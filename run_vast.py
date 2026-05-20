#!/usr/bin/env python3
"""
Vast.ai stochastic training launcher (Python version).

This script:
  - Finds an interruptible Vast.ai offer with 8x H100 GPUs and bid price <= $5
  - Creates an instance from the provided template hash
  - Waits for SSH to become available
  - SSHes in, clones https://github.com/AeroX2/stochastic, and runs:
      ./setup_and_train.sh --variant={baseline,spiking,stochastic,both}
    with 8-way distributed training and the provided Hugging Face repos
  - Streams remote logs to both stdout and a local log file so eval results
    are persisted for later analysis
  - Destroys the instance on completion or error

Prerequisites (local machine running this script):
  - Python 3.10+
  - Vast.ai CLI installed and on PATH (`pip install vastai`)
  - `ssh` client installed
  - `requests` installed (`pip install requests`)
  - Environment variables:
      VAST_API_KEY  - your Vast.ai API key
      HF_TOKEN      - your Hugging Face token

Note:
  - This script replaces direct `curl` calls with the Python `requests` library.
  - The Vast.ai Python SDK can be used in addition to this, but is not required
    for the core flow implemented here.
"""

from __future__ import annotations

import datetime as _dt
import os
import pathlib
import shlex
import argparse
import sys
import time

# Windows console defaults to cp1252 which can't encode ANSI/Unicode characters
# emitted by the remote training (rustbpe progress lines, tmux escapes, etc.).
# Force UTF-8 with replace so streaming never crashes the local process.
try:
  sys.stdout.reconfigure(encoding="utf-8", errors="replace")
  sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
  pass
from typing import Optional

import paramiko
from vastai_sdk import VastAI as VastClient


TEMPLATE_HASH = "cf10248a1d803b250a4382ca71fa9c50"
# MAX_BID is the hard cap $/hr per machine. Raised from 5.0 to 10.0 so we can
# accept on-demand 8x A100 (~$8.81/hr) and not just bid-priced 4-GPU offers.
# On-demand mode (VAST_OFFER_TYPE=on-demand) does not auto-preempt mid-run, so
# the extra cost buys reliability vs the bid market's spurious preemptions.
MAX_BID = float(os.environ.get("VAST_MAX_BID", "10.0"))
DISK_GB = 100          # local disk size in GB when creating a new instance
BID_HEADROOM = 0.35    # bid up to 35% above min_bid (bid market only)

DEFAULT_REPO_URL = "https://github.com/AeroX2/stochastic.git"
DEFAULT_GIT_REF = "main"

# Minimum useful GPU count. Below 4, training stretches into 6+ hour ranges per
# variant which makes the iteration loop impractical. Iter 0 hit a 1-GPU offer
# that would have been ~13 hours wall-clock for four variants.
MIN_GPUS = 4

# Minimum host disk_space (GB) the offer must report. The 30GB threshold matches
# the remote heredoc's df check. Offers with disk_space=11 returned 11GB total /
# which crashed training mid-checkpoint; filter them out at search time.
MIN_DISK_GB = int(os.environ.get("VAST_MIN_DISK_GB", "100"))

# H-series + B-series (Hopper + Blackwell). A100 dropped because it was ~3x
# slower than H100 at similar bid prices. B200 is the fastest and often the
# cheapest per FLOP on Vast.ai, but availability fluctuates -- include both
# so the search finds whichever is currently rentable.
GPU_QUERY = (
  f"gpu_name in [\"H100_PCIE\", \"H100_SXM\", \"H100_NVL\", \"H200\", \"B200\", \"B200_SXM\"] "
  f"num_gpus>={MIN_GPUS} min_bid<={MAX_BID} reliability>0.95 disk_space>={MIN_DISK_GB}"
)

# Default SSH private key (generated in repo root via ssh-keygen)
SSH_KEY_PATH = pathlib.Path(__file__).with_name("vast_vastai_key")

# Hugging Face repos for each variant
HF_REPO_BASELINE = "aerox2/baseline-nanogpt"
HF_REPO_SPIKING = "aerox2/spiking-nanogpt"
HF_REPO_STOCHASTIC = "aerox2/stochastic-nanogpt"
HF_REPO_BOTH = "aerox2/both-nanogpt"


def _require_env(name: str) -> str:
  value = os.environ.get(name)
  if not value:
    raise SystemExit(f"Environment variable {name} must be set.")
  return value


def _get_vast_client(api_key: str) -> VastClient:
  """Construct a VastAI SDK client (CLI-backed or standalone)."""
  # Force raw=True so SDK methods return JSON/Response objects instead of only printing.
  # This matches vastai-0.5.0's CLI functions, which return rows/Response when raw=True.
  return VastClient(api_key=api_key, raw=True)


def find_offer_id(vast: VastClient) -> int:
  """Find a suitable multi-GPU interruptible offer.

  Strategy: search with GPU_QUERY (which already filters num_gpus>=MIN_GPUS,
  bid<=MAX_BID, reliability>0.95), then rank by (num_gpus desc, min_bid asc).
  Higher GPU count first because per-iter wall-clock dominates the budget, then
  cheapest-of-the-best so we don't overpay.

  Honors VAST_OFFER_BLACKLIST env var (comma-separated offer ids) to skip
  offers whose underlying host was just discovered to have full disk or other
  per-host issues in this session.
  """
  blacklist_raw = os.environ.get("VAST_OFFER_BLACKLIST", "")
  blacklist = {s.strip() for s in blacklist_raw.split(",") if s.strip()}
  if blacklist:
    print(f"Offer blacklist: {sorted(blacklist)}")
  offer_type = os.environ.get("VAST_OFFER_TYPE", "bid")
  print(f"Searching Vast {offer_type} offers (>={MIN_GPUS} GPUs, reliability>0.95)...")
  try:
    out = vast.search_offers(query=GPU_QUERY, type=offer_type, order="dph-")
  except Exception as e:
    raise SystemExit(f"vast_sdk.search_offers failed: {e}")

  if not isinstance(out, list) or not out:
    raise SystemExit(
      f"No offers matched: {GPU_QUERY}. "
      f"The market may not currently have {MIN_GPUS}+ GPU bid offers; try lowering MIN_GPUS or raising MAX_BID."
    )

  filtered = [o for o in out if str(o.get("id")) not in blacklist]
  if not filtered:
    raise SystemExit(
      f"All {len(out)} offers matching {GPU_QUERY!r} are blacklisted: {sorted(blacklist)}."
    )

  def _key(o):
    try:
      gpus = int(o.get("num_gpus", 0))
    except Exception:
      gpus = 0
    try:
      bid = float(o.get("min_bid", 0.0))
    except Exception:
      bid = 0.0
    # (more GPUs first, then cheapest)
    return (-gpus, bid)

  ranked = sorted(filtered, key=_key)
  best = ranked[0]
  offer_id = int(best["id"])
  print(
    f"Selected offer id={offer_id} "
    f"(gpu={best.get('gpu_name')}, n={best.get('num_gpus')}, "
    f"min_bid={best.get('min_bid')}, dph={best.get('dph_total')}, "
    f"reliability={best.get('reliability2')})"
  )
  return offer_id


def create_instance(offer_id: int, vast: VastClient) -> tuple[int, int, bool]:
  """Create an instance from the given offer using the VastAI SDK and
  infer its id by diffing `show_instances` before/after.

  Returns (instance_id, num_gpus, created_here=True).

  This avoids depending on the exact JSON shape / return value of
  `create_instance`, which differs across Vast.ai versions.
  """
  offer_type = os.environ.get("VAST_OFFER_TYPE", "bid")
  print(f"Creating instance from template hash {TEMPLATE_HASH} (type={offer_type})...")

  # Look up the chosen offer so we can scale price with GPU count and floor price.
  try:
    offer_info = vast.search_offers(query=f"id={offer_id}", type=offer_type, order="dph-")
  except Exception as e:
    print(f"Warning: vast.search_offers(id={offer_id}) failed; falling back to flat MAX_BID: {e}")
    offer_info = []

  num_gpus = 1
  min_bid = 0.0
  dph_total = 0.0
  if isinstance(offer_info, list) and offer_info:
    offer = offer_info[0]
    try:
      num_gpus = int(offer.get("num_gpus", 1))
    except Exception:
      num_gpus = 1
    try:
      min_bid = float(offer.get("min_bid", 0.0))
    except Exception:
      min_bid = 0.0
    try:
      dph_total = float(offer.get("dph_total", 0.0))
    except Exception:
      dph_total = 0.0

  if offer_type == "on-demand":
    # On-demand offers are billed at dph_total. Pass that as bid_price so the
    # SDK accepts the price; instance won't be preempted on bid auctions.
    base_price = dph_total if dph_total > 0 else MAX_BID
    bid_price = min(MAX_BID, base_price)
  else:
    # Bid market: pay min_bid + headroom to win interrupts, capped at MAX_BID.
    base_price = min_bid if min_bid > 0 else MAX_BID
    bid_price = min(MAX_BID, base_price * (1.0 + BID_HEADROOM))

  print(
    f"Selected offer {offer_id} type={offer_type} num_gpus={num_gpus} "
    f"min_bid={min_bid:.4f} dph={dph_total:.4f} bid_price={bid_price:.4f}"
  )

  # Snapshot existing instance IDs first
  try:
    existing = vast.show_instances()
  except Exception as e:
    print(f"Warning: vast.show_instances failed before create_instance: {e}")
    existing = []

  prev_ids: set[int] = set()
  if isinstance(existing, list):
    for row in existing:
      if isinstance(row, dict) and "id" in row:
        try:
          prev_ids.add(int(row["id"]))
        except Exception:
          continue

  try:
    # This mirrors: `vastai create instance OFFER_ID --template_hash TEMPLATE_HASH --bid_price MAX_BID`
    vast.create_instance(
      id=offer_id,
      template_hash=TEMPLATE_HASH,
      bid_price=bid_price,
      disk=DISK_GB,
    )
  except Exception as e:
    raise SystemExit(f"Error creating instance via VastAI SDK: {e}")

  # Poll `show_instances` until we see a new id that wasn't present before.
  print("Waiting for new instance to appear in show_instances...")
  deadline = time.time() + 10 * 60  # 10 minutes
  while time.time() < deadline:
    try:
      current = vast.show_instances()
    except Exception as e:
      print(f"Warning: vast.show_instances failed while waiting for new instance: {e}")
      current = []

    if isinstance(current, list):
      for row in current:
        if not isinstance(row, dict) or "id" not in row:
          continue
        try:
          inst_id = int(row["id"])
        except Exception:
          continue
        if inst_id not in prev_ids:
          print(f"Instance created with id: {inst_id}")
          # Some Vast offers (on-demand especially) come up in `intended=stopped`
          # state and need an explicit start to run the container. Send start
          # unconditionally -- it's a no-op for already-running instances.
          try:
            vast.start_instance(id=inst_id)
            print(f"  Sent start_instance({inst_id}).")
          except Exception as e:
            print(f"  Warning: start_instance({inst_id}) failed: {e}")
          return inst_id, num_gpus, True

    print("  New instance not visible yet; sleeping 10s...")
    time.sleep(10)

  raise SystemExit("Timed out waiting for new instance to appear in show_instances after create_instance.")


def find_or_create_instance(vast: VastClient) -> tuple[int, int, bool]:
  """Reuse an existing instance if available, otherwise create a new one.

  Returns (instance_id, num_gpus, created_here).
  """
  print("Checking for existing Vast instances...")
  try:
    existing = vast.show_instances()
  except Exception as e:
    print(f"Warning: vast.show_instances failed, will always create a new instance: {e}")
    existing = []

  # vastai-0.5.0 show__instances with raw=True returns a list[dict]
  if isinstance(existing, list) and existing:
    # Prefer running instances if we can detect them; otherwise just take the first.
    def is_running(row: dict) -> bool:
      status = str(row.get("actual_status") or row.get("status") or "").lower()
      return status in {"running", "active"}

    running = [row for row in existing if isinstance(row, dict) and is_running(row)]
    chosen = (running or existing)[0]
    inst_id = int(chosen["id"])
    try:
      num_gpus = int(chosen.get("num_gpus", 1))
    except Exception:
      num_gpus = 1
    print(f"Reusing existing instance id: {inst_id} with num_gpus={num_gpus}")
    # Nudge instance to running if it's stopped (idempotent on already-running).
    intended = chosen.get("intended_status")
    if intended != "running":
      try:
        vast.start_instance(id=inst_id)
        print(f"  Sent start_instance({inst_id}); was intended={intended}.")
      except Exception as e:
        print(f"  Warning: start_instance({inst_id}) failed: {e}")
    return inst_id, num_gpus, False

  # No existing instance; go through the offer → create flow.
  offer_id = find_offer_id(vast)
  return create_instance(offer_id, vast)


def wait_for_ssh_details(instance_id: int, vast: VastClient, timeout_minutes: int = 20) -> dict:
  """Poll until the instance is actually_status=running, then return SSH details
  from show_instances (more reliable than ssh_url which has an off-by-one port quirk).
  """
  print(f"Waiting for instance {instance_id} to reach actual_status=running...")
  deadline = time.time() + timeout_minutes * 60
  inst: Optional[dict] = None

  while time.time() < deadline:
    try:
      rows = vast.show_instances()
    except Exception as e:
      print(f"  show_instances failed: {e}; retry in 15s...")
      time.sleep(15)
      continue

    for row in (rows or []):
      if isinstance(row, dict) and int(row.get("id", -1)) == instance_id:
        inst = row
        break

    if inst is None:
      print(f"  instance {instance_id} not found in show_instances; retry in 15s...")
      time.sleep(15)
      continue

    actual = inst.get("actual_status")
    intended = inst.get("intended_status")
    if actual == "running":
      # Prefer direct SSH (public_ipaddr + port 22's host mapping) over the
      # proxy hostname. Vast.ai's proxies (ssh<N>.vast.ai) drop connections
      # mid-handshake on many on-demand hosts, while direct works fine.
      ports = inst.get("ports") or {}
      direct = (ports.get("22/tcp") or [{}])[0]
      direct_port = direct.get("HostPort")
      public_ip = inst.get("public_ipaddr")
      if public_ip and direct_port:
        print(f"SSH endpoint ready (direct): ssh://root@{public_ip}:{direct_port}")
        return {
          "hostname": str(public_ip),
          "username": "root",
          "port": int(direct_port),
          "key_filename": str(SSH_KEY_PATH),
        }
      # Fallback to proxy if direct port mapping not exposed.
      ssh_host = inst.get("ssh_host")
      ssh_port = inst.get("ssh_port")
      if ssh_host and ssh_port:
        print(f"SSH endpoint ready (proxy fallback): ssh://root@{ssh_host}:{ssh_port}")
        return {
          "hostname": str(ssh_host),
          "username": "root",
          "port": int(ssh_port),
          "key_filename": str(SSH_KEY_PATH),
        }
      print(f"  instance running but no SSH details yet; retry in 15s...")
    else:
      print(f"  actual={actual} intended={intended}; retry in 15s...")
    time.sleep(15)

  raise SystemExit(
    f"Timed out waiting for instance {instance_id} actual_status=running "
    f"(last actual={inst.get('actual_status') if inst else None})."
  )


def run_remote_training(
  ssh_info: dict,
  hf_token: str,
  log_dir: pathlib.Path,
  *,
  repo_url: str = DEFAULT_REPO_URL,
  git_ref: str = DEFAULT_GIT_REF,
) -> None:
  """Run the remote bootstrap + multi-variant training over Paramiko and save logs locally."""
  log_dir.mkdir(parents=True, exist_ok=True)

  # Which variants to run, in order. Default = all four. Per-iter override via
  # VAST_VARIANTS=spiking (or "spiking,stochastic", etc) saves GPU time when we
  # already know the baseline number and only need to measure the others.
  _all_variant_repos = {
    "baseline": HF_REPO_BASELINE,
    "spiking": HF_REPO_SPIKING,
    "stochastic": HF_REPO_STOCHASTIC,
    "both": HF_REPO_BOTH,
  }
  variants_env = os.environ.get("VAST_VARIANTS", "baseline,spiking,stochastic,both")
  variants_to_run: list[str] = []
  for v in variants_env.split(","):
    v = v.strip()
    if v and v in _all_variant_repos:
      variants_to_run.append(v)
  if not variants_to_run:
    raise SystemExit(f"VAST_VARIANTS produced no valid variants: {variants_env!r}")
  variant_commands = "\n".join(
    f"run_variant {v} {_all_variant_repos[v]}" for v in variants_to_run
  )
  print(f"Will run variants: {variants_to_run}")
  timestamp = _dt.datetime.utcnow().strftime("%Y%m%d-%H%M%S")
  log_path = log_dir / f"vast_run_{timestamp}.log"

  remote_script = f"""set -euo pipefail

export HF_TOKEN={hf_token}

REPO_URL={shlex.quote(repo_url)}
GIT_REF={shlex.quote(git_ref)}

# Disk health check: bail fast on hosts whose overlay is already near-full.
# Other tenants on the same Vast host can saturate the shared overlay even when
# our instance quota is 100GB. Training will crash mid-checkpoint with
# "inline_container.cc:672 unexpected pos N vs M" if disk runs out.
FREE_KB=$(df --output=avail / | tail -1 | tr -d ' ')
FREE_GB=$((FREE_KB / 1024 / 1024))
echo "Remote: free disk on / = ${{FREE_GB}}GB"
if [ "$FREE_GB" -lt 30 ]; then
  echo "Remote: ABORT - free disk ${{FREE_GB}}GB < 30GB threshold. Host overlay too full for safe training. Exiting so launcher can pick a different offer."
  exit 42
fi

echo "Remote: checking for git and python3..."
if ! command -v git &>/dev/null; then
  if command -v apt-get &>/dev/null; then
    if command -v sudo &>/dev/null; then
      sudo apt-get update -y
      sudo apt-get install -y git
    else
      apt-get update -y
      apt-get install -y git
    fi
  else
    echo "Warning: git not found and apt-get is not available; assuming git already present in image." >&2
  fi
fi

if ! command -v python3 &>/dev/null; then
  if command -v apt-get &>/dev/null; then
    if command -v sudo &>/dev/null; then
      sudo apt-get update -y
      sudo apt-get install -y python3 python3-pip
    else
      apt-get update -y
      apt-get install -y python3 python3-pip
    fi
  else
    echo "Warning: python3 not found and apt-get is not available; setup_and_train.sh may fail." >&2
  fi
fi

echo "Remote: syncing repo ($REPO_URL) at ref ($GIT_REF)..."
if [[ ! -d stochastic/.git ]]; then
  rm -rf stochastic
  echo "Remote: cloning stochastic repo..."
  git clone "$REPO_URL" stochastic
fi

cd stochastic

# Always sync to requested ref (branch/tag/sha). If it's a branch on origin, prefer origin/<branch>.
git remote set-url origin "$REPO_URL" || true
git fetch --prune origin
if git show-ref --verify --quiet "refs/remotes/origin/$GIT_REF"; then
  git checkout -B "$GIT_REF" "origin/$GIT_REF"
  git reset --hard "origin/$GIT_REF"
else
  # Could be a tag or sha (or a non-origin remote ref)
  git checkout --detach "$GIT_REF"
fi
git clean -fdx
echo "Remote: checked out $(git rev-parse --short HEAD)"

# If a training process is already running, just stream its log and exit.
if pgrep -f "experiments.train" >/dev/null 2>&1; then
  echo "Remote: training already running, streaming log..."
  if [[ -f train.log ]]; then
    tail -n 200 -f train.log
  else
    echo "Remote: train.log not found; attach manually via ssh."
  fi
  exit 0
fi

chmod +x setup_and_train.sh

# Run each variant in a subshell so a failure (set -e or non-zero exit through pipefail)
# doesn't abort the remaining variants. Print a clear marker on failure so log parsing
# can distinguish "variant ran and got a CORE metric" from "variant crashed".
run_variant() {{
  local variant="$1"
  local hf_repo="$2"
  echo "Remote: running $variant variant..."
  if (set -eo pipefail; ./setup_and_train.sh --variant="$variant" --hf-repo="$hf_repo" --save-every=500 2>&1 | tee -a train.log); then
    echo "Remote: $variant variant SUCCEEDED"
  else
    echo "Remote: $variant variant FAILED (continuing to next variant)"
  fi
}}

{variant_commands}

echo "Remote: all variants attempted."
# Print a sentinel line as the very last thing iter_run.sh emits. The local
# tail -F --pid=$PID can hang on a static log even after PID dies (it only
# polls on I/O); writing this final line wakes tail up so it notices the pid
# is gone and exits cleanly. The local launcher parses this to know we hit a
# normal completion vs a network disconnect.
echo "Remote: __ITER_RUN_DONE__"
"""

  print(f"Starting remote bootstrap and training via Paramiko (logs -> {log_path})...")

  client = paramiko.SSHClient()
  client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

  # SSH endpoints can take a bit to become reachable even after ssh-url is ready.
  # Retry connect for a while before giving up.
  last_err: Optional[Exception] = None
  for attempt in range(10):
    try:
      client.connect(
        hostname=ssh_info["hostname"],
        port=ssh_info["port"],
        username=ssh_info["username"],
        key_filename=ssh_info.get("key_filename"),
        look_for_keys=False,
        timeout=30,
        # Send TCP keep-alive at the OS layer so idle NAT/proxy timeouts don't
        # silently kill the connection. Windows was emitting winsock 10054
        # within 1-2 minutes when tail was idle waiting for new lines.
        banner_timeout=60,
        auth_timeout=30,
      )
      # paramiko-level keep-alive: send an SSH ping every 30s on the Transport.
      # This keeps the channel alive across NAT/proxy idle timeouts.
      try:
        client.get_transport().set_keepalive(30)
      except Exception:
        pass
      break
    except Exception as e:
      last_err = e
      print(f"SSH connect attempt {attempt + 1} failed: {e}. Retrying in 15s...")
      time.sleep(15)
  else:
    raise SystemExit(f"Failed to connect via SSH using Paramiko after multiple attempts: {last_err}")

  # Strategy: write the script to the remote via SFTP, launch with nohup so it
  # survives SSH/network disconnects, then tail -F the remote log. If our local
  # tail dies (timeout, network), the remote training keeps running and we can
  # re-attach via another invocation (the script's pgrep branch resumes the tail).
  REMOTE_SCRIPT = "/workspace/iter_run.sh"
  REMOTE_LOG = "/workspace/iter_run.log"
  REMOTE_PIDFILE = "/workspace/iter_run.pid"

  try:
    # Disable auto-tmux on first connect; otherwise the welcome-tmux corrupts logs.
    client.exec_command("touch ~/.no_auto_tmux")

    # 1. Upload the remote script via SFTP.
    sftp = client.open_sftp()
    try:
      with sftp.open(REMOTE_SCRIPT, "w") as f:
        f.write(remote_script)
      sftp.chmod(REMOTE_SCRIPT, 0o755)
    finally:
      sftp.close()

    # 2. If no prior training is running, start a new one detached via nohup.
    submit_cmd = (
      f"set -e; "
      f"if [ -f {REMOTE_PIDFILE} ] && kill -0 $(cat {REMOTE_PIDFILE}) 2>/dev/null; then "
      f"  echo \"Remote: training already in progress (pid=$(cat {REMOTE_PIDFILE}))\"; "
      f"else "
      f"  nohup bash {REMOTE_SCRIPT} > {REMOTE_LOG} 2>&1 < /dev/null & "
      f"  echo $! > {REMOTE_PIDFILE}; "
      f"  echo \"Remote: training submitted, pid=$(cat {REMOTE_PIDFILE}), log={REMOTE_LOG}\"; "
      f"fi"
    )
    _stdin, _stdout, _stderr = client.exec_command(submit_cmd)
    submit_out = _stdout.read().decode("utf-8", errors="replace")
    submit_err = _stderr.read().decode("utf-8", errors="replace")
    submit_rc = _stdout.channel.recv_exit_status()
    print(submit_out, end="")
    if submit_err:
      print(submit_err, end="", file=sys.stderr)
    if submit_rc != 0:
      raise SystemExit(f"Submit failed (exit {submit_rc}). stdout: {submit_out!r} stderr: {submit_err!r}")

    # 3. Tail -F the remote log; tail exits when the training pid is gone.
    tail_cmd = (
      f"PID=$(cat {REMOTE_PIDFILE} 2>/dev/null); "
      f"tail -n +1 -F {REMOTE_LOG} ${{PID:+--pid=$PID}} 2>/dev/null"
    )
    stdin, stdout, stderr = client.exec_command(tail_cmd, bufsize=1)

    # Stream logs to console and file. flush() on every line so the local file
    # is always usable for parsing CORE metrics even mid-run.
    with log_path.open("w", encoding="utf-8") as f:
      for line in iter(stdout.readline, ""):
        if not line:
          break
        sys.stdout.write(line)
        f.write(line)
        f.flush()
      err_rest = stderr.read()
      if err_rest:
        sys.stdout.write(err_rest.decode("utf-8", errors="replace") if isinstance(err_rest, bytes) else err_rest)
        f.write(err_rest.decode("utf-8", errors="replace") if isinstance(err_rest, bytes) else err_rest)

    exit_status = stdout.channel.recv_exit_status()
  finally:
    client.close()

  # Distinguish "training actually finished" from "our local view got cut off".
  # tail's normal clean exit when --pid dies is 0. Paramiko returns -1 when the
  # SSH channel was torn down without an exit status (network blip, Bash-tool
  # SIGTERM, peer reset). In that case the remote training is almost certainly
  # still alive (it's nohup'd) and we must NOT let main()'s finally auto-destroy
  # the instance. Raise so the outer try sees an exception and skips destroy.
  print(f"Remote training log captured to {log_path}")
  if exit_status == 0:
    return
  if exit_status == 130:
    # tail was SIGINT'd locally; remote process state unknown -- treat as interrupted.
    raise RuntimeError(f"Local tail interrupted (SIGINT). Remote training may still be running. See {log_path}.")
  if exit_status == -1:
    raise RuntimeError(
      f"SSH channel closed without exit status (likely local timeout / network blip). "
      f"Remote training is nohup'd and may still be running. Re-run run_vast.py to resume tailing. See {log_path}."
    )
  raise RuntimeError(f"tail exited with non-zero status {exit_status}. See {log_path}.")


def destroy_instance(instance_id: int, vast: VastClient) -> None:
  """Best-effort destroy of the Vast instance via VastAI SDK."""
  print(f"Destroying Vast instance {instance_id} via VastAI SDK...")
  try:
    vast.destroy_instance(id=instance_id)
  except Exception as e:
    print(f"Warning: failed to destroy instance {instance_id} via SDK: {e}", file=sys.stderr)


def _prompt_destroy_with_timeout(instance_id: int, timeout_seconds: int = 300) -> bool:
  """Ask whether to destroy the instance, with a timeout.

  Returns True if we should destroy, False otherwise. If there is no
  input within `timeout_seconds`, defaults to destroying the instance.

  When run non-interactively (no TTY on stdin) or VAST_AUTO_DESTROY=1, skip the
  prompt entirely and destroy immediately. This keeps automated loops from
  waiting on a tty that will never come.
  """
  auto_destroy = os.environ.get("VAST_AUTO_DESTROY") == "1"
  try:
    is_tty = sys.stdin.isatty()
  except Exception:
    is_tty = False
  if auto_destroy or not is_tty:
    print(f"\nInstance {instance_id}: non-interactive run, auto-destroying.", flush=True)
    return True

  print(
    f"\nInstance {instance_id} is still running.\n"
    f"Destroy it now? [Y/n] (auto-destroy in {timeout_seconds // 60} minutes if no input)...",
    end=" ",
    flush=True,
  )

  # Windows: use msvcrt to implement a simple timed input loop.
  if os.name == "nt":
    import msvcrt  # type: ignore[import]

    line = ""
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
      if msvcrt.kbhit():
        ch = msvcrt.getwche()
        if ch in ("\r", "\n"):
          break
        # Handle backspace
        if ch == "\b":
          if line:
            line = line[:-1]
          continue
        line += ch
      else:
        time.sleep(0.1)

    answer = line.strip().lower()
    if not answer:
      print("\nNo input received; defaulting to destroy.")
      return True
    return answer.startswith("y")

  # POSIX fallback: best-effort blocking input (no real timeout)
  try:
    answer = input()
  except EOFError:
    return True
  answer = answer.strip().lower()
  if not answer:
    return True
  return answer.startswith("y")


def main() -> None:
  parser = argparse.ArgumentParser(description="Vast.ai stochastic training launcher")
  parser.add_argument(
    "--repo-url",
    default=DEFAULT_REPO_URL,
    help=f"Git repo URL to clone on remote (default: {DEFAULT_REPO_URL})",
  )
  parser.add_argument(
    "--git-ref",
    default=DEFAULT_GIT_REF,
    help=f"Git ref to checkout on remote (branch/tag/sha). Default: {DEFAULT_GIT_REF}",
  )
  args = parser.parse_args()

  vast_api_key = _require_env("VAST_API_KEY")
  hf_token = _require_env("HF_TOKEN")
  vast = _get_vast_client(vast_api_key)

  instance_id: Optional[int] = None
  created_here = False
  training_returned = False
  try:
    instance_id, _, created_here = find_or_create_instance(vast)
    ssh_info = wait_for_ssh_details(instance_id, vast)
    logs_dir = pathlib.Path("vast_logs")
    run_remote_training(
      ssh_info,
      hf_token,
      logs_dir,
      repo_url=args.repo_url,
      git_ref=args.git_ref,
    )
    training_returned = True
  finally:
    if instance_id is not None and created_here:
      if not training_returned:
        # Local launcher was interrupted (Bash-tool timeout, network blip, SIGTERM, etc.)
        # The remote nohup'd training may still be running. Leave the instance alive so
        # a subsequent run_vast.py call can re-attach via the existing pidfile/log.
        print(
          f"Leaving instance {instance_id} running: local launcher exited before training returned. "
          f"Re-run `python run_vast.py --git-ref {args.git_ref}` to resume tailing the remote log."
        )
      elif _prompt_destroy_with_timeout(instance_id):
        destroy_instance(instance_id, vast)
      else:
        print(f"Leaving instance {instance_id} running.")


if __name__ == "__main__":
  main()

