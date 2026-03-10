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
import sys
import time
from typing import Optional

import paramiko
from vastai_sdk import VastAI as VastClient


TEMPLATE_HASH = "cf10248a1d803b250a4382ca71fa9c50"
MAX_BID = 5.0          # absolute cap $/hr per machine
DISK_GB = 100          # local disk size in GB when creating a new instance
BID_HEADROOM = 0.35    # bid up to 15% above min_bid to win interrupts

# Search query for H100 / H200 / A100 class at or below MAX_BID (bid / interruptible pricing)
GPU_QUERY = (
  f"gpu_name in [\"H100_PCIE\", \"H100_SXM\", \"H100_NVL\", \"H200\", \"A100_PCIE\", \"A100_SXM4\"] "
  f"num_gpus>=1 min_bid<={MAX_BID}"
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
  """Use VastAI SDK to find a suitable 8x H100 interruptible offer."""
  print(f"Searching Vast offers for 8x H100 (bid, min_bid <= ${MAX_BID})...")
  try:
    # With raw=True this returns a list[dict] of offers (see search__offers in vast.py).
    out = vast.search_offers(query=GPU_QUERY, type="bid", order="dph-")
  except Exception as e:
    raise SystemExit(f"vast_sdk.search_offers failed: {e}")

  if not isinstance(out, list) or not out:
    raise SystemExit("vast_sdk.search_offers returned no offers; cannot select offer id.")

  offer_id = int(out[0]["id"])
  print(f"Selected offer id: {offer_id}")
  return offer_id


def create_instance(offer_id: int, vast: VastClient) -> int:
  """Create an instance from the given offer using the VastAI SDK and
  infer its id by diffing `show_instances` before/after.

  This avoids depending on the exact JSON shape / return value of
  `create_instance`, which differs across Vast.ai versions.
  """
  print(f"Creating instance from template hash {TEMPLATE_HASH} with dynamic bid scaling...")

  # Look up the chosen offer so we can scale bid price with GPU count and min_bid.
  try:
    offer_info = vast.search_offers(query=f"id={offer_id}", type="bid", order="dph-")
  except Exception as e:
    print(f"Warning: vast.search_offers(id={offer_id}) failed; falling back to flat MAX_BID: {e}")
    offer_info = []

  num_gpus = 1
  min_bid = 0.0
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

  # Use the market's own min_bid as baseline and add a small headroom,
  # clamped by a global MAX_BID cap. This automatically accounts for
  # H200 > H100 > A100 price tiers and scales with GPU count.
  if min_bid <= 0:
    # Fallback if API doesn't return min_bid: conservative flat cap.
    base_bid = MAX_BID
  else:
    base_bid = min_bid

  bid_price = min(MAX_BID, base_bid * (1.0 + BID_HEADROOM))

  print(f"Selected offer {offer_id} with num_gpus={num_gpus}, min_bid={min_bid:.4f}, bid_price={bid_price:.4f}")

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
          return inst_id

    print("  New instance not visible yet; sleeping 10s...")
    time.sleep(10)

  raise SystemExit("Timed out waiting for new instance to appear in show_instances after create_instance.")


def find_or_create_instance(vast: VastClient) -> int:
  """Reuse an existing instance if available, otherwise create a new one."""
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
    print(f"Reusing existing instance id: {inst_id}")
    return inst_id

  # No existing instance; go through the offer → create flow.
  offer_id = find_offer_id(vast)
  return create_instance(offer_id, vast)


def wait_for_ssh_details(instance_id: int, vast: VastClient, timeout_minutes: int = 20) -> dict:
  """Poll VastAI SDK `ssh_url` until SSH is ready, then parse into Paramiko connection details."""
  print("Waiting for SSH endpoint to become available...")
  deadline = time.time() + timeout_minutes * 60
  ssh_descriptor: Optional[str] = None

  while time.time() < deadline:
    try:
      # `ssh_url` prints the URL (e.g. ssh://root@host:port) to stdout;
      # the SDK captures that in `last_output`.
      vast.ssh_url(id=instance_id)
      out = getattr(vast, "last_output", "") or ""
    except Exception:
      out = ""

    out = out.strip()
    if out:
      # Take the last non-empty line in case of extra output
      for ln in reversed(out.splitlines()):
        if ln.strip():
          ssh_descriptor = ln.strip()
          break

    if ssh_descriptor and not ssh_descriptor.lower().startswith("error:"):
      print(f"SSH endpoint ready: {ssh_descriptor}")
      break

    print("  SSH not ready yet; sleeping 30s...")
    time.sleep(30)

  if not ssh_descriptor or ssh_descriptor.lower().startswith("error:"):
    raise SystemExit(f"Timed out waiting for SSH on instance {instance_id}.")

  # Two possible formats:
  # 1) URL:  ssh://user@host:port
  # 2) Full ssh command: ssh -p 4242 -i /path/to/key user@host
  if ssh_descriptor.startswith("ssh://"):
    # URL-style from `_ssh_url`
    rest = ssh_descriptor[len("ssh://") :]
    try:
      user_host, port_str = rest.rsplit(":", 1)
      user, host = user_host.split("@", 1)
      port = int(port_str)
    except Exception as e:
      raise SystemExit(f"Could not parse ssh-url output into host/user/port: {ssh_descriptor} ({e})")
    return {
      "hostname": host,
      "username": user,
      "port": port,
      "key_filename": str(SSH_KEY_PATH),
    }

  # Fallback: parse as full ssh command (if Vast ever emits one)
  tokens = shlex.split(ssh_descriptor)
  tokens_iter = iter(tokens[1:])  # skip leading "ssh"
  host = None
  user = None
  port = 22
  key_path: Optional[str] = None

  for tok in tokens_iter:
    if tok == "-p":
      port = int(next(tokens_iter))
    elif tok == "-i":
      key_path = next(tokens_iter)
    elif "@" in tok:
      user, host = tok.split("@", 1)

  if not host or not user:
    raise SystemExit(f"Could not parse ssh-url output into host/user: {ssh_descriptor}")

  return {
    "hostname": host,
    "username": user,
    "port": port,
    "key_filename": key_path or str(SSH_KEY_PATH),
  }


def run_remote_training(ssh_info: dict, hf_token: str, log_dir: pathlib.Path) -> None:
  """Run the remote bootstrap + multi-variant training over Paramiko and save logs locally."""
  log_dir.mkdir(parents=True, exist_ok=True)
  timestamp = _dt.datetime.utcnow().strftime("%Y%m%d-%H%M%S")
  log_path = log_dir / f"vast_run_{timestamp}.log"

  remote_script = f"""set -euo pipefail

export HF_TOKEN={hf_token}

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

if [[ ! -d stochastic ]]; then
  echo "Remote: cloning stochastic repo..."
  git clone https://github.com/AeroX2/stochastic.git
fi

cd stochastic
chmod +x setup_and_train.sh

echo "Remote: running baseline variant..."
./setup_and_train.sh --variant=baseline   --nproc-per-node=8 --hf-repo={HF_REPO_BASELINE} --save-every=500

echo "Remote: running spiking variant..."
./setup_and_train.sh --variant=spiking    --nproc-per-node=8 --hf-repo={HF_REPO_SPIKING} --save-every=500

echo "Remote: running stochastic variant..."
./setup_and_train.sh --variant=stochastic --nproc-per-node=8 --hf-repo={HF_REPO_STOCHASTIC} --save-every=500

echo "Remote: running both variant..."
./setup_and_train.sh --variant=both       --nproc-per-node=8 --hf-repo={HF_REPO_BOTH} --save-every=500

echo "Remote: all variants completed successfully."
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
      )
      break
    except Exception as e:
      last_err = e
      print(f"SSH connect attempt {attempt + 1} failed: {e}. Retrying in 15s...")
      time.sleep(15)
  else:
    raise SystemExit(f"Failed to connect via SSH using Paramiko after multiple attempts: {last_err}")

  try:
    stdin, stdout, stderr = client.exec_command("bash -s", get_pty=True)
    stdin.write(remote_script)
    stdin.channel.shutdown_write()

    # Stream logs to console and file
    with log_path.open("w", encoding="utf-8") as f:
      for line in iter(stdout.readline, ""):
        if not line:
          break
        sys.stdout.write(line)
        f.write(line)
      # Drain any remaining stderr at the end
      err_rest = stderr.read()
      if err_rest:
        sys.stdout.write(err_rest)
        f.write(err_rest)

    exit_status = stdout.channel.recv_exit_status()
  finally:
    client.close()

  if exit_status != 0:
    raise SystemExit(f"Remote training script exited with code {exit_status}. See {log_path}.")

  print(f"Remote training completed successfully. Logs saved to {log_path}")


def destroy_instance(instance_id: int, vast: VastClient) -> None:
  """Best-effort destroy of the Vast instance via VastAI SDK."""
  print(f"Destroying Vast instance {instance_id} via VastAI SDK...")
  try:
    vast.destroy_instance(id=instance_id)
  except Exception as e:
    print(f"Warning: failed to destroy instance {instance_id} via SDK: {e}", file=sys.stderr)


def main() -> None:
  vast_api_key = _require_env("VAST_API_KEY")
  hf_token = _require_env("HF_TOKEN")
  vast = _get_vast_client(vast_api_key)

  instance_id: Optional[int] = None
  success = False
  try:
    instance_id = find_or_create_instance(vast)
    ssh_info = wait_for_ssh_details(instance_id, vast)
    logs_dir = pathlib.Path("vast_logs")
    run_remote_training(ssh_info, hf_token, logs_dir)
    success = True
  finally:
    # Only auto-destroy the instance if training completed successfully.
    # On errors (SSH/connectivity, script failures), leave the instance
    # running so it can be debugged or reused manually.
    if instance_id is not None and success:
      destroy_instance(instance_id, vast)


if __name__ == "__main__":
  main()

