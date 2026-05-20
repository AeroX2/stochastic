#!/usr/bin/env python3
"""
Ralph loop runner for Vast training.

This script is intentionally separate from `run_vast.py`.
It repeatedly calls `run_vast.py`, parses metrics out of the newest
`vast_logs/vast_run_*.log`, appends a JSONL "memory" file, and stops when
any candidate beats a baseline.

The score we currently key off is nanochat's printed:
  CORE metric: <float>
which is produced by `nanochat/scripts/base_eval.py`.

Typical usage:
  python ralph_loop.py --baseline-variant baseline --candidates spiking,stochastic,both

Notes:
  - `run_vast.py` currently runs all variants sequentially on the remote machine.
    This loop assumes that behavior and extracts one CORE metric per variant run.
  - Optionally, this script can call an external "Claude Code" command to create/push
    a branch each iteration. See --claude-cmd and --claude-prompt-file.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


RE_VARIANT_START = re.compile(r"Remote:\s+running\s+(baseline|spiking|stochastic|both)\s+variant\.\.\.")
RE_CORE_METRIC = re.compile(r"\bCORE metric:\s*([0-9]*\.?[0-9]+)\b")
RE_CLAUDE_BRANCH = re.compile(r"^\s*BRANCH:\s*(\S+)\s*$", re.IGNORECASE)
RE_CLAUDE_COMMIT = re.compile(r"^\s*COMMIT:\s*([0-9a-f]{7,40})\s*$", re.IGNORECASE)
RE_CLAUDE_SUMMARY = re.compile(r"^\s*SUMMARY:\s*(.+?)\s*$", re.IGNORECASE)


@dataclass(frozen=True)
class VariantResult:
    variant: str
    core_metric: float
    log_path: str


def _utcnow_iso() -> str:
    return dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc).isoformat()


def _list_vast_logs(log_dir: Path) -> list[Path]:
    if not log_dir.exists():
        return []
    return sorted(log_dir.glob("vast_run_*.log"), key=lambda p: p.stat().st_mtime, reverse=True)


def _pick_newest_log_since(log_dir: Path, since_epoch: float) -> Path:
    logs = _list_vast_logs(log_dir)
    if not logs:
        raise SystemExit(f"No logs found in {log_dir}. Did `run_vast.py` create one?")
    for p in logs:
        if p.stat().st_mtime >= since_epoch - 5:
            return p
    return logs[0]


def parse_core_metrics_by_variant(log_text: str) -> dict[str, float]:
    """
    Parse CORE metrics per variant segment.

    We pick the *last* CORE metric seen for a variant segment (after
    "Remote: running <variant> variant..." and before the next variant marker).
    This is robust to cases where eval is re-run or printed multiple times.

    Vast logs can contain ANSI escape sequences and tmux status lines; this parser
    stays robust by simply scanning for the known markers.
    """
    out: dict[str, float] = {}
    current_variant: str | None = None

    for line in log_text.splitlines():
        m = RE_VARIANT_START.search(line)
        if m:
            current_variant = m.group(1)
            continue

        if current_variant is None:
            continue

        mm = RE_CORE_METRIC.search(line)
        if mm:
            out[current_variant] = float(mm.group(1))

    return out


def _append_jsonl(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _load_last_baseline(memory_path: Path, baseline_variant: str) -> float | None:
    if not memory_path.exists():
        return None
    last = None
    with memory_path.open("r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                obj = json.loads(ln)
            except Exception:
                continue
            if obj.get("type") == "result" and obj.get("baseline_variant") == baseline_variant:
                b = obj.get("baseline_core_metric")
                if isinstance(b, (int, float)):
                    last = float(b)
    return last


def run_once(run_vast_py: Path, *, repo_url: str | None = None, git_ref: str | None = None) -> int:
    cmd = [sys.executable, str(run_vast_py)]
    if repo_url:
        cmd += ["--repo-url", repo_url]
    if git_ref:
        cmd += ["--git-ref", git_ref]
    print(f"[ralph] running: {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd)
    return int(proc.returncode)


def run_claude_iteration(*, claude_cmd: str, prompt_text: str) -> dict:
    """
    Run an external Claude Code command and parse its required 3-line output.
    """
    proc = subprocess.run(
        claude_cmd,
        input=prompt_text,
        text=True,
        capture_output=True,
        shell=True,
    )
    out_lines = (proc.stdout or "").splitlines()
    branch = None
    commit = None
    summary = None
    for ln in out_lines:
        if branch is None:
            m = RE_CLAUDE_BRANCH.match(ln)
            if m:
                branch = m.group(1).strip()
                continue
        if commit is None:
            m = RE_CLAUDE_COMMIT.match(ln)
            if m:
                commit = m.group(1).strip()
                continue
        if summary is None:
            m = RE_CLAUDE_SUMMARY.match(ln)
            if m:
                summary = m.group(1).strip()
                continue

    if proc.returncode != 0:
        raise SystemExit(
            "[ralph] Claude command failed.\n"
            f"cmd: {claude_cmd}\n"
            f"exit_code: {proc.returncode}\n"
            f"stderr:\n{proc.stderr}"
        )

    if not branch or not commit or not summary:
        raise SystemExit(
            "[ralph] Claude output did not match required format.\n"
            "Expected lines (any order):\n"
            "BRANCH: <branch>\nCOMMIT: <sha>\nSUMMARY: <text>\n\n"
            f"stdout:\n{proc.stdout}\n\nstderr:\n{proc.stderr}"
        )

    return {"branch": branch, "commit": commit, "summary": summary}


def iter_loop(
    *,
    run_vast_py: Path,
    log_dir: Path,
    memory_path: Path,
    repo_url: str | None,
    baseline_variant: str,
    candidates: Iterable[str],
    min_improvement: float,
    max_iters: int,
    sleep_seconds: float,
    claude_cmd: str | None,
    claude_prompt_file: Path | None,
) -> None:
    candidates = [c.strip() for c in candidates if c.strip()]
    if baseline_variant not in {"baseline", "spiking", "stochastic", "both"}:
        raise SystemExit(f"Unknown baseline variant: {baseline_variant}")
    for c in candidates:
        if c not in {"baseline", "spiking", "stochastic", "both"}:
            raise SystemExit(f"Unknown candidate variant: {c}")

    baseline_core = _load_last_baseline(memory_path, baseline_variant)
    if baseline_core is not None:
        print(f"[ralph] loaded baseline from memory: {baseline_variant} CORE={baseline_core:.6f}")

    for it in range(1, max_iters + 1):
        claude_info = None
        git_ref = None
        if claude_cmd:
            if not claude_prompt_file or not claude_prompt_file.exists():
                raise SystemExit("[ralph] --claude-cmd requires --claude-prompt-file to exist.")
            prompt_text = claude_prompt_file.read_text(encoding="utf-8")
            claude_info = run_claude_iteration(claude_cmd=claude_cmd, prompt_text=prompt_text)
            git_ref = claude_info["branch"]
            _append_jsonl(
                memory_path,
                {
                    "type": "claude",
                    "ts_utc": _utcnow_iso(),
                    "iter": it,
                    "claude_cmd": claude_cmd,
                    "branch": claude_info["branch"],
                    "commit": claude_info["commit"],
                    "summary": claude_info["summary"],
                },
            )

        started = time.time()
        rc = run_once(run_vast_py, repo_url=repo_url, git_ref=git_ref)

        log_path = _pick_newest_log_since(log_dir, since_epoch=started)
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        metrics = parse_core_metrics_by_variant(log_text)

        # Baseline for comparison: prefer the metric from this run, otherwise the last saved one.
        if baseline_variant in metrics:
            baseline_core = metrics[baseline_variant]
        if baseline_core is None:
            raise SystemExit(
                f"[ralph] baseline CORE metric not found for variant={baseline_variant}. "
                f"Parsed variants: {sorted(metrics.keys())}. Log: {log_path}"
            )

        best_candidate = None
        best_score = None
        for c in candidates:
            if c in metrics:
                s = metrics[c]
                if best_score is None or s > best_score:
                    best_score = s
                    best_candidate = c

        event = {
            "type": "result",
            "ts_utc": _utcnow_iso(),
            "iter": it,
            "run_vast_returncode": rc,
            "log_path": str(log_path),
            "repo_url": repo_url,
            "git_ref": git_ref,
            "claude": claude_info,
            "parsed_core_by_variant": metrics,
            "baseline_variant": baseline_variant,
            "baseline_core_metric": baseline_core,
            "candidates": candidates,
            "best_candidate_variant": best_candidate,
            "best_candidate_core_metric": best_score,
            "min_improvement": min_improvement,
        }
        _append_jsonl(memory_path, event)

        # Stop condition.
        if best_candidate is not None and best_score is not None:
            improvement = best_score - baseline_core
            print(
                f"[ralph] iter={it} baseline({baseline_variant})={baseline_core:.6f} "
                f"best({best_candidate})={best_score:.6f} improvement={improvement:+.6f}",
                flush=True,
            )
            if improvement >= min_improvement:
                _append_jsonl(
                    memory_path,
                    {
                        "type": "stop",
                        "ts_utc": _utcnow_iso(),
                        "iter": it,
                        "reason": "beat_baseline",
                        "baseline_variant": baseline_variant,
                        "baseline_core_metric": baseline_core,
                        "winner_variant": best_candidate,
                        "winner_core_metric": best_score,
                        "improvement": improvement,
                        "log_path": str(log_path),
                    },
                )
                print("[ralph] stop: baseline beaten", flush=True)
                return
        else:
            print(f"[ralph] iter={it} no candidate metrics parsed; variants={sorted(metrics.keys())}", flush=True)

        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    _append_jsonl(
        memory_path,
        {
            "type": "stop",
            "ts_utc": _utcnow_iso(),
            "reason": "max_iters",
            "max_iters": max_iters,
        },
    )
    print("[ralph] stop: reached max iters", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-vast", default="run_vast.py", help="Path to run_vast.py (default: run_vast.py)")
    ap.add_argument("--log-dir", default="vast_logs", help="Directory where vast logs are written (default: vast_logs)")
    ap.add_argument(
        "--memory",
        default="ralph_memory.jsonl",
        help="JSONL memory file to append results to (default: ralph_memory.jsonl)",
    )
    ap.add_argument(
        "--repo-url",
        default=None,
        help="Repo URL to pass through to run_vast.py --repo-url (default: run_vast.py default)",
    )
    ap.add_argument("--baseline-variant", default="baseline", choices=["baseline", "spiking", "stochastic", "both"])
    ap.add_argument(
        "--candidates",
        default="spiking,stochastic,both",
        help="Comma-separated candidate variants to compare vs baseline (default: spiking,stochastic,both)",
    )
    ap.add_argument(
        "--min-improvement",
        type=float,
        default=0.0,
        help="Minimum CORE improvement over baseline required to stop (default: 0.0)",
    )
    ap.add_argument("--max-iters", type=int, default=50, help="Maximum loop iterations (default: 50)")
    ap.add_argument("--sleep-seconds", type=float, default=0.0, help="Sleep between iterations (default: 0)")
    ap.add_argument(
        "--claude-cmd",
        default=None,
        help=(
            "Shell command to invoke Claude Code. The prompt is sent on stdin. "
            "It must print BRANCH:/COMMIT:/SUMMARY: lines."
        ),
    )
    ap.add_argument(
        "--claude-prompt-file",
        default=None,
        help="Path to a prompt file to feed to Claude Code (default: none).",
    )
    args = ap.parse_args()

    run_vast_py = Path(args.run_vast).resolve()
    if not run_vast_py.exists():
        raise SystemExit(f"run_vast.py not found at: {run_vast_py}")

    log_dir = Path(args.log_dir).resolve()
    memory_path = Path(args.memory).resolve()
    prompt_file = Path(args.claude_prompt_file).resolve() if args.claude_prompt_file else None

    # Ensure we preserve the environment `run_vast.py` expects.
    # (VAST_API_KEY, HF_TOKEN, etc. are validated inside run_vast.py.)
    if not os.environ.get("VAST_API_KEY"):
        print("[ralph] warning: VAST_API_KEY not set; run_vast.py will fail.", file=sys.stderr)
    if not os.environ.get("HF_TOKEN"):
        print("[ralph] warning: HF_TOKEN not set; run_vast.py will fail.", file=sys.stderr)

    candidates = [s.strip() for s in str(args.candidates).split(",") if s.strip()]
    iter_loop(
        run_vast_py=run_vast_py,
        log_dir=log_dir,
        memory_path=memory_path,
        repo_url=args.repo_url,
        baseline_variant=args.baseline_variant,
        candidates=candidates,
        min_improvement=float(args.min_improvement),
        max_iters=int(args.max_iters),
        sleep_seconds=float(args.sleep_seconds),
        claude_cmd=args.claude_cmd,
        claude_prompt_file=prompt_file,
    )


if __name__ == "__main__":
    main()

