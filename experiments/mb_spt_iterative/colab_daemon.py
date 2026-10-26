#!/usr/bin/env python3
"""Persistent GPU evaluation daemon on Colab A100 for sub-10-second MB-SPT iterations."""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, "/content")
from colab_init_engine import ROOT_DIR, ensure_deps, init_globals

CMD_FILE = ROOT_DIR / "next_cmd.json"
STATUS_FILE = ROOT_DIR / "daemon_status.json"


def write_status(state: str, **extra: object) -> None:
    ROOT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"state": state, "timestamp": time.time(), **extra}
    STATUS_FILE.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    write_status("installing_deps")
    ensure_deps()
    # Spawn fresh process if torch was upgraded so C++/CUDA extensions match cleanly
    if os.environ.get("MB_SPT_DEPS_READY") != "1":
        env = dict(os.environ)
        env["MB_SPT_DEPS_READY"] = "1"
        os.execve(sys.executable, [sys.executable, __file__], env)

    write_status("loading_engine")
    state_ns: dict[str, object] = {}
    init_globals(state_ns)
    write_status("ready")
    print("Daemon is READY and waiting for jobs...", flush=True)

    while True:
        if CMD_FILE.exists():
            try:
                cmd = json.loads(CMD_FILE.read_text(encoding="utf-8"))
                CMD_FILE.unlink()
            except Exception as exc:
                print(f"Failed reading cmd: {exc}", flush=True)
                time.sleep(0.5)
                continue

            action = cmd.get("action", "run_script")
            if action == "stop":
                write_status("stopped")
                print("Stopping daemon.", flush=True)
                break

            script_path = Path(cmd["script"])
            tag = cmd.get("tag", script_path.stem)
            subset = int(cmd.get("subset_per_benchmark", 16))
            write_status("running", tag=tag, script=str(script_path), subset=subset)
            print(f"\n=== Running job {tag} ({script_path}, subset={subset}) ===", flush=True)
            t0 = time.time()
            try:
                code = script_path.read_text(encoding="utf-8")
                job_ns = dict(state_ns)
                job_ns["__name__"] = "__job__"
                exec(compile(code, str(script_path), "exec"), job_ns)
                if "run_full_evaluation" in job_ns:
                    res = job_ns["run_full_evaluation"](state_ns, tag=tag)
                elif "run_iteration" in job_ns:
                    res = job_ns["run_iteration"](state_ns, subset_per_benchmark=subset, tag=tag)
                else:
                    raise ValueError("Script contains neither run_full_evaluation nor run_iteration")
                write_status(
                    "idle",
                    last_tag=tag,
                    last_status="ok",
                    elapsed_sec=round(time.time() - t0, 2),
                    overall=res.get("overall_by_arm"),
                    bootstrap=res.get("paired_bootstrap_vs_aligncoder_code"),
                )
            except Exception as exc:
                tb = traceback.format_exc()
                print(f"ERROR in job {tag}:\n{tb}", flush=True)
                write_status("idle", last_tag=tag, last_status="error", error=str(exc), traceback=tb)
        time.sleep(0.5)


if __name__ == "__main__":
    main()
