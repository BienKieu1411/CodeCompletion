#!/usr/bin/env python3
"""Submit job to Colab daemon via REST upload and poll status via REST download (zero WebSockets)."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

COLAB_BIN = "/Users/kieugiangbien/bienkieu_env/bin/colab"


def colab_upload(session: str, local_path: str | Path, remote_path: str) -> None:
    subprocess.check_call([COLAB_BIN, "upload", "-s", session, str(local_path), remote_path])


def colab_download(session: str, remote_path: str, local_path: str | Path) -> bool:
    res = subprocess.run(
        [COLAB_BIN, "download", "-s", session, remote_path, str(local_path)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return res.returncode == 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-s", "--session", default="cceval")
    ap.add_argument("--script", required=True, help="Local script path to upload and run")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--subset", type=int, required=True)
    ap.add_argument("--timeout", type=float, default=600.0)
    args = ap.parse_args()

    local_script = Path(args.script)
    remote_script = f"/content/{local_script.name}"
    colab_upload(args.session, local_script, remote_script)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        cmd_file = tmp_dir / "next_cmd.json"
        cmd_file.write_text(
            json.dumps(
                {
                    "action": "run_script",
                    "script": remote_script,
                    "tag": args.tag,
                    "subset_per_benchmark": args.subset,
                }
            ),
            encoding="utf-8",
        )
        colab_upload(args.session, cmd_file, "/content/mb_spt/next_cmd.json")
        print(f"Submitted job tag={args.tag} (subset={args.subset}/benchmark) via REST...", flush=True)

        t0 = time.time()
        last_state = None
        status_local = tmp_dir / "daemon_status.json"
        while time.time() - t0 < args.timeout:
            if colab_download(args.session, "/content/mb_spt/daemon_status.json", status_local):
                try:
                    st = json.loads(status_local.read_text(encoding="utf-8"))
                except Exception:
                    st = {}
                state = st.get("state")
                if state != last_state:
                    print(f"[{time.time() - t0:5.1f}s] Daemon state: {state}", flush=True)
                    last_state = state
                if state == "idle" and st.get("last_tag") == args.tag:
                    if st.get("last_status") != "ok":
                        print(f"Job {args.tag} FAILED:\n{st.get('traceback', st.get('error'))}", flush=True)
                        sys.exit(1)
                    out_dir = Path(__file__).resolve().parent / "outputs"
                    out_dir.mkdir(parents=True, exist_ok=True)
                    local_metrics = out_dir / f"{args.tag}_metrics.json"
                    colab_download(
                        args.session,
                        f"/content/mb_spt/outputs/{args.tag}_metrics.json",
                        local_metrics,
                    )
                    print(f"\n=== {args.tag} METRICS ===", flush=True)
                    print(local_metrics.read_text(encoding="utf-8"), flush=True)
                    return
            time.sleep(4.0)
        raise TimeoutError(f"Timed out waiting for job {args.tag}")


if __name__ == "__main__":
    main()
