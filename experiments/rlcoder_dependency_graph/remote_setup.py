#!/usr/bin/env python3
"""Extract the uploaded experiment bundle and launch it beyond CLI disconnects."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path


ROOT = Path("/content/rlcoder_dependency_graph")
ARCHIVE = Path("/content/rlcoder_dependency_graph_bundle.tar.gz")

if not ARCHIVE.is_file():
    raise FileNotFoundError(f"Upload the bundle first: {ARCHIVE}")
if ROOT.exists():
    raise FileExistsError(f"Refusing to reuse existing experiment directory: {ROOT}")
ROOT.mkdir(parents=True)
with tarfile.open(ARCHIVE, "r:gz") as bundle:
    bundle.extractall(ROOT, filter="data")

runner = ROOT / "runner.py"
if not runner.is_file():
    raise FileNotFoundError(f"Bundle did not contain {runner}")
log_path = ROOT / "runner.log"
pid_path = ROOT / "runner.pid"
with log_path.open("ab", buffering=0) as log_file:
    process = subprocess.Popen(
        [sys.executable, str(runner)], cwd=str(ROOT), stdin=subprocess.DEVNULL,
        stdout=log_file, stderr=subprocess.STDOUT, start_new_session=True,
        close_fds=True, env=os.environ.copy())
pid_path.write_text(f"{process.pid}\n", encoding="utf-8")
print(json.dumps({"status": "launched", "pid": process.pid,
                  "root": str(ROOT), "log": str(log_path)}), flush=True)
