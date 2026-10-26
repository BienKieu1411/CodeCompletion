#!/usr/bin/env python3
"""Print process state and the recent detached Colab runner log."""

from __future__ import annotations

import json
import os
from pathlib import Path


ROOT = Path("/content/rlcoder_dependency_graph")
pid_file = ROOT / "runner.pid"
log_file = ROOT / "runner.log"
pid = int(pid_file.read_text().strip()) if pid_file.exists() else None
alive = False
cmdline = ""
if pid is not None:
    try:
        os.kill(pid, 0)
        alive = True
    except ProcessLookupError:
        alive = False
    proc_cmdline = Path(f"/proc/{pid}/cmdline")
    if proc_cmdline.exists():
        cmdline = proc_cmdline.read_bytes().replace(b"\0", b" ").decode("utf-8", "replace")
tail = ""
if log_file.exists():
    with log_file.open("rb") as handle:
        handle.seek(max(0, log_file.stat().st_size - 4000))
        tail = handle.read().decode("utf-8", "replace")
print(json.dumps({"pid": pid, "alive": alive, "cmdline": cmdline,
                  "log_bytes": log_file.stat().st_size if log_file.exists() else 0,
                  "log_tail": tail}, ensure_ascii=False), flush=True)
