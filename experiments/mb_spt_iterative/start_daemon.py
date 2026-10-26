#!/usr/bin/env python3
import sys
sys.path.insert(0, "/content")
import submit_job
submit_job.ensure_daemon()
print("Daemon spawned in background.", flush=True)
