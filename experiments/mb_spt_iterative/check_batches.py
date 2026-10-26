from pathlib import Path
LOG_FILE = Path("/content/mb_spt/daemon.log")
if LOG_FILE.exists():
    text = LOG_FILE.read_text(errors="replace")
    import re
    batches = re.findall(r"\[Batch \d+\] Processing tasks.*", text)
    progress = re.findall(r"Progress Running EM \[.*?\].*", text)
    print("Found batches:", len(batches))
    for b in batches[-5:]:
        print(" ", b)
    print("\nRecent progress reports:")
    for p in progress[-6:]:
        print(" ", p)
