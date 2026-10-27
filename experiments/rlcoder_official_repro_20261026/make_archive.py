
import shutil, os
from pathlib import Path
out_path = Path("/content/rlcoder-results/full-1793066757")
if out_path.is_dir():
    shutil.make_archive("/content/rlcoder-results/full-1793066757", "zip", "/content/rlcoder-results/full-1793066757")
    print("Created archive /content/rlcoder-results/full-1793066757.zip", flush=True)
