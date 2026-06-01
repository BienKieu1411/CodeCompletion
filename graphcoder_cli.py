from __future__ import annotations

import os
import sys


def _bootstrap_src_path() -> None:
    root = os.path.dirname(os.path.abspath(__file__))
    src_path = os.path.join(root, "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)


def main() -> int:
    _bootstrap_src_path()
    from graphcoder_rl.cli.graphcoder_cli import main as _main

    return _main()


if __name__ == "__main__":
    raise SystemExit(main())
