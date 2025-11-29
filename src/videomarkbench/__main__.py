"""
Entry point for running videomarkbench as a module.

This allows running: python -m videomarkbench.main
"""

import sys
from pathlib import Path

# Ensure src directory is in path
src_dir = Path(__file__).parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))
if str(src_dir / "videoseal") not in sys.path:
    sys.path.insert(0, str(src_dir / "videoseal"))

# Import and run main
from videomarkbench.main import main  # noqa: E402

if __name__ == "__main__":
    main()
