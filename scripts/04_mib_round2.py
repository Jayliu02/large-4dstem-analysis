"""Run or verify multi-ROI detection and provisional Bragg coordinate correction."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fourdstem_pipeline.mib_round2 import main

if __name__ == "__main__":
    main()
