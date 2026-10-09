"""Third-round weak-peak detection and calibration-readiness entry point."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fourdstem_pipeline.mib_round3 import main

if __name__ == "__main__":
    main()
