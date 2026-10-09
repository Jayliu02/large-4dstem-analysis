"""Run real-background injection diagnostics or import a human peak review."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fourdstem_pipeline.mib_round4 import main

if __name__ == "__main__":
    main()
