#!/usr/bin/env python3
"""Build/validate linked RoboDojo history indices without copying raw data."""
import argparse
from pathlib import Path

from starVLA.utils.robodojo_memory import RoboDojoMemorySpec
from starVLA.utils.robodojo_memory_sidecar import build_sidecar, validate_sidecar


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--data-root", type=Path, required=True, help="Original LeRobot dataset (contains meta/ and data/)")
    parser.add_argument("--sidecar", type=Path, required=True, help="New versioned directory outside the original dataset")
    parser.add_argument("--recent-records", type=int, default=16)
    parser.add_argument("--decision-stride", type=int, default=8)
    parser.add_argument("--use-pose", action="store_true", help="Indices only; genuine FK still comes from the spatial sidecar")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--no-visual", action="store_true", help="Build legacy index-only sidecar")
    parser.add_argument("--validate", action="store_true", help="Validate an existing sidecar without writing anything")
    parser.add_argument("--resume-from", type=Path, help="Resume an interrupted unpublished .building-* directory")
    args = parser.parse_args()
    spec = RoboDojoMemorySpec(args.recent_records, args.decision_stride, args.use_pose)
    if not args.validate:
        build_sidecar(args.data_root, args.sidecar, spec, args.workers, visual=not args.no_visual,
                      resume_from=args.resume_from)
    validate_sidecar(args.sidecar, args.data_root, spec)


if __name__ == "__main__":
    main()
