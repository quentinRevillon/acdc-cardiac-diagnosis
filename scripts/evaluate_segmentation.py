"""Evaluate the LV reconstruction on the training set.

The LV label is removed from the ground-truth segmentations of the training
patients, reconstructed with acdc_diagnosis.segmentation, and compared with the
ground truth (exact equality of the label volumes).

Usage:
    python scripts/evaluate_segmentation.py --data-dir data
"""

import argparse
from pathlib import Path

import numpy as np

from acdc_diagnosis.data import PHASES, load_split
from acdc_diagnosis.segmentation import fill_lv_volume, remove_lv


def mismatched_slices(seg):
    """Indices of the slices where the reconstructed LV differs from the ground truth.

    Args:
        seg (numpy.ndarray): ground-truth label volume (H, W, slices).

    Returns:
        list[int]: slice indices with at least one differing pixel.
    """
    diff = fill_lv_volume(remove_lv(seg)) != seg
    return np.flatnonzero(diff.any(axis=(0, 1))).tolist()


def main():
    """Print the patients and slices whose LV is not reconstructed exactly."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args()

    patients, _ = load_split(args.data_dir / "Train", args.data_dir / "metaDataTrain.csv")
    for phase in PHASES:
        failures = {p["id"]: mismatched_slices(p[phase][0]) for p in patients}
        failures = {pid: slices for pid, slices in failures.items() if slices}
        exact = len(patients) - len(failures)
        print(f"{phase}: {exact}/{len(patients)} patients reconstructed exactly")
        for pid, slices in failures.items():
            print(f"  patient {pid}: slices {slices}")


if __name__ == "__main__":
    main()
