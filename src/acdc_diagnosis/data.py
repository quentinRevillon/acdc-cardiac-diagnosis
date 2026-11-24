"""Load ACDC segmentations and patient metadata.

Expected layout (as distributed for the challenge):
    data/metaDataTrain.csv          Id, Category, Height, Weight
    data/metaDataTest.csv           Id, Height, Weight
    data/Train/001/001_ED_seg.nii   one folder per patient, ED and ES phases
    data/Test/101/101_ES_seg.nii

Usage:
    metadata = load_metadata("data/metaDataTrain.csv")
    patient = load_patient("data/Train", metadata.iloc[0])
    seg, voxel_volume = patient["ED"]
"""

from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

PHASES = ("ED", "ES")


def load_metadata(csv_path):
    """Read the metadata CSV into a DataFrame indexed in file order.

    Args:
        csv_path (str | Path): path to metaDataTrain.csv or metaDataTest.csv.

    Returns:
        pandas.DataFrame: columns Id, Height, Weight and Category (train only).
    """
    return pd.read_csv(csv_path)


def load_segmentation(nifti_path):
    """Load a segmentation volume and its voxel volume.

    Args:
        nifti_path (str | Path): path to a *_seg.nii file.

    Returns:
        tuple[numpy.ndarray, float]: label volume (H, W, slices) and voxel volume in mm^3.
    """
    image = nib.load(nifti_path)
    return image.get_fdata(), float(np.prod(image.header.get_zooms()))


def load_patient(split_dir, row):
    """Load the ED and ES segmentations and the metadata of one patient.

    Args:
        split_dir (str | Path): data/Train or data/Test.
        row (pandas.Series): one row of the metadata DataFrame.

    Returns:
        dict: {"id", "height", "weight", "ED": (seg, voxel_volume), "ES": (seg, voxel_volume)}.
    """
    patient_id = f"{int(row['Id']):03d}"
    folder = Path(split_dir) / patient_id
    patient = {"id": patient_id, "height": row["Height"], "weight": row["Weight"]}
    for phase in PHASES:
        patient[phase] = load_segmentation(folder / f"{patient_id}_{phase}_seg.nii")
    return patient


def load_split(split_dir, csv_path):
    """Load every patient listed in a metadata file.

    Args:
        split_dir (str | Path): data/Train or data/Test.
        csv_path (str | Path): matching metadata CSV.

    Returns:
        tuple[list[dict], pandas.DataFrame]: patients and their metadata.
    """
    metadata = load_metadata(csv_path)
    patients = [load_patient(split_dir, row) for _, row in metadata.iterrows()]
    return patients, metadata
