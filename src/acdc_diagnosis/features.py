"""Cardiac biomarkers computed from ED/ES segmentations and patient metadata.

The 42 features follow the static parameters used by Khened et al. (2017),
Wolterink et al. (2017) and Isensee et al. (2017): ventricular volumes, ejection
fractions, myocardial mass, volume ratios, volumes indexed to body surface area,
perimeters, circularity and myocardial wall thickness. Volumes are in mL, mass in g,
perimeters and thicknesses in pixels (in-plane).

Usage:
    from acdc_diagnosis.features import features_table
    table = features_table(patients)        # patients from data.load_split
"""

import cv2
import numpy as np
import pandas as pd

from acdc_diagnosis import LV, MYO, RV

MYOCARDIUM_DENSITY = 1.05  # g/mL


def volume_ml(seg, label, voxel_volume):
    """Volume of one structure.

    Args:
        seg (numpy.ndarray): label volume.
        label (int): structure label.
        voxel_volume (float): voxel volume in mm^3.

    Returns:
        float: volume in mL.
    """
    return np.sum(seg == label) * voxel_volume / 1000


def ejection_fraction(ed_volume, es_volume):
    """Ejection fraction (EDV - ESV) / EDV.

    Args:
        ed_volume (float): end-diastolic volume.
        es_volume (float): end-systolic volume.

    Returns:
        float: ejection fraction in [0, 1].
    """
    return (ed_volume - es_volume) / ed_volume


def body_surface_area(height, weight):
    """Body surface area with the Mosteller formula.

    Args:
        height (float): height in cm.
        weight (float): weight in kg.

    Returns:
        float: body surface area in m^2.
    """
    return np.sqrt(height * weight / 3600)


def slice_perimeter(seg_slice, label):
    """Perimeter of the first external contour of a structure in a 2D slice.

    Args:
        seg_slice (numpy.ndarray): 2D label slice.
        label (int): structure label.

    Returns:
        float: perimeter in pixels, 0 if the structure is absent.
    """
    mask = (seg_slice == label).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return cv2.arcLength(contours[0], True) if contours else 0.0


def slice_circularity(seg_slice, label):
    """Circularity 4*pi*area / perimeter^2 of a structure in a 2D slice.

    Args:
        seg_slice (numpy.ndarray): 2D label slice.
        label (int): structure label.

    Returns:
        float: circularity, 0 if the structure is absent.
    """
    perimeter = slice_perimeter(seg_slice, label)
    return 4 * np.pi * np.sum(seg_slice == label) / perimeter**2 if perimeter else 0.0


def slice_wall_thickness(seg_slice):
    """Distances from the outer to the inner myocardial contour in a 2D slice.

    Args:
        seg_slice (numpy.ndarray): 2D label slice.

    Returns:
        list[float]: one distance (pixels) per outer contour point, empty if the
        myocardium does not form a ring.
    """
    mask = (seg_slice == MYO).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) < 2:
        return []
    outer, inner = sorted(contours, key=cv2.contourArea, reverse=True)[:2]
    return [abs(cv2.pointPolygonTest(inner, (int(p[0][0]), int(p[0][1])), True)) for p in outer]


def per_slice(seg, function, *args):
    """Apply a slice-level measure to every slice of a volume.

    Args:
        seg (numpy.ndarray): label volume (H, W, slices).
        function (callable): measure taking (seg_slice, *args).
        *args: extra arguments of the measure.

    Returns:
        list: one value per slice.
    """
    return [function(np.ascontiguousarray(seg[:, :, k]), *args) for k in range(seg.shape[2])]


def phase_features(seg, voxel_volume):
    """Shape and volume measures of one cardiac phase.

    Args:
        seg (numpy.ndarray): label volume with RV, MYO and LV labels.
        voxel_volume (float): voxel volume in mm^3.

    Returns:
        dict[str, float]: measures keyed by structure and measure name.
    """
    thickness = np.concatenate([np.asarray(t) for t in per_slice(seg, slice_wall_thickness)])
    out = {
        "RV_volumes": volume_ml(seg, RV, voxel_volume),
        "MYO_volumes": volume_ml(seg, MYO, voxel_volume),
        "LV_volumes": volume_ml(seg, LV, voxel_volume),
        "MYO_thickness_min": thickness.min(),
        "MYO_thickness_mean": thickness.mean(),
        "MYO_thickness_max": thickness.max(),
        "MYO_thickness_std": thickness.std(),
    }
    for name, label in (("RV", RV), ("MYO", MYO)):
        perimeters = per_slice(seg, slice_perimeter, label)
        out[f"{name}_max_perimeter"] = np.max(perimeters)
        out[f"{name}_mean_perimeter"] = np.mean(perimeters)
        out[f"{name}_mean_circularity"] = np.mean(per_slice(seg, slice_circularity, label))
    return out


def patient_features(patient):
    """The 42 diagnosis features of one patient.

    Args:
        patient (dict): output of data.load_patient, with LV labels present.

    Returns:
        dict[str, float]: feature name -> value.
    """
    ed, es = (phase_features(*patient[phase]) for phase in ("ED", "ES"))
    bsa = body_surface_area(patient["height"], patient["weight"])
    features = {f"ED_{k}": v for k, v in ed.items()} | {f"ES_{k}": v for k, v in es.items()}
    features |= {
        "height": patient["height"],
        "weight": patient["weight"],
        "EF_LV": ejection_fraction(ed["LV_volumes"], es["LV_volumes"]),
        "EF_RV": ejection_fraction(ed["RV_volumes"], es["RV_volumes"]),
        "ED_MYO_mass": ed["MYO_volumes"] * MYOCARDIUM_DENSITY,
        "ED_RV_volumes_div_ED_LV_volumes": ed["RV_volumes"] / ed["LV_volumes"],
        "ES_RV_volumes_div_ED_LV_volumes": es["RV_volumes"] / ed["LV_volumes"],
        "ED_MYO_volumes_div_ED_LV_volumes": ed["MYO_volumes"] / ed["LV_volumes"],
        "ES_MYO_volumes_div_ES_LV_volumes": es["MYO_volumes"] / es["LV_volumes"],
        "ED_MYO_volumes_div_ES_LV_volumes": ed["MYO_volumes"] / es["LV_volumes"],
    }
    for phase, measures in (("ED", ed), ("ES", es)):
        for name in ("RV", "MYO", "LV"):
            features[f"{phase}_{name}_volume_by_BSA"] = measures[f"{name}_volumes"] / bsa
    return features


def features_table(patients):
    """Feature table of a list of patients.

    Args:
        patients (list[dict]): outputs of data.load_patient, with LV labels present.

    Returns:
        pandas.DataFrame: one row per patient, 42 feature columns.
    """
    return pd.DataFrame([patient_features(p) for p in patients])
