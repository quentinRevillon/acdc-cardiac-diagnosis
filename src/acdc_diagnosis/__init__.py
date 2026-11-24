"""Cardiac pathology classification on the ACDC dataset.

Modules:
    data            load segmentations and patient metadata
    segmentation    reconstruct the left ventricle from the myocardium mask
    features        compute cardiac biomarkers from the segmentations
    classification  preprocessing, feature selection and classifiers

Usage:
    from acdc_diagnosis.data import load_patient
    from acdc_diagnosis.segmentation import fill_lv_volume
"""

RV, MYO, LV = 1, 2, 3
