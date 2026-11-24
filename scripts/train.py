"""Train and evaluate a diagnosis classifier, optionally write a Kaggle submission.

Pipeline: features from the ground-truth training segmentations, stratified
90/10 train/hold-out split, preprocessing and feature selection fitted on the
90%, grid-searched classifier, 5-fold cross-validated accuracy on the 90% and
accuracy on the 10% hold-out. For the submission, the LV of the test patients is
reconstructed first, then the classifier fitted on the 90% predicts their class.

Usage:
    python scripts/train.py --data-dir data --model rf
    python scripts/train.py --data-dir data --model voting --submission submission.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_predict, train_test_split

from acdc_diagnosis.classification import CLASS_NAMES, build_model, build_preprocessor, cross_validator
from acdc_diagnosis.data import PHASES, load_split
from acdc_diagnosis.features import features_table
from acdc_diagnosis.segmentation import fill_lv_volume


def parse_args():
    """Parse command-line arguments.

    Returns:
        argparse.Namespace: data_dir, model, submission, seed.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--model", choices=["rf", "mlp", "svm", "xgb", "voting"], default="rf")
    parser.add_argument("--submission", type=Path, help="write test predictions to this CSV")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def test_features(data_dir):
    """Features of the test patients after LV reconstruction.

    Args:
        data_dir (Path): dataset root.

    Returns:
        tuple[pandas.DataFrame, pandas.DataFrame]: features and test metadata.
    """
    patients, metadata = load_split(data_dir / "Test", data_dir / "metaDataTest.csv")
    for patient in patients:
        for phase in PHASES:
            seg, voxel_volume = patient[phase]
            patient[phase] = (fill_lv_volume(seg), voxel_volume)
    return features_table(patients), metadata


def report(title, y_true, y_pred):
    """Print accuracy and per-class metrics.

    Args:
        title (str): section title.
        y_true (numpy.ndarray): true labels.
        y_pred (numpy.ndarray): predicted labels.
    """
    print(f"\n{title}: accuracy {accuracy_score(y_true, y_pred):.2f} (n={len(y_true)})")
    labels = range(len(CLASS_NAMES))
    print(classification_report(y_true, y_pred, labels=labels, target_names=CLASS_NAMES, zero_division=0))


def main():
    """Run the classification pipeline."""
    args = parse_args()
    patients, metadata = load_split(args.data_dir / "Train", args.data_dir / "metaDataTrain.csv")
    X, y = features_table(patients), metadata["Category"].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, stratify=y, random_state=args.seed
    )

    preprocessor = build_preprocessor(seed=args.seed).fit(X_train, y_train)
    names = preprocessor[:-1].get_feature_names_out()[preprocessor[-1].get_support()]
    print(f"{len(names)} selected features: {', '.join(names)}")
    S_train, S_test = preprocessor.transform(X_train), preprocessor.transform(X_test)

    model = build_model(args.model, S_train, y_train, args.seed)
    cv_predictions = cross_val_predict(model, S_train, y_train, cv=cross_validator(args.seed))
    report("5-fold cross-validation", y_train, cv_predictions)
    model.fit(S_train, y_train)
    report("Hold-out", y_test, model.predict(S_test))

    if args.submission:
        X_sub, test_metadata = test_features(args.data_dir)
        predictions = model.predict(preprocessor.transform(X_sub))
        submission = pd.DataFrame({"Id": test_metadata["Id"], "Category": predictions})
        submission.to_csv(args.submission, index=False)
        counts = np.bincount(predictions, minlength=len(CLASS_NAMES))
        print(f"Wrote {args.submission}: class counts {counts}")


if __name__ == "__main__":
    main()
