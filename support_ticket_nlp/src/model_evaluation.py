"""
Reusable classification evaluation utilities.
"""

from typing import Dict

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


def calculate_classification_metrics(
    y_true,
    y_pred,
) -> Dict[str, float]:
    """
    Calculate core classification metrics.
    """

    return {
        "accuracy": accuracy_score(
            y_true,
            y_pred,
        ),
        "macro_f1": f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),
        "weighted_f1": f1_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        ),
    }


def build_classification_report(
    y_true,
    y_pred,
) -> pd.DataFrame:
    """
    Build a classification report as a
    display-friendly pandas DataFrame.
    """

    report_dict = classification_report(
        y_true,
        y_pred,
        output_dict=True,
        zero_division=0,
    )

    return (
        pd.DataFrame(report_dict)
        .transpose()
        .rename_axis("class")
        .reset_index()
    )


def build_confusion_matrix(
    y_true,
    y_pred,
    class_labels,
) -> pd.DataFrame:
    """
    Build a labeled confusion matrix.
    """

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=class_labels,
    )

    return (
        pd.DataFrame(
            matrix,
            index=class_labels,
            columns=[
                f"Predicted_{label}"
                for label in class_labels
            ],
        )
        .rename_axis("Actual_Category")
        .reset_index()
    )


def build_prediction_results(
    ticket_ids,
    text,
    y_true,
    y_pred,
    probabilities,
    class_labels,
) -> pd.DataFrame:
    """
    Build row-level prediction results including
    predicted class probabilities.
    """

    results = pd.DataFrame(
        {
            "ticket_id": ticket_ids,
            "clean_text": text,
            "category": y_true,
            "predicted_category": y_pred,
        }
    )

    results["is_correct"] = (
        results["category"]
        == results["predicted_category"]
    )

    results["prediction_confidence"] = (
        np.max(
            probabilities,
            axis=1,
        )
    )

    for index, label in enumerate(
        class_labels
    ):
        column_name = (
            f"prob_{label.lower()}"
        )

        results[column_name] = (
            probabilities[:, index]
        )

    return results