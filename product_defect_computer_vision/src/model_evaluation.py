"""
Model evaluation utilities for the Product Defect
Computer Vision project.
"""

from dataclasses import dataclass

import numpy as np
import torch

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)


@dataclass
class EvaluationResult:
    """
    Container for classification evaluation results.
    """

    labels: np.ndarray
    predictions: np.ndarray
    probabilities: np.ndarray

    accuracy: float

    precision: np.ndarray
    recall: np.ndarray
    f1: np.ndarray
    support: np.ndarray

    confusion_matrix: np.ndarray


def evaluate_classifier(
    model: torch.nn.Module,
    data_loader,
    device: torch.device,
    num_classes: int,
) -> EvaluationResult:
    """
    Evaluate a PyTorch multiclass classifier.

    Parameters
    ----------
    model:
        Trained PyTorch classification model.

    data_loader:
        DataLoader containing the evaluation dataset.

    device:
        Device used for inference.

    num_classes:
        Number of target classes.

    Returns
    -------
    EvaluationResult
        Predictions, probabilities, and classification metrics.
    """

    model = model.to(device)
    model.eval()

    all_labels = []
    all_predictions = []
    all_probabilities = []

    with torch.no_grad():

        for images, labels in data_loader:

            images = images.to(device)

            logits = model(images)

            probabilities = torch.softmax(
                logits,
                dim=1,
            )

            predictions = torch.argmax(
                logits,
                dim=1,
            )

            all_labels.extend(
                labels.cpu().numpy()
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_probabilities.extend(
                probabilities.cpu().numpy()
            )

    labels_array = np.asarray(
        all_labels
    )

    predictions_array = np.asarray(
        all_predictions
    )

    probabilities_array = np.asarray(
        all_probabilities
    )

    accuracy = accuracy_score(
        labels_array,
        predictions_array,
    )

    precision, recall, f1, support = (
        precision_recall_fscore_support(
            labels_array,
            predictions_array,
            labels=range(num_classes),
            zero_division=0,
        )
    )

    matrix = confusion_matrix(
        labels_array,
        predictions_array,
        labels=range(num_classes),
    )

    return EvaluationResult(
        labels=labels_array,
        predictions=predictions_array,
        probabilities=probabilities_array,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        support=support,
        confusion_matrix=matrix,
    )