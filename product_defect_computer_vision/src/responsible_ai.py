"""
Responsible AI assessment utilities for the Product Defect
Computer Vision project.
"""

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@dataclass
class PredictionRecord:
    actual_label: int
    predicted_label: int
    confidence: float
    probabilities: np.ndarray


@dataclass
class ConfidenceSummary:
    total_predictions: int
    correct_predictions: int
    incorrect_predictions: int
    mean_confidence: float
    mean_correct_confidence: float | None
    mean_incorrect_confidence: float | None
    high_confidence_errors: int


def collect_predictions(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
) -> list[PredictionRecord]:
    """
    Collect predictions, probabilities, and confidence
    values for a classification dataset.
    """

    model.eval()

    records = []

    with torch.no_grad():

        for images, labels in data_loader:

            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)

            probabilities = torch.softmax(
                logits,
                dim=1,
            )

            confidence, predictions = torch.max(
                probabilities,
                dim=1,
            )

            for index in range(labels.shape[0]):

                records.append(
                    PredictionRecord(
                        actual_label=int(
                            labels[index].item()
                        ),
                        predicted_label=int(
                            predictions[index].item()
                        ),
                        confidence=float(
                            confidence[index].item()
                        ),
                        probabilities=(
                            probabilities[index]
                            .detach()
                            .cpu()
                            .numpy()
                        ),
                    )
                )

    return records


def summarize_confidence(
    records: list[PredictionRecord],
    high_confidence_threshold: float = 0.90,
) -> ConfidenceSummary:
    """
    Summarize model confidence for correct and
    incorrect predictions.
    """

    if not records:
        raise ValueError(
            "records cannot be empty."
        )

    correct_confidences = [
        record.confidence
        for record in records
        if (
            record.actual_label
            == record.predicted_label
        )
    ]

    incorrect_confidences = [
        record.confidence
        for record in records
        if (
            record.actual_label
            != record.predicted_label
        )
    ]

    high_confidence_errors = sum(
        1
        for record in records
        if (
            record.actual_label
            != record.predicted_label
            and record.confidence
            >= high_confidence_threshold
        )
    )

    return ConfidenceSummary(
        total_predictions=len(records),
        correct_predictions=len(
            correct_confidences
        ),
        incorrect_predictions=len(
            incorrect_confidences
        ),
        mean_confidence=float(
            np.mean(
                [
                    record.confidence
                    for record in records
                ]
            )
        ),
        mean_correct_confidence=(
            float(np.mean(correct_confidences))
            if correct_confidences
            else None
        ),
        mean_incorrect_confidence=(
            float(np.mean(incorrect_confidences))
            if incorrect_confidences
            else None
        ),
        high_confidence_errors=(
            high_confidence_errors
        ),
    )


def get_error_indices(
    records: list[PredictionRecord],
) -> list[int]:
    """
    Return indices corresponding to incorrect
    predictions.
    """

    return [
        index
        for index, record in enumerate(records)
        if (
            record.actual_label
            != record.predicted_label
        )
    ]


def get_false_negative_indices(
    records: list[PredictionRecord],
    positive_class_index: int,
) -> list[int]:
    """
    Return indices where the actual positive class
    was predicted as another class.
    """

    return [
        index
        for index, record in enumerate(records)
        if (
            record.actual_label
            == positive_class_index
            and record.predicted_label
            != positive_class_index
        )
    ]