"""
Pretrained vision model utilities for the Product Defect
Computer Vision project.
"""

import torch.nn as nn

from torchvision.models import (
    ResNet18_Weights,
    resnet18,
)


def build_resnet18_classifier(
    num_classes: int,
    freeze_backbone: bool = True,
) -> nn.Module:
    """
    Build a pretrained ResNet18 classifier.

    Parameters
    ----------
    num_classes:
        Number of output classes.

    freeze_backbone:
        If True, freeze pretrained ResNet18 parameters
        before replacing the final classifier.

    Returns
    -------
    nn.Module
        ResNet18 adapted to the target classification task.
    """

    if num_classes < 2:
        raise ValueError(
            "num_classes must be at least 2."
        )

    weights = ResNet18_Weights.DEFAULT

    model = resnet18(
        weights=weights
    )

    if freeze_backbone:

        for parameter in model.parameters():
            parameter.requires_grad = False

    num_features = (
        model.fc.in_features
    )

    model.fc = nn.Linear(
        in_features=num_features,
        out_features=num_classes,
    )

    return model