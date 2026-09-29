"""
Image preprocessing utilities for the Product Defect
Computer Vision project.

Separate preprocessing pipelines are provided for:

    - custom CNN
    - pretrained ResNet18

Training transforms may include augmentation.

Validation and test transforms remain deterministic.
"""

from torchvision import transforms

from torchvision.models import (
    ResNet18_Weights,
)

from src.project_config import (
    CUSTOM_NORMALIZE_MEAN,
    CUSTOM_NORMALIZE_STD,
    IMAGE_SIZE,
)


# ============================================================
# CUSTOM CNN TRANSFORMS
# ============================================================

def build_custom_train_transform():
    """
    Build the training transform for the custom CNN.

    Includes lightweight image augmentation.
    """

    return transforms.Compose(
        [
            transforms.Resize(
                IMAGE_SIZE
            ),

            transforms.RandomHorizontalFlip(
                p=0.5
            ),

            transforms.RandomRotation(
                degrees=10
            ),

            transforms.ToTensor(),

            transforms.Normalize(
                mean=CUSTOM_NORMALIZE_MEAN,
                std=CUSTOM_NORMALIZE_STD,
            ),
        ]
    )


def build_custom_eval_transform():
    """
    Build deterministic validation/test preprocessing
    for the custom CNN.
    """

    return transforms.Compose(
        [
            transforms.Resize(
                IMAGE_SIZE
            ),

            transforms.ToTensor(),

            transforms.Normalize(
                mean=CUSTOM_NORMALIZE_MEAN,
                std=CUSTOM_NORMALIZE_STD,
            ),
        ]
    )


# ============================================================
# PRETRAINED RESNET18 TRANSFORMS
# ============================================================

def build_pretrained_train_transform():
    """
    Build the training transform for pretrained ResNet18.

    Uses training augmentation while preserving the RGB
    normalization expected by the pretrained weights.
    """

    weights = (
        ResNet18_Weights.DEFAULT
    )

    pretrained_transform = (
        weights.transforms()
    )

    return transforms.Compose(
    [
        transforms.RandomResizedCrop(
            size=IMAGE_SIZE,
            scale=(0.8, 1.0),
            ratio=(0.9, 1.1),
        ),

        transforms.RandomHorizontalFlip(
            p=0.5
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=pretrained_transform.mean,
            std=pretrained_transform.std,
        ),
    ]
)


def build_pretrained_eval_transform():
    """
    Build deterministic validation/test preprocessing
    associated with the pretrained ResNet18 weights.
    """

    weights = (
        ResNet18_Weights.DEFAULT
    )

    return weights.transforms()