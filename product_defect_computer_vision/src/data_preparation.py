"""
Dataset and DataLoader utilities for the Product Defect
Computer Vision project.

The original dataset provides:

    casting_data/
        train/
            def_front/
            ok_front/
        test/
            def_front/
            ok_front/

The original training set is deterministically split into
training and validation subsets.

The provided test set remains untouched for final evaluation.

The data pipeline supports preprocessing for:

    - custom CNN
    - pretrained vision models
"""

from dataclasses import dataclass

import numpy as np
import torch

from sklearn.model_selection import train_test_split

from torch.utils.data import (
    DataLoader,
    Subset,
)

from torchvision.datasets import ImageFolder

from src.image_preprocessing import (
    build_custom_eval_transform,
    build_custom_train_transform,
    build_pretrained_eval_transform,
    build_pretrained_train_transform,
)

from src.project_config import (
    BATCH_SIZE,
    NUM_WORKERS,
    RANDOM_SEED,
    RAW_DATA_DIR,
    VALIDATION_RATIO,
)


# ============================================================
# DATA CONTAINER
# ============================================================

@dataclass
class ImageDataLoaders:
    """
    Container for datasets, DataLoaders, and class metadata.
    """

    train_dataset: Subset
    val_dataset: Subset
    test_dataset: ImageFolder

    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader

    class_names: list[str]
    class_to_idx: dict[str, int]

    num_classes: int


# ============================================================
# TRANSFORM SELECTION
# ============================================================

def _get_image_transforms(
    preprocessing_mode: str,
):
    """
    Return training and evaluation transforms for the
    requested preprocessing mode.

    Parameters
    ----------
    preprocessing_mode:
        Supported values:

        - "custom_cnn"
        - "pretrained"

    Returns
    -------
    tuple
        Training transform and evaluation transform.
    """

    if preprocessing_mode == "custom_cnn":

        return (
            build_custom_train_transform(),
            build_custom_eval_transform(),
        )

    if preprocessing_mode == "pretrained":

        return (
            build_pretrained_train_transform(),
            build_pretrained_eval_transform(),
        )

    raise ValueError(
        "preprocessing_mode must be "
        "'custom_cnn' or 'pretrained'."
    )


# ============================================================
# MAIN DATA PREPARATION FUNCTION
# ============================================================

def create_image_dataloaders(
    validation_ratio: float = VALIDATION_RATIO,
    preprocessing_mode: str = "custom_cnn",
) -> ImageDataLoaders:
    """
    Create train, validation, and test datasets and DataLoaders.

    The dataset's original test split is preserved.

    The original training split is divided into training and
    validation subsets using a deterministic stratified split.

    The preprocessing pipeline can be selected for either the
    custom CNN or a pretrained vision model.

    Parameters
    ----------
    validation_ratio:
        Fraction of the original training data assigned to
        validation.

    preprocessing_mode:
        Image preprocessing pipeline to use.

        Supported values:

        - "custom_cnn"
        - "pretrained"

        The default is "custom_cnn" so that existing custom-CNN
        notebooks continue to behave as before.

    Returns
    -------
    ImageDataLoaders
        Datasets, DataLoaders, and class metadata.
    """

    if not 0.0 < validation_ratio < 1.0:
        raise ValueError(
            "validation_ratio must be between 0 and 1."
        )


    # ========================================================
    # DATA DIRECTORIES
    # ========================================================

    train_directory = (
        f"{RAW_DATA_DIR}/train"
    )

    test_directory = (
        f"{RAW_DATA_DIR}/test"
    )


    # ========================================================
    # TRANSFORMS
    # ========================================================

    train_transform, eval_transform = (
        _get_image_transforms(
            preprocessing_mode
        )
    )


    # ========================================================
    # BASE DATASETS
    # ========================================================
    #
    # Two ImageFolder objects intentionally point to the same
    # original training directory.
    #
    # This allows:
    #
    # training subset   → training preprocessing / augmentation
    # validation subset → deterministic evaluation preprocessing
    #
    # Both datasets contain the same source images. Different
    # transforms are applied when images are loaded.
    #

    train_source_dataset = ImageFolder(
        root=train_directory,
        transform=train_transform,
    )

    val_source_dataset = ImageFolder(
        root=train_directory,
        transform=eval_transform,
    )

    test_dataset = ImageFolder(
        root=test_directory,
        transform=eval_transform,
    )


    # ========================================================
    # CLASS CONSISTENCY
    # ========================================================

    if (
        train_source_dataset.class_to_idx
        != val_source_dataset.class_to_idx
    ):
        raise ValueError(
            "Training and validation class mappings differ."
        )

    if (
        train_source_dataset.class_to_idx
        != test_dataset.class_to_idx
    ):
        raise ValueError(
            "Training and test class mappings differ."
        )


    # ========================================================
    # STRATIFIED TRAIN / VALIDATION SPLIT
    # ========================================================

    targets = np.asarray(
        train_source_dataset.targets
    )

    all_indices = np.arange(
        len(train_source_dataset)
    )

    train_indices, val_indices = (
        train_test_split(
            all_indices,
            test_size=validation_ratio,
            random_state=RANDOM_SEED,
            stratify=targets,
        )
    )


    # ========================================================
    # SUBSETS
    # ========================================================

    train_dataset = Subset(
        train_source_dataset,
        train_indices.tolist(),
    )

    val_dataset = Subset(
        val_source_dataset,
        val_indices.tolist(),
    )


    # ========================================================
    # DATALOADER SETTINGS
    # ========================================================

    pin_memory = (
        torch.cuda.is_available()
    )


    # ========================================================
    # DATALOADERS
    # ========================================================

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
    )


    # ========================================================
    # CLASS METADATA
    # ========================================================

    class_names = (
        train_source_dataset.classes
    )

    class_to_idx = (
        train_source_dataset.class_to_idx
    )

    num_classes = len(
        class_names
    )


    # ========================================================
    # RETURN DATA CONTAINER
    # ========================================================

    return ImageDataLoaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,

        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,

        class_names=class_names,
        class_to_idx=class_to_idx,

        num_classes=num_classes,
    )