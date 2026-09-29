"""
Central project configuration for the Product Defect
Computer Vision project.

This module contains shared constants used across notebooks
and reusable source modules.

Do not place credentials, access tokens, or secrets here.
"""

from pathlib import Path


# ============================================================
# 1. PROJECT
# ============================================================

PROJECT_NAME = "product_defect_computer_vision"

RANDOM_SEED = 42


# ============================================================
# 2. DATABRICKS / UNITY CATALOG
# ============================================================

CATALOG = "dbw_agentic_ai_dev"
SCHEMA = "product_defect_ai"

VOLUME_NAME = "product_defect_data"

VOLUME_PATH = (
    f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME_NAME}"
)


# ============================================================
# 3. DATA PATHS
# ============================================================

RAW_DATA_DIR = f"{VOLUME_PATH}/raw/casting_data"

TRAIN_DATA_DIR = f"{VOLUME_PATH}/train"
VAL_DATA_DIR = f"{VOLUME_PATH}/validation"
TEST_DATA_DIR = f"{VOLUME_PATH}/test"


# ============================================================
# 4. IMAGE CONFIGURATION
# ============================================================

IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224

IMAGE_SIZE = (
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
)

IMAGE_CHANNELS = 3


# ============================================================
# 5. DATA SPLIT
# ============================================================

VALIDATION_RATIO  = 0.15

# ============================================================
# 6. TRAINING CONFIGURATION
# ============================================================

BATCH_SIZE = 32

NUM_WORKERS = 0

LEARNING_RATE = 1e-3

NUM_EPOCHS = 5


# ============================================================
# 7. CUSTOM CNN ARCHITECTURE
# ============================================================

CNN_CHANNELS = (
    32,
    64,
    128,
)

CLASSIFIER_HIDDEN_DIM = 64


# ============================================================
# 8. IMAGE NORMALIZATION
# ============================================================
#
# These values are intentionally separated from pretrained-model
# normalization.
#
# For the custom CNN, normalization statistics should ultimately
# be chosen based on the training pipeline/data.
#
# Pretrained models will use the preprocessing associated with
# their pretrained weights instead of assuming these values.

CUSTOM_NORMALIZE_MEAN = (
    0.5,
    0.5,
    0.5,
)

CUSTOM_NORMALIZE_STD = (
    0.5,
    0.5,
    0.5,
)


# ============================================================
# 9. MLFLOW
# ============================================================

MLFLOW_EXPERIMENT_NAME = (
    "/Shared/product_defect_computer_vision"
)

REGISTERED_MODEL_NAME = (
    f"{CATALOG}.{SCHEMA}.product_defect_classifier"
)


# ============================================================
# 10. SERVING
# ============================================================

SERVING_ENDPOINT_NAME = (
    "product-defect-classifier-endpoint"
)


# ============================================================
# 11. VALIDATION
# ============================================================


if IMAGE_CHANNELS != 3:
    raise ValueError(
        "This project currently expects RGB images "
        "with 3 channels."
    )

# ============================================================
# TRANSFER LEARNING
# ============================================================

FEATURE_EXTRACTION_LEARNING_RATE = 1e-3
FEATURE_EXTRACTION_EPOCHS = 3

FINE_TUNING_LEARNING_RATE = 1e-4
FINE_TUNING_EPOCHS = 2