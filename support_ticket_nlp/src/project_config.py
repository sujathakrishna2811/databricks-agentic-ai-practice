"""
Central configuration for the Support Ticket NLP project.

This module contains project-wide constants such as
Unity Catalog objects, column names, split names,
feature-engineering settings, and model settings.
"""

# ============================================================
# Project
# ============================================================

PROJECT_NAME = "support_ticket_nlp"

RANDOM_SEED = 42


# ============================================================
# Unity Catalog
# ============================================================

CATALOG = "dbw_agentic_ai_dev"

SCHEMA = "support_ticket_ai"

FULL_SCHEMA = f"{CATALOG}.{SCHEMA}"


# ============================================================
# Source Tables
# ============================================================

SOURCE_TABLE_NAME = "bronze_support_tickets"

SOURCE_TABLE = (
    f"{FULL_SCHEMA}.{SOURCE_TABLE_NAME}"
)


# ============================================================
# NLP Tables
# ============================================================

NLP_PREPROCESSED_TABLE_NAME = (
    "nlp_preprocessed_tickets"
)

NLP_PREPROCESSED_TABLE = (
    f"{FULL_SCHEMA}."
    f"{NLP_PREPROCESSED_TABLE_NAME}"
)


NLP_MODELING_TABLE_NAME = (
    "nlp_modeling_dataset"
)

NLP_MODELING_TABLE = (
    f"{FULL_SCHEMA}."
    f"{NLP_MODELING_TABLE_NAME}"
)


# ============================================================
# Canonical Columns
# ============================================================

TICKET_ID_COL = "ticket_id"

TEXT_COL = "ticket_text"

TARGET_COL = "category"

CLEAN_TEXT_COL = "clean_text"

TOKENS_COL = "tokens"

TOKEN_COUNT_COL = "token_count"

SPLIT_COL = "dataset_split"


# ============================================================
# Dataset Splits
# ============================================================

TRAIN_SPLIT = "train"

VALIDATION_SPLIT = "validation"

TEST_SPLIT = "test"


# ============================================================
# Bag of Words
# ============================================================

BOW_MAX_FEATURES = None


# ============================================================
# TF-IDF
# ============================================================

TFIDF_NGRAM_RANGE = (1, 1)

TFIDF_MIN_DF = 1

TFIDF_MAX_DF = 1.0

TFIDF_MAX_FEATURES = None


# ============================================================
# Logistic Regression
# ============================================================

LOGISTIC_REGRESSION_MAX_ITER = 1000


# ============================================================
# Expected Categories
# ============================================================

EXPECTED_CATEGORIES = [
    "Billing",
    "Cancellation",
    "Login",
    "Technical",
]

# ---------------------------------------------------------
# Embedding Configuration
# ---------------------------------------------------------

EMBEDDING_MODEL_NAME = (
    "sentence-transformers/"
    "all-MiniLM-L6-v2"
)

EMBEDDING_BATCH_SIZE = 32

# ---------------------------------------------------------
# Transformer classifier
# ---------------------------------------------------------

TRANSFORMER_MODEL_NAME = (
    "sentence-transformers/"
    "all-MiniLM-L6-v2"
)

TRANSFORMER_MAX_LENGTH = 128
TRANSFORMER_BATCH_SIZE = 8

TRANSFORMER_LEARNING_RATE = 2e-5
TRANSFORMER_WEIGHT_DECAY = 0.01

TRANSFORMER_MAX_EPOCHS = 8
TRANSFORMER_EARLY_STOPPING_PATIENCE = 2

TRANSFORMER_DROPOUT = 0.1
TRANSFORMER_MAX_GRAD_NORM = 1.0

CLASS_NAMES = [
    "Billing",
    "Cancellation",
    "Login",
    "Technical",
]