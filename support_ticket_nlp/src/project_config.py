"""
Shared project configuration for the Support Ticket NLP project.

This module is the single source of truth for stable project-wide
configuration such as:

- Unity Catalog objects
- source and output tables
- canonical column names
- expected ticket categories
- dataset split settings
- traditional NLP feature settings
- MLflow resources
- model registration
- model serving

Experimental model hyperparameters should remain in their respective
training notebooks and be logged to MLflow.
"""


# ============================================================
# 1. Project Identification
# ============================================================

PROJECT_NAME = "support_ticket_nlp"

RANDOM_SEED = 42


# ============================================================
# 2. Unity Catalog Configuration
# ============================================================

CATALOG = "dbw_agentic_ai_dev"

SCHEMA = "support_ticket_ai"

FULL_SCHEMA = f"{CATALOG}.{SCHEMA}"


# ============================================================
# 3. Source Table Configuration
# ============================================================

SOURCE_TABLE_NAME = "bronze_support_tickets"

SOURCE_TABLE = (
    f"{FULL_SCHEMA}."
    f"{SOURCE_TABLE_NAME}"
)

# Result:
# dbw_agentic_ai_dev.support_ticket_ai.bronze_support_tickets


# ============================================================
# 4. NLP Preprocessed Table Configuration
# ============================================================

NLP_CLEAN_TABLE_NAME = "nlp_preprocessed_tickets"

NLP_CLEAN_TABLE = (
    f"{FULL_SCHEMA}."
    f"{NLP_CLEAN_TABLE_NAME}"
)

# Result:
# dbw_agentic_ai_dev.support_ticket_ai.nlp_preprocessed_tickets


# ============================================================
# 5. Canonical NLP Column Names
# ============================================================

TICKET_ID_COL = "ticket_id"

TEXT_COL = "ticket_text"

TARGET_COL = "category"

CLEAN_TEXT_COL = "clean_text"

TOKENS_COL = "tokens"

TOKEN_COUNT_COL = "token_count"


# ============================================================
# 6. Expected Ticket Categories
# ============================================================

EXPECTED_CATEGORIES = (
    "Billing",
    "Cancellation",
    "Login",
    "Technical",
)

# ============================================================
# 7. Train / Validation / Test Configuration
# ============================================================

TRAIN_SIZE = 0.70

VALIDATION_SIZE = 0.15

TEST_SIZE = 0.15


# Validate split configuration.

_SPLIT_TOTAL = (
    TRAIN_SIZE
    + VALIDATION_SIZE
    + TEST_SIZE
)

if abs(_SPLIT_TOTAL - 1.0) >= 1e-9:
    raise ValueError(
        "Train, validation, and test "
        "fractions must sum to 1.0."
    )


# ============================================================
# 8. Traditional NLP Feature Configuration
# ============================================================

BOW_MAX_FEATURES = None

TFIDF_MAX_FEATURES = None

TFIDF_NGRAM_RANGE = (1, 1)

TFIDF_MIN_DF = 1

TFIDF_MAX_DF = 1.0


# ============================================================
# 9. Baseline Model Configuration
# ============================================================

BASELINE_MODEL_NAME = "logistic_regression_tfidf"


# ============================================================
# 10. MLflow Configuration
# ============================================================

MLFLOW_EXPERIMENT_NAME = (
    "/Users/sujathakrishna2811@gmail.com/"
    "support_ticket_nlp_experiment"
)


# ============================================================
# 11. Registered Model Configuration
# ============================================================

REGISTERED_MODEL_NAME = (
    f"{FULL_SCHEMA}."
    "support_ticket_classifier"
)

# Result:
# dbw_agentic_ai_dev.support_ticket_ai.support_ticket_classifier


# ============================================================
# 12. Serving Endpoint Configuration
# ============================================================

SERVING_ENDPOINT_NAME = (
    "support-ticket-classifier-endpoint"
)


# ============================================================
# 13. Model Input Contract
# ============================================================

MODEL_INPUT_COL = TEXT_COL


# ============================================================
# 14. Project-Wide Configuration Validation
# ============================================================

_REQUIRED_STRING_SETTINGS = {
    "PROJECT_NAME": PROJECT_NAME,
    "CATALOG": CATALOG,
    "SCHEMA": SCHEMA,
    "SOURCE_TABLE": SOURCE_TABLE,
    "NLP_CLEAN_TABLE": NLP_CLEAN_TABLE,
    "TICKET_ID_COL": TICKET_ID_COL,
    "TEXT_COL": TEXT_COL,
    "TARGET_COL": TARGET_COL,
    "CLEAN_TEXT_COL": CLEAN_TEXT_COL,
    "TOKENS_COL": TOKENS_COL,
    "TOKEN_COUNT_COL": TOKEN_COUNT_COL,
    "BASELINE_MODEL_NAME": BASELINE_MODEL_NAME,
    "MLFLOW_EXPERIMENT_NAME": MLFLOW_EXPERIMENT_NAME,
    "REGISTERED_MODEL_NAME": REGISTERED_MODEL_NAME,
    "SERVING_ENDPOINT_NAME": SERVING_ENDPOINT_NAME,
    "MODEL_INPUT_COL": MODEL_INPUT_COL,
}


for _name, _value in _REQUIRED_STRING_SETTINGS.items():

    if (
        not isinstance(_value, str)
        or not _value.strip()
    ):
        raise ValueError(
            "Invalid project configuration: "
            f"{_name}"
        )


if not EXPECTED_CATEGORIES:
    raise ValueError(
        "EXPECTED_CATEGORIES cannot be empty."
    )


if len(set(EXPECTED_CATEGORIES)) != len(
    EXPECTED_CATEGORIES
):
    raise ValueError(
        "EXPECTED_CATEGORIES contains "
        "duplicate values."
    )

# ============================================================
# Modeling Dataset Configuration
# ============================================================

MODELING_TABLE_NAME = "nlp_modeling_dataset"

MODELING_TABLE = (
    f"{FULL_SCHEMA}."
    f"{MODELING_TABLE_NAME}"
)

SPLIT_COL = "dataset_split"