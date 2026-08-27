from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(
    numerical_features,
    categorical_features,
    binary_features,
):
    """
    Build the preprocessing pipeline for the Telco churn model.
    """

    numerical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                numerical_pipeline,
                numerical_features,
            ),
            (
                "cat",
                categorical_pipeline,
                categorical_features,
            ),
            (
                "binary",
                "passthrough",
                binary_features,
            ),
        ]
    )

    return preprocessor


def validate_feature_columns(
    dataframe,
    required_features,
):
    """
    Validate that required model features exist.
    """

    missing_columns = [
        column
        for column in required_features
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required feature columns: "
            f"{missing_columns}"
        )