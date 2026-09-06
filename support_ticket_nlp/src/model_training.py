"""
Reusable model-training utilities.
"""

from sklearn.linear_model import (
    LogisticRegression,
)

from src.project_config import (
    RANDOM_SEED,
    LOGISTIC_REGRESSION_MAX_ITER,
)


def create_logistic_regression_classifier(
) -> LogisticRegression:
    """
    Create the project's baseline
    Logistic Regression classifier.
    """

    return LogisticRegression(
        max_iter=LOGISTIC_REGRESSION_MAX_ITER,
        random_state=RANDOM_SEED,
    )


def train_logistic_regression(
    X_train,
    y_train,
) -> LogisticRegression:
    """
    Train a Logistic Regression classifier.
    """

    classifier = (
        create_logistic_regression_classifier()
    )

    classifier.fit(
        X_train,
        y_train,
    )

    return classifier