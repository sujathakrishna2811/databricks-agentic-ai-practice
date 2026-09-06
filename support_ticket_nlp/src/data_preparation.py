"""
Dataset loading and preparation utilities
for the Support Ticket NLP project.
"""

from typing import Tuple

import pandas as pd

from pyspark.sql import DataFrame, SparkSession

from src.project_config import (
    NLP_MODELING_TABLE,
    TICKET_ID_COL,
    CLEAN_TEXT_COL,
    TARGET_COL,
    SPLIT_COL,
    TRAIN_SPLIT,
    VALIDATION_SPLIT,
    TEST_SPLIT,
)


REQUIRED_MODELING_COLUMNS = [
    TICKET_ID_COL,
    CLEAN_TEXT_COL,
    TARGET_COL,
    SPLIT_COL,
]


def load_modeling_dataset(
    spark: SparkSession,
) -> DataFrame:
    """
    Load the persisted NLP modeling dataset
    from Unity Catalog.
    """

    return spark.table(
        NLP_MODELING_TABLE
    )


def validate_modeling_schema(
    df: DataFrame,
) -> None:
    """
    Validate that required modeling columns exist.
    """

    missing_columns = [
        column
        for column in REQUIRED_MODELING_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Modeling dataset is missing required "
            f"columns: {missing_columns}"
        )


def split_modeling_dataset(
    df: DataFrame,
) -> Tuple[
    DataFrame,
    DataFrame,
    DataFrame,
]:
    """
    Split the persisted modeling dataset using
    the existing dataset_split assignments.

    No new random split is created.
    """

    train_df = df.filter(
        df[SPLIT_COL] == TRAIN_SPLIT
    )

    validation_df = df.filter(
        df[SPLIT_COL] == VALIDATION_SPLIT
    )

    test_df = df.filter(
        df[SPLIT_COL] == TEST_SPLIT
    )

    return (
        train_df,
        validation_df,
        test_df,
    )


def to_pandas_modeling_data(
    df: DataFrame,
) -> pd.DataFrame:
    """
    Convert only the columns required for modeling
    from Spark to pandas.
    """

    return (
        df.select(
            TICKET_ID_COL,
            CLEAN_TEXT_COL,
            TARGET_COL,
            SPLIT_COL,
        )
        .toPandas()
    )