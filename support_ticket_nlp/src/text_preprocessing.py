"""
Reusable text preprocessing utilities.
"""

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from src.project_config import (
    TEXT_COL,
    CLEAN_TEXT_COL,
    TOKENS_COL,
    TOKEN_COUNT_COL,
)


def clean_text_column(
    df: DataFrame,
) -> DataFrame:
    """
    Clean and tokenize support-ticket text.

    Parameters
    ----------
    df:
        Spark DataFrame containing the raw ticket text.

    Returns
    -------
    DataFrame
        DataFrame containing clean text, tokens,
        and token count.
    """

    cleaned_text = F.lower(
        F.trim(
            F.regexp_replace(
                F.regexp_replace(
                    F.regexp_replace(
                        F.col(TEXT_COL),
                        r"https?://\S+|www\.\S+",
                        " url ",
                    ),
                    r"\b[\w\.-]+@[\w\.-]+\.\w+\b",
                    " email ",
                ),
                r"[^a-z0-9\s]",
                " ",
            )
        )
    )

    df = df.withColumn(
        CLEAN_TEXT_COL,
        cleaned_text,
    )

    df = df.withColumn(
        CLEAN_TEXT_COL,
        F.regexp_replace(
            F.col(CLEAN_TEXT_COL),
            r"\s+",
            " ",
        ),
    )

    df = df.withColumn(
        TOKENS_COL,
        F.split(
            F.col(CLEAN_TEXT_COL),
            r"\s+",
        ),
    )

    df = df.withColumn(
        TOKEN_COUNT_COL,
        F.size(
            F.col(TOKENS_COL)
        ),
    )

    return df