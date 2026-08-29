from pyspark.sql import functions as F


def clean_text_column(
    column: F.Column,
) -> F.Column:
    """
    Normalize support-ticket text.

    Operations:
    - lowercase
    - replace URLs
    - replace email addresses
    - normalize punctuation
    - collapse repeated whitespace
    - trim leading/trailing whitespace
    """

    cleaned = F.lower(column)

    cleaned = F.regexp_replace(
        cleaned,
        r"https?://\S+|www\.\S+",
        " url ",
    )

    cleaned = F.regexp_replace(
        cleaned,
        r"\b[\w\.-]+@[\w\.-]+\.\w+\b",
        " email ",
    )

    cleaned = F.regexp_replace(
        cleaned,
        r"[^a-z0-9\s']",
        " ",
    )

    cleaned = F.regexp_replace(
        cleaned,
        r"\s+",
        " ",
    )

    cleaned = F.trim(
        cleaned
    )

    return cleaned