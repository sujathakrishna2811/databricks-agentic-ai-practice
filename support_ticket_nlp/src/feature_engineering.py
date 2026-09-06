"""
Feature engineering utilities for NLP models.
"""

from typing import Tuple

import numpy as np

from scipy.sparse import spmatrix

from sklearn.feature_extraction.text import (
    CountVectorizer,
    TfidfVectorizer,
)

from src.project_config import (
    BOW_MAX_FEATURES,
    TFIDF_NGRAM_RANGE,
    TFIDF_MIN_DF,
    TFIDF_MAX_DF,
    TFIDF_MAX_FEATURES,
)

from sentence_transformers import (
    SentenceTransformer,
)

from src.project_config import (
    EMBEDDING_MODEL_NAME,
    EMBEDDING_BATCH_SIZE,
)


def create_bow_vectorizer() -> CountVectorizer:
    """
    Create the project's Bag-of-Words vectorizer.

    Text is already lowercased and cleaned,
    therefore sklearn preprocessing is disabled.
    """

    return CountVectorizer(
        lowercase=False,
        tokenizer=str.split,
        preprocessor=None,
        token_pattern=None,
        max_features=BOW_MAX_FEATURES,
    )


def create_tfidf_vectorizer() -> TfidfVectorizer:
    """
    Create the project's TF-IDF vectorizer.
    """

    return TfidfVectorizer(
        lowercase=False,
        tokenizer=str.split,
        preprocessor=None,
        token_pattern=None,
        ngram_range=TFIDF_NGRAM_RANGE,
        min_df=TFIDF_MIN_DF,
        max_df=TFIDF_MAX_DF,
        max_features=TFIDF_MAX_FEATURES,
    )


def build_tfidf_features(
    train_text,
    validation_text,
    test_text,
) -> Tuple[
    TfidfVectorizer,
    spmatrix,
    spmatrix,
    spmatrix,
]:
    """
    Fit TF-IDF using training text only and
    transform validation and test text.

    This prevents data leakage.
    """

    vectorizer = create_tfidf_vectorizer()

    X_train = vectorizer.fit_transform(
        train_text
    )

    X_validation = vectorizer.transform(
        validation_text
    )

    X_test = vectorizer.transform(
        test_text
    )

    return (
        vectorizer,
        X_train,
        X_validation,
        X_test,
    )

def load_embedding_model(
    model_name: str = EMBEDDING_MODEL_NAME,
) -> SentenceTransformer:
    """
    Load the pretrained sentence embedding model.

    Parameters
    ----------
    model_name:
        Hugging Face / Sentence Transformers
        model identifier.

    Returns
    -------
    SentenceTransformer
        Loaded pretrained embedding model.
    """

    return SentenceTransformer(
        model_name
    )

def generate_embeddings(
    embedding_model: SentenceTransformer,
    text,
) -> np.ndarray:
    """
    Convert text documents into dense
    sentence embeddings.
    """

    text_values = [
        str(value)
        for value in text
    ]

    embeddings = embedding_model.encode(
        text_values,
        batch_size=EMBEDDING_BATCH_SIZE,
        show_progress_bar=False,
        convert_to_numpy=True,
    )

    return embeddings


def build_embedding_features(
    embedding_model: SentenceTransformer,
    train_text,
    validation_text,
    test_text,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Generate embedding matrices for the
    persisted train, validation, and test
    datasets.

    The embedding model is pretrained.
    No local fitting is performed.

    Returns
    -------
    tuple
        X_train_embeddings,
        X_validation_embeddings,
        X_test_embeddings
    """

    X_train_embeddings = (
        generate_embeddings(
            embedding_model,
            train_text,
        )
    )

    X_validation_embeddings = (
        generate_embeddings(
            embedding_model,
            validation_text,
        )
    )

    X_test_embeddings = (
        generate_embeddings(
            embedding_model,
            test_text,
        )
    )

    return (
        X_train_embeddings,
        X_validation_embeddings,
        X_test_embeddings,
    )