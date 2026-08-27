import torch

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from src.preprocessing import (
    build_preprocessor,
    validate_feature_columns,
)


def prepare_training_data(
    spark,
    gold_table,
    target_column,
    columns_to_drop,
    numerical_features,
    binary_features,
    categorical_features,
    test_size,
    validation_test_size,
    random_state,
    batch_size,
):
    """
    Load and prepare Telco churn data for PyTorch training.

    Returns:
        train_loader
        val_loader
        test_loader
        input_size
        fitted_preprocessor
    """

    # ---------------------------------------------------------
    # Load Gold table
    # ---------------------------------------------------------

    df = (
        spark.table(gold_table)
        .toPandas()
    )

    # ---------------------------------------------------------
    # Create X and y
    # ---------------------------------------------------------

    y = df[target_column]

    X = df.drop(
        columns=columns_to_drop,
        errors="ignore",
    )

    # ---------------------------------------------------------
    # Validate feature schema
    # ---------------------------------------------------------

    required_features = (
        numerical_features
        + binary_features
        + categorical_features
    )

    validate_feature_columns(
        X,
        required_features,
    )

    # ---------------------------------------------------------
    # Train / Validation / Test Split
    # ---------------------------------------------------------

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=validation_test_size,
        random_state=random_state,
        stratify=y_temp,
    )

    # ---------------------------------------------------------
    # Preprocessing
    # ---------------------------------------------------------

    preprocessor = build_preprocessor(
        numerical_features=numerical_features,
        categorical_features=categorical_features,
        binary_features=binary_features,
    )

    X_train_processed = preprocessor.fit_transform(
        X_train
    )

    X_val_processed = preprocessor.transform(
        X_val
    )

    X_test_processed = preprocessor.transform(
        X_test
    )

    # ---------------------------------------------------------
    # Convert Features to PyTorch Tensors
    # ---------------------------------------------------------

    X_train_tensor = torch.tensor(
        X_train_processed,
        dtype=torch.float32,
    )

    X_val_tensor = torch.tensor(
        X_val_processed,
        dtype=torch.float32,
    )

    X_test_tensor = torch.tensor(
        X_test_processed,
        dtype=torch.float32,
    )

    # ---------------------------------------------------------
    # Convert Targets to PyTorch Tensors
    # ---------------------------------------------------------

    y_train_tensor = torch.tensor(
        y_train.to_numpy(),
        dtype=torch.float32,
    ).unsqueeze(1)

    y_val_tensor = torch.tensor(
        y_val.to_numpy(),
        dtype=torch.float32,
    ).unsqueeze(1)

    y_test_tensor = torch.tensor(
        y_test.to_numpy(),
        dtype=torch.float32,
    ).unsqueeze(1)

    # ---------------------------------------------------------
    # TensorDataset
    # ---------------------------------------------------------

    train_dataset = TensorDataset(
        X_train_tensor,
        y_train_tensor,
    )

    val_dataset = TensorDataset(
        X_val_tensor,
        y_val_tensor,
    )

    test_dataset = TensorDataset(
        X_test_tensor,
        y_test_tensor,
    )

    # ---------------------------------------------------------
    # DataLoader
    # ---------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    # ---------------------------------------------------------
    # Neural Network Input Size
    # ---------------------------------------------------------

    input_size = X_train_tensor.shape[1]

    return (
        train_loader,
        val_loader,
        test_loader,
        input_size,
        preprocessor,
    )