"""
Reusable PyTorch neural-network components
for the Support Ticket NLP project.
"""

from copy import deepcopy
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import (
    DataLoader,
    TensorDataset,
)


class TicketClassifierNN(nn.Module):
    """
    Feed-forward neural network for multiclass
    support-ticket classification.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size_1: int,
        hidden_size_2: int,
        num_classes: int,
    ):
        super().__init__()

        self.layer1 = nn.Linear(
            input_size,
            hidden_size_1,
        )

        self.relu1 = nn.ReLU()

        self.layer2 = nn.Linear(
            hidden_size_1,
            hidden_size_2,
        )

        self.relu2 = nn.ReLU()

        self.output = nn.Linear(
            hidden_size_2,
            num_classes,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        x = self.layer1(x)
        x = self.relu1(x)

        x = self.layer2(x)
        x = self.relu2(x)

        logits = self.output(x)

        return logits


def set_torch_seed(
    seed: int,
) -> None:
    """
    Set reproducibility seeds for NumPy and PyTorch.
    """

    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_tensor_dataset(
    features: np.ndarray,
    labels: np.ndarray,
) -> TensorDataset:
    """
    Convert NumPy feature and label arrays
    into a PyTorch TensorDataset.
    """

    X_tensor = torch.tensor(
        features,
        dtype=torch.float32,
    )

    y_tensor = torch.tensor(
        labels,
        dtype=torch.long,
    )

    return TensorDataset(
        X_tensor,
        y_tensor,
    )


def create_data_loader(
    dataset: TensorDataset,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    """
    Create a PyTorch DataLoader.
    """

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
    )


def evaluate_loss_and_accuracy(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Evaluate average loss and classification accuracy.
    """

    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    with torch.no_grad():

        for X_batch, y_batch in data_loader:

            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(
                X_batch
            )

            loss = criterion(
                logits,
                y_batch,
            )

            predictions = torch.argmax(
                logits,
                dim=1,
            )

            batch_size = y_batch.size(0)

            total_loss += (
                loss.item()
                * batch_size
            )

            total_correct += (
                predictions == y_batch
            ).sum().item()

            total_examples += batch_size

    average_loss = (
        total_loss
        / total_examples
    )

    accuracy = (
        total_correct
        / total_examples
    )

    return (
        average_loss,
        accuracy,
    )


def train_neural_network(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_epochs: int,
    patience: int,
) -> List[Dict]:
    """
    Train a neural network using validation-loss
    based early stopping.

    The best validation model state is restored
    before returning.
    """

    history = []

    best_validation_loss = float(
        "inf"
    )

    best_model_state = None

    epochs_without_improvement = 0

    for epoch in range(
        1,
        num_epochs + 1,
    ):

        model.train()

        total_train_loss = 0.0
        total_train_correct = 0
        total_train_examples = 0

        for X_batch, y_batch in train_loader:

            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()

            logits = model(
                X_batch
            )

            loss = criterion(
                logits,
                y_batch,
            )

            loss.backward()

            optimizer.step()

            predictions = torch.argmax(
                logits,
                dim=1,
            )

            batch_size = y_batch.size(0)

            total_train_loss += (
                loss.item()
                * batch_size
            )

            total_train_correct += (
                predictions == y_batch
            ).sum().item()

            total_train_examples += (
                batch_size
            )

        train_loss = (
            total_train_loss
            / total_train_examples
        )

        train_accuracy = (
            total_train_correct
            / total_train_examples
        )

        (
            validation_loss,
            validation_accuracy,
        ) = evaluate_loss_and_accuracy(
            model=model,
            data_loader=validation_loader,
            criterion=criterion,
            device=device,
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "validation_loss": validation_loss,
                "validation_accuracy": validation_accuracy,
            }
        )

        if (
            validation_loss
            < best_validation_loss
        ):

            best_validation_loss = (
                validation_loss
            )

            best_model_state = deepcopy(
                model.state_dict()
            )

            epochs_without_improvement = 0

        else:

            epochs_without_improvement += 1

            if (
                epochs_without_improvement
                >= patience
            ):
                break

    if best_model_state is not None:
        model.load_state_dict(
            best_model_state
        )

    return history


def predict_neural_network(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return predicted class IDs and class probabilities.
    """

    model.eval()

    prediction_batches = []
    probability_batches = []

    with torch.no_grad():

        for X_batch, _ in data_loader:

            X_batch = X_batch.to(
                device
            )

            logits = model(
                X_batch
            )

            probabilities = torch.softmax(
                logits,
                dim=1,
            )

            predictions = torch.argmax(
                logits,
                dim=1,
            )

            prediction_batches.append(
                predictions.cpu().numpy()
            )

            probability_batches.append(
                probabilities.cpu().numpy()
            )

    predictions = np.concatenate(
        prediction_batches
    )

    probabilities = np.concatenate(
        probability_batches
    )

    return (
        predictions,
        probabilities,
    )