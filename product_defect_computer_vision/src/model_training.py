"""
Reusable PyTorch training utilities for the Product Defect
Computer Vision project.

This module contains:

- one-epoch training
- one-epoch validation
- multi-epoch model training
- training history
- best validation model restoration

The functions are model-agnostic and can be reused by:

- custom CNN models
- pretrained vision models
- fine-tuned vision models
"""

from copy import deepcopy
from dataclasses import dataclass

import torch

from torch import nn
from torch.utils.data import DataLoader


# ============================================================
# 1. EPOCH METRICS
# ============================================================

@dataclass
class EpochMetrics:
    """
    Metrics collected during one complete epoch.
    """

    loss: float
    accuracy: float


# ============================================================
# 2. TRAINING RESULT
# ============================================================

@dataclass
class TrainingResult:
    """
    Result returned after multi-epoch model training.

    The model itself is updated in place and restored to the
    state with the lowest validation loss.
    """

    history: dict[str, list[float]]

    best_epoch: int
    best_val_loss: float
    best_val_accuracy: float

def _set_frozen_batchnorm_to_eval(
    model: nn.Module,
) -> None:
    """
    Keep fully frozen BatchNorm layers in evaluation mode.

    Calling model.train() normally places every BatchNorm layer
    into training mode, which allows its running mean and
    running variance to change.

    A BatchNorm layer whose learnable parameters are completely
    frozen should retain its pretrained running statistics.
    """

    for module in model.modules():

        if isinstance(
            module,
            nn.modules.batchnorm._BatchNorm,
        ):

            parameters = list(
                module.parameters(
                    recurse=False
                )
            )

            is_frozen = (
                parameters
                and all(
                    not parameter.requires_grad
                    for parameter
                    in parameters
                )
            )

            if is_frozen:
                module.eval()
# ============================================================
# 3. TRAIN ONE EPOCH
# ============================================================

def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> EpochMetrics:
    """
    Train a model for one complete epoch.

    Parameters
    ----------
    model:
        PyTorch model to train.

    data_loader:
        Training DataLoader.

    criterion:
        Loss function.

    optimizer:
        Optimizer used to update model parameters.

    device:
        CPU or CUDA device.

    Returns
    -------
    EpochMetrics
        Average training loss and accuracy.
    """

    model.train()

    _set_frozen_batchnorm_to_eval(
    model
)

    running_loss = 0.0
    correct_predictions = 0
    total_examples = 0

    for images, labels in data_loader:

        images = images.to(
            device,
            non_blocking=True,
        )

        labels = labels.to(
            device,
            non_blocking=True,
        )

        # ----------------------------------------------------
        # Clear gradients from the previous batch
        # ----------------------------------------------------

        optimizer.zero_grad()

        # ----------------------------------------------------
        # Forward pass
        # ----------------------------------------------------

        logits = model(
            images
        )

        # ----------------------------------------------------
        # Calculate loss
        # ----------------------------------------------------

        loss = criterion(
            logits,
            labels,
        )

        # ----------------------------------------------------
        # Backpropagation
        # ----------------------------------------------------

        loss.backward()

        # ----------------------------------------------------
        # Update trainable model parameters
        # ----------------------------------------------------

        optimizer.step()

        # ----------------------------------------------------
        # Accumulate metrics
        # ----------------------------------------------------

        batch_size = (
            labels.size(0)
        )

        running_loss += (
            loss.item()
            * batch_size
        )

        predictions = torch.argmax(
            logits,
            dim=1,
        )

        correct_predictions += (
            predictions
            == labels
        ).sum().item()

        total_examples += (
            batch_size
        )

    if total_examples == 0:

        raise ValueError(
            "Training DataLoader contains no examples."
        )

    epoch_loss = (
        running_loss
        / total_examples
    )

    epoch_accuracy = (
        correct_predictions
        / total_examples
    )

    return EpochMetrics(
        loss=epoch_loss,
        accuracy=epoch_accuracy,
    )


# ============================================================
# 4. VALIDATE ONE EPOCH
# ============================================================

def validate_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> EpochMetrics:
    """
    Evaluate a model for one complete validation epoch.

    Model parameters are not updated.
    """

    model.eval()

    running_loss = 0.0
    correct_predictions = 0
    total_examples = 0

    with torch.no_grad():

        for images, labels in data_loader:

            images = images.to(
                device,
                non_blocking=True,
            )

            labels = labels.to(
                device,
                non_blocking=True,
            )

            # ------------------------------------------------
            # Forward pass
            # ------------------------------------------------

            logits = model(
                images
            )

            loss = criterion(
                logits,
                labels,
            )

            # ------------------------------------------------
            # Accumulate metrics
            # ------------------------------------------------

            batch_size = (
                labels.size(0)
            )

            running_loss += (
                loss.item()
                * batch_size
            )

            predictions = torch.argmax(
                logits,
                dim=1,
            )

            correct_predictions += (
                predictions
                == labels
            ).sum().item()

            total_examples += (
                batch_size
            )

    if total_examples == 0:

        raise ValueError(
            "Validation DataLoader contains no examples."
        )

    epoch_loss = (
        running_loss
        / total_examples
    )

    epoch_accuracy = (
        correct_predictions
        / total_examples
    )

    return EpochMetrics(
        loss=epoch_loss,
        accuracy=epoch_accuracy,
    )


# ============================================================
# 5. TRAIN MODEL
# ============================================================

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_epochs: int,
) -> TrainingResult:
    """
    Train and validate a model across multiple epochs.

    The model state with the lowest validation loss is retained
    and restored after training.

    Parameters
    ----------
    model:
        PyTorch model.

    train_loader:
        Training DataLoader.

    val_loader:
        Validation DataLoader.

    criterion:
        Loss function.

    optimizer:
        Optimizer containing the parameters that should be
        updated during training.

    device:
        CPU or CUDA device.

    num_epochs:
        Number of complete training epochs.

    Returns
    -------
    TrainingResult
        Training history and information about the best
        validation model.
    """

    if num_epochs < 1:

        raise ValueError(
            "num_epochs must be at least 1."
        )

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
    }

    best_epoch = 0
    best_val_loss = float("inf")
    best_val_accuracy = 0.0

    best_model_state = deepcopy(
        model.state_dict()
    )

    for epoch in range(
        1,
        num_epochs + 1,
    ):

        # ====================================================
        # TRAIN
        # ====================================================

        train_metrics = train_one_epoch(
            model=model,
            data_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        # ====================================================
        # VALIDATE
        # ====================================================

        val_metrics = validate_one_epoch(
            model=model,
            data_loader=val_loader,
            criterion=criterion,
            device=device,
        )

        # ====================================================
        # HISTORY
        # ====================================================

        history[
            "train_loss"
        ].append(
            train_metrics.loss
        )

        history[
            "train_accuracy"
        ].append(
            train_metrics.accuracy
        )

        history[
            "val_loss"
        ].append(
            val_metrics.loss
        )

        history[
            "val_accuracy"
        ].append(
            val_metrics.accuracy
        )

        # ====================================================
        # BEST VALIDATION MODEL
        # ====================================================

        if (
            val_metrics.loss
            < best_val_loss
        ):

            best_epoch = (
                epoch
            )

            best_val_loss = (
                val_metrics.loss
            )

            best_val_accuracy = (
                val_metrics.accuracy
            )

            best_model_state = deepcopy(
                model.state_dict()
            )

        # ====================================================
        # EPOCH SUMMARY
        # ====================================================

        print(
            f"Epoch {epoch:02d}/{num_epochs} | "
            f"Train Loss: "
            f"{train_metrics.loss:.4f} | "
            f"Train Acc: "
            f"{train_metrics.accuracy:.4f} | "
            f"Val Loss: "
            f"{val_metrics.loss:.4f} | "
            f"Val Acc: "
            f"{val_metrics.accuracy:.4f}"
        )

    # ========================================================
    # RESTORE BEST MODEL
    # ========================================================

    model.load_state_dict(
        best_model_state
    )

    print(
        "\nBest epoch:",
        best_epoch,
    )

    print(
        "Best validation loss:",
        f"{best_val_loss:.4f}",
    )

    print(
        "Best validation accuracy:",
        f"{best_val_accuracy:.4f}",
    )

    print(
        "Best validation model restored."
    )

    return TrainingResult(
        history=history,
        best_epoch=best_epoch,
        best_val_loss=best_val_loss,
        best_val_accuracy=best_val_accuracy,
    )