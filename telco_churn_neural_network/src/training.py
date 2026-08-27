import torch


def train_one_epoch(
    model,
    train_loader,
    criterion,
    optimizer,
):
    """
    Train the model for one complete epoch.

    Parameters
    ----------
    model:
        PyTorch neural-network model.

    train_loader:
        DataLoader containing training batches.

    criterion:
        Loss function used to measure prediction error.

    optimizer:
        Optimizer used to update model parameters.

    Returns
    -------
    float
        Average training loss for the epoch.
    """

    # Put model into training mode
    model.train()

    total_loss = 0.0

    for X_batch, y_batch in train_loader:

        # -----------------------------------------------------
        # 1. Clear gradients from previous batch
        # -----------------------------------------------------
        optimizer.zero_grad()

        # -----------------------------------------------------
        # 2. Forward propagation
        # -----------------------------------------------------
        logits = model(X_batch)

        # -----------------------------------------------------
        # 3. Calculate loss
        # -----------------------------------------------------
        loss = criterion(
            logits,
            y_batch,
        )

        # -----------------------------------------------------
        # 4. Backpropagation
        # -----------------------------------------------------
        loss.backward()

        # -----------------------------------------------------
        # 5. Update weights and biases
        # -----------------------------------------------------
        optimizer.step()

        # -----------------------------------------------------
        # 6. Track batch loss
        # -----------------------------------------------------
        total_loss += loss.item()

    average_loss = (
        total_loss
        / len(train_loader)
    )

    return average_loss


def validate_one_epoch(
    model,
    val_loader,
    criterion,
):
    """
    Evaluate the model for one complete validation epoch.

    Parameters
    ----------
    model:
        PyTorch neural-network model.

    val_loader:
        DataLoader containing validation batches.

    criterion:
        Loss function used to measure validation error.

    Returns
    -------
    float
        Average validation loss for the epoch.
    """

    # Put model into evaluation mode
    model.eval()

    total_loss = 0.0

    # No gradients are required during validation
    with torch.no_grad():

        for X_batch, y_batch in val_loader:

            # -------------------------------------------------
            # 1. Forward propagation only
            # -------------------------------------------------
            logits = model(X_batch)

            # -------------------------------------------------
            # 2. Calculate validation loss
            # -------------------------------------------------
            loss = criterion(
                logits,
                y_batch,
            )

            # -------------------------------------------------
            # 3. Track validation loss
            # -------------------------------------------------
            total_loss += loss.item()

    average_loss = (
        total_loss
        / len(val_loader)
    )

    return average_loss