from __future__ import annotations

import copy
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import (
    Dataset,
    DataLoader,
)

from transformers import (
    AutoModel,
    AutoTokenizer,
)

from src.project_config import (
    TRANSFORMER_MODEL_NAME,
    TRANSFORMER_MAX_LENGTH,
    TRANSFORMER_BATCH_SIZE,
    TRANSFORMER_LEARNING_RATE,
    TRANSFORMER_WEIGHT_DECAY,
    TRANSFORMER_DROPOUT,
    TRANSFORMER_MAX_GRAD_NORM,
)


# ---------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------

def set_transformer_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------
# Dataset
# ---------------------------------------------------------

class SupportTicketTransformerDataset(Dataset):

    def __init__(
        self,
        texts,
        labels,
        tokenizer,
        max_length: int = TRANSFORMER_MAX_LENGTH,
    ):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):

        text = str(self.texts[index])
        label = int(self.labels[index])

        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None,
        )

        encoded["labels"] = label

        return encoded


# ---------------------------------------------------------
# Dynamic batch collation
# ---------------------------------------------------------

def create_collate_fn(tokenizer):

    def collate_fn(batch):

        labels = torch.tensor(
            [item.pop("labels") for item in batch],
            dtype=torch.long,
        )

        padded = tokenizer.pad(
            batch,
            padding=True,
            return_tensors="pt",
        )

        padded["labels"] = labels

        return padded

    return collate_fn


# ---------------------------------------------------------
# DataLoader
# ---------------------------------------------------------

def create_transformer_dataloader(
    texts,
    labels,
    tokenizer,
    batch_size: int = TRANSFORMER_BATCH_SIZE,
    shuffle: bool = False,
):

    dataset = SupportTicketTransformerDataset(
        texts=texts,
        labels=labels,
        tokenizer=tokenizer,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=create_collate_fn(tokenizer),
    )


# ---------------------------------------------------------
# Transformer classifier
# ---------------------------------------------------------

class TransformerTicketClassifier(nn.Module):

    def __init__(
        self,
        model_name: str,
        num_classes: int,
        dropout_rate: float = TRANSFORMER_DROPOUT,
    ):
        super().__init__()

        self.transformer = AutoModel.from_pretrained(
            model_name
        )

        hidden_size = (
            self.transformer.config.hidden_size
        )

        self.dropout = nn.Dropout(
            dropout_rate
        )

        self.classifier = nn.Linear(
            hidden_size,
            num_classes,
        )

    @staticmethod
    def mean_pool(
        token_embeddings,
        attention_mask,
    ):
        """
        Mean-pool contextual token representations,
        excluding padding positions.
        """

        mask = (
            attention_mask
            .unsqueeze(-1)
            .expand(token_embeddings.size())
            .float()
        )

        masked_embeddings = (
            token_embeddings * mask
        )

        summed_embeddings = (
            masked_embeddings.sum(dim=1)
        )

        token_counts = (
            mask.sum(dim=1)
            .clamp(min=1e-9)
        )

        return (
            summed_embeddings
            / token_counts
        )

    def forward(
        self,
        input_ids,
        attention_mask,
    ):

        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        token_embeddings = (
            outputs.last_hidden_state
        )

        pooled = self.mean_pool(
            token_embeddings,
            attention_mask,
        )

        pooled = self.dropout(
            pooled
        )

        logits = self.classifier(
            pooled
        )

        return logits


# ---------------------------------------------------------
# Tokenizer + model creation
# ---------------------------------------------------------

def create_transformer_model(
    num_classes: int,
    model_name: str = TRANSFORMER_MODEL_NAME,
):

    tokenizer = AutoTokenizer.from_pretrained(
        model_name
    )

    model = TransformerTicketClassifier(
        model_name=model_name,
        num_classes=num_classes,
    )

    return tokenizer, model


# ---------------------------------------------------------
# Trainable parameter information
# ---------------------------------------------------------

def count_parameters(model):

    total = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "frozen_parameters": total - trainable,
    }


# ---------------------------------------------------------
# One training epoch
# ---------------------------------------------------------

def train_transformer_epoch(
    model,
    dataloader,
    optimizer,
    criterion,
    device,
):

    model.train()

    total_loss = 0.0
    total_examples = 0
    total_correct = 0

    for batch in dataloader:

        input_ids = batch[
            "input_ids"
        ].to(device)

        attention_mask = batch[
            "attention_mask"
        ].to(device)

        labels = batch[
            "labels"
        ].to(device)

        optimizer.zero_grad()

        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        loss = criterion(
            logits,
            labels,
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            TRANSFORMER_MAX_GRAD_NORM,
        )

        optimizer.step()

        predictions = torch.argmax(
            logits,
            dim=1,
        )

        batch_size = labels.size(0)

        total_loss += (
            loss.item() * batch_size
        )

        total_examples += batch_size

        total_correct += (
            predictions == labels
        ).sum().item()

    return {
        "loss": total_loss / total_examples,
        "accuracy": (
            total_correct / total_examples
        ),
    }


# ---------------------------------------------------------
# Evaluation
# ---------------------------------------------------------

def evaluate_transformer(
    model,
    dataloader,
    criterion,
    device,
):

    model.eval()

    total_loss = 0.0
    total_examples = 0
    total_correct = 0

    all_labels = []
    all_predictions = []
    all_probabilities = []

    with torch.no_grad():

        for batch in dataloader:

            input_ids = batch[
                "input_ids"
            ].to(device)

            attention_mask = batch[
                "attention_mask"
            ].to(device)

            labels = batch[
                "labels"
            ].to(device)

            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            loss = criterion(
                logits,
                labels,
            )

            probabilities = torch.softmax(
                logits,
                dim=1,
            )

            predictions = torch.argmax(
                probabilities,
                dim=1,
            )

            batch_size = labels.size(0)

            total_loss += (
                loss.item() * batch_size
            )

            total_examples += batch_size

            total_correct += (
                predictions == labels
            ).sum().item()

            all_labels.extend(
                labels.cpu().numpy()
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_probabilities.extend(
                probabilities.cpu().numpy()
            )

    return {
        "loss": total_loss / total_examples,
        "accuracy": (
            total_correct / total_examples
        ),
        "labels": np.array(all_labels),
        "predictions": np.array(
            all_predictions
        ),
        "probabilities": np.array(
            all_probabilities
        ),
    }


# ---------------------------------------------------------
# Full fine-tuning
# ---------------------------------------------------------

def fine_tune_transformer(
    model,
    train_loader,
    validation_loader,
    device,
    max_epochs,
    patience,
):

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        [
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad
        ],
        lr=TRANSFORMER_LEARNING_RATE,
        weight_decay=TRANSFORMER_WEIGHT_DECAY,
    )

    history = []

    best_validation_loss = float("inf")
    best_model_state = None

    epochs_without_improvement = 0

    for epoch in range(
        1,
        max_epochs + 1,
    ):

        train_result = (
            train_transformer_epoch(
                model=model,
                dataloader=train_loader,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
            )
        )

        validation_result = (
            evaluate_transformer(
                model=model,
                dataloader=validation_loader,
                criterion=criterion,
                device=device,
            )
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": (
                    train_result["loss"]
                ),
                "train_accuracy": (
                    train_result["accuracy"]
                ),
                "validation_loss": (
                    validation_result["loss"]
                ),
                "validation_accuracy": (
                    validation_result["accuracy"]
                ),
            }
        )

        print(
            f"Epoch {epoch:02d} | "
            f"Train Loss: "
            f"{train_result['loss']:.4f} | "
            f"Train Acc: "
            f"{train_result['accuracy']:.4f} | "
            f"Val Loss: "
            f"{validation_result['loss']:.4f} | "
            f"Val Acc: "
            f"{validation_result['accuracy']:.4f}"
        )

        if (
            validation_result["loss"]
            < best_validation_loss
        ):

            best_validation_loss = (
                validation_result["loss"]
            )

            best_model_state = copy.deepcopy(
                model.state_dict()
            )

            epochs_without_improvement = 0

        else:

            epochs_without_improvement += 1

        if (
            epochs_without_improvement
            >= patience
        ):

            print(
                "Early stopping triggered."
            )

            break

    if best_model_state is not None:

        model.load_state_dict(
            best_model_state
        )

    return model, history