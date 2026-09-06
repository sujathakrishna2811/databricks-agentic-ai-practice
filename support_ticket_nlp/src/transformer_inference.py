"""
Inference utilities for the fine-tuned support-ticket
Transformer classifier.
"""

from __future__ import annotations

import json

import mlflow.pyfunc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from transformers import (
    AutoModel,
    AutoTokenizer,
)

from src.project_config import (
    TEXT_COL,
)


class TransformerTicketInferenceModel(
    nn.Module
):
    """
    Transformer encoder + masked mean pooling +
    classification head used during MLflow inference.
    """

    def __init__(
        self,
        transformer,
        hidden_size: int,
        num_classes: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.transformer = transformer

        self.dropout = nn.Dropout(
            dropout
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
        mask = (
            attention_mask
            .unsqueeze(-1)
            .expand(
                token_embeddings.size()
            )
            .float()
        )

        summed = (
            token_embeddings
            * mask
        ).sum(dim=1)

        counts = (
            mask.sum(dim=1)
            .clamp(min=1e-9)
        )

        return summed / counts

    def forward(
        self,
        input_ids,
        attention_mask,
    ):
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        pooled = self.mean_pool(
            outputs.last_hidden_state,
            attention_mask,
        )

        pooled = self.dropout(
            pooled
        )

        return self.classifier(
            pooled
        )


class TransformerTicketPyFunc(
    mlflow.pyfunc.PythonModel
):
    """
    MLflow PyFunc wrapper that accepts raw ticket text
    and returns category predictions and probabilities.
    """

    def load_context(
        self,
        context,
    ):
        with open(
            context.artifacts[
                "metadata"
            ],
            "r",
            encoding="utf-8",
        ) as file:
            metadata = json.load(
                file
            )

        self.class_names = (
            metadata["class_names"]
        )

        self.max_length = int(
            metadata["max_length"]
        )

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        self.tokenizer = (
            AutoTokenizer.from_pretrained(
                context.artifacts[
                    "tokenizer"
                ]
            )
        )

        transformer = (
            AutoModel.from_pretrained(
                context.artifacts[
                    "transformer"
                ]
            )
        )

        self.model = (
            TransformerTicketInferenceModel(
                transformer=transformer,
                hidden_size=int(
                    transformer.config.hidden_size
                ),
                num_classes=len(
                    self.class_names
                ),
                dropout=0.0,
            )
        )

        classifier_state = torch.load(
            context.artifacts[
                "classifier"
            ],
            map_location="cpu",
            weights_only=True,
        )

        self.model.classifier.load_state_dict(
            classifier_state
        )

        self.model = self.model.to(
            self.device
        )

        self.model.eval()

    def predict(
        self,
        context,
        model_input,
        params=None,
    ):
        if TEXT_COL not in model_input.columns:
            raise ValueError(
                f"Input must contain "
                f"'{TEXT_COL}' column."
            )

        texts = (
            model_input[TEXT_COL]
            .astype(str)
            .tolist()
        )

        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        input_ids = encoded[
            "input_ids"
        ].to(self.device)

        attention_mask = encoded[
            "attention_mask"
        ].to(self.device)

        with torch.no_grad():

            logits = self.model(
                input_ids=input_ids,
                attention_mask=(
                    attention_mask
                ),
            )

            probabilities = (
                torch.softmax(
                    logits,
                    dim=1,
                )
                .cpu()
                .numpy()
            )

        predicted_ids = np.argmax(
            probabilities,
            axis=1,
        )

        predicted_categories = [
            self.class_names[index]
            for index in predicted_ids
        ]

        confidence = np.max(
            probabilities,
            axis=1,
        )

        result = pd.DataFrame(
            {
                "predicted_category":
                    predicted_categories,

                "confidence":
                    confidence,
            }
        )

        for index, class_name in enumerate(
            self.class_names
        ):
            result[
                f"probability_{class_name}"
            ] = probabilities[:, index]

        return result