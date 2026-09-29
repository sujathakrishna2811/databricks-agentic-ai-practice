"""
Inference utilities for the Product Defect
Computer Vision project.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
from PIL import Image

import base64
import io

import pandas as pd
from PIL import Image
import mlflow

from src.image_preprocessing import (
    build_pretrained_eval_transform,
)


@dataclass
class PredictionResult:
    predicted_class_index: int
    predicted_class: str
    confidence: float
    probabilities: dict[str, float]


def predict_image(
    model: nn.Module,
    image: Image.Image,
    class_names: list[str],
    device: torch.device,
) -> PredictionResult:
    """
    Run deterministic inference for one product image.
    """

    if not class_names:
        raise ValueError(
            "class_names cannot be empty."
        )

    model = model.to(device)
    model.eval()

    image = image.convert("RGB")

    transform = (
        build_pretrained_eval_transform()
    )

    image_tensor = (
        transform(image)
        .unsqueeze(0)
        .to(device)
    )

    with torch.no_grad():
        logits = model(
            image_tensor
        )

        probabilities = torch.softmax(
            logits,
            dim=1,
        )[0]

    predicted_class_index = int(
        torch.argmax(
            probabilities
        ).item()
    )

    if predicted_class_index >= len(
        class_names
    ):
        raise ValueError(
            "Model output does not match "
            "class_names."
        )

    predicted_class = class_names[
        predicted_class_index
    ]

    confidence = float(
        probabilities[
            predicted_class_index
        ].item()
    )

    probability_mapping = {
        class_name: float(
            probabilities[index].item()
        )
        for index, class_name
        in enumerate(class_names)
    }

    return PredictionResult(
        predicted_class_index=(
            predicted_class_index
        ),
        predicted_class=predicted_class,
        confidence=confidence,
        probabilities=probability_mapping,
    )

def decode_base64_image(
    image_base64: str,
) -> Image.Image:
    """
    Decode a base64-encoded image into a PIL image.
    """

    if not image_base64:
        raise ValueError(
            "image_base64 cannot be empty."
        )

    try:
        image_bytes = base64.b64decode(
            image_base64,
            validate=True,
        )

        image = Image.open(
            io.BytesIO(image_bytes)
        )

        image.load()

    except Exception as exc:
        raise ValueError(
            "Invalid base64 image."
        ) from exc

    return image.convert("RGB")

class ProductDefectServingModel(
    mlflow.pyfunc.PythonModel
):
    """
    MLflow serving wrapper for the Product Defect
    image classifier.
    """

    def __init__(
        self,
        model,
        class_names: list[str],
    ):
        self.model = model
        self.class_names = class_names
        self.device = None

    def load_context(
        self,
        context,
    ):
        """
        Prepare the model when the serving
        environment loads it.
        """

        self.device = torch.device("cpu")

        self.model = self.model.to(
            self.device
        )

        self.model.eval()

    def predict(
        self,
        context,
        model_input: pd.DataFrame,
        params=None,
    ) -> pd.DataFrame:
        """
        Predict one or more base64-encoded images.
        """

        if "image_base64" not in model_input.columns:
            raise ValueError(
                "Input must contain an "
                "'image_base64' column."
            )

        predictions = []

        for image_base64 in model_input[
            "image_base64"
        ].tolist():

            image = decode_base64_image(
                image_base64
            )

            result = predict_image(
                model=self.model,
                image=image,
                class_names=self.class_names,
                device=self.device,
            )

            predictions.append({
                "predicted_class_index": (
                    result.predicted_class_index
                ),
                "predicted_class": (
                    result.predicted_class
                ),
                "confidence": (
                    result.confidence
                ),
                "defect_probability": (
                    result.probabilities[
                        "def_front"
                    ]
                ),
                "ok_probability": (
                    result.probabilities[
                        "ok_front"
                    ]
                ),
            })

        return pd.DataFrame(
            predictions
        )