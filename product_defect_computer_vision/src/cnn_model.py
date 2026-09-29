"""
CNN model architectures for the Product Defect
Computer Vision project.
"""

import torch
import torch.nn as nn

from src.project_config import (
    CLASSIFIER_HIDDEN_DIM,
    CNN_CHANNELS,
    IMAGE_CHANNELS,
)


class ProductDefectCNN(nn.Module):
    """
    Custom convolutional neural network for multiclass
    product defect image classification.
    """

    def __init__(
        self,
        num_classes: int,
    ):
        super().__init__()

        if num_classes < 2:
            raise ValueError(
                "num_classes must be at least 2."
            )

        channel_1, channel_2, channel_3 = CNN_CHANNELS

        self.features = nn.Sequential(
            nn.Conv2d(
                in_channels=IMAGE_CHANNELS,
                out_channels=channel_1,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2,
            ),

            nn.Conv2d(
                in_channels=channel_1,
                out_channels=channel_2,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2,
            ),

            nn.Conv2d(
                in_channels=channel_2,
                out_channels=channel_3,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2,
            ),
        )

        self.global_pool = nn.AdaptiveAvgPool2d(
            output_size=(1, 1)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),

            nn.Linear(
                in_features=channel_3,
                out_features=CLASSIFIER_HIDDEN_DIM,
            ),
            nn.ReLU(),

            nn.Linear(
                in_features=CLASSIFIER_HIDDEN_DIM,
                out_features=num_classes,
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run the forward pass and return raw class logits.
        """

        x = self.features(x)

        x = self.global_pool(x)

        logits = self.classifier(x)

        return logits