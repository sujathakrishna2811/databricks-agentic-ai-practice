"""
Explainability utilities for the Product Defect
Computer Vision project.

This module implements Grad-CAM for convolutional
image classifiers.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GradCAMResult:
    heatmap: torch.Tensor
    predicted_class: int
    target_class: int
    probabilities: torch.Tensor
    activation_shape: tuple[int, ...]
    gradient_shape: tuple[int, ...]


class GradCAM:
    """
    Generate Grad-CAM explanations for a CNN classifier.

    The target layer should be a convolutional layer whose
    output preserves spatial dimensions.
    """

    def __init__(
        self,
        model: nn.Module,
        target_layer: nn.Module,
    ):
        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        self._forward_hook = (
            self.target_layer.register_forward_hook(
                self._capture_activations
            )
        )

    def _capture_activations(
        self,
        module,
        inputs,
        output,
    ):
        """
        Save target-layer activations from the forward pass
        and attach a hook to capture their gradients.
        """
        self.activations = output

        if output.requires_grad:
            output.register_hook(
                self._capture_gradients
            )

    def _capture_gradients(
        self,
        gradient,
    ):
        """
        Save gradients with respect to the target-layer
        activations during the backward pass.
        """
        self.gradients = gradient

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: int | None = None,
    ) -> GradCAMResult:
        """
        Generate a Grad-CAM heatmap for one image.

        Parameters
        ----------
        input_tensor:
            Image tensor with shape [1, C, H, W].

        target_class:
            Class to explain. If None, explain the
            model's predicted class.
        """

        if input_tensor.ndim != 4:
            raise ValueError(
                "input_tensor must have shape [1, C, H, W]."
            )

        if input_tensor.shape[0] != 1:
            raise ValueError(
                "GradCAM currently expects exactly one image."
            )

        self.model.eval()

        device = next(
            self.model.parameters()
        ).device

        input_tensor = (
            input_tensor
            .detach()
            .clone()
            .to(device)
        )

        # Ensures a gradient graph exists even if some
        # model parameters are frozen.
        input_tensor.requires_grad_(True)

        self.activations = None
        self.gradients = None

        self.model.zero_grad(set_to_none=True)

        with torch.enable_grad():
            logits = self.model(input_tensor)

            probabilities = torch.softmax(
                logits,
                dim=1,
            )

            predicted_class = int(
                torch.argmax(
                    probabilities,
                    dim=1,
                ).item()
            )

            if target_class is None:
                target_class = predicted_class

            if not (
                0 <= target_class < logits.shape[1]
            ):
                raise ValueError(
                    "target_class is outside the model's "
                    "class range."
                )

            class_score = logits[
                0,
                target_class,
            ]

            class_score.backward()

        if self.activations is None:
            raise RuntimeError(
                "Target-layer activations were not captured."
            )

        if self.gradients is None:
            raise RuntimeError(
                "Target-layer gradients were not captured."
            )

        activation_shape = tuple(
            self.activations.shape
        )

        gradient_shape = tuple(
            self.gradients.shape
        )

        # Average gradients across height and width.
        # [1, 512, 7, 7]
        #          ↓
        # [1, 512, 1, 1]
        channel_importance = self.gradients.mean(
            dim=(2, 3),
            keepdim=True,
        )

        # Apply one importance value to every spatial
        # activation in the corresponding channel,
        # then combine all channels.
        #
        # [1, 512, 7, 7]
        #          ↓
        # [1, 1, 7, 7]
        cam = (
            channel_importance
            * self.activations
        ).sum(
            dim=1,
            keepdim=True,
        )

        # Keep positive contributions to the
        # selected class.
        cam = F.relu(cam)

        # Resize from target-layer spatial dimensions
        # to the input image dimensions.
        cam = F.interpolate(
            cam,
            size=input_tensor.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        # Normalize to [0, 1].
        cam_min = cam.min()
        cam_max = cam.max()

        if cam_max > cam_min:
            cam = (
                (cam - cam_min)
                / (cam_max - cam_min)
            )
        else:
            cam = torch.zeros_like(cam)

        heatmap = (
            cam[0, 0]
            .detach()
            .cpu()
        )

        return GradCAMResult(
            heatmap=heatmap,
            predicted_class=predicted_class,
            target_class=target_class,
            probabilities=(
                probabilities[0]
                .detach()
                .cpu()
            ),
            activation_shape=activation_shape,
            gradient_shape=gradient_shape,
        )

    def close(self):
        """
        Remove the forward hook from the target layer.
        """
        self._forward_hook.remove()