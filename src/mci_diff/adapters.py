"""Adapters for external sMRI feature extractors.

The project expects longitudinal scan features from an HFCN implementation or
another compatible 3D feature network.  This module only defines the small
interface used by the rest of the codebase; model definitions and checkpoints
stay in their original repositories.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Union

import torch
from torch import nn


def _unwrap_feature_output(
    output: Any,
    output_key: str = "features",
    output_index: int = 0,
) -> torch.Tensor:
    """Convert common external-network outputs to a ``[B, D]`` tensor."""

    if isinstance(output, Mapping):
        if output_key not in output:
            available = ", ".join(str(key) for key in output.keys())
            raise ValueError(
                "feature output has no key %s (got %s)" % (output_key, available)
            )
        output = output[output_key]
    elif isinstance(output, (tuple, list)):
        if len(output) <= output_index:
            raise ValueError("feature output tuple is too short")
        output = output[output_index]

    if not isinstance(output, torch.Tensor):
        raise TypeError("feature extractor must return a tensor, tuple or mapping")
    if output.ndim > 2:
        output = output.flatten(1)
    if output.ndim != 2:
        raise ValueError("feature extractor output must reduce to [B, D]")
    return output


class FeatureExtractorAdapter(ABC):
    """Minimal interface expected by the feature-extraction command."""

    @abstractmethod
    def encode(self, volumes: torch.Tensor) -> torch.Tensor:
        """Encode a batch of registered sMRI volumes as feature vectors."""


class HFCNFeatureAdapter(FeatureExtractorAdapter):
    """Wrap an external HFCN or another compatible PyTorch module."""

    def __init__(
        self,
        model: nn.Module,
        device: Union[str, torch.device] = "cpu",
        output_key: str = "features",
        output_index: int = 0,
        preprocess: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.output_key = str(output_key)
        self.output_index = int(output_index)
        self.preprocess = preprocess

    @torch.no_grad()
    def encode(self, volumes: torch.Tensor) -> torch.Tensor:
        if volumes.ndim != 5:
            raise ValueError("HFCN input must have shape [B, C, D, H, W]")
        values = volumes
        if self.preprocess is not None:
            values = self.preprocess(values)
        output = self.model(values.to(self.device))
        features = _unwrap_feature_output(
            output,
            self.output_key,
            self.output_index,
        )
        if features.shape[0] != volumes.shape[0]:
            raise ValueError("feature extractor changed the batch size")
        return features.detach().cpu().float()


class TorchScriptFeatureAdapter(HFCNFeatureAdapter):
    """Load a TorchScript export of an external sMRI feature network."""

    def __init__(
        self,
        checkpoint: Union[str, Path],
        device: Union[str, torch.device] = "cpu",
        output_key: str = "features",
        output_index: int = 0,
        preprocess: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        path = Path(checkpoint)
        if not path.is_file():
            raise FileNotFoundError("TorchScript checkpoint does not exist: %s" % path)
        model = torch.jit.load(str(path), map_location=torch.device(device))
        super().__init__(model, device, output_key, output_index, preprocess)


__all__ = [
    "FeatureExtractorAdapter",
    "HFCNFeatureAdapter",
    "TorchScriptFeatureAdapter",
]
