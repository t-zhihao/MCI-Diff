"""Core feature diffusion model used by MCI-Diff.

The paper diffuses HFCN features instead of full MRI volumes.  Time points are
fixed to 0/6/12/18/24/36 months; a visit can be absent, but its token stays in
the Transformer and receives a missing-value embedding.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F


VISIT_MONTHS = (0, 6, 12, 18, 24, 36)
INTERPOLATION = 0
EXTRAPOLATION = 1


def make_beta_schedule(
    name: str = "cosine",
    timesteps: int = 40,
    beta_start: float = 1.0e-4,
    beta_end: float = 2.0e-2,
) -> torch.Tensor:
    """Build a linear or improved-DDPM cosine schedule."""

    if int(timesteps) < 1:
        raise ValueError("timesteps must be positive")
    name = str(name).lower()
    if name == "linear":
        if not 0.0 < beta_start <= beta_end < 1.0:
            raise ValueError("linear beta bounds must satisfy 0 < start <= end < 1")
        betas = torch.linspace(beta_start, beta_end, int(timesteps), dtype=torch.float64)
    elif name == "cosine":
        # alpha_bar(t)=cos^2(((t/T+s)/(1+s))*pi/2), s=0.008
        points = torch.linspace(0, int(timesteps), int(timesteps) + 1, dtype=torch.float64)
        angles = ((points / int(timesteps) + 0.008) / 1.008) * math.pi / 2.0
        alpha_bar = torch.cos(angles).pow(2)
        alpha_bar = alpha_bar / alpha_bar[0].clone()
        betas = 1.0 - alpha_bar[1:] / alpha_bar[:-1]
        betas = betas.clamp(1.0e-8, 0.999)
    else:
        raise ValueError("schedule must be 'linear' or 'cosine'")
    return betas.float()


class DiffusionSchedule(nn.Module):
    """DDPM forward and exact q(x[t-1] | x[t], x[0]) coefficients."""

    def __init__(
        self,
        timesteps: int = 40,
        name: str = "cosine",
        beta_start: float = 1.0e-4,
        beta_end: float = 2.0e-2,
    ) -> None:
        super().__init__()
        betas = make_beta_schedule(name, timesteps, beta_start, beta_end)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        alpha_bars_prev = torch.cat(
            [torch.ones_like(alpha_bars[:1]), alpha_bars[:-1]], dim=0
        )
        one_minus_bar = (1.0 - alpha_bars).clamp_min(1.0e-20)

        # q(x[t-1] | x[t], x[0]) = N(c1*x[0] + c2*x[t], posterior_var)
        posterior_variance = betas * (1.0 - alpha_bars_prev) / one_minus_bar
        posterior_coef1 = betas * alpha_bars_prev.sqrt() / one_minus_bar
        posterior_coef2 = (1.0 - alpha_bars_prev) * alphas.sqrt() / one_minus_bar

        self.timesteps = int(timesteps)
        self.name = str(name).lower()
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("alpha_bars_prev", alpha_bars_prev)
        self.register_buffer("sqrt_alpha_bars", alpha_bars.sqrt())
        self.register_buffer("sqrt_one_minus_alpha_bars", (1.0 - alpha_bars).sqrt())
        self.register_buffer("sqrt_recip_alpha_bars", alpha_bars.rsqrt())
        self.register_buffer(
            "sqrt_recipm1_alpha_bars", (alpha_bars.reciprocal() - 1.0).sqrt()
        )
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer(
            "posterior_log_variance",
            posterior_variance.clamp_min(1.0e-20).log(),
        )
        self.register_buffer("posterior_coef1", posterior_coef1)
        self.register_buffer("posterior_coef2", posterior_coef2)

    @staticmethod
    def extract(values: torch.Tensor, timestep: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if timestep.ndim != 1 or timestep.shape[0] != target.shape[0]:
            raise ValueError("timestep must have shape [B]")
        selected = values.gather(0, timestep.long())
        shape = (target.shape[0],) + (1,) * (target.ndim - 1)
        return selected.reshape(shape).to(device=target.device, dtype=target.dtype)

    def q_mean_variance(
        self, clean: torch.Tensor, timestep: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mean = self.extract(self.sqrt_alpha_bars, timestep, clean) * clean
        variance = self.extract(1.0 - self.alpha_bars, timestep, clean)
        return mean, variance

    def q_sample(
        self,
        clean: torch.Tensor,
        timestep: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """x[t] = sqrt(alpha_bar[t])*x[0] + sqrt(1-alpha_bar[t])*eps."""

        if noise is None:
            noise = randn_like(clean, generator)
        if noise.shape != clean.shape:
            raise ValueError("noise and clean features must have the same shape")
        noisy = (
            self.extract(self.sqrt_alpha_bars, timestep, clean) * clean
            + self.extract(self.sqrt_one_minus_alpha_bars, timestep, clean) * noise
        )
        return noisy, noise

    def predict_x0(
        self,
        noisy: torch.Tensor,
        timestep: torch.Tensor,
        predicted_noise: torch.Tensor,
    ) -> torch.Tensor:
        if noisy.shape != predicted_noise.shape:
            raise ValueError("predicted noise has the wrong shape")
        return (
            self.extract(self.sqrt_recip_alpha_bars, timestep, noisy) * noisy
            - self.extract(self.sqrt_recipm1_alpha_bars, timestep, noisy)
            * predicted_noise
        )

    def q_posterior(
        self,
        clean: torch.Tensor,
        noisy: torch.Tensor,
        timestep: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean = (
            self.extract(self.posterior_coef1, timestep, noisy) * clean
            + self.extract(self.posterior_coef2, timestep, noisy) * noisy
        )
        variance = self.extract(self.posterior_variance, timestep, noisy)
        log_variance = self.extract(self.posterior_log_variance, timestep, noisy)
        return mean, variance, log_variance


def randn_like(
    value: torch.Tensor, generator: Optional[torch.Generator] = None
) -> torch.Tensor:
    # Use an explicit shape so the optional generator works on every device.
    return torch.randn(
        value.shape,
        dtype=value.dtype,
        device=value.device,
        generator=generator,
    )


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        if int(dimension) < 4:
            raise ValueError("time embedding dimension must be at least 4")
        self.dimension = int(dimension)

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        half = self.dimension // 2
        exponent = -math.log(10000.0) / max(half - 1, 1)
        frequencies = torch.exp(
            torch.arange(half, device=timestep.device, dtype=torch.float32) * exponent
        )
        phase = timestep.float().unsqueeze(-1) * frequencies.unsqueeze(0)
        result = torch.cat([phase.sin(), phase.cos()], dim=-1)
        if self.dimension % 2:
            result = F.pad(result, (0, 1))
        return result


def _init_linear_layers(module: nn.Module) -> None:
    for layer in module.modules():
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)


class ConditionalTransformerDenoiser(nn.Module):
    """Predict epsilon for one target feature given the six-visit sequence.

    ``condition_observed`` is True where an actual or previously imputed feature
    may be used. Missing visits remain normal Transformer tokens; they are never
    passed as a key-padding mask.
    """

    def __init__(
        self,
        feature_dim: int,
        model_dim: int = 512,
        num_visits: int = 6,
        num_layers: int = 6,
        num_heads: int = 8,
        feedforward_dim: int = 2048,
        dropout: float = 0.1,
        time_dim: int = 128,
    ) -> None:
        super().__init__()
        if min(feature_dim, model_dim, num_visits, num_layers, num_heads, time_dim) < 1:
            raise ValueError("model dimensions must be positive")
        if model_dim % num_heads:
            raise ValueError("model_dim must be divisible by num_heads")
        if num_visits != len(VISIT_MONTHS):
            raise ValueError("MCI-Diff uses exactly six visits")

        self.feature_dim = int(feature_dim)
        self.model_dim = int(model_dim)
        self.num_visits = int(num_visits)

        self.feature_projection = nn.Linear(feature_dim, model_dim)
        self.noisy_projection = nn.Linear(feature_dim, model_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, model_dim))
        self.position_embedding = nn.Embedding(num_visits, model_dim)
        self.missing_embedding = nn.Embedding(2, model_dim)
        self.task_embedding = nn.Embedding(2, model_dim)
        self.target_embedding = nn.Embedding(num_visits, model_dim)
        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, model_dim),
            nn.SiLU(),
            nn.Linear(model_dim, model_dim),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.output = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, model_dim),
            nn.SiLU(),
            nn.Linear(model_dim, feature_dim),
        )
        _init_linear_layers(self)
        nn.init.normal_(self.mask_token, std=0.02)
        nn.init.normal_(self.output[-1].weight, std=1.0e-3)

    def forward(
        self,
        noisy_target: torch.Tensor,
        timestep: torch.Tensor,
        condition_features: torch.Tensor,
        condition_observed: torch.Tensor,
        target_index: torch.Tensor,
        task_id: torch.Tensor,
    ) -> torch.Tensor:
        batch = noisy_target.shape[0]
        if noisy_target.ndim != 2 or noisy_target.shape[1] != self.feature_dim:
            raise ValueError("noisy_target must have shape [B, feature_dim]")
        if condition_features.shape != (batch, self.num_visits, self.feature_dim):
            raise ValueError("condition_features must have shape [B, 6, feature_dim]")
        if condition_observed.shape != (batch, self.num_visits):
            raise ValueError("condition_observed must have shape [B, 6]")
        if timestep.shape != (batch,) or target_index.shape != (batch,):
            raise ValueError("timestep and target_index must have shape [B]")
        if task_id.shape != (batch,):
            raise ValueError("task_id must have shape [B]")
        if bool(((target_index < 0) | (target_index >= self.num_visits)).any()):
            raise ValueError("target visit is outside the six-visit sequence")
        if bool(((task_id < 0) | (task_id > 1)).any()):
            raise ValueError("task_id must be interpolation=0 or extrapolation=1")

        observed = condition_observed.bool()
        rows = torch.arange(batch, device=target_index.device)
        if bool(observed[rows, target_index.long()].any()):
            raise ValueError("target feature leaked into the condition sequence")

        feature_tokens = self.feature_projection(condition_features)
        masked_tokens = self.mask_token.expand(batch, self.num_visits, -1)
        condition_tokens = torch.where(
            observed.unsqueeze(-1), feature_tokens, masked_tokens
        )
        positions = torch.arange(self.num_visits, device=noisy_target.device)
        task = self.task_embedding(task_id.long())
        condition_tokens = (
            condition_tokens
            + self.position_embedding(positions).unsqueeze(0)
            + self.missing_embedding((~observed).long())
            + task.unsqueeze(1)
        )

        query = (
            self.noisy_projection(noisy_target)
            + self.time_embedding(timestep)
            + self.target_embedding(target_index.long())
            + task
        )
        tokens = torch.cat([query.unsqueeze(1), condition_tokens], dim=1)
        encoded = self.transformer(tokens.transpose(0, 1)).transpose(0, 1)
        return self.output(encoded[:, 0])


@dataclass
class ReconstructionBatch:
    clean_target: torch.Tensor
    condition_features: torch.Tensor
    condition_observed: torch.Tensor
    target_index: torch.Tensor
    task_id: torch.Tensor


@dataclass
class DiffusionLoss:
    loss: torch.Tensor
    per_sample: torch.Tensor
    predicted_noise: torch.Tensor
    noise: torch.Tensor
    noisy_target: torch.Tensor
    timestep: torch.Tensor
    target_index: torch.Tensor
    condition_observed: torch.Tensor


def _draw_index(
    indices: torch.Tensor, generator: Optional[torch.Generator]
) -> int:
    if indices.numel() == 0:
        raise ValueError("no valid visit is available for this reconstruction task")
    pick = torch.randint(
        0,
        int(indices.numel()),
        (1,),
        device=indices.device,
        generator=generator,
    ).item()
    return int(indices[int(pick)].item())


def _coerce_target_indices(
    target_index: Union[int, torch.Tensor], batch: int, device: torch.device
) -> torch.Tensor:
    if isinstance(target_index, int):
        result = torch.full((batch,), target_index, dtype=torch.long, device=device)
    else:
        result = target_index.to(device=device, dtype=torch.long)
    if result.shape != (batch,):
        raise ValueError("target_index must be an int or a [B] tensor")
    return result


def make_interpolation_mask(
    observed: torch.Tensor,
    target_index: torch.Tensor,
    difficulty: int = 1,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Mask the target plus up to d-1 other intermediate visits."""

    if observed.ndim != 2 or observed.shape[1] != len(VISIT_MONTHS):
        raise ValueError("observed must have shape [B, 6]")
    if not 1 <= int(difficulty) <= len(VISIT_MONTHS) - 2:
        raise ValueError("interpolation difficulty must lie in [1, 4]")
    result = observed.bool().clone()
    for row in range(result.shape[0]):
        target = int(target_index[row].item())
        if target < 1 or target > len(VISIT_MONTHS) - 2:
            raise ValueError("interpolation targets are visits 1..4")
        if not bool(observed[row, target]):
            raise ValueError("interpolation target is not present in the source sequence")
        candidates = torch.nonzero(observed[row, 1:-1].bool(), as_tuple=False).flatten() + 1
        candidates = candidates[candidates != target]
        result[row, target] = False
        extra = min(int(difficulty) - 1, int(candidates.numel()))
        if extra:
            order = torch.randperm(
                int(candidates.numel()),
                device=candidates.device,
                generator=generator,
            )[:extra]
            result[row, candidates[order]] = False
    return result


def make_extrapolation_mask(
    observed: torch.Tensor, target_index: torch.Tensor
) -> torch.Tensor:
    """Hide target and all later visits. Target 5 (36 months) is valid."""

    if observed.ndim != 2 or observed.shape[1] != len(VISIT_MONTHS):
        raise ValueError("observed must have shape [B, 6]")
    result = observed.bool().clone()
    for row in range(result.shape[0]):
        target = int(target_index[row].item())
        if target < 1 or target >= len(VISIT_MONTHS):
            raise ValueError("extrapolation targets are visits 1..5")
        if not bool(observed[row, target]):
            raise ValueError("extrapolation target is not present in the source sequence")
        result[row, target:] = False
    return result


def make_reconstruction_batch(
    features: torch.Tensor,
    observed: torch.Tensor,
    task: str,
    difficulty: int = 1,
    target_index: Optional[Union[int, torch.Tensor]] = None,
    generator: Optional[torch.Generator] = None,
) -> ReconstructionBatch:
    """Prepare interpolation or extrapolation examples for epsilon regression."""

    if features.ndim != 3 or features.shape[1] != len(VISIT_MONTHS):
        raise ValueError("features must have shape [B, 6, D]")
    if observed.shape != features.shape[:2]:
        raise ValueError("observed mask has the wrong shape")
    batch = features.shape[0]
    observed_bool = observed.bool()
    task_name = str(task).lower()
    if task_name in ("extrapolation", "ext") and not 1 <= int(difficulty) <= len(VISIT_MONTHS) - 1:
        raise ValueError("extrapolation difficulty must lie in [1, 5]")

    if target_index is None:
        chosen = []
        for row in range(batch):
            if task_name in ("interpolation", "int"):
                eligible = torch.nonzero(
                    observed_bool[row, 1:-1], as_tuple=False
                ).flatten() + 1
            elif task_name in ("extrapolation", "ext"):
                # At difficulty d the randomly chosen horizon lies in the final d visits.
                first = max(1, len(VISIT_MONTHS) - int(difficulty))
                eligible = torch.nonzero(
                    observed_bool[row, first:], as_tuple=False
                ).flatten() + first
            else:
                raise ValueError("task must be interpolation or extrapolation")
            chosen.append(_draw_index(eligible, generator))
        targets = torch.tensor(chosen, dtype=torch.long, device=features.device)
    else:
        targets = _coerce_target_indices(target_index, batch, features.device)

    if task_name in ("interpolation", "int"):
        condition_observed = make_interpolation_mask(
            observed_bool, targets, difficulty, generator
        )
        task_ids = torch.full(
            (batch,), INTERPOLATION, dtype=torch.long, device=features.device
        )
    elif task_name in ("extrapolation", "ext"):
        if not 1 <= int(difficulty) <= len(VISIT_MONTHS) - 1:
            raise ValueError("extrapolation difficulty must lie in [1, 5]")
        condition_observed = make_extrapolation_mask(observed_bool, targets)
        task_ids = torch.full(
            (batch,), EXTRAPOLATION, dtype=torch.long, device=features.device
        )
    else:
        raise ValueError("task must be interpolation or extrapolation")

    rows = torch.arange(batch, device=features.device)
    clean_target = features[rows, targets]
    return ReconstructionBatch(
        clean_target=clean_target,
        condition_features=features,
        condition_observed=condition_observed,
        target_index=targets,
        task_id=task_ids,
    )


class ConditionalFeatureDDPM(nn.Module):
    """Small wrapper joining the denoiser, DDPM loss and reverse sampler."""

    def __init__(
        self,
        denoiser: ConditionalTransformerDenoiser,
        schedule: Optional[DiffusionSchedule] = None,
    ) -> None:
        super().__init__()
        self.denoiser = denoiser
        self.schedule = schedule or DiffusionSchedule()

    def loss_from_condition(
        self,
        clean_target: torch.Tensor,
        condition_features: torch.Tensor,
        condition_observed: torch.Tensor,
        target_index: torch.Tensor,
        task_id: torch.Tensor,
        generator: Optional[torch.Generator] = None,
        reduction: str = "mean",
    ) -> DiffusionLoss:
        """Dataset-facing loss using the phase mask selected by the trainer."""

        return self.denoising_loss(
            clean_target,
            condition_features,
            condition_observed,
            target_index,
            task_id,
            generator=generator,
            reduction=reduction,
        )

    def denoising_loss(
        self,
        clean_target: torch.Tensor,
        condition_features: torch.Tensor,
        condition_observed: torch.Tensor,
        target_index: torch.Tensor,
        task_id: torch.Tensor,
        timestep: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        reduction: str = "mean",
    ) -> DiffusionLoss:
        """Epsilon loss for masks supplied by ``LongitudinalTaskDataset``."""

        batch = clean_target.shape[0]
        if timestep is None:
            timestep = torch.randint(
                0,
                self.schedule.timesteps,
                (batch,),
                dtype=torch.long,
                device=clean_target.device,
                generator=generator,
            )
        noisy, used_noise = self.schedule.q_sample(
            clean_target,
            timestep,
            noise=noise,
            generator=generator,
        )
        predicted = self.denoiser(
            noisy,
            timestep,
            condition_features,
            condition_observed,
            target_index,
            task_id,
        )
        per_sample = (predicted - used_noise).pow(2).flatten(1).mean(dim=1)
        if reduction == "mean":
            loss = per_sample.mean()
        elif reduction == "sum":
            loss = per_sample.sum()
        elif reduction == "none":
            loss = per_sample
        else:
            raise ValueError("reduction must be mean, sum or none")
        return DiffusionLoss(
            loss=loss,
            per_sample=per_sample,
            predicted_noise=predicted,
            noise=used_noise,
            noisy_target=noisy,
            timestep=timestep,
            target_index=target_index,
            condition_observed=condition_observed,
        )

    def reconstruction_loss(
        self,
        features: torch.Tensor,
        observed: torch.Tensor,
        task: str,
        difficulty: int = 1,
        target_index: Optional[Union[int, torch.Tensor]] = None,
        generator: Optional[torch.Generator] = None,
        reduction: str = "mean",
    ) -> DiffusionLoss:
        item = make_reconstruction_batch(
            features,
            observed,
            task,
            difficulty,
            target_index,
            generator,
        )
        return self.denoising_loss(
            item.clean_target,
            item.condition_features,
            item.condition_observed,
            item.target_index,
            item.task_id,
            generator=generator,
            reduction=reduction,
        )

    def p_mean_variance(
        self,
        noisy: torch.Tensor,
        timestep: torch.Tensor,
        condition_features: torch.Tensor,
        condition_observed: torch.Tensor,
        target_index: torch.Tensor,
        task_id: torch.Tensor,
        clip_x0: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        predicted_noise = self.denoiser(
            noisy,
            timestep,
            condition_features,
            condition_observed,
            target_index,
            task_id,
        )
        predicted_x0 = self.schedule.predict_x0(noisy, timestep, predicted_noise)
        if clip_x0 is not None:
            if float(clip_x0) <= 0:
                raise ValueError("clip_x0 must be positive")
            predicted_x0 = predicted_x0.clamp(-float(clip_x0), float(clip_x0))
        mean, variance, log_variance = self.schedule.q_posterior(
            predicted_x0, noisy, timestep
        )
        return mean, variance, log_variance, predicted_x0

    @torch.no_grad()
    def sample(
        self,
        condition_features: torch.Tensor,
        condition_observed: torch.Tensor,
        target_index: Union[int, torch.Tensor],
        task: str = "extrapolation",
        initial_noise: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        clip_x0: Optional[float] = None,
    ) -> torch.Tensor:
        if condition_features.ndim != 3:
            raise ValueError("condition_features must have shape [B, 6, D]")
        batch = condition_features.shape[0]
        targets = _coerce_target_indices(
            target_index, batch, condition_features.device
        )
        if task.lower() in ("interpolation", "int"):
            task_value = INTERPOLATION
        elif task.lower() in ("extrapolation", "ext"):
            task_value = EXTRAPOLATION
        else:
            raise ValueError("task must be interpolation or extrapolation")
        task_ids = torch.full(
            (batch,), task_value, dtype=torch.long, device=condition_features.device
        )
        observed = condition_observed.bool().clone()
        rows = torch.arange(batch, device=condition_features.device)
        observed[rows, targets] = False
        if task_value == EXTRAPOLATION:
            for row in range(batch):
                observed[row, int(targets[row].item()) :] = False

        if initial_noise is None:
            value = torch.randn(
                (batch, self.denoiser.feature_dim),
                dtype=condition_features.dtype,
                device=condition_features.device,
                generator=generator,
            )
        else:
            value = initial_noise
        expected = (batch, self.denoiser.feature_dim)
        if value.shape != expected:
            raise ValueError("initial_noise must have shape [B, feature_dim]")

        for step in range(self.schedule.timesteps - 1, -1, -1):
            timestep = torch.full(
                (batch,), step, dtype=torch.long, device=condition_features.device
            )
            mean, variance, _, predicted_x0 = self.p_mean_variance(
                value,
                timestep,
                condition_features,
                observed,
                targets,
                task_ids,
                clip_x0,
            )
            if step:
                value = mean + variance.clamp_min(0.0).sqrt() * randn_like(
                    value, generator
                )
            else:
                value = predicted_x0
        return value

    @torch.no_grad()
    def sample_candidates(
        self,
        condition_features: torch.Tensor,
        condition_observed: torch.Tensor,
        target_index: Union[int, torch.Tensor],
        num_candidates: int = 20,
        task: str = "extrapolation",
        generator: Optional[torch.Generator] = None,
        clip_x0: Optional[float] = None,
    ) -> torch.Tensor:
        """Return candidate features with shape [B, N, D]."""

        if int(num_candidates) < 1:
            raise ValueError("num_candidates must be positive")
        batch = condition_features.shape[0]
        targets = _coerce_target_indices(
            target_index, batch, condition_features.device
        )
        repeated_features = condition_features.repeat_interleave(num_candidates, dim=0)
        repeated_observed = condition_observed.repeat_interleave(num_candidates, dim=0)
        repeated_targets = targets.repeat_interleave(num_candidates, dim=0)
        samples = self.sample(
            repeated_features,
            repeated_observed,
            repeated_targets,
            task=task,
            generator=generator,
            clip_x0=clip_x0,
        )
        return samples.reshape(batch, int(num_candidates), -1)


class TrajectoryClassifier(nn.Module):
    """Conversion head over baseline and generated visits."""

    def __init__(
        self,
        feature_dim: int,
        model_dim: int = 256,
        num_layers: int = 2,
        num_heads: int = 4,
        feedforward_dim: int = 1024,
        dropout: float = 0.1,
        num_classes: int = 2,
        num_visits: int = 6,
        architecture: str = "transformer",
        pooling: str = "cls",
    ) -> None:
        super().__init__()
        architecture = str(architecture).lower()
        pooling = str(pooling).lower()
        if architecture not in ("transformer", "gru", "mlp"):
            raise ValueError("architecture must be transformer, gru or mlp")
        if pooling not in ("cls", "mean", "last"):
            raise ValueError("pooling must be cls, mean or last")
        if min(feature_dim, model_dim, num_layers, num_classes, num_visits) < 1:
            raise ValueError("classifier dimensions must be positive")
        if architecture == "transformer" and model_dim % num_heads:
            raise ValueError("model_dim must be divisible by num_heads")
        self.feature_dim = int(feature_dim)
        self.model_dim = int(model_dim)
        self.num_visits = int(num_visits)
        self.architecture = architecture
        self.pooling = pooling
        self.input_projection = nn.Linear(feature_dim, model_dim)
        self.position_embedding = nn.Embedding(num_visits, model_dim)
        self.missing_embedding = nn.Embedding(2, model_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, model_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, model_dim))
        self.transformer = None  # type: Optional[nn.Module]
        self.gru = None  # type: Optional[nn.GRU]
        self.mlp_encoder = None  # type: Optional[nn.Module]
        if architecture == "transformer":
            layer = nn.TransformerEncoderLayer(
                d_model=model_dim,
                nhead=num_heads,
                dim_feedforward=feedforward_dim,
                dropout=dropout,
                activation="gelu",
            )
            self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        elif architecture == "gru":
            self.gru = nn.GRU(
                input_size=model_dim,
                hidden_size=model_dim,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0.0,
                batch_first=True,
            )
        else:
            input_width = model_dim * num_visits if pooling == "cls" else model_dim
            blocks = []
            for layer_index in range(num_layers):
                blocks.extend(
                    [
                        nn.Linear(input_width if layer_index == 0 else model_dim, model_dim),
                        nn.SiLU(),
                        nn.Dropout(dropout),
                    ]
                )
            self.mlp_encoder = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.LayerNorm(model_dim), nn.Dropout(dropout), nn.Linear(model_dim, num_classes)
        )
        _init_linear_layers(self)
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)

    def forward(
        self, features: torch.Tensor, observed: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if features.ndim != 3 or features.shape[1:] != (
            self.num_visits,
            self.feature_dim,
        ):
            raise ValueError("features must have shape [B, 6, feature_dim]")
        batch = features.shape[0]
        if observed is None:
            observed = torch.ones(
                (batch, self.num_visits), dtype=torch.bool, device=features.device
            )
        if observed.shape != features.shape[:2]:
            raise ValueError("observed has the wrong shape")
        observed = observed.bool()
        projected = self.input_projection(features)
        tokens = torch.where(
            observed.unsqueeze(-1),
            projected,
            self.mask_token.expand(batch, self.num_visits, -1),
        )
        positions = torch.arange(self.num_visits, device=features.device)
        tokens = (
            tokens
            + self.position_embedding(positions).unsqueeze(0)
            + self.missing_embedding((~observed).long())
        )
        if self.architecture == "transformer":
            assert self.transformer is not None
            sequence = torch.cat(
                [self.cls_token.expand(batch, -1, -1), tokens], dim=1
            )
            encoded = self.transformer(sequence.transpose(0, 1)).transpose(0, 1)
            if self.pooling == "cls":
                pooled = encoded[:, 0]
            elif self.pooling == "mean":
                pooled = encoded[:, 1:].mean(dim=1)
            else:
                pooled = encoded[:, -1]
        elif self.architecture == "gru":
            assert self.gru is not None
            # Appended CLS sees all earlier visits in a unidirectional GRU.
            sequence = torch.cat(
                [tokens, self.cls_token.expand(batch, -1, -1)], dim=1
            )
            encoded, _ = self.gru(sequence)
            if self.pooling == "cls":
                pooled = encoded[:, -1]
            elif self.pooling == "mean":
                pooled = encoded[:, :-1].mean(dim=1)
            else:
                pooled = encoded[:, -2]
        else:
            assert self.mlp_encoder is not None
            if self.pooling == "cls":
                pooled_input = tokens.reshape(batch, -1)
            elif self.pooling == "mean":
                pooled_input = tokens.mean(dim=1)
            else:
                pooled_input = tokens[:, -1]
            pooled = self.mlp_encoder(pooled_input)
        return self.head(pooled)


__all__ = [
    "EXTRAPOLATION",
    "INTERPOLATION",
    "VISIT_MONTHS",
    "ConditionalFeatureDDPM",
    "ConditionalTransformerDenoiser",
    "DiffusionLoss",
    "DiffusionSchedule",
    "ReconstructionBatch",
    "SinusoidalTimeEmbedding",
    "TrajectoryClassifier",
    "make_beta_schedule",
    "make_extrapolation_mask",
    "make_interpolation_mask",
    "make_reconstruction_batch",
    "randn_like",
]
