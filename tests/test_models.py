import numpy as np
import pytest
import torch

from mci_diff.models import (
    ConditionalFeatureDDPM,
    ConditionalTransformerDenoiser,
    DiffusionSchedule,
    TrajectoryClassifier,
    make_extrapolation_mask,
    make_interpolation_mask,
    make_reconstruction_batch,
)


def small_diffusion(steps=3):
    denoiser = ConditionalTransformerDenoiser(
        feature_dim=8,
        model_dim=16,
        num_layers=1,
        num_heads=4,
        feedforward_dim=32,
        dropout=0.0,
        time_dim=8,
    )
    return ConditionalFeatureDDPM(
        denoiser,
        DiffusionSchedule(timesteps=steps, name="linear", beta_end=0.01),
    )


def test_forward_diffusion_closed_form_and_last_posterior():
    schedule = DiffusionSchedule(timesteps=4, name="linear")
    clean = torch.tensor([[1.0, -2.0], [0.5, 0.25]])
    noise = torch.tensor([[0.2, 0.1], [-0.3, 0.4]])
    timestep = torch.tensor([0, 3])
    noisy, returned_noise = schedule.q_sample(clean, timestep, noise=noise)

    a = schedule.alpha_bars[timestep].sqrt().unsqueeze(1)
    b = (1.0 - schedule.alpha_bars[timestep]).sqrt().unsqueeze(1)
    assert torch.allclose(noisy, a * clean + b * noise)
    assert torch.equal(returned_noise, noise)

    posterior_mean, posterior_var, _ = schedule.q_posterior(
        clean[:1], noisy[:1], torch.zeros(1, dtype=torch.long)
    )
    assert torch.allclose(posterior_mean, clean[:1], atol=2e-4)
    assert torch.equal(posterior_var, torch.zeros_like(posterior_var))


def test_interpolation_and_extrapolation_masks():
    observed = torch.ones(2, 6, dtype=torch.bool)
    target = torch.tensor([2, 4])
    interpolation = make_interpolation_mask(
        observed,
        target,
        difficulty=4,
        generator=torch.Generator().manual_seed(7),
    )
    assert not interpolation[0, 2]
    assert not interpolation[1, 4]
    assert torch.equal(interpolation[:, 0], torch.ones(2, dtype=torch.bool))
    assert torch.equal(interpolation[:, 5], torch.ones(2, dtype=torch.bool))
    assert torch.equal(interpolation[:, 1:5].sum(1), torch.zeros(2, dtype=torch.long))

    extrapolation = make_extrapolation_mask(observed[:1], torch.tensor([5]))
    assert extrapolation[0, :5].all()
    assert not extrapolation[0, 5]


def test_reconstruction_batch_never_exposes_target():
    features = torch.randn(3, 6, 8)
    observed = torch.ones(3, 6, dtype=torch.bool)
    batch = make_reconstruction_batch(
        features, observed, "extrapolation", difficulty=5, target_index=5
    )
    rows = torch.arange(3)
    assert torch.equal(batch.clean_target, features[:, 5])
    assert not batch.condition_observed[rows, batch.target_index].any()
    assert torch.equal(batch.task_id, torch.ones(3, dtype=torch.long))


def test_denoiser_and_loss_shapes():
    model = small_diffusion()
    features = torch.randn(4, 6, 8)
    observed = torch.ones(4, 6, dtype=torch.bool)
    loss = model.reconstruction_loss(
        features,
        observed,
        "interpolation",
        difficulty=2,
        target_index=torch.tensor([1, 2, 3, 4]),
    )
    assert loss.loss.ndim == 0
    assert loss.per_sample.shape == (4,)
    assert loss.predicted_noise.shape == (4, 8)
    assert torch.isfinite(loss.loss)

    with pytest.raises(ValueError, match="leaked"):
        model.denoiser(
            torch.randn(4, 8),
            torch.zeros(4, dtype=torch.long),
            features,
            observed,
            torch.tensor([1, 2, 3, 4]),
            torch.zeros(4, dtype=torch.long),
        )


def test_candidate_sampling_and_classifier_shapes():
    model = small_diffusion(steps=2).eval()
    context = torch.zeros(2, 6, 8)
    observed = torch.zeros(2, 6, dtype=torch.bool)
    observed[:, 0] = True
    candidates = model.sample_candidates(
        context,
        observed,
        target_index=1,
        num_candidates=3,
        generator=torch.Generator().manual_seed(11),
    )
    assert candidates.shape == (2, 3, 8)
    assert np.isfinite(candidates.detach().numpy()).all()

    classifier = TrajectoryClassifier(
        feature_dim=8,
        model_dim=16,
        num_layers=1,
        num_heads=4,
        feedforward_dim=32,
        dropout=0.0,
    )
    logits = classifier(torch.randn(2, 6, 8))
    assert logits.shape == (2, 2)
