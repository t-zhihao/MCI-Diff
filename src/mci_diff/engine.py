"""Training loops, progressive augmentation and evaluation for MCI-Diff."""

from __future__ import annotations

import csv
import inspect
import json
import logging
import math
import os
import random
import tempfile
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset


LOGGER = logging.getLogger("mci_diff")


# Small I/O helpers. Stage markers only list outputs that must still exist.


def _json_safe(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _json_safe(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set)):
        return [_json_safe(item) for item in value]
    return value


def atomic_json_dump(payload: Any, path: Union[str, Path]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix="." + destination.name + ".", suffix=".tmp", dir=str(destination.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(
                _json_safe(payload),
                stream,
                indent=2,
                ensure_ascii=False,
                sort_keys=True,
                allow_nan=False,
            )
            stream.write("\n")
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return destination


def atomic_torch_save(payload: Any, path: Union[str, Path]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix="." + destination.name + ".", suffix=".tmp", dir=str(destination.parent)
    )
    os.close(descriptor)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return destination


def torch_load(
    path: Union[str, Path], map_location: Union[str, torch.device] = "cpu"
) -> Any:
    try:
        return torch.load(Path(path), map_location=map_location, weights_only=False)
    except TypeError:  # PyTorch before weights_only was added.
        return torch.load(Path(path), map_location=map_location)


def stage_marker_path(directory: Union[str, Path], stage: str) -> Path:
    if not stage or any(part in stage for part in ("/", "\\", "..")):
        raise ValueError("stage must be a plain non-empty name")
    return Path(directory) / "stages" / (stage + ".complete.json")


def mark_stage_complete(
    directory: Union[str, Path],
    stage: str,
    outputs: Sequence[Union[str, Path]],
    metadata: Optional[Mapping[str, Any]] = None,
) -> Path:
    paths = [Path(item).expanduser().resolve() for item in outputs]
    if not paths:
        raise ValueError("a completed stage needs at least one output")
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("cannot complete %s; missing %s" % (stage, missing))
    return atomic_json_dump(
        {
            "schema_version": 1,
            "stage": stage,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "outputs": [str(path) for path in paths],
            "metadata": dict(metadata or {}),
        },
        stage_marker_path(directory, stage),
    )


def stage_is_complete(
    directory: Union[str, Path],
    stage: str,
    required_outputs: Optional[Sequence[Union[str, Path]]] = None,
) -> bool:
    marker = stage_marker_path(directory, stage)
    if not marker.is_file():
        return False
    try:
        with marker.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
    except (OSError, ValueError, TypeError):
        return False
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        return False
    if payload.get("stage") != stage:
        return False
    declared = payload.get("outputs")
    if not isinstance(declared, list) or not declared:
        return False
    declared_paths = [Path(item).expanduser().resolve() for item in declared if isinstance(item, str)]
    if len(declared_paths) != len(declared) or not all(path.exists() for path in declared_paths):
        return False
    if required_outputs is None:
        return True
    required = [Path(item).expanduser().resolve() for item in required_outputs]
    return bool(required) and set(required).issubset(set(declared_paths)) and all(
        path.exists() for path in required
    )


def remove_stage_marker(directory: Union[str, Path], stage: str) -> bool:
    path = stage_marker_path(directory, stage)
    if not path.exists():
        return False
    path.unlink()
    return True


# Runtime and checkpoints


def resolve_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def amp_enabled(precision: str, device: torch.device) -> bool:
    if precision not in ("fp32", "fp16", "bf16", "amp"):
        raise ValueError("precision must be fp32, fp16, bf16 or amp")
    return precision != "fp32" and device.type == "cuda"


def seed_everything(seed: int, deterministic: bool = True) -> None:
    os.environ["PYTHONHASHSEED"] = str(int(seed))
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = not bool(deterministic)
    if hasattr(torch, "use_deterministic_algorithms"):
        try:
            torch.use_deterministic_algorithms(bool(deterministic), warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(bool(deterministic))


def seed_worker(worker_id: int) -> None:
    del worker_id
    value = torch.initial_seed() % (2**32)
    random.seed(value)
    np.random.seed(value)


def _generator_on_device(
    generator: Optional[torch.Generator], device: torch.device
) -> Optional[torch.Generator]:
    if generator is None:
        return None
    generator_device = torch.device(getattr(generator, "device", "cpu"))
    if generator_device.type == device.type:
        return generator
    converted = torch.Generator(device=device)
    converted.manual_seed(generator.initial_seed())
    return converted


def move_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, Mapping):
        return {key: move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    return value


def _cfg(config: Any, path: str, default: Any = None) -> Any:
    value = config
    for name in path.split("."):
        if isinstance(value, Mapping):
            if name not in value:
                return default
            value = value[name]
        elif hasattr(value, name):
            value = getattr(value, name)
        else:
            return default
    return value


def _batch_value(batch: Any, names: Sequence[str], default: Any = None) -> Any:
    for name in names:
        if isinstance(batch, Mapping) and name in batch:
            return batch[name]
        if hasattr(batch, name):
            return getattr(batch, name)
    return default


def _sequence_batch(batch: Any) -> Tuple[torch.Tensor, torch.Tensor]:
    if isinstance(batch, (tuple, list)) and len(batch) >= 2:
        features, observed = batch[0], batch[1]
    else:
        features = _batch_value(batch, ("features", "sequence", "trajectory"))
        observed = _batch_value(
            batch, ("available_mask", "observed", "observed_mask", "visit_mask", "mask")
        )
    if not isinstance(features, torch.Tensor) or not isinstance(observed, torch.Tensor):
        raise TypeError("batch must provide tensor features and observed mask")
    if features.ndim != 3 or observed.shape != features.shape[:2]:
        raise ValueError("expected features [B, visits, D] and observed [B, visits]")
    return features.float(), observed.bool()


def _classifier_batch(
    batch: Any,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[List[str]]]:
    features, observed = _sequence_batch(batch)
    if isinstance(batch, (tuple, list)) and len(batch) >= 3:
        labels = batch[2]
        identifiers = batch[3] if len(batch) >= 4 else None
    else:
        labels = _batch_value(batch, ("label", "labels", "target", "targets"))
        identifiers = _batch_value(
            batch, ("patient_id", "patient_ids", "subject_id", "subject_ids")
        )
    if not isinstance(labels, torch.Tensor):
        labels = torch.as_tensor(labels)
    labels = labels.long().reshape(-1)
    if labels.shape[0] != features.shape[0]:
        raise ValueError("one binary label is required for each sequence")
    if identifiers is None:
        ids = None
    elif isinstance(identifiers, torch.Tensor):
        ids = [str(item) for item in identifiers.detach().cpu().tolist()]
    elif isinstance(identifiers, str):
        ids = [identifiers]
    else:
        ids = [str(item) for item in identifiers]
    return features, observed, labels, ids


def _make_scaler(enabled: bool) -> Any:
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _autocast(enabled: bool, dtype: torch.dtype = torch.float16) -> Any:
    if not enabled:
        return nullcontext()
    try:
        return torch.autocast(device_type="cuda", dtype=dtype)
    except AttributeError:
        return torch.cuda.amp.autocast(enabled=True)


def build_epoch_scheduler(
    optimizer: torch.optim.Optimizer, total_epochs: int, warmup_epochs: int = 0
) -> torch.optim.lr_scheduler.LambdaLR:
    if total_epochs < 1 or not 0 <= warmup_epochs < total_epochs:
        raise ValueError("warmup_epochs must lie in [0, total_epochs)")

    def scale(epoch: int) -> float:
        if warmup_epochs and epoch < warmup_epochs:
            return max(1.0e-8, float(epoch + 1) / warmup_epochs)
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)


class ExponentialMovingAverage:
    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        if not 0.0 <= decay < 1.0:
            raise ValueError("EMA decay must lie in [0, 1)")
        self.decay = float(decay)
        self.average = {
            name: value.detach().clone()
            for name, value in model.named_parameters()
            if value.requires_grad
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for name, value in model.named_parameters():
            if name not in self.average:
                continue
            current = self.average[name]
            if current.device != value.device or current.dtype != value.dtype:
                current = current.to(value)
                self.average[name] = current
            current.lerp_(value.detach(), 1.0 - self.decay)

    def state_dict(self) -> Dict[str, Any]:
        return {"decay": self.decay, "average": self.average}

    def load_state_dict(self, state: Mapping[str, Any], model: Optional[nn.Module] = None) -> None:
        self.decay = float(state["decay"])
        parameters = dict(model.named_parameters()) if model is not None else {}
        raw = state.get("average")
        if not isinstance(raw, Mapping):
            raise ValueError("EMA checkpoint is missing averaged parameters")
        self.average = {}
        for name, value in raw.items():
            if not isinstance(value, torch.Tensor):
                raise TypeError("EMA parameters must be tensors")
            target = parameters.get(str(name))
            self.average[str(name)] = value.detach().clone().to(target) if target is not None else value

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        for name, value in model.named_parameters():
            if name in self.average:
                value.copy_(self.average[name].to(value))

    @contextmanager
    def average_parameters(self, model: nn.Module) -> Iterator[None]:
        original: Dict[str, torch.Tensor] = {}
        with torch.no_grad():
            for name, value in model.named_parameters():
                if name in self.average:
                    original[name] = value.detach().clone()
                    value.copy_(self.average[name].to(value))
        try:
            yield
        finally:
            with torch.no_grad():
                for name, value in model.named_parameters():
                    if name in original:
                        value.copy_(original[name])


@dataclass
class EarlyStopping:
    patience: int
    mode: str = "min"
    minimum_delta: float = 0.0
    best: Optional[float] = None
    bad_epochs: int = 0

    def __post_init__(self) -> None:
        if self.patience < 1:
            raise ValueError("patience must be positive")
        if self.mode not in ("min", "max"):
            raise ValueError("mode must be min or max")

    def update(self, value: float) -> Tuple[bool, bool]:
        if not math.isfinite(value):
            self.bad_epochs += 1
            return False, self.bad_epochs >= self.patience
        if self.best is None:
            improved = True
        elif self.mode == "min":
            improved = value < self.best - self.minimum_delta
        else:
            improved = value > self.best + self.minimum_delta
        if improved:
            self.best = float(value)
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        return improved, self.bad_epochs >= self.patience

    def state_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.best = None if state.get("best") is None else float(state["best"])
        self.bad_epochs = int(state.get("bad_epochs", 0))


def rng_state() -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Mapping[str, Any]) -> None:
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch" in state:
        torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


@dataclass(frozen=True)
class ResumeState:
    epoch: int = -1
    global_step: int = 0
    best_metric: Optional[float] = None
    phase: Optional[str] = None
    extra: Optional[Dict[str, Any]] = None


def save_checkpoint(
    path: Union[str, Path],
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[Any] = None,
    ema: Optional[ExponentialMovingAverage] = None,
    early_stopping: Optional[EarlyStopping] = None,
    epoch: int = -1,
    global_step: int = 0,
    best_metric: Optional[float] = None,
    phase: Optional[str] = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> Path:
    payload: Dict[str, Any] = {
        "model": model.state_dict(),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "best_metric": best_metric,
        "phase": phase,
        "rng_state": rng_state(),
        "extra": dict(extra or {}),
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    if scaler is not None:
        payload["scaler"] = scaler.state_dict()
    if ema is not None:
        payload["ema"] = ema.state_dict()
    if early_stopping is not None:
        payload["early_stopping"] = early_stopping.state_dict()
    return atomic_torch_save(payload, path)


def restore_training_checkpoint(
    path: Union[str, Path],
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[Any] = None,
    ema: Optional[ExponentialMovingAverage] = None,
    early_stopping: Optional[EarlyStopping] = None,
    device: Union[str, torch.device] = "cpu",
    restore_rng: bool = True,
) -> ResumeState:
    payload = torch_load(path, map_location=device)
    if not isinstance(payload, Mapping) or "model" not in payload:
        raise ValueError("not an MCI-Diff training checkpoint")
    model.load_state_dict(payload["model"])
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and "scheduler" in payload:
        scheduler.load_state_dict(payload["scheduler"])
    if scaler is not None and "scaler" in payload:
        scaler.load_state_dict(payload["scaler"])
    if ema is not None and "ema" in payload:
        ema.load_state_dict(payload["ema"], model)
    if early_stopping is not None and "early_stopping" in payload:
        early_stopping.load_state_dict(payload["early_stopping"])
    if restore_rng and isinstance(payload.get("rng_state"), Mapping):
        restore_rng_state(payload["rng_state"])
    extra = payload.get("extra")
    return ResumeState(
        epoch=int(payload.get("epoch", -1)),
        global_step=int(payload.get("global_step", 0)),
        best_metric=payload.get("best_metric"),
        phase=payload.get("phase"),
        extra=dict(extra) if isinstance(extra, Mapping) else None,
    )


def load_model_checkpoint(
    path: Union[str, Path],
    model: nn.Module,
    device: Union[str, torch.device] = "cpu",
    use_ema: bool = False,
) -> ResumeState:
    payload = torch_load(path, map_location=device)
    if isinstance(payload, Mapping) and "model" in payload:
        state = payload["model"]
        if use_ema and isinstance(payload.get("ema"), Mapping):
            state = payload["ema"].get("average", state)
        model.load_state_dict(state)
        return ResumeState(
            epoch=int(payload.get("epoch", -1)),
            global_step=int(payload.get("global_step", 0)),
            best_metric=payload.get("best_metric"),
            phase=payload.get("phase"),
        )
    if not isinstance(payload, Mapping):
        raise ValueError("checkpoint does not contain a state dict")
    model.load_state_dict(payload)
    return ResumeState()


# Diffusion training


@dataclass(frozen=True)
class PhaseResult:
    phase: str
    epochs_ran: int
    best_validation_loss: float
    best_checkpoint: str
    last_checkpoint: str
    history: List[Dict[str, float]]


class DiffusionTrainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: Union[str, torch.device] = "auto",
        precision: str = "amp",
        ema_decay: float = 0.999,
        gradient_clip_norm: Optional[float] = 1.0,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.device = resolve_device(device) if isinstance(device, str) else device
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.use_amp = amp_enabled(precision, self.device)
        self.amp_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
        self.scaler = _make_scaler(self.use_amp)
        self.ema = ExponentialMovingAverage(self.model, ema_decay)
        self.gradient_clip_norm = gradient_clip_norm
        self.logger = logger or LOGGER
        self.global_step = 0
        self.base_lrs = [float(group["lr"]) for group in optimizer.param_groups]

    def _epoch(
        self,
        loader: Iterable[Any],
        task: str,
        difficulty: int,
        training: bool,
    ) -> float:
        self.model.train(training)
        total = 0.0
        count = 0
        context = nullcontext() if training else self.ema.average_parameters(self.model)
        with context:
            for raw_batch in loader:
                batch = move_to_device(raw_batch, self.device)
                features, observed = _sequence_batch(batch)
                batch_size = int(features.shape[0])
                if training:
                    self.optimizer.zero_grad(set_to_none=True)
                with _autocast(self.use_amp, self.amp_dtype):
                    conditioned = (
                        isinstance(batch, Mapping)
                        and hasattr(self.model, "loss_from_condition")
                        and "input_features" in batch
                        and "input_mask" in batch
                        and "target_indices" in batch
                    )
                    if conditioned:
                        raw_indices = batch["target_indices"]
                        if not isinstance(raw_indices, (list, tuple)):
                            raw_indices = [item.reshape(-1) for item in raw_indices]
                        chosen = []
                        for item in raw_indices:
                            values = torch.as_tensor(item, device=self.device).long().reshape(-1)
                            if values.numel() == 0:
                                raise ValueError("diffusion task item has no target visit")
                            pick = int(torch.randint(0, values.numel(), (1,), device=self.device))
                            chosen.append(values[pick])
                        targets = torch.stack(chosen)
                        rows = torch.arange(features.shape[0], device=self.device)
                        task_ids = batch.get("task_id")
                        if task_ids is None:
                            task_ids = torch.full(
                                (features.shape[0],),
                                0 if task == "interpolation" else 1,
                                dtype=torch.long,
                                device=self.device,
                            )
                        result = self.model.loss_from_condition(
                            clean_target=features[rows, targets],
                            condition_features=batch["input_features"].float(),
                            condition_observed=batch["input_mask"].bool(),
                            target_index=targets,
                            task_id=task_ids.long(),
                            reduction="mean",
                        )
                    else:
                        result = self.model.reconstruction_loss(
                            features,
                            observed,
                            task=task,
                            difficulty=int(difficulty),
                            reduction="mean",
                        )
                    loss = result.loss if hasattr(result, "loss") else result
                if not isinstance(loss, torch.Tensor) or loss.ndim != 0:
                    raise ValueError("reconstruction_loss must return a scalar loss")
                if not torch.isfinite(loss):
                    raise FloatingPointError("non-finite diffusion loss")
                if training:
                    self.scaler.scale(loss).backward()
                    if self.gradient_clip_norm is not None:
                        self.scaler.unscale_(self.optimizer)
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.ema.update(self.model)
                    self.global_step += 1
                total += float(loss.detach()) * batch_size
                count += batch_size
        if not count:
            raise ValueError("empty diffusion data loader")
        return total / count

    def evaluate(self, loader: Iterable[Any], task: str, difficulty: int) -> float:
        with torch.no_grad():
            return self._epoch(loader, task, difficulty, training=False)

    def fit_phase(
        self,
        train_loader: Iterable[Any],
        validation_loader: Iterable[Any],
        task: str,
        difficulty: int,
        epochs: int,
        patience: int,
        checkpoint_dir: Union[str, Path],
        phase_name: Optional[str] = None,
        warmup_epochs: int = 0,
        checkpoint_every: int = 10,
        resume: bool = True,
    ) -> PhaseResult:
        if epochs < 1:
            raise ValueError("epochs must be positive")
        phase = phase_name or "%s_d%d" % (task, difficulty)
        directory = Path(checkpoint_dir)
        directory.mkdir(parents=True, exist_ok=True)
        best_path = directory / "best.pt"
        last_path = directory / "last.pt"
        for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            group["lr"] = base_lr
        scheduler = build_epoch_scheduler(self.optimizer, epochs, warmup_epochs)
        stopping = EarlyStopping(patience=patience, mode="min")
        start_epoch = 0
        history: List[Dict[str, float]] = []
        if resume and last_path.exists():
            state = restore_training_checkpoint(
                last_path,
                self.model,
                self.optimizer,
                scheduler,
                self.scaler,
                self.ema,
                stopping,
                device=self.device,
            )
            if state.phase not in (None, phase):
                raise ValueError("checkpoint phase does not match %s" % phase)
            self.global_step = state.global_step
            start_epoch = state.epoch + 1
            if state.extra and isinstance(state.extra.get("history"), list):
                history = list(state.extra["history"])

        for epoch in range(start_epoch, epochs):
            train_loss = self._epoch(train_loader, task, difficulty, training=True)
            validation_loss = self.evaluate(validation_loader, task, difficulty)
            scheduler.step()
            row = {
                "epoch": float(epoch),
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "learning_rate": float(self.optimizer.param_groups[0]["lr"]),
            }
            history.append(row)
            improved, stop = stopping.update(validation_loss)
            if improved:
                save_checkpoint(
                    best_path,
                    self.model,
                    self.optimizer,
                    scheduler,
                    self.scaler,
                    self.ema,
                    stopping,
                    epoch,
                    self.global_step,
                    stopping.best,
                    phase,
                    {"history": history},
                )
            if (epoch + 1) % max(1, checkpoint_every) == 0 or stop or epoch + 1 == epochs:
                save_checkpoint(
                    last_path,
                    self.model,
                    self.optimizer,
                    scheduler,
                    self.scaler,
                    self.ema,
                    stopping,
                    epoch,
                    self.global_step,
                    stopping.best,
                    phase,
                    {"history": history},
                )
            self.logger.info(
                "%s epoch %d/%d train %.5f val %.5f",
                phase,
                epoch + 1,
                epochs,
                train_loss,
                validation_loss,
            )
            if stop:
                break
        if best_path.exists():
            best_state = restore_training_checkpoint(
                best_path,
                self.model,
                self.optimizer,
                scaler=self.scaler,
                ema=self.ema,
                early_stopping=stopping,
                device=self.device,
                restore_rng=False,
            )
            self.global_step = best_state.global_step
        best_value = float(stopping.best) if stopping.best is not None else float("nan")
        return PhaseResult(
            phase=phase,
            epochs_ran=len(history),
            best_validation_loss=best_value,
            best_checkpoint=str(best_path),
            last_checkpoint=str(last_path),
            history=history,
        )


@torch.no_grad()
def standard_ddpm_impute(
    model: nn.Module,
    condition_features: torch.Tensor,
    condition_observed: torch.Tensor,
    target_index: Union[int, torch.Tensor],
    task: str = "extrapolation",
    generator: Optional[torch.Generator] = None,
    clip_x0: Optional[float] = None,
) -> torch.Tensor:
    """Run the model's ordinary stochastic DDPM reverse chain."""

    if not hasattr(model, "sample"):
        raise TypeError("diffusion model must provide sample()")
    return model.sample(
        condition_features=condition_features,
        condition_observed=condition_observed,
        target_index=target_index,
        task=task,
        generator=generator,
        clip_x0=clip_x0,
    )


@dataclass
class ProgressiveTrainingPool:
    """Sequences from one training fold. Validation/test subjects never enter here."""

    features: torch.Tensor
    observed: torch.Tensor
    labels: Optional[torch.Tensor] = None
    patient_ids: Optional[List[str]] = None

    def __post_init__(self) -> None:
        if self.features.ndim != 3 or self.observed.shape != self.features.shape[:2]:
            raise ValueError("pool expects features [N, visits, D] and observed [N, visits]")
        if self.labels is not None and len(self.labels) != len(self.features):
            raise ValueError("labels and features have different lengths")
        if self.patient_ids is not None and len(self.patient_ids) != len(self.features):
            raise ValueError("patient_ids and features have different lengths")

    def clone(self) -> "ProgressiveTrainingPool":
        return ProgressiveTrainingPool(
            self.features.detach().clone(),
            self.observed.detach().clone().bool(),
            None if self.labels is None else self.labels.detach().clone(),
            None if self.patient_ids is None else list(self.patient_ids),
        )

    def state_dict(self) -> Dict[str, Any]:
        return {
            "features": self.features.detach().cpu(),
            "observed": self.observed.detach().cpu(),
            "labels": None if self.labels is None else self.labels.detach().cpu(),
            "patient_ids": self.patient_ids,
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "ProgressiveTrainingPool":
        return cls(
            features=state["features"],
            observed=state["observed"].bool(),
            labels=state.get("labels"),
            patient_ids=state.get("patient_ids"),
        )


def pool_from_records(records: Sequence[Any]) -> ProgressiveTrainingPool:
    if not records:
        raise ValueError("record collection is empty")
    return ProgressiveTrainingPool(
        features=torch.stack(
            [torch.as_tensor(record.features).float() for record in records]
        ),
        observed=torch.stack(
            [
                torch.as_tensor(
                    getattr(record, "available", getattr(record, "observed", None))
                ).bool()
                for record in records
            ]
        ),
        labels=torch.tensor([int(record.label) for record in records], dtype=torch.long),
        patient_ids=[str(record.subject_id) for record in records],
    )


class _PoolDataset(Dataset):
    def __init__(self, pool: ProgressiveTrainingPool) -> None:
        self.pool = pool

    def __len__(self) -> int:
        return int(self.pool.features.shape[0])

    def __getitem__(self, index: int) -> Dict[str, Any]:
        item: Dict[str, Any] = {
            "features": self.pool.features[index],
            "observed": self.pool.observed[index],
        }
        if self.pool.labels is not None:
            item["label"] = self.pool.labels[index]
        if self.pool.patient_ids is not None:
            item["patient_id"] = self.pool.patient_ids[index]
        return item


class _PoolTaskDataset(Dataset):
    def __init__(
        self,
        pool: ProgressiveTrainingPool,
        task: str,
        difficulty: int,
        expand_targets: bool = False,
    ) -> None:
        from mci_diff.data import make_extrapolation_masks, make_interpolation_masks

        self.pool = pool
        self.task = task
        self.items: List[Tuple[int, Any, Optional[int]]] = []
        builder = (
            make_interpolation_masks if task == "interpolation" else make_extrapolation_masks
        )
        for row in range(len(pool.features)):
            masks = builder(pool.observed[row].cpu().numpy(), max_difficulty=difficulty)
            for mask in masks:
                if int(mask.difficulty) != int(difficulty):
                    continue
                if expand_targets:
                    self.items.extend((row, mask, target) for target in mask.target_indices)
                else:
                    self.items.append((row, mask, None))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row, mask, fixed_target = self.items[index]
        features = self.pool.features[row]
        input_features = torch.zeros_like(features)
        input_mask = torch.from_numpy(mask.input_mask.copy()).bool()
        input_features[input_mask] = features[input_mask]
        item: Dict[str, Any] = {
            "features": features,
            "available_mask": self.pool.observed[row],
            "input_features": input_features,
            "input_mask": input_mask,
            "target_indices": torch.tensor(
                mask.target_indices if fixed_target is None else (fixed_target,),
                dtype=torch.long,
            ),
            "task_id": torch.tensor(0 if self.task == "interpolation" else 1),
        }
        if self.pool.labels is not None:
            item["label"] = self.pool.labels[row]
        if self.pool.patient_ids is not None:
            item["subject_id"] = self.pool.patient_ids[row]
        return item


def _default_pool_loader(
    pool: ProgressiveTrainingPool,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    task: Optional[str] = None,
    difficulty: Optional[int] = None,
) -> DataLoader:
    from mci_diff.data import longitudinal_collate

    dataset: Dataset = (
        _PoolTaskDataset(pool, task, int(difficulty), expand_targets=not shuffle)
        if task is not None and difficulty is not None
        else _PoolDataset(pool)
    )
    if len(dataset) == 0:
        raise ValueError("no %s examples are available at difficulty %s" % (task, difficulty))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker if num_workers else None,
        collate_fn=longitudinal_collate if isinstance(dataset, _PoolTaskDataset) else None,
    )


def _call_factory(factory: Callable[..., Any], **values: Any) -> Any:
    signature = inspect.signature(factory)
    if any(item.kind == inspect.Parameter.VAR_KEYWORD for item in signature.parameters.values()):
        return factory(**values)
    usable = {name: value for name, value in values.items() if name in signature.parameters}
    return factory(**usable)


@dataclass(frozen=True)
class AugmentationRecord:
    row: int
    target_index: int
    task: str
    difficulty: int


def _eligible_augmentation_rows(
    observed: torch.Tensor, task: str, difficulty: int
) -> torch.Tensor:
    """Rows assigned to the current missingness difficulty."""

    task_name = task.lower()
    if task_name in ("interpolation", "int"):
        missing_middle = (~observed[:, 1:-1]).sum(dim=1)
        eligible = missing_middle == int(difficulty)
        # Interpolation requires real/imputed context on both sides of every gap.
        for row in torch.nonzero(eligible, as_tuple=False).flatten().tolist():
            gaps = torch.nonzero(~observed[row, 1:-1], as_tuple=False).flatten() + 1
            if gaps.numel() and not (
                bool(observed[row, : int(gaps[0])].any())
                and bool(observed[row, int(gaps[-1]) + 1 :].any())
            ):
                eligible[row] = False
        return eligible
    if task_name in ("extrapolation", "ext"):
        eligible = torch.zeros(observed.shape[0], dtype=torch.bool, device=observed.device)
        for row in range(observed.shape[0]):
            known = torch.nonzero(observed[row], as_tuple=False).flatten()
            if known.numel() == 0:
                continue
            last = int(known[-1])
            trailing = observed.shape[1] - last - 1
            # Extrapolation rows have a complete prefix and a missing suffix of size d.
            eligible[row] = trailing == int(difficulty) and bool(observed[row, : last + 1].all())
        return eligible
    raise ValueError("task must be interpolation or extrapolation")


@torch.no_grad()
def augment_training_pool(
    model: nn.Module,
    pool: ProgressiveTrainingPool,
    task: str,
    difficulty: int,
    device: Union[str, torch.device] = "auto",
    mode: str = "next_only",
    max_augmented_future_points: int = 1,
    generator: Optional[torch.Generator] = None,
    clip_x0: Optional[float] = None,
) -> Tuple[ProgressiveTrainingPool, List[AugmentationRecord]]:
    """Fill eligible entries in a training-fold copy with DDPM samples."""

    target_device = resolve_device(device) if isinstance(device, str) else device
    generator = _generator_on_device(generator, target_device)
    result = pool.clone()
    features = result.features.to(target_device)
    observed = result.observed.to(target_device).bool()
    eligible_rows = _eligible_augmentation_rows(observed, task, difficulty)
    records: List[AugmentationRecord] = []
    task_name = task.lower()
    if mode not in ("next_only", "iterative_d"):
        raise ValueError("augmentation mode must be next_only or iterative_d")
    if difficulty < 1:
        raise ValueError("difficulty must be positive")

    if task_name in ("interpolation", "int"):
        # Only genuine gaps with observations on both sides are interpolated.
        for target in range(1, features.shape[1] - 1):
            left = observed[:, :target].any(dim=1)
            right = observed[:, target + 1 :].any(dim=1)
            rows = torch.nonzero(
                eligible_rows & (~observed[:, target]) & left & right,
                as_tuple=False,
            ).flatten()
            if rows.numel() == 0:
                continue
            sampled = standard_ddpm_impute(
                model,
                features[rows],
                observed[rows],
                target,
                task="interpolation",
                generator=generator,
                clip_x0=clip_x0,
            )
            features[rows, target] = sampled
            observed[rows, target] = True
            records.extend(
                AugmentationRecord(int(row), target, "interpolation", difficulty)
                for row in rows.detach().cpu().tolist()
            )
    elif task_name in ("extrapolation", "ext"):
        steps = 1 if mode == "next_only" else int(difficulty)
        steps = min(steps, max(1, int(max_augmented_future_points)))
        for _ in range(steps):
            targets = torch.full((features.shape[0],), -1, dtype=torch.long, device=target_device)
            for row in range(features.shape[0]):
                if not bool(eligible_rows[row]):
                    continue
                known = torch.nonzero(observed[row], as_tuple=False).flatten()
                if known.numel() and int(known[-1]) + 1 < features.shape[1]:
                    targets[row] = int(known[-1]) + 1
            for target in range(1, features.shape[1]):
                rows = torch.nonzero(targets == target, as_tuple=False).flatten()
                if rows.numel() == 0:
                    continue
                sampled = standard_ddpm_impute(
                    model,
                    features[rows],
                    observed[rows],
                    target,
                    task="extrapolation",
                    generator=generator,
                    clip_x0=clip_x0,
                )
                features[rows, target] = sampled
                observed[rows, target] = True
                records.extend(
                    AugmentationRecord(int(row), target, "extrapolation", difficulty)
                    for row in rows.detach().cpu().tolist()
                )
    else:
        raise ValueError("task must be interpolation or extrapolation")

    result.features = features.cpu()
    result.observed = observed.cpu()
    return result, records


def _pool_rows(pool: ProgressiveTrainingPool, rows: Sequence[int]) -> ProgressiveTrainingPool:
    index = torch.as_tensor(list(rows), dtype=torch.long)
    return ProgressiveTrainingPool(
        features=pool.features[index],
        observed=pool.observed[index],
        labels=None if pool.labels is None else pool.labels[index],
        patient_ids=(
            None if pool.patient_ids is None else [pool.patient_ids[item] for item in index.tolist()]
        ),
    )


def _merge_pool_rows(
    active: ProgressiveTrainingPool,
    source: ProgressiveTrainingPool,
    rows: Sequence[int],
) -> ProgressiveTrainingPool:
    if not rows:
        return active
    if active.patient_ids is None or source.patient_ids is None:
        raise ValueError("patient_ids are required when progressive rows are merged")
    features = [item.clone() for item in active.features]
    observed = [item.clone() for item in active.observed]
    labels = None if active.labels is None else [item.clone() for item in active.labels]
    identifiers = list(active.patient_ids)
    positions = {identifier: index for index, identifier in enumerate(identifiers)}
    for row in rows:
        identifier = source.patient_ids[row]
        if identifier in positions:
            target = positions[identifier]
            features[target] = source.features[row].clone()
            observed[target] = source.observed[row].clone()
            if labels is not None and source.labels is not None:
                labels[target] = source.labels[row].clone()
        else:
            positions[identifier] = len(identifiers)
            identifiers.append(identifier)
            features.append(source.features[row].clone())
            observed.append(source.observed[row].clone())
            if labels is not None and source.labels is not None:
                labels.append(source.labels[row].clone())
    return ProgressiveTrainingPool(
        torch.stack(features),
        torch.stack(observed),
        None if labels is None else torch.stack(labels),
        identifiers,
    )


@dataclass(frozen=True)
class ProgressiveResult:
    phases: List[Dict[str, Any]]
    final_pool_path: str
    generated_points: int
    active_size: int
    source_size: int


def _run_progressive_training_core(
    trainer: DiffusionTrainer,
    training_pool: ProgressiveTrainingPool,
    validation: Union[ProgressiveTrainingPool, Iterable[Any]],
    config: Any,
    output_dir: Union[str, Path],
    train_loader_factory: Optional[Callable[..., Iterable[Any]]] = None,
    validation_loader_factory: Optional[Callable[..., Iterable[Any]]] = None,
    resume: bool = True,
    generator: Optional[torch.Generator] = None,
) -> ProgressiveResult:
    """Algorithm 1: interpolation then extrapolation for d=1,...,D_max."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    generator = _generator_on_device(generator, trainer.device)
    source_pool = training_pool.clone()
    if source_pool.patient_ids is None:
        source_pool.patient_ids = [
            "train_%06d" % index for index in range(len(source_pool.features))
        ]
    complete_rows = torch.nonzero(
        source_pool.observed.bool().all(dim=1), as_tuple=False
    ).flatten().tolist()
    if not complete_rows:
        raise ValueError(
            "progressive training needs at least one complete training-fold sequence"
        )
    pool = _pool_rows(source_pool, complete_rows)
    maximum = int(_cfg(config, "progressive.max_difficulty", 4))
    epochs = int(_cfg(config, "training.diffusion.epochs_per_phase", 60))
    patience = int(_cfg(config, "training.diffusion.patience", 15))
    batch_size = int(_cfg(config, "training.diffusion.batch_size", 64))
    warmup = int(_cfg(config, "training.diffusion.warmup_epochs", 5))
    checkpoint_every = int(_cfg(config, "training.checkpoint_every", 10))
    num_workers = int(_cfg(config, "training.num_workers", 0))
    pin_memory = bool(_cfg(config, "training.pin_memory", True))
    mode = str(_cfg(config, "progressive.augmentation_mode", "next_only"))
    max_future = int(_cfg(config, "progressive.max_augmented_future_points", 1))
    clip_x0 = _cfg(config, "diffusion.clip_x0", None)

    flags = {
        "interpolation_task": bool(_cfg(config, "progressive.interpolation_task", True))
        and bool(_cfg(config, "ablation.interpolation_task", True)),
        "interpolation_augmentation": bool(
            _cfg(config, "progressive.interpolation_augmentation", True)
        )
        and bool(_cfg(config, "ablation.interpolation_augmentation", True)),
        "extrapolation_task": bool(_cfg(config, "progressive.extrapolation_task", True))
        and bool(_cfg(config, "ablation.extrapolation_task", True)),
        "extrapolation_augmentation": bool(
            _cfg(config, "progressive.extrapolation_augmentation", True)
        )
        and bool(_cfg(config, "ablation.extrapolation_augmentation", True)),
    }
    if not flags["interpolation_task"] and not flags["extrapolation_task"]:
        raise ValueError("at least one progressive reconstruction task must be enabled")

    phase_rows: List[Dict[str, Any]] = []
    generated_points = 0

    def make_loader(source: Any, task: str, difficulty: int, training: bool) -> Iterable[Any]:
        factory = train_loader_factory if training else validation_loader_factory
        if factory is not None:
            return _call_factory(
                factory,
                pool=source,
                features=getattr(source, "features", None),
                observed=getattr(source, "observed", None),
                labels=getattr(source, "labels", None),
                task=task,
                difficulty=difficulty,
                batch_size=batch_size,
                shuffle=training,
            )
        if isinstance(source, ProgressiveTrainingPool):
            return _default_pool_loader(
                source,
                batch_size,
                training,
                num_workers,
                pin_memory,
                task,
                difficulty,
            )
        return source

    for difficulty in range(1, maximum + 1):
        for task in ("interpolation", "extrapolation"):
            if not flags[task + "_task"]:
                continue
            phase = "%s_d%d" % (task, difficulty)
            checkpoint_dir = output / "checkpoints" / phase
            best_path = checkpoint_dir / "best.pt"
            train_stage = "train_" + phase
            if resume and stage_is_complete(output, train_stage, [best_path]):
                restore_training_checkpoint(
                    best_path,
                    trainer.model,
                    optimizer=trainer.optimizer,
                    scaler=trainer.scaler,
                    ema=trainer.ema,
                    device=trainer.device,
                    restore_rng=False,
                )
                phase_rows.append({"phase": phase, "resumed": True})
            else:
                result = trainer.fit_phase(
                    make_loader(pool, task, difficulty, True),
                    make_loader(validation, task, difficulty, False),
                    task=task,
                    difficulty=difficulty,
                    epochs=epochs,
                    patience=patience,
                    checkpoint_dir=checkpoint_dir,
                    phase_name=phase,
                    warmup_epochs=warmup,
                    checkpoint_every=checkpoint_every,
                    resume=resume,
                )
                phase_rows.append(_json_safe(result))
                mark_stage_complete(output, train_stage, [best_path], {"difficulty": difficulty})

            if not flags[task + "_augmentation"]:
                continue
            pool_path = output / "pools" / (phase + ".pt")
            augment_stage = "augment_" + phase
            if resume and stage_is_complete(output, augment_stage, [pool_path]):
                payload = torch_load(pool_path, "cpu")
                if "active_pool" not in payload or "source_pool" not in payload:
                    raise ValueError("old progressive cache is missing active/source pools")
                pool = ProgressiveTrainingPool.from_state_dict(payload["active_pool"])
                source_pool = ProgressiveTrainingPool.from_state_dict(payload["source_pool"])
                generated_points += int(payload.get("generated_points", 0))
            else:
                with trainer.ema.average_parameters(trainer.model):
                    source_pool, records = augment_training_pool(
                        trainer.model,
                        source_pool,
                        task,
                        difficulty,
                        trainer.device,
                        mode,
                        max_future,
                        generator,
                        clip_x0,
                    )
                changed_rows = sorted({item.row for item in records})
                pool = _merge_pool_rows(pool, source_pool, changed_rows)
                generated_points += len(records)
                atomic_torch_save(
                    {
                        "active_pool": pool.state_dict(),
                        "source_pool": source_pool.state_dict(),
                        "records": [asdict(item) for item in records],
                        "generated_points": len(records),
                    },
                    pool_path,
                )
                mark_stage_complete(
                    output,
                    augment_stage,
                    [pool_path],
                    {"task": task, "difficulty": difficulty, "points": len(records)},
                )

    final_pool = output / "training_pool_final.pt"
    atomic_torch_save(
        {"active_pool": pool.state_dict(), "source_pool": source_pool.state_dict()},
        final_pool,
    )
    atomic_json_dump(phase_rows, output / "progressive_history.json")
    trainer.ema.copy_to(trainer.model)
    return ProgressiveResult(
        phase_rows,
        str(final_pool),
        generated_points,
        len(pool.features),
        len(source_pool.features),
    )


def run_progressive_training(
    model: Union[nn.Module, DiffusionTrainer],
    train_records: Union[Sequence[Any], ProgressiveTrainingPool],
    config: Any,
    output_dir: Union[str, Path],
    device: Optional[Union[str, torch.device]] = None,
    resume: Optional[bool] = None,
    validation_records: Optional[
        Union[Sequence[Any], ProgressiveTrainingPool, Iterable[Any]]
    ] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    train_loader_factory: Optional[Callable[..., Iterable[Any]]] = None,
    validation_loader_factory: Optional[Callable[..., Iterable[Any]]] = None,
    generator: Optional[torch.Generator] = None,
) -> ProgressiveResult:
    """Convenient Algorithm 1 entry point used by the CLI."""

    if output_dir is None:
        raise ValueError("output_dir is required")
    source = (
        train_records
        if isinstance(train_records, ProgressiveTrainingPool)
        else pool_from_records(train_records)
    )
    if validation_records is None:
        complete = torch.nonzero(source.observed.all(dim=1), as_tuple=False).flatten().tolist()
        validation: Union[ProgressiveTrainingPool, Iterable[Any]] = _pool_rows(source, complete)
    elif isinstance(validation_records, ProgressiveTrainingPool):
        validation = validation_records
    elif isinstance(validation_records, Sequence):
        validation = pool_from_records(validation_records)
    else:
        validation = validation_records

    selected_device = device or str(_cfg(config, "experiment.device", "auto"))
    do_resume = bool(_cfg(config, "experiment.resume", True)) if resume is None else resume
    if isinstance(model, DiffusionTrainer):
        trainer = model
    else:
        if optimizer is None:
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=float(_cfg(config, "training.diffusion.learning_rate", 2.0e-4)),
                weight_decay=float(_cfg(config, "training.diffusion.weight_decay", 1.0e-5)),
            )
        trainer = DiffusionTrainer(
            model,
            optimizer,
            selected_device,
            str(_cfg(config, "experiment.precision", "amp")),
            float(_cfg(config, "training.ema_decay", 0.999)),
            _cfg(config, "training.gradient_clip_norm", 1.0),
        )
    return _run_progressive_training_core(
        trainer,
        source,
        validation,
        config,
        output_dir,
        train_loader_factory,
        validation_loader_factory,
        do_resume,
        generator,
    )


# Candidate selection. The scoring model is supplied by the caller.


def _score_number(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float, np.number)) and not isinstance(value, bool):
        score = float(value)
    elif isinstance(value, Mapping):
        if value.get("valid") is False or value.get("parse_ok") is False:
            return None
        score = value.get("score", value.get("value"))
        if score is None:
            return None
        score = float(score)
    else:
        if getattr(value, "valid", True) is False or getattr(value, "parse_ok", True) is False:
            return None
        score = getattr(value, "score", getattr(value, "value", None))
        if score is None:
            return None
        score = float(score)
    return score if math.isfinite(score) else None


def _score_candidates(
    scorer: Any,
    history: torch.Tensor,
    history_observed: torch.Tensor,
    candidates: torch.Tensor,
    target_index: int,
    patient_id: str,
    visit_months: Sequence[int] = (0, 6, 12, 18, 24, 36),
) -> Tuple[int, List[Optional[float]]]:
    if scorer is None:
        return 0, [None] * int(candidates.shape[0])
    method = getattr(scorer, "score_candidates", None)
    if method is not None:
        try:
            raw = _call_factory(
                method,
                baseline_feature=history[0],
                history=history,
                condition_features=history,
                history_observed=history_observed,
                condition_observed=history_observed,
                history_features=history[1:target_index][history_observed[1:target_index]],
                candidates=candidates,
                target_index=target_index,
                target_month=int(visit_months[target_index]),
                patient_id=patient_id,
            )
            if isinstance(raw, tuple) and len(raw) == 2 and isinstance(raw[0], int):
                values = raw[1]
            else:
                values = raw
        except Exception as error:
            LOGGER.warning("candidate scoring failed for %s visit %d: %s", patient_id, target_index, error)
            return 0, [None] * int(candidates.shape[0])
    else:
        single = getattr(scorer, "score", None)
        if single is None:
            raise TypeError("scorer must provide score_candidates() or score()")
        values = []
        for candidate in candidates:
            try:
                values.append(
                    _call_factory(
                        single,
                        history=history,
                        history_observed=history_observed,
                        candidate=candidate,
                        target_index=target_index,
                        patient_id=patient_id,
                    )
                )
            except Exception as error:
                LOGGER.warning("candidate score failed: %s", error)
                values.append(None)
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().tolist()
    values = list(values)
    scores = [_score_number(item) for item in values[: candidates.shape[0]]]
    scores.extend([None] * (int(candidates.shape[0]) - len(scores)))
    best_index = 0
    best_score: Optional[float] = None
    for index, score in enumerate(scores):
        # Strict comparison gives the first candidate on a tie.
        if score is not None and (best_score is None or score > best_score):
            best_index, best_score = index, score
    return best_index, scores


@dataclass
class TrajectoryGenerationResult:
    trajectories: torch.Tensor
    observed: torch.Tensor
    selected_indices: torch.Tensor
    scores: List[List[List[Optional[float]]]]
    patient_ids: List[str]
    cache_path: Optional[str] = None

    @property
    def trajectory_by_subject(self) -> Dict[str, torch.Tensor]:
        return {
            identifier: self.trajectories[index]
            for index, identifier in enumerate(self.patient_ids)
        }

    @property
    def score_log(self) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for subject, visits in enumerate(self.scores):
            for target, scores in enumerate(visits):
                if scores:
                    rows.append(
                        {
                            "subject_id": self.patient_ids[subject],
                            "target_index": target,
                            "selected_index": int(self.selected_indices[subject, target]),
                            "scores": scores,
                        }
                    )
        return rows


@torch.no_grad()
def _generate_trajectory_tensors(
    diffusion: nn.Module,
    features: torch.Tensor,
    observed: torch.Tensor,
    scorer: Any = None,
    patient_ids: Optional[Sequence[str]] = None,
    num_candidates: int = 20,
    candidate_batch_size: Optional[int] = None,
    target_indices: Optional[Sequence[int]] = None,
    device: Union[str, torch.device] = "auto",
    use_guidance: bool = True,
    fallback: str = "first",
    tie_break: str = "first",
    cache_path: Optional[Union[str, Path]] = None,
    resume: bool = True,
    save_all_candidates: bool = False,
    generator: Optional[torch.Generator] = None,
    clip_x0: Optional[float] = None,
    visit_months: Sequence[int] = (0, 6, 12, 18, 24, 36),
) -> TrajectoryGenerationResult:
    """Sample future candidates and keep the highest external score."""

    if fallback != "first" or tie_break != "first":
        raise ValueError("fallback and tie_break must both be first")
    if num_candidates < 1:
        raise ValueError("num_candidates must be positive")
    chunk_size = num_candidates if candidate_batch_size is None else int(candidate_batch_size)
    if chunk_size < 1:
        raise ValueError("candidate_batch_size must be positive")
    if features.ndim != 3 or observed.shape != features.shape[:2]:
        raise ValueError("features and observed have incompatible shapes")
    if len(visit_months) != features.shape[1]:
        raise ValueError("visit_months does not match trajectory length")
    identifiers = (
        [str(item) for item in patient_ids]
        if patient_ids is not None
        else ["subject_%06d" % index for index in range(features.shape[0])]
    )
    if len(identifiers) != len(features):
        raise ValueError("patient_ids has the wrong length")
    cache = Path(cache_path) if cache_path is not None else None
    if cache is not None and resume and cache.exists():
        payload = torch_load(cache, "cpu")
        required = ("trajectories", "observed", "selected_indices", "scores", "patient_ids")
        if isinstance(payload, Mapping) and all(name in payload for name in required):
            cached_features = payload["trajectories"]
            cached_observed = payload["observed"]
            if (
                cached_features.shape == features.shape
                and cached_observed.shape == observed.shape
                and list(payload["patient_ids"]) == identifiers
            ):
                return TrajectoryGenerationResult(
                    cached_features,
                    cached_observed.bool(),
                    payload["selected_indices"],
                    payload["scores"],
                    list(payload["patient_ids"]),
                    str(cache),
                )

    target_device = resolve_device(device) if isinstance(device, str) else device
    generator = _generator_on_device(generator, target_device)
    diffusion = diffusion.to(target_device).eval()
    trajectories = features.detach().clone().to(target_device)
    known = observed.detach().clone().bool().to(target_device)
    if not bool(known[:, 0].all()):
        raise ValueError("future generation requires a baseline visit for every subject")
    targets = list(target_indices) if target_indices is not None else list(range(1, features.shape[1]))
    selected = torch.full(features.shape[:2], -1, dtype=torch.long)
    all_scores: List[List[List[Optional[float]]]] = [
        [[] for _ in range(features.shape[1])] for _ in range(features.shape[0])
    ]
    candidate_cache: Dict[str, torch.Tensor] = {}

    for target in targets:
        if target < 1 or target >= features.shape[1]:
            raise ValueError("future target index is outside the sequence")
        for row in range(features.shape[0]):
            if bool(known[row, target]):
                continue
            chunks = []
            remaining = num_candidates
            while remaining:
                current = min(chunk_size, remaining)
                chunks.append(
                    diffusion.sample_candidates(
                        condition_features=trajectories[row : row + 1],
                        condition_observed=known[row : row + 1],
                        target_index=target,
                        num_candidates=current,
                        task="extrapolation",
                        generator=generator,
                        clip_x0=clip_x0,
                    )[0]
                )
                remaining -= current
            candidates = torch.cat(chunks, dim=0)
            if candidates.shape[0] != num_candidates:
                raise ValueError("diffusion returned the wrong number of candidates")
            if use_guidance:
                chosen, scores = _score_candidates(
                    scorer,
                    trajectories[row].detach().cpu(),
                    known[row].detach().cpu(),
                    candidates.detach().cpu(),
                    target,
                    identifiers[row],
                    visit_months,
                )
            else:
                chosen, scores = 0, [None] * num_candidates
            trajectories[row, target] = candidates[chosen]
            known[row, target] = True
            selected[row, target] = int(chosen)
            all_scores[row][target] = scores
            if save_all_candidates:
                candidate_cache["%d:%d" % (row, target)] = candidates.detach().cpu()

    result = TrajectoryGenerationResult(
        trajectories.detach().cpu(),
        known.detach().cpu(),
        selected,
        all_scores,
        identifiers,
        None if cache is None else str(cache),
    )
    if cache is not None:
        atomic_torch_save(
            {
                "trajectories": result.trajectories,
                "observed": result.observed,
                "selected_indices": result.selected_indices,
                "scores": result.scores,
                "patient_ids": result.patient_ids,
                "all_candidates": candidate_cache if save_all_candidates else None,
            },
            cache,
        )
    return result


def generate_trajectories(
    diffusion: nn.Module,
    records: Union[Sequence[Any], torch.Tensor],
    scorer: Any,
    config: Any = None,
    output_dir: Optional[Union[str, Path]] = None,
    device: Optional[Union[str, torch.device]] = None,
    resume: Optional[bool] = None,
    *,
    observed: Optional[torch.Tensor] = None,
    patient_ids: Optional[Sequence[str]] = None,
    cache_path: Optional[Union[str, Path]] = None,
    target_indices: Optional[Sequence[int]] = None,
    num_candidates: Optional[int] = None,
    candidate_batch_size: Optional[int] = None,
    use_guidance: Optional[bool] = None,
    generator: Optional[torch.Generator] = None,
    fallback: Optional[str] = None,
    tie_break: Optional[str] = None,
    save_all_candidates: Optional[bool] = None,
    clip_x0: Optional[float] = None,
    visit_months: Optional[Sequence[int]] = None,
) -> TrajectoryGenerationResult:
    """Generate longitudinal trajectories from records or raw tensors."""

    if isinstance(records, torch.Tensor):
        features = records
        if observed is None:
            raise ValueError("observed is required when records is a tensor")
        mask = observed
        identifiers = patient_ids
    else:
        if not records:
            raise ValueError("record collection is empty")
        features = torch.stack(
            [torch.as_tensor(record.features).float() for record in records]
        )
        mask = torch.stack(
            [
                torch.as_tensor(
                    getattr(record, "available", getattr(record, "observed", None))
                ).bool()
                for record in records
            ]
        )
        identifiers = [str(record.subject_id) for record in records]
    selected_device = device or str(_cfg(config, "experiment.device", "auto"))
    do_resume = bool(_cfg(config, "experiment.resume", True)) if resume is None else resume
    count = int(_cfg(config, "sampling.num_candidates", 20)) if num_candidates is None else num_candidates
    batch = (
        int(_cfg(config, "sampling.candidate_batch_size", count))
        if candidate_batch_size is None
        else candidate_batch_size
    )
    guidance = scorer is not None if use_guidance is None else use_guidance
    if cache_path is None and output_dir is not None:
        cache_path = Path(output_dir) / "trajectories.pt"
    return _generate_trajectory_tensors(
        diffusion=diffusion,
        features=features,
        observed=mask,
        scorer=scorer,
        patient_ids=identifiers,
        num_candidates=count,
        candidate_batch_size=batch,
        target_indices=target_indices,
        device=selected_device,
        use_guidance=guidance,
        fallback=str(_cfg(config, "sampling.fallback", "first") if fallback is None else fallback),
        tie_break=str(
            _cfg(config, "sampling.tie_break", "first") if tie_break is None else tie_break
        ),
        cache_path=cache_path,
        resume=do_resume,
        save_all_candidates=(
            bool(_cfg(config, "sampling.save_all_candidates", False))
            if save_all_candidates is None
            else save_all_candidates
        ),
        generator=generator,
        clip_x0=_cfg(config, "diffusion.clip_x0", None) if clip_x0 is None else clip_x0,
        visit_months=tuple(
            _cfg(config, "data.visit_months", (0, 6, 12, 18, 24, 36))
            if visit_months is None
            else visit_months
        ),
    )


# Conversion classifier


def balanced_class_weights(labels: Union[torch.Tensor, np.ndarray, Sequence[int]]) -> torch.Tensor:
    values = torch.as_tensor(labels).long().reshape(-1)
    if values.numel() == 0 or bool(((values < 0) | (values > 1)).any()):
        raise ValueError("balanced weights need non-empty binary labels")
    counts = torch.bincount(values, minlength=2).float()
    if bool((counts == 0).any()):
        raise ValueError("both classes are needed for balanced class weights")
    return values.numel() / (2.0 * counts)


@dataclass(frozen=True)
class ClassifierFitResult:
    epochs_ran: int
    best_validation_loss: float
    best_checkpoint: str
    history: List[Dict[str, float]]


@dataclass(frozen=True)
class PredictionResult:
    patient_ids: List[str]
    labels: np.ndarray
    probabilities: np.ndarray
    predictions: np.ndarray


class ClassifierTrainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: Union[str, torch.device] = "auto",
        precision: str = "amp",
        gradient_clip_norm: Optional[float] = 1.0,
        class_weights: Optional[Union[torch.Tensor, Sequence[float]]] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.device = resolve_device(device) if isinstance(device, str) else device
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.use_amp = amp_enabled(precision, self.device)
        self.amp_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
        self.scaler = _make_scaler(self.use_amp)
        self.gradient_clip_norm = gradient_clip_norm
        self.class_weights = (
            None if class_weights is None else torch.as_tensor(class_weights).float().to(self.device)
        )
        self.logger = logger or LOGGER
        self.global_step = 0

    @classmethod
    def from_config(
        cls,
        model: nn.Module,
        config: Any,
        labels: Optional[Union[torch.Tensor, Sequence[int]]] = None,
        device: Optional[Union[str, torch.device]] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> "ClassifierTrainer":
        selected_device = device or str(_cfg(config, "experiment.device", "auto"))
        if optimizer is None:
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=float(_cfg(config, "training.classifier.learning_rate", 1.0e-4)),
                weight_decay=float(_cfg(config, "training.classifier.weight_decay", 1.0e-4)),
            )
        weights = None
        if str(_cfg(config, "classifier.class_weight", "balanced")) == "balanced":
            if labels is None:
                raise ValueError("training labels are required for balanced class weights")
            weights = balanced_class_weights(labels)
        return cls(
            model,
            optimizer,
            selected_device,
            str(_cfg(config, "experiment.precision", "amp")),
            _cfg(config, "training.gradient_clip_norm", 1.0),
            weights,
        )

    def fit_records(
        self,
        train_records: Sequence[Any],
        validation_records: Sequence[Any],
        config: Any,
        checkpoint_dir: Union[str, Path],
        resume: Optional[bool] = None,
    ) -> ClassifierFitResult:
        from mci_diff.data import SequenceDataset, longitudinal_collate

        if not train_records or not validation_records:
            raise ValueError("classifier train and validation records cannot be empty")
        if self.class_weights is None and str(
            _cfg(config, "classifier.class_weight", "balanced")
        ) == "balanced":
            self.class_weights = balanced_class_weights(
                [int(record.label) for record in train_records]
            ).to(self.device)
        batch_size = int(_cfg(config, "training.classifier.batch_size", 32))
        workers = int(_cfg(config, "training.num_workers", 0))
        pin_memory = bool(_cfg(config, "training.pin_memory", True))
        train_loader = DataLoader(
            SequenceDataset(train_records),
            batch_size=batch_size,
            shuffle=True,
            num_workers=workers,
            pin_memory=pin_memory,
            collate_fn=longitudinal_collate,
            worker_init_fn=seed_worker if workers else None,
        )
        validation_loader = DataLoader(
            SequenceDataset(validation_records),
            batch_size=batch_size,
            shuffle=False,
            num_workers=workers,
            pin_memory=pin_memory,
            collate_fn=longitudinal_collate,
            worker_init_fn=seed_worker if workers else None,
        )
        do_resume = bool(_cfg(config, "experiment.resume", True)) if resume is None else resume
        return self.fit(
            train_loader,
            validation_loader,
            epochs=int(_cfg(config, "training.classifier.epochs", 100)),
            patience=int(_cfg(config, "training.classifier.patience", 20)),
            checkpoint_dir=checkpoint_dir,
            warmup_epochs=int(_cfg(config, "training.classifier.warmup_epochs", 5)),
            checkpoint_every=int(_cfg(config, "training.checkpoint_every", 10)),
            resume=do_resume,
        )

    def predict_records(
        self, records: Sequence[Any], config: Any
    ) -> PredictionResult:
        from mci_diff.data import SequenceDataset, longitudinal_collate

        if not records:
            raise ValueError("prediction records cannot be empty")
        loader = DataLoader(
            SequenceDataset(records),
            batch_size=int(_cfg(config, "training.classifier.batch_size", 32)),
            shuffle=False,
            num_workers=int(_cfg(config, "training.num_workers", 0)),
            pin_memory=bool(_cfg(config, "training.pin_memory", True)),
            collate_fn=longitudinal_collate,
        )
        return self.predict(loader, float(_cfg(config, "evaluation.threshold", 0.5)))

    def _epoch(self, loader: Iterable[Any], training: bool) -> Tuple[float, np.ndarray, np.ndarray]:
        self.model.train(training)
        total = 0.0
        count = 0
        labels_out: List[np.ndarray] = []
        probabilities_out: List[np.ndarray] = []
        for raw_batch in loader:
            batch = move_to_device(raw_batch, self.device)
            features, observed, labels, _ = _classifier_batch(batch)
            if training:
                self.optimizer.zero_grad(set_to_none=True)
            with _autocast(self.use_amp, self.amp_dtype):
                logits = self.model(features, observed)
                if logits.shape != (features.shape[0], 2):
                    raise ValueError("classifier must return [B, 2] logits")
                loss = F.cross_entropy(logits, labels, weight=self.class_weights)
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite classifier loss")
            if training:
                self.scaler.scale(loss).backward()
                if self.gradient_clip_norm is not None:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.global_step += 1
            batch_size = int(features.shape[0])
            total += float(loss.detach()) * batch_size
            count += batch_size
            labels_out.append(labels.detach().cpu().numpy())
            probabilities_out.append(logits.softmax(dim=1)[:, 1].detach().cpu().numpy())
        if not count:
            raise ValueError("empty classifier data loader")
        return total / count, np.concatenate(labels_out), np.concatenate(probabilities_out)

    def fit(
        self,
        train_loader: Iterable[Any],
        validation_loader: Iterable[Any],
        epochs: int,
        patience: int,
        checkpoint_dir: Union[str, Path],
        warmup_epochs: int = 0,
        checkpoint_every: int = 10,
        resume: bool = True,
    ) -> ClassifierFitResult:
        directory = Path(checkpoint_dir)
        directory.mkdir(parents=True, exist_ok=True)
        best_path, last_path = directory / "best.pt", directory / "last.pt"
        scheduler = build_epoch_scheduler(self.optimizer, epochs, warmup_epochs)
        stopping = EarlyStopping(patience, "min")
        history: List[Dict[str, float]] = []
        start_epoch = 0
        if resume and last_path.exists():
            state = restore_training_checkpoint(
                last_path,
                self.model,
                self.optimizer,
                scheduler,
                self.scaler,
                early_stopping=stopping,
                device=self.device,
            )
            start_epoch = state.epoch + 1
            self.global_step = state.global_step
            if state.extra and isinstance(state.extra.get("history"), list):
                history = list(state.extra["history"])

        for epoch in range(start_epoch, epochs):
            train_loss, _, _ = self._epoch(train_loader, True)
            with torch.no_grad():
                validation_loss, labels, probabilities = self._epoch(validation_loader, False)
            scheduler.step()
            metrics = binary_metrics(labels, probabilities)
            row = {
                "epoch": float(epoch),
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "validation_auc": metrics["auc"],
            }
            history.append(row)
            improved, stop = stopping.update(validation_loss)
            if improved:
                save_checkpoint(
                    best_path,
                    self.model,
                    self.optimizer,
                    scheduler,
                    self.scaler,
                    epoch=epoch,
                    global_step=self.global_step,
                    best_metric=stopping.best,
                    phase="classifier",
                    early_stopping=stopping,
                    extra={"history": history},
                )
            if (epoch + 1) % max(1, checkpoint_every) == 0 or stop or epoch + 1 == epochs:
                save_checkpoint(
                    last_path,
                    self.model,
                    self.optimizer,
                    scheduler,
                    self.scaler,
                    epoch=epoch,
                    global_step=self.global_step,
                    best_metric=stopping.best,
                    phase="classifier",
                    early_stopping=stopping,
                    extra={"history": history},
                )
            self.logger.info(
                "classifier epoch %d/%d train %.5f val %.5f",
                epoch + 1,
                epochs,
                train_loss,
                validation_loss,
            )
            if stop:
                break
        if best_path.exists():
            load_model_checkpoint(best_path, self.model, self.device)
        return ClassifierFitResult(
            len(history),
            float(stopping.best) if stopping.best is not None else float("nan"),
            str(best_path),
            history,
        )

    @torch.no_grad()
    def predict(self, loader: Iterable[Any], threshold: float = 0.5) -> PredictionResult:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must lie in [0, 1]")
        self.model.eval()
        labels_out: List[np.ndarray] = []
        probabilities_out: List[np.ndarray] = []
        identifiers: List[str] = []
        offset = 0
        for raw_batch in loader:
            batch = move_to_device(raw_batch, self.device)
            features, observed, labels, ids = _classifier_batch(batch)
            with _autocast(self.use_amp, self.amp_dtype):
                logits = self.model(features, observed)
            probability = logits.softmax(dim=1)[:, 1]
            labels_out.append(labels.detach().cpu().numpy())
            probabilities_out.append(probability.detach().cpu().numpy())
            if ids is None:
                identifiers.extend("subject_%06d" % (offset + index) for index in range(len(labels)))
            else:
                identifiers.extend(ids)
            offset += len(labels)
        if not labels_out:
            raise ValueError("empty prediction data loader")
        labels = np.concatenate(labels_out).astype(int)
        probabilities = np.concatenate(probabilities_out).astype(float)
        return PredictionResult(
            identifiers,
            labels,
            probabilities,
            (probabilities >= threshold).astype(int),
        )


# Metrics and fold reports


def _binary_arrays(
    labels: Union[np.ndarray, Sequence[int]], probabilities: Union[np.ndarray, Sequence[float]]
) -> Tuple[np.ndarray, np.ndarray]:
    truth = np.asarray(labels).reshape(-1)
    score = np.asarray(probabilities, dtype=float).reshape(-1)
    if len(truth) != len(score) or not len(truth):
        raise ValueError("labels and probabilities need equal non-zero lengths")
    if not np.isin(truth, [0, 1]).all():
        raise ValueError("labels must be binary 0/1")
    if not np.isfinite(score).all():
        raise ValueError("probabilities must be finite")
    return truth.astype(int), score


def binary_metrics(
    labels: Union[np.ndarray, Sequence[int]],
    probabilities: Union[np.ndarray, Sequence[float]],
    threshold: float = 0.5,
) -> Dict[str, float]:
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must lie in [0, 1]")
    truth, score = _binary_arrays(labels, probabilities)
    predicted = score >= threshold
    positive = truth == 1
    negative = ~positive
    tp = int(np.sum(predicted & positive))
    tn = int(np.sum((~predicted) & negative))
    fp = int(np.sum(predicted & negative))
    fn = int(np.sum((~predicted) & positive))
    try:
        auc = float(roc_auc_score(truth, score))
    except ValueError:
        auc = float("nan")
    return {
        "accuracy": float((tp + tn) / len(truth)),
        "sensitivity": float(tp / (tp + fn)) if tp + fn else float("nan"),
        "specificity": float(tn / (tn + fp)) if tn + fp else float("nan"),
        "auc": auc,
    }


@dataclass(frozen=True)
class BootstrapEstimate:
    estimate: float
    lower: float
    upper: float
    successful_samples: int


def bootstrap_binary_metrics(
    labels: Union[np.ndarray, Sequence[int]],
    probabilities: Union[np.ndarray, Sequence[float]],
    threshold: float = 0.5,
    samples: int = 1000,
    confidence: float = 0.95,
    seed: int = 2026,
) -> Dict[str, BootstrapEstimate]:
    truth, score = _binary_arrays(labels, probabilities)
    if samples < 1 or not 0.0 < confidence < 1.0:
        raise ValueError("invalid bootstrap settings")
    point = binary_metrics(truth, score, threshold)
    collected: Dict[str, List[float]] = {name: [] for name in point}
    rng = np.random.default_rng(seed)
    for _ in range(samples):
        indices = rng.integers(0, len(truth), len(truth))
        values = binary_metrics(truth[indices], score[indices], threshold)
        for name, value in values.items():
            if math.isfinite(value):
                collected[name].append(value)
    alpha = (1.0 - confidence) / 2.0
    result: Dict[str, BootstrapEstimate] = {}
    for name, estimate in point.items():
        values = np.asarray(collected[name], dtype=float)
        result[name] = BootstrapEstimate(
            estimate=estimate,
            lower=float(np.quantile(values, alpha)) if len(values) else float("nan"),
            upper=float(np.quantile(values, 1.0 - alpha)) if len(values) else float("nan"),
            successful_samples=int(len(values)),
        )
    return result


def write_prediction_csv(
    result: PredictionResult,
    path: Union[str, Path],
    fold: Optional[int] = None,
    datasets: Optional[Sequence[str]] = None,
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if datasets is not None and len(datasets) != len(result.labels):
        raise ValueError("datasets has the wrong length")
    descriptor, temporary = tempfile.mkstemp(
        prefix="." + destination.name + ".", suffix=".tmp", dir=str(destination.parent)
    )
    try:
        with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            columns = ["patient_id", "label", "probability", "prediction"]
            if fold is not None:
                columns.append("fold")
            if datasets is not None:
                columns.append("dataset")
            writer.writerow(columns)
            for index, identifier in enumerate(result.patient_ids):
                row: List[Any] = [
                    identifier,
                    int(result.labels[index]),
                    "%.10g" % float(result.probabilities[index]),
                    int(result.predictions[index]),
                ]
                if fold is not None:
                    row.append(int(fold))
                if datasets is not None:
                    row.append(str(datasets[index]))
                writer.writerow(row)
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return destination


def _load_metric_payload(value: Union[str, Path, Mapping[str, Any]]) -> Dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    with Path(value).open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise ValueError("fold metric file must contain an object")
    return payload


def aggregate_fold_metrics(
    folds: Sequence[Union[str, Path, Mapping[str, Any]]]
) -> Dict[str, Any]:
    if not folds:
        raise ValueError("at least one fold is required")
    payloads = [_load_metric_payload(item) for item in folds]
    names = ("accuracy", "sensitivity", "specificity", "auc")
    output: Dict[str, Any] = {"folds": len(payloads), "per_fold": payloads}
    fold_ids = [item.get("fold") for item in payloads if item.get("fold") is not None]
    if len(fold_ids) != len(set(fold_ids)):
        raise ValueError("duplicate fold identifiers")
    if fold_ids:
        output["fold_ids"] = sorted(fold_ids)
    for name in names:
        values: List[float] = []
        for payload in payloads:
            raw = payload.get(name)
            if isinstance(raw, Mapping):
                raw = raw.get("estimate")
            if raw is not None and math.isfinite(float(raw)):
                values.append(float(raw))
        array = np.asarray(values, dtype=float)
        output[name] = {
            "mean": float(array.mean()) if len(array) else None,
            "std": float(array.std(ddof=1)) if len(array) > 1 else (0.0 if len(array) else None),
            "valid_folds": int(len(array)),
        }
    return output


def write_fold_aggregate(
    folds: Sequence[Union[str, Path, Mapping[str, Any]]], path: Union[str, Path]
) -> Dict[str, Any]:
    result = aggregate_fold_metrics(folds)
    atomic_json_dump(result, path)
    return result


__all__ = [
    "AugmentationRecord",
    "BootstrapEstimate",
    "ClassifierFitResult",
    "ClassifierTrainer",
    "DiffusionTrainer",
    "EarlyStopping",
    "ExponentialMovingAverage",
    "PhaseResult",
    "PredictionResult",
    "ProgressiveResult",
    "ProgressiveTrainingPool",
    "ResumeState",
    "TrajectoryGenerationResult",
    "aggregate_fold_metrics",
    "amp_enabled",
    "atomic_json_dump",
    "atomic_torch_save",
    "augment_training_pool",
    "balanced_class_weights",
    "binary_metrics",
    "bootstrap_binary_metrics",
    "build_epoch_scheduler",
    "generate_trajectories",
    "load_model_checkpoint",
    "mark_stage_complete",
    "move_to_device",
    "pool_from_records",
    "remove_stage_marker",
    "resolve_device",
    "restore_rng_state",
    "restore_training_checkpoint",
    "rng_state",
    "run_progressive_training",
    "save_checkpoint",
    "seed_everything",
    "seed_worker",
    "stage_is_complete",
    "stage_marker_path",
    "standard_ddpm_impute",
    "torch_load",
    "write_fold_aggregate",
    "write_prediction_csv",
]
