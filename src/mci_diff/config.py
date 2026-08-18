"""Configuration loading and the few checks shared by all entry points."""

from __future__ import annotations

import copy
import math
from collections.abc import Iterator, Mapping, MutableMapping, Sequence
from difflib import get_close_matches
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

import yaml


VISIT_MONTHS = (0, 6, 12, 18, 24, 36)
TOP_LEVEL_SECTIONS = (
    "experiment",
    "data",
    "split",
    "normalization",
    "diffusion",
    "progressive",
    "training",
    "sampling",
    "classifier",
    "evaluation",
    "ablation",
    "matrix",
)
PROTOCOLS = ("adni_transfer", "adni_5fold")
ABLATION_SWITCHES = (
    "interpolation_task",
    "interpolation_augmentation",
    "extrapolation_task",
    "extrapolation_augmentation",
)


class Config(MutableMapping[str, Any]):
    """A recursively wrapped dictionary with attribute access."""

    def __init__(
        self,
        values: Optional[Mapping[str, Any]] = None,
        *,
        config_path: Optional[Union[str, Path]] = None,
        project_root: Optional[Union[str, Path]] = None,
    ) -> None:
        object.__setattr__(self, "_values", {})
        object.__setattr__(
            self,
            "_config_path",
            None if config_path is None else Path(config_path).expanduser().resolve(),
        )
        object.__setattr__(
            self,
            "_project_root",
            None if project_root is None else Path(project_root).expanduser().resolve(),
        )
        for key, value in (values or {}).items():
            self._values[str(key)] = self._wrap(value)

    @classmethod
    def _wrap(cls, value: Any) -> Any:
        if isinstance(value, Config):
            return value.clone()
        if isinstance(value, Mapping):
            return cls(value)
        if isinstance(value, list):
            return [cls._wrap(item) for item in value]
        if isinstance(value, tuple):
            return tuple(cls._wrap(item) for item in value)
        return value

    @classmethod
    def _unwrap(cls, value: Any) -> Any:
        if isinstance(value, Config):
            return {key: cls._unwrap(item) for key, item in value.items()}
        if isinstance(value, list):
            return [cls._unwrap(item) for item in value]
        if isinstance(value, tuple):
            return [cls._unwrap(item) for item in value]
        return copy.deepcopy(value)

    def __getitem__(self, key: str) -> Any:
        return self._values[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self._values[key] = self._wrap(value)

    def __delitem__(self, key: str) -> None:
        del self._values[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def __getattr__(self, name: str) -> Any:
        try:
            return self._values[name]
        except KeyError as error:
            raise AttributeError(name) from error

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_"):
            object.__setattr__(self, name, value)
        else:
            self._values[name] = self._wrap(value)

    @property
    def config_path(self) -> Optional[Path]:
        return self._config_path

    @property
    def project_root(self) -> Optional[Path]:
        return self._project_root

    def to_dict(self) -> Dict[str, Any]:
        return self._unwrap(self)

    def clone(self) -> "Config":
        return Config(
            self.to_dict(),
            config_path=self.config_path,
            project_root=self.project_root,
        )

    def dump(self, path: Union[str, Path]) -> None:
        dump_config(self, path)


def _hint(name: str, choices: Sequence[str]) -> str:
    match = get_close_matches(name, list(choices), n=1)
    return f"; did you mean {match[0]!r}?" if match else ""


def _deep_merge(
    base: Dict[str, Any],
    override: Mapping[str, Any],
    *,
    reject_unknown: bool,
    prefix: str = "",
) -> Dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        dotted = f"{prefix}.{key}" if prefix else str(key)
        if reject_unknown and key not in merged:
            raise KeyError(
                f"Unknown configuration key {dotted!r}"
                f"{_hint(str(key), [str(item) for item in merged])}"
            )
        if key in merged and isinstance(merged[key], dict) and isinstance(value, Mapping):
            merged[key] = _deep_merge(
                merged[key],
                value,
                reject_unknown=reject_unknown,
                prefix=dotted,
            )
        elif key in merged and isinstance(merged[key], dict) != isinstance(value, Mapping):
            raise TypeError(f"Configuration key {dotted!r} cannot change mapping shape")
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _read_yaml(path: Path, active: Optional[Set[Path]] = None) -> Dict[str, Any]:
    source = path.expanduser().resolve()
    chain = set() if active is None else active
    if source in chain:
        raise ValueError(f"Cyclic _base_ chain at {source}")
    if not source.is_file():
        raise FileNotFoundError(f"Configuration file does not exist: {source}")
    chain.add(source)
    try:
        with source.open("r", encoding="utf-8") as stream:
            current = yaml.safe_load(stream) or {}
        if not isinstance(current, dict):
            raise TypeError(f"Top-level YAML value must be a mapping: {source}")
        base_value = current.pop("_base_", None)
        if base_value is None:
            return current
        if isinstance(base_value, str):
            base_names = [base_value]
        elif isinstance(base_value, list) and all(isinstance(item, str) for item in base_value):
            base_names = base_value
        else:
            raise TypeError(f"_base_ must be a path or list of paths: {source}")
        if not base_names:
            raise ValueError(f"_base_ cannot be an empty list: {source}")
        merged: Dict[str, Any] = {}
        for base_name in base_names:
            inherited = _read_yaml(source.parent / base_name, chain)
            merged = _deep_merge(merged, inherited, reject_unknown=False)
        return _deep_merge(merged, current, reject_unknown=True)
    finally:
        chain.remove(source)


def _compatible_override(old: Any, new: Any) -> bool:
    if old is None:
        return True
    if isinstance(old, bool):
        return isinstance(new, bool)
    if isinstance(old, (int, float)) and not isinstance(old, bool):
        return isinstance(new, (int, float)) and not isinstance(new, bool)
    if isinstance(old, str):
        return isinstance(new, str)
    if isinstance(old, list):
        return isinstance(new, list)
    return isinstance(new, type(old))


def _apply_override(values: Dict[str, Any], expression: str) -> None:
    if "=" not in expression:
        raise ValueError(f"Override must be key=value, got {expression!r}")
    dotted, raw_value = expression.split("=", 1)
    parts = dotted.split(".")
    if not dotted or any(not part for part in parts):
        raise ValueError(f"Override has an empty key component: {expression!r}")
    cursor: Dict[str, Any] = values
    for offset, part in enumerate(parts[:-1]):
        if part not in cursor:
            prefix = ".".join(parts[: offset + 1])
            raise KeyError(
                f"Unknown override component {prefix!r}"
                f"{_hint(part, [str(item) for item in cursor])}"
            )
        if not isinstance(cursor[part], dict):
            raise TypeError(f"Cannot descend through non-mapping key {part!r}")
        cursor = cursor[part]
    leaf = parts[-1]
    if leaf not in cursor:
        raise KeyError(
            f"Unknown override key {dotted!r}"
            f"{_hint(leaf, [str(item) for item in cursor])}"
        )
    if isinstance(cursor[leaf], dict):
        raise TypeError(f"--set can only replace a leaf, not {dotted!r}")
    parsed = yaml.safe_load(raw_value)
    if not _compatible_override(cursor[leaf], parsed):
        raise TypeError(
            f"Override {dotted!r} changes {type(cursor[leaf]).__name__} "
            f"to {type(parsed).__name__}"
        )
    cursor[leaf] = parsed


def _project_root(config_path: Path) -> Path:
    for candidate in (config_path.parent, *config_path.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return config_path.parent


def _number(
    errors: List[str],
    value: Any,
    name: str,
    *,
    minimum: Optional[float] = None,
    maximum: Optional[float] = None,
    integer: bool = False,
) -> None:
    if isinstance(value, bool):
        errors.append(f"{name} must be numeric")
        return
    try:
        number = float(value)
    except (TypeError, ValueError):
        errors.append(f"{name} must be numeric")
        return
    if not math.isfinite(number):
        errors.append(f"{name} must be finite")
    elif integer and number != int(number):
        errors.append(f"{name} must be an integer")
    elif minimum is not None and number < minimum:
        errors.append(f"{name} must be >= {minimum:g}")
    elif maximum is not None and number > maximum:
        errors.append(f"{name} must be <= {maximum:g}")


def _boolean(errors: List[str], value: Any, name: str) -> None:
    if not isinstance(value, bool):
        errors.append(f"{name} must be true or false")


def _choice(errors: List[str], value: Any, name: str, choices: Sequence[str]) -> None:
    if str(value) not in choices:
        errors.append(f"{name} must be one of {', '.join(choices)}")


def _section(values: Mapping[str, Any], name: str, errors: List[str]) -> Mapping[str, Any]:
    value = values.get(name)
    if not isinstance(value, Mapping):
        errors.append(f"{name} must be a mapping")
        return {}
    return value


def _require_keys(section: Mapping[str, Any], name: str, keys: Sequence[str], errors: List[str]) -> None:
    for key in keys:
        if key not in section:
            errors.append(f"Missing configuration key {name}.{key}")


def validate_config(config: Config) -> None:
    """Validate the configuration."""

    values = config.to_dict()
    errors: List[str] = []
    unknown_sections = sorted(set(values) - set(TOP_LEVEL_SECTIONS))
    missing_sections = sorted(set(TOP_LEVEL_SECTIONS) - set(values))
    if unknown_sections:
        errors.append(f"Unknown top-level sections: {', '.join(unknown_sections)}")
    if missing_sections:
        errors.append(f"Missing top-level sections: {', '.join(missing_sections)}")

    experiment = _section(values, "experiment", errors)
    _require_keys(experiment, "experiment", ("name", "seed", "device", "precision"), errors)
    if not str(experiment.get("name", "")).strip():
        errors.append("experiment.name cannot be empty")
    if not str(experiment.get("output_dir", "")).strip():
        errors.append("experiment.output_dir cannot be empty")
    _number(errors, experiment.get("seed"), "experiment.seed", minimum=0, integer=True)
    _choice(errors, experiment.get("precision"), "experiment.precision", ("fp32", "amp"))
    for key in ("deterministic", "resume"):
        if key in experiment:
            _boolean(errors, experiment[key], f"experiment.{key}")

    data = _section(values, "data", errors)
    _require_keys(
        data,
        "data",
        ("manifest", "visit_months", "feature_dim", "allowed_datasets", "positive_label", "negative_label", "encoder"),
        errors,
    )
    try:
        visits = tuple(int(item) for item in data.get("visit_months", ()))
    except (TypeError, ValueError):
        visits = ()
    if visits != VISIT_MONTHS:
        errors.append(f"data.visit_months must be exactly {list(VISIT_MONTHS)}")
    _number(errors, data.get("feature_dim"), "data.feature_dim", minimum=1, maximum=65536, integer=True)
    allowed = data.get("allowed_datasets")
    if not isinstance(allowed, list) or set(allowed) != {"ADNI1", "ADNI2", "AIBL"}:
        errors.append("data.allowed_datasets must contain ADNI1, ADNI2, and AIBL")
    if not str(data.get("positive_label", "")).strip() or not str(data.get("negative_label", "")).strip():
        errors.append("data class labels cannot be empty")
    elif str(data.get("positive_label")).casefold() == str(data.get("negative_label")).casefold():
        errors.append("data positive and negative labels must differ")
    encoder = data.get("encoder", {})
    if not isinstance(encoder, Mapping):
        errors.append("data.encoder must be a mapping")
    else:
        _choice(
            errors,
            encoder.get("backend"),
            "data.encoder.backend",
            ("precomputed", "python_factory", "torchscript"),
        )
        if encoder.get("backend") == "python_factory" and not encoder.get("python_factory"):
            errors.append("data.encoder.python_factory is required for python_factory backend")
        if encoder.get("backend") == "torchscript" and not encoder.get("checkpoint"):
            errors.append("data.encoder.checkpoint is required for torchscript backend")
        if not str(encoder.get("output_key", "")).strip():
            errors.append("data.encoder.output_key cannot be empty")

    split = _section(values, "split", errors)
    _require_keys(split, "split", ("protocol", "file", "n_folds", "validation_fraction", "stratify"), errors)
    _choice(errors, split.get("protocol"), "split.protocol", PROTOCOLS)
    _number(errors, split.get("n_folds"), "split.n_folds", minimum=2, maximum=20, integer=True)
    if split.get("protocol") == "adni_5fold" and split.get("n_folds") != 5:
        errors.append("split.n_folds must be 5 for the adni_5fold protocol")
    _number(errors, split.get("validation_fraction"), "split.validation_fraction", minimum=0.0, maximum=0.5)
    fraction = split.get("validation_fraction")
    if isinstance(fraction, (int, float)) and not isinstance(fraction, bool) and fraction <= 0:
        errors.append("split.validation_fraction must be greater than zero")
    _choice(errors, split.get("stratify"), "split.stratify", ("label",))

    normalization = _section(values, "normalization", errors)
    _choice(errors, normalization.get("method"), "normalization.method", ("standard",))
    _choice(errors, normalization.get("fit_on"), "normalization.fit_on", ("train_only",))
    _number(errors, normalization.get("epsilon"), "normalization.epsilon", minimum=1.0e-12, maximum=1.0)
    if "enabled" in normalization:
        _boolean(errors, normalization["enabled"], "normalization.enabled")

    diffusion = _section(values, "diffusion", errors)
    _number(errors, diffusion.get("num_steps"), "diffusion.num_steps", minimum=1, maximum=10000, integer=True)
    _choice(errors, diffusion.get("schedule"), "diffusion.schedule", ("linear", "cosine"))
    _choice(errors, diffusion.get("prediction_type"), "diffusion.prediction_type", ("epsilon",))
    _number(
        errors,
        diffusion.get("hidden_dim"),
        "diffusion.hidden_dim",
        minimum=1,
        maximum=16384,
        integer=True,
    )
    _number(
        errors,
        diffusion.get("time_embedding_dim"),
        "diffusion.time_embedding_dim",
        minimum=4,
        maximum=4096,
        integer=True,
    )
    _number(errors, diffusion.get("denoiser_layers"), "diffusion.denoiser_layers", minimum=1, maximum=64, integer=True)
    _number(errors, diffusion.get("attention_heads"), "diffusion.attention_heads", minimum=1, maximum=128, integer=True)
    _number(errors, diffusion.get("feedforward_multiplier"), "diffusion.feedforward_multiplier", minimum=1, maximum=32)
    _number(errors, diffusion.get("dropout"), "diffusion.dropout", minimum=0.0, maximum=0.95)
    if diffusion.get("schedule") == "linear":
        start, end = diffusion.get("beta_start"), diffusion.get("beta_end")
        _number(errors, start, "diffusion.beta_start", minimum=1.0e-8, maximum=0.999)
        _number(errors, end, "diffusion.beta_end", minimum=1.0e-8, maximum=0.999)
        if isinstance(start, (int, float)) and isinstance(end, (int, float)) and start > end:
            errors.append("diffusion.beta_start cannot exceed diffusion.beta_end")
    if diffusion.get("clip_x0") is not None:
        _number(
            errors,
            diffusion.get("clip_x0"),
            "diffusion.clip_x0",
            minimum=1.0e-12,
        )
    hidden, heads = diffusion.get("hidden_dim"), diffusion.get("attention_heads")
    if isinstance(hidden, int) and isinstance(heads, int) and heads > 0 and hidden % heads:
        errors.append("diffusion.hidden_dim must be divisible by diffusion.attention_heads")

    progressive = _section(values, "progressive", errors)
    _number(errors, progressive.get("max_difficulty"), "progressive.max_difficulty", minimum=1, maximum=4, integer=True)
    _choice(errors, progressive.get("augmentation_mode"), "progressive.augmentation_mode", ("next_only", "iterative_d"))
    _number(
        errors,
        progressive.get("max_augmented_future_points"),
        "progressive.max_augmented_future_points",
        minimum=1,
        maximum=5,
        integer=True,
    )
    if (
        progressive.get("augmentation_mode") == "next_only"
        and progressive.get("max_augmented_future_points") != 1
    ):
        errors.append("progressive.max_augmented_future_points must be 1 for next_only")
    for key in (
        "interpolation_task",
        "interpolation_augmentation",
        "extrapolation_task",
        "extrapolation_augmentation",
    ):
        _boolean(errors, progressive.get(key), f"progressive.{key}")

    training = _section(values, "training", errors)
    _number(errors, training.get("num_workers"), "training.num_workers", minimum=0, maximum=256, integer=True)
    _number(
        errors,
        training.get("gradient_clip_norm"),
        "training.gradient_clip_norm",
        minimum=1.0e-12,
        maximum=1000,
    )
    _number(errors, training.get("ema_decay"), "training.ema_decay", minimum=0.0, maximum=0.999999)
    _number(errors, training.get("checkpoint_every"), "training.checkpoint_every", minimum=1, integer=True)
    _boolean(errors, training.get("pin_memory"), "training.pin_memory")
    for stage_name in ("diffusion", "classifier"):
        stage = training.get(stage_name)
        if not isinstance(stage, Mapping):
            errors.append(f"training.{stage_name} must be a mapping")
            continue
        epoch_key = "epochs_per_phase" if stage_name == "diffusion" else "epochs"
        _number(errors, stage.get(epoch_key), f"training.{stage_name}.{epoch_key}", minimum=1, maximum=100000, integer=True)
        _number(errors, stage.get("batch_size"), f"training.{stage_name}.batch_size", minimum=1, maximum=8192, integer=True)
        _number(errors, stage.get("learning_rate"), f"training.{stage_name}.learning_rate", minimum=1.0e-10, maximum=1.0)
        _number(errors, stage.get("weight_decay"), f"training.{stage_name}.weight_decay", minimum=0.0, maximum=10.0)
        if "warmup_epochs" in stage:
            _number(errors, stage["warmup_epochs"], f"training.{stage_name}.warmup_epochs", minimum=0, integer=True)
            epochs = stage.get(epoch_key)
            if (
                isinstance(stage["warmup_epochs"], (int, float))
                and not isinstance(stage["warmup_epochs"], bool)
                and isinstance(epochs, (int, float))
                and not isinstance(epochs, bool)
                and float(stage["warmup_epochs"]) >= float(epochs)
            ):
                errors.append(
                    f"training.{stage_name}.warmup_epochs must be smaller than {epoch_key}"
                )
        if "patience" in stage:
            _number(errors, stage["patience"], f"training.{stage_name}.patience", minimum=1, integer=True)
    sampling = _section(values, "sampling", errors)
    _number(errors, sampling.get("num_candidates"), "sampling.num_candidates", minimum=1, maximum=4096, integer=True)
    _number(errors, sampling.get("candidate_batch_size"), "sampling.candidate_batch_size", minimum=1, maximum=4096, integer=True)
    candidates, candidate_batch = sampling.get("num_candidates"), sampling.get("candidate_batch_size")
    if isinstance(candidates, int) and isinstance(candidate_batch, int) and candidate_batch > candidates:
        errors.append("sampling.candidate_batch_size cannot exceed sampling.num_candidates")
    _boolean(errors, sampling.get("save_all_candidates"), "sampling.save_all_candidates")
    scorer_factory = sampling.get("scorer_factory")
    if scorer_factory is not None and not str(scorer_factory).strip():
        errors.append("sampling.scorer_factory must be null or module.path:function")
    elif scorer_factory is not None and ":" not in str(scorer_factory):
        errors.append("sampling.scorer_factory must be written as module.path:function")
    _choice(errors, sampling.get("fallback"), "sampling.fallback", ("first",))
    _choice(errors, sampling.get("tie_break"), "sampling.tie_break", ("first",))

    classifier = _section(values, "classifier", errors)
    _choice(
        errors,
        classifier.get("architecture"),
        "classifier.architecture",
        ("transformer", "gru", "mlp"),
    )
    _number(errors, classifier.get("hidden_dim"), "classifier.hidden_dim", minimum=1, maximum=16384, integer=True)
    _number(errors, classifier.get("num_layers"), "classifier.num_layers", minimum=1, maximum=64, integer=True)
    _number(errors, classifier.get("attention_heads"), "classifier.attention_heads", minimum=1, maximum=128, integer=True)
    _number(errors, classifier.get("feedforward_multiplier"), "classifier.feedforward_multiplier", minimum=1, maximum=32)
    _number(errors, classifier.get("dropout"), "classifier.dropout", minimum=0.0, maximum=0.95)
    classifier_hidden, classifier_heads = classifier.get("hidden_dim"), classifier.get("attention_heads")
    if (
        classifier.get("architecture") == "transformer"
        and isinstance(classifier_hidden, int)
        and isinstance(classifier_heads, int)
        and classifier_heads > 0
        and classifier_hidden % classifier_heads
    ):
        errors.append("classifier.hidden_dim must be divisible by classifier.attention_heads")
    _choice(errors, classifier.get("pooling"), "classifier.pooling", ("cls", "mean", "last"))
    _choice(errors, classifier.get("class_weight"), "classifier.class_weight", ("balanced", "none"))

    evaluation = _section(values, "evaluation", errors)
    metrics = evaluation.get("metrics")
    expected_metrics = {"accuracy", "sensitivity", "specificity", "auc"}
    if not isinstance(metrics, list) or set(metrics) != expected_metrics or len(metrics) != 4:
        errors.append("evaluation.metrics must contain accuracy, sensitivity, specificity, and auc")
    _number(errors, evaluation.get("threshold"), "evaluation.threshold", minimum=0.0, maximum=1.0)
    _number(errors, evaluation.get("bootstrap_samples"), "evaluation.bootstrap_samples", minimum=0, integer=True)
    _number(errors, evaluation.get("bootstrap_confidence"), "evaluation.bootstrap_confidence", minimum=0.5, maximum=0.999999)
    _boolean(errors, evaluation.get("save_predictions"), "evaluation.save_predictions")

    ablation = _section(values, "ablation", errors)
    missing_switches = sorted(set(ABLATION_SWITCHES) - set(ablation))
    unknown_switches = sorted(set(ablation) - set(ABLATION_SWITCHES))
    if missing_switches:
        errors.append(f"Missing ablation switches: {', '.join(missing_switches)}")
    if unknown_switches:
        errors.append(f"Unknown ablation switches: {', '.join(unknown_switches)}")
    for key in ABLATION_SWITCHES:
        if key in ablation:
            _boolean(errors, ablation[key], f"ablation.{key}")

    matrix = _section(values, "matrix", errors)
    matrix_keys = {
        "enabled",
        "protocols",
        "ablations",
        "diffusion_steps",
        "max_difficulties",
        "candidate_sizes",
        "denoiser_layers",
    }
    unknown_matrix_keys = sorted(set(matrix) - matrix_keys)
    if unknown_matrix_keys:
        errors.append("Unknown matrix keys: %s" % ", ".join(unknown_matrix_keys))
    _boolean(errors, matrix.get("enabled"), "matrix.enabled")
    for key, value in matrix.items():
        if key == "enabled":
            continue
        if not isinstance(value, list) or not value:
            errors.append(f"matrix.{key} must be a non-empty experiment list")
    for protocol in matrix.get("protocols", []):
        _choice(errors, protocol, "matrix.protocols entry", PROTOCOLS)
    for value in matrix.get("diffusion_steps", []):
        _number(errors, value, "matrix.diffusion_steps entry", minimum=1, maximum=10000, integer=True)
    for value in matrix.get("max_difficulties", []):
        _number(errors, value, "matrix.max_difficulties entry", minimum=1, maximum=4, integer=True)
    for value in matrix.get("candidate_sizes", []):
        _number(errors, value, "matrix.candidate_sizes entry", minimum=1, maximum=4096, integer=True)
    for value in matrix.get("denoiser_layers", []):
        _number(errors, value, "matrix.denoiser_layers entry", minimum=1, maximum=64, integer=True)

    if errors:
        raise ValueError("Invalid configuration:\n- " + "\n- ".join(errors))


def load_config(
    path: Union[str, Path],
    overrides: Optional[Sequence[str]] = None,
    *,
    project_root: Optional[Union[str, Path]] = None,
) -> Config:
    config_path = Path(path).expanduser().resolve()
    values = _read_yaml(config_path)
    for expression in overrides or ():
        _apply_override(values, expression)
    root = _project_root(config_path) if project_root is None else Path(project_root).expanduser().resolve()
    config = Config(values, config_path=config_path, project_root=root)
    validate_config(config)
    return config


def dump_config(config: Config, path: Union[str, Path]) -> None:
    destination = Path(path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(
            config.to_dict(),
            stream,
            sort_keys=False,
            allow_unicode=True,
            default_flow_style=False,
        )


__all__ = [
    "ABLATION_SWITCHES",
    "PROTOCOLS",
    "TOP_LEVEL_SECTIONS",
    "VISIT_MONTHS",
    "Config",
    "dump_config",
    "load_config",
    "validate_config",
]
