from __future__ import annotations
import yaml
from pathlib import Path
from pydantic import ValidationError
from .schema import ExperimentConfig


class ConfigLoadError(Exception):
    pass


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    """
    Loads and validates an experiment definition from YAML/JSON.
    Raises ConfigLoadError with a clear message on any schema violation —
    never partially loads a malformed procedure graph.
    """
    path = Path(path)
    if not path.exists():
        raise ConfigLoadError(f"Experiment config not found: {path}")

    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    try:
        cfg = ExperimentConfig(**raw)
    except ValidationError as e:
        raise ConfigLoadError(f"Invalid experiment config {path}:\n{e}") from e

    # Structural checks beyond field-level validation
    step_ids = {s.id for s in cfg.steps}
    for step in cfg.steps:
        target = step.next or step.on_success
        if target and target not in step_ids:
            raise ConfigLoadError(
                f"Step '{step.id}' points to unknown next step '{target}'"
            )
        for cond, target_id in step.branches.items():
            if target_id not in step_ids:
                raise ConfigLoadError(
                    f"Step '{step.id}' branch '{cond}' points to unknown step '{target_id}'"
                )
        if step.recovery_step and step.recovery_step not in step_ids:
            raise ConfigLoadError(
                f"Step '{step.id}' recovery_step points to unknown step '{step.recovery_step}'"
            )

    return cfg
