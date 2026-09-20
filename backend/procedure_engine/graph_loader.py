from backend.config.loader import load_experiment_config
from backend.config.schema import ExperimentConfig


def load_graph(config_path: str) -> ExperimentConfig:
    """Thin wrapper kept separate so the procedure engine doesn't import
    the config module directly — keeps the engine testable with in-memory
    ExperimentConfig objects too (e.g. for unit tests)."""
    return load_experiment_config(config_path)
