# src/minilm/trainer/__init__.py

__version__ = "0.1.0"

from .distillation import DistillationConfig, DistillationTrainer

__all__ = ["DistillationConfig", "DistillationTrainer"]
