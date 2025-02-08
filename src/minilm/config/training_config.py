# src/minilm/config/training_config.py

from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainingConfig:
    """Configuration for training parameters and hyperparameters."""

    # Data configuration
    train_config: str
    max_seq_len: int = 512
    val_config: Optional[str] = None

    # Training hyperparameters
    per_device_train_batch_size: int = 256
    learning_rate: float = 6e-4
    adam_epsilon: float = 1e-6
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    weight_decay: float = 0.01
    max_steps: int = 400_000
    save_steps: int = 50_000
    logging_steps: int = 1_000
    warmup_steps: int = 4_000

    # Training settings
    ddp_find_unused_parameters: bool = True
    output_dir: str = "./out"
    seed: int = 42

    @classmethod
    def from_args(cls, args):
        """Creates config from parsed command-line arguments."""
        return cls(
            train_config=args.train_config,
            max_seq_len=getattr(args, "max_seq_len", 512),
            val_config=getattr(args, "val_config", None),
            per_device_train_batch_size=getattr(
                args, "per_device_train_batch_size", 256
            ),
            learning_rate=float(getattr(args, "learning_rate", 6e-4)),
            adam_epsilon=float(getattr(args, "adam_epsilon", 1e-6)),
            adam_beta1=float(getattr(args, "adam_beta1", 0.9)),
            adam_beta2=float(getattr(args, "adam_beta2", 0.999)),
            weight_decay=float(getattr(args, "weight_decay", 0.01)),
            max_steps=int(getattr(args, "max_steps", 400000)),
            save_steps=int(getattr(args, "save_steps", 50000)),
            logging_steps=int(getattr(args, "logging_steps", 1000)),
            warmup_steps=int(getattr(args, "warmup_steps", 4000)),
            ddp_find_unused_parameters=getattr(
                args, "ddp_find_unused_parameters", True
            ),
            output_dir=getattr(args, "output_dir", "./out"),
            seed=int(getattr(args, "seed", 42)),
        )
