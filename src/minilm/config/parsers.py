# src/minilm/config/parsers.py

import argparse
from typing import Dict


def create_model_parser() -> argparse.ArgumentParser:
    """Creates argument parser for model configuration.

    Returns:
        ArgumentParser configured for model parameters
    """
    parser = argparse.ArgumentParser(description="Parameters for model configuration")

    # Model paths
    parser.add_argument(
        "--input_model_dir",
        type=str,
        required=True,
        help="Directory containing the pre-trained teacher model",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=False,
        help="Optional checkpoint directory for resuming training",
    )
    parser.add_argument(
        "--tokenizer_dir",
        type=str,
        required=False,
        help="Optional tokenizer directory (defaults to input_model_dir)",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        required=False,
        help="Optional directory for caching model files",
    )

    # Student model architecture
    parser.add_argument(
        "--student_hidden_size",
        type=int,
        required=True,
        help="Hidden layer size for student model",
    )
    parser.add_argument(
        "--student_num_layers",
        type=int,
        required=True,
        help="Number of layers in student model",
    )
    parser.add_argument(
        "--student_attention_heads",
        type=int,
        required=True,
        help="Number of attention heads in student model",
    )

    # Distillation parameters
    parser.add_argument(
        "--L",
        type=int,
        required=True,
        help="Teacher layer to distill from (1-based indexing)",
    )
    parser.add_argument(
        "--num_relation_heads",
        type=int,
        required=True,
        help="Number of relation heads for MiniLM",
    )
    parser.add_argument(
        "--minilm_relations",
        type=str,
        required=False,
        default="{(1, 1): 1, (2, 2): 1, (3, 3): 1}",
        help="Relations configuration as a dictionary mapping (query_id, key_id) "
        "to weights. Query=1, Key=2, Value=3",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["bert", "modernbert"],
        default="bert",
        help="Model type for MiniLM distillation",
    )

    return parser


def create_training_parser() -> argparse.ArgumentParser:
    """Creates argument parser for training configuration."""
    parser = argparse.ArgumentParser(
        description="Parameters for training configuration"
    )

    # Data configuration
    parser.add_argument(
        "--train_config",
        type=str,
        required=True,
        help="Path to training dataset configuration file",
    )
    parser.add_argument(
        "--val_config",
        type=str,
        help="Optional path to validation dataset configuration",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=512,
        help="Maximum sequence length for inputs",
    )

    # Training hyperparameters
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=32,
        help="Batch size per device for training",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=6e-4, help="Learning rate for training"
    )
    parser.add_argument(
        "--adam_epsilon", type=float, default=1e-6, help="Epsilon for Adam optimizer"
    )
    parser.add_argument(
        "--adam_beta1", type=float, default=0.9, help="Beta1 for Adam optimizer"
    )
    parser.add_argument(
        "--adam_beta2", type=float, default=0.999, help="Beta2 for Adam optimizer"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.01, help="Weight decay for optimization"
    )
    parser.add_argument(
        "--max_steps", type=int, default=400000, help="Maximum number of training steps"
    )
    parser.add_argument(
        "--save_steps", type=int, default=50000, help="Save checkpoint every X steps"
    )
    parser.add_argument(
        "--logging_steps", type=int, default=1000, help="Log every X steps"
    )
    parser.add_argument(
        "--warmup_steps", type=int, default=4000, help="Number of warmup steps"
    )

    # Training settings
    parser.add_argument(
        "--ddp_find_unused_parameters",
        type=str,
        default="true",
        help="Find unused parameters in DDP",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./out",
        help="Output directory for checkpoints and logs",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )

    return parser


def parse_grouped_args(
    args: list, group_keys: list[str]
) -> Dict[str, argparse.Namespace]:
    """Parses command-line arguments into groups based on specified keys.

    Args:
        args: List of command-line arguments
        group_keys: List of keys that mark the start of each argument group

    Returns:
        Dictionary mapping group keys to parsed arguments

    Raises:
        ValueError: If a required group key is missing from arguments
    """
    # Find positions of group keys
    key_positions = {key: args.index(key) if key in args else -1 for key in group_keys}

    # Verify all required keys are present
    missing_keys = [key for key, pos in key_positions.items() if pos == -1]
    if missing_keys:
        raise ValueError(f"Required argument groups missing: {missing_keys}")

    # Sort keys by position
    sorted_keys = sorted(key_positions.items(), key=lambda x: x[1])

    # Parse each group
    parsed_args = {}
    parsers = {"--model": create_model_parser(), "--training": create_training_parser()}

    for i, (key, start_pos) in enumerate(sorted_keys):
        # Get end position (next key or end of args)
        end_pos = len(args) if i == len(sorted_keys) - 1 else sorted_keys[i + 1][1]

        # Parse group arguments
        group_args = args[start_pos + 1 : end_pos]
        parsed_args[key] = parsers[key].parse_args(group_args)

    return parsed_args
