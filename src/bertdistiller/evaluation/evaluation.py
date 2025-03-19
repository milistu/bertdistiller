import json
import tempfile
from dataclasses import asdict
from itertools import product
from pathlib import Path
from typing import List, Optional
import subprocess

from loguru import logger
from transformers import TrainingArguments

from .run_glue import DataTrainingArguments, ModelArguments


def evaluate(
    model_name_or_path: str,
    tasks: List[str] = ["mnli", "qnli", "qqp", "rte", "sst2", "mrpc", "cola", "stsb"],
    learning_rate: List[float] = [1e-5, 3e-5, 5e-5],
    epochs: List[int] = [3, 5, 10],
    output_dir: str = "./evaluation_results",
    per_device_train_batch_size: int = 32,
    per_device_eval_batch_size: int = 32,
    max_seq_length: int = 128,
    seed: int = 42,
    model_args: Optional[ModelArguments] = None,
    data_args: Optional[DataTrainingArguments] = None,
    training_args: Optional[TrainingArguments] = None,
    run_glue_script_path: str = "src/bertdistiller/evaluation/run_glue.py",
    cache_dir: Optional[str] = None,
    **additional_args,
) -> None:
    """
    Evaluate a model on GLUE tasks with various hyperparameter combinations.

    Args:
        model_name_or_path: Path to the pretrained model or model identifier from huggingface.co/models
        tasks: List of GLUE tasks to evaluate on
        learning_rates: List of learning rates to try
        epochs: List of number of epochs to try
        output_dir: Base directory to save evaluation results
        per_device_train_batch_size: Batch size for training
        per_device_eval_batch_size: Batch size for evaluation
        max_seq_length: Maximum sequence length
        seed: Random seed for reproducibility
        model_args: Optional ModelArguments to override defaults
        data_args: Optional DataTrainingArguments to override defaults
        training_args: Optional TrainingArguments to override defaults
        run_glue_script_path: Path to the run_glue.py script
        cache_dir: Directory to cache model and datasets
        **additional_args: Additional arguments to pass to the script
    """
    short_model_name = (
        model_name_or_path
        if "/" not in model_name_or_path
        else model_name_or_path.split("/")[-1]
    )
    output_dir = Path(output_dir)
    output_dir = output_dir / short_model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # all_results = {}

    # Create parameter grid
    param_grid = list(product(tasks, learning_rate, epochs))
    total_runs = len(param_grid)

    logger.info(
        f"Starting evaluation of model {short_model_name} on {len(tasks)} tasks with {total_runs} parameter combinations"
    )

    for i, (task, lr, num_epochs) in enumerate(param_grid):
        logger.info(
            f"Run {i+1}/{total_runs}: Task: {task}, Learning Rate: {lr}, Epochs: {num_epochs}"
        )

        task_output_dir = (
            output_dir / task / f"{num_epochs}_{str(lr).replace('.', '_')}"
        )
        task_output_dir.mkdir(parents=True, exist_ok=True)

        model_args_dict = {
            "model_name_or_path": model_name_or_path,
            "cache_dir": cache_dir,
        }

        data_args_dict = {
            "task_name": task,
            "max_seq_length": max_seq_length,
        }

        training_args_dict = {
            "output_dir": str(task_output_dir),
            "do_train": True,
            "do_eval": True,
            "learning_rate": lr,
            "num_train_epochs": num_epochs,
            "per_device_train_batch_size": per_device_train_batch_size,
            "per_device_eval_batch_size": per_device_eval_batch_size,
            "warmup_ratio": 0.1,
            "weight_decay": 0.01,
            "evaluation_strategy": "epoch",
            "eval_steps": 1_000_000_000,
            "save_steps": 1_000_000_000,
            # "save_strategy": "epoch",
            # "load_best_model_at_end": True,
            "seed": seed,
            "overwrite_output_dir": True,
        }

        if model_args:
            for key, value in asdict(model_args).items():
                if value is not None:
                    model_args_dict[key] = value

        if data_args:
            for key, value in asdict(data_args).items():
                if value is not None:
                    data_args_dict[key] = value

        if training_args:
            for key, value in asdict(training_args).items():
                if value is not None:
                    training_args_dict[key] = value

        # Create temporary JSON file with all the arguments
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(
                # {
                #     "model_args": asdict(default_model_args),
                #     "data_args": asdict(default_data_args),
                #     "training_args": asdict(default_training_args),
                # },
                {
                    **model_args_dict,
                    **data_args_dict,
                    **training_args_dict,
                },
                f,
                indent=2,
            )
            args_file = f.name

        try:
            # Run the evaluation script
            cmd = ["python", run_glue_script_path, args_file]
            logger.info(f"Running command: {' '.join(cmd)}")

            # Run the command
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
            )
            # Print output in real-time
            while True:
                output = process.stdout.readline()
                if output == "" and process.poll() is not None:
                    break
                if output:
                    print(output.strip(), flush=True)

            # Get return code
            return_code = process.poll()

            if return_code != 0:
                logger.error(f"Process exited with code {return_code}")

        finally:
            # Remove the temporary file
            Path(args_file).unlink()
