import json
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional, Union, Literal

import pandas as pd
from loguru import logger
from transformers import TrainingArguments

from .run_glue import DataTrainingArguments, ModelArguments, run_glue


def evaluation(
    model_name_or_path: str,
    tokenizer_name: Optional[str] = None,
    tasks: Union[List[str], str] = [
        "mnli",
        "qnli",
        "qqp",
        "rte",
        "sst2",
        "mrpc",
        "cola",
        "stsb",
    ],
    learning_rate: Union[List[float], float] = [1e-5, 3e-5, 5e-5],
    epochs: Union[List[int], int] = [3, 5, 10],
    output_dir: str = "./evaluation_results",
    per_device_train_batch_size: int = 32,
    per_device_eval_batch_size: int = 32,
    max_seq_length: int = 128,
    seed: int = 42,
    cache_dir: Optional[str] = None,
) -> None:
    """
    Evaluate a model on GLUE tasks with various hyperparameter combinations.

    Args:
        model_name_or_path: Path to the pretrained model or model identifier
        tokenizer_name: Optional tokenizer name if different from model
        tasks: List of GLUE tasks to evaluate on
        learning_rate: Learning rate(s) to try
        epochs: Number of epochs to try
        output_dir: Base directory to save evaluation results
        per_device_train_batch_size: Batch size for training
        per_device_eval_batch_size: Batch size for evaluation
        max_seq_length: Maximum sequence length
        seed: Random seed for reproducibility
        cache_dir: Directory to cache model and datasets

    Returns:
        Dictionary with results for each task and hyperparameter combination
    """

    if isinstance(tasks, str):
        tasks = [tasks]
    if isinstance(learning_rate, (int, float)):
        learning_rate = [learning_rate]
    if isinstance(epochs, int):
        epochs = [epochs]

    short_model_name = (
        model_name_or_path
        if "/" not in model_name_or_path
        else model_name_or_path.split("/")[-1]
    )
    base_output_dir = Path(output_dir) / short_model_name

    # Prepare parameter grid
    param_grid = list(product(tasks, learning_rate, epochs))
    total_runs = len(param_grid)

    logger.info(f"Starting evaluation of model {short_model_name}")
    logger.info(f"Will run {total_runs} evaluations across {len(tasks)} tasks")

    for i, (task, lr, num_epochs) in enumerate(param_grid):
        logger.info(
            f"Run {i+1}/{total_runs}: Task: {task}, LR: {lr}, Epochs: {num_epochs}"
        )

        task_output_dir = (
            base_output_dir
            / task
            / f"epochs_{num_epochs}_lr_{str(lr).replace('.', '_')}"
        )
        task_output_dir.mkdir(parents=True, exist_ok=True)

        # Create arguments for run_glue
        model_args = ModelArguments(
            model_name_or_path=model_name_or_path,
            tokenizer_name=tokenizer_name,
            cache_dir=cache_dir,
        )

        data_args = DataTrainingArguments(
            task_name=task,
            max_seq_length=max_seq_length,
        )

        training_args = TrainingArguments(
            output_dir=str(task_output_dir),
            overwrite_output_dir=True,
            do_train=True,
            do_eval=True,
            eval_strategy="epoch",
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            learning_rate=lr,
            weight_decay=0.01,
            num_train_epochs=num_epochs,
            warmup_ratio=0.1,
            eval_steps=1_000_000_000,
            save_steps=1_000_000_000,
            seed=seed,
        )

        try:
            run_glue(
                model_args=model_args, data_args=data_args, training_args=training_args
            )
        except Exception as e:
            logger.error(
                f"Error evaluating {task} with lr={lr}, epochs={num_epochs}: {str(e)}"
            )


def create_summary_table(
    results_dir: Union[str, Path],
    save: bool = True,
    output_file: Optional[str] = None,
    include_average: bool = True,
    metrics_file: str = "eval_results.json",
    metric: str = "eval_accuracy",
    round_decimals: int = 2,
    aggregation: Literal["mean", "best"] = "mean",
) -> pd.DataFrame:
    """
    Create a simple summary table of model performance across GLUE tasks.

    Args:
        results_dir: Directory containing evaluation results organized by model/task
        save: Whether to save the summary table to CSV
        output_file: Custom filename for saving the summary (default: "summary.csv")
        include_average: Whether to include an average column across all tasks (default: True)
        metrics_file: Name of the file containing evaluation metrics (default: "eval_results.json")
        metric: Name of the metric to include in the summary (default: "eval_accuracy")
        round_decimals: Number of decimal places to round the scores to (default: 2)
        aggregation: Method to aggregate multiple runs for each task (default: "mean")
                     Options: "mean" - average of all runs, "best" - best score from all runs

    Returns:
        DataFrame containing model performance with models as rows and tasks as columns

    The function expects the following directory structure:
    results_dir/
    ├── model1/
    │   ├── task1/
    │   │   ├── epochs_X_lr_Y/
    │   │   │   └── eval_results.json
    │   │   └── ...
    │   ├── task2/
    │   │   └── ...
    │   └── ...
    ├── model2/
    │   └── ...
    └── ...
    """
    results_dir = Path(results_dir) if isinstance(results_dir, str) else results_dir
    assert results_dir.is_dir(), f"Directory {results_dir} does not exist"
    assert aggregation in [
        "mean",
        "best",
    ], f"Invalid aggregation method: {aggregation}, must be 'mean' or 'best'"

    # Collect results for each model and task
    model_results: Dict[str, Dict[str, float]] = {}

    for model_dir in results_dir.iterdir():
        if not model_dir.is_dir():
            continue

        model_name = model_dir.name
        model_results[model_name] = {}

        for task_dir in model_dir.iterdir():
            if not task_dir.is_dir():
                continue

            task_name = task_dir.name

            # Find the best result across all hyperparameter runs
            all_scores = []
            best_score = None

            for run_dir in task_dir.iterdir():
                if not run_dir.is_dir():
                    continue

                metrics_path = run_dir / metrics_file
                if not metrics_path.exists():
                    continue

                try:
                    with open(metrics_path, "r") as f:
                        metrics = json.load(f)

                    score = metrics.get(metric, None)
                    if score is not None:
                        all_scores.append(score)

                        if best_score is None or score > best_score:
                            best_score = score

                except (json.JSONDecodeError, IOError):
                    continue

            if all_scores:
                if aggregation == "mean":
                    final_score = sum(all_scores) / len(all_scores)
                else:
                    final_score = best_score

                model_results[model_name][task_name] = final_score * 100

    df = pd.DataFrame.from_dict(model_results, orient="index")

    if include_average:
        df["Avg"] = df.mean(axis=1)

    df = df.round(round_decimals)

    if save:
        save_path = results_dir / (output_file or "summary.csv")
        df.to_csv(save_path)
        logger.info(f"Summary table saved to {save_path}")

    return df
