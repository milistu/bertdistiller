import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from datasets import Dataset, load_dataset
from loguru import logger
from transformers import PreTrainedTokenizer


@dataclass
class DataConfig:
    """Configuration for dataset preparation.

    Attributes:
        dataset_name: Name of the dataset in HuggingFace hub
        column: Column name containing the text data (defaults to 'text')
        max_samples: Maximum number of samples to use (optional)
    """

    dataset_name: str
    column: str = "text"
    max_samples: Optional[int] = None


@dataclass
class DataArguments:
    """Arguments for dataset processing.

    Attributes:
        train_config: Configuration for training dataset
        val_config: Configuration for validation dataset (optional)
        max_seq_len: Maximum sequence length for tokenization
    """

    train_config: DataConfig
    val_config: Optional[DataConfig] = None
    max_seq_len: int = 512


def prepare_dataset(
    tokenizer: PreTrainedTokenizer,
    config: DataConfig,
    max_seq_len: int,
    tokenization_kwargs: Optional[Dict] = None,
    seed: int = 41,
) -> Dataset:
    """Prepare and tokenize a dataset for training or validation.

    Args:
        tokenizer: Tokenizer instance to process the text
        config: Dataset configuration
        max_seq_len: Maximum sequence length for tokenization
        tokenization_kwargs: Additional arguments for tokenizer
        seed: Random seed for dataset shuffling

    Returns:
        Tokenized dataset ready for training

    Raises:
        ValueError: If dataset_name is not found or column doesn't exist
    """

    # Disable tokenizers parallelism for better memory usage
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    try:
        # Load and optionally limit dataset size
        dataset = load_dataset(config.dataset_name, split="train").shuffle(seed=seed)
        if config.max_samples:
            dataset = dataset.select(range(min(len(dataset), config.max_samples)))

        logger.info(f"Loaded dataset {config.dataset_name} with {len(dataset)} samples")

        # Validate column exists
        if config.column not in dataset.column_names:
            raise ValueError(
                f"Column '{config.column}' not found in dataset. "
                f"Available columns: {dataset.column_names}"
            )

        # Tokenization Function
        def tokenize_batch(examples):
            """Tokenize a batch of examples."""
            # Join multi-line text with newlines
            texts = [
                "\n".join(text) if isinstance(text, (list, tuple)) else text
                for text in examples[config.column]
            ]
            # Apply tokenization with default

            return tokenizer(
                texts,
                truncation=True,
                max_length=max_seq_len,
                **(tokenization_kwargs or {}),
            )

        # Apply tokenization and convert to PyTorch format
        return dataset.map(
            tokenize_batch,
            batched=True,
            desc="Tokenizing dataset",
            num_proc=os.cpu_count(),
        ).with_format("torch")

    except Exception as e:
        logger.error(f"Error preparing dataset: {str(e)}")
        raise


def get_tokenized_datasets(
    data_args: DataArguments,
    tokenizer: PreTrainedTokenizer,
    tokenization_kwargs: Optional[Dict] = None,
) -> Tuple[Dataset, Optional[Dataset]]:
    """Prepare tokenized datasets for training and validation.

    Args:
        data_args: Dataset processing arguments
        tokenizer: Tokenizer to process the text
        tokenization_kwargs: Additional arguments for tokenizer

    Returns:
        Tuple of (train_dataset, val_dataset)
        val_dataset will be None if no validation config is provided
    """
    train_dataset = prepare_dataset(
        tokenizer=tokenizer,
        config=data_args.train_config,
        max_seq_len=data_args.max_seq_len,
        tokenization_kwargs=tokenization_kwargs,
    )

    val_dataset = None
    if data_args.val_config:
        val_dataset = prepare_dataset(
            tokenizer=tokenizer,
            config=data_args.val_config,
            max_seq_len=data_args.max_seq_len,
            tokenization_kwargs=tokenization_kwargs,
        )

    return train_dataset, val_dataset
