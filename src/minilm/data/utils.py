import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from datasets import Dataset, concatenate_datasets, load_dataset
from loguru import logger
from transformers import PreTrainedTokenizer


@dataclass
class DatasetSource:
    """Configuration for a single dataset source.

    This class represents one dataset that can come from either Hugging Face
    or a custom URL. It handles both cases uniformly.

    Attributes:
        name: Dataset name (if from Hugging Face) or URL
        column: Column containing the text data
        subset: Optional subset name for Hugging Face datasets
        is_hf: Whether this is a Hugging Face dataset
    """

    name: str
    column: str
    subset: Optional[str] = None
    is_hf: bool = True

    def load(self, cache_dir: Optional[str] = None) -> Dataset:
        """Load the dataset from its source.

        Args:
        cache_dir: Optional directory to cache the dataset

        Returns:
            Loaded dataset ready for processing

        Raises:
            ValueError: If dataset cannot be loaded
        """
        try:
            if self.is_hf:
                # Load from Hugging Face
                return load_dataset(
                    self.name,
                    self.subset,
                    split="train",
                    cache_dir=cache_dir,
                )
            else:
                # Load from URL using datasets' built-in text loader
                return load_dataset(
                    "text",
                    data_files={"train": self.name},
                    split="train",
                    cache_dir=cache_dir,
                )
        except Exception as e:
            logger.error(f"Failed to load dataset {self.name}: {str(e)}")
            raise


@dataclass
class DataConfig:
    """Configuration for dataset preparation.

    This class can handle multiple dataset sources and combine them
    into a single training dataset.

    Attributes:
        sources: List of dataset sources to combine
        max_samples: Maximum number of samples per dataset to use (optional)
        cache_dir: Optional cache directory for datasets
    """

    sources: List[DatasetSource]
    max_samples: Optional[int] = None
    cache_dir: Optional[str] = None


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
    """Prepare and tokenize multiple datasets for training or validation.

    This function loads multiple datasets, combines them, and applies
    consistent tokenization across the combined dataset.

    Args:
        tokenizer: Tokenizer instance to process the text
        config: Dataset configuration containing multiple sources
        max_seq_len: Maximum sequence length for tokenization
        tokenization_kwargs: Additional arguments for tokenizer
        seed: Random seed for dataset shuffling

    Returns:
        Combined and tokenized dataset ready for training
    """

    # Disable tokenizers parallelism for better memory usage
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    try:
        # Load and optionally limit dataset size
        datasets = []
        for source in config.sources:
            dataset = source.load(cache_dir=config.cache_dir)

            # Limit samples if needed
            if config.max_samples:
                dataset = dataset.select(range(min(len(dataset), config.max_samples)))

            # Tokenization Function
            def tokenize_batch(examples):
                """Tokenize a batch of examples."""
                # Join multi-line text with newlines
                texts = [
                    "\n".join(text) if isinstance(text, (list, tuple)) else text
                    for text in examples[source.column]
                ]
                # Apply tokenization with default

                return tokenizer(
                    texts,
                    truncation=True,
                    max_length=max_seq_len,
                    **(tokenization_kwargs or {}),
                )

            tokenized = dataset.map(
                tokenize_batch,
                batched=True,
                desc=f"Tokenizing {source.name}",
                num_proc=os.cpu_count(),
            )
            datasets.append(tokenized)

        # Combine all datasets
        combined_dataset = concatenate_datasets(datasets)
        # Shuffle
        combined_dataset = combined_dataset.shuffle(seed=seed)

        logger.info(
            f"Created dataset with {len(combined_dataset)} samples "
            f"from {len(config.sources)} sources"
        )

        return combined_dataset.with_format("torch")

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
