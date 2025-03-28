import os
from typing import Dict, List, Optional, Tuple, Union

from datasets import Dataset, concatenate_datasets
from transformers import PreTrainedTokenizer


def prepare_dataset(
    datasets: Union[List[Dataset], List[Tuple[Dataset, str]]],
    tokenizer: PreTrainedTokenizer,
    max_seq_len: int,
    default_column: str = "text",
    tokenization_kwargs: Optional[Dict] = None,
    seed: int = 42,
) -> Dataset:
    """Prepare and tokenize multiple datasets for training or validation.

    Args:
        datasets: List of datasets to process, either as Dataset objects or
                 tuples of (Dataset, column_name) to specify different text columns
        tokenizer: Tokenizer instance to process the text
        max_seq_len: Maximum sequence length for tokenization
        default_column: Default text column name if not specified per dataset
        tokenization_kwargs: Additional arguments for tokenizer
        seed: Random seed for dataset shuffling

    Returns:
        Combined and tokenized dataset ready for training
    """
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    tokenized_datasets = []

    for dataset_item in datasets:
        # Handle either Dataset or (Dataset, column) format
        if isinstance(dataset_item, tuple) and len(dataset_item) == 2:
            dataset, column = dataset_item
        else:
            dataset = dataset_item
            column = default_column

        if column not in dataset.column_names:
            raise ValueError(
                f"Column '{column}' not found in dataset. Available columns: {dataset.column_names}"
            )

        def tokenize_batch(examples):
            texts = [
                "\n".join(text) if isinstance(text, (list, tuple)) else text
                for text in examples[column]
            ]
            return tokenizer(
                texts,
                truncation=True,
                max_length=max_seq_len,
                **(tokenization_kwargs or {}),
            )

        tokenized = dataset.map(
            tokenize_batch,
            batched=True,
            desc=f"Tokenizing dataset: {dataset.info.dataset_name}",
            num_proc=os.cpu_count(),
            remove_columns=dataset.column_names,
        )
        tokenized_datasets.append(tokenized)

    combined_dataset = concatenate_datasets(tokenized_datasets)
    combined_dataset = combined_dataset.shuffle(seed=seed)

    return combined_dataset.with_format("torch")
