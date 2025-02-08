# src/minilm/config/data_config.py

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class DatasetSource:
    """Configuration for a single dataset source.

    This class represents one dataset that can come from either Hugging Face
    or a custom URL. It handles both cases uniformly.

    Attributes:
        name (str): Dataset name (if from Hugging Face) or URL
        column (str): Column containing the text data
        subset (Optional[str]): Optional subset name for Hugging Face datasets
        is_hf (bool): Whether this is a Hugging Face dataset

    Example:
        ```python
        # Hugging Face dataset
        wiki_source = DatasetSource(
            name="wikipedia",
            column="text",
            subset="20220301.en",
            is_hf=True
        )

        # Custom dataset
        custom_source = DatasetSource(
            name="path/to/data.csv",
            column="text_column",
            is_hf=False
        )
        ```
    """

    name: str
    column: str
    subset: Optional[str] = None
    is_hf: bool = True


@dataclass
class DataConfig:
    """Configuration for dataset preparation.

    This class can handle multiple dataset sources and combine them
    into a single training dataset.

    Attributes:
        sources (List[DatasetSource]): List of dataset sources to combine
        max_samples (Optional[int): Maximum number of samples per dataset to use (optional)
        cache_dir (Optional[str]): Optional cache directory for datasets

    Example:
        ```python
        # Using all samples, default cache
        config = DataConfig(
            sources=[wiki_source, custom_source]
        )

        # Limiting samples with custom cache
        config = DataConfig(
            sources=[wiki_source, custom_source],
            max_samples=1000000,  # Use 1M samples per source
            cache_dir="./cache"   # Cache datasets here
        )
        ```
    """

    sources: List[DatasetSource]
    max_samples: Optional[int] = None
    cache_dir: Optional[str] = None


@dataclass
class DataArguments:
    """Arguments for dataset processing.

    Attributes:
        train_config (DataConfig): Configuration for training dataset
        val_config (Optional[DataConfig]): Configuration for validation dataset (optional)
        max_seq_len (int): Maximum sequence length for tokenization

    Example:
        ```python
        args = DataArguments(
            train_config=train_config,
            val_config=val_config,  # Optional, set to None if not using validation
            max_seq_len=512
        )
        ```
    """

    train_config: DataConfig
    val_config: Optional[DataConfig] = None
    max_seq_len: int = 512
