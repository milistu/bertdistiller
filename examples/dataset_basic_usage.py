from minilm.config import DatasetSource, DataConfig, DataArguments

# Define Source
wiki_source = DatasetSource(
    name="bookcorpus/bookcorpus",
    column="text",
    subset="20220301.en",
    is_hf=True,
)

# custom_source = DatasetSource(
#     name="path/to/data.csv", column="text_column", is_hf=False
# )

train_config = DataConfig(
    sources=[wiki_source], cache_dir=".cache"
)  # , custom_source])


# Create validation config (with sample limit and custom cache)
val_config = DataConfig(
    sources=[wiki_source],
    max_samples=10000,  # Only use 10K samples for validation
    cache_dir=".cache",  # Custom cache location
)

# Create data arguments
data_args = DataArguments(
    train_config=train_config, val_config=val_config, max_seq_len=512
)
