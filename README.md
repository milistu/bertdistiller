# MiniLM - Distillation of BERT models

## Configuration

### Dataset Configuration

The dataset configuration is specified in JSON format. You can find example configurations in the `configs/` directory:

- `configs/default/`: Contains default configurations for training and validation
- `configs/examples/`: Contains example configurations for different use cases

#### Basic Configuration Structure

```json
{
    "sources": [
        {
            "name": "wikipedia",    // Dataset name or path
            "column": "text",       // Column containing the text
            "subset": "20220301.en", // Optional: dataset subset
            "is_hf": true          // true for HuggingFace datasets, false for custom
        }
    ],
    "max_samples": null,  // Optional: limit samples per dataset
    "cache_dir": null    // Optional: custom cache directory
}
```

#### Using Example Configurations

1. Copy an example configuration from `configs/examples/`
2. Modify according to your needs
3. Pass the path to your configuration file:

```bash
python train.py --train_config path/to/your/config.json
```

#### Configuration Tips

- Remove `max_samples` to use all available samples
- Remove `cache_dir` to use the default cache location
- For custom datasets, set `is_hf: false` and provide the path in `name`