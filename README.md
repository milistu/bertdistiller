# BertDistiller: Knowledge Distillation for BERT Models

[![HF Models](https://img.shields.io/badge/%F0%9F%A4%97-models-yellow)](https://huggingface.co/bertdistiller)
[![Python Versions](https://img.shields.io/pypi/pyversions/bertdistiller?logo=python&logoColor=white)](https://pypi.org/project/bertdistiller/)
[![License](https://img.shields.io/badge/License-Apache_2.0-green.svg)](LICENSE)

A flexible framework for distilling BERT models using various distillation techniques, built on the Hugging Face Transformers library.

Currently implements:
- [MiniLMv2](https://arxiv.org/abs/2012.15828): Multi-Head Self-Attention Relation Distillation for compressing pretrained Transformers.

## Overview

BertDistiller is a framework for knowledge distillation of BERT models. It currently implements the MiniLMv2 technique - a task-agnostic distillation approach that compresses large pretrained transformer models into smaller, faster models while maintaining comparable performance.

Key features:
- **Built on Hugging Face Transformers**: Seamless integration with the popular transformers library
- **Task-agnostic distillation**: Compress models without task-specific fine-tuning (task-specific distillation planned for future releases)
- **Flexible architecture**: Compress models to different sizes by configuring layers and dimensions
- **Teacher weight inheritance**: Option to initialize student with teacher weights for better performance
- **Support for various teacher models**: Compatible with BERT-based architectures

## Installation

```bash
# From PyPI
pip install bertdistiller

# From source
git clone https://github.com/milistu/bertdistiller.git
cd bertdistiller
pip install -e .
```

## Quick Start

### Complete Example

See the [examples/minilm_distillation.py](examples/minilm_distillation.py) file for a complete distillation example.

### Preparing Datasets

The library includes utilities for preparing datasets for distillation:

```python
from bertdistiller.data import prepare_dataset
from transformers import AutoTokenizer
from datasets import load_dataset

# Load datasets
dataset_bc = load_dataset("bookcorpus/bookcorpus", split="train")
dataset_wiki = load_dataset("legacy-datasets/wikipedia", "20220301.en", split="train")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")

# Prepare and combine datasets
combined_dataset = prepare_dataset(
    datasets=[dataset_bc, dataset_wiki],  # List of datasets
    tokenizer=tokenizer,
    max_seq_len=128,
    tokenization_kwargs={"padding": "do_not_pad"},
)

# Split into train/test
dataset = combined_dataset.train_test_split(test_size=0.01, seed=42)
train_dataset = dataset["train"]
test_dataset = dataset["test"]
```

### Training the Model

```python
from bertdistiller import MiniLMTrainer, MiniLMTrainingArguments, create_student

# Load teacher model and prepare datasets
from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset

# 1. Create student configuration
args = MiniLMTrainingArguments(
    # Distillation parameters
    teacher_layer=12,                # Which teacher layer to transfer from
    student_layer=6,                 # Number of layers in student model
    student_hidden_size=384,         # Hidden size of student model
    student_attention_heads=12,      # Number of attention heads in student
    num_relation_heads=48,           # Number of relation heads for distillation
    relations={                      # Which attention relations to use
        (1, 1): 1.0,                 # Q-Q relation with weight 1.0
        (2, 2): 1.0,                 # K-K relation with weight 1.0
        (3, 3): 1.0,                 # V-V relation with weight 1.0
    },
    
    # Regular training arguments
    output_dir="./output",
    per_device_train_batch_size=256,
    learning_rate=6e-4,
    max_steps=400_000,
    # ... other training arguments
)

# 2. Create the student model
teacher_model_name = "google-bert/bert-base-uncased"
teacher = AutoModel.from_pretrained(teacher_model_name)
student = create_student(
    teacher_model_name,
    args,
    use_teacher_weights=True,  # Initialize with teacher weights for better performance
    # Setting to False will initialize with random weights
)

# 3. Create the trainer and start training
trainer = MiniLMTrainer(
    args=args,
    teacher_model=teacher,
    model=student,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
)

# 4. Train the model
trainer.train()

# 5. Save the distilled model
student.save_pretrained("./bert-base-uncased-L6-H384")
```

## Available Distillation Methods

### MiniLMv2

MiniLMv2 transfers knowledge from a teacher model to a student model by matching the self-attention relation patterns. Unlike previous approaches, MiniLMv2:

1. Uses multi-head **self-attention relations** computed by scaled dot-product between pairs of query, key, and value vectors
2. Does not require the student to have the same number of attention heads as the teacher
3. Provides more fine-grained knowledge transfer through relation heads
4. Strategically selects which teacher layer to distill from (typically an upper-middle layer for large models)

The implementation supports three primary relation types:
- Query-Query (Q-Q) relations
- Key-Key (K-K) relations
- Value-Value (V-V) relations

### Teacher Weight Inheritance

BertDistiller supports initializing the student model with the teacher's weights, which can improve distillation performance. The `create_student` function handles weight transfer, including cases where the student has different hidden dimensions and fewer layers than the teacher.

## Configuration Options

### MiniLMTrainingArguments

This class extends Hugging Face's `TrainingArguments` with MiniLMv2-specific parameters:

| Parameter | Description |
|-----------|-------------|
| `teacher_layer` | Which layer of the teacher model to distill from (1-indexed) |
| `student_layer` | Number of layers in the student model |
| `student_hidden_size` | Hidden dimension size of the student model |
| `student_attention_heads` | Number of attention heads in the student model |
| `num_relation_heads` | Number of relation heads for distillation (more heads = more fine-grained knowledge) |
| `relations` | Dictionary mapping relation types to weights, e.g., `{(1,1): 1.0}` for Q-Q relation |

## Evaluation

BertDistiller provides utilities for evaluating distilled models on common benchmarks like GLUE:

```python
from bertdistiller.evaluation import evaluate

# Evaluate a model on all GLUE tasks with different hyperparameters
evaluate(
    model_name_or_path="microsoft/MiniLM-L12-H384-uncased", # or path to your distilled model
    tasks=["mnli", "qnli", "qqp", "rte", "sst2", "mrpc", "cola", "stsb"],
    learning_rate=[1e-5, 2e-5, 3e-5],
    epochs=[3, 5, 10],
)

# Evaluate on specific tasks with custom parameters
evaluate(
    model_name_or_path="microsoft/MiniLM-L12-H384-uncased",
    tasks=["sst2", "mrpc"],  # Subset of tasks
    learning_rate=3e-5,    # Specific learning rate
    epochs=3,              # Specific number of epochs
    per_device_train_batch_size=16,
    cache_dir=".cache",
)
```
The evaluation utility:
- Runs fine-tuning on GLUE tasks with specified hyperparameters
- Organizes results by task and hyperparameter combination
- Displays training progress in real-time
- Saves results in a structured format for easy comparison

Results are saved in your specified output directory with the following structure:
```text
evaluation_results/
└── model_name/
    ├── task1/
    │   ├── epochs_learning_rate/
    │   │   └── (evaluation results)
    │   └── ...
    ├── task2/
    │   └── ...
    └── ...
```
This evaluation helps compare the performance of distilled models against other models or their teacher models.

### Create Evaluation Summary
After running evaluations, you can generate a summary table to easily compare model performance across tasks:
```python
from bertdistiller.evaluation import create_summary_table

# Create summary table from evaluation results
summary = create_summary_table(
    results_dir="./evaluation_results",
    save=True,
    include_average=True,
)

print(summary)
```
The summary table displays the best scores for each model across all tasks, showing:
- Model names as rows
- GLUE tasks as columns
- Optional "Avg" column with mean performance across tasks
- All scores displayed as percentages rounded to 2 decimal places

## Recommendations

- For large-size teachers (24 layers), transferring from an upper-middle layer (e.g., layer 21 for BERT-large) typically works best
- For base-size teachers (12 layers), using the last layer usually performs better
- Using more relation heads (e.g., 48 for base models, 64 for large models) generally improves performance
- The student can use a standard number of attention heads (e.g., 12) regardless of relation heads

## Future Plans

- **Task-specific distillation**: Implement distillation methods for fine-tuned models
- **Additional distillation techniques**: Support for other knowledge distillation methods beyond MiniLMv2
- **More model architectures**: Extend support to other transformer architectures (ModernBERT)
- **Evaluation utilities**: Tools for evaluating distilled models on common benchmarks

## Acknowledgements

- This implementation is inspired by [minilmv2.bb](https://github.com/bloomberg/minilmv2.bb) by Bloomberg
- Built using the [Hugging Face Transformers](https://github.com/huggingface/transformers) library

## Citation

If you use this implementation in your research, please cite the original paper:

```bibtex
@article{wang2020minilmv2,
  title={MINILMv2: Multi-Head Self-Attention Relation Distillation for Compressing Pretrained Transformers},
  author={Wang, Wenhui and Bao, Hangbo and Huang, Shaohan and Dong, Li and Wei, Furu},
  journal={arXiv preprint arXiv:2012.15828},
  year={2020}
}
```

## License

This project is licensed under the Apache License Version 2.0 - see the LICENSE file for details.