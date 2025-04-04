# BertDistiller: Knowledge Distillation for BERT Models

[![HF Models](https://img.shields.io/badge/%F0%9F%A4%97-models-yellow)](https://huggingface.co/bertdistiller)
[![Python Versions](https://img.shields.io/pypi/pyversions/bertdistiller?logo=python&logoColor=white)](https://pypi.org/project/bertdistiller/)
[![License](https://img.shields.io/badge/License-Apache_2.0-green.svg)](LICENSE)

A flexible framework for distilling BERT models using various distillation techniques, built on the Hugging Face Transformers library.

Currently implements:
- [MiniLMv2](https://arxiv.org/abs/2012.15828): Multi-Head Self-Attention Relation Distillation for compressing pretrained Transformers.

## Overview

BertDistiller enables knowledge distillation of BERT models using the MiniLMv2 technique - a task-agnostic approach that compresses large transformer models into smaller, faster models while maintaining comparable performance.

Key features:
- **Built on Hugging Face Transformers**: Seamless integration with the transformers ecosystem
- **Task-agnostic distillation**: Compress models without task-specific fine-tuning
- **Flexible architecture**: Configure student models with different layer counts and dimensions
- **Teacher weight inheritance**: Option to initialize student with teacher weights

## Experimental Results

The following table compares our implementation's results with Microsoft's original MiniLM implementations on the GLUE benchmark:

| Model | STSB | RTE | CoLA | QQP | SST-2 | MNLI | QNLI | MRPC | Avg |
|-------|------|-----|------|-----|-------|------|------|------|-----|
| MiniLM-L6-H768-distilled-from-BERT-Base | 88.66 | 67.11 | 72.90 | 87.18 | 91.55 | 83.58 | 90.20 | 89.17 | 83.79 |
| MiniLM-L6-H384-distilled-from-BERT-Base | 87.33 | 64.74 | 66.63 | 85.72 | 90.58 | 81.85 | 89.55 | 88.00 | 81.80 |
| **Our Model (L6-H384)** | 85.29 | 59.81 | 70.04 | 85.22 | 90.62 | 81.03 | 87.69 | 86.66 | 80.80 |

With just a 1% difference in average score, our model was trained with a maximum sequence length of 128 tokens (vs 512 in the original paper) and was distilled on a single RTX A6000 GPU, demonstrating the efficiency and accessibility of our approach. 


## Installation

```bash
pip install bertdistiller
```

## Quick Start

See the [examples/minilm_distillation.py](examples/minilm_distillation.py) for a complete working example. Here's a simplified version:

```python
from bertdistiller import MiniLMTrainer, MiniLMTrainingArguments, create_student
from transformers import AutoModel, DataCollatorWithPadding

# 1. Create configuration
args = MiniLMTrainingArguments(
    teacher_layer=12,                # Which teacher layer to transfer from
    student_layer=6,                 # Number of layers in student model
    student_hidden_size=384,         # Hidden size of student model
    num_relation_heads=48,           # Number of relation heads for distillation
    relations={(1,1): 1.0, (2,2): 1.0, (3,3): 1.0},  # Q-Q, K-K, V-V relations
    
    # Training parameters
    output_dir="./output",
    per_device_train_batch_size=256,
    learning_rate=6e-4,
    max_steps=400_000,
)

# 2. Create models & trainer
teacher = AutoModel.from_pretrained("google-bert/bert-base-uncased")
student = create_student("google-bert/bert-base-uncased", args)

trainer = MiniLMTrainer(
    args=args,
    teacher_model=teacher,
    model=student,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
)

# 3. Train and save
trainer.train()
student.save_pretrained("./distilled-model")
```

## How MiniLMv2 Works

MiniLMv2 transfers knowledge using **self-attention relations** - interactions between query, key, and value vectors within transformer layers. The implementation:

1. Computes relation patterns using scaled dot-product between Q-Q, K-K, and V-V pairs
2. Creates flexible "relation heads" that don't require matching teacher/student attention head counts
3. Strategically selects which teacher layer to distill from (typically layer 12 for base models, an upper-middle layer for large models)

This approach provides more fine-grained knowledge transfer than traditional attention distillation methods.


## Evaluation

BertDistiller includes utilities to evaluate distilled models on GLUE benchmark tasks:

```python
from bertdistiller.evaluation import evaluate, create_summary_table

# Evaluate on GLUE tasks
evaluate(
    model_name_or_path="your-distilled-model",
    tasks=["mnli", "qnli", "qqp", "sst2"],
    learning_rate=[1e-5, 3e-5],
    epochs=[3, 5],
)

# Generate comparison table
summary = create_summary_table("./evaluation_results")
print(summary)
```


## Recommendations

- For base-size teachers (12 layers), use the last layer for distillation
- For large-size teachers (24 layers), use an upper-middle layer (e.g., layer 21)
- Using more relation heads (48+) generally improves performance
- Initialize with teacher weights when possible

## Acknowledgements & Citation
Built using [Hugging Face Transformers](https://github.com/huggingface/transformers) and inspired by [minilmv2.bb](https://github.com/bloomberg/minilmv2.bb) implementation and the original MiniLMv2 paper:


```bibtex
@article{wang2020minilmv2,
  title={MINILMv2: Multi-Head Self-Attention Relation Distillation for Compressing Pretrained Transformers},
  author={Wang, Wenhui and Bao, Hangbo and Huang, Shaohan and Dong, Li and Wei, Furu},
  journal={arXiv preprint arXiv:2012.15828},
  year={2020}
}
```

## License

Apache License Version 2.0