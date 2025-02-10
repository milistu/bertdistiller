# src/minilm/trainer/distillation.py

import os
from dataclasses import dataclass
from typing import Optional

import torch
from datasets import Dataset
from loguru import logger
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    DataCollatorWithPadding,
    PreTrainedTokenizer,
    Trainer,
    TrainingArguments,
)

from ..config import DataArguments, ModelConfig
from ..models import MiniLM


@dataclass
class DistillationConfig:
    """Configuration for the distillation process.

    This class combines all necessary configurations and handles
    distributed training setup.

    Args:
        model_config: Configuration for teacher/student models
        data_args: Configuration for dataset processing
        training_args: HuggingFace training arguments
        local_rank: Process rank for distributed training (-1 for non-distributed)
    """

    model_config: ModelConfig
    data_args: DataArguments
    training_args: TrainingArguments
    local_rank: int = -1

    @property
    def is_distributed(self) -> bool:
        """Whether we're running in distributed mode."""
        return int(os.environ.get("WORLD_SIZE", "1")) > 1

    @property
    def is_primary(self) -> bool:
        """Whether this is the primary process."""
        return self.local_rank in [-1, 0]


class DistillationTrainer:
    """Handles the model distillation process.

    This class encapsulates the logic for loading models, preparing data,
    and running the distillation training process.
    """

    def __init__(self, config: DistillationConfig):
        """Initialize the trainer with configuration."""
        self.config = config

        # Initialize components
        self.tokenizer = self._create_tokenizer()
        self.teacher = self._create_teacher()
        self.student = self._create_student()
        self.distiller = self._create_distiller()

    def _create_tokenizer(self) -> PreTrainedTokenizer:
        """Create and configure the tokenizer."""
        tokenizer_path = (
            self.config.model_config.tokenizer_dir
            or self.config.model_config.input_model_dir
        )

        return AutoTokenizer.from_pretrained(
            tokenizer_path,
            use_fast=True,
            max_length=self.config.data_args.max_seq_len,
            cache_dir=self.config.model_config.cache_dir,
        )

    def _create_teacher(self) -> AutoModel:
        """Load the teacher model."""
        logger.info("Loading teacher model...")
        return AutoModel.from_pretrained(
            self.config.model_config.input_model_dir,
            cache_dir=self.config.model_config.cache_dir,
        )

    def _create_student(self) -> AutoModel:
        """Create the student model with smaller configuration."""
        logger.info("Creating student model...")

        # Get base config from teacher and modify for student
        student_config = AutoConfig.from_pretrained(
            self.config.model_config.input_model_dir,
            cache_dir=self.config.model_config.cache_dir,
        )
        student_config.hidden_size = self.config.model_config.student_hidden_size
        student_config.num_hidden_layers = self.config.model_config.student_num_layers
        student_config.num_attention_heads = (
            self.config.model_config.student_attention_heads
        )

        logger.info(f"Student configuration:\n{student_config}")
        return AutoModel.from_config(student_config)

    def _create_distiller(self) -> MiniLM:
        """Create the MiniLM distillation model."""
        logger.info("Initializing MiniLM distiller...")

        distiller = MiniLM(
            teacher=self.teacher,
            student=self.student,
            teacher_layer=self.config.model_config.teacher_layer,
            student_layer=self.config.model_config.student_num_layers,
            relations=self.config.model_config.minilm_relations,
            num_relation_heads=self.config.model_config.num_relation_heads,
            model_type=self.config.model_config.model_type,
        )

        # Load checkpoint if provided
        if self.config.model_config.checkpoint_dir:
            logger.info("Loading from checkpoint...")
            checkpoint_path = os.path.join(
                self.config.model_config.checkpoint_dir, "pytorch_model.bin"
            )
            distiller.load_state_dict(torch.load(checkpoint_path))

        return distiller

    def train(
        self, train_dataset: Dataset, eval_dataset: Optional[Dataset] = None
    ) -> None:
        """Run the distillation training process.

        Args:
            train_dataset: Dataset for training
            eval_dataset: Optional dataset for evaluation
        """
        if self.config.is_distributed:
            logger.info("Running in distributed mode")
            os.environ["TOKENIZERS_PARALLELISM"] = "false"

        # Create trainer
        trainer = Trainer(
            model=self.distiller,
            args=self.config.training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=DataCollatorWithPadding(self.tokenizer, padding="longest"),
        )

        # Start training
        logger.info("Starting training...")
        trainer.train(resume_from_checkpoint=self.config.model_config.checkpoint_dir)
        logger.info("Training completed!")
