import math
from abc import ABC, abstractmethod
from typing import Tuple

import torch
from torch import Tensor, nn
from transformers import PreTrainedModel, Trainer


class AttentionExtractor(ABC):
    """Abstract base class for extracting attention components from different model architectures."""

    @abstractmethod
    def get_attention_module(self, model: PreTrainedModel, layer_idx: int) -> nn.Module:
        """Gets the attention module for a specific layer."""
        pass

    @abstractmethod
    def get_qkv_projections(
        self, attention_module: nn.Module
    ) -> Tuple[nn.Module, nn.Module, nn.Module]:
        """Extracts Q/K/V projection matrices from attention module."""
        pass


class BertAttentionExtractor(AttentionExtractor):
    """Handles attention extraction for traditional BERT models."""

    def get_attention_module(self, model: PreTrainedModel, layer_idx: int) -> nn.Module:
        return model.encoder.layer[layer_idx].attention.self

    def get_qkv_projections(
        self, attention_module: nn.Module
    ) -> Tuple[nn.Module, nn.Module, nn.Module]:
        return attention_module.query, attention_module.key, attention_module.value


class MiniLMTrainer(Trainer):
    def __init__(self, *args, teacher_model: PreTrainedModel, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher = teacher_model
        self._validate_params()
        # Move the teacher model to the same device as the student model
        self._move_model_to_device(self.teacher, self.model.device)
        self.teacher.eval()

        self.attention_extractor = self._get_attention_extractor()

        self.kl_loss = nn.KLDivLoss(reduction="sum")

    def _validate_params(self):
        if hasattr(self.teacher, "encoder"):
            max_teacher_layers = len(self.teacher.encoder.layer)
        else:
            max_teacher_layers = len(self.teacher.layers)

        assert (
            self.args.teacher_layer <= max_teacher_layers
        ), f"Teacher layer {self.args.teacher_layer} exceeds available layers ({max_teacher_layers})"

    def _get_attention_extractor(self):
        architecture = self.teacher.config.architectures[0]
        if "ModernBertForMaskedLM" == architecture:
            raise NotImplementedError("ModernBERT distillation is not yet supported.")
        elif "BertForMaskedLM" == architecture:
            return BertAttentionExtractor()
        else:
            raise ValueError(f"Unsupported teacher model architecture: {architecture}")

    def _get_relation_vectors(
        self,
        attention_module: nn.Module,
        hidden_states: Tensor,
        head_size: int,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Extracts query/key/value vectors for relation projection.

        This method handles the extraction of Q/K/V vectors.
        It uses the appropriate attention extractor to get the projection matrices and applies them to the hidden states.

        Args:
            attention_module: Self-attention module from transformer layer
            hidden_states: Output from previous layer [batch_size, seq_len, hidden_size]
            head_size: Size per attention head

        Returns:
            Tuple of (query, key, value) tensors shaped [batch_size, num_heads, seq_len, head_size]
        """
        # Get Q/K/V projections for the current architecture
        q_proj, k_proj, v_proj = self.attention_extractor.get_qkv_projections(
            attention_module
        )

        # Apply projections to hidden states
        query = q_proj(hidden_states)
        key = k_proj(hidden_states)
        value = v_proj(hidden_states)

        # Reshape for multi-head attention
        query = self._transpose_for_scores_relations(query, head_size)
        key = self._transpose_for_scores_relations(key, head_size)
        value = self._transpose_for_scores_relations(value, head_size)

        return query, key, value

    def _transpose_for_scores_relations(self, x: Tensor, head_size: int) -> Tensor:
        """Reshapes tensor for multi-head relation processing.

        Args:
            x (Tensor): Input tensor [batch_size, seq_len, hidden_size]
            head_size (int): Target size per relation head

        Returns:
            Reshaped tensor [batch_size, num_heads, seq_len, head_size]
        """
        batch_size, seq_len, _ = x.size()
        # Reshape to separate heads
        new_shape = (batch_size, seq_len, self.args.num_relation_heads, head_size)
        x = x.view(*new_shape)

        # Transpose to get [batch_size, num_heads, seq_len, head_size]
        return x.permute(0, 2, 1, 3)

    def _compute_kl_divergence(
        self,
        teacher_relations: Tensor,
        student_relations: Tensor,
        attention_mask: Tensor,
    ) -> Tensor:
        """Computes masked KL divergence between teacher and student relation matrices.

        This method calculates the KL divergence loss between teacher and student
        attention relation matrices, properly handling padding and normalization.

        Args:
            teacher_relations (Tensor): Relation matrix from teacher model
            student_relations (Tensor): Relation matrix from student model
            attention_mask (Tensor): Binary mask for valid tokens (1 for valid, 0 for padding)

        Returns:
            Scalar loss value normalized by number of valid elements
        """
        batch_size, num_heads, seq_len, _ = teacher_relations.size()

        # If no attention mask provided, assume all tokens are valid
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_len), dtype=torch.bool, device=teacher_relations.device
            )

        # Get valid sequence lengths for each batch
        seq_lens = attention_mask.sum(dim=1)  # [batch_size]

        total_loss = 0.0
        for batch_idx in range(batch_size):
            valid_len = seq_lens[batch_idx].item()

            # # Compute distributions
            teacher_probs = nn.Softmax(dim=-1)(
                teacher_relations[batch_idx, :, :valid_len, :valid_len]
            )
            student_probs = nn.functional.log_softmax(
                student_relations[batch_idx, :, :valid_len, :valid_len], dim=-1
            )

            # Calculate KL divergence
            loss = self.kl_loss(
                student_probs.flatten(end_dim=-2), teacher_probs.flatten(end_dim=-2)
            )

            # Normalize by valid elements
            # We divide by (num_heads * valid_len) because each head processes valid_len tokens
            total_loss += loss / (self.args.num_relation_heads * valid_len)

        # Return average loss across batch
        return total_loss / batch_size

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        loss = self.compute_loss(model, inputs)
        loss = loss.mean().detach()

        return (loss, None, None)

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        """Computes distillation loss for given inputs.

        This method performs a forward pass through both teacher and student models,
        computes relation matrices, and returns the distillation loss.

        Args:"
            model: The student model.
            inputs: The input data.
            return_outputs: Whether to return the outputs of the model.
            num_items_in_batch: The number of items in the batch.
        """

        # Forward pass through models
        with torch.no_grad():
            teacher_outputs = self.teacher(**inputs, output_hidden_states=True)
        student_outputs = model(**inputs, output_hidden_states=True)

        # Get hidden states from specified layers
        if hasattr(teacher_outputs, "hidden_states"):
            teacher_hidden = teacher_outputs.hidden_states[self.args.teacher_layer - 1]
            student_hidden = student_outputs.hidden_states[self.args.student_layer - 1]
        else:
            # Some models return hidden states directly
            teacher_hidden = teacher_outputs[self.args.teacher_layer - 1]
            student_hidden = student_outputs[self.args.student_layer - 1]

        # Calculate head sizes
        teacher_head_size = (
            self.teacher.config.hidden_size // self.args.num_relation_heads
        )
        student_head_size = model.config.hidden_size // self.args.num_relation_heads

        # Get relation vectors
        relation_vectors_T = self._get_relation_vectors(
            attention_module=self.attention_extractor.get_attention_module(
                self.teacher, self.args.teacher_layer - 1
            ),
            hidden_states=teacher_hidden,
            head_size=teacher_head_size,
        )
        relation_vectors_S = self._get_relation_vectors(
            attention_module=self.attention_extractor.get_attention_module(
                model, self.args.student_layer - 1
            ),
            hidden_states=student_hidden,
            head_size=student_head_size,
        )

        # Loss Calculation
        total_loss = 0.0
        for (m, n), weight in self.args.relations.items():
            # Compute scaled dot-product attention
            teacher_sim = torch.matmul(
                relation_vectors_T[m - 1],  # -1 because relations are 1-based
                relation_vectors_T[n - 1].transpose(-1, -2),
            ) / math.sqrt(teacher_head_size)

            student_sim = torch.matmul(
                relation_vectors_S[m - 1],
                relation_vectors_S[n - 1].transpose(-1, -2),
            ) / math.sqrt(student_head_size)

            # Calculate KL divergence loss
            relation_loss = self._compute_kl_divergence(
                teacher_relations=teacher_sim.detach(),
                student_relations=student_sim,
                attention_mask=inputs["attention_mask"],
            )

            total_loss += weight * relation_loss

        return (total_loss,) if return_outputs else total_loss
