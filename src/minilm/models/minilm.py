import math
from abc import ABC, abstractmethod
from typing import Dict, Literal, Optional, Tuple

import torch

# from loguru import logger
from torch import Tensor, nn
from transformers import PreTrainedModel


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


class ModernBertAttentionExtractor(AttentionExtractor):
    """Handles attention extraction for ModernBERT models."""

    def get_attention_module(self, model: PreTrainedModel, layer_idx: int) -> nn.Module:
        return model.layers[layer_idx].attn

    def get_qkv_projections(
        self, attention_module: nn.Module
    ) -> Tuple[nn.Module, nn.Module, nn.Module]:
        # ModernBERT uses a combined QKV projection
        # We need to split it into individual Q, K, V components
        hidden_size = attention_module.Wqkv.out_features // 3

        class QKVSplit(nn.Module):
            def __init__(self, start_idx: int):
                super().__init__()
                self.start_idx = start_idx

            def forward(self, x: Tensor) -> Tensor:
                qkv = attention_module.Wqkv(x)
                return qkv[
                    :,
                    :,
                    self.start_idx * hidden_size : (self.start_idx + 1) * hidden_size,
                ]

        return (QKVSplit(0), QKVSplit(1), QKVSplit(2))  # Query  # Key  # Value


class MiniLM(nn.Module):
    """
    Distills knowledge from teacher model to student using MiniLMv2 technique.
    Supports both BERT and ModernBERT architectures.

    Attributes:
        teacher (nn.Module): Pre-trained teacher model
        student (nn.Module): Smaller student model to train
        teacher_layer (int): Layer index in teacher to distill from (1-based)
        student_layer (int): Layer index in student to distill to (1-based)
        relations (Dict[Tuple[int, int], float]): Dictionary mapping (teacher_relation, student_relation) pairs to weights
        num_relation_heads (int): Number of attention heads for relation projection
        kl_loss (nn.KLDivLoss): KL divergence loss module
    """

    def __init__(
        self,
        teacher: PreTrainedModel,
        student: PreTrainedModel,
        teacher_layer: int,
        student_layer: int,
        relations: Dict[Tuple[int, int], float],
        num_relation_heads: int,
        model_type: Literal["bert", "modernbert"] = "bert",
    ) -> None:
        """Initializes distiller with configuration."""
        super().__init__()

        # Select appropriate attention extractor based on model type
        if model_type.lower() == "bert":
            self.attention_extractor = BertAttentionExtractor()
        elif model_type.lower() == "modernbert":
            self.attention_extractor = ModernBertAttentionExtractor()
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        # Basic validation
        if hasattr(teacher, "encoder"):
            max_teacher_layers = len(teacher.encoder.layer)
        else:
            max_teacher_layers = len(teacher.layers)
        assert (
            teacher_layer <= max_teacher_layers
        ), f"Teacher layer {teacher_layer} exceeds available layers ({max_teacher_layers})"

        self.teacher = teacher
        self.student = student
        self.teacher_layer = teacher_layer - 1
        self.student_layer = student_layer - 1
        self.relations = relations
        self.num_relation_heads = num_relation_heads

        self.kl_loss = nn.KLDivLoss(reduction="sum")

        # Freeze teacher model
        self._freeze_teacher()

    def _freeze_teacher(self) -> None:
        """Freezes all teacher parameters."""
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher.eval()
        # logger.info("Teacher model is frozen and set to evaluation mode.")

    def _get_relation_vectors(
        self,
        attention_module: nn.Module,
        hidden_states: Tensor,
        head_size: int,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Extracts query/key/value vectors for relation projection.

        This method handles the extraction of Q/K/V vectors from both BERT and ModernBERT
        architectures. It uses the appropriate attention extractor to get the projection
        matrices and applies them to the hidden states.

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
        new_shape = (batch_size, seq_len, self.num_relation_heads, head_size)
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

            # Extract valid portions of relations
            t_rel = teacher_relations[batch_idx, :, :valid_len, :valid_len]
            s_rel = student_relations[batch_idx, :, :valid_len, :valid_len]

            # Compute distributions
            teacher_probs = torch.softmax(t_rel, dim=-1)
            student_probs = torch.log_softmax(s_rel, dim=-1)

            # Calculate KL divergence
            loss = self.kl_loss(
                student_probs.flatten(end_dim=-2), teacher_probs.flatten(end_dim=-2)
            )

            # Normalize by valid elements
            # We divide by (num_heads * valid_len) because each head processes valid_len tokens
            total_loss += loss / (self.num_relation_heads * valid_len)

        # Return average loss across batch
        return total_loss / batch_size

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
    ) -> Tuple[Tensor]:
        """Computes distillation loss for given inputs.

        This method performs a forward pass through both teacher and student models,
        computes relation matrices, and returns the distillation loss.

        Args:
            input_ids (Tensor): Token indices [batch_size, seq_len]
            attention_mask (Optional[Tensor]): Attention mask [batch_size, seq_len]
            token_type_ids (Optional[Tensor]): Token type IDs (only for BERT, not ModernBERT) [batch_size, seq_len]

        Returns:
            Tuple containing scalar loss tensor
        """
        # Prepare inputs based on model architecture
        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "output_hidden_states": True,
        }

        # Add token_type_ids only for BERT models
        if (
            isinstance(self.attention_extractor, BertAttentionExtractor)
            and token_type_ids is not None
        ):
            inputs["token_type_ids"] = token_type_ids

        # Forward pass through models
        with torch.no_grad():
            teacher_outputs = self.teacher(**inputs)
        student_outputs = self.student(**inputs)

        # Get hidden states from specified layers
        if hasattr(teacher_outputs, "hidden_states"):
            teacher_hidden = teacher_outputs.hidden_states[self.teacher_layer]
            student_hidden = student_outputs.hidden_states[self.student_layer]
        else:
            # Some models return hidden states directly
            teacher_hidden = teacher_outputs[self.teacher_layer]
            student_hidden = student_outputs[self.student_layer]

        # Get attention modules
        teacher_attn = self.attention_extractor.get_attention_module(
            self.teacher, self.teacher_layer
        )
        student_attn = self.attention_extractor.get_attention_module(
            self.student, self.student_layer
        )

        # Calculate head sizes
        teacher_head_size = self.teacher.config.hidden_size // self.num_relation_heads
        student_head_size = self.student.config.hidden_size // self.num_relation_heads

        # Get relation vectors
        t_query, t_key, t_value = self._get_relation_vectors(
            attention_module=teacher_attn,
            hidden_states=teacher_hidden,
            head_size=teacher_head_size,
        )
        s_query, s_key, s_value = self._get_relation_vectors(
            attention_module=student_attn,
            hidden_states=student_hidden,
            head_size=student_head_size,
        )

        # Loss Calculation
        total_loss = 0.0
        for (m, n), weight in self.relations.items():
            # Get corresponding vectors based on relation indices
            t_vectors = [t_query, t_key, t_value]
            s_vectors = [s_query, s_key, s_value]

            # Compute scaled dot-product attention
            teacher_sim = torch.matmul(
                t_vectors[m - 1],  # -1 because relations are 1-based
                t_vectors[n - 1].transpose(-1, -2),
            ) / math.sqrt(teacher_head_size)

            student_sim = torch.matmul(
                s_vectors[m - 1],
                s_vectors[n - 1].transpose(-1, -2),
            ) / math.sqrt(student_head_size)

            # Calculate KL divergence loss
            relation_loss = self._compute_kl_divergence(
                teacher_relations=teacher_sim,
                student_relations=student_sim,
                attention_mask=attention_mask,
            )

            total_loss += weight * relation_loss

        return (total_loss,)
