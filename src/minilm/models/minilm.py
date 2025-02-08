import math
from typing import Dict, Tuple

import torch
from loguru import logger
from torch import Tensor, nn


class MiniLM(nn.Module):
    """Distills knowledge from teacher model to student using MiniLMv2 technique.

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
        teacher: nn.Module,
        student: nn.Module,
        teacher_layer: int,
        student_layer: int,
        relations: Dict[Tuple[int, int], float],
        num_relation_heads: int,
    ) -> None:
        """Initializes distiller with configuration."""
        super().__init__()
        assert teacher_layer <= len(
            teacher.encoder.layer
        ), "Teacher layer exceeds available layers"

        self.teacher = teacher
        self.student = student
        self.teacher_layer = teacher_layer
        self.student_layer = student_layer
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
        logger.info("Teacher model is frozen and set to evaluation mode.")

    def _get_relation_vectors(
        self,
        attention_module: nn.Module,
        hidden_states: Tensor,
        head_size: int,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Extracts query/key/value vectors for relation projection.

        Args:
            attention_module (nn.Module): Self-attention module from transformer layer
            hidden_states (Tensor): Output from previous layer
            head_size (int): Size per attention head

        Returns:
            Tuple of (query, key, value) tensors shaped
            [batch_size, num_heads, seq_len, head_size]
        """
        query = self._transpose_for_scores_relations(
            attention_module.query(hidden_states), head_size
        )
        key = self._transpose_for_scores_relations(
            attention_module.key(hidden_states), head_size
        )
        value = self._transpose_for_scores_relations(
            attention_module.value(hidden_states), head_size
        )
        return query, key, value

    def _transpose_for_scores_relations(self, x: Tensor, head_size: int) -> Tensor:
        """Reshapes tensor for multi-head relation processing.

        Args:
            x (Tensor): Input tensor [batch_size, seq_len, hidden_size]
            head_size (int): Target size per relation head

        Returns:
            Reshaped tensor [batch_size, num_heads, seq_len, head_size]
        """
        new_shape = (*x.shape[:-1], self.num_relation_heads, head_size)
        return x.view(new_shape).permute(0, 2, 1, 3)

    def _compute_kl_divergence(
        self,
        teacher_relations: Tensor,
        student_relations: Tensor,
        attention_mask: Tensor,
    ) -> Tensor:
        """Computes masked KL divergence between relation matrices.

        Args:
            teacher_relations (Tensor): Relation matrix from teacher
            student_relations (Tensor): Relation matrix from student
            attention_mask (Tensor): Binary mask for valid tokens

        Returns:
            Scalar loss value normalized by valid elements
        """
        batch_size, _, seq_len, _ = teacher_relations.size()
        seq_lens = attention_mask.sum(dim=1)  # [batch_size]

        total_loss = 0.0
        for batch_idx in range(batch_size):
            valid_len = seq_lens[batch_idx].item()
            valid_slice = slice(None, valid_len)

            # Extract valid portions
            t_rel = teacher_relations[batch_idx, :, valid_slice, valid_slice]
            s_rel = student_relations[batch_idx, :, valid_slice, valid_slice]

            # Compute distributions
            teacher_probs = torch.softmax(t_rel, dim=-1)
            student_probs = torch.log_softmax(s_rel, dim=-1)

            # Calculate KL divergence
            loss = self.kl_loss(
                student_probs.flatten(end_dim=-2), teacher_probs.flatten(end_dim=-2)
            )

            # Normalize by valid elements
            total_loss += loss / (self.num_relation_heads * valid_len)

        return total_loss / batch_size

    def forward(
        self,
        input_ids: Tensor,
        token_type_ids: Tensor,
        attention_mask: Tensor,
    ) -> Tuple[Tensor]:
        """Computes distillation loss for given inputs.

        Args:
            input_ids: Token indices [batch_size, seq_len]
            token_type_ids: Segment indices [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]

        Returns:
            Tuple containing scalar loss tensor
        """
        # Common input format for both models
        inputs = {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask,
            "output_hidden_states": True,
            "output_attentions": True,
        }

        # Model Forward Pass
        with torch.no_grad():
            teacher_outputs = self.teacher(**inputs)

        student_outputs = self.student(**inputs)

        # Hidden State Extraction
        teacher_hidden = teacher_outputs.hidden_states[self.teacher_layer - 1]
        student_hidden = student_outputs.hidden_states[self.student_layer - 1]

        # Relation Projections
        teacher_attn = self.teacher.encoder.layer[self.teacher_layer - 1].attention.self
        student_attn = self.student.encoder.layer[self.student_layer - 1].attention.self

        # Extract relation head size
        teacher_head_size = self.teacher.config.hidden_size // self.num_relation_heads
        student_head_size = self.student.config.hidden_size // self.num_relation_heads

        # Relation Vectors
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
            teacher_m_idx = m - 1
            teacher_n_idx = n - 1
            student_m_idx = m - 1
            student_n_idx = n - 1

            # Scaled Dot-Product Similarity
            teacher_sim = torch.matmul(
                [t_query, t_key, t_value][teacher_m_idx],
                [t_query, t_key, t_value][teacher_n_idx].transpose(-1, -2),
            ) / math.sqrt(t_query.size(-1))

            student_sim = torch.matmul(
                [s_query, s_key, s_value][student_m_idx],
                [s_query, s_key, s_value][student_n_idx].transpose(-1, -2),
            ) / math.sqrt(s_query.size(-1))

            # KL Divergence Loss
            relation_loss = self._compute_kl_divergence(
                teacher_relations=teacher_sim.detach(),
                student_relations=student_sim,
                attention_mask=attention_mask,
            )

            total_loss += weight * relation_loss

        return (total_loss,)
