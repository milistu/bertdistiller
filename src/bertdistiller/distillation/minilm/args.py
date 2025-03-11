from typing import Dict, Tuple

from transformers import TrainingArguments


class MiniLMTrainingArguments(TrainingArguments):
    def __init__(
        self,
        *args,
        teacher_layer: int,
        student_layer: int,
        student_hidden_size: int,
        student_attention_heads: int,
        num_relation_heads: int,
        relations: Dict[Tuple[int, int], float] = {
            (1, 1): 1.0,
            (2, 2): 1.0,
            (3, 3): 1.0,
        },
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.teacher_layer = teacher_layer
        self.student_layer = student_layer
        self.student_hidden_size = student_hidden_size
        self.student_attention_heads = student_attention_heads
        self.relations = relations
        self.num_relation_heads = num_relation_heads
