# src/minilm/config/model_config.py

import argparse
import ast
from dataclasses import dataclass, field
from typing import Dict, Literal, Optional, Tuple


@dataclass
class ModelConfig:
    """Configuration for model architecture and distillation settings."""

    input_model_dir: str
    student_hidden_size: int
    student_num_layers: int
    student_attention_heads: int
    teacher_layer: int
    num_relation_heads: int
    cache_dir: Optional[str] = None
    checkpoint_dir: Optional[str] = None
    tokenizer_dir: Optional[str] = None
    minilm_relations: Dict[Tuple[int, int], float] = field(
        default_factory=lambda: {(1, 1): 1.0, (2, 2): 1.0, (3, 3): 1.0}
    )
    model_type: Literal["bert", "modernbert"] = "bert"

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "ModelConfig":
        """Creates config from parsed command-line arguments."""
        # Parse relation string into dictionary
        relations_str = getattr(
            args, "minilm_relations", "{(1, 1): 1, (2, 2): 1, (3, 3): 1}"
        )
        relations = ast.literal_eval(relations_str)

        return cls(
            input_model_dir=args.input_model_dir,
            student_hidden_size=args.student_hidden_size,
            student_num_layers=args.student_num_layers,
            student_attention_heads=args.student_attention_heads,
            teacher_layer=args.L,
            num_relation_heads=args.num_relation_heads,
            cache_dir=getattr(args, "cache_dir", None),
            checkpoint_dir=getattr(args, "checkpoint_dir", None),
            tokenizer_dir=getattr(args, "tokenizer_dir", None),
            minilm_relations=relations,
            model_type=args.model_type,
        )
