from transformers import AutoConfig, AutoModel
from transformers.models.bert import BertModel

from .args import MiniLMTrainingArguments


def create_student(
    teacher_model_name_or_path: str,
    args: MiniLMTrainingArguments,
    use_teacher_weights: bool = False,
    cache_dir: str = None,
) -> BertModel:
    """Create the student model with smaller configuration.

    This function creates a student model that can be either initialized with random
    weights or with weights from the teacher model. When using teacher weights, it
    handles both traditional BERT and ModernBERT architectures appropriately.

    Args:
        teacher_model_name_or_path: The teacher model to use as a base.
        args: Training arguments containing student model configuration
        use_teacher_weights: Whether to initialize student with teacher weights
        cache_dir: Directory for caching model configurations

    Returns:
        The initialized student model
    """
    # Load the teacher model
    teacher_config = AutoConfig.from_pretrained(
        teacher_model_name_or_path, cache_dir=cache_dir
    )
    # Create the base student configuration
    student_config = AutoConfig.from_pretrained(
        teacher_model_name_or_path, cache_dir=cache_dir
    )
    student_config.hidden_size = args.student_hidden_size
    student_config.num_hidden_layers = args.student_layer
    student_config.num_attention_heads = args.student_attention_heads

    # Adjust intermediate_size based on model type
    if "ModernBert" in teacher_config.architectures[0]:
        raise NotImplementedError("ModernBERT distillation is not yet supported.")
    else:
        # Original BERT: intermediate_size = 4 * hidden_size
        student_config.intermediate_size = 4 * args.student_hidden_size

    # Create the student model
    student_model = AutoModel.from_config(student_config)

    if use_teacher_weights:
        teacher_model = AutoModel.from_pretrained(
            teacher_model_name_or_path, cache_dir=cache_dir
        )
        # Determine if we're using ModernBERT or traditional BERT
        is_modern_bert = (
            "ModernBertForMaskedLM" == teacher_model.config.architectures[0]
        )

        if is_modern_bert:
            raise NotImplementedError("ModernBERT distillation is not yet supported.")
        else:
            _init_bert_student(teacher_model, student_model, args)

    return student_model


def _init_bert_student(teacher_model, student_model, args):
    """Initialize traditional BERT student model with teacher weights."""
    # Handle embeddings
    if teacher_model.config.hidden_size == args.student_hidden_size:
        # Direct copy for same size
        student_model.embeddings.load_state_dict(teacher_model.embeddings.state_dict())
    else:
        # For different sizes, properly access the embedding layers
        student_model.embeddings.word_embeddings.weight.data.copy_(
            teacher_model.embeddings.word_embeddings.weight.data[
                :, : args.student_hidden_size
            ]
        )
        student_model.embeddings.position_embeddings.weight.data.copy_(
            teacher_model.embeddings.position_embeddings.weight.data[
                :, : args.student_hidden_size
            ]
        )
        student_model.embeddings.token_type_embeddings.weight.data.copy_(
            teacher_model.embeddings.token_type_embeddings.weight.data[
                :, : args.student_hidden_size
            ]
        )

    # Handle encoder layers
    layers_per_block = len(teacher_model.encoder.layer) // args.student_layer

    for student_idx in range(args.student_layer):
        teacher_idx = student_idx * layers_per_block
        teacher_layer = teacher_model.encoder.layer[teacher_idx]
        student_layer = student_model.encoder.layer[student_idx]

        # Copy attention weights
        if teacher_model.config.hidden_size == args.student_hidden_size:
            student_layer.attention.self.query.load_state_dict(
                teacher_layer.attention.self.query.state_dict()
            )
            student_layer.attention.self.key.load_state_dict(
                teacher_layer.attention.self.key.state_dict()
            )
            student_layer.attention.self.value.load_state_dict(
                teacher_layer.attention.self.value.state_dict()
            )
        else:
            # For different sizes, copy partial weights
            for qkv in ["query", "key", "value"]:
                teacher_proj = getattr(teacher_layer.attention.self, qkv)
                student_proj = getattr(student_layer.attention.self, qkv)
                student_proj.weight.data.copy_(
                    teacher_proj.weight.data[
                        : args.student_hidden_size, : args.student_hidden_size
                    ]
                )
                student_proj.bias.data.copy_(
                    teacher_proj.bias.data[: args.student_hidden_size]
                )

        # Copy output projection and LayerNorm
        if teacher_model.config.hidden_size == args.student_hidden_size:
            student_layer.attention.output.dense.load_state_dict(
                teacher_layer.attention.output.dense.state_dict()
            )
            student_layer.attention.output.LayerNorm.load_state_dict(
                teacher_layer.attention.output.LayerNorm.state_dict()
            )
        else:
            student_layer.attention.output.dense.weight.data.copy_(
                teacher_layer.attention.output.dense.weight.data[
                    : args.student_hidden_size, : args.student_hidden_size
                ]
            )
            student_layer.attention.output.dense.bias.data.copy_(
                teacher_layer.attention.output.dense.bias.data[
                    : args.student_hidden_size
                ]
            )
            # Copy LayerNorm parameters
            student_layer.attention.output.LayerNorm.weight.data.copy_(
                teacher_layer.attention.output.LayerNorm.weight.data[
                    : args.student_hidden_size
                ]
            )
            student_layer.attention.output.LayerNorm.bias.data.copy_(
                teacher_layer.attention.output.LayerNorm.bias.data[
                    : args.student_hidden_size
                ]
            )

        # Copy FFN weights
        if teacher_model.config.hidden_size == args.student_hidden_size:
            student_layer.intermediate.dense.load_state_dict(
                teacher_layer.intermediate.dense.state_dict()
            )
            student_layer.output.dense.load_state_dict(
                teacher_layer.output.dense.state_dict()
            )
        else:
            intermediate_size = student_layer.intermediate.dense.out_features
            student_layer.intermediate.dense.weight.data.copy_(
                teacher_layer.intermediate.dense.weight.data[
                    :intermediate_size, : args.student_hidden_size
                ]
            )
            student_layer.intermediate.dense.bias.data.copy_(
                teacher_layer.intermediate.dense.bias.data[:intermediate_size]
            )
            student_layer.output.dense.weight.data.copy_(
                teacher_layer.output.dense.weight.data[
                    : args.student_hidden_size, :intermediate_size
                ]
            )
            student_layer.output.dense.bias.data.copy_(
                teacher_layer.output.dense.bias.data[: args.student_hidden_size]
            )
            # Copy the output LayerNorm parameters
            student_layer.output.LayerNorm.weight.data.copy_(
                teacher_layer.output.LayerNorm.weight.data[: args.student_hidden_size]
            )
            student_layer.output.LayerNorm.bias.data.copy_(
                teacher_layer.output.LayerNorm.bias.data[: args.student_hidden_size]
            )
