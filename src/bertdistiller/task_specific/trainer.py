import torch
import torch.nn.functional as F
from torch import nn
from transformers import PreTrainedModel, Trainer, AutoTokenizer


class DistillationTrainer(Trainer):
    def __init__(self, *args, teacher_model: PreTrainedModel, **kwargs):
        super().__init__(*args, **kwargs)

        self.teacher = teacher_model
        # Move the teacher model to the same device as the student model
        self._move_model_to_device(self.teacher, self.model.device)
        self.teacher.eval()

        if self.args.ensure_same_tokenizers:
            teacher_tokenizer = AutoTokenizer.from_pretrained(self.teacher.name_or_path)
            student_tokenizer = AutoTokenizer.from_pretrained(self.model.name_or_path)

            sample_text = "This is a sample text to check if tokenizers are the same."

            assert teacher_tokenizer(sample_text) == student_tokenizer(
                sample_text
            ), "Tokenizers are not the same. Please ensure that the teacher and student models have compatible tokenizers."
        else:
            raise ValueError(
                "BertDistiller currently only supports the same tokenizer for teacher and student models."
            )

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        output_student = model(**inputs)
        student_loss = output_student.loss
        with torch.no_grad():
            output_teacher = self.teacher(**inputs)

        assert (
            output_student.logits.size() == output_teacher.logits.size()
        ), f"Output size mismatch: student {output_student.logits.size()} vs teacher {output_teacher.logits.size()}"

        loss_function = nn.KLDivLoss(reduction="batchmean")
        loss_logits = loss_function(
            F.log_softmax(output_student.logits / self.args.temperature, dim=-1),
            F.softmax(output_teacher.logits / self.args.temperature, dim=-1),
        ) * (self.args.temperature**2)
        # Weighted student loss
        loss = self.args.alpha * student_loss + (1.0 - self.args.alpha) * loss_logits
        return (loss, output_student) if return_outputs else loss
