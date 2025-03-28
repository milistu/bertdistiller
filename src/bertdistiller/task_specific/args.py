from transformers import TrainingArguments


class DistillationTrainingArguments(TrainingArguments):
    def __init__(
        self,
        *args,
        alpha: int = 0.5,
        temperature: float = 2.0,
        ensure_same_tokenizers: bool = True,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.alpha = alpha
        self.temperature = temperature
        self.ensure_same_tokenizers = ensure_same_tokenizers
