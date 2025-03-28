from typing import Dict, Optional

from transformers import PreTrainedTokenizer


def process_dataset(
    examples,
    tokenizer: PreTrainedTokenizer,
    column: str,
    max_seq_len: int,
    tokenization_kwargs: Optional[Dict] = None,
):
    tokenized_inputs = tokenizer(
        examples[column],
        truncation=True,
        max_length=max_seq_len,
        **(tokenization_kwargs or {})
    )
    return tokenized_inputs
