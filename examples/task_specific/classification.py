from typing import Dict

import evaluate
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
)

from bertdistiller.task_specific import (
    DistillationTrainer,
    DistillationTrainingArguments,
    process_dataset,
)


def main():
    dataset_id = "glue"
    dataset_config = "sst2"

    student_id = "google/bert_uncased_L-2_H-128_A-2"
    teacher_id = "textattack/bert-base-uncased-SST-2"

    max_seq_len = 128
    distilled_model_name = "tiny-bert-sst2-distilled"

    teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_id)

    dataset = load_dataset(dataset_id, dataset_config, cache_dir=".cache")

    tokenized_dataset = dataset.map(
        lambda batch: process_dataset(
            batch,
            tokenizer=teacher_tokenizer,
            column="sentence",
            max_seq_len=max_seq_len,
        ),
        batched=True,
        remove_columns=["sentence"],
    )

    labels = tokenized_dataset["train"].features["label"].names
    num_labels = len(labels)
    label2id, id2label = dict(), dict()

    for i, label in enumerate(labels):
        label2id[label] = str(i)
        id2label[str(i)] = label

    args = DistillationTrainingArguments(
        # Distillation parameters
        alpha=0.5,
        temperature=2.0,
        # Training parameters
        output_dir=f"./results/task_specific/{distilled_model_name}",
        num_train_epochs=3,
        per_device_train_batch_size=128,
        per_device_eval_batch_size=128,
        # fp16=True, # if you have a GPU
        learning_rate=6e-5,
        seed=42,
        logging_dir="./logs",
        logging_strategy="epoch",
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
    )

    data_collator = DataCollatorWithPadding(tokenizer=teacher_tokenizer)

    teacher_model = AutoModelForSequenceClassification.from_pretrained(
        teacher_id,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        cache_dir=".cache",
    )

    student_model = AutoModelForSequenceClassification.from_pretrained(
        student_id,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        cache_dir=".cache",
    )

    accuracy_metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred) -> Dict[str, float]:
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return accuracy_metric.compute(predictions=predictions, references=labels)

    trainer = DistillationTrainer(
        args=args,
        model=student_model,
        teacher_model=teacher_model,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        processing_class=teacher_tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()


if __name__ == "__main__":
    main()
