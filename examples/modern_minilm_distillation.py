from datetime import datetime

from datasets import load_dataset
from loguru import logger
from transformers import (
    AutoModel,
    AutoTokenizer,
    DataCollatorWithPadding,
)

from bertdistiller import (
    MiniLMTrainingArguments,
    ModernMiniLMTrainer,
    create_student,
    prepare_dataset,
)


def main():
    model_name = "answerdotai/ModernBERT-base"
    max_seq_len = 128
    seed = 42
    use_teacher_weights = False

    dt = datetime.now().strftime("%Y%m%d-%H%M%S")

    # 1. Load the dataset
    dataset_bc_id = "bookcorpus/bookcorpus"
    dataset_wiki_id = "legacy-datasets/wikipedia"

    dataset_bc = load_dataset(dataset_bc_id, split="train", cache_dir=".cache")
    dataset_wiki = load_dataset(
        dataset_wiki_id, "20220301.en", split="train", cache_dir=".cache"
    )

    # 2. Prepare the dataset
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=".cache")

    dataset = prepare_dataset(
        datasets=[dataset_bc, dataset_wiki],
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        tokenization_kwargs={"padding": "do_not_pad"},
    )
    dataset = dataset.train_test_split(test_size=0.01, seed=seed)
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]
    logger.info(f"Train dataset: {train_dataset}")
    logger.info(f"Test dataset: {test_dataset}")

    # 3. Define the training arguments
    short_model_name = (
        model_name if "/" not in model_name else model_name.split("/")[-1]
    )
    run_name = f"ModernMiniLM-L6-H384-{short_model_name}"
    args = MiniLMTrainingArguments(
        # Distillation arguments
        teacher_layer=22,
        student_layer=6,
        student_hidden_size=384,
        student_attention_heads=12,
        num_relation_heads=48,
        relations={
            (1, 1): 1.0,
            (2, 2): 1.0,
            (3, 3): 1.0,
        },
        # Training arguments
        output_dir=f"models/{run_name}_{dt}",
        per_device_train_batch_size=256,
        per_device_eval_batch_size=256,
        learning_rate=6e-4,
        weight_decay=0.01,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-6,
        max_steps=400_000,
        warmup_ratio=0.01,
        logging_steps=1_000,
        save_steps=10_000,
        seed=42,
        ddp_find_unused_parameters=True,
        save_total_limit=10,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        prediction_loss_only=True,
        greater_is_better=False,
        save_strategy="steps",
        eval_strategy="steps",
        eval_steps=10_000,
    )

    # 4. Define our models
    teacher = AutoModel.from_pretrained(model_name, cache_dir=".cache")
    student = create_student(
        model_name,
        args,
        use_teacher_weights=use_teacher_weights,
        cache_dir=".cache",
    )

    # 5. Create the trainer & start training
    trainer = ModernMiniLMTrainer(
        args=args,
        teacher_model=teacher,
        model=student,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=DataCollatorWithPadding(tokenizer, padding="longest"),
    )
    trainer.train()

    # 6. Save the final model
    final_output_dir = f"models/{run_name}_{dt}/final"
    student.save_pretrained(final_output_dir)


if __name__ == "__main__":
    main()
