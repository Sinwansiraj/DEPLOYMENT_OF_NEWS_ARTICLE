"""
Training script for News Article Categorization.
Fine-tunes DistilBERT on AG News using Hugging Face Trainer API.

Usage:
    python -m src.train
    python -m src.train --subset 5000   # quick smoke test
"""

import argparse
import logging
import os
import tarfile

import numpy as np
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
import evaluate

from config import MODEL_CONFIG, LABEL_MAP
from src.data_preprocessing import load_and_prepare_dataset
from src.aws_utils import upload_model_to_s3

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

METRIC = evaluate.load("accuracy")


def compute_metrics(eval_pred) -> dict:
    """
    Compute accuracy for Trainer evaluation.

    Args:
        eval_pred: EvalPrediction namedtuple from Trainer.

    Returns:
        Dict with 'accuracy' key.
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return METRIC.compute(predictions=predictions, references=labels)


def archive_model(model_dir: str, output_path: str) -> str:
    """
    Compress model directory into a .tar.gz archive.

    Args:
        model_dir: Path to the saved model directory.
        output_path: Destination .tar.gz file path.

    Returns:
        Path to the created archive.
    """
    logger.info("Archiving model from %s → %s", model_dir, output_path)
    with tarfile.open(output_path, "w:gz") as tar:
        tar.add(model_dir, arcname=os.path.basename(model_dir))
    return output_path


def train(subset_size: int = None, upload_to_s3: bool = True) -> None:
    """
    Full training pipeline: load data → fine-tune → save → upload.

    Args:
        subset_size: Limit dataset rows for quick testing.
        upload_to_s3: Whether to upload trained model to S3 after training.
    """
    os.makedirs(MODEL_CONFIG.local_model_dir, exist_ok=True)

    logger.info("Loading tokenizer: %s", MODEL_CONFIG.model_name)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CONFIG.model_name)

    train_ds, eval_ds = load_and_prepare_dataset(tokenizer, subset_size)

    logger.info("Loading base model: %s", MODEL_CONFIG.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_CONFIG.model_name,
        num_labels=MODEL_CONFIG.num_labels,
        id2label=LABEL_MAP,
        label2id={v: k for k, v in LABEL_MAP.items()},
    )

    training_args = TrainingArguments(
        output_dir=MODEL_CONFIG.local_model_dir,
        num_train_epochs=MODEL_CONFIG.num_epochs,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        learning_rate=MODEL_CONFIG.learning_rate,
        weight_decay=MODEL_CONFIG.weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        warmup_steps=100,
        logging_steps=100,
        report_to="none",
        fp16=False,
        use_cpu=True,   # Set to False if GPU is available
        dataloader_num_workers=0,  # Set to >0 for faster data loading if supported
        gradient_accumulation_steps=4,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    logger.info("Starting training...")
    trainer.train()

    final_model_path = os.path.join(MODEL_CONFIG.local_model_dir, "final_model")
    logger.info("Saving model to %s", final_model_path)
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)

    eval_results = trainer.evaluate()
    logger.info("Final eval results: %s", eval_results)

    if upload_to_s3:
        archive_path = "model_artifacts/model.tar.gz"
        archive_model(final_model_path, archive_path)
        upload_model_to_s3(archive_path)
    else:
        logger.info("Skipping S3 upload (upload_to_s3=False).")

    logger.info("Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train News Categorizer")
    parser.add_argument(
        "--subset",
        type=int,
        default=None,
        help="Limit dataset rows (e.g. 5000 for quick test)",
    )
    parser.add_argument(
        "--no-s3",
        action="store_true",
        help="Skip S3 upload after training",
    )
    args = parser.parse_args()
    train(subset_size=args.subset, upload_to_s3=not args.no_s3)