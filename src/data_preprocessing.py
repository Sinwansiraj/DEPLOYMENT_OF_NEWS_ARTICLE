"""
Data preprocessing module for AG News dataset.
Handles loading, cleaning, and tokenization.
"""

import re
import logging
from typing import Dict, Tuple

import datasets
from transformers import AutoTokenizer

from config import MODEL_CONFIG, LABEL_MAP

logger = logging.getLogger(__name__)


def clean_text(text: str) -> str:
    """
    Clean raw article text.

    Args:
        text: Raw input string.

    Returns:
        Cleaned string with normalized whitespace and removed HTML.
    """
    if not isinstance(text, str):
        return ""
    text = re.sub(r"<.*?>", " ", text)          # strip HTML tags
    text = re.sub(r"&[a-z]+;", " ", text)       # strip HTML entities
    text = re.sub(r"http\S+|www\.\S+", " ", text)  # strip URLs
    text = re.sub(r"\s+", " ", text)             # collapse whitespace
    return text.strip()


def combine_title_description(example: Dict) -> Dict:
    """
    Merge title and description into a single `text` field.

    Args:
        example: A single dataset row with 'text' (title) and 'description'.

    Returns:
        Same row with added 'combined_text' and 0-indexed 'label'.
    """
    title = clean_text(example.get("text", ""))
    desc = clean_text(example.get("description", ""))
    example["combined_text"] = f"{title} [SEP] {desc}"
    example["label"] = example["label"] - 1   # AG News labels are 1-indexed
    return example


def tokenize_batch(batch: Dict, tokenizer: AutoTokenizer) -> Dict:
    """
    Tokenize a batch of combined texts.

    Args:
        batch: Batch dict containing 'combined_text'.
        tokenizer: Hugging Face tokenizer instance.

    Returns:
        Tokenized encodings dict.
    """
    return tokenizer(
        batch["combined_text"],
        padding="max_length",
        truncation=True,
        max_length=MODEL_CONFIG.max_length,
    )


def load_and_prepare_dataset(
    tokenizer: AutoTokenizer,
    subset_size: int = None,
) -> Tuple[datasets.Dataset, datasets.Dataset]:
    """
    Load AG News, preprocess, and tokenize.

    Args:
        tokenizer: Hugging Face tokenizer.
        subset_size: Optional integer to limit dataset size (for quick tests).

    Returns:
        Tuple of (train_dataset, eval_dataset).
    """
    logger.info("Loading AG News dataset...")
    raw = datasets.load_dataset("ag_news")

    train_ds = raw["train"]
    test_ds = raw["test"]

    if subset_size:
        train_ds = train_ds.select(range(min(subset_size, len(train_ds))))
        test_ds = test_ds.select(range(min(subset_size // 5, len(test_ds))))

    logger.info("Applying text cleaning and combining fields...")
    # AG News has 'text' (title+desc combined by HF) and 'label'
    # We clean and re-label to 0-indexed
    train_ds = train_ds.map(
        lambda ex: {
            "combined_text": clean_text(ex["text"]),
            "label": ex["label"],  # already 0-indexed in HF ag_news
        }
    )
    test_ds = test_ds.map(
        lambda ex: {
            "combined_text": clean_text(ex["text"]),
            "label": ex["label"],
        }
    )

    logger.info("Tokenizing datasets...")
    train_ds = train_ds.map(
        lambda b: tokenize_batch(b, tokenizer),
        batched=True,
        batch_size=256,
    )
    test_ds = test_ds.map(
        lambda b: tokenize_batch(b, tokenizer),
        batched=True,
        batch_size=256,
    )

    train_ds.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "label"],
    )
    test_ds.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "label"],
    )

    logger.info(
        "Dataset prepared. Train: %d | Eval: %d",
        len(train_ds),
        len(test_ds),
    )
    return train_ds, test_ds