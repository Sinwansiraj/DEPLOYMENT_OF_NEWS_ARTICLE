"""
Inference module for News Article Categorization.
Loads model from local path or S3 and returns predictions with confidence scores.
"""

import logging
import os
import tarfile
from typing import Dict, List

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config import MODEL_CONFIG, LABEL_MAP, AWS_CONFIG
from src.aws_utils import download_model_from_s3

logger = logging.getLogger(__name__)

_MODEL = None
_TOKENIZER = None


def _extract_archive(archive_path: str, extract_dir: str) -> None:
    """
    Extract a .tar.gz model archive.

    Args:
        archive_path: Path to .tar.gz file.
        extract_dir: Destination directory.
    """
    logger.info("Extracting %s → %s", archive_path, extract_dir)
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(extract_dir)


def load_model(model_path: str = None) -> None:
    """
    Load tokenizer and model into module-level globals.
    Downloads from S3 if local path not found.

    Args:
        model_path: Optional local path to model directory.
                    Falls back to S3 download if None or missing.
    """
    global _MODEL, _TOKENIZER

    if _MODEL is not None and _TOKENIZER is not None:
        logger.debug("Model already loaded; skipping reload.")
        return

    if model_path is None:
        model_path = os.path.join(MODEL_CONFIG.local_model_dir, "final_model")

    if not os.path.isdir(model_path):
        logger.info("Model not found locally. Downloading from S3...")
        archive_path = os.path.join(
            MODEL_CONFIG.local_model_dir, "model.tar.gz"
        )
        os.makedirs(MODEL_CONFIG.local_model_dir, exist_ok=True)
        download_model_from_s3(archive_path)
        _extract_archive(archive_path, MODEL_CONFIG.local_model_dir)

    logger.info("Loading tokenizer from %s", model_path)
    _TOKENIZER = AutoTokenizer.from_pretrained(model_path)

    logger.info("Loading model from %s", model_path)
    _MODEL = AutoModelForSequenceClassification.from_pretrained(model_path)
    _MODEL.eval()

    if torch.cuda.is_available():
        _MODEL = _MODEL.cuda()
        logger.info("Model loaded on GPU.")
    else:
        logger.info("Model loaded on CPU.")


def predict(text: str) -> Dict:
    """
    Run inference on a single article text.

    Args:
        text: Raw article text (title + body or combined).

    Returns:
        Dict with keys:
            - 'predicted_label': human-readable category string
            - 'predicted_id': integer label index
            - 'confidence': float confidence of top prediction
            - 'all_scores': list of dicts [{label, score}, ...]
    """
    if _MODEL is None or _TOKENIZER is None:
        raise RuntimeError("Model not loaded. Call load_model() first.")

    if not text or not text.strip():
        raise ValueError("Input text cannot be empty.")

    inputs = _TOKENIZER(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=MODEL_CONFIG.max_length,
    )

    device = next(_MODEL.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = _MODEL(**inputs)

    probs = torch.softmax(outputs.logits, dim=-1).squeeze().cpu().tolist()

    if isinstance(probs, float):          # single-label edge case
        probs = [probs]

    predicted_id = int(torch.argmax(torch.tensor(probs)).item())
    predicted_label = LABEL_MAP.get(predicted_id, "Unknown")
    confidence = probs[predicted_id]

    all_scores = [
        {"label": LABEL_MAP[i], "score": round(probs[i], 4)}
        for i in range(len(probs))
    ]
    all_scores.sort(key=lambda x: x["score"], reverse=True)

    return {
        "predicted_label": predicted_label,
        "predicted_id": predicted_id,
        "confidence": round(confidence, 4),
        "all_scores": all_scores,
    }


def batch_predict(texts: List[str]) -> List[Dict]:
    """
    Run inference on a list of article texts.

    Args:
        texts: List of article strings.

    Returns:
        List of prediction dicts (same structure as predict()).
    """
    return [predict(t) for t in texts]