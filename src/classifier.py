"""
classifier.py — LegalBERT multi-label clause classifier for ClauseWise.

Loads the fine-tuned model from HuggingFace Hub and exposes a single
`classify_clause(text)` function that returns a list of clause-type labels.
"""

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Model config
MODEL_ID = "ClauseWise/legalbert-clause-classifier"
DEFAULT_THRESHOLD = 0.5
MAX_LENGTH = 512

# Module-level state (loaded once on first use)
_tokenizer: AutoTokenizer | None = None
_model: AutoModelForSequenceClassification | None = None
_device: torch.device | None = None


def _load_model() -> None:
    """Load tokenizer and model into module-level variables."""
    global _tokenizer, _model, _device
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    _model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID,
        problem_type="multi_label_classification",
    ).to(_device)
    _model.eval()

def _ensure_loaded() -> None:
    if _model is None:
        _load_model()

# Public API
def classify_clause(text: str, threshold: float = DEFAULT_THRESHOLD) -> list[str]:
    """
    Classify a single clause and return a list of matching label strings.
    Args:
        text:      Raw clause text (will be truncated to 512 tokens).
        threshold: Sigmoid probability threshold for a label to be included.
                   Defaults to 0.5.
    Returns:
        List of label strings, e.g. ["Termination For Convenience", "Non-Compete"].
        Falls back to the single highest-probability label if nothing exceeds
        the threshold.
    """
    _ensure_loaded()

    inputs = _tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
        padding=True,
    ).to(_device)

    with torch.no_grad():
        logits = _model(**inputs).logits

    probs: np.ndarray = torch.sigmoid(logits).cpu().numpy()[0]

    labels = [
        _model.config.id2label[i]
        for i, prob in enumerate(probs)
        if prob > threshold
    ]

    # Guarantee at least one label so downstream code never gets an empty list
    if not labels:
        labels = [_model.config.id2label[int(np.argmax(probs))]]

    return labels


def classify_clauses_batch(texts: list[str], threshold: float = DEFAULT_THRESHOLD, batch_size: int = 16) -> list[list[str]]:
    """
    Classify a list of clauses in batches.
    Args:
        texts:      List of clause strings.
        threshold:  Per-label sigmoid threshold.
        batch_size: Number of clauses to process per forward pass.
    Returns:
        List of label-lists, one per input clause.
    """
    _ensure_loaded()

    all_results: list[list[str]] = []

    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]

        inputs = _tokenizer(
            batch_texts,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LENGTH,
            padding=True,
        ).to(_device)

        with torch.no_grad():
            logits = _model(**inputs).logits

        probs: np.ndarray = torch.sigmoid(logits).cpu().numpy()

        for row in probs:
            labels = [
                _model.config.id2label[i]
                for i, p in enumerate(row)
                if p > threshold
            ]
            if not labels:
                labels = [_model.config.id2label[int(np.argmax(row))]]
            all_results.append(labels)

    return all_results