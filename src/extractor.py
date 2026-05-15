"""
extractor.py — FLAN-T5 + LoRA clause information extractor for ClauseWise.

Loads the fine-tuned model from HuggingFace Hub and exposes a single
`extract_clause(category, text)` function that returns an extracted string
for the 8 free-text CUAD categories.
"""

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Model config
MODEL_ID = "ClauseWise/flan-t5-cuad-clause-extractor-lora"
MAX_INPUT_LENGTH = 512
MAX_OUTPUT_LENGTH = 250

# Module-level state (loaded once on first use)
_tokenizer: AutoTokenizer | None = None
_model: AutoModelForSeq2SeqLM | None = None
_device: torch.device | None = None


def _load_model() -> None:
    """Load tokenizer and model into module-level variables."""
    global _tokenizer, _model, _device
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    _model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID).to(_device)
    _model.eval()


def _ensure_loaded() -> None:
    if _model is None:
        _load_model()


# Public API
def extract_clause(category: str, text: str) -> str:
    """
    Extract information for a given category from a clause.

    Mirrors the prompt format used during fine-tuning:
        "Extract <category> from the following clause:\\n<text>"

    Args:
        category: One of the 8 extraction categories, e.g. "Governing Law",
                  "Parties", "Agreement Date". Also accepts arbitrary questions
                  since the underlying model is FLAN-T5.
        text:     Raw clause text.

    Returns:
        Extracted string. Empty string if the model produces nothing.
    """
    _ensure_loaded()

    prompt = f"Extract {category} from the following clause:\n{text}"

    inputs = _tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_LENGTH,
    ).to(_device)

    with torch.no_grad():
        outputs = _model.generate(**inputs, max_length=MAX_OUTPUT_LENGTH)

    return _tokenizer.decode(outputs[0], skip_special_tokens=True)


def extract_clause_qa(question: str, context: str) -> str:
    """
    General question-answering extraction over a context block.

    Used by the pipeline for queries like "what", "when", "who", "how much"
    where the question is the user's raw input and the context is the
    assembled clause text.

    Prompt format:
        "Question: <question>\\nContext: <context>\\nAnswer:"

    Args:
        question: User's natural language question.
        context:  Assembled clause text (may span multiple clauses).

    Returns:
        Extracted answer string.
    """
    _ensure_loaded()

    prompt = f"Question: {question}\nContext: {context}\nAnswer:"

    inputs = _tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_LENGTH,
    ).to(_device)

    with torch.no_grad():
        outputs = _model.generate(**inputs, max_length=MAX_OUTPUT_LENGTH)

    return _tokenizer.decode(outputs[0], skip_special_tokens=True)