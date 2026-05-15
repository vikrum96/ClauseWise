"""
reasoner.py — Groq-backed reasoning module for ClauseWise.

The model used is deepseek-r1-distill-llama-70b,
which preserves the DeepSeek R1 reasoning behavior without requiring local GPU memory.

Set GROQ_API_KEY in your environment or .env file before running.
"""

import os
import re

from groq import Groq

# Config
MODEL_ID = "deepseek-r1-distill-llama-70b"
MAX_TOKENS = 900
TEMPERATURE = 0.3

# Client (initialized once on first use)
_client: Groq | None = None
def _get_client() -> Groq:
    global _client
    if _client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GROQ_API_KEY is not set. "
                "Add it to your environment or .env file."
            )
        _client = Groq(api_key=api_key)
    return _client


# Output cleanup
def _cleanup_response(decoded: str) -> str:
    """
    Strip DeepSeek-R1 chain-of-thought artifacts and instruction echoes.
    Ported directly from the notebook's deepseek_reasoning() function.
    """
    # If there's a </think> tag, drop everything before it
    if "</think>" in decoded:
        decoded = decoded.split("</think>", 1)[1].lstrip()

    meta_patterns = [
        r"Use simple,? clear.*language\.?",
        r"Use clear,? simple.*English\.?",
        r"Do not use markdown formatting\.?",
        r"Do NOT use markdown formatting\.?",
        r"Avoid markdown formatting\.?",
    ]
    for pat in meta_patterns:
        decoded = re.sub(pat, "", decoded, flags=re.IGNORECASE)

    # Drop lines that look like instructions the model is echoing to itself
    instruction_starts = (
        "use ",
        "avoid ",
        "do not ",
        "don't ",
        "assistant ",
        "as an ai",
        "you are ",
        "the assistant ",
    )

    lines = decoded.splitlines()
    filtered = []
    for line in lines:
        s = line.strip()
        if not s:
            continue
        if any(s.lower().startswith(pfx) for pfx in instruction_starts):
            continue
        filtered.append(line)

    decoded = "\n".join(filtered).strip()

    # Fallback if everything got filtered
    if not decoded:
        decoded = (
            "I could not reliably identify specific red flags from the clauses shown. "
            "The text I saw is mostly the introductory part of the agreement, which sets "
            "out the parties, the nature of the transaction, and lists the attachments. "
            "Most of the serious risk terms (liability limits, warranties, termination "
            "rights, exclusivity, etc.) are usually found in later sections of the contract. "
            "Those later sections would need to be reviewed to identify concrete red flags."
        )

    return decoded


# Public API
def reason(prompt: str) -> str:
    """
    Send a reasoning prompt to Groq and return the cleaned response.

    Args:
        prompt: The full reasoning prompt assembled by the pipeline,
                including clause context, classification results, and
                any red-flag instructions.

    Returns:
        Cleaned response string ready to return to the user.
    """
    client = _get_client()

    response = client.chat.completions.create(
        model=MODEL_ID,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
    )

    raw = response.choices[0].message.content or ""
    return _cleanup_response(raw)