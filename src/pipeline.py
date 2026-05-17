"""
pipeline.py — Core query routing and inference pipeline for ClauseWise.

Contains route_user_query() and all helper functions ported directly from
the notebook's Chatbot Loop section. Depends on classifier, extractor,
and reasoner modules. Segmentation is handled upstream (see segmenter.py).
"""

import re
from typing import Any, Dict, List, Optional

from src.classifier import classify_clause, classify_clauses_batch
from src.extractor import extract_clause_qa
from src.reasoner import reason
from src.retriever import search


# Red flag categories (programmer-defined, same as notebook)
RED_FLAG_TYPES = [
    "Uncapped Liability",
    "Cap On Liability",
    "Termination For Convenience",
    "Exclusivity",
    "Non-Compete",
    "No-Solicit Of Employees",
    "Change Of Control",
    "Anti-Assignment",
    "Ip Ownership Assignment",
    "Source Code Escrow",
    "Audit Rights",
]

# Helper functions
def limit_clause_blocks(text: str, max_clauses: int = 5) -> str:
    """
    Keep at most `max_clauses` 'Clause <id>:' blocks in the output.
    Always keep any intro text before the first 'Clause ' line.
    """
    lines = text.splitlines()
    out_lines = []
    clause_count = 0

    for line in lines:
        stripped = line.lstrip()

        # Detect start of a new clause block
        if stripped.startswith("Clause "):
            clause_count += 1
            if clause_count > max_clauses:
                break

        out_lines.append(line)

    return "\n".join(out_lines).strip()


def retrieve_relevant_clauses(user_input: str, clauses: List[Dict[str, Any]], k: int = 5) -> List[Dict[str, Any]]:
    scores = []
    user_words = set(user_input.lower().split())
    for clause in clauses:
        clause_words = clause["text"].lower().split()
        overlap = len(user_words.intersection(clause_words))
        scores.append((overlap, clause))

    scores.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scores[:k]]


def detect_requested_clause(user_input: str, clauses: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    match = re.search(r"clause\s+(\d+)", user_input.lower())
    if match:
        target_id = int(match.group(1))
        for c in clauses:
            if c["clause_id"] == target_id:
                return c
    return None


def detect_summary_request(user_input):
    summary_triggers = [
        "summary", "summarize", "summarise", "overview", "key points",
        "explain the contract", "what is this contract about",
        "what is the contract about",
    ]
    return any(t in user_input.lower() for t in summary_triggers)


def get_representative_clauses(clauses, k=5):
    return clauses[:k]


def detect_red_flag_request(user_input: str) -> bool:
    triggers = [
        "red flag",
        "red flags",
        "dangerous clause",
        "dangerous clauses",
        "risky clause",
        "risky clauses",
        "worst clauses",
        "most dangerous",
        "biggest risks",
        "major risks",
    ]
    text = user_input.lower()
    return any(t in text for t in triggers)


def find_top_red_flag_clauses(clauses: List[Dict[str, Any]], top_k: int = 8) -> tuple[List[Dict[str, Any]], Dict[int, List[str]]]:
    texts = [c["text"] for c in clauses]
    all_labels = classify_clauses_batch(texts)  # one batched call
    
    classification_results = {}
    red_flag_clauses = []
    
    for c, labels in zip(clauses, all_labels):
        classification_results[c["clause_id"]] = labels
        overlap = [lbl for lbl in labels if lbl in RED_FLAG_TYPES]
        if overlap:
            red_flag_clauses.append((len(overlap), c))
    
    if not red_flag_clauses:
        return [], classification_results
    
    red_flag_clauses.sort(key=lambda x: x[0], reverse=True)
    top_clauses = [c for _, c in red_flag_clauses[:top_k]]
    top_classification_results = {
        c["clause_id"]: classification_results[c["clause_id"]]
        for c in top_clauses
    }
    return top_clauses, top_classification_results


# Red flag instruction strings (ported from notebook)
_RED_FLAG_INSTRUCTION = (
    "Focus ONLY on clauses that have designer-defined red-flag labels "
    "in `Clause classifications`. You may ONLY refer to a red-flag "
    "category (such as 'Uncapped Liability', 'Termination For Convenience', "
    "'Non-Compete', etc.) for a clause if that exact category name appears "
    "in the list of labels for that clause.\n"
    "The only REAL contract clauses are the ones shown in the block "
    "titled 'Relevant contract clauses', and each of those clauses starts "
    "with a line like: [Clause <id>] ...\n"
    "You MUST NOT treat ANY other text (such as 'Clause classifications' "
    "or any dictionaries or lists) as clauses.\n"
    "Ignore everything except the text after '[Clause <id>]' in the "
    " 'Relevant contract clauses' block.\n"
    "For EACH such risky clause, you MUST output it EXACTLY in this format:\n"
    "Clause <clause_id>: \"<first part of clause>\"\n"
    "    Explanation: <short explanation of why it is risky for the user>\n"
    "Leave a blank line between different clauses.\n\n"
    "At the very beginning of your answer, explicitly state that these are "
    "only the most important red flags based on the clauses provided, and "
    "that there may be additional risks elsewhere in the contract that are "
    "not covered here.\n"
)

_RED_FLAG_FALLBACK_INSTRUCTION = (
    "Based on the clauses provided, the classifier did NOT identify any "
    "clause matching the predefined red-flag categories. This is NOT an "
    "exhaustive legal review, and there may be risks in other clauses or "
    "sections of the contract not shown in the provided text.\n"
    "You must NOT invent red-flag labels or assume the presence of any "
    "risk category unless it appears directly in the clause text.\n\n"
    "The only REAL contract clauses are the ones shown in the block "
    "titled 'Relevant contract clauses', and each of those clauses starts "
    "with a line like: [Clause <id>] ...\n"
    "Ignore everything except the text after '[Clause <id>]' in the "
    "'Relevant contract clauses' block.\n\n"
    "Your task:\n"
    "  - Identify clauses that appear important or potentially risky in a "
    "    practical sense (e.g., broad responsibilities, redacted content, "
    "    asset transfer, unclear obligations).\n"
    "  - For EACH clause you highlight, output it EXACTLY in this format:\n"
    "Clause <clause_id>: \"<first part of clause>\"\n"
    "    Explanation: <short explanation of why it might be important or risky>\n"
    "  - Leave a blank line between clauses.\n"
    "  - Always ground explanations ONLY in the clause text provided.\n"
    "At the very beginning of your answer, explicitly state these are not "
    "red-flag clauses but things to potentially look out for and that there "
    "may be additional risks elsewhere in the contract that are not covered here.\n"
)

# Main pipeline entrypoint
def route_user_query(user_input: str, clauses: List[Dict[str, Any]]) -> str:
    """
    Route a user query through the full ClauseWise inference pipeline.

    Steps:
    1. If user asks about a specific clause, select it (classify if requested).
    2. If user asks for red flags, scan all clauses via LegalBERT.
    3. If user asks for a summary, use representative clauses.
    4. Otherwise retrieve top-k relevant clauses via FAISS semantic search
   (falls back to keyword overlap if index is unavailable).
    5. Run FLAN-T5 extraction for question-type queries.
    6. Assemble reasoning prompt and call Groq (DeepSeek-R1 distill).

    Args:
        user_input: Raw user question string.
        clauses:    List of clause dicts from segmenter.segment_contract().
                    Each dict has keys: clause_id, section_id, section_heading,
                    local_index, label, text.

    Returns:
        Final answer string, trimmed to at most 5 clause blocks.
    """
    text_lower = user_input.lower()
    is_red_flag_request = detect_red_flag_request(user_input)

    specific_clause = detect_requested_clause(user_input, clauses)
    classification_results: Dict[int, List[str]] = {}
    no_model_red_flags = False

    if specific_clause:
        # User asked about a specific clause number
        selected_clauses = [specific_clause]

        need_classification = "classify" in text_lower or "type" in text_lower
        if need_classification:
            classification_results[specific_clause["clause_id"]] = classify_clause(
                specific_clause["text"]
            )

    elif is_red_flag_request:
        # User explicitly wants red flags -> scan all clauses
        selected_clauses, classification_results = find_top_red_flag_clauses(clauses, top_k=5)
        if not selected_clauses:
            # Model didn't find any of programmer-defined red-flag categories
            no_model_red_flags = True
            # Fallback: still give them something useful (e.g. important clauses)
            selected_clauses = get_representative_clauses(clauses, k=5)

    else:
        # Summary vs general question
        if detect_summary_request(user_input):
            selected_clauses = get_representative_clauses(clauses, k=5)
        else:
            selected_clauses = search(user_input, k=5)
            if not selected_clauses:  # fallback if index not built
                selected_clauses = retrieve_relevant_clauses(user_input, clauses, k=5)

        need_classification = "classify" in text_lower or "type" in text_lower
        if need_classification:
            for clause in selected_clauses:
                classification_results[clause["clause_id"]] = classify_clause(
                    clause["text"]
                )

    # Build context from selected clauses (full text)
    context = "\n\n".join(
        f"[Clause {c['clause_id']}] {c['text']}" for c in selected_clauses
    )

    # Basic heuristics for extraction
    need_extraction = (
        not is_red_flag_request
        and not detect_summary_request(user_input)
        and any(q in text_lower for q in ["what", "when", "who", "how much", "define", "meaning", "obligation"])
    )
    extracted_answer = extract_clause_qa(user_input, context) if need_extraction else None

    # Select red flag instruction
    if is_red_flag_request and not no_model_red_flags:
        red_flag_instruction = _RED_FLAG_INSTRUCTION
    elif is_red_flag_request and no_model_red_flags:
        red_flag_instruction = _RED_FLAG_FALLBACK_INSTRUCTION
    else:
        red_flag_instruction = ""

    reasoning_prompt = f"""
  You are a legal reasoning engine. The user asked:

  {user_input}

  Relevant contract clauses (the ONLY real clauses you may discuss):
  {context}

  # Metadata (NOT clauses):
  Clause classifications (per clause_id):
  {classification_results}

  Extracted answer (if needed): {extracted_answer}

  Designer-defined red-flag categories:
  {RED_FLAG_TYPES}

  {red_flag_instruction}
  Provide a clear, accurate, non-legal-advice explanation.
  Do NOT hallucinate facts or clause types that are not supported by the clause text
  or the classifier labels.
  """

    final_answer = reason(reasoning_prompt)
    final_answer = limit_clause_blocks(final_answer, max_clauses=5)
    return final_answer