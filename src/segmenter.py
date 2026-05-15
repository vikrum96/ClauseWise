from __future__ import annotations
import os
from PyPDF2 import PdfReader
import re
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any

@dataclass
class Section:
    section_id: int
    heading: Optional[str]
    text: str


@dataclass
class Clause:
    clause_id: int            
    section_id: int          
    section_heading: Optional[str]
    local_index: int             # index within section (1-based)
    label: Optional[str]         # e.g. "(a)" or "a)" or None
    text: str

# "SECTION 1. TERM", "Article II – Payment"
SECTION_HEADING_PATTERN = re.compile(
    r'^\s*(section|article)\s+([0-9ivxlcdm]+)[\.\-)]?\s+.*',
    re.IGNORECASE
)

# "1. TERM", "1.1 Scope", "2.3.4 Some Heading"
NUMBERED_HEADING_PATTERN = re.compile(
    r'^\s*\d+(\.\d+)*\s+.+'
)

# ALL CAPS headings: "CONFIDENTIALITY", "LIMITATION OF LIABILITY"
# We infer heading if line has mostly uppercase letters.
def looks_like_all_caps_heading(line: str, min_len: int = 5, min_upper_ratio: float = 0.8) -> bool:
    s = line.strip()
    if len(s) < min_len:
        return False
    letters = [c for c in s if c.isalpha()]
    if not letters:
        return False
    upper_ratio = sum(c.isupper() for c in letters) / len(letters)
    return upper_ratio >= min_upper_ratio

#   "(a) Text", "(1) Text", "(i) Text"
SUBCLAUSE_PAREN_PATTERN = re.compile(r'^\s*\(([a-zA-Z0-9ivxlcdm]+)\)\s+')

#   "a) Text", "1) Text"
SUBCLAUSE_SUFFIX_PATTERN = re.compile(r'^\s*([a-zA-Z0-9ivxlcdm]+)\)\s+')

class ContractSegmenter:
    def __init__(
        self,
        min_clause_len_chars: int = 25,
        merge_short_clauses: bool = True,
    ) -> None:

        self.min_clause_len_chars = min_clause_len_chars
        self.merge_short_clauses = merge_short_clauses

    def segment_contract(self, text: str) -> List[Dict[str, Any]]:
        text = self._preprocess(text)
        sections = self._segment_sections(text)
        clauses = self._segment_clauses_within_sections(sections)
        clauses = self._cleanup_clauses(clauses)

        # Return list of plain dicts
        return [asdict(c) for c in clauses]

    def _preprocess(self, text: str) -> str:
        if not text:
            return ""

        # Normalize line endings
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"[ \t]+", " ", text)

        # Strip trailing spaces on each line
        lines = [ln.rstrip() for ln in text.split("\n")]
        return "\n".join(lines)

    def _is_section_heading(self, line: str) -> bool:
        """
        Check: does this line look like a section heading?
        """
        s = line.strip()
        if not s:
            return False

        # "Section 1. Term", "Article II - Definitions"
        if SECTION_HEADING_PATTERN.match(s):
            return True

        # "1. Term", "1.1 Grant of License"
        if NUMBERED_HEADING_PATTERN.match(s):
            return True

        # ALL CAPS headings
        if looks_like_all_caps_heading(s):
            return True

        return False

    def _segment_sections(self, text: str) -> List[Section]:
        lines = text.split("\n")

        sections: List[Section] = []
        current_heading: Optional[str] = None
        current_lines: List[str] = []
        section_id = 0

        for line in lines:
            if self._is_section_heading(line):
                # Flush previous section (if any)
                if current_heading is not None or current_lines:
                    section_text = "\n".join(current_lines).strip()
                    if section_text:
                        section_id += 1
                        sections.append(
                            Section(
                                section_id=section_id,
                                heading=current_heading,
                                text=section_text,
                            )
                        )

                # Start new section
                current_heading = line.strip()
                current_lines = []
            else:
                if line.strip() or current_lines:
                    current_lines.append(line)

        # Flush last section
        if current_heading is not None or current_lines:
            section_text = "\n".join(current_lines).strip()
            if section_text:
                section_id += 1
                sections.append(
                    Section(
                        section_id=section_id,
                        heading=current_heading,
                        text=section_text,
                    )
                )

        # If we never detected any heading at all, treat whole doc as one section
        if not sections and text.strip():
            sections.append(
                Section(section_id=1, heading=None, text=text.strip())
            )

        return sections

    def _parse_subclause_label(self, line: str) -> Optional[str]:
        """
        Check if a line starts a subclause (e.g., "(a) Text" or "a) Text").
        Returns the label (e.g., "(a)", "a)", "(1)") or None.
        """
        m = SUBCLAUSE_PAREN_PATTERN.match(line)
        if m:
            raw = m.group(1)
            # Return normalized label with parentheses: e.g. "(a)"
            return f"({raw})"

        m = SUBCLAUSE_SUFFIX_PATTERN.match(line)
        if m:
            raw = m.group(1)
            # Return normalized label with suffix: e.g. "a)"
            return f"{raw})"

        return None

    def _strip_subclause_marker(self, line: str) -> str:
        line = SUBCLAUSE_PAREN_PATTERN.sub("", line, count=1)
        line = SUBCLAUSE_SUFFIX_PATTERN.sub("", line, count=1)
        return line.lstrip()

    def _segment_clauses_within_sections(
        self, sections: List[Section]
    ) -> List[Clause]:
        """
        For each section, split into clauses based on subclause markers.
        If no markers are found, treat the whole section as a single clause.
        """
        clauses: List[Clause] = []
        global_clause_id = 0

        for sec in sections:
            lines = sec.text.split("\n")
            local_index = 0
            current_label: Optional[str] = None
            current_lines: List[str] = []

            def flush_current():
                nonlocal global_clause_id, local_index, current_lines, current_label
                text_block = "\n".join(current_lines).strip()
                if text_block:
                    global_clause_id += 1
                    local_index += 1
                    clauses.append(
                        Clause(
                            clause_id=global_clause_id,
                            section_id=sec.section_id,
                            section_heading=sec.heading,
                            local_index=local_index,
                            label=current_label,
                            text=text_block,
                        )
                    )
                current_lines = []
                current_label = None

            # Track if we ever saw any subclause pattern in this section
            saw_subclause_pattern = False

            for line in lines:
                label = self._parse_subclause_label(line)
                if label is not None:
                    saw_subclause_pattern = True
                    # New subclause: flush previous
                    flush_current()
                    current_label = label
                    # Add line without the label
                    stripped = self._strip_subclause_marker(line)
                    if stripped:
                        current_lines.append(stripped)
                else:
                    # Continuation of current subclause (if any)
                    # or plain text inside section without explicit labels
                    if line.strip() or current_lines:
                        current_lines.append(line)

            # Flush last clause in this section
            flush_current()

        return clauses

    def _cleanup_clauses(self, clauses: List[Clause]) -> List[Clause]:
        if not self.merge_short_clauses or not clauses:
            return clauses

        cleaned: List[Clause] = []
        for clause in clauses:
            # If this clause is very short and we have a previous one, merge
            if (
                cleaned
                and len(clause.text.strip()) < self.min_clause_len_chars
                and clause.section_id == cleaned[-1].section_id
            ):
                # Merge into previous clause (same section)
                prev = cleaned[-1]
                merged_text = prev.text.rstrip() + "\n" + clause.text.lstrip()
                cleaned[-1] = Clause(
                    clause_id=prev.clause_id,
                    section_id=prev.section_id,
                    section_heading=prev.section_heading,
                    local_index=prev.local_index,
                    label=prev.label,  # keep previous label
                    text=merged_text,
                )
            else:
                cleaned.append(clause)

        # Re-number global clause_id to keep it simple (optional)
        for idx, cl in enumerate(cleaned, start=1):
            cleaned[idx - 1] = Clause(
                clause_id=idx,
                section_id=cl.section_id,
                section_heading=cl.section_heading,
                local_index=cl.local_index,
                label=cl.label,
                text=cl.text,
            )

        return cleaned


def load_contract_text(input_path: str) -> str:
    ext = os.path.splitext(input_path)[1].lower()

    if ext == ".txt":
        with open(input_path, "r", encoding="utf-8") as f:
            return f.read()

    elif ext == ".pdf":
        reader = PdfReader(input_path)
        parts = []
        for page in reader.pages:
            # page.extract_text() returns a string or None
            text = page.extract_text() or ""
            parts.append(text)
        return "\n\n".join(parts)

    else:
        raise ValueError(f"Unsupported file type: {ext}. Use .txt or .pdf")

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python Contract_Parser_V2.py <input_file.(txt|pdf)> <output_file.txt>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    # Load contract text from .txt or .pdf
    try:
        contract_text = load_contract_text(input_path)
    except Exception as e:
        print(f"Error reading input file: {e}")
        sys.exit(1)

    segmenter = ContractSegmenter()
    clause_dicts = segmenter.segment_contract(contract_text)

    # Write readable output text file
    with open(output_path, "w", encoding="utf-8") as out:
        for c in clause_dicts:
            out.write(f"Clause {c['clause_id']}\n")
            out.write(f"Section: {c['section_heading']}\n")
            out.write(f"Label: {c['label']}\n")
            out.write("-" * 50 + "\n")
            out.write(c["text"] + "\n\n")