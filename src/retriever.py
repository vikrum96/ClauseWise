"""
retriever.py — FAISS-backed semantic clause retrieval for ClauseWise.
 
Encodes clause texts into dense embeddings using a sentence transformer,
builds a FAISS index per contract upload, and retrieves the top-k most
semantically similar clauses for a given query.
 
The sentence transformer (all-MiniLM-L6-v2) is loaded once at server
startup via _ensure_loaded(). The FAISS index is rebuilt per contract
upload via build_index() since clauses change with every new document.
"""

from __future__ import annotations

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import Any, Dict, List

# Model Config
MODEL_ID = "all-MiniLM-L6-v2"

# Module-level state
_encoder: SentenceTransformer | None = None

# Per-contract index state — rebuilt on every build_index() call
_index: faiss.IndexFlatIP | None = None
_indexed_clauses: List[Dict[str, Any]] = []

def _ensure_loaded() -> None:
    """Load the sentence transformer model."""
    global _encoder
    if _encoder is None:
        _encoder = SentenceTransformer(MODEL_ID)

# Public API
def build_index(clauses) -> None:
    """
    Encode all clause texts and build a FAISS index for the current contract.
    
    Called once per contract upload in app.py, after segmentation.
    Replaces any previously built index (only one contract is indexed at a time)
    
    Args:
        clauses: List of clause dicts from ContractSegmenter.segment_contract().
                 Each dict must have a "text" key.
    """

    global _index, _indexed_clauses
    _ensure_loaded()
    texts = [c["text"] for c in clauses]

    # Encode all clause texts into dense embeddings
    embeddings = _encoder.encode(
        texts, batch_size=32, 
        show_progress_bar=False, normalize_embeddings=True, 
        convert_to_numpy=True
    )
    embeddings = embeddings.astype(np.float32)

    # IndexFlatIP = exact inner product search
    # With normalized embeddings this is equivalent to cosine similarity
    dimension = embeddings.shape[1]
    _index = faiss.IndexFlatIP(dimension)
    _index.add(embeddings)
    _indexed_clauses = clauses

def search(query, k=5):
    """
    Retrieve the top-k clauses most semantically similar for a given query.
    
    Args:
        query: User query string.
        k: Number of top results to return.

    Returns:
        List of clause dicts ordered by cosine similarity (higest first)
        Returns an empty list if no index is built or if the query is empty.
    """
    if _index is None or not _indexed_clauses or not query.strip():
        return []
    _ensure_loaded()

    k_clamped = min(k, len(_indexed_clauses))
    query_embedding = _encoder.encode([query], normalize_embeddings=True, convert_to_numpy=True)
    query_embedding = query_embedding.astype(np.float32)

    # (distances, indices) 
    # distances represent cosine similarities
    _, indices = _index.search(query_embedding, k_clamped)

    return [_indexed_clauses[i] for i in indices[0] if i != -1]