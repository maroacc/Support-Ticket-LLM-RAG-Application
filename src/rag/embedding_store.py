import json
import numpy as np
from sentence_transformers import SentenceTransformer

from src.data_utils import load_data
from src.rag.config import (
    EMBEDDING_MODEL,
    EMBEDDINGS_PATH,
    EMBEDDING_INDEX_PATH,
)
from src.rag.build_embeddings import build_ticket_text


# ============================================================
# LOAD PRE-COMPUTED DATA
# ============================================================

# These are loaded once when the module is first used (via load())
_embeddings    = None   # np.ndarray (N, 384)
_index_map     = None   # dict: ticket_id -> row number
_reverse_index = None   # dict: row number -> ticket_id
_tickets       = None   # dict: ticket_id -> full ticket dict
_model         = None   # SentenceTransformer (loaded lazily on first query)


def load():
    """
    Load embeddings, index, and ticket data into memory.
    Call this once before calling find_similar().

    What gets loaded:
      - embeddings.npy       (~160MB, the 110K x 384 matrix)
      - embedding_index.json (ticket_id <-> row mapping)
      - support_tickets.json (full ticket data to return in results)
    """
    global _embeddings, _index_map, _reverse_index, _tickets

    # Load the embedding matrix
    _embeddings = np.load(EMBEDDINGS_PATH)
    print(f"Loaded embeddings: {_embeddings.shape}")

    # Load the ticket_id <-> row number mapping
    with open(EMBEDDING_INDEX_PATH) as f:
        _index_map = json.load(f)
    _reverse_index = {v: k for k, v in _index_map.items()}

    # Load full ticket data so we can return complete ticket info
    df = load_data()
    _tickets = {row["ticket_id"]: row for _, row in df.iterrows()}
    print(f"Loaded {len(_tickets)} tickets for lookup")


# ============================================================
# ENCODE A QUERY
# ============================================================

def _get_model() -> SentenceTransformer:
    """Load the embedding model lazily (only when first query happens)."""
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def _encode_query(text: str) -> np.ndarray:
    """
    Encode a single query string into a 384-dim vector.
    Normalized so dot product = cosine similarity.
    """
    model = _get_model()
    embedding = model.encode([text], normalize_embeddings=True)
    return embedding[0].astype(np.float32)


# ============================================================
# FIND SIMILAR TICKETS
# ============================================================

def find_similar(ticket: dict, top_k: int = 10) -> list[dict]:
    """
    Find the most similar historical tickets to the one provided.

    Args:
        ticket: A dict with the same structure as the dataset tickets.
                At minimum needs the text fields used for embedding:
                  - subject      : ticket title
                  - description  : detailed description of the issue
                  - error_logs   : any error log text (optional, can be "")
                  - resolution   : known resolution text (optional, can be "")
        top_k:  Number of similar tickets to return (default 10).

    Returns:
        List of dicts, each containing:
          - ticket_id          : ID of the similar ticket
          - similarity_score   : cosine similarity (0 to 1, higher = more similar)
          - ticket_data        : full ticket dict with all original fields

    Example:
        load()  # call once at startup

        results = find_similar({
            "subject": "Database sync failing with timeout error",
            "description": "Getting ERROR_TIMEOUT_429 when syncing large datasets",
            "error_logs": "ERROR_TIMEOUT_429: Connection timeout after 30s",
            "resolution": "",
        })

        for r in results:
            print(r["similarity_score"], r["ticket_data"]["subject"])
    """
    if _embeddings is None:
        raise RuntimeError("Call load() before find_similar()")

    # Step 1 — Build query text from the ticket fields (same way we built embeddings)
    query_text = build_ticket_text(ticket)

    # Step 2 — Encode the query into a 384-dim vector
    query_vec = _encode_query(query_text)

    # Step 3 — Dot product against all stored embeddings
    #          Since both are L2-normalized, this equals cosine similarity
    similarities = _embeddings @ query_vec   # shape: (N,)

    # Step 4 — Get the top_k highest scores
    #          argpartition is O(N) vs O(N log N) for full sort
    top_indices = np.argpartition(similarities, -top_k)[-top_k:]
    # Sort those top_k by score descending
    top_indices = top_indices[np.argsort(similarities[top_indices])[::-1]]

    # Step 5 — Build results with full ticket data
    results = []
    for idx in top_indices:
        ticket_id = _reverse_index[int(idx)]
        results.append({
            "ticket_id": ticket_id,
            "similarity_score": round(float(similarities[idx]), 4),
            "ticket_data": _tickets[ticket_id],
        })

    return results


# ============================================================
# MAIN — quick test
# ============================================================

if __name__ == "__main__":
    load()

    # Simulate a new incoming ticket (same format as the dataset)
    new_ticket = {
        "subject": "Database sync failing with timeout error",
        "description": "Getting ERROR_TIMEOUT_429 when syncing large datasets. "
                       "This started happening after the recent update.",
        "error_logs": "ERROR_TIMEOUT_429: Connection timeout after 30s\n"
                      "RETRY_FAILED: Max retries exceeded",
        "resolution": "",
    }

    print(f"\nQuery ticket:")
    print(f"  Subject: {new_ticket['subject']}")
    print(f"  Description: {new_ticket['description'][:80]}...")
    print("=" * 70)

    results = find_similar(new_ticket, top_k=10)

    print(f"\nTop {len(results)} similar tickets:\n")
    for i, r in enumerate(results, 1):
        t = r["ticket_data"]
        print(f"#{i}  Score: {r['similarity_score']}  ID: {r['ticket_id']}")
        print(f"    Category:   {t['category']} / {t['subcategory']}")
        print(f"    Subject:    {t['subject']}")
        print(f"    Resolution: {str(t['resolution'])[:100]}...")
        print()
