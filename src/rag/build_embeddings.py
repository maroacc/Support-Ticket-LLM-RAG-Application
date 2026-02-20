import json
import numpy as np
from sentence_transformers import SentenceTransformer

from src.data_utils import load_data
from src.rag.config import (
    EMBEDDING_MODEL,
    EMBEDDING_BATCH_SIZE,
    TEXT_FIELDS_FOR_EMBEDDING,
    RAG_DATA_DIR,
    EMBEDDINGS_PATH,
    EMBEDDING_INDEX_PATH,
)


# ============================================================
# STEP 1  BUILD TEXT FOR EACH TICKET
# ============================================================

def build_ticket_text(ticket: dict) -> str:
    """
    Concatenate the relevant text fields into a single string.
    Skips empty/null fields so we don't embed meaningless whitespace.

    Example output:
      "Database sync failing with timeout error Getting ERROR_TIMEOUT_429
       when syncing large datasets... Increased batch size limits in config.yaml..."
    """
    parts = []
    for field in TEXT_FIELDS_FOR_EMBEDDING:
        value = ticket.get(field, "") or ""
        value = str(value).strip()
        if value:
            parts.append(value)
    return " ".join(parts)


# ============================================================
# STEP 2  GENERATE EMBEDDINGS
# ============================================================

def generate_embeddings(tickets: list[dict]) -> tuple[np.ndarray, dict]:
    """
    Encode all tickets into dense vectors using sentence-transformers.

    What happens here:
      1. Load the all-MiniLM-L6-v2 model (22M params, 384-dim output)
      2. Build a text string for each ticket (subject + description + error_logs + resolution)
      3. Encode all texts in batches
      4. L2-normalize so dot product = cosine similarity (faster at query time)

    Returns:
      embeddings : np.ndarray of shape (num_tickets, 384), dtype float32
      index_map  : dict mapping ticket_id -> row number in the matrix
    """
    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    # Build text for each ticket
    texts = [build_ticket_text(t) for t in tickets]
    print(f"Built text for {len(texts)} tickets")

    # Map each ticket_id to its row index
    index_map = {t["ticket_id"]: i for i, t in enumerate(tickets)}

    # Encode  normalize_embeddings=True does L2 normalization
    # so later we can use simple dot product instead of cosine similarity
    print(f"Encoding {len(texts)} tickets (batch_size={EMBEDDING_BATCH_SIZE})...")
    embeddings = model.encode(
        texts,
        batch_size=EMBEDDING_BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    return embeddings.astype(np.float32), index_map


# ============================================================
# STEP 3  SAVE TO DISK
# ============================================================

def save_embeddings(embeddings: np.ndarray, index_map: dict):
    """
    Save the two files:
      - embeddings.npy    : the raw matrix (N x 384 floats)
      - embedding_index.json : ticket_id -> row number mapping

    Together they let us look up any ticket's vector by ID,
    and translate search results (row numbers) back to ticket IDs.
    """
    RAG_DATA_DIR.mkdir(parents=True, exist_ok=True)

    np.save(EMBEDDINGS_PATH, embeddings)
    print(f"Saved embeddings: {EMBEDDINGS_PATH}  shape={embeddings.shape}")

    with open(EMBEDDING_INDEX_PATH, "w") as f:
        json.dump(index_map, f)
    print(f"Saved index: {EMBEDDING_INDEX_PATH}  entries={len(index_map)}")


# ============================================================
# MAIN  run this script to build embeddings
# ============================================================

if __name__ == "__main__":
    # Reuse load_data from data_utils (returns a DataFrame)
    df = load_data()

    # Convert DataFrame rows to list of dicts for easy field access
    tickets = df.to_dict(orient="records")

    # Generate and save
    embeddings, index_map = generate_embeddings(tickets)
    save_embeddings(embeddings, index_map)

    print(f"\nDone! Files saved to {RAG_DATA_DIR}/")
    print(f"  embeddings.npy       : {embeddings.shape[0]} vectors x {embeddings.shape[1]} dims")
    print(f"  embedding_index.json : {len(index_map)} ticket mappings")
