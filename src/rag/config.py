from pathlib import Path


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR     = PROJECT_ROOT / "data"
RAG_DATA_DIR = DATA_DIR / "rag"

# Input
TICKETS_PATH = DATA_DIR / "support_tickets.json"

# Output artifacts
EMBEDDINGS_PATH       = RAG_DATA_DIR / "embeddings.npy"
EMBEDDING_INDEX_PATH  = RAG_DATA_DIR / "embedding_index.json"
KNOWLEDGE_GRAPH_PATH  = RAG_DATA_DIR / "knowledge_graph.json"
RESOLUTION_STATS_PATH = RAG_DATA_DIR / "resolution_stats.json"


# ============================================================
# EMBEDDING MODEL
# ============================================================

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM   = 384          # output dimensions for this model
EMBEDDING_BATCH_SIZE = 256     # sentences per batch during encoding


# ============================================================
# TEXT FIELDS
# ============================================================

# Which ticket fields to concatenate for embedding
# We include resolution here (unlike classification) because
# for RAG we want to match against known solutions too
TEXT_FIELDS_FOR_EMBEDDING = ["subject", "description", "error_logs", "resolution"]


# ============================================================
# RE-RANKING WEIGHTS
# ============================================================

# How much each signal contributes to the final score
# Similarity = semantic meaning from embeddings (primary signal)
# Match = knowledge graph field overlap (secondary confirmation)
SIMILARITY_WEIGHT = 0.6
MATCH_WEIGHT      = 0.4
