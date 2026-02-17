import json
import re

from src.data_utils import load_data
from src.rag.config import RAG_DATA_DIR, KNOWLEDGE_GRAPH_PATH


# ============================================================
# CONFIGURATION
# ============================================================

# Regex to extract error codes like ERROR_TIMEOUT_429, RETRY_FAILED, etc.
# Matches: uppercase letters/underscores, ending with digits
# Examples: ERROR_TIMEOUT_429, ERROR_RATELIMIT_429, RETRY_FAILED
ERROR_CODE_PATTERN = re.compile(r"[A-Z][A-Z_]+\d{2,}")


# ============================================================
# EXTRACT ERROR CODES
# ============================================================

def extract_error_codes(error_logs: str) -> list[str]:
    """
    Pull error codes from the error_logs field using regex.

    Example:
      Input:  "2024-01-15 10:25:33 ERROR_TIMEOUT_429: Connection timeout after 30s\n
               2024-01-15 10:25:34 RETRY_FAILED: Max retries exceeded"
      Output: ["ERROR_TIMEOUT_429"]

    Note: RETRY_FAILED won't match because the pattern requires
    ending digits (to avoid matching generic uppercase words).
    """
    if not error_logs:
        return []
    return list(set(ERROR_CODE_PATTERN.findall(str(error_logs))))


# ============================================================
# BUILD THE RELATIONSHIP MAP
# ============================================================

def build_knowledge_graph(tickets: list[dict]) -> list[dict]:
    """
    For each ticket, extract the key fields that define relationships:
      - ticket_id
      - product
      - product_version
      - product_module
      - category
      - subcategory
      - error_codes (extracted from error_logs via regex)

    Returns a list of dicts, one per ticket.
    """
    graph = []

    for ticket in tickets:
        entry = {
            "ticket_id":       ticket.get("ticket_id", ""),
            "product":         ticket.get("product", ""),
            "product_version": ticket.get("product_version", ""),
            "product_module":  ticket.get("product_module", ""),
            "category":        ticket.get("category", ""),
            "subcategory":     ticket.get("subcategory", ""),
            "error_codes":     extract_error_codes(ticket.get("error_logs", "")),
        }
        graph.append(entry)

    return graph


# ============================================================
# SAVE TO DISK
# ============================================================

def save_knowledge_graph(graph: list[dict]):
    """Save the knowledge graph as a JSON file."""
    RAG_DATA_DIR.mkdir(parents=True, exist_ok=True)

    with open(KNOWLEDGE_GRAPH_PATH, "w") as f:
        json.dump(graph, f, indent=2)

    print(f"Saved knowledge graph: {KNOWLEDGE_GRAPH_PATH}")
    print(f"  Entries: {len(graph)}")

    # Quick stats
    with_errors = sum(1 for entry in graph if entry["error_codes"])
    all_codes = set()
    for entry in graph:
        all_codes.update(entry["error_codes"])
    print(f"  Tickets with error codes: {with_errors}")
    print(f"  Unique error codes found: {len(all_codes)}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    df = load_data()
    tickets = df.to_dict(orient="records")

    graph = build_knowledge_graph(tickets)
    save_knowledge_graph(graph)
