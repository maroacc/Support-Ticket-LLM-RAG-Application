import json

from src.rag.config import KNOWLEDGE_GRAPH_PATH, SIMILARITY_WEIGHT, MATCH_WEIGHT
from src.rag.embedding_store import load as load_embeddings, find_similar
from src.rag.build_knowledge_graph import extract_error_codes


# ============================================================
# LOAD KNOWLEDGE GRAPH
# ============================================================

_knowledge_graph = None   # dict: ticket_id -> knowledge graph entry


def load():
    """
    Load everything needed for the solution finder:
      1. Embeddings + ticket data (via embedding_store)
      2. Knowledge graph (ticket_id -> product, category, error_codes, etc.)

    Call once before using find_solutions().
    """
    global _knowledge_graph

    # Load embeddings and ticket data
    load_embeddings()

    # Load knowledge graph as a lookup by ticket_id
    with open(KNOWLEDGE_GRAPH_PATH) as f:
        graph_list = json.load(f)
    _knowledge_graph = {entry["ticket_id"]: entry for entry in graph_list}
    print(f"Loaded knowledge graph: {len(_knowledge_graph)} entries")


# ============================================================
# COMPARE A NEW TICKET AGAINST A SIMILAR TICKET'S GRAPH ENTRY
# ============================================================

def _compare_fields(new_ticket: dict, graph_entry: dict) -> dict:
    """
    Compare the new ticket's fields against a similar ticket's
    knowledge graph entry. Returns which fields match.

    Checks:
      - product         : exact match
      - product_version : exact match
      - product_module  : exact match
      - category        : exact match
      - error_codes     : any overlap between the two sets

    Returns a dict with:
      - matches     : dict of field_name -> bool (True if they match)
      - match_count : how many fields matched (0-5)
      - match_ratio : match_count / total fields checked
    """
    # Extract the new ticket's error codes from its error_logs
    new_error_codes = set(extract_error_codes(new_ticket.get("error_logs", "")))
    similar_error_codes = set(graph_entry.get("error_codes", []))

    matches = {
        "product":         new_ticket.get("product", "") == graph_entry.get("product", ""),
        "product_version": new_ticket.get("product_version", "") == graph_entry.get("product_version", ""),
        "product_module":  new_ticket.get("product_module", "") == graph_entry.get("product_module", ""),
        "category":        new_ticket.get("category", "") == graph_entry.get("category", ""),
        "error_codes":     bool(new_error_codes & similar_error_codes) if new_error_codes else False,
    }

    match_count = sum(matches.values())

    return {
        "matches": matches,
        "match_count": match_count,
        "match_ratio": round(match_count / len(matches), 4),
    }


# ============================================================
# FIND SOLUTIONS
# ============================================================

def find_solutions(ticket: dict, top_k: int = 10) -> list[dict]:
    """
    Full RAG pipeline: find similar tickets, then enrich each result
    with knowledge graph comparison.

    Args:
        ticket: A new incoming ticket dict. Should have at minimum:
                  - subject         : ticket title
                  - description     : detailed description
                  - error_logs      : error log text (can be "")
                  - resolution      : "" (unknown for new tickets)
                And optionally (for knowledge graph matching):
                  - product         : e.g. "DataSync Pro"
                  - product_version : e.g. "3.2.1"
                  - product_module  : e.g. "sync_engine"
                  - category        : predicted by classifier (or "")
                  - subcategory     : predicted by classifier (or "")
        top_k:  Number of results to return.

    Returns:
        List of dicts sorted by final_score (descending), each containing:
          - ticket_id         : ID of the similar ticket
          - similarity_score  : cosine similarity (0-1)
          - match_ratio       : knowledge graph field overlap (0-1)
          - final_score       : weighted combination of both signals
          - ticket_data       : full ticket dict with all fields
          - graph_comparison  : which fields match the new ticket
    """
    if _knowledge_graph is None:
        raise RuntimeError("Call load() before find_solutions()")

    # Step 1 — Find the most similar tickets by embedding similarity
    similar_tickets = find_similar(ticket, top_k=top_k)

    # Step 2 — For each similar ticket, compare its knowledge graph entry
    results = []
    for result in similar_tickets:
        tid = result["ticket_id"]
        graph_entry = _knowledge_graph.get(tid, {})

        comparison = _compare_fields(ticket, graph_entry)

        # Step 3 — Calculate final score combining both signals
        similarity = result["similarity_score"]
        match_ratio = comparison["match_ratio"]
        final_score = (SIMILARITY_WEIGHT * similarity) + (MATCH_WEIGHT * match_ratio)

        results.append({
            "ticket_id":        tid,
            "similarity_score": similarity,
            "match_ratio":      match_ratio,
            "final_score":      round(final_score, 4),
            "ticket_data":      result["ticket_data"],
            "graph_comparison": comparison,
        })

    # Step 4 — Re-sort by final_score (not just similarity)
    results.sort(key=lambda r: r["final_score"], reverse=True)

    return results


# ============================================================
# MAIN — test
# ============================================================

if __name__ == "__main__":
    load()

    # Simulate a new incoming ticket
    new_ticket = {
        "subject": "Database sync failing with timeout error",
        "description": "Getting ERROR_TIMEOUT_429 when syncing large datasets. "
                       "This started happening after the recent update.",
        "error_logs": "ERROR_TIMEOUT_429: Connection timeout after 30s\n"
                      "RETRY_FAILED: Max retries exceeded",
        "resolution": "",
        "product": "DataSync Pro",
        "product_version": "3.2.1",
        "product_module": "sync_engine",
        "category": "Technical Issue",
        "subcategory": "Configuration",
    }

    print(f"\nNew ticket:")
    print(f"  Subject:  {new_ticket['subject']}")
    print(f"  Product:  {new_ticket['product']} v{new_ticket['product_version']}")
    print(f"  Category: {new_ticket['category']} / {new_ticket['subcategory']}")
    print("=" * 70)

    results = find_solutions(new_ticket, top_k=10)

    print(f"\nTop {len(results)} solutions:\n")
    for i, r in enumerate(results, 1):
        t = r["ticket_data"]
        comp = r["graph_comparison"]

        # Show which fields matched with checkmarks
        match_symbols = []
        for field, matched in comp["matches"].items():
            symbol = "Y" if matched else "N"
            match_symbols.append(f"{field}={symbol}")

        print(f"#{i}  Final: {r['final_score']}  Similarity: {r['similarity_score']}  Match: {comp['match_ratio']*100:.0f}%  ID: {r['ticket_id']}")
        print(f"    Fields:  {', '.join(match_symbols)}")
        print(f"    Subject: {t['subject']}")
        print(f"    Resolution: {str(t['resolution'])[:100]}...")
        print()
