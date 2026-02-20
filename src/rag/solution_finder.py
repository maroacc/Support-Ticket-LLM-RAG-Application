import json

from src.rag.config import KNOWLEDGE_GRAPH_PATH, RESOLUTION_STATS_PATH, SIMILARITY_WEIGHT, MATCH_WEIGHT
from src.rag.embedding_store import load as load_embeddings, find_similar
from src.rag.build_knowledge_graph import extract_error_codes


# ============================================================
# LOAD KNOWLEDGE GRAPH + RESOLUTION STATS
# ============================================================

_knowledge_graph  = None   # dict: ticket_id -> knowledge graph entry
_resolution_stats = None   # dict: kb_article_id -> {times_used, times_helpful, success_rate}


def load():
    """
    Load everything needed for the solution finder:
      1. Embeddings + ticket data (via embedding_store)
      2. Knowledge graph (ticket_id -> product, category, error_codes, etc.)
      3. Resolution stats (kb_article_id -> success_rate)

    Call once before using find_solutions().
    """
    global _knowledge_graph, _resolution_stats

    # Load embeddings and ticket data
    load_embeddings()

    # Load knowledge graph as a lookup by ticket_id
    with open(KNOWLEDGE_GRAPH_PATH) as f:
        graph_list = json.load(f)
    _knowledge_graph = {entry["ticket_id"]: entry for entry in graph_list}
    print(f"Loaded knowledge graph: {len(_knowledge_graph)} entries")

    # Load resolution stats (KB article success rates)
    with open(RESOLUTION_STATS_PATH) as f:
        _resolution_stats = json.load(f)
    print(f"Loaded resolution stats: {len(_resolution_stats)} KB articles")


# ============================================================
# EXTRACT STRUCTURED SOLUTION FROM A SIMILAR TICKET
# ============================================================

def _extract_solution(ticket_data: dict) -> dict:
    """
    Pull solution-relevant fields out of a historical ticket and enrich
    the KB articles with their success rates from resolution_stats.

    Returns:
        {
          "resolution":          str    what was done to fix the issue
          "resolution_code":     str    e.g. "CONFIG_CHANGE"
          "resolution_template": str    e.g. "TEMPLATE-DB-TIMEOUT"
          "resolution_helpful":  bool   whether the resolution actually worked
          "kb_articles":         list   helpful KB articles sorted by success_rate,
                                         each: {"article": "KB-887", "success_rate": 0.85}
        }
    """
    kb_articles_raw = ticket_data.get("kb_articles_helpful") or []
    kb_articles = []
    for article in kb_articles_raw:
        stats = _resolution_stats.get(article, {})
        kb_articles.append({
            "article":      article,
            "success_rate": stats.get("success_rate"),
        })
    kb_articles.sort(key=lambda x: (x["success_rate"] or 0), reverse=True)

    return {
        "resolution":          ticket_data.get("resolution", ""),
        "resolution_code":     ticket_data.get("resolution_code"),
        "resolution_template": ticket_data.get("resolution_template_used"),
        "resolution_helpful":  ticket_data.get("resolution_helpful"),
        "kb_articles":         kb_articles,
    }


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
    Full RAG pipeline: find similar tickets, enrich with knowledge graph
    comparison, and extract structured solutions (resolution text + KB articles).

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
        top_k:  Number of results to return.

    Returns:
        List of dicts sorted by final_score (descending), each containing:
          - ticket_id         : ID of the similar ticket
          - similarity_score  : cosine similarity (0-1)
          - match_ratio       : knowledge graph field overlap (0-1)
          - final_score       : weighted combination of both signals
          - ticket_data       : full ticket dict with all fields
          - graph_comparison  : which fields match the new ticket
          - solution          : structured solution extracted from the similar ticket:
                                  resolution, resolution_code, resolution_template,
                                  resolution_helpful, kb_articles (with success_rate)
    """
    if _knowledge_graph is None:
        raise RuntimeError("Call load() before find_solutions()")

    # Step 1  Find the most similar tickets by embedding similarity
    similar_tickets = find_similar(ticket, top_k=top_k)

    # Step 2  For each similar ticket, compare its knowledge graph entry
    results = []
    for result in similar_tickets:
        tid = result["ticket_id"]
        graph_entry = _knowledge_graph.get(tid, {})

        comparison = _compare_fields(ticket, graph_entry)

        # Step 3  Calculate final score combining both signals
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
            "solution":         _extract_solution(result["ticket_data"]),
        })

    # Step 4  Re-sort by final_score (not just similarity)
    results.sort(key=lambda r: r["final_score"], reverse=True)

    return results


# ============================================================
# MAIN  test
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

        sol = r["solution"]
        kb_str = ", ".join(
            f"{kb['article']}({kb['success_rate']:.0%})" for kb in sol["kb_articles"]
        ) or "none"

        print(f"#{i}  Final: {r['final_score']}  Similarity: {r['similarity_score']}  Match: {comp['match_ratio']*100:.0f}%  ID: {r['ticket_id']}")
        print(f"    Fields:      {', '.join(match_symbols)}")
        print(f"    Subject:     {t['subject']}")
        print(f"    Resolution:  {str(sol['resolution'])[:100]}...")
        print(f"    Code:        {sol['resolution_code']}  |  Template: {sol['resolution_template']}")
        print(f"    KB articles: {kb_str}")
        print()
