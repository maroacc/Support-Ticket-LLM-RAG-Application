import json
from collections import defaultdict

from src.data_utils import load_data
from src.rag.config import RAG_DATA_DIR, RESOLUTION_STATS_PATH


# ============================================================
# BUILD STATS
# ============================================================

def build_resolution_stats(tickets: list[dict]) -> dict:
    """
    For each distinct KB article that appears in kb_articles_helpful,
    calculate its success rate: how often resolution_helpful was True
    when that article was used.

    Example output:
      {
        "KB-0218": { "times_used": 150, "times_helpful": 120, "success_rate": 0.8 },
        "KB-0397": { "times_used": 95,  "times_helpful": 70,  "success_rate": 0.7368 },
        ...
      }
    """
    # Track per KB article: total times used + times resolution was helpful
    stats = defaultdict(lambda: {"times_used": 0, "times_helpful": 0})

    for ticket in tickets:
        kb_articles = ticket.get("kb_articles_helpful", []) or []
        helpful = ticket.get("resolution_helpful", False)

        for article in kb_articles:
            stats[article]["times_used"] += 1
            if helpful:
                stats[article]["times_helpful"] += 1

    # Calculate success rate
    result = {}
    for article, counts in sorted(stats.items()):
        result[article] = {
            "times_used": counts["times_used"],
            "times_helpful": counts["times_helpful"],
            "success_rate": round(
                counts["times_helpful"] / counts["times_used"], 4
            ),
        }

    return result


# ============================================================
# SAVE TO DISK
# ============================================================

def save_resolution_stats(stats: dict):
    """Save resolution stats as a JSON file."""
    RAG_DATA_DIR.mkdir(parents=True, exist_ok=True)

    with open(RESOLUTION_STATS_PATH, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"Saved resolution stats: {RESOLUTION_STATS_PATH}")
    print(f"  Distinct KB articles: {len(stats)}")

    # Quick summary
    rates = [s["success_rate"] for s in stats.values()]
    if rates:
        avg_rate = sum(rates) / len(rates)
        print(f"  Avg success rate: {avg_rate:.4f}")
        print(f"  Min success rate: {min(rates):.4f}")
        print(f"  Max success rate: {max(rates):.4f}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    df = load_data()
    tickets = df.to_dict(orient="records")

    stats = build_resolution_stats(tickets)
    save_resolution_stats(stats)
