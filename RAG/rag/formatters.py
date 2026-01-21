from typing import List, Dict

def format_ecli_citations(results: List[Dict], *, show_snippet: bool = True) -> str:
    if not results:
        return "No relevant ECLI numbers found."

    lines = [f"Found {len(results)} ECLI:"]
    for i, r in enumerate(results, 1):
        lines.append(f"{i}. {r['ecli_number']}  (score={r['score']:.4f})")
        if show_snippet and r.get("text_snippet"):
            lines.append(f"   {r['text_snippet'].strip()}...")
    return "\n".join(lines)
