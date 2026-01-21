# ============================================
# High-Citation ECLI Support Functions
# ============================================
# These functions add support for including popular ECLI in search results

from typing import Optional, Set
from .ground_truth import load_ground_truth_ecli, GroundTruth


def get_popular_ecli(
    min_citations=10,
    top_k=None,
    train_ids: Optional[Set[str]] = None
):
    """
    Get popular ECLI numbers based on ground truth citation frequency.
    
    **IMPORTANT**: To prevent data leakage, only use training set data.
    Pass train_ids to filter ground truth to training set only.
    
    Parameters:
    -----------
    min_citations : int
        Minimum number of citations to be considered "popular" (default: 10)
    top_k : int, optional
        If specified, return only top K most cited ECLI
    train_ids : Set[str], optional
        Set of training advice IDs. If provided, only counts citations from
        training set to prevent data leakage. If None, uses ALL data (⚠️ data leakage risk!)
    
    Returns:
    --------
    list : List of tuples (ecli_number, citation_count) sorted by frequency
    """
    ground_truth = load_ground_truth_ecli()
    if not ground_truth:
        return []
    
    # Filter to training set only if train_ids provided
    if train_ids is not None:
        ground_truth = {aid: ecli_list for aid, ecli_list in ground_truth.items() if aid in train_ids}
    
    # Count ECLI frequency in ground truth
    ecli_frequency = {}
    for zaaknummer, ecli_list in ground_truth.items():
        for ecli in ecli_list:
            ecli_frequency[ecli] = ecli_frequency.get(ecli, 0) + 1
    
    # Filter by minimum citations
    popular_ecli = [(ecli, count) for ecli, count in ecli_frequency.items() 
                    if count >= min_citations]
    
    # Sort by frequency (descending)
    popular_ecli.sort(key=lambda x: x[1], reverse=True)
    
    # Return top K if specified
    if top_k:
        popular_ecli = popular_ecli[:top_k]
    
    return popular_ecli

print("✓ Popular ECLI functions ready!")
print("\nTo get popular ECLI list:")
print("  popular_ecli = get_popular_ecli(min_citations=10)")
print("  for ecli, count in popular_ecli[:5]:")
print("      print(f'{ecli}: cited {count} times')")
