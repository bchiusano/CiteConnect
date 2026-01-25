# RAG enhancements: citation context, issue boost, popularity, fallback
# Adapted from dsp/RAG/rag/retrieval.py for Chroma/BM25

import os
import re
import ast
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict
import pandas as pd
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Citation prototype collection config
CITATION_PERSIST_DIR = os.path.join(os.path.dirname(__file__), "resources", "chroma_citation_prototypes")
CITATION_COLLECTION_NAME = "citation_prototypes"


def clean_ecli(text):
    if pd.isna(text) or str(text).lower() == 'nan':
        return ""
    cleaned = str(text).upper().replace("ECLI:", "").strip()
    cleaned = re.sub(r'[^A-Z0-9:]', '', cleaned)
    return cleaned


def extract_citation_contexts(advice_text: str, ecli_list: List[str], *, window_chars: int = 800) -> List[Tuple[str, str]]:
    # Returns [(ecli, context_text), ...]
    out = []
    if not advice_text:
        return out
    text_upper = advice_text.upper()

    for ecli in ecli_list:
        ne = clean_ecli(ecli)
        if not ne:
            continue
        idx = text_upper.find(ne)
        if idx == -1:
            continue
        start = max(0, idx - window_chars)
        end = min(len(advice_text), idx + len(ne) + window_chars)
        ctx = advice_text[start:end].strip()
        if ctx:
            out.append((ne, ctx))
    return out


def build_citation_prototypes(
    letters_path: str,
    embedder: HuggingFaceEmbeddings,
    train_ids: Set[str],
    window_chars: int = 800
) -> Chroma:
    """Build citation prototype collection from training data."""
    assert train_ids is not None and len(train_ids) > 0, \
        "train_ids required to build citation prototypes (prevents data leakage)"
    print(f"Building citation prototypes from {len(train_ids)} training documents...")
    
    df = pd.read_excel(letters_path)
    
    # Find ID column
    id_col = None
    for col in ["zaaknummer", "Octopus zaaknummer", "doc_id", "id"]:
        if col in df.columns:
            id_col = col
            break
    if id_col is None:
        id_col = df.columns[0]
    
    documents = []
    metadatas = []
    ids = []
    
    for idx, row in df.iterrows():
        doc_id = str(row[id_col])
        if doc_id not in train_ids:
            continue
        
        advice_text = row.get('geanonimiseerd_doc_inhoud', '')
        if pd.isna(advice_text) or not advice_text:
            continue
        
        ecli_value = row.get('ECLI')
        if pd.isna(ecli_value):
            continue
        
        # Parse ECLI list
        ecli_list = []
        if isinstance(ecli_value, str):
            try:
                parsed = ast.literal_eval(ecli_value)
                if isinstance(parsed, list):
                    ecli_list = [str(e) for e in parsed]
                else:
                    ecli_list = [str(parsed)]
            except:
                ecli_list = [ecli_value]
        elif isinstance(ecli_value, list):
            ecli_list = [str(e) for e in ecli_value]
        else:
            ecli_list = [str(ecli_value)]
        
        # Extract citation contexts
        contexts = extract_citation_contexts(str(advice_text), ecli_list, window_chars=window_chars)
        
        for ecli, ctx_text in contexts:
            doc_unique_id = f"{doc_id}_{ecli}_{len(ids)}"
            documents.append(ctx_text)
            
            # predict the issue typle of citation context
            issue_scores = predict_issues_simple(ctx_text)
            issue_id = issue_scores[0][0] if issue_scores else None
            
            metadatas.append({
                "ecli_nummer": ecli,
                "source_doc_id": doc_id,
                "doc_type": "citation_context",
                "issue_id": issue_id  
            })
            ids.append(doc_unique_id)
    
    print(f"Extracted {len(documents)} citation contexts")
    
    if not documents:
        print("Warning: No citation contexts found, creating empty collection")
    
    # Create or overwrite collection
    if os.path.exists(CITATION_PERSIST_DIR):
        import shutil
        shutil.rmtree(CITATION_PERSIST_DIR)
    
    citation_db = Chroma.from_texts(
        texts=documents if documents else ["placeholder"],
        metadatas=metadatas if metadatas else [{"doc_type": "placeholder"}],
        ids=ids if ids else ["placeholder_id"],
        embedding=embedder,
        collection_name=CITATION_COLLECTION_NAME,
        persist_directory=CITATION_PERSIST_DIR
    )
    
    print(f"Citation prototypes saved to {CITATION_PERSIST_DIR}")
    return citation_db


def load_or_build_citation_db(
    embedder: HuggingFaceEmbeddings,
    letters_path: str = None,
    train_ids: Set[str] = None,
    force_rebuild: bool = False
) -> Optional[Chroma]:
    """Load existing citation DB or build if needed."""
    
    # Try to load existing
    if os.path.exists(CITATION_PERSIST_DIR) and not force_rebuild:
        try:
            citation_db = Chroma(
                collection_name=CITATION_COLLECTION_NAME,
                persist_directory=CITATION_PERSIST_DIR,
                embedding_function=embedder
            )
            count = citation_db._collection.count()
            if count > 1:  # More than placeholder
                print(f"Loaded existing citation prototypes: {count} documents")
                return citation_db
        except Exception as e:
            print(f"Could not load citation DB: {e}")
    
    # Build new if we have the data
    if letters_path and train_ids and len(train_ids) > 0:
        return build_citation_prototypes(letters_path, embedder, train_ids)
    
    print("Warning: No citation prototype DB available")
    return None


def search_citation_context_prototypes(
    citation_db: Optional[Chroma],
    query_text: str,
    k: int = 50,
    train_ids: Optional[Set[str]] = None
) -> List[Dict]:
    """Search citation prototypes (separate collection, no domain filtering)."""
    if citation_db is None:
        return []
    
    try:
        hits = citation_db.similarity_search_with_score(query_text, k=k)
        
        results = []
        for doc, score in hits:
            metadata = doc.metadata
            # Skip placeholder
            if metadata.get('doc_type') == 'placeholder':
                continue
            # Filter by train_ids if specified
            if train_ids and metadata.get('source_doc_id'):
                if metadata['source_doc_id'] not in train_ids:
                    continue
            
            results.append({
                "ecli": clean_ecli(metadata.get('ecli_nummer', '')),
                "issue_id": metadata.get('issue_id'),
                "sim": float(1 - score),  # Distance to similarity
                "text": doc.page_content,
            })
        
        return results
    except Exception as e:
        print(f"Citation search error: {e}")
        return []


def predict_issues_simple(
    text_input: str,
    issue_keywords: Optional[Dict[str, List[str]]] = None
) -> List[Tuple[str, float]]:
    # Simple issue prediction by keyword matching, returns [(issue_id, score), ...]
    if issue_keywords is None:
        # Default issues for bicycle domain
        issue_keywords = {
            "procedural": ["procedure", "procedural", "zorgvuldigheid", "motivering", 
                          "hoorplicht", "bezwaarprocedure", "proportionaliteit", 
                          "evenredigheid", "kennisgeving", "termijn", "besluit"],
            "costs": ["kosten", "proceskosten", "vergoeding", "griffierecht", 
                     "restitut", "sleepkosten", "stallingskosten", "kostenverhaal"],
            "substance": ["fiets", "stalling", "parkeren", "verwijderd", "weesfiets", 
                         "verbod", "borden", "feitelijk"]
        }
    
    text_lower = text_input.lower()
    scores = []
    
    for issue_id, keywords in issue_keywords.items():
        hits = sum(1 for kw in keywords if kw.lower() in text_lower)
        score = min(1.0, hits / 5.0)  # 5 hits => 1.0
        if score > 0:
            scores.append((issue_id, score))
    
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:3]  # Top 3 issues


def get_popular_ecli_from_data(
    letters_path: str,
    min_citations: int = 5,
    train_ids: Optional[Set[str]] = None
) -> List[Tuple[str, int]]:
    """Get popular ECLI from training data, returns [(ecli, count), ...]"""
    assert train_ids is not None and len(train_ids) > 0, \
        "train_ids required for popularity prior (prevents data leakage)"
    
    try:
        df = pd.read_excel(letters_path)
        # Filter to training set
        id_col = None
        for col in ["zaaknummer", "Octopus zaaknummer", "doc_id", "id"]:
            if col in df.columns:
                id_col = col
                break
        if id_col:
            df = df[df[id_col].astype(str).isin(train_ids)]
        else:
            print("Warning: Could not find ID column for popularity, skipping")
            return []
        
        # Count ECLI frequency
        ecli_frequency = {}
        for _, row in df.iterrows():
            ecli_value = row.get('ECLI')
            if pd.isna(ecli_value):
                continue
            
            # Parse ECLI list
            ecli_list = []
            if isinstance(ecli_value, str):
                try:
                    parsed = ast.literal_eval(ecli_value)
                    if isinstance(parsed, list):
                        ecli_list = [str(e) for e in parsed]
                    else:
                        ecli_list = [str(parsed)]
                except Exception:
                    ecli_list = [ecli_value]
            elif isinstance(ecli_value, list):
                ecli_list = [str(e) for e in ecli_value]
            else:
                ecli_list = [str(ecli_value)]
            
            # Normalize and count
            for ecli in ecli_list:
                ecli_clean = clean_ecli(ecli)
                if ecli_clean:
                    ecli_frequency[ecli_clean] = ecli_frequency.get(ecli_clean, 0) + 1
        
        # Filter by minimum citations
        popular_ecli = [(ecli, count) for ecli, count in ecli_frequency.items() 
                       if count >= min_citations]
        popular_ecli.sort(key=lambda x: x[1], reverse=True)
        
        return popular_ecli
    except Exception as e:
        print(f"Warning: Could not load popular ECLI: {e}")
        return []


def apply_citation_context_boost(
    chunk_hits: List[Dict],
    proto_hits: List[Dict],
    boost_weight_high: float = 0.75,  # 增加 citation boost
    boost_weight_normal: float = 0.55,
    similarity_threshold: float = 0.70  # 稍微降低阈值
) -> Tuple[List[Dict], Dict[str, float], Dict[str, str], Dict[str, str]]:
    # Apply citation context boost, returns (hits, proto_best_sim, proto_issue, proto_best_text)
    # Build prototype maps
    proto_best_sim: Dict[str, float] = {}
    proto_issue: Dict[str, str] = {}
    proto_best_text: Dict[str, str] = {}
    
    for p in proto_hits:
        ecli = p.get("ecli")
        if not ecli:
            continue
        sim = float(p.get("sim", 0.0))
        if ecli not in proto_best_sim or sim > proto_best_sim[ecli]:
            proto_best_sim[ecli] = sim
            raw_issue = p.get("issue_id")
            proto_issue[ecli] = str(raw_issue) if raw_issue is not None and not isinstance(raw_issue, str) else raw_issue
            proto_best_text[ecli] = p.get("text", "")
    
    # Apply boosts to existing candidates
    for h in chunk_hits:
        ecli = h.get("ecli")
        if not ecli or ecli not in proto_best_sim:
            continue
        
        sim = proto_best_sim[ecli]
        if sim > similarity_threshold:
            h["score"] = h.get("score", 0.0) + boost_weight_high * sim
        else:
            h["score"] = h.get("score", 0.0) + boost_weight_normal * sim
    
    return chunk_hits, proto_best_sim, proto_issue, proto_best_text


def apply_issue_aware_boost(
    chunk_hits: List[Dict],
    issue_score_map: Dict[str, float],
    proto_issue: Dict[str, str],
    boost_weight: float = 0.10
) -> List[Dict]:
    # Boost ECLI if prototype issue matches predicted issues
    for h in chunk_hits:
        ecli = h.get("ecli")
        if not ecli:
            continue
        iid = proto_issue.get(ecli)
        if iid is not None:
            iid = str(iid) if not isinstance(iid, str) else iid
            if iid in issue_score_map:
                h["score"] = h.get("score", 0.0) + boost_weight * issue_score_map[iid]
    
    return chunk_hits


def apply_popularity_prior(
    ecli_hits: List[Dict],
    popular_ecli: List[Tuple[str, int]],
    boost_weight: float = 0.12
) -> List[Dict]:
    # Boost frequently cited ECLI
    if not popular_ecli:
        return ecli_hits
    
    pop_counts: Dict[str, int] = {e: c for e, c in popular_ecli}
    max_pop = max(pop_counts.values()) if pop_counts else 0
    
    if max_pop > 0:
        for h in ecli_hits:
            ecli = h.get("ecli")
            if not ecli:
                continue
            cnt = pop_counts.get(ecli, 0)
            if cnt <= 0:
                continue
            # Normalize and apply concave prior (sqrt to reduce dominance)
            pop_norm = (cnt / max_pop) ** 0.5
            h["score"] = h.get("score", 0.0) + boost_weight * pop_norm
    
    return ecli_hits


def apply_fallback(
    ecli_hits: List[Dict],
    proto_best_sim: Dict[str, float],
    popular_ecli: List[Tuple[str, int]],
    fallback_threshold: float = 0.7,
    top_ecli: int = 10
) -> List[Dict]:
    # Include high-sim citation ECLI even without chunk evidence
    existing_eclis = {h.get("ecli") for h in ecli_hits if h.get("ecli")}
    fallback_records: List[Dict] = []
    
    pop_counts: Dict[str, int] = {e: c for e, c in popular_ecli} if popular_ecli else {}
    max_pop = max(pop_counts.values()) if pop_counts else 0
    
    for ecli, sim in proto_best_sim.items():
        if ecli in existing_eclis:
            continue
        if sim < fallback_threshold:
            continue
        
        # Use prototype similarity + popularity as prior
        base_prior = 0.75 * float(sim)
        
        # Add popularity prior if available
        pop_prior = 0.0
        if max_pop > 0 and ecli in pop_counts:
            cnt = pop_counts[ecli]
            pop_norm = (cnt / max_pop) ** 0.5
            pop_prior = 0.10 * pop_norm
        
        prior_score = base_prior + pop_prior
        
        fallback_records.append({
            "ecli": ecli,
            "score": prior_score,
            "text": "",  # No chunk text available
            "is_fallback": True,
        })
    
    if fallback_records:
        # Mark existing hits as non-fallback
        for h in ecli_hits:
            h["is_fallback"] = False
        
        # Merge fallback records
        ecli_hits.extend(fallback_records)
        ecli_hits.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        
        # Ensure fallback entries are included in top results
        fallback_ids = {r["ecli"] for r in fallback_records}
        fallback_list = [r for r in ecli_hits if r.get("ecli") in fallback_ids]
        non_fallback_list = [r for r in ecli_hits if r.get("ecli") not in fallback_ids]
        
        keep_non = max(0, top_ecli - len(fallback_list))
        trimmed_non_fallback = non_fallback_list[:keep_non]
        
        combined = trimmed_non_fallback + fallback_list
        combined.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        ecli_hits = combined[:top_ecli]
    else:
        # Mark all as non-fallback
        for h in ecli_hits:
            h["is_fallback"] = False
    
    return ecli_hits


def enhance_retrieval_results(
    chunk_results: List[Tuple[str, float]],
    letter_text: str,
    citation_db: Optional[Chroma],
    letters_path: Optional[str] = None,
    train_ids: Optional[Set[str]] = None,
    proto_k: int = 50,
    top_ecli: int = 10,
    use_citation_context: bool = True,
    use_issue_boost: bool = True,
    use_popularity: bool = True,
    use_fallback: bool = True
) -> List[Dict]:
    """Apply all enhancements. citation_db is a separate collection for citation prototypes."""
    # Guard: require train_ids when using features that depend on training data
    if use_citation_context or use_popularity:
        assert train_ids is not None and len(train_ids) > 0, \
            "train_ids required when use_citation_context or use_popularity is True (prevents data leakage)"
    
    chunk_hits = [
        {"ecli": clean_ecli(ecli), "score": score, "text": "", "is_fallback": False}
        for ecli, score in chunk_results
    ]
    
    proto_hits = []
    proto_best_sim = {}
    proto_issue = {}
    proto_best_text = {}
    
    if use_citation_context and citation_db is not None:
        try:
            proto_hits = search_citation_context_prototypes(
                citation_db, letter_text, k=proto_k, train_ids=train_ids
            )
            if proto_hits:
                chunk_hits, proto_best_sim, proto_issue, proto_best_text = apply_citation_context_boost(
                    chunk_hits, proto_hits
                )
        except Exception as e:
            print(f"Warning: Citation context search failed: {e}")
    
    # 2. Issue-aware boost
    if use_issue_boost:
        try:
            issue_scores = predict_issues_simple(letter_text)
            issue_score_map = {iid: s for iid, s in issue_scores}
            chunk_hits = apply_issue_aware_boost(chunk_hits, issue_score_map, proto_issue)
        except Exception as e:
            print(f"Warning: Issue prediction failed: {e}")
    
    ecli_best: Dict[str, Dict] = {}
    for h in chunk_hits:
        ecli = h.get("ecli")
        if not ecli:
            continue
        if ecli not in ecli_best or h.get("score", 0.0) > ecli_best[ecli].get("score", 0.0):
            ecli_best[ecli] = h.copy()
    
    ecli_hits = list(ecli_best.values())
    ecli_hits.sort(key=lambda x: x.get("score", 0.0), reverse=True)
    
    if use_popularity and letters_path and train_ids:
        try:
            popular_ecli = get_popular_ecli_from_data(letters_path, min_citations=5, train_ids=train_ids)
            ecli_hits = apply_popularity_prior(ecli_hits, popular_ecli)
        except Exception as e:
            print(f"Warning: Popularity prior failed: {e}")
            popular_ecli = []
    else:
        popular_ecli = []
    
    if use_fallback and proto_best_sim:
        ecli_hits = apply_fallback(
            ecli_hits, proto_best_sim, popular_ecli, 
            fallback_threshold=0.7, top_ecli=top_ecli
        )
    else:
        ecli_hits = ecli_hits[:top_ecli]
        for h in ecli_hits:
            h["is_fallback"] = False
    
    return ecli_hits
