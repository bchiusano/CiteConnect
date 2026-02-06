import os
import sys
import pickle
import re
import time
import random
import ast
import spacy
import platform
import pandas as pd
import numpy as np
from typing import Dict, List, Set, Tuple
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank
from langchain_core.documents import Document

# --- PATH FIX ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '')))

# --- CONFIGURATION ---
from rag_pipeline_infloat_multilingual import PERSIST_DIR, EMBEDDING_MODEL, COLLECTION_NAME, letters_path
from resources.domain_config import DOMAIN_MAP, ACTIVE_DOMAIN
from rag_index_infloat_multilingual import BM25_INDEX_PATH, CORPUS_PATH
from rag_enhancements import enhance_retrieval_results, load_or_build_citation_db

RERANK_MODEL = "ms-marco-TinyBERT-L-2-v2"

# --- PARAMETERS ---
NUM_TEST_ROWS = 30
RANDOM_SEED = 40
TEST_RATIO = 0.2
SEARCH_K = 200
RERANK_TOP_N = 50
CANDIDATE_LIMIT = 50
MIN_SCORE = 0.3
RERANK_WEIGHT = 0.8


def get_device():
    """Automatically detect the best available device for the current platform."""
    if platform.system() == "Darwin":  # macOS
        return "mps"
    elif platform.system() == "Windows":
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
        except ImportError:
            pass
    return "cpu"


# --- SPACY INITIALIZATION ---
try:
    nlp = spacy.load("nl_core_news_md")
except:
    print("nl_core_news_md not found, falling back to small model.")
    nlp = spacy.load("nl_core_news_sm")


def clean_ecli(text):
    if pd.isna(text) or str(text).lower() == 'nan': return ""
    cleaned = str(text).upper().replace("ECLI:", "").strip()
    cleaned = re.sub(r'[^A-Z0-9:]', '', cleaned)
    return cleaned


def load_ground_truth_from_excel(excel_path: str, id_column: str = None) -> Dict[str, List[str]]:
    # Load ground truth from Excel, returns {advice_id: [ecli_list]}
    df = pd.read_excel(excel_path)
    ground_truth = {}

    if id_column is None:
        for col in ["zaaknummer", "Octopus zaaknummer", "doc_id", "id"]:
            if col in df.columns:
                id_column = col
                break
        if id_column is None:
            id_column = df.columns[0]

    for _, row in df.iterrows():
        advice_id = str(row[id_column]) if id_column in df.columns else str(row.index[0])
        ecli_value = row.get("ECLI")

        if pd.isna(ecli_value):
            continue

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

        ecli_list = [clean_ecli(e) for e in ecli_list if clean_ecli(e)]
        if ecli_list:
            ground_truth[advice_id] = ecli_list

    return ground_truth


def split_train_test(
        ground_truth: Dict[str, List[str]],
        test_ratio: float = 0.2,
        random_seed: int = 42
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    # Split into train/test sets
    if not ground_truth:
        return {}, {}

    advice_ids = list(ground_truth.keys())
    random.seed(random_seed)
    shuffled_ids = advice_ids.copy()
    random.shuffle(shuffled_ids)

    split_idx = int(len(shuffled_ids) * (1 - test_ratio))
    train_ids = set(shuffled_ids[:split_idx])
    test_ids = set(shuffled_ids[split_idx:])

    train_gt = {aid: ecli_list for aid, ecli_list in ground_truth.items() if aid in train_ids}
    test_gt = {aid: ecli_list for aid, ecli_list in ground_truth.items() if aid in test_ids}

    return train_gt, test_gt


def get_train_ids(ground_truth: Dict[str, List[str]], test_ratio: float = 0.2, random_seed: int = 42) -> Set[str]:
    # Get train IDs to prevent data leakage
    train_gt, _ = split_train_test(ground_truth, test_ratio, random_seed)
    return set(train_gt.keys())


class LegalRAGSystem:
    def __init__(self):
        print("Initializing Engines...")
        device = get_device()
        print(f"Using device: {device}")

        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={'device': device}
        )
        self.db = Chroma(
            collection_name=COLLECTION_NAME,
            persist_directory=PERSIST_DIR,
            embedding_function=self.embeddings
        )

        with open(BM25_INDEX_PATH, "rb") as f: self.bm25_model = pickle.load(f)
        with open(CORPUS_PATH, "rb") as f: self.legal_corpus = pickle.load(f)

        cache_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'flashrank_cache'))
        if not os.path.exists(cache_path):
            os.makedirs(cache_path)

        self.reranker = FlashrankRerank(
            model=RERANK_MODEL, top_n=RERANK_TOP_N
        )

        # Citation DB initialized lazily after train/test split
        self.citation_db = None

    def init_citation_db(self, train_ids: Set[str], force_rebuild: bool = False):
        """Initialize citation prototype DB (call after train/test split)."""
        self.citation_db = load_or_build_citation_db(
            embedder=self.embeddings,
            letters_path=letters_path,
            train_ids=train_ids,
            force_rebuild=force_rebuild
        )
        # print("Citation_db working with ids: ", train_ids)

    def load_citation_db_for_ui(self):
        self.citation_db = load_or_build_citation_db(
            embedder=self.embeddings,
            letters_path=None,
            train_ids=None,
            force_rebuild=False
        )

    def get_top_10_for_letter(self, letter_text, domain="bicycle", train_ids=None, use_enhancements=True):
        # Safeguard: ensure citation DB is initialized when using enhancements
        if use_enhancements and self.citation_db is None:
            raise RuntimeError("Citation DB not initialized. Call init_citation_db(train_ids) first.")

        keywords = DOMAIN_MAP.get(domain, {}).get("keywords", [])
        anchor = " ".join(keywords)

        # 1. Split into 5 issues
        doc = nlp(letter_text)
        sentences = [s.text.strip() for s in doc.sents if s.text.strip()]
        num_issues = 5
        chunk_size = max(1, len(sentences) // num_issues)
        # Last chunk absorbs remainder to avoid dropping sentences
        issues = []
        for i in range(num_issues):
            start = i * chunk_size
            end = (i + 1) * chunk_size if i < num_issues - 1 else len(sentences)
            issues.append(" ".join(sentences[start:end]))

        ecli_best_chunks = {}

        for issue_text in issues:
            try:
                # 2. Parallel Search
                v_hits = self.db.similarity_search(f"query: {issue_text}", k=SEARCH_K)
                tokens = re.findall(r'\w+', (f"{anchor} {issue_text}").lower())
                b_hits = self.bm25_model.get_top_n(tokens, self.legal_corpus, n=SEARCH_K)

                # 3. Hybrid Fusion (0.7 / 0.3)
                fused = self._rrf_fusion(v_hits, b_hits)

                candidate_map = {clean_ecli(d.metadata['ecli_nummer']): d for d in v_hits}
                for d in b_hits:
                    eid = clean_ecli(d['metadata'].get('ecli_nummer'))
                    if eid not in candidate_map:
                        candidate_map[eid] = Document(page_content=d['content'], metadata=d['metadata'])

                # 4. Neural Reranking (Weighted 0.2 / 0.8)
                rerank_candidates = []
                for eid, hybrid_score in fused[:RERANK_TOP_N]:
                    if eid in candidate_map:
                        rerank_candidates.append((candidate_map[eid], hybrid_score))

                if rerank_candidates:
                    docs_to_rerank = [item[0] for item in rerank_candidates]
                    refined = self.reranker.compress_documents(docs_to_rerank, issue_text)

                    for r in refined:
                        ecli = clean_ecli(r.metadata.get('ecli_nummer', ''))
                        rerank_score = r.metadata.get('relevance_score', 0)

                        # Find original hybrid score
                        original_hybrid_score = next((score for eid, score in fused if eid == ecli), 0)

                        # Apply Weighting: 20% Original, 80% Reranker
                        final_score = ((1 - RERANK_WEIGHT) * original_hybrid_score) + (RERANK_WEIGHT * rerank_score)

                        # Aggregation: Keep the best result across all 5 segments
                        if final_score >= MIN_SCORE:
                            if ecli not in ecli_best_chunks or final_score > ecli_best_chunks[ecli]:
                                ecli_best_chunks[ecli] = final_score
            except:
                continue

        base_results = [(eid, score) for eid, score in
                        sorted(ecli_best_chunks.items(), key=lambda x: x[1], reverse=True)]

        # Apply enhancements if enabled (uses separate citation_db)
        if use_enhancements:
            try:
                enhanced_results = enhance_retrieval_results(
                    chunk_results=base_results,
                    letter_text=letter_text,
                    citation_db=self.citation_db,
                    letters_path=letters_path,
                    train_ids=train_ids,
                    proto_k=50,
                    top_ecli=10,
                    use_citation_context=True,
                    use_issue_boost=True,
                    use_popularity=True,
                    use_fallback=True
                )
                return [(r.get("ecli", ""), r.get("score", 0.0)) for r in enhanced_results]
            except Exception as e:
                print(f"Warning: Enhancements failed, using base results: {e}")
                return base_results[:10]
        else:
            return base_results[:10]

    def _rrf_fusion(self, v_hits, b_hits, k=60):
        scores = {}
        # 0.7 Semantic / 0.3 Keyword split
        for r, d in enumerate(v_hits):
            eid = clean_ecli(d.metadata.get('ecli_nummer'))
            scores[eid] = scores.get(eid, 0) + 0.7 * (1 / (r + k))
        for r, d in enumerate(b_hits):
            eid = clean_ecli(d['metadata'].get('ecli_nummer'))
            scores[eid] = scores.get(eid, 0) + 0.3 * (1 / (r + k))
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)

    def prepare_train_ids_for_citation_db(self):
        ground_truth = load_ground_truth_from_excel(letters_path)
        train_gt, test_gt = split_train_test(ground_truth, test_ratio=TEST_RATIO, random_seed=RANDOM_SEED)
        train_ids = set(train_gt.keys())
        return train_ids

    def run_evaluation(self, mode="full"):
        print(f"--- STARTING EVALUATION MODE: {mode.upper()} ---")

        # Load data and prepare evaluation set
        df = pd.read_excel(letters_path)
        data = df.dropna(subset=['ECLI', 'geanonimiseerd_doc_inhoud'])

        # Split train/test to prevent data leakage
        ground_truth = load_ground_truth_from_excel(letters_path)
        train_gt, test_gt = split_train_test(ground_truth, test_ratio=TEST_RATIO, random_seed=RANDOM_SEED)
        train_ids = set(train_gt.keys())
        test_ids = set(test_gt.keys())
        print(f"Data split: {len(train_ids)} training, {len(test_ids)} test (total {len(ground_truth)})")

        # Build citation prototype DB from training data
        self.init_citation_db(train_ids, force_rebuild=False)

        id_col = None
        for col in ["zaaknummer", "Octopus zaaknummer", "doc_id", "id"]:
            if col in data.columns:
                id_col = col
                break
        if id_col:
            data = data[data[id_col].astype(str).isin(test_ids)]
        else:
            print("Warning: Could not find ID column, using all data (may cause data leakage)")

        if mode == "sample":
            data = data.sample(n=min(NUM_TEST_ROWS, len(data)), random_state=RANDOM_SEED)

        results = []
        # Metrics for both Top-5 and Top-10
        metrics = {
            "reciprocal_ranks_5": [],
            "reciprocal_ranks_10": [],
            "hits_at_5": 0,
            "hits_at_10": 0,
            "total_targets": 0,
            "precisions_5": [],
            "precisions_10": []
        }

        start_time = time.time()

        for idx, row in data.iterrows():
            # 1. Prepare standardized targets
            targets = [clean_ecli(e) for e in str(row['ECLI']).replace(';', ',').split(',') if clean_ecli(e)]

            # 2. Retrieval using the Issues-Based strategy
            # MODIFIED: Pass train_ids to prevent data leakage
            found_raw = self.get_top_10_for_letter(
                str(row['geanonimiseerd_doc_inhoud']),
                ACTIVE_DOMAIN,
                train_ids=train_ids,
                use_enhancements=True
            )
            top_10 = [clean_ecli(f[0] if isinstance(f, tuple) else f) for f in found_raw]
            top_5 = top_10[:5]

            # 3. Calculate Row Metrics for Top-10
            hits_10 = [t for t in targets if t in top_10]
            row_recall_10 = len(hits_10) / len(targets) if len(targets) > 0 else 0
            row_precision_10 = len(hits_10) / len(top_10) if len(top_10) > 0 else 0

            # Calculate Row Metrics for Top-5
            hits_5 = [t for t in targets if t in top_5]
            row_recall_5 = len(hits_5) / len(targets) if len(targets) > 0 else 0
            row_precision_5 = len(hits_5) / len(top_5) if len(top_5) > 0 else 0

            # Accumulate for global metrics
            metrics["hits_at_10"] += len(hits_10)
            metrics["hits_at_5"] += len(hits_5)
            metrics["total_targets"] += len(targets)
            metrics["precisions_10"].append(row_precision_10)
            metrics["precisions_5"].append(row_precision_5)

            # 4. Calculate Reciprocal Rank for MRR (Top-10)
            rank_score_10 = 0
            for i, ecli in enumerate(top_10, 1):
                if ecli in targets:
                    rank_score_10 = 1 / i
                    break
            metrics["reciprocal_ranks_10"].append(rank_score_10)

            # Calculate Reciprocal Rank for MRR (Top-5)
            rank_score_5 = 0
            for i, ecli in enumerate(top_5, 1):
                if ecli in targets:
                    rank_score_5 = 1 / i
                    break
            metrics["reciprocal_ranks_5"].append(rank_score_5)

            # --- DETAILED OUTPUT PER ROW ---
            print(f"\nRow ID: {idx}")
            print(f"Target ECLIs:  {targets}")
            print(f"Top 5 Found:   {top_5}")
            print(f"Top 10 Found:  {top_10}")
            print(f"Result @5:     {len(hits_5)}/{len(targets)} hits | Recall: {row_recall_5:.4f}")
            print(f"Result @10:    {len(hits_10)}/{len(targets)} hits | Recall: {row_recall_10:.4f}")
            print("-" * 30)

            # Store results for CSV export
            results.append({
                "row_id": idx,
                "targets": "; ".join(targets),
                "top_5": "; ".join(top_5),
                "top_10": "; ".join(top_10),
                "recall_at_5": row_recall_5,
                "recall_at_10": row_recall_10,
                "precision_at_5": row_precision_5,
                "precision_at_10": row_precision_10,
                "mrr_5": rank_score_5,
                "mrr_10": rank_score_10
            })

            # Progress tracker
            if len(results) % 5 == 0:
                current_recall_5 = metrics['hits_at_5'] / metrics['total_targets'] if metrics[
                                                                                          'total_targets'] > 0 else 0
                current_recall_10 = metrics['hits_at_10'] / metrics['total_targets'] if metrics[
                                                                                            'total_targets'] > 0 else 0
                print(
                    f"\n>>> PROGRESS: {len(results)}/{len(data)} | Recall@5: {current_recall_5:.2%} | Recall@10: {current_recall_10:.2%}")

        # --- FINAL SUMMARY STATISTICS ---
        final_recall_5 = metrics["hits_at_5"] / metrics["total_targets"] if metrics["total_targets"] > 0 else 0
        final_recall_10 = metrics["hits_at_10"] / metrics["total_targets"] if metrics["total_targets"] > 0 else 0
        final_precision_5 = np.mean(metrics["precisions_5"])
        final_precision_10 = np.mean(metrics["precisions_10"])
        final_mrr_5 = np.mean(metrics["reciprocal_ranks_5"])
        final_mrr_10 = np.mean(metrics["reciprocal_ranks_10"])

        print("\n" + "=" * 60)
        print(f"FINAL EVALUATION METRICS (SEED: {RANDOM_SEED})")
        print("=" * 60)
        print(f"{'Metric':<25} {'Top-5':<15} {'Top-10':<15}")
        print("-" * 60)
        print(f"{'Recall (Accuracy)':<25} {final_recall_5 * 100:.2f}%{'':<9} {final_recall_10 * 100:.2f}%")
        print(f"{'Precision':<25} {final_precision_5 * 100:.2f}%{'':<9} {final_precision_10 * 100:.2f}%")
        print(f"{'MRR':<25} {final_mrr_5:.4f}{'':<10} {final_mrr_10:.4f}")
        print("=" * 60)
        print(f"Total Test Samples: {len(data)}")
        print(f"Total Evaluation Time: {(time.time() - start_time) / 60:.2f} mins")
        print("=" * 60)

        # Save detailed results to CSV
        # 1. Get root directory (one level up from /src)
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ''))
        # 2. Define the evaluation directory
        eval_dir = os.path.join(root_dir, 'data', 'evaluation')
        # 3. Create the folder if it doesn't exist
        os.makedirs(eval_dir, exist_ok=True)

        # 4. Construct filename
        if mode == "sample":
            filename = f"eval_{mode}_seed{RANDOM_SEED}_results.csv"
        else:
            filename = f"eval_{mode}_results.csv"

        # 5. Save to the specific folder
        save_path = os.path.join(eval_dir, filename)
        pd.DataFrame(results).to_csv(save_path, index=False)
        print(f"Detailed results saved to: {save_path}")


# --- EXECUTION ---
if __name__ == "__main__":
    rag = LegalRAGSystem()
    # rag.run_evaluation(mode="sample")
    rag.run_evaluation(mode="full")
