import os
import sys
import pickle
import re
import platform
import pandas as pd
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank
from langchain_core.documents import Document

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- CONFIGURATION ---
from rag_pipeline_infloat_multilingual import PERSIST_DIR, EMBEDDING_MODEL, COLLECTION_NAME, letters_path
from resources.domain_config import DOMAIN_MAP, ACTIVE_DOMAIN
from rag_index_infloat_multilingual import BM25_INDEX_PATH, CORPUS_PATH

RERANK_MODEL = "ms-marco-TinyBERT-L-2-v2"

# --- PARAMETERS ---
SEARCH_K = 400
RERANK_TOP_N = 60
CANDIDATE_LIMIT = 100


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


def _rrf_fusion(v_hits, b_hits, k=60):
    scores = {}
    for r, d in enumerate(v_hits):
        eid = d.metadata.get('ecli_nummer')
        scores[eid] = scores.get(eid, 0) + (1 / (r + k))
    for r, d in enumerate(b_hits):
        eid = d['metadata'].get('ecli_nummer')
        # 1.2 boost for lexical matching consistency
        scores[eid] = scores.get(eid, 0) + 1.2 * (1 / (r + k))
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def clean_ecli(text):
    """Standardizes ECLI strings for comparison."""
    if pd.isna(text) or str(text).lower() == 'nan': return ""
    text = str(text).upper().strip().replace("ECLI:", "")
    for char in ["[", "]", "'", '"', " ", "\n", "\r", ";"]:
        text = text.replace(char, "")
    return text


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

        with open(BM25_INDEX_PATH, "rb") as f: 
            self.bm25_model = pickle.load(f)
        with open(CORPUS_PATH, "rb") as f: 
            self.legal_corpus = pickle.load(f)

        cache_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'flashrank_cache'))
        self.reranker = FlashrankRerank(model=RERANK_MODEL, top_n=RERANK_TOP_N)

    def get_top_10_for_letter(self, letter_text, domain="bicycle"):
        """
        UI METHOD: Takes a single letter, splits into 5 issues, and returns best 10 ECLIs.
        """
        # 1. Prepare Domain Anchors
        keywords = DOMAIN_MAP.get(domain, {}).get("keywords", [])
        anchor = " ".join(keywords)

        # 2. Split FULL letter into exactly 5 claims/issues
        # First, clean text and split into sentences (using a basic split for logic consistency)
        sentences = [s.strip() for s in re.split(r'[.!?]\s+', letter_text) if len(s.strip()) > 10]

        num_issues = 5
        if len(sentences) < num_issues:
            claims = sentences
        else:
            # Divide sentences into 5 roughly equal chunks
            chunk_size = len(sentences) // num_issues
            claims = [" ".join(sentences[i * chunk_size: (i + 1) * chunk_size]) for i in range(num_issues)]

        ecli_scores = {}

        for claim in claims:
            try:
                # Query Expansion with anchor terms
                enhanced_query = f"{anchor} {claim}"

                # Retrieval using SEARCH_K parameter
                v_hits = self.db.similarity_search(f"query: {enhanced_query}", k=SEARCH_K)
                tokens = re.findall(r'\w+', enhanced_query.lower())
                b_hits = self.bm25_model.get_top_n(tokens, self.legal_corpus, n=SEARCH_K)

                # Hybrid Fusion (RRF)
                fused_results = _rrf_fusion(v_hits, b_hits)

                # Prepare candidates for Reranker
                candidate_map = {doc.metadata['ecli_nummer']: doc for doc in v_hits}
                for d in b_hits:
                    eid = d['metadata']['ecli_nummer']
                    if eid not in candidate_map:
                        candidate_map[eid] = Document(page_content=d['content'], metadata=d['metadata'])

                # Using CANDIDATE_LIMIT parameter
                top_cands = [candidate_map[eid] for eid, _ in fused_results[:CANDIDATE_LIMIT] if eid in candidate_map]

                # Neural Rerank
                refined = self.reranker.compress_documents(top_cands, enhanced_query)
                for r in refined:
                    ecli = clean_ecli(r.metadata.get('ecli_nummer', ''))
                    score = r.metadata.get('relevance_score', 0)
                    # Keep the best score for an ECLI found across the 5 claims
                    if ecli not in ecli_scores or score > ecli_scores[ecli]:
                        ecli_scores[ecli] = score
            except:
                continue

        # Sort and return top 10
        sorted_eclis = sorted(ecli_scores.items(), key=lambda x: x[1], reverse=True)
        return [item[0] for item in sorted_eclis[:10]]


# --- EXECUTION ---
if __name__ == "__main__":
    rag = LegalRAGSystem()

    # Verify with sample first, then switch to 'full'
    rag.run_evaluation(mode="sample")