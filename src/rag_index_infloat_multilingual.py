import os
import pickle
import re
from pathlib import Path
from rank_bm25 import BM25Okapi
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from rag_pipeline_infloat_multilingual import PERSIST_DIR

EMBEDDING_MODEL = "intfloat/multilingual-e5-large"
COLLECTION_NAME = "legal_rag"
BM25_INDEX_PATH = Path("resources/legal_bm25_index.pkl")
CORPUS_PATH = Path("resources/legal_corpus_docs.pkl")


def build_lexical_layer():
    """
    Connects to the existing Chroma DB, extracts all text, 
    and builds a persistent BM25 index.
    """
    print(f"⚡ Loading existing Chroma DB from: {PERSIST_DIR}")

    # We initialize embeddings just to access the existing collection
    # but we won't be generating any new vectors.
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'mps'}
    )

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR
    )

    # 1. Fetch all documents and metadata from your 75k collection
    print("⚡ Fetching raw text chunks from Chroma...")
    data = vector_store.get(include=['documents', 'metadatas'])

    documents = data['documents']
    metadatas = data['metadatas']

    if not documents:
        print("❌ Error: No documents found in Chroma. Check your PERSIST_DIR.")
        return

    # 2. Tokenize the corpus for BM25
    # We use a simple regex to catch ECLI patterns and Law articles specifically
    print(f"⚡ Tokenizing {len(documents)} chunks...")
    tokenized_corpus = [re.findall(r'\w+', doc.lower()) for doc in documents]

    # 3. Create the BM25 Index
    print("⚡ Calculating BM25 frequency tables...")
    bm25 = BM25Okapi(tokenized_corpus)

    # 4. Save the Index and a "Reference Corpus" to disk
    # We save a simplified version of the corpus so the search script 
    # can return text even if it only uses the BM25 index.
    print(f"💾 Saving BM25 index to {BM25_INDEX_PATH}...")
    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump(bm25, f)

    print(f"💾 Saving reference corpus to {CORPUS_PATH}...")
    corpus_bundle = [{"content": doc, "metadata": meta} for doc, meta in zip(documents, metadatas)]
    with open(CORPUS_PATH, "wb") as f:
        pickle.dump(corpus_bundle, f)

    print("✅ SUCCESS: Hybrid Lexical Layer is ready.")


if __name__ == "__main__":
    # Ensure the directory exists before attempting to load
    if not os.path.exists(PERSIST_DIR):
        print(f"❌ Error: {PERSIST_DIR} does not exist. Run your rag_pipeline first.")
    else:
        try:
            build_lexical_layer()
        except Exception as e:
            print(f"❌ Error building index: {e}")
