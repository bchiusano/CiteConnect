import os
import sys
import time
import pandas as pd

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# Data paths
DATA_DIR = "../data/"
EMBEDDING_MODEL = "intfloat/multilingual-e5-large"

ecli_path = os.path.join(DATA_DIR, "DATA ecli_nummers juni 2025 v1 (version 1).xlsx")
letters_path = os.path.join(DATA_DIR, "Dataset Advice letters on objections towing of bicycles.xlsx")

# Directory to persist Chroma database
PERSIST_DIR = "resources/chroma_db_infloat_multilingual"

# Global placeholder for embeddings object (set by create_embeddings)
embeddings = None


def load_data_and_prepare_documents():
    """Load Excel files and prepare Document objects for ALL texts."""
    df_ecli = pd.read_excel(ecli_path)

    # REMOVED: bicycle_keywords filter is gone to ensure a general-purpose database

    print(f"Total ECLI cases being indexed: {len(df_ecli)}")

    ecli_texts = df_ecli['ecli_tekst'].astype(str).tolist()

    case_docs = [
        Document(
            page_content=f"passage: {text}",
            metadata={
                "source": "case",
                "ecli_nummer": str(df_ecli.iloc[i]["ecli_nummer"]).replace("ECLI:", "").strip()
            }
        )
        for i, text in enumerate(ecli_texts)
    ]
    return case_docs


# Create a text splitter
# chunk_size = maximum number of characters per chunk
# chunk_overlap = number of characters overlapping between chunks to preserve context
def split_documents(all_docs, chunk_size=800, chunk_overlap=150):
    """Split documents into chunks using RecursiveCharacterTextSplitter."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""]  # Prioritize paragraph/line breaks
    )

    # Split all documents (ECLI cases + Advice letters) into smaller chunks
    # Splitting preserves metadata for each chunk
    split_docs = text_splitter.split_documents(all_docs)

    print("Original number of documents:", len(all_docs))
    print("Number of chunks created after splitting:", len(split_docs))

    if len(split_docs) > 0:
        print("\nSample chunk text (first 200 chars):")
        print(split_docs[0].page_content[:200])
        print("\nSample chunk metadata:")
        print(split_docs[0].metadata)

    return split_docs


# Create Embeddings
# Purpose: Convert each text chunk into a numeric vector (embedding) that captures its semantic meaning
# These embeddings will be stored in a vector store (Chroma) for similarity search
def create_embeddings(split_docs, model_name):
    """
    Create an embeddings object using the model given and test it on the first document chunk.
    Returns the embeddings object.
    """
    global embeddings

    print("Initializing embeddings model...")
    print(f"Using model: {model_name}")

    # Initialize embeddings
    # embeddings = HuggingFaceEmbeddings(model_name=model_name)
    # Below one is specific to 'clips/e5-large-trm-nl' model
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={'device': 'mps'},
        encode_kwargs={'normalize_embeddings': True},
        show_progress=True
    )

    # Test on the first chunk to confirm model works
    if len(split_docs) > 0:
        print("\nTesting embedding on first chunk...")
        sample_text = split_docs[0].page_content
        try:
            test_embedding = embeddings.embed_documents([sample_text])
            print("Embedding created successfully!")
            print("Embedding vector length:", len(test_embedding[0]))
            print("First 5 values:", test_embedding[0][:5])
        except Exception as e:
            print("Warning: embedding test failed:", e)

    return embeddings


# Create Chroma vector store from split documents
# Purpose: Store embeddings of all document chunks for fast similarity search
# Also preserves metadata (like source, case number) for context during retrieval
def create_vector_store_prev(documents, persist_directory=PERSIST_DIR):
    """Create a new Chroma vector store and persist it to disk."""
    start = time.time()

    # Create directory if it does not exist
    os.makedirs(persist_directory, exist_ok=True)

    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name="legal_rag",
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"}
    )
    print("Chroma vector store created.")

    try:
        count = vector_store._collection.count()
        print("Number of documents in vector store:", count)
    except Exception:
        count = "unknown"

    # Persist the vector store to disk
    try:
        print("Starting to persist vector store...")
        vector_store.persist()
        print("Completed persisting vector store.")
    except Exception as e:
        print("Warning: could not persist vector store:", e)

    print("Chroma vector store created and persisted successfully!")
    print("Number of documents stored in vector store:", count)
    print(f"Vector store saved at: {persist_directory}")
    print(f"Time taken: {time.time() - start:.2f} seconds")

    return vector_store


def create_vector_store(documents, persist_directory=PERSIST_DIR):
    start = time.time()
    os.makedirs(persist_directory, exist_ok=True)

    # 1. Initialize the store
    print("Creating vector store..")
    vector_store = Chroma(
        collection_name="legal_rag",
        embedding_function=embeddings,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"}
    )
    print("Chroma vector store created.")

    # 2. Add documents 
    print("Adding documents to vector store...")
    # Add documents in chunks of 500 to stay well under the 5461 limit
    step = 500
    for i in range(0, len(documents), step):
        batch = documents[i: i + step]
        print(f"Adding batch {i // step + 1}...")
        vector_store.add_documents(batch)
    print("--- SUCCESS: Documents added ---")
    print(f"Time taken: {time.time() - start:.2f} seconds")

    try:
        count = vector_store._collection.count()
        print("Number of documents in vector store:", count)
    except Exception:
        count = "unknown"

    # Persist the vector store to disk
    try:
        print("Starting to persist vector store...")
        vector_store.persist()
        print("Completed persisting vector store.")
    except Exception as e:
        print("Warning: could not persist vector store:", e)

    print("Chroma vector store created and persisted successfully!")
    print("Number of documents stored in vector store:", count)
    print(f"Vector store saved at: {persist_directory}")

    return vector_store


def load_vector_store(persist_directory=PERSIST_DIR):
    """Load an existing Chroma vector store."""
    vector_store = Chroma(
        collection_name="legal_rag",
        embedding_function=embeddings,
        persist_directory=persist_directory
    )
    print("Loaded existing Chroma DB.")
    return vector_store


def update_vector_store(new_documents, persist_directory=PERSIST_DIR):
    """Add new documents into an existing Chroma store and persist."""
    start = time.time()

    vector_store = load_vector_store(persist_directory)

    vector_store.add_documents(new_documents)

    # Persist the vector store to disk
    try:
        vector_store.persist()
    except Exception as e:
        print("Warning: could not persist updated vector store:", e)

    try:
        count = vector_store._collection.count()
    except Exception:
        count = "unknown"

    print("Chroma vector store updated with new documents!")
    print("Total number of documents stored in vector store:", count)
    print(f"Update time: {time.time() - start:.2f} seconds")

    return vector_store


def main():
    # Load and prepare documents
    all_docs = load_data_and_prepare_documents()

    # Split documents
    split_docs = split_documents(all_docs)

    # Create embeddings 
    create_embeddings(split_docs, model_name=EMBEDDING_MODEL)

    # Create/persist Chroma vector store
    create_vector_store(split_docs)

    # Example: load the store
    # store = load_vector_store()

    # To update the store with new docs
    # update_vector_store(new_docs)


if __name__ == "__main__":
    main()
