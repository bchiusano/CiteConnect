import pickle

""" Code Generated with Claude to check what is inside the pickle files """

# View the corpus file
with open('resources/legal_corpus_docs.pkl', 'rb') as f:
    corpus = pickle.load(f)

print(f"Type: {type(corpus)}")
print(f"Total documents: {len(corpus)}")
print("\n--- First 3 documents ---")
for i, doc in enumerate(corpus[:3]):
    print(f"\nDocument {i}:")
    print(f"Content preview: {doc['content'][:200]}...")
    print(f"Metadata: {doc['metadata']}")

# View the BM25 index
with open('resources/legal_bm25_index.pkl', 'rb') as f:
    bm25 = pickle.load(f)

print(f"\n--- BM25 Index ---")
print(f"Type: {type(bm25)}")
print(f"Number of documents indexed: {len(bm25.doc_len)}")
print(f"Average document length: {bm25.avgdl:.2f}")
print(f"Total unique terms: {len(bm25.idf)}")