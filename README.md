# Linking Legal Advice to Case Law Using RAG
In legal advice letters, lawyers frequently reference case law using ECLI citations. This project involves developing a tool that links legal arguments in advice documents to relevant jurisprudence. The goal is to make it easier for legal professionals to find pertinent case law quickly. A Retrieval-Augmented Generation (RAG) approach is envisioned for this solution.

# Required python version (to be documented more properly: issues with chromadb vector store)
3.12

# Setup
Run the following command:

```python
pip install pandas openpyxl langchain langchain-classic langchain-chroma langchain-text-splitters langchain-huggingface langchain-community chromadb openai sentence-transformers tiktoken huggingface_hub flashrank
```
pip install rank-bm25


