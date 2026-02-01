# Linking Legal Advice to Case Law Using RAG
In legal advice letters, lawyers frequently reference case law using ECLI citations. This project involves developing a tool that links legal arguments in advice documents to relevant jurisprudence. The goal is to make it easier for legal professionals to find pertinent case law quickly. A Retrieval-Augmented Generation (RAG) approach is envisioned for this solution.

# Required python version (to be documented more properly: issues with chromadb vector store)
3.12

# Setup
Run the following commands:

```bash
pip install pandas openpyxl langchain langchain-classic langchain-chroma langchain-text-splitters langchain-huggingface langchain-community chromadb openai sentence-transformers tiktoken huggingface_hub flashrank rank-bm25 nlp spacy transformers torch bitsandbytes accelerate mlx-lm

pip install sentencepiece protobuf
```

```bash
python -m spacy download nl_core_news_md
```

## Legal Text Summarizer

This module uses a fine-tuned **Llama-3b-DUTCH** model to transform raw objection letters into technical legal search queries for ECLI retrieval.

## 🛠 Setup Instructions

### 1. Install Dependencies
Ensure you are using Python 3.12 and install the required high-speed transfer tools.
To avoid slow downloads, download the model locally using your Hugging Face token.

Create Account: Sign up at huggingface.co.

Request Access: Go to the Llama 3.2 3B page and click Accept on the license agreement (usually instant).

Generate Token:

Go to Settings > Tokens > New Token.
Type: Read.

Add to Environment: Copy the token (starts with hf_...) and use it in the download command:

```bash
pip install hf_transfer

export HF_HUB_ENABLE_HF_TRANSFER=1
hf download BramVanroy/fietje-2-chat --local-dir ./models/fietje --token <YOUR_HF_TOKEN>
```
hf download BramVanroy/fietje-2-chat --local-dir ./models/fietje --token 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

huggingface-cli download BramVanroy/fietje-1b-chat --local-dir ./models/fietje-1b --token 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

huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct --local-dir ./models/qwen-1.5b --token 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

huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct --local-dir ./models/qwen-0.5b