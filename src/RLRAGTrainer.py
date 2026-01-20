import torch
from torch import optim
from tqdm import tqdm
from test_rag_infloat_multilingual import get_device, clean_ecli, LegalRAGSystem
from rag_pipeline_infloat_multilingual import letters_path
from DocumentScorer import DocumentRelevanceScorer
from resources.domain_config import DOMAIN_MAP, ACTIVE_DOMAIN
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from langchain_core.documents import Document
import re
import pickle


# TODO: don't we need to remove the ecli's from the query?

def prepare_training_data(train_size, random_seed):
    df = pd.read_excel(letters_path)
    data = df.dropna(subset=['ECLI', 'geanonimiseerd_doc_inhoud'])
    print(len(data))
    query_target_data = []

    for idx, row in data.iterrows():

        targets = [clean_ecli(e) for e in str(row['ECLI']).replace(';', ',').split(',') if clean_ecli(e)]

        # if there is more than one target
        if targets:
            query_target_data.append({
                'query': str(row['geanonimiseerd_doc_inhoud']),
                'targets': targets
            })

    # split into train/test
    train, test = train_test_split(
        query_target_data,
        train_size=train_size,
        random_state=random_seed
    )

    print(f"Data split: {len(train)} train, {len(test)} test")

    return train, test


def compute_reward(predicted_top_10, target_eclis):
    # reward based on how many targets were retrieved.
    predicted_set = set(predicted_top_10)
    target_set = set(target_eclis)

    hits = len(predicted_set & target_set)
    recall = hits / len(target_set) if target_set else 0
    precision = hits / len(predicted_set) if predicted_set else 0

    return recall, precision



def extract_document_features(doc, query_text, bm25_score, vector_score,
                              rerank_score, bm25_rank, vector_rank,
                              num_keywords_matched):
    """
    Extract features for a single document-query pair.

    Returns:
        torch.tensor of shape [feature_dim]
    """
    doc_length = len(doc.page_content.split())
    query_length = len(query_text.split())

    # Calculate query-document overlap
    query_words = set(query_text.lower().split())
    doc_words = set(doc.page_content.lower().split())
    overlap_ratio = len(query_words & doc_words) / len(query_words) if query_words else 0

    # Extract metadata
    year = doc.metadata.get('year', 2020)
    normalized_year = year / 2025.0  # Normalize to roughly 0-1

    # Check if ECLI is valid format
    ecli = clean_ecli(doc.metadata.get('ecli_nummer', ''))
    valid_ecli = 1.0 if ecli and len(ecli.split(':')) >= 4 else 0.0

    features = torch.tensor([
        bm25_score,  # BM25 score (already normalized by RRF)
        vector_score,  # Vector similarity score
        rerank_score,  # Reranker score
        1.0 / (bm25_rank + 1),  # Reciprocal rank in BM25
        1.0 / (vector_rank + 1),  # Reciprocal rank in vector search
        doc_length / 1000.0,  # Normalized document length
        query_length / 100.0,  # Normalized query length
        overlap_ratio,  # Query-document word overlap
        num_keywords_matched / 10.0,  # Domain keywords matched
        normalized_year,  # Document year
        valid_ecli,  # ECLI validity flag
        doc_length / (query_length + 1)  # Length ratio
    ], dtype=torch.float32)

    return features


class RLRAGTrainer:
    """Trains a neural document scorer using policy gradient RL."""

    def __init__(self, rag_system, feature_dim=12, hidden_dim=64,
                 learning_rate=0.001, device='cpu'):
        self.rag = rag_system
        self.device = torch.device(device)
        self.policy = DocumentRelevanceScorer(feature_dim, hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)

        # NEW: Cache for candidates
        print("loading candidate cache")
        try:
            with open("candidate_cache.pkl", "rb") as f: self.candidates_cache = pickle.load(f)
        except:
            print("uncomment the part on creating the candidate cache in the main section and try again, and comment this part")

        # Training history
        self.training_history = {
            'episode_rewards': [],
            'recall_at_10': [],
            'precision_at_10': [],
            'losses': []
        }

    def precompute_candidates(self, training_data):
        """Pre-compute candidates for all training queries ONCE."""
        print("\nPre-computing candidates for all queries...")

        for i, data in enumerate(tqdm(training_data, desc="Caching candidates")):
            query = data['query']

            # Compute and store
            candidates = self.get_candidates_with_features(query)
            self.candidates_cache[i] = candidates

        print(f"✓ Cached {len(self.candidates_cache)} query results")

    def get_candidates_with_features(self, query_text):
        """
        Retrieve candidates and extract features for each.

        Returns:
            candidates: List of dicts with 'doc', 'features', 'ecli'
        """

        keywords = DOMAIN_MAP.get(ACTIVE_DOMAIN, {}).get("keywords", [])
        anchor = " ".join(keywords)

        # Get retrieval results
        v_hits = self.rag.db.similarity_search(f"query: {query_text}", k=100) # TODO: changed from 200 to 100 for computing purposes
        tokens = re.findall(r'\w+', (f"{anchor} {query_text}").lower())
        b_hits = self.rag.bm25_model.get_top_n(tokens, self.rag.legal_corpus, n=100) # TODO: changed from 200 to 100

        # Compute hybrid scores using RRF
        fused = self.rag._rrf_fusion(v_hits, b_hits)

        # Create candidate map
        candidate_map = {}
        for rank, d in enumerate(v_hits):
            eid = clean_ecli(d.metadata['ecli_nummer'])
            candidate_map[eid] = {
                'doc': d,
                'vector_rank': rank,
                'vector_score': 1.0 / (rank + 60),  # Approximate RRF score
                'bm25_rank': 999,
                'bm25_score': 0
            }

        for rank, d in enumerate(b_hits):
            eid = clean_ecli(d['metadata'].get('ecli_nummer'))
            if eid in candidate_map:
                candidate_map[eid]['bm25_rank'] = rank
                candidate_map[eid]['bm25_score'] = 1.0 / (rank + 60)
            else:
                doc = Document(page_content=d['content'], metadata=d['metadata'])
                candidate_map[eid] = {
                    'doc': doc,
                    'vector_rank': 999,
                    'vector_score': 0,
                    'bm25_rank': rank,
                    'bm25_score': 1.0 / (rank + 60)
                }

        # Get top candidates by hybrid score
        top_candidates = []
        for eid, hybrid_score in fused[:20]:  # TODO: changed from 50 to 20
            if eid in candidate_map:
                top_candidates.append((candidate_map[eid], hybrid_score))

        # Rerank
        if top_candidates:
            docs_to_rerank = [item[0]['doc'] for item in top_candidates]
            reranked = self.rag.reranker.compress_documents(docs_to_rerank, query_text)

            # Update with rerank scores
            for r in reranked:
                ecli = clean_ecli(r.metadata.get('ecli_nummer', ''))
                if ecli in candidate_map:
                    candidate_map[ecli]['rerank_score'] = r.metadata.get('relevance_score', 0)

        # Count keyword matches
        query_lower = query_text.lower()
        for eid, cand in candidate_map.items():
            matched = sum(1 for kw in keywords if kw.lower() in query_lower)
            cand['num_keywords'] = matched

        # Extract features for all candidates
        candidates = []
        for eid, cand_info in candidate_map.items():
            features = extract_document_features(
                doc=cand_info['doc'],
                query_text=query_text,
                bm25_score=cand_info.get('bm25_score', 0),
                vector_score=cand_info.get('vector_score', 0),
                rerank_score=cand_info.get('rerank_score', 0),
                bm25_rank=cand_info.get('bm25_rank', 999),
                vector_rank=cand_info.get('vector_rank', 999),
                num_keywords_matched=cand_info.get('num_keywords', 0)
            )

            candidates.append({
                'ecli': eid,
                'features': features,
                'doc': cand_info['doc']
            })

        return candidates

    def train_single(self, query_id, target):
        # train on a single query
        # returns episode reward (recall) and gradient loss

        self.policy.train()

        # candidate_ecli = self.get_candidates_with_features(query)
        candidate_ecli = self.candidates_cache.get(query_id, [])

        # print("Candidate_ecli: ", candidate_ecli)
        if not candidate_ecli:
            print("no ecli's found for this query")
            return 0.0, 0.0

        # Stack features into batch
        features_batch = torch.stack([c['features'] for c in candidate_ecli]).to(self.device)

        # Forward pass: get relevance scores
        with torch.set_grad_enabled(True):
            scores = self.policy(features_batch).squeeze()  # [num_candidates]

        # Select top 10 based on neural scores
        top_10_indices = torch.topk(scores, min(10, len(scores))).indices
        predicted_top_10 = [candidate_ecli[i]['ecli'] for i in top_10_indices.cpu().numpy()]

        # For debug purposes
        print("target: ", target)
        print("predicted_top_10: ", predicted_top_10)
        print("Scores: ", scores)
        # Compute reward
        recall, _ = compute_reward(predicted_top_10, target)
        print("Recall: ", recall)
        # Policy Gradient Loss (REINFORCE)
        # We want to maximize reward, so minimize negative log probability weighted by reward
        log_probs = torch.log(scores + 1e-8)  # Add epsilon for numerical stability
        print("Log_probs: ", log_probs)
        # Only consider the top 10 documents we selected
        selected_log_probs = log_probs[top_10_indices]
        print("Selected_log_probs: ", selected_log_probs)
        # Policy gradient: maximize reward for selected actions
        loss = -torch.sum(selected_log_probs) * recall
        print("Loss: ", loss, ", Loss item: ", loss.item())
        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
        self.optimizer.step()
        # TODO: why is loss zero ???
        return recall, loss.item()

    def train(self, data, epochs, batch_size):
        # data is a list of dictionaries with query and targets

        if not self.candidates_cache:
            self.precompute_candidates(data)

        for epoch in range(epochs):
            epoch_rewards = []
            epoch_losses = []

            # Create index mapping
            indices = list(range(len(data)))
            np.random.shuffle(indices)

            progress_bar = tqdm(indices, desc=f"Epoch {epoch + 1}/{epochs}")
            for i, idx in enumerate(progress_bar):
                targets = data[idx]['targets']

                # print("QUERY: ", query)
                # print("TARGET: ", targets)

                # Train on this episode
                recall, loss = self.train_single(idx, targets)
                # print("Episode recall: ", recall)

                epoch_rewards.append(recall)
                epoch_losses.append(loss)

                # Update progress bar
                if (i + 1) % batch_size == 0:
                    avg_reward = np.mean(epoch_rewards[-batch_size:])
                    avg_loss = np.mean(epoch_losses[-batch_size:])
                    progress_bar.set_postfix({
                        'Recall@10': f'{avg_reward:.3f}',
                        'Loss': f'{avg_loss:.3f}'
                    })

            avg_epoch_reward = np.mean(epoch_rewards)
            avg_epoch_loss = np.mean(epoch_losses)

            self.training_history['episode_rewards'].append(avg_epoch_reward)
            self.training_history['losses'].append(avg_epoch_loss)

            print(f"\nEpoch {epoch + 1} Summary:")
            print(f"  Avg Recall@10: {avg_epoch_reward:.4f}")
            print(f"  Avg Loss: {avg_epoch_loss:.4f}")
            print("-" * 60)

    def evaluate(self, test_data):
        """
        Evaluate the trained policy on test data.

        Returns:
            metrics: Dict with recall, precision, MRR
        """
        self.policy.eval()

        recalls = []
        precisions = []
        mrrs = []

        print(f"\n{'=' * 60}")
        print(f"Evaluating on {len(test_data)} test queries")
        print(f"{'=' * 60}\n")

        with torch.no_grad():
            for data in tqdm(test_data, desc="Evaluating"):
                query = data['query']
                targets = data['targets']

                # Get candidates
                candidates = self.get_candidates_with_features(query)

                if not candidates:
                    recalls.append(0)
                    precisions.append(0)
                    mrrs.append(0)
                    continue

                # Score with policy
                features_batch = torch.stack([c['features'] for c in candidates]).to(self.device)
                scores = self.policy(features_batch).squeeze()

                # Get top 10
                top_10_indices = torch.topk(scores, min(10, len(scores))).indices
                predicted_top_10 = [candidates[i]['ecli'] for i in top_10_indices.cpu().numpy()]

                # Compute metrics
                recall, precision = compute_reward(predicted_top_10, targets)
                recalls.append(recall)
                precisions.append(precision)

                # MRR
                mrr = 0
                for rank, ecli in enumerate(predicted_top_10, 1):
                    if ecli in targets:
                        mrr = 1.0 / rank
                        break
                mrrs.append(mrr)

        metrics = {
            'recall_at_10': np.mean(recalls),
            'precision_at_10': np.mean(precisions),
            'mrr': np.mean(mrrs)
        }

        print(f"\nEvaluation Results:")
        print(f"  Recall@10:    {metrics['recall_at_10']:.4f}")
        print(f"  Precision@10: {metrics['precision_at_10']:.4f}")
        print(f"  MRR:          {metrics['mrr']:.4f}")
        print(f"{'=' * 60}\n")

        return metrics

    def save_model(self, path):
        """Save the trained policy network."""
        torch.save({
            'model_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_history': self.training_history
        }, path)
        print(f"Model saved to: {path}")

    def load_model(self, path):
        """Load a trained policy network."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_history = checkpoint['training_history']
        print(f"Model loaded from: {path}")


if __name__ == "__main__":
    device = get_device()
    print(f"Using device: {device}")

    # Initialize RAG system
    rag = LegalRAGSystem()

    # Prepare train and test data
    train_data, test_data = prepare_training_data(train_size=0.8, random_seed=42)

    # Trainer
    trainer = RLRAGTrainer(
        rag_system=rag,
        feature_dim=12,
        hidden_dim=64,
        learning_rate=0.001,
        device=device
    )

    # ADD THIS: Pre-compute before training
    #print("\n🔄 Pre-computing candidates (this takes time but only once)...")
    #trainer.precompute_candidates(train_data)

    # Optional: Save cache to disk
    #with open('candidate_cache.pkl', 'wb') as f:
    #    pickle.dump(trainer.candidates_cache, f)
    #print("✓ Cache saved to disk")

    # Benchmark metric
    print("BASELINE EVALUATION (Before Training)")
    # baseline_metrics = trainer.evaluate(test_data[:5])

    # Train the model
    trainer.train(
        data=train_data,
        epochs=20,
        batch_size=5
    )

    # Evaluate
    # Final evaluation
    print("\n" + "=" * 60)
    print("FINAL EVALUATION (After Training)")
    print("=" * 60)
    final_metrics = trainer.evaluate(test_data)

    # Save model
    model_path = os.path.join('models', 'rl_document_scorer.pt')
    os.makedirs('models', exist_ok=True)
    trainer.save_model(model_path)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Final Recall@10:    {final_metrics['recall_at_10']:.4f}")
