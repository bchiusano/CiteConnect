import torch.nn as nn
import torch
from torch import optim
from tqdm import tqdm
from test_rag_infloat_multilingual import get_device, clean_ecli, LegalRAGSystem
from rag_pipeline_infloat_multilingual import letters_path
from resources.domain_config import DOMAIN_MAP, ACTIVE_DOMAIN
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from langchain_core.documents import Document
import re
import pickle
from RLRAGTrainer import prepare_training_data


class DocumentScorer(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1))

        # Xavier initialization
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0.1)

    def forward(self, state):
        return self.net(state)


class RLSimple:
    def __init__(self, learning_rate):
        self.rag_system = LegalRAGSystem()
        self.device = torch.device(get_device())
        self.policy_network = DocumentScorer(input_dim=1024 * 2, hidden_dim=128).to(self.device)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=learning_rate)

    # Calculating the reward (based on retrieval)
    def calculate_reward(self):
        pass

    # Training the policy network using the reward
    def train_single(self, query, target):
        self.policy_network.train()

        retrieved_ecli = self.rag_system.get_top_10_for_letter(query)

        # The relevance score is the state
        valid_states = [torch.tensor(score, dtype=torch.float32).to(self.device) for ecli, score in retrieved_ecli]

        print("Actual Target: ", target)
        print("Retrieved_ECLI: ", retrieved_ecli)
        print("Valid_states: ", valid_states)

        scores = torch.stack([self.policy_network(state) for state in valid_states]).squeeze(-1)
        probs = torch.softmax(scores, dim=0)
        top_10_indices = torch.topk(scores, min(10, len(scores))).indices
        predicted_top_10 = [retrieved_ecli[i]['ecli'] for i in top_10_indices.cpu().numpy()]

        # debug

        print("Scores: ", scores)
        print("Probs: ", probs)
        print("Predicted_top_10: ", predicted_top_10)

        # TODO: use softmax?
        if not retrieved_ecli:
            print("no ecli list found for this query")
            return 0.0, 0.0


if __name__ == "__main__":
    train_data, test_data = prepare_training_data(train_size=0.8, random_seed=42)

    rl_pipeline = RLSimple(learning_rate=0.0001)

    sample = train_data[0]
    q = sample['query']
    t = sample['targets']
    rl_pipeline.train_single(q, t)
