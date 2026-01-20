import torch.nn as nn


class DocumentRelevanceScorer(nn.Module):
    def __init__(self, feature_dim=12, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # Output 0-1 relevance probability
        )

        # Xavier initialization
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0.1)

    def forward(self, features):
        """
        Args:
            features: [batch_size, feature_dim] tensor
        Returns:
            scores: [batch_size, 1] relevance scores
        """
        return self.net(features)