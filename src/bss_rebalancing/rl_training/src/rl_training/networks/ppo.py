import torch.nn as nn
import torch
from torch_geometric.nn.conv import GATv2Conv
from torch_geometric.nn import global_mean_pool
from torch.distributions.categorical import Categorical

class PPO(nn.Module):
    """
    Actor-Critic Network for Graph-based BSS Rebalancing.
    
    Parameters:
    - num_actions: Number of possible rebalancing moves (cells/stations).
    - node_features: Input features per node (bikes, demand, etc.).
    """
    def __init__(self, num_actions: int, node_features: int = 4):
        super(PPO, self).__init__()

        # Store device (will be set when .to(device) is called)
        self.device = torch.device('cpu')

        # First GAT layer:  3 input features -> 64 features, heads=4 -> 64 * 4 = 256 if concat=True
        self.gat1 = GATv2Conv(in_channels=node_features, out_channels=64, heads=4, edge_dim=1, concat=True)
        # Second GAT layer: 256 -> 64 features, heads=4 -> 64 * 4 = 256
        self.gat2 = GATv2Conv(in_channels=256, out_channels=64, heads=4, edge_dim=1, concat=True)

        # Critic (Estimates a scalar value V(s), representing the expected total return starting from that state):
        # Maps the global graph embedding to a single scalar reward estimate
        self.critic = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        # Actor (Policy pi(a|s)):
        # Outputs logits for a probability distribution over actions
        # Maps the global graph embedding to action probabilities
        self.actor = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions) # If action is choosing a cell/station 
        )

    def extract_features(self, batch_data):
        """
        Processes graph data through GAT layers and performs global pooling.
        """
        x, edge_index, edge_attr = batch_data.x, batch_data.edge_index, batch_data.edge_attr
        
        # Graph Convolutions
        h = torch.relu(self.gat1(x, edge_index, edge_attr))
        h = torch.relu(self.gat2(h, edge_index, edge_attr))
        
        # Global Pooling (Batch handling for PyG)
        batch_idx = batch_data.batch if hasattr(batch_data, 'batch') and batch_data.batch is not None \
                else torch.zeros(h.size(0), dtype=torch.long, device=self.device)
        
        pooled_h = global_mean_pool(h, batch_idx)
        return pooled_h

    def get_value(self, graph_data):
        """Returns only the state value (used during GAE calculation)."""
        features = self.extract_features(graph_data)
        return self.critic(features)

    def get_action_and_value(self, graph_data, action=None):
        """
        Main method for both rollout collection and optimization.
        Returns: action, log_prob, entropy, and state_value.
        """
        features = self.extract_features(graph_data)
        
        logits = self.actor(features)
        probs = Categorical(logits=logits)
        
        if action is None:
            action = probs.sample()
            
        return action, probs.log_prob(action), probs.entropy(), self.critic(features)

    def to(self, device):
        """Override to() to track device."""
        self.device = device
        return super().to(device)

    def cuda(self, device=None):
        """Override cuda() to track device."""
        self.device = torch.device('cuda' if device is None else f'cuda:{device}')
        return super().cuda(device)

    def cpu(self):
        """Override cpu() to track device."""
        self.device = torch.device('cpu')
        return super().cpu()