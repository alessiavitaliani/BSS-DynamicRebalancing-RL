import torch.nn as nn
import torch
from torch_geometric.nn.conv import GATv2Conv
from torch_geometric.nn.aggr import AttentionalAggregation
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, num_actions: int, observation_space_len = 256):
        super(DQN, self).__init__()

        # Store device (will be set when .to(device) is called)
        self.device = torch.device('cpu')

        # First GAT layer:  3 input features -> 64 features, heads=4 -> 64 * 4 = 256 if concat=True
        self.gat1 = GATv2Conv(
            # # MODIFY THE FIRST PARAMETER wrt THE DIMENTION OF THE GRAPH OBSERVATION SPACE
            in_channels=4,

            out_channels=64,
            heads=4,
            edge_dim=1,
            concat=True
        )

        # Second GAT layer: 256 -> 64 features, heads=4 -> 64 * 4 = 256
        self.gat2 = GATv2Conv(
            in_channels=256,
            out_channels=64,
            heads=4,
            edge_dim=1,
            concat=True
        )

        # Optional third GAT layer: 256 -> 128 features, heads=2 -> 128 * 2 = 256
        self.gat3 = GATv2Conv(
            in_channels=256,
            out_channels=128,
            heads=2,
            edge_dim=1,
            concat=True
        )

        # ------------------------------------------------------------------------------
        # 2) Pooling Layer (GlobalAttention)
        # ------------------------------------------------------------------------------
        self.pooling_gate_nn = nn.Linear(256, 1)
        self.global_attention_pool = AttentionalAggregation(
            gate_nn=self.pooling_gate_nn
        )  # returns (batch_size, 256)

        # ------------------------------------------------------------------------------
        # 3) Fully Connected Layers for Graph Embedding
        # ------------------------------------------------------------------------------
        self.graph_fc = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )

        # ------------------------------------------------------------------------------
        # 4) MLP for Agent State
        # ------------------------------------------------------------------------------
        self.agent_fc = nn.Sequential(
            nn.Linear(observation_space_len, 256),

            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64)  # final agent embedding
        )

        # ------------------------------------------------------------------------------
        # 5) Fusion & Q-Value Output
        # ------------------------------------------------------------------------------
        self.fc_output = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions)
        )

        # Initialize weights
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)

    def forward(self, batch, key=None):
        if key == 's':
            x, edge_index, edge_attr, agent_state, pool_batch = (
                batch.x_s,
                batch.edge_index_s,
                batch.edge_attr_s,
                batch.agent_state,
                batch.x_s_batch
            )
        elif key == 't':
            x, edge_index, edge_attr, agent_state, pool_batch = (
                batch.x_t,
                batch.edge_index_t,
                batch.edge_attr_t,
                batch.agent_next_state,
                batch.x_t_batch
            )
        else:
            x, edge_index, edge_attr, agent_state, pool_batch = (
                batch.x,
                batch.edge_index,
                batch.edge_attr,
                batch.agent_state,
                batch.batch
            )

        # -----------------------------
        # 1) Graph Embedding with GAT
        # -----------------------------
        x = self.gat1(x, edge_index, edge_attr)
        x = F.relu(x)

        x = self.gat2(x, edge_index, edge_attr)
        x = F.relu(x)

        x = self.gat3(x, edge_index, edge_attr)
        x = F.relu(x)

        # ---------------------------------------
        # 2) Graph-Level Embedding (GlobalAttention)
        # ---------------------------------------
        x = self.global_attention_pool(x, pool_batch)  # shape => (batch_size, 256)
        x = self.graph_fc(x)  # shape => (batch_size, 64)

        # -----------------------------
        # 3) Agent State Path
        # -----------------------------
        agent_state = self.agent_fc(agent_state)  # shape => (batch_size, 64)

        # -----------------------------
        # 4) Fusion
        # -----------------------------
        fused = torch.cat([x, agent_state], dim=-1)  # shape => (batch_size, 128)

        # -----------------------------
        # 5) Q-Value Output
        # -----------------------------
        q_values = self.fc_output(fused)

        return q_values

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

    def update_agent_fc_input_dim(self, new_input_dim: int):
        """
        Update the input dimension of the agent_fc MLP.
        Keeps the new layers on the same device as the model.
        """

        self.agent_fc = nn.Sequential(
            nn.Linear(new_input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64)  # final agent embedding
        )

        # Move to correct device
        self.agent_fc = self.agent_fc.to(self.device)

        # Initialize only the new agent_fc weights
        for layer in self.agent_fc.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
