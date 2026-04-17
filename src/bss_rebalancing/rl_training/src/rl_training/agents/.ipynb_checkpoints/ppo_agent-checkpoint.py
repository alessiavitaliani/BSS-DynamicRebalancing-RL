import torch
import numpy as np
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

# Difference from DQNAgent:
# - Exploration: In your DQN, we used epsilon. In PPO, the Actor generates a probability distribution: if the entropy of the distribution is high, the agent explores; if it is low, the agent is confident.
# - Reward (GAE): Instead of using the direct Bellman equation (R + gamma*Q), we use the GAE: this calculates how much better an action was than the “average” (V(s)).
# - Epoch-wise updating: While DQN performed a single gradient step per batch, PPO reuses rollout data across multiple epochs (usually 10), making learning much more stable in complex environments.

class PPOAgent:
    def __init__(self, network, lr=3e-4, gamma=0.99, gae_lambda=0.95, 
                 clip_coef=0.2, ent_coef=0.01, vf_coef=0.5, update_epochs=10, device='cpu'):
        """
        Initializes the PPOAgent.
        """
        self.network = network.to(device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr, eps=1e-5)
        
        # Hyperparameters
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_coef = clip_coef
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.update_epochs = update_epochs
        self.device = device

    def select_action(self, state_graph, avoid_action: list = None):
        """
        Selects an action using the stochastic policy (Actor).
        Implements action masking for forbidden rebalancing moves.
        """
        if avoid_action is None:
            avoid_action = []

        self.network.eval()
        with torch.no_grad():
            # Get logits and state value
            features = self.network.extract_features(state_graph)
            logits = self.network.actor(features).squeeze(0)
            value = self.network.critic(features)

            # Masking: set logits of forbidden actions to -infinity
            if avoid_action:
                mask = torch.ones_like(logits, dtype=torch.bool)
                for a in avoid_action:
                    mask[a] = False
                logits[~mask] = -float('inf')

            # Sample from the distribution
            probs = Categorical(logits=logits)
            action = probs.sample()
            logprob = probs.log_prob(action)

        return action.item(), logprob, value

    def update(self, buffer):
        """
        Performs the PPO update using GAE and Clipped Objective.
        """
        self.network.train()
        
        # Get all data from the rollout buffer
        batch = buffer.get_all().to(self.device)
        
        # GAE (Generalized Advantage Estimation):
        # δ_t = r_t + γ * V(s_{t+1}) - V(s_t)
        # A_t = δ_t + (γ * λ) * δ_{t+1} + ...
        with torch.no_grad():
            rewards = batch.reward.flatten()
            values = batch.value.flatten()
            dones = batch.done.flatten()
            
            advantages = torch.zeros_like(rewards).to(self.device)
            lastgaelam = 0
            
            # Recursive GAE calculation
            for t in reversed(range(len(rewards))):
                if t == len(rewards) - 1:
                    nextnonterminal = 1.0 - dones[t]
                    nextvalues = 0 # End of rollout bootstrap
                else:
                    nextnonterminal = 1.0 - dones[t]
                    nextvalues = values[t+1]
                
                delta = rewards[t] + self.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam
            
            returns = advantages + values

        # OPTIMIZATION LOOP (Multiple epochs)
        # Standard PPO reuses the rollout data for N epochs
        for epoch in range(self.update_epochs):
            # Forward pass with current network
            _, newlogprob, entropy, newvalue = self.network.get_action_and_value(batch, batch.action.flatten())
            
            # Policy Ratio:
            # r_t(θ) = π_new(a|s) / π_old(a|s)
            logratio = newlogprob - batch.logprob.flatten()
            ratio = logratio.exp()

            # Normalized Advantages (improves stability)
            mb_advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # PPO Clipped Loss:
            # L^CLIP = min( r_t * A_t, clip(r_t, 1-ε, 1+ε) * A_t )
            pg_loss1 = -mb_advantages * ratio
            pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

            # Value Loss (MSE):
            # L^VF = (V_predicted - V_target)^2
            v_loss = self.vf_coef * ((newvalue.view(-1) - returns) ** 2).mean()

            # Entropy Bonus (prevents premature convergence)
            ent_loss = self.ent_coef * entropy.mean()

            # Loss = L^CLIP - c2 * Entropy + c1 * L^VF
            total_loss = pg_loss + v_loss - ent_loss

            # Gradient Step
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
            self.optimizer.step()

        return pg_loss.item(), v_loss.item(), ent_loss.item()

    def save_model(self, file_path):
        """Save network weights."""
        torch.save(self.network.state_dict(), file_path)

    def load_model(self, file_path):
        """Load network weights."""
        self.network.load_state_dict(torch.load(file_path, map_location=self.device))