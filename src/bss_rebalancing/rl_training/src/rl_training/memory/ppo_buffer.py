import torch
import numpy as np
from torch_geometric.data import Data, Batch
import os
from tqdm import tqdm

# We drop the PairData structure. In DQN, we needed (s, s') to compute TD-errors. In PPO, we process sequences: we store state s, and the return is calculated later from the rewards sequence.
# DQN is off-policy, PPO is on-policy.

class PPOBuffer:
    def __init__(self):
        """
        Rollout buffer for PPO. Unlike DQN, PPO buffer is cleared after every update. 
        """
        self.buffer = []

    def push(self, state, action, logprob, reward, value, done):
        """
        Adds a transition to the buffer.
        
        Parameters:
            - state (Data): Current graph state (PyG Data object).
            - action (int): Action taken by the Actor.
            - logprob (float): Log-probability of that action.
            - reward (float): Reward received.
            - value (float): Value estimated by the Critic for this state.
            - done (bool): Terminal flag.
        """
        # We store everything inside the Data object to use PyG's Batching later
        transition = state.clone() # Clone to avoid reference issues
        
        transition.action = torch.tensor([action], dtype=torch.long)
        transition.logprob = torch.tensor([logprob], dtype=torch.float32)
        transition.reward = torch.tensor([reward], dtype=torch.float32)
        transition.value = torch.tensor([value], dtype=torch.float32)
        transition.done = torch.tensor([done], dtype=torch.float32)

        self.buffer.append(transition)

    def get_all(self):
        """
        Returns the entire rollout as a single large PyG Batch.
        """
        if len(self.buffer) == 0:
            return None
            
        # Converts list of Data objects into one Batch for the GPU
        batch = Batch.from_data_list(self.buffer)
        return batch

    def clear(self):
        """Clears the buffer. Must be called after each PPO update."""
        self.buffer = []

    def __len__(self):
        return len(self.buffer)

    def save_to_files(self, folder_path, chunk_size=1000):
        """
        Saves the current rollout buffer to disk. 
        Useful for offline analysis of the agent's decisions.
        """
        os.makedirs(folder_path, exist_ok=True)
        # PPO buffers are usually smaller than DQN, so we use a smaller default chunk_size
        num_chunks = (len(self.buffer) + chunk_size - 1) // chunk_size

        for i in range(num_chunks):
            chunk = self.buffer[i * chunk_size : (i + 1) * chunk_size]
            file_path = os.path.join(folder_path, f"rollout_chunk_{i}.pt")
        
            # Use torch.save instead of pickle for better compatibility with PyG Data
            torch.save(chunk, file_path)

    def load_from_files(self, folder_path):
        """
        Loads saved rollout data back into the buffer.
        """
        self.buffer = []
        # Look for .pt files (PyTorch) instead of .pkl
        buffer_files = [f for f in os.listdir(folder_path)
                    if f.startswith("rollout_chunk_") and f.endswith(".pt")]
        buffer_files = sorted(buffer_files)

        tbar = tqdm(total=len(buffer_files), desc="Loading rollout files")
        for file_name in buffer_files:
            file_path = os.path.join(folder_path, file_name)
            try:
                # Load back to CPU to avoid filling up GPU memory
                chunk = torch.load(file_path, map_location='cpu')
                self.buffer.extend(chunk)
            except Exception as e:
                print(f"Error: Failed to load file {file_name} due to {e}")
            tbar.update(1)