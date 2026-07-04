# rl_training/networks/__init__.py
from rl_training.networks.dqn import DQN
from rl_training.networks.ppo import PPO

__all__ = ["DQN", "PPO"]