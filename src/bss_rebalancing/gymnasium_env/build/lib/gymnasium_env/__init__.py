"""
BSS Gymnasium Environment

Gymnasium environments for the BSS Dynamic Rebalancing RL project.
"""

__version__ = "1.1.0"
__author__ = "Edoardo Scarpel"

from gymnasium.envs.registration import register

register(
    id="gymnasium_env/FullyDynamicEnv-v0",
    entry_point="gymnasium_env.envs:FullyDynamicEnv",
    kwargs={'data_path': 'data/', 'results_path': 'results/', 'seed': 42, 'logging_enabled': False},
)

register(
    id="gymnasium_env/StaticEnv-v0",
    entry_point="gymnasium_env.envs:StaticEnv",
    kwargs={'data_path': 'data/', 'results_path': 'results/', 'seed': 42},
)

from gymnasium_env.envs import FullyDynamicEnv, StaticEnv

__all__ = ["FullyDynamicEnv", "StaticEnv"]
