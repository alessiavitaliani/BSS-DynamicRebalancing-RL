# gymnasium_env/envs/__init__.py
from gymnasium_env.envs.fully_dynamic_env import FullyDynamicEnv
from gymnasium_env.envs.static_env import StaticEnv

# Package metadata
__version__ = "1.1.0"
__author__ = "Edoardo Scarpel"

__all__ = ["FullyDynamicEnv", "StaticEnv"]