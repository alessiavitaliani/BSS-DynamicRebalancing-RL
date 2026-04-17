
import sys
import os
sys.path.extend(['C:\\Users\\Alessia Vitaliani\\Downloads\\Research training\\Bike_sharing\\gymnasium_env\\src', 'C:\\Users\\Alessia Vitaliani\\Downloads\\Research training\\Bike_sharing\\rl_training\\src', 'C:\\Users\\Alessia Vitaliani\\Downloads\\Research training\\Bike_sharing'])

import gymnasium
from gymnasium.envs.registration import register

# Registrazione manuale forzata
try:
    register(
        id='gymnasium_env/FullyDynamicEnv-v0',
        entry_point='gymnasium_env.envs:FullyDynamicEnv',
    )
    print('✅ Namespace gymnasium_env registrato correttamente.')
except Exception as e:
    print(f'ℹ️ Nota sulla registrazione: {e}')

# Avvio del tuo script originale
import rl_training.train as train
if __name__ == "__main__":
    train.main()
