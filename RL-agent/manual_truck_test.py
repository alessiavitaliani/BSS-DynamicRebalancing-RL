import os, shutil
import pickle
import threading
import torch
import argparse
import gc
import warnings
import logging
import random

import gymnasium_env.register_env

import gymnasium as gym
import numpy as np

from tqdm.contrib.telegram import tqdm as tqdm_telegram
from tqdm import tqdm
from agent import DQNAgent
from utils import convert_graph_to_data, convert_seconds_to_hours_minutes, send_telegram_message, Actions
from replay_memory import ReplayBuffer
from torch_geometric.data import Data
from gymnasium_env.simulator.utils import initialize_cells_subgraph
from torch.nn import functional as F

# ----------------------------------------------------------------------------------------------------------------------

data_path = "data/"
results_path = "results/"
run_id = 999
printing = False
benchmark_mode = True

# if GPU is to be used
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    # "mps" if torch.backends.mps.is_available() else
    "cpu"
)
print("Device available: " + device.__getattribute__("type"))
# SEED SETTING (Some setting are put after the generation of the enviroment)
seed = 31
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
# torch.use_deterministic_algorithms(True)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
"""
 The following is not a real command but it's already satisified from
 - random.seed(seed)
 - np.random.seed(seed)
 - torch.manual_seed(seed)
 >>'chatGPT'
"""
# torch.geometric.seed(seed) 

params = {
    "num_episodes": 5,                              # Total number of test episodes
    "total_timeslots": 56,                          # Total number of time slots in one episode (1 month)
    "maximum_number_of_bikes": 250,                 # Maximum number of bikes in the system
    "results_path": results_path,                   # Path to save results
    "batch_size": 64,                               # Batch size for replay buffer sampling
    "replay_buffer_capacity": int(1e5),             # Capacity of replay buffer: 0.1 million transitions
    "gamma": 0.95,                                  # Discount factor
    "epsilon_start": 1.0,                           # Starting exploration rate
    "epsilon_delta": 0.05,                          # Epsilon decay rate
    "epsilon_end": 0.00,                            # Minimum exploration rate
    "epsilon_decay": 1e-5,                          # Epsilon decay constant
    "exploration_time": 0.6,                        # Fraction of total training time for exploration
    "lr": 1e-4,                                     # Learning rate
    "soft_update": True,                            # Use soft update for target network
    "tau": 0.005,                                   # Tau parameter for soft update
}

reward_params = {
    'W_ZERO_BIKES': 1.0,
    'W_CRITICAL_ZONES': 1.0,
    'W_DROP_PICKUP': 0.9,
    'W_MOVEMENT': 0.7,
    'W_CHARGE_BIKE': 0.9,
    'W_STAY': 0.7,
}

enable_telegram = False
BOT_TOKEN = '7963966364:AAFfUAeWj40QrtL8KRYN8BJIXhHD15bnsNU'    # Aquarum2 bot
CHAT_ID = '671757146'                                           # Alberto Pettena chat

enable_checkpoint = False
restore_from_checkpoint = False
enable_logging = False

formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')

# ----------------------------------------------------------------------------------------------------------------------

def train_dqn(env: gym, agent: DQNAgent, batch_size: int, episode: int, tbar = None) -> dict:
    # Initialize episode metrics
    timeslot = 0
    timeslots_completed = 0
    rewards_per_timeslot = []
    total_reward = 0
    failures_per_timeslot = []
    total_failures = 0
    q_values_per_timeslot = []
    action_per_step = []
    losses = []
    reward_tracking = [[] for _ in range(len(Actions))]
    deployed_bikes = []
    epsilon_per_timeslot = []

    # Reset environment and agent state
    options ={
        'total_timeslots': params["total_timeslots"],
        'maximum_number_of_bikes': params["maximum_number_of_bikes"],
        'logging': enable_logging,
        'depot_id': 18,         # 491 back
        'initial_cell': 18,     # 185 back
    }

    node_features = [
        'truck_cell',
        'critic_score',
        'eligibility_score',
        'bikes',
    ]

    agent_state, info = env.reset(options=options)

    state = convert_graph_to_data(info['cells_subgraph'], node_features=node_features)
    state.agent_state = agent_state
    state.steps = info['steps']
    cell_dict = info['cell_dict']
    nodes_dict = info['nodes_dict']
    distance_matrix = info['distance_matrix']
    custom_features = {
        'visits': 0.0,
        'failures': 0,
        'critic_score': 0.0,
        'num_bikes': 0.0,
    }
    cell_graph = initialize_cells_subgraph(cell_dict, nodes_dict, distance_matrix, custom_features)

    not_done = True

    iterations = 0
    while not_done:
        if printing:
            print(state.x)
        # Prepare the state for the agent
        single_state = Data(
            x=state.x.to(device),
            edge_index=state.edge_index.to(device),
            edge_attr=state.edge_attr.to(device),
            agent_state=torch.tensor(state.agent_state, dtype=torch.float32).unsqueeze(dim=0).to(device),
            batch=torch.zeros(state.x.size(0), dtype=torch.long).to(device),
        )

        if printing:
            # What would have done the agent at this point
            agent_action = agent.select_action(single_state, greedy=True)
            q_values = agent.get_q_values(single_state).squeeze(0)
            print(f"The agent suggest action {agent_action}")
            print(f"Q values of policy net are: {q_values}")

        # Select an action using the agent
        if benchmark_mode:
            action = 0
        else:
            try:
                action = int(input("Choose truck Action.value : "))
            except ValueError:
                print("Invalid input! Please enter a valid integer.")

        # Step the environment with the chosen action
        agent_state, reward, done, timeslot_terminated, info = env.step(action)

        # print the consequences
        if printing:
            print(f"Reward given = {reward}")


        # Update state with new information
        env_cells_subgraph = info['cells_subgraph']
        next_state = convert_graph_to_data(env_cells_subgraph, node_features=node_features)
        next_state.agent_state = agent_state
        state.steps = info['steps']

        if not benchmark_mode:
            # Store the transition in the replay buffer
            agent.replay_buffer.push(state, action, reward, next_state, done)

            # Agent train step simulaiton
            # Prepare the next state for the agent
            single_next_state = Data(
                x=next_state.x.to(device),
                edge_index=next_state.edge_index.to(device),
                edge_attr=next_state.edge_attr.to(device),
                agent_state=torch.tensor(next_state.agent_state, dtype=torch.float32).unsqueeze(dim=0).to(device),
                batch=torch.zeros(next_state.x.size(0), dtype=torch.long).to(device),
            )

            policy_q_value = q_values[action].item()
            agent_next_actions = agent.select_action(single_next_state, greedy=True)
            next_q_values = agent.get_q_values(single_next_state).squeeze(0)
            target_q_value = next_q_values[agent_next_actions].item()

            discount = params["gamma"] ** info['steps']
            bellman_q_target = reward + discount * target_q_value

            print(f"Q(S, a)= {policy_q_value}, Q(S', a')= {target_q_value}, Steps = {info['steps']}")
            print(f"Discount = gamma ** steps = {discount}")
            print(f"bellman_q_target = reward + discount * target_q_values = {bellman_q_target}")


            # Train the agent with a batch from the replay buffer
            _ = agent.train_step(batch_size)

        if printing:
            # Print informations about the next state
            print("-------------------------------------------------------------------")
            print(f"State: Load= {agent_state[0]}"
                f" | is_surplus= {agent_state[1]}"
                f" | is_empty= {agent_state[2]}"
                f" | Criticals = {agent_state[3]}"
                # f" | day= {ohe2num(agent_state[2:9])} | hour= {ohe2num(agent_state[9:33])}"
                f" | prevous_act= {ohe2num(agent_state[4:12])} | cell position= {ohe2cell(agent_state[12:39])} | borders= {agent_state[66:70]}"
                f" | critical cells = {agent_state[39:66]}")
            # print(info['cells_subgraph'])

            if timeslot_terminated:
                print("\n########### TIMESLOT TERMINATED ###############\n")
            if done:
                print("\n########### EPISODE IS DONE ###################\n")

        if not benchmark_mode:
            # Update the state
            state = next_state

        total_failures += sum(info['failures'])

        # Check if the episode is complete
        not_done = not done

        for cell_id, cell in cell_dict.items():
            center_node = cell.get_center_node()
            if center_node in cell_graph:
                cell_graph.nodes[center_node]['critic_score'] += env_cells_subgraph.nodes[center_node]['critic_score']
                cell_graph.nodes[center_node]['num_bikes'] += env_cells_subgraph.nodes[center_node]['bikes']#*params["maximum_number_of_bikes"]
            else:
                raise ValueError(f"Node {center_node} not found in the subgraph.")

        if timeslot_terminated:
            timeslots_completed += 1

            # Record metrics for the current time slot
            failures_per_timeslot.append(total_failures)
            epsilon_per_timeslot.append(0)
            deployed_bikes.append(info['number_of_system_bikes'])

            # Reset time slot metrics
            total_reward = 0
            total_failures = 0
            timeslot = 0 if timeslot == 7 else timeslot + 1

            # Update progress bar
            if isinstance(tbar, (tqdm, tqdm_telegram)):
                # tbar.set_description(f"Test Episode {episode}, Week {info['week'] % 52}, {info['day'].capitalize()} "
                #                      f"at {convert_seconds_to_hours_minutes(info['time'])}")
                tbar.set_description(f"Run {run_id} cuda {args.cuda_device}. Epis {episode}, Week {info['week'] % 52}, {info['day'].capitalize()} "
                                    f"at {convert_seconds_to_hours_minutes(info['time'])}")

                tbar.update(1)

        iterations += 1

        if done:
            for cell_id, cell in cell_dict.items():
                center_node = cell.get_center_node()
                if center_node in cell_graph:
                    cell_graph.nodes[center_node]['visits'] = env_cells_subgraph.nodes[center_node]['visits']
                    cell_graph.nodes[center_node]['operations'] = env_cells_subgraph.nodes[center_node]['operations']
                    cell_graph.nodes[center_node]['rebalanced'] = env_cells_subgraph.nodes[center_node]['rebalanced']
                    cell_graph.nodes[center_node]['failures'] = env_cells_subgraph.nodes[center_node]['failures']
                    cell_graph.nodes[center_node]['critic_score'] = cell_graph.nodes[center_node]['critic_score'] / iterations
                    cell_graph.nodes[center_node]['num_bikes'] = cell_graph.nodes[center_node]['num_bikes'] / iterations
                else:
                    raise ValueError(f"Node {center_node} not found in the subgraph.")

        # Explicitly delete single_state
        del single_state

    env.close()
    torch.cuda.empty_cache()

    results = {
        "rewards_per_timeslot": rewards_per_timeslot,
        "failures_per_timeslot": failures_per_timeslot,
        "q_values_per_timeslot": 0,
        "action_per_step": action_per_step,
        "losses": 0,
        "reward_tracking": reward_tracking,
        "epsilon_per_timeslot": epsilon_per_timeslot,
        "deployed_bikes": deployed_bikes,
        "cell_subgraph": cell_graph,
    }

    return results

# ----------------------------------------------------------------------------------------------------------------------

def ohe2num(ohe):
    num, = np.where(ohe == 1.0)
    return num

def ohe2cell(ohe):
    num, = np.where(ohe == 1.0)
    if 0 <= num <= 3:
        return num
    if 4 <= num <= 8:
        return num+1
    if 9 <= num <= 12:
        return num+2
    if 13 <= num <= 15:
        return num+4
    if 16 <= num <= 19:
        return num+9
    if 20 <= num <= 23:
        return num+10
    if 24 <= num <= 26:
        return num+12

def setup_logger(name, log_file, level=logging.INFO):
    """To setup as many loggers as you want"""

    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger

# ----------------------------------------------------------------------------------------------------------------------

def main():
    warnings.filterwarnings("ignore")
    print(f"Device in use: {device}\n")

    # Print parameters
    print(f"{params}\n")


    # Create results path
    test_results_path = str(results_path + 'test_' + str(run_id) + '/')

    # Remove existing results
    if os.path.exists(test_results_path):
        shutil.rmtree(test_results_path)

    if not os.path.exists(test_results_path):
        os.makedirs(test_results_path)

    # Create the environment
    env = gym.make('gymnasium_env/FullyDynamicEnv-v0', data_path=data_path, results_path=test_results_path)
    env.unwrapped.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    starting_episode = 0

    # Set up replay buffer
    replay_buffer = ReplayBuffer(params["replay_buffer_capacity"])

    # Initialize the DQN agent
    agent = DQNAgent(
        replay_buffer=replay_buffer,
        num_actions=env.action_space.n,
        gamma=params["gamma"],
        epsilon_start=params["epsilon_start"],
        epsilon_end=params["epsilon_end"],
        epsilon_decay=params["epsilon_decay"],
        lr=params["lr"],
        device=device,
        tau=params["tau"],
        soft_update=params["soft_update"],
        # input_dim=params["input_dimension"],
    )

    # Restore from checkpoint
    starting_episode = 0
    if restore_from_checkpoint:
        main_variables = restore_checkpoint(agent, replay_buffer)
        starting_episode = main_variables['episode'] + 1
        print(f"Restored from checkpoint. Resuming training from episode {starting_episode}.")

    # Train the agent using the test loop
    try:
        # Progress bar for the episode
        if enable_telegram:
            tbar = tqdm_telegram(
                range(params["num_episodes"]),
                desc="Test is starting - TELEGRAM ENABLED ",
                initial=starting_episode,
                position=0,
                leave=True,
                dynamic_ncols=True,
                token=BOT_TOKEN,
                chat_id=CHAT_ID
            )
        else:
            tbar = tqdm(
                range(params["total_timeslots"]*params["num_episodes"]),
                desc="Test computation is starting ... ... ... ...",
                initial=starting_episode*params["total_timeslots"],
                position=0,
                leave=False,
                dynamic_ncols=True
            )

        logger = setup_logger('Test_logger', test_results_path + 'test.log', level=logging.INFO)

        logger.info(f"Test started with the following parameters: {params}")
        if enable_telegram:
            tbar.set_description(f"Run {run_id} cuda {args.cuda_device}. Epis 0")

        # Test episodes loop
        for episode in range(starting_episode, params["num_episodes"]):
            # Test one episode
            if enable_telegram:
                test_results = train_dqn(env, agent, params["batch_size"], episode)
                tbar.set_description(f"Run {run_id} cuda {args.cuda_device}. Epis {episode}")
                tbar.update(1)
            else:
                test_results = train_dqn(env, agent, params["batch_size"], episode, tbar)

            # Save test result lists
            ep_results_path = test_results_path + 'data/'+ str(episode).zfill(2) + '/'
            if not os.path.exists(ep_results_path):
                os.makedirs(ep_results_path)
            for key, value in test_results.items():
                with open(ep_results_path + key + '.pkl', 'wb') as f:
                    pickle.dump(value, f)

            total_train_failures = sum(test_results['failures_per_timeslot'])
            mean_train_failures = total_train_failures / params["total_timeslots"]

            logger.info(
                f"Episode {episode}: Mean Failures = {mean_train_failures:.2f}, Total Failures = {total_train_failures}"
            )

            gc.collect()

        tbar.set_description(f"Run {run_id} cuda {args.cuda_device}. Epis {episode}")
        tbar.close()
    except Exception as e:
        if enable_telegram:
            send_telegram_message(f"An error occurred during test: {e}", BOT_TOKEN, CHAT_ID)
        raise e
    except KeyboardInterrupt:
        if enable_telegram:
            send_telegram_message("Test interrupted by user.", BOT_TOKEN, CHAT_ID)
        print("\nTest interrupted by user.")
        return

    # Print test is completed
    print(f"\nTest {run_id} completed.")
    if enable_telegram:
        send_telegram_message(f"Test run_id = {run_id} completed.", BOT_TOKEN, CHAT_ID)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the Dueling DQN agent.')
    parser.add_argument('--enable_telegram', action='store_true', help='Enable Telegram notifications.')
    parser.add_argument('--data_path', type=str, default=data_path, help='Path to the data folder.')
    parser.add_argument('--results_path', type=str, default=results_path, help='Path to the results folder.')
    parser.add_argument('--cuda_device', type=int, default=1, help='CUDA device to use.')
    parser.add_argument('--enable_logging', action='store_true', help='Enable logging.')
    parser.add_argument('--restore_from_checkpoint', action='store_true', help='Restore from checkpoint.')
    parser.add_argument('--num_episodes', type=int, default=params['num_episodes'], help='Number of episodes to train.')
    parser.add_argument('--run_id', type=int, default=run_id, help='Run ID for the experiment.')

    args = parser.parse_args()

    # Assign variables based on the parsed arguments
    enable_telegram = args.enable_telegram
    data_path = args.data_path
    results_path = args.results_path
    enable_logging = True
    restore_from_checkpoint = args.restore_from_checkpoint
    params["num_episodes"] = args.num_episodes
    run_id = args.run_id

    # Ensure the data path exists
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"The specified data path does not exist: {data_path}")

    # Set up the device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_device )
    torch.cuda.device_count()  # print 1
    device = torch.device(f"cuda:{args.cuda_device}" if torch.cuda.is_available() else "cpu")

    main()
