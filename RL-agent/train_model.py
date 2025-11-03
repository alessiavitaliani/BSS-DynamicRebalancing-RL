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

# ----------------------------------------------------------------------------------------------------------------------

data_path = "data/"
results_path = "results/"
run_id = 999

# if GPU is to be used
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    # "mps" if torch.backends.mps.is_available() else
    "cpu"
)
print("Device available: " + device.__getattribute__("type"))
# SEED SETTING (Some setting are put after the generation of the enviroment)
seed = 43
print(seed)
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
    "num_episodes": 200,                            # Total number of training episodes
    "batch_size": 64,                               # Batch size for replay buffer sampling
    "replay_buffer_capacity": int(1e5),             # Capacity of replay buffer: 0.1 million transitions
    # "input_dimentions": 72,
    "gamma": 0.95,                                  # Discount factor
    "epsilon_start": 1.0,                           # Starting exploration rate
    "epsilon_delta": 0.05,                          # Epsilon decay rate
    "epsilon_end": 0.01,                            # Minimum exploration rate
    "epsilon_decay": 1e-5,                          # Epsilon decay constant
    "exploration_time": 0.6,                        # Fraction of total training time for exploration
    "lr": 1e-4,                                     # Learning rate
    "total_timeslots": 56,                          # Total number of time slots in one episode (1 month)
    "maximum_number_of_bikes": 250,                 # Maximum number of bikes in the system
    "results_path": results_path,                   # Path to save results
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
    global_criticals = []
    reward_tracking = [[] for _ in range(len(Actions))]
    epsilon_per_timeslot = []
    deployed_bikes = []

    # Reset environment and agent state
    options ={
        'total_timeslots': params["total_timeslots"],
        'maximum_number_of_bikes': params["maximum_number_of_bikes"],
        'discount_factor': params["gamma"],
        'logging': enable_logging,
        'depot_id': 18,         # 491 back
        # 'initial_cell': 18,     # 185 back
        'reward_params': reward_params,
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
        'visits': 0,
        'operations': 0,
        'rebalanced': 0,
        'failures': 0,
        'critic_score': 0.0,
        'num_bikes': 0.0,
    } # to see new custom feature on the results, remember to add them in the webserver
    cell_graph = initialize_cells_subgraph(cell_dict, nodes_dict, distance_matrix, custom_features)

    not_done = True

    iterations = 0
    while not_done:
        # Prepare the state for the agent
        # print(state.x)
        single_state = Data(
            x=state.x.to(device),
            edge_index=state.edge_index.to(device),
            edge_attr=state.edge_attr.to(device),
            agent_state=torch.tensor(state.agent_state, dtype=torch.float32).unsqueeze(dim=0).to(device),
            batch=torch.zeros(state.x.size(0), dtype=torch.long).to(device),
        )

        # CENSORING
        # Remove actions that are not allowed
        # avoid_actions = []

        # Avoid moving in directions where the truck cannot move
        # truck_adjacent_cells = info['truck_neighbor_cells']
        
        # if truck_adjacent_cells['down'] is None:
        #     avoid_actions.append(Actions.DOWN.value)

        # if truck_adjacent_cells['up'] is None:
        #     avoid_actions.append(Actions.UP.value)

        # if truck_adjacent_cells['left'] is None:
        #     avoid_actions.append(Actions.LEFT.value)

        # if truck_adjacent_cells['right'] is None:
        #     avoid_actions.append(Actions.RIGHT.value)

        # Select an action using the agent
        # action = agent.select_action(single_state, avoid_action=avoid_actions)
        action = agent.select_action(single_state, epsilon_greedy=True)#, avoid_action=avoid_actions)

        # Step the environment with the chosen action
        agent_state, reward, done, timeslot_terminated, info = env.step(action)

        # Update state with new information
        env_cells_subgraph = info['cells_subgraph']
        next_state = convert_graph_to_data(env_cells_subgraph, node_features=node_features)
        next_state.agent_state = agent_state
        state.steps = info['steps']

        # Store the transition in the replay buffer
        agent.replay_buffer.push(state, action, reward, next_state, done)

        # Train the agent with a batch from the replay buffer
        loss = agent.train_step(batch_size)

        # Update the state
        state = next_state

        # Update the metrics
        action_per_step.append(action)
        reward_tracking[action].append(reward)
        losses.append(loss)
        global_criticals.append(info['global_critic_score'])

        total_reward += reward
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

            # Update target network periodically
            if timeslots_completed % 8 == 0:
                agent.update_target_network()
            if agent.epsilon <= 0.05:
                agent.epsilon = 0.05
            elif agent.epsilon > 0.05:
                agent.update_epsilon()
                # agent.epsilon = 1 - episode*0.01

            # Get Q-values for the current state
            with torch.no_grad():
                q_values = agent.get_q_values(single_state)
                q_values_per_timeslot.append(q_values[0].squeeze().cpu().numpy())
                del q_values

            # Record metrics for the current time slot
            rewards_per_timeslot.append(total_reward/360)
            failures_per_timeslot.append(total_failures)
            epsilon_per_timeslot.append(agent.epsilon)
            deployed_bikes.append(info['number_of_system_bikes'])

            # Reset time slot metrics
            total_reward = 0
            total_failures = 0
            timeslot = 0 if timeslot == 7 else timeslot + 1

            # Update progress bar
            if isinstance(tbar, (tqdm, tqdm_telegram)):
                # tbar.set_description(f"Training Episode {episode}, Week {info['week'] % 52}, {info['day'].capitalize()} "
                #                      f"at {convert_seconds_to_hours_minutes(info['time'])}")
                tbar.set_description(f"Run {run_id} cuda {args.cuda_device}. Epis {episode}, Week {info['week'] % 52}, {info['day'].capitalize()} "
                                    f"at {convert_seconds_to_hours_minutes(info['time'])}")
                tbar.set_postfix({'eps': agent.epsilon})

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
        "total_trips": info["total_trips"],
        "total_invalid": info["total_invalid"],
        "q_values_per_timeslot": q_values_per_timeslot,
        "action_per_step": action_per_step,
        "global_criticals": global_criticals,
        "losses": losses,
        "reward_tracking": reward_tracking,
        "epsilon_per_timeslot": epsilon_per_timeslot,
        "deployed_bikes": deployed_bikes,
        "cell_subgraph": cell_graph,
    }

    return results


def validate_dqn(env: gym, agent: DQNAgent, episode: int, tbar: tqdm | tqdm_telegram, enable_val_logging: bool) -> dict:
    # Initialize episode metrics
    timeslot = 0
    timeslots_completed = 0
    rewards_per_timeslot = []
    total_reward = 0
    failures_per_timeslot = []
    total_failures = 0
    action_per_step = []
    global_criticals = []
    reward_tracking = [[] for _ in range(len(Actions))]
    deployed_bikes = []

    # Reset environment and agent state
    options ={
        'total_timeslots': params["total_timeslots"],
        'maximum_number_of_bikes': params["maximum_number_of_bikes"],
        'discount_factor': params["gamma"],
        'logging': enable_val_logging,
        'depot_id': 18,         # 491 back
        'initial_cell': 18,     # 185 back
        'reward_params': reward_params,
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
        'visits': 0,
        'operations': 0,
        'rebalanced': 0,
        'failures': 0,
        'critic_score': 0.0,
        'num_bikes': 0.0,
    }
    cell_graph = initialize_cells_subgraph(cell_dict, nodes_dict, distance_matrix, custom_features)

    not_done = True

    previous_epsilon = agent.epsilon

    iterations = 0
    while not_done:
        # Prepare the state for the agent
        single_state = Data(
            x=state.x.to(device),
            edge_index=state.edge_index.to(device),
            edge_attr=state.edge_attr.to(device),
            agent_state=torch.tensor(state.agent_state, dtype=torch.float32).unsqueeze(dim=0).to(device),
            batch=torch.zeros(state.x.size(0), dtype=torch.long).to(device),
        )

        # CENSORING
        # Remove actions that are not allowed
        # avoid_actions = []

        # Avoid moving in directions where the truck cannot move
        # truck_adjacent_cells = info['truck_neighbor_cells']

        # if truck_adjacent_cells['down'] is None:
        #     avoid_actions.append(Actions.DOWN.value)

        # if truck_adjacent_cells['up'] is None:
        #     avoid_actions.append(Actions.UP.value)

        # if truck_adjacent_cells['left'] is None:
        #     avoid_actions.append(Actions.LEFT.value)

        # if truck_adjacent_cells['right'] is None:
        #     avoid_actions.append(Actions.RIGHT.value)

        # Select an action using the agent
        agent.epsilon = 0.05
        action = agent.select_action(single_state, epsilon_greedy=True)
        # action = agent.select_action(single_state, greedy=True, avoid_action=avoid_actions)

        # Step the environment with the chosen action
        agent_state, reward, done, timeslot_terminated, info = env.step(action)

        # Update state with new information
        env_cells_subgraph = info['cells_subgraph']
        next_state = convert_graph_to_data(env_cells_subgraph, node_features=node_features)
        next_state.agent_state = agent_state
        next_state.steps = info['steps']

        # Update the state
        state = next_state

        # Update the metrics
        action_per_step.append(action)
        global_criticals.append(info['global_critic_score'])
        reward_tracking[action].append(reward)

        total_reward += reward
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
            rewards_per_timeslot.append(total_reward/360)
            failures_per_timeslot.append(total_failures)
            deployed_bikes.append(info['number_of_system_bikes'])

            # Reset time slot metrics
            total_reward = 0
            total_failures = 0
            timeslot = 0 if timeslot == 7 else timeslot + 1

            # Update progress bar
            tbar.set_description(f"Validating Episode {episode}, Week {info['week'] % 52}, {info['day'].capitalize()} "
                                 f"at {convert_seconds_to_hours_minutes(info['time'])}")
            tbar.set_postfix({'eps': agent.epsilon})

            # tbar.update(1)

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

    agent.epsilon = previous_epsilon

    results = {
        "rewards_per_timeslot": rewards_per_timeslot,
        "failures_per_timeslot": failures_per_timeslot,
        "total_trips": info["total_trips"],
        "total_invalid": info["total_invalid"],
        "action_per_step": action_per_step,
        "global_criticals": global_criticals,
        "reward_tracking": reward_tracking,
        "deployed_bikes": deployed_bikes,
        "cell_subgraph": cell_graph,
    }

    return results


# ----------------------------------------------------------------------------------------------------------------------

def save_checkpoint(main_variables: dict, agent: DQNAgent, buffer: ReplayBuffer):
    checkpoint_path = results_path + 'checkpoints_' + str(run_id) + '/'
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
        # print(f"Directory '{checkpoint_path}' created.")

    with open(checkpoint_path + 'main_variables.pkl', 'wb') as f:
        pickle.dump(main_variables, f)
    agent.save_checkpoint(checkpoint_path + 'agent.pt')
    buffer.save_to_files(checkpoint_path + 'replay_buffer/')

    print("Checkpoint saved.")


def restore_checkpoint(agent: DQNAgent, buffer: ReplayBuffer) -> dict:
    print("Restoring checkpoint...", end=' ')
    checkpoint_path = results_path + 'checkpoints_' + str(run_id) + '/'

    with open(checkpoint_path + 'main_variables.pkl', 'rb') as f:
        main_variables = pickle.load(f)
    print("Main variables loaded.", end=' ')
    agent.load_checkpoint(checkpoint_path + 'agent.pt')
    print("Agent loaded.", end=' ')
    buffer.load_from_files(checkpoint_path + 'replay_buffer/')
    print("Replay buffer loaded.")

    return main_variables

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

    # At 60% of the total timeslots (60% of the training) the epsilon should be 0.1
    params["epsilon_decay"] = (params["exploration_time"] * params["num_episodes"] * params["total_timeslots"]) / np.log(10)
    print(f"{params}\n")
    print(f"{reward_params}\n")

    # Create results path
    training_results_path = str(results_path + 'training_' + str(run_id) + '/')
    validation_results_path = str(results_path + 'validation_' + str(run_id) + '/')

    # Remove existing results
    if os.path.exists(training_results_path):
        shutil.rmtree(training_results_path)
    if os.path.exists(validation_results_path):
        shutil.rmtree(validation_results_path)

    if not os.path.exists(training_results_path):
        os.makedirs(training_results_path)

    if not os.path.exists(validation_results_path):
        os.makedirs(validation_results_path)

    # Create the environment
    env = gym.make('gymnasium_env/FullyDynamicEnv-v0', data_path=data_path, results_path=training_results_path)
    env.unwrapped.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    # Set up replay buffer
    replay_buffer = ReplayBuffer(params["replay_buffer_capacity"])

    # Create background thread for checkpointing
    checkpoint_background_thread = None

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

    # Train the agent using the training loop
    try:
        # Progress bar for the episode
        if enable_telegram:
            tbar = tqdm_telegram(
                range(params["num_episodes"]),
                desc="Training is starting - TELEGRAM ENABLED ",
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
                desc="Training computation is starting ... ... ... ...",
                initial=starting_episode*params["total_timeslots"],
                position=0,
                leave=True,
                dynamic_ncols=True
            )

        logger = setup_logger('training_logger', training_results_path + 'training.log', level=logging.INFO)

        logger.info(f"Training started with the following parameters: {params}")
        if enable_telegram:
            tbar.set_description(f"Run {run_id} cuda {args.cuda_device}. Epis 0")
            tbar.set_postfix({'eps': agent.epsilon})

        # Train and validation loop
        best_validation_score = 1e4
        for episode in range(starting_episode, params["num_episodes"]):
            # if episode > 133:
            #     enable_logging = True
            # Train the agent for one episode, TRAIN_DQN
            if enable_telegram:
                training_results = train_dqn(env, agent, params["batch_size"], episode)
                tbar.set_description(f"Run {run_id} cuda {args.cuda_device}. Epis {episode}")
                tbar.set_postfix({'eps': agent.epsilon})
                tbar.update(1)
            else:
                training_results = train_dqn(env, agent, params["batch_size"], episode, tbar)

            # Save training result lists
            ep_results_path = training_results_path + 'data/'+ str(episode).zfill(2) + '/'
            if not os.path.exists(ep_results_path):
                os.makedirs(ep_results_path)
            for key, value in training_results.items():
                with open(ep_results_path + key + '.pkl', 'wb') as f:
                    pickle.dump(value, f)

            total_trips = training_results['total_trips']
            total_invalid = training_results['total_invalid']
            total_train_failures = sum(training_results['failures_per_timeslot'])
            mean_train_failures = total_train_failures / params["total_timeslots"]

            logger.info(
                f"Episode {episode}: Mean Failures = {mean_train_failures:.2f}, Total Failures = {total_train_failures}/{total_trips}, Invalid Actions = {total_invalid}"
            )

            # Save checkpoint if the training and validation score is better
            if (episode%10 == 0 and (agent.epsilon < 0.15 or mean_train_failures < 10)) or episode == params["num_episodes"]: 
                enable_val_logging = enable_logging 
                if episode > params["num_episodes"]*0.9: 
                    enable_val_logging = True
                # validate the training with a greedy validation, VALIDATE_DQN             
                validation_results = validate_dqn(env, agent, episode, tbar, enable_val_logging)

                total_trips = training_results['total_trips']
                total_invalid = training_results['total_invalid']
                val_failures_per_timeslot = validation_results['failures_per_timeslot']
                mean_val_failures_per_timeslot = sum(val_failures_per_timeslot) / params["total_timeslots"]
                total_val_failures = sum(val_failures_per_timeslot)

                logger.info(
                    f"Episode {episode}: Mean Validation Failures = {mean_val_failures_per_timeslot:.2f}, "
                    f"Total Validation Failures = {total_val_failures}/{total_trips}, Invalid Actions = {total_invalid}, "
                    f"Best Validation Failures = {best_validation_score}"
                )

                if total_val_failures < best_validation_score or episode > 125:
                    if total_val_failures < best_validation_score:
                        best_validation_score = total_val_failures

                    # Save validation result lists
                    ep_results_path = validation_results_path + 'data/' + str(episode).zfill(2) + '/'
                    if not os.path.exists(ep_results_path):
                        os.makedirs(ep_results_path)
                    for key, value in validation_results.items():
                        with open(ep_results_path + key + '.pkl', 'wb') as f:
                            pickle.dump(value, f)

                    # Save the trained model
                    trained_models_folder = validation_results_path + 'trained_models/' + str(episode).zfill(2) + '/'
                    if not os.path.exists(trained_models_folder):
                        os.makedirs(trained_models_folder)
                    agent.save_model(trained_models_folder + 'trained_agent.pt')

            # Save checkpoint
            if enable_checkpoint:
                main_variables = {
                    'episode': episode,
                    'tbar_progress': tbar.n,
                }
                checkpoint_background_thread = threading.Thread(target=save_checkpoint,
                                                                args=(main_variables, agent, replay_buffer))
                checkpoint_background_thread.start()

            gc.collect()

        tbar.set_description(f"Run {run_id} cuda {args.cuda_device}. Epis {episode}")
        tbar.set_postfix({'eps': agent.epsilon})
        tbar.close()
    except Exception as e:
        if enable_telegram:
            send_telegram_message(f"An error occurred during training {run_id}: {e}", BOT_TOKEN, CHAT_ID)
        raise e
    except KeyboardInterrupt:
        if enable_telegram:
            send_telegram_message(f"Training {run_id} interrupted by user.", BOT_TOKEN, CHAT_ID)
        print("\nTraining interrupted by user.")
        return

    # Save the trained model
    trained_models_folder = results_path + 'trained_models'

    if not os.path.exists(trained_models_folder):
        os.makedirs(trained_models_folder)
        print(f"Directory '{trained_models_folder}' created.")

    agent.save_model(trained_models_folder + '/DuelingDQN.pt')

    # Wait for the background threads to finish
    if checkpoint_background_thread is not None:
        checkpoint_background_thread.join()

    # Print the rewards after training
    print(f"\nTraining {run_id} completed.")
    if enable_telegram:
        send_telegram_message(f"Training run_id = {run_id} completed.", BOT_TOKEN, CHAT_ID)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the Dueling DQN agent.')
    parser.add_argument('--enable_telegram', action='store_true', help='Enable Telegram notifications.')
    parser.add_argument('--data_path', type=str, default=data_path, help='Path to the data folder.')
    parser.add_argument('--results_path', type=str, default=results_path, help='Path to the results folder.')
    parser.add_argument('--cuda_device', type=int, default=0, help='CUDA device to use.')
    parser.add_argument('--enable_logging', action='store_true', help='Enable logging.')
    parser.add_argument('--enable_checkpoint', action='store_true', help='Enable checkpointing.')
    parser.add_argument('--restore_from_checkpoint', action='store_true', help='Restore from checkpoint.')
    parser.add_argument('--num_episodes', type=int, default=params['num_episodes'], help='Number of episodes to train.')
    parser.add_argument('--run_id', type=int, default=run_id, help='Run ID for the experiment.')
    parser.add_argument('--exploration_time', type=float, default=params['exploration_time'], help='Number of episodes to explore.')

    args = parser.parse_args()

    # Assign variables based on the parsed arguments
    enable_telegram = args.enable_telegram
    data_path = args.data_path
    results_path = args.results_path
    enable_logging = args.enable_logging
    enable_checkpoint = args.enable_checkpoint
    restore_from_checkpoint = args.restore_from_checkpoint
    params["num_episodes"] = args.num_episodes
    run_id = args.run_id
    params["exploration_time"] = args.exploration_time

    # Ensure the data path exists
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"The specified data path does not exist: {data_path}")

    # Set up the device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_device )
    torch.cuda.device_count()  # print 1
    device = torch.device(f"cuda:{args.cuda_device}" if torch.cuda.is_available() else "cpu")

    main()
