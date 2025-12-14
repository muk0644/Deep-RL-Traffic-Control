import numpy as np
from utils import *
import os
from utils import import_train_configuration, import_test_configuration, set_train_path, set_test_path
from shutil import copyfile
from collections import deque
import matplotlib.pyplot as plt

# Load configuration from INI file
config = import_train_configuration(config_file='training_settings.ini')

# If not training, load test configuration
if not config['is_train']:
    config = import_test_configuration(config_file_path=config['test_model_path_name'])

# Import environment only if training is active
if config['is_train']:
    from Environment.SUMO_train import SUMO
    env = SUMO()
else:
    # For test mode, import appropriate environment
    pass

print(f'State shape: {env.num_states}')
print(f'Number of actions: {env.action_space.n}')

# Initialize agent (DQN Agent)
if config['agent_type'] == 'DQN':
    from dqn_agent import DQNAgent
    agent = DQNAgent(env.num_states, env.action_space.n, config['hidden_dim'], config['fixed_action_space'], env.TL_list,
                     config['memory_size_max'], config['batch_size'], config['gamma'], config['tau'],
                     config['learning_rate'], config['target_update'])

# Set storage paths (training or test)
if config['is_train']:
    path = set_train_path(config['models_path_name'])
    print(f'Training results will be saved in: {path}')
    os.makedirs(path, exist_ok=True)
    copyfile(src='training_settings.ini', dst=os.path.join(path, 'training_settings.ini'))
else:
    test_path, plot_path = set_test_path(config['test_model_path_name'])
    print(f'Test results will be saved in: {plot_path}')

# DQN training function
def DQNRL(n_episodes=config['total_episodes'], max_t=config['max_steps'] + 1000, eps_start=config['eps_start'],
          eps_end=config['eps_end'], eps_decay=config['eps_decay'], single_agent=config['single_agent']):
    """
    Train DQN agent for traffic signal control.
    
    Args:
        n_episodes: Number of training episodes
        max_t: Maximum steps per episode
        eps_start: Starting epsilon for exploration
        eps_end: Minimum epsilon value
        eps_decay: Epsilon decay rate per episode
        single_agent: Single vs multi-agent mode
    
    Returns:
        scores: List of episode scores
    """
    scores = []  # Scores for all episodes
    scores_window = deque(maxlen=100)  # Sliding window for average
    eps = eps_start  # Starting epsilon value
    episode_start = 1 if config['is_train'] else n_episodes + 1
    episode_end = n_episodes + 1 if config['is_train'] else n_episodes + 11

    for i_episode in range(episode_start, episode_end):
        # Generate new route file for this episode
        env._TrafficGen.generate_routefile(model_path=env.model_path, model_id=env.model_id, seed=i_episode)
        state = env.reset()
        score = 0

        for t in range(max_t):
            # ---- Reinforcement Learning Step ----
            action = agent.act(np.array(state), eps)                  # Select action
            next_state, reward, done, _ = env.step(action)           # Environment step
            agent.step(state, action, reward, next_state, done)      # Store experience and learn
            state = next_state                                       # Update state
            # -------------------------------------

            score += sum(list(reward.values()))
            if done:
                break

        scores_window.append(score)
        scores.append(score)

        # Decay epsilon (exploration to exploitation)
        eps = max(eps_end, eps_decay * eps)

        print(f'\rEpisode {i_episode}\tAverage Score: {np.mean(scores_window):.2f}', end="")
        if i_episode % 100 == 0:
            print(f'\rEpisode {i_episode}\tAverage Score: {np.mean(scores_window):.2f}')

    env.reset()
    env.close()

    return scores

# Execute training if DQN selected
if 'DQN' in config['agent_type']:
    scores = DQNRL()

# Visualize results
import plotly.express as px
fig = px.line(x=np.arange(len(scores)), y=scores)
fig.show()
fig.write_html(os.path.join(path if config['is_train'] else plot_path, 'training_reward.html'))

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.savefig(os.path.join(path if config['is_train'] else plot_path, 'training_reward.png'))
# plt.show()