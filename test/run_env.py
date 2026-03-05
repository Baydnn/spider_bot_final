import numpy as np
import os
import yaml

from ppo import PPOAgent
from spider_env import SpiderEnv
from rollout_buffer import RolloutBuffer

DIR_PATH = os.path.dirname(os.path.abspath(__file__))
SRC_PATH = os.path.join(DIR_PATH, "..", "src")
CONFIG_PATH = os.path.join(SRC_PATH, "config.yaml")

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

N_ROLLOUT_STEPS = config.get("n_rollout_steps", 2048)
TOTAL_TIMESTEPS = 500_000
LOG_FILE = "reward_log.txt"

def log_episode(episode, ep_reward, mean_reward):
    with open(LOG_FILE, "a") as f:
        f.write(f"Episode {episode}: reward={ep_reward:.2f}  mean={mean_reward:.2f}\n")

env = SpiderEnv(render_mode="human")
obs, info = env.reset()

input_dims = env.observation_space.shape[0]
n_actions = env.action_space.shape[0]

agent = PPOAgent(n_inputs=input_dims, n_actions=n_actions)

train = True
episode = 0
episode_reward = 0.0
reward_history = []
best_mean_reward = -np.inf
rollout_steps = 0
num_updates = 0

for global_step in range(TOTAL_TIMESTEPS):
    state = obs

    raw_action, log_prob, value = agent.select_action(state)

    env_action = np.clip(raw_action, -1.0, 1.0)
    obs, reward, terminated, truncated, _ = env.step(env_action)
    episode_reward += reward
    done = terminated or truncated
    agent.memory.store(state, action, value, reward, log_prob, done)

    if done:
        episode += 1

        reward_history.append(episode_reward)
        recent = reward_history[-100:]
        mean_reward = np.mean(recent)

        log_episode(episode, episode_reward, mean_reward)

        if mean_reward > best_mean_reward and train and len(reward_history) >= 10:
            best_mean_reward = mean_reward
            agent.save_best_checkpoint()

        obs, info = env.reset()
        episode_reward = 0.0

        log_episode_reward(episode, mean_reward)
        if train:
            agent.ppo_update()

        if episode % 10 == 0 and train:
            print(f"EPISODE: {episode}")
            agent.save_checkpoint()

        obs, info = env.reset()
        episode_reward = 0

env.close()
