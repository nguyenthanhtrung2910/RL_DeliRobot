import random
import dill
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from Cell import Cell
from ControlSystem import ControlSystem
import DeliRobotEnv
# Training parameters
n_training_episodes = 50000  # Total training episodes
learning_rate = 0.7          # Learning rate

# Evaluation parameters
n_eval_episodes = 100        # Total number of test episodes

# Environment parameters
max_steps = 300               # Max steps per episode
gamma = 0.95                 # Discounting rate
eval_seed = []               # The evaluation seed of the environment

# Exploration parameters
max_epsilon = 1.0             # Exploration probability at start
min_epsilon = 0.05            # Minimum exploration probability
decay_rate = 0.0005            # Exponential decay rate for exploration prob


env = DeliRobotEnv.DeliRobotEnv('field.txt')
env = gym.wrappers.FlattenObservation(env)

control_system = ControlSystem(Qtable_size=((env.observation_space.high[0]+1)*(env.observation_space.high[1]+1), env.action_space.n),
                               transform_state2scalar=lambda s: s[0]*(env.observation_space.high[1]+1) + s[1])


control_system.learn(env, n_training_episodes, min_epsilon, max_epsilon, decay_rate, max_steps, learning_rate, gamma)

mean_reward, std_reward = control_system.evaluate(env, max_steps, n_eval_episodes, eval_seed)
print(f"Mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")
file_name = 'controller.pkl'
with open(file_name, 'wb') as file:
    dill.dump(control_system, file)
