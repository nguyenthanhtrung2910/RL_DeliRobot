import numpy as np
import random
from gymnasium import spaces
class ControlSystem:
    def __init__(self, Qtable_size, transform_state2scalar = lambda x:x) -> None:
        self.Qtable = np.zeros(Qtable_size)
        self.action_space = spaces.Discrete(Qtable_size[1])
        self.transform_state2scalar = transform_state2scalar

    def greedy_policy(self, state):
        # Exploitation: take the action with the highest state, action value
        action = np.argmax(self.Qtable[state][:])
        return action
    
    def epsilon_greedy_policy(self, state, epsilon):
        # Randomly generate a number between 0 and 1
        random_num = random.uniform(0,1)
        # if random_num > greater than epsilon --> exploitation
        if random_num > epsilon:
            # Take the action with the highest value given a state
            # np.argmax can be useful here
            action = self.greedy_policy(state)
        # else --> exploration
        else:
            action = self.action_space.sample()
        return action
    
    def learn(self, env, n_training_episodes, min_epsilon, max_epsilon, decay_rate, max_steps, learning_rate, gamma):
        for episode in range(n_training_episodes):
            # Reduce epsilon (because we need less and less exploration)
            epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*episode)
            # Reset the environment
            state, _ = env.reset()
            state = self.transform_state2scalar(state)
            step = 0
            terminated = False
            truncated = False

            # repeat
            for step in range(max_steps):
                # Choose the action At using epsilon greedy policy
                action = self.epsilon_greedy_policy(state, epsilon)

                # Take action At and observe Rt+1 and St+1
                # Take the action (a) and observe the outcome state(s') and reward (r)
                new_state, reward, terminated, truncated, _ = env.step(action)
                new_state = self.transform_state2scalar(new_state)
                # Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
                self.Qtable[state][action] = self.Qtable[state][action] + learning_rate * (reward + gamma * np.max(self.Qtable[new_state]) - self.Qtable[state][action])

                # If terminated or truncated finish the episode
                if terminated or truncated:
                    break

                # Our next state is the new state
                state = new_state

    def evaluate(self, env, max_steps, n_eval_episodes, seed):
        """
        Evaluate the agent for ``n_eval_episodes`` episodes and returns average reward and std of reward.
        :param env: The evaluation environment
        :param max_steps: Maximum number of steps per episode
        :param n_eval_episodes: Number of episode to evaluate the agent
        :param Q: The Q-table
        :param seed: The evaluation seed array (for taxi-v3)
        """
        episode_rewards = []
        for episode in range(n_eval_episodes):
            if seed:
                state, _ = env.reset(seed=seed[episode])
            else:
                state, _ = env.reset()
            state = self.transform_state2scalar(state)
            step = 0
            truncated = False
            terminated = False
            total_rewards_ep = 0

            for step in range(max_steps):
                # Take the action (index) that have the maximum expected future reward given that state
                action = self.greedy_policy(state)
                new_state, reward, terminated, truncated, _ = env.step(action)
                new_state = self.transform_state2scalar(new_state)
                total_rewards_ep += reward

                if terminated or truncated:
                    break
                state = new_state
            episode_rewards.append(total_rewards_ep)
        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)

        return mean_reward, std_reward
