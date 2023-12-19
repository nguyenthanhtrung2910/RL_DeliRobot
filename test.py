import random
import dill
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from Cell import Cell
from ControlSystem import ControlSystem
import DeliRobotEnv
with open("controller.pkl", "rb") as file:
    control_system = dill.load(file)

max_steps = 300   
int2action = {0:'up',
              1:'right',
              2:'down',
              3:'left',
              4:'pickup',
              5:'dropoff'}
env = DeliRobotEnv.DeliRobotEnv('field.txt')
env = gym.wrappers.FlattenObservation(env)

state, info = env.reset()
state = control_system.transform_state2scalar(state)
print('Init agent in position:{}'.format(info['agent_location']), file=open('working_process_example.txt', 'w'))
step = 0
truncated = False
terminated = False
for step in range(max_steps):
      # Take the action (index) that have the maximum expected future reward given that state
      action = control_system.greedy_policy(state)
      new_state, reward, terminated, truncated, info = env.step(action)
      new_state = control_system.transform_state2scalar(new_state)
      if action == 4:
        print('agent in location {} do action {} {}'.format(info['agent_location'], int2action[action], info['mail']), file=open('working_process_example.txt', 'a'))
      elif action == 5:
        print('agent in location {} do action {} {}'.format(info['agent_location'], int2action[action], info['deliveried_mail']), file=open('working_process_example.txt', 'a'))
      else:
        print('agent in location {} do action {}'.format(info['agent_location'], int2action[action]), file=open('working_process_example.txt', 'a')) 
      if terminated or truncated:
        break
      state = new_state