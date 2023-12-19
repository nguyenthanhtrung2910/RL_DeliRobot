import numpy as np
import Cell
import gymnasium as gym
from gymnasium import spaces

class DeliRobotEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, file):
        self._load_from_file(file)
        self.size = len(self._cells)

        self._green_cells = self._get_cells_by_color('gr')
        self._yellow_cells = self._get_cells_by_color('y')

        #A list of cells for render
        width = int(self.size**(1/2))
        self._repr_cells = [[self._get_cells_by_coordinate(chr(i)+str(j))[0] 
                       for i in range(97,97+width)] for j in range(width, 0, -1)] 
        
        # Observations are dictionaries with the agent's location and agent's mail.
        self.observation_space = spaces.Dict(
            {
                "agent_location": spaces.Box(0, self.size - 1, shape=(1,), dtype=int),
                "mail": spaces.Box(0, len(self._yellow_cells), shape=(1,), dtype=int)
            }
        )

        # We have 6 actions, corresponding to "right", "up", "left", "down","pickup","dropoff"
        self.action_space = spaces.Discrete(6)

    def _get_cells_by_coordinate(self, *coordinates):
        res = []
        for c in coordinates:
            if c == 'None':
                res.append(None)
            else:
                res = res + list(filter(lambda cell: cell.coordinate == c, self._cells))
        return res
    
    def _get_cells_by_color(self, color):
        return list(filter(lambda cell: cell.color == color, self._cells))
    
    def _get_cells_by_target(self, target):
        return list(filter(lambda cell: cell.target == target, self._cells))
    
    def _load_from_file(self, file):
        self._cells = []
        with open(file, 'r') as f:
            text = f.read().split('\n')
        for cell_datas in text:
            self._cells.append(Cell.Cell(*cell_datas.split()[:3]))
        for cell, cell_datas in list(zip(self._cells, text)):
            cell.set_adj(*self._get_cells_by_coordinate(*cell_datas.split()[4:]))

    def _get_obs(self):
        return {"agent_location": self._cells.index(self._agent_location),
                "mail":self._mail
                }
    
    def _get_info(self):
        return {'agent_location':self._agent_location,
                'mail':self._mail,
                'deliveried_mail':self._deliveried_mail,
                'count':self._count} 
      
    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        
        # Choose the agent's location uniformly at random
        self._agent_location = self._green_cells[0]
        while (self._agent_location in self._yellow_cells) or (self._agent_location in self._green_cells):
            self._agent_location = self.np_random.choice(self._cells)
        self._agent_location.located = 'agent'
        
        self._mail = 0
        self._deliveried_mail = 0
        self._count = 0

        observation = self._get_obs()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action):
        assert action in range(6)
        if (self._agent_location.target == str(self._mail)) and (self._mail != 0) and (action == 5):
            reward = 50
        #punishment for illegal pickup   
        elif ((self._agent_location not in self._green_cells) or (self._mail != 0)) and (action == 4):
            reward = -10
        #punishment for illegal dropoff   
        elif ((self._agent_location.target != str(self._mail)) or (self._mail == 0)) and (action == 5):
            reward = -10
        else:
            reward = -1 
        
        if action == 0:
            if self._agent_location.front: 
                if self._agent_location.front.color != 'y' or self._agent_location.front.target == str(self._mail):
                    self._agent_location = self._agent_location.front
        if action == 1:
            if self._agent_location.right :
                if self._agent_location.right.color != 'y' or self._agent_location.right.target == str(self._mail):
                    self._agent_location = self._agent_location.right
        if action == 2:
            if self._agent_location.back:
                if self._agent_location.back.color != 'y' or self._agent_location.back.target == str(self._mail):
                    self._agent_location = self._agent_location.back
        if action == 3:
            if self._agent_location.left:
                if self._agent_location.left.color != 'y' or self._agent_location.left.target == str(self._mail):
                    self._agent_location = self._agent_location.left
        if action == 4:
            if (self._agent_location in self._green_cells) and (self._mail == 0):
                self._mail = self.np_random.choice(range(1, len(self._yellow_cells)+1))
        if action == 5:
            if (self._agent_location.target == str(self._mail)) and (self._mail != 0):
                self._deliveried_mail = self._mail
                self._mail = 0
                self._count += 1
        self._agent_location.located = 'agent'

        # An episode is done if the agent has reached required mails
        terminated = self._count == 3

        observation = self._get_obs()
        info = self._get_info()
        
        return observation, reward, terminated, False, info
    
    def render(self):
        for ele in self._repr_cells:
            for cell in ele:
                print(cell, end = '   ')
            print('\n')