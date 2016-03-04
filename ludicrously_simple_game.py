import numpy as np

"""
Five sequentially connected states, with reward in the last state. Actions are 'left and 'right'.
"""


class LudicrouslySimpleGame(object):
    def __init__(self):
        self.name = "ludicrous"
        self.actions = ['left', 'right']
        self.observation_size = 1
        self.state_reward = np.zeros(5)
        self.state_reward[4] = 1.0
        self.last_state = len(self.state_reward) - 1
        self.state = 0

    def score(self):
        return self.state_reward[self.state]

    def terminal(self):
        return self.state == self.last_state

    def new_game(self):
        self.state = 0
        return [self.state], self.score(), self.terminal()

    def do(self, action_idx):

        if action_idx == 0:  # left
            self.state -= 1
            if self.state < 0:
                self.state = 0
        else:                # right
            self.state += 1
            if self.state > self.last_state:
                self.state = self.last_state

        return [self.state], self.score(), self.terminal()

