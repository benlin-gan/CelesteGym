import numpy as np
import random
import struct
import collections

# Action masks
ACTIONS = {
    "left": 0x01,
    "right": 0x02,
    "up": 0x04,
    "down": 0x08,
    "jump": 0x10,
    "dash": 0x20,
    "grab": 0x40
}

class Greedy_learning:
    def __init__(self, epsilon, duration_sec, sleep_time):
        self.epsilon = epsilon # parameter for epsilon greedy exploration
        self.num_frames = duration_sec / sleep_time
        self.q_values = collections.defaultdict(lambda: dict())
        self.time_count = 0
        self.visited = set()

    def generate_action(self, state):
        rand_num = random.random()
        if rand_num <= self.epsilon or state not in self.q_values:
            rand_action = random.randint(0, 127)
            while not self.is_valid_action(state, rand_action):
                rand_action = random.randint(0, 127)
            return rand_action
        else:
            best_score = float("-inf")
            best_action = 0
            for (action, succ_state) in self.q_values[state]:
                if self.q_values[state][(action, succ_state)] > best_score:
                    best_score = self.q_values[state][(action, succ_state)]
                    best_action = action
    def is_valid_action(self, state, action):
        left = action & ACTIONS["left"]
        right = action & ACTIONS["right"]
        up = action & ACTIONS["up"]
        down = action & ACTIONS["down"]
        dash = action & ACTIONS["dash"]
        # check directions
        if left and right:
            return False
        if up and down:
            return False
        if (not state.can_dash or not state.dashes > 0) and dash:
            return False
        return True

    def compute_score(self, prev_state, action, state):
        self.time_count += 1
        time_penalty = -2

        # currently mannually set the goal
        goalXmin = 248
        goalXmax = 287
        goalYmin = 0
        goalYmax = 3
        if state.dead: # heavy penalty if dead
            self.q_values[prev_state][(action, state)] = -1000
            self.__restart__()
            return
        else:
            prev_visited_count = len(self.visited)
            self.visited.add((state.pos_x, state.pos_y))
            if state.pos_x > goalXmin and state.pos_y < goalYmax: # strong reward if pass
                self.q_values[prev_state][(action, state)] = 1000
                self.__restart__()
                return
            score = 0
            dist = abs(state.pos_x - prev_state.pos_x) + abs(state.pos_y - prev_state.pos_y)
            score += time_penalty * self.time_count + (len(self.visited) - prev_visited_count) * 3
            score += (state.pos_y) * (-5) + (state.pos_x) * 8
            self.q_values[prev_state][(action, state)] = score
    def __restart__(self):
        self.visited = set()
        self.time_count = 0



