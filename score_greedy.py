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

# import torch
# import torch.nn as nn

class FourierFeatures():#nn.Module):
    def __init__(self, num_freqs=6):
        # super().__init__()
        self.freq_bands = 2**np.arange(num_freqs)
        # self.register_buffer("freq_bands", freq_bands)

    def forward(self, x, y):
        x_proj = x * self.freq_bands
        y_proj = y * self.freq_bands

        return np.concatenate([
            np.sin(x_proj), np.cos(x_proj),
            np.sin(y_proj), np.cos(y_proj)
        ])
class ReducedGameState:
    def __init__(self, big_state):
        """Parse game state from raw bytes."""
        # Player state (32 bytes)
        self.pos_x = big_state.pos_x
        x_norm = self.pos_x / 2559
        
        self.pos_y = big_state.pos_y
        y_norm = self.pos_y / 1538

        ff = FourierFeatures()
        self.pos_freqs = ff.forward(x_norm, y_norm)

        self.dashes = big_state.dashes
        self.on_ground = big_state.on_ground
        self.wall_slide_dir = big_state.wall_slide_dir
        self.state = big_state.state
        self.dead = big_state.dead
        self.can_dash = big_state.can_dash
        self.local_grid = big_state.local_grid

        # print("pos_freqs", self.pos_freqs.shape)
        # print("dashes", type(self.dashes))
        # print("on_ground", type(self.on_ground))
        # print("wall_slide_dir", type(self.wall_slide_dir))
        # print("state", type(self.state))
        # print("dead", type(self.dead))
        # print("can_dash", type(self.can_dash))
        # print("local_grid", self.local_grid.shape)

        self.features = np.concatenate([
            self.pos_freqs.ravel(),
            [self.dashes,
            self.on_ground,
            self.wall_slide_dir,
            self.state,
            self.dead,
            self.can_dash],
            self.local_grid.ravel()
        ])
        # print("***FEATURE LEN***", len(self.features)) 
        # 1054

class FunctionApproxQLearning():
    def __init__(self, discount=0.95, exploration_prob=0.2):
        # self.actions = actions
        self.discount = discount
        self.exploration_prob = exploration_prob
        self.weights = np.random.standard_normal(size=(1054, 128))
        self.max_score = float("-inf") # for debugging

    def get_q(self, state, action):
        Q_vals = state.features @ self.weights
        return Q_vals[action]

    def get_action(self,  big_state):
        state = ReducedGameState(big_state)
        if (random.random() < self.exploration_prob):
            return random.randint(0, 127)
        else:
            Q_vals = state.features @ self.weights
            return np.argmax(Q_vals)
    
    def get_reward(self, state):
        reward = (1538-state.pos_y) + (state.pos_x)
        if state.state == 1:# if next to wall and climbing
            reward += 50
        if state.on_ground == 1:
            reward += 5
        if reward > self.max_score:
            print(reward)
            self.max_score = reward

        return reward

    def incorporate_feedback(self, big_state, action, big_next_state):
        state = ReducedGameState(big_state)
        next_state = ReducedGameState(big_next_state)

        Q_vals = state.features @ self.weights
        Q_vals_next = next_state.features @ self.weights

        reward = self.get_reward(state)
        # reward_next = get_reward(next_state)

        max_future_Q = np.max(Q_vals_next)
        target = reward + self.discount * (max_future_Q - reward)
        self.weights[:, action] += 0.01 * (target-Q_vals[action])*state.features


class Greedy_learning:
    def __init__(self, epsilon, duration_sec, sleep_time):
        self.epsilon = epsilon # parameter for epsilon greedy exploration
        self.num_frames = duration_sec / sleep_time
        self.q_values = collections.defaultdict(lambda: dict())
        self.time_count = 0
        self.visited = set()
        self.max_score = float("-inf") # for debugging

    def generate_action(self, big_state):
        state = ReducedGameState(big_state)
        # print(state.local_grid)
        rand_num = random.random()
        # print(rand_num, self.epsilon)
        if rand_num <= self.epsilon: #or state not in self.q_values:
            rand_action = random.randint(0, 127)
            while not self.is_valid_action(state, rand_action):
                rand_action = random.randint(0, 127)
            return rand_action
        else:
            best_score = float("-inf")
            best_action = 0
            for action in self.q_values[state]:
                if self.q_values[state][action] > best_score:
                    best_score = self.q_values[state][action]
                    best_action = action
            # print("best_action", best_action)
            return best_action
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
        goalXmax = 2300
        goalYmin = 0
        goalYmax = 3
        if state.dead: # heavy penalty if dead
            self.q_values[prev_state][action] = -1000
            self.__restart__()
            return
        else:
            prev_visited_count = len(self.visited)
            self.visited.add((state.pos_x, state.pos_y))
            if state.pos_x > goalXmax and state.pos_y < goalYmin: # strong reward if pass
                self.q_values[prev_state][action] = 1000
                self.__restart__()
                return
            score = 0
            # dist = abs(state.pos_x - prev_state.pos_x) + abs(state.pos_y - prev_state.pos_y)
            # score += state.frame_count * time_penalty #+ (len(self.visited) - prev_visited_count) * 3
            score += (1538-state.pos_y) + (state.pos_x)
            # if state.wall_slide_dir != 0:
            #     score += 5
            # print(state.pos_x, state.pos_y)
            if (prev_state.state == 1 and action|0x40 != 0) or state.state == 1:# if next to wall and climbing
                score += 50
            if state.on_ground:
                score += 5
            self.q_values[prev_state][action] = score
            if score > self.max_score:
                print(score)
                self.max_score = score
        # print(state.pos_x, state.pos_y)
    def __restart__(self):
        self.visited = set()
        self.time_count = 0



