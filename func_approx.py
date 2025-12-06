import numpy as np
import random
import struct
import collections

np.random.seed(1)
random.seed(1)

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

# import torch45040.0
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

        self.vel_x = big_state.vel_x
        self.vel_y = big_state.vel_y

        norm_vel_x = self.vel_x / 300
        norm_vel_y = self.vel_y / 300

        ff = FourierFeatures()
        self.pos_freqs = ff.forward(x_norm, y_norm)
        self.vel_freqs = ff.forward(norm_vel_x, norm_vel_y)

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
            # self.pos_freqs.ravel(),
            self.vel_freqs.ravel(),
            [
                self.dashes,
                self.on_ground,
                self.wall_slide_dir,
                self.state,
                self.dead,
                self.can_dash
            ],
            self.local_grid.ravel()
        ])
        # print("***FEATURE LEN***", len(self.features)) 
        # 1054
        #1078

import json
from collections import defaultdict
class FunctionApproxQLearning():
    def __init__(self, discount=0.90, exploration_prob=0.2):
        # self.actions = actions
        self.discount = discount
        self.exploration_prob = exploration_prob
        self.weights = np.random.standard_normal(size=(1054, 128))
        self.max_score = float("-inf") # for debugging
        self.actions = [] # for saving
        self.running_max = float("-inf")
        self.action_file = open("best_actions.json", 'a')
        self.ground_x = 0
        self.ground_y = 500
        self.ignore_save = 0
        self.ground_dict = defaultdict(int)
        self.sa_buffer = []
        self.action_repeat = 4
        self.counter = 0

    def get_q(self, state, action):
        Q_vals = state.features @ self.weights
        return Q_vals[action]

    def get_action(self,  big_state):
        if (big_state.state == 13 or big_state.state == 14):
            self.ground_y = big_state.pos_y
            return 0
        
        # self.print_action(0x45)
        # return 0x45
        state = ReducedGameState(big_state)
        if (random.random() < self.exploration_prob):
            action = random.randint(0, 127)
            # self.print_action(action)
            # if (action&0x01 !=0):
            #     action -= 0x01
            # action |= 0x02
            if (action&0x01 != 0) and (action&0x02 !=0):
                # print(action, end="->")
                action -= 0x01
                # print(action)
            if (action&0x04 != 0) and (action&0x08 !=0):
                # print(action, end="->")
                action -= 0x08
                # print(action)
            if state.dashes == 0:
                action &= 0x5F
            # self.print_action(action)
            self.save_action(action)
            self.sa_buffer.append((big_state, action))
            return action
        else:
            Q_vals = state.features @ self.weights
            action = np.argmax(Q_vals)
            # self.print_action(np.argmax(Q_vals))
            self.save_action(action)
            if (action&0x01 !=0):
                action -= 0x01
            action |= 0x02
            self.sa_buffer.append((big_state, action))
            return action
    
    def get_reward(self, state, action):
        reward = ((1538-state.pos_y) + (state.pos_x)*10) *10
        if state.state == 1:# if next to wall and climbing
            reward += 10
            if (action&0x04 != 0):
                reward += 50
        if state.on_ground == 1 and state.pos_y < self.ground_y:
            # self.ground_x = state.pos_x
            self.ground_y = state.pos_y
            reward += 1000
            # input("Breakpoint Ground")
            print(f"Ground, {state.pos_y} {self.ground_dict[state.pos_y]}")
            self.ground_dict[state.pos_y] += 1
        if reward > self.max_score:
            print(reward)
            self.max_score = reward

        if state.pos_y < 10 and self.ignore_save == 0:
            self.save_weights()
            self.ignore_save = 1
            input("Breakpoint Save Weights")

        # Test climbing
        # reward = (1538-state.pos_y)
        # reward = 0
        # if state.state == 1:# if next to wall and climbing
        #     reward += 1000
        #     print(reward)
        #     # input("Breakpoint")

        # Test left
        # reward = - state.pos_x
        return reward

    def save_action(self, action): # for debugging, print back
        self.actions.append(int(action))

    def save_actions(self):
        # if self.running_max > .93*self.max_score:
        data_record = {
            'reward': int(self.running_max),
            'actions': self.actions
        }

        json_string = json.dumps(data_record, separators=(',', ':'))

        # with open("best_actions.json", 'a') as f:
        self.action_file.write(json_string + '\n')

        self.actions = []
        if self.running_max > self.max_score:
            self.max_score = self.running_max
        self.running_max = 0
    
    def save_weights(self):
        # json_string = json.dumps(self.weights)
        # print(self.weights)
        # with open("weights.json", 'w') as f:
        #     json.dump(self.weights, f, indent=4)
        np.save('weights.npy', self.weights)

    def print_action(self, action):
        action_str = ""
        if (action&0x01 != 0): action_str += "left "
        if (action&0x02 != 0): action_str += "right "
        if (action&0x04 != 0): action_str += "up "
        if (action&0x08 != 0): action_str += "down "
        if (action&0x10 != 0): action_str += "jump "
        if (action&0x20 != 0): action_str += "dash "
        if (action&0x40 != 0): action_str += "grab "
        print(action_str)


    def incorporate_feedback(self):
        # self.counter += 1
        # if self.counter % 100 == 0:
        #     print(self.weights)
        #     self.counter = 0
        if len(self.sa_buffer) < 2:
            return
        big_state, action = self.sa_buffer[-2]
        big_next_state, _ = self.sa_buffer[-1]
        state = ReducedGameState(big_state)
        # print(state.state)
        next_state = ReducedGameState(big_next_state)

        Q_vals = state.features @ self.weights
        Q_vals_next = next_state.features @ self.weights

        reward = self.get_reward(state, action)
        next_reward = self.get_reward(next_state, 0)
        # reward = self.get_reward(next_state) - self.get_reward(state)
        # print(state.pos_x, next_state.pos_x, next_reward-reward)
        if reward > self.running_max:
            self.running_max = reward
        # reward_next = get_reward(next_state)
        
        if (next_state.state == 13 or next_state.state == 14) and self.actions != []:
            # input("Breakpoint")
            self.save_actions()

        max_future_Q = np.max(Q_vals_next)
        target = (next_reward) + self.discount * max_future_Q
        self.weights[:, action] += 0.001 * (target-Q_vals[action])*state.features


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



