import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import numpy as np
import os
import json
from collections import defaultdict

# Minimal state
class MinimalState:
    def __init__(self, big_state):
        self.grid = big_state.local_grid[8:24, 8:24]  # 16×16
        self.vel_x = big_state.vel_x / 100.0
        self.vel_y = big_state.vel_y / 100.0
        self.dashes = big_state.dashes / 2.0
        self.state = big_state.state
        self.pos_x = big_state.pos_x
        self.pos_y = big_state.pos_y
        self.on_ground = big_state.on_ground
        self.dead = big_state.dead

# Minimal DQN
class CelesteDQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2),  # 16->7
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1), # 7->5
            nn.ReLU(),
            nn.Flatten()  # 64×5×5 = 1600
        )
        
        self.fc = nn.Sequential(
            nn.Linear(1600 + 4, 128),  # +4 for [vel_x, vel_y, dashes, state]
            nn.ReLU(),
            nn.Linear(128, 128)  # 128 actions
        )
    
    def forward(self, grid, other):
        grid = grid.unsqueeze(1)
        spatial = self.conv(grid)
        combined = torch.cat([spatial, other], dim=1)
        return self.fc(combined)

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)


class DQNLearning:
    def __init__(self, seed, discount=0.99, exploration_prob=1.0):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        # DQN components
        self.q_network = CelesteDQN()
        self.target_network = CelesteDQN()
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=1e-3)
        self.replay_buffer = ReplayBuffer(capacity=10000)
        
        # Hyperparameters
        self.discount = discount
        self.exploration_prob = exploration_prob
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.1
        self.gamma = 0.99
        self.batch_size = 1024
        
        # Episode tracking (matches other algorithms)
        self.episode = 1
        self.room = 1
        self.actions = []
        self.max_score = float("-inf")
        self.running_max = float("-inf")
        self.ground_y = 500
        self.ground_dict = defaultdict(int)
        
        # State buffer for temporal learning
        self.sa_buffer = []
        self.action_hold_count = 0
        self.current_action = 0
        
        # Logging
        self.dirname = "dqn"
        os.makedirs(self.dirname, exist_ok=True)
        self.action_file = open(f"{self.dirname}/{seed}_seed_episode_log.json", 'w')
        
        # Training state
        self.step_count = 0
        
    def get_action(self, big_state):
        """Get action for current state (matches interface)"""
        # Skip during respawn/transition states
        if big_state.state == 13 or big_state.state == 14:
            self.ground_y = big_state.pos_y
            return 0
        
        state = MinimalState(big_state)
        
        # Epsilon-greedy action selection
        if random.random() < self.exploration_prob:
            action = random.randint(0, 127)
            # Enforce action constraints (no left+right, no up+down)
            if (action & 0x01 != 0) and (action & 0x02 != 0):
                action -= 0x01
            if (action & 0x04 != 0) and (action & 0x08 != 0):
                action -= 0x08
            if state.dashes == 0:
                action &= 0x5F  # Disable dash
        else:
            with torch.no_grad():
                grid = torch.tensor(state.grid, dtype=torch.float32).unsqueeze(0)
                other = torch.tensor([
                    state.vel_x,
                    state.vel_y,
                    state.dashes,
                    state.state / 30.0
                ], dtype=torch.float32).unsqueeze(0)
                
                q_values = self.q_network(grid, other)
                action = q_values.argmax(dim=1).item()
        
        self.save_action(action)
        self.sa_buffer.append((big_state, action))
        return action
    
    def get_reward(self, state, action):
        """Compute reward (matches interface)"""
        reward = ((1538 - state.pos_y) + (state.pos_x) * 10) * 10
        
        # Bonus for wall climbing
        if state.state == 1:
            reward += 10
            if (action & 0x04 != 0):  # Pressing up while climbing
                reward += 50
        
        # Bonus for reaching new low ground
        if state.on_ground == 1 and state.pos_y < self.ground_y:
            self.ground_y = state.pos_y
            reward += 1000
            print(f"Ground, {state.pos_y} {self.ground_dict[state.pos_y]}")
            self.ground_dict[state.pos_y] += 1
        
        if reward > self.max_score:
            print(reward)
            self.max_score = reward
        
        return reward
    
    def incorporate_feedback(self, episode_done, room_won):
        """Update Q-network and handle episode transitions (matches interface)"""
        if len(self.sa_buffer) < 2:
            return
        
        big_state, action = self.sa_buffer[-2]
        big_next_state, _ = self.sa_buffer[-1]
        
        state = MinimalState(big_state)
        next_state = MinimalState(big_next_state)
        
        # Compute rewards
        reward = self.get_reward(state, action)
        next_reward = self.get_reward(next_state, 0)
        
        # Track max reward for episode
        if reward > self.running_max:
            self.running_max = reward
        
        # Check for death
        done = next_state.dead
        
        # Store transition in replay buffer
        state_tuple = (state.grid, [state.vel_x, state.vel_y, state.dashes, state.state / 30.0])
        next_state_tuple = (next_state.grid, [next_state.vel_x, next_state.vel_y, 
                                               next_state.dashes, next_state.state / 30.0])
        
        self.replay_buffer.push(state_tuple, action, next_reward, next_state_tuple, done)
        
        # Train every 10 episodes if buffer is large enough
        if episode_done and self.episode % 10 == 0 and len(self.replay_buffer) > self.batch_size:
            self._train_step()
            # Update target network
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Handle episode end
        if episode_done:
            self.save_actions()
            # Decay exploration
            self.exploration_prob = max(self.epsilon_min, self.exploration_prob * self.epsilon_decay)
            # Special: disable exploration after episode 500
            if self.episode > 500:
                self.exploration_prob = 0.0
                self.epsilon_min = 0.0
        
        if room_won:
            self.room += 1
        
        self.step_count += 1
    
    def _train_step(self):
        """Internal: perform one gradient update"""
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # Convert to tensors
        grids = torch.stack([torch.tensor(s[0], dtype=torch.float32) for s in states])
        others = torch.stack([torch.tensor(s[1], dtype=torch.float32) for s in states])
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_grids = torch.stack([torch.tensor(s[0], dtype=torch.float32) for s in next_states])
        next_others = torch.stack([torch.tensor(s[1], dtype=torch.float32) for s in next_states])
        dones = torch.tensor(dones, dtype=torch.float32)
        
        # Current Q values
        q_values = self.q_network(grids, others)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Target Q values
        with torch.no_grad():
            next_q_values = self.target_network(next_grids, next_others)
            max_next_q = next_q_values.max(dim=1)[0]
            targets = rewards + self.gamma * max_next_q * (1 - dones)
        
        # Loss and update
        loss = nn.MSELoss()(q_values, targets)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 10.0)
        self.optimizer.step()
    
    def save_action(self, action):
        """Save action for logging (matches interface)"""
        self.actions.append(int(action))
    
    def save_actions(self):
        """Write episode data to JSON file (matches interface)"""
        data_record = {
            'episode': self.episode,
            'room': self.room,
            'ground_y': self.ground_y,
            'reward': int(self.running_max),
            'actions': self.actions
        }
        
        json_string = json.dumps(data_record, separators=(',', ':'))
        self.action_file.write(json_string + '\n')
        
        self.actions = []
        if self.running_max > self.max_score:
            self.max_score = self.running_max
        self.running_max = 0
        self.episode += 1
    
    def save_weights(self):
        """Save model checkpoint (matches interface)"""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'episode': self.episode
        }, f'{self.dirname}/checkpoint.pt')
    
    def print_action(self, action):
        """Debug: print action as string (matches interface)"""
        action_str = ""
        if (action & 0x01 != 0): action_str += "left "
        if (action & 0x02 != 0): action_str += "right "
        if (action & 0x04 != 0): action_str += "up "
        if (action & 0x08 != 0): action_str += "down "
        if (action & 0x10 != 0): action_str += "jump "
        if (action & 0x20 != 0): action_str += "dash "
        if (action & 0x40 != 0): action_str += "grab "
        print(action_str)