import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import numpy as np
from matplotlib import pyplot as plt
from shared_mem import SharedMemoryBridge
from tqdm import tqdm

import time
# Minimal state
class MinimalState:
    def __init__(self, big_state):
        self.grid = big_state.local_grid[8:24, 8:24]  # 16×16
        self.vel_x = big_state.vel_x / 100.0
        self.vel_y = big_state.vel_y / 100.0
        self.dashes = big_state.dashes / 2.0
        self.state = big_state.state  # KEEP THIS for now (to debug reward)

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

def get_reward(state, next_state):
    """Compute reward based on state transition"""
    reward = 0
    
    # Height reward (lower Y is better in Celeste coordinates)
    height_gain = state.pos_y - next_state.pos_y
    reward += height_gain * 0.01
    
    # Horizontal progress
    horizontal_gain = next_state.pos_x - state.pos_x
    reward += horizontal_gain * 0.1
    
    return reward

def train():
    # Setup
    bridge = SharedMemoryBridge()
    bridge.open(timeout_sec=10.0)
    
    q_network = CelesteDQN()
    target_network = CelesteDQN()
    target_network.load_state_dict(q_network.state_dict())
    
    optimizer = optim.Adam(q_network.parameters(), lr=1e-3)
    replay_buffer = ReplayBuffer(capacity=10000)
    
    # Hyperparameters
    epsilon = 1  # Start with full exploration
    epsilon_decay = 0.995
    epsilon_min = 0.1
    gamma = 0.99
    batch_size = 1024
    
    # Metrics
    episode_rewards = []
    episode_lengths = []
    
    step_count = 0
    episode = 0
    
    while episode < 520:  
        if episode > 500:
            epsilon = 0 #do some epsilon=0 test at the end.
            epsilon_min = 0
        while bridge.shm[bridge.ACTION_READY_OFFSET] == 1:
            pass    
        state = bridge.read_state()
        episode_reward = 0
        total_steps = 0
        
        qsave = None
        episode_length = 500
        for step in range(500):  # Max 500 steps per episode
            
            # Convert state
            min_state = MinimalState(state)
            
            # Epsilon-greedy action selection
            if random.random() < epsilon:
                action = random.randint(0, 127)
            else:
                with torch.no_grad():
                    grid = torch.tensor(min_state.grid, dtype=torch.float32).unsqueeze(0)
                    other = torch.tensor([
                        min_state.vel_x,
                        min_state.vel_y,
                        min_state.dashes,
                        min_state.state / 30.0
                    ], dtype=torch.float32).unsqueeze(0)
                    
                    q_values = q_network(grid, other)
                    if qsave is None:
                        qsave = q_values
                    action = q_values.argmax(dim=1).item()
            #consume action for current state
            bridge.write_action(action)
            for _ in range(14):
                while bridge.shm[bridge.ACTION_READY_OFFSET] == 1:
                    pass   
                next_state = bridge.read_state()
                bridge.write_action(action)

            #skip respawn states
            reward = get_reward(state, next_state)
            done = False
            while next_state.state == 14:
                reward = -10
                done=True
                while bridge.shm[bridge.ACTION_READY_OFFSET] == 1:
                    pass
                next_state = bridge.read_state()
                bridge.write_action(0)

            
            # Track metrics
            episode_reward += reward
            total_steps += 1
            
            # Store transition
            next_min_state = MinimalState(next_state)
            replay_buffer.push(
                (min_state.grid, [min_state.vel_x, min_state.vel_y, 
                                  min_state.dashes, min_state.state / 30.0]),
                action,
                reward,
                (next_min_state.grid, [next_min_state.vel_x, next_min_state.vel_y,
                                       next_min_state.dashes, next_min_state.state / 30.0]),
                done
            )
            #print(f"Step {step}: state_id={state.state}, action={action}, reward={reward}, done={done}")
            state = next_state
            if done:
                episode_length = step
                break
        #return
        #print(qsave)
        # Train
        if episode % 10 == 9 and len(replay_buffer) > batch_size:
            train_step(q_network, target_network, replay_buffer, 
                        optimizer, batch_size, gamma)
            target_network.load_state_dict(q_network.state_dict())
        
        state = next_state
        
        # Episode complete
        episode += 1
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        #print(f"episode", episode)

        # Decay epsilon
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        
        # Log progress
        if episode % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            avg_length = np.mean(episode_lengths[-10:])
            print(f"Episode {episode}: "
                  f"Avg Reward={avg_reward:.2f}, "
                  f"Average Length={avg_length:.2f}, "
                  f"Epsilon={epsilon:.3f}")
        
    
    bridge.close()
    return episode_rewards, episode_lengths


def train_step(q_network, target_network, replay_buffer, optimizer, batch_size, gamma):
    states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
    #print(states[0][0])
    
    # Convert to tensors
    grids = torch.stack([torch.tensor(s[0], dtype=torch.float32) for s in states])
    others = torch.stack([torch.tensor(s[1], dtype=torch.float32) for s in states])
    actions = torch.tensor(actions, dtype=torch.long)
    rewards = torch.tensor(rewards, dtype=torch.float32)
    next_grids = torch.stack([torch.tensor(s[0], dtype=torch.float32) for s in next_states])
    next_others = torch.stack([torch.tensor(s[1], dtype=torch.float32) for s in next_states])
    dones = torch.tensor(dones, dtype=torch.float32)
    
    # Current Q values
    q_values = q_network(grids, others)
    q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
    
    # Target Q values
    with torch.no_grad():
        next_q_values = target_network(next_grids, next_others)
        max_next_q = next_q_values.max(dim=1)[0]
        targets = rewards + gamma * max_next_q * (1 - dones)
        #print(rewards)
    # Loss and update
    loss = nn.MSELoss()(q_values, targets)
    
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(q_network.parameters(), 10.0)
    optimizer.step()


if __name__ == "__main__":
    rewards, episode_lengths = train()
    
    # Plot results
    import matplotlib.pyplot as plt
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    ax1.plot(rewards)
    ax1.set_title('Episode Rewards')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    
    ax2.plot(episode_lengths)
    ax2.set_title('Episode Lengths')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Number of Steps Survived')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('episode_lengths.png')