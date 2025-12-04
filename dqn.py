import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import numpy as np
from matplotlib import pyplot as plt
from shared_mem import SharedMemoryBridge

import time
# Minimal state
class MinimalState:
    def __init__(self, big_state):
        self.grid = big_state.local_grid[8:24, 8:24]  # 16×16
        self.vel_x = big_state.vel_x / 100.0
        self.vel_y = big_state.vel_y / 100.0
        self.dashes = big_state.dashes / 2.0
        self.state = big_state.state  # KEEP THIS for now (to debug reward)

    # Simple reward 
    def reward(self):
        if self.state == 0: #StNormal
            return 1.0 
        return 0.0

# Minimal DQN
class ClimbingDQN(nn.Module):
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
            nn.Linear(1600 + 4, 8),  # +4 for [vel_x, vel_y, dashes, state]
            nn.ReLU(),
            nn.Linear(8, 2)  # 128 actions
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


def train_climbing():
    # Setup
    bridge = SharedMemoryBridge()
    bridge.open(timeout_sec=10.0)
    
    q_network = ClimbingDQN()
    target_network = ClimbingDQN()
    target_network.load_state_dict(q_network.state_dict())
    
    optimizer = optim.Adam(q_network.parameters(), lr=1e-3)
    replay_buffer = ReplayBuffer(capacity=10000)
    
    # Hyperparameters
    epsilon = 1.0  # Start with full exploration
    epsilon_decay = 0.995
    epsilon_min = 0.1
    gamma = 0.99
    batch_size = 32
    target_update_freq = 100
    
    # Metrics
    episode_rewards = []
    climbing_rates = []
    
    step_count = 0
    episode = 0
    
    print("Starting climbing training...")
    
    while episode < 1000:  # 1000 episodes
        # Reset (manually reset game or just continue)
        state = bridge.read_state()
        episode_reward = 0
        climbing_steps = 0
        total_steps = 0
        
        qsave = None
        for step in range(500):  # Max 500 steps per episode

            # Convert state
            min_state = MinimalState(state)
            
            # Epsilon-greedy action selection
            if random.random() < epsilon:
                action = random.randint(0, 1)
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
                    qsave = q_values
                    action = q_values.argmax(dim=1).item()
            
            # Take action
            bridge.write_action(action)
            next_state = bridge.read_state()

            #skip respawn states
            reward = 0
            done = False
            while next_state.state == 14:
                reward = -1
                done=True
                bridge.write_action(0)
                next_state = bridge.read_state()

            
            # Track metrics
            episode_reward += reward
            if min_state.state == 1:
                climbing_steps += 1
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
                print(f"ending episode early at step {step}")
                break
        #return
        print(qsave)
        # Train
        if episode % 10 == 9 and len(replay_buffer) > batch_size:
            print("starting training")
            for ts in range(250):
                if ts % 10 == 0:
                    print(f"on step {ts}")
                train_step(q_network, target_network, replay_buffer, 
                        optimizer, batch_size, gamma)
            print("finished training")
        
        # Update target network
        step_count += 1
        if step_count % target_update_freq == 0:
            target_network.load_state_dict(q_network.state_dict())
        
        state = next_state
        
        # Episode complete
        episode += 1
        climbing_rate = climbing_steps / total_steps if total_steps > 0 else 0
        episode_rewards.append(episode_reward)
        climbing_rates.append(climbing_rate)
        print(f"episode", episode)

        # Decay epsilon
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        
        # Log progress
        if episode % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            avg_climbing = np.mean(climbing_rates[-10:])
            print(f"Episode {episode}: "
                  f"Avg Reward={avg_reward:.2f}, "
                  f"Climbing Rate={avg_climbing:.2%}, "
                  f"Epsilon={epsilon:.3f}")
        
        # Check convergence
        if episode >= 100:
            recent_climbing = np.mean(climbing_rates[-50:])
            if recent_climbing > 0.9:
                print(f"✓ CONVERGED! Climbing {recent_climbing:.1%} of the time")
                break
    
    bridge.close()
    return episode_rewards, climbing_rates


def train_step(q_network, target_network, replay_buffer, optimizer, batch_size, gamma):
    states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
    
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
    rewards, climbing_rates = train_climbing()
    
    # Plot results
    import matplotlib.pyplot as plt
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    ax1.plot(rewards)
    ax1.set_title('Episode Rewards')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    
    ax2.plot(climbing_rates)
    ax2.axhline(y=0.9, color='r', linestyle='--', label='90% threshold')
    ax2.set_title('Climbing Rate')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('% of Steps Climbing')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('climbing_training.png')
    print("Saved plot to climbing_training.png")