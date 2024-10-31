import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
import random
import math
import time

Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class InvertedPendulumSim:
    def __init__(self):
        # Physical parameters
        self.g = 9.81  # gravity (m/s^2)
        self.m_wheel = 0.1  # wheel mass (kg)
        self.m_body = 0.3   # body mass (kg)
        self.r_wheel = 0.05 # wheel radius (m)
        self.h_body = 0.15  # body height (m)
        self.I_wheel = 0.5 * self.m_wheel * self.r_wheel**2  # wheel inertia
        self.I_body = self.m_body * self.h_body**2 / 12      # body inertia
        self.dt = 0.02  # simulation timestep (s)
        
        # Motor parameters
        self.motor_torque_constant = 0.1  # Nm/A
        self.max_torque = 0.5  # Nm
        
    def step(self, state, action):
        theta, theta_dot, wheel_omega, _ = state
        
        # Convert action (-4 to 4) to torque
        torque = (action / 4.0) * self.max_torque
        
        # Physics simulation (simplified equations of motion)
        theta_ddot = (self.m_body * self.h_body * self.g * math.sin(theta) - torque) / \
                    (self.I_body + self.m_body * self.h_body**2)
        
        wheel_alpha = torque / self.I_wheel
        
        # Euler integration
        theta_new = theta + theta_dot * self.dt
        theta_dot_new = theta_dot + theta_ddot * self.dt
        wheel_omega_new = wheel_omega + wheel_alpha * self.dt
        wheel_pos_new = (state[3] + wheel_omega * self.dt) % (2 * math.pi)
        
        return np.array([theta_new, theta_dot_new, wheel_omega_new, wheel_pos_new])

class BalanceNetV2(nn.Module):
    def __init__(self, state_size, action_size):
        super(BalanceNetV2, self).__init__()
        # Shared features
        self.features = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        )
        
    def forward(self, x):
        features = self.features(x)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        return values + (advantages - advantages.mean(dim=1, keepdim=True))

class BalanceBotV2:
    def __init__(self, load_pretrained=True):
        # RL parameters
        self.state_size = 4
        self.action_size = 9  # -4 to +4 in steps of 1
        self.memory = deque(maxlen=100000)
        self.batch_size = 128
        self.gamma = 0.99
        self.tau = 0.001  # Soft update parameter
        
        # Exploration parameters
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        
        # Networks
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = BalanceNetV2(self.state_size, self.action_size).to(self.device)
        self.target_net = BalanceNetV2(self.state_size, self.action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Simulation
        self.sim = InvertedPendulumSim()
        
        if load_pretrained:
            self.pretrain()
        
        # Optimizer with smaller learning rate for transfer learning
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.0001)
        
    def pretrain(self):
        """Pre-train on simulated data"""
        print("Pre-training on simulated data...")
        optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001)
        
        for episode in range(1000):  # Pre-training episodes
            state = np.array([random.uniform(-0.1, 0.1), 0, 0, 0])  # Start near balanced
            total_reward = 0
            
            for step in range(500):  # Steps per episode
                action = self.choose_action(state, epsilon=0.3)  # More exploration in pre-training
                next_state = self.sim.step(state, action - 4)  # Convert action to -4 to 4 range
                reward = self.calculate_reward(next_state, simulated=True)
                done = abs(next_state[0]) > math.pi/3  # Fall if > 60 degrees
                
                self.remember(state, action, reward, next_state, done)
                self.learn(optimizer)
                
                total_reward += reward
                state = next_state
                
                if done:
                    break
            
            if episode % 100 == 0:
                print(f"Pre-training Episode: {episode}, Total Reward: {total_reward}")
        
        # Save pre-trained model
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
        }, 'pretrained_balance_model.pth')
    
    def calculate_reward(self, state, simulated=False):
        angle = state[0]
        angular_vel = state[1]
        wheel_vel = state[2]
        
        # Base reward for staying upright
        angle_reward = math.cos(angle)  # 1 when vertical, decreases as it tilts
        
        # Penalty for excessive motion
        stability_penalty = -0.1 * (abs(angular_vel) + 0.1 * abs(wheel_vel))
        
        # Energy efficiency reward (penalize high wheel velocities)
        efficiency_reward = -0.05 * wheel_vel**2
        
        # Different reward scaling for sim vs real
        if simulated:
            reward = 2.0 * angle_reward + stability_penalty + efficiency_reward
        else:
            reward = angle_reward + 0.5 * stability_penalty + 0.2 * efficiency_reward
        
        # Immediate failure for falling
        if abs(angle) > math.pi/3:  # 60 degrees
            reward = -10
        
        return reward
    
    def choose_action(self, state, epsilon=None):
        if epsilon is None:
            epsilon = self.epsilon
            
        if random.random() < epsilon:
            return random.randrange(self.action_size)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax().item()
    
    def learn(self, optimizer=None):
        if len(self.memory) < self.batch_size:
            return
            
        if optimizer is None:
            optimizer = self.optimizer
            
        batch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor([e.state for e in batch]).to(self.device)
        actions = torch.LongTensor([e.action for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e.reward for e in batch]).to(self.device)
        next_states = torch.FloatTensor([e.next_state for e in batch]).to(self.device)
        dones = torch.FloatTensor([e.done for e in batch]).to(self.device)
        
        # Double DQN
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
        next_actions = self.policy_net(next_states).argmax(1).unsqueeze(1)
        next_q_values = self.target_net(next_states).gather(1, next_actions).squeeze(1)
        target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))
        
        # Huber loss for robustness
        loss = nn.SmoothL1Loss()(current_q_values.squeeze(), target_q_values)
        
        optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        optimizer.step()
        
        # Soft update of target network
        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(self.tau * policy_param.data + (1 - self.tau) * target_param.data)
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append(Experience(state, action, reward, next_state, done))
    
    def load_pretrained(self, path='pretrained_balance_model.pth'):
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        print("Loaded pre-trained model")
    
    def fine_tune(self, episodes=500, max_steps=1000):
        print("Starting fine-tuning on real hardware...")
        for episode in range(episodes):
            state = self.get_state()  # Implement this for your hardware
            total_reward = 0
            
            for step in range(max_steps):
                action = self.choose_action(state)
                self.apply_action(action)  # Implement this for your hardware
                
                time.sleep(0.02)  # Control loop delay
                
                next_state = self.get_state()
                reward = self.calculate_reward(next_state, simulated=False)
                done = abs(next_state[0]) > math.pi/3
                
                self.remember(state, action, reward, next_state, done)
                self.learn()
                
                total_reward += reward
                state = next_state
                
                if done:
                    break
            
            print(f"Fine-tuning Episode: {episode}, Total Reward: {total_reward}")
            
            # Save periodically
            if episode % 50 == 0:
                torch.save({
                    'policy_net': self.policy_net.state_dict(),
                    'target_net': self.target_net.state_dict(),
                }, f'finetuned_balance_model_{episode}.pth')

if __name__ == "__main__":
    bot = BalanceBotV2(load_pretrained=True)
    bot.fine_tune()