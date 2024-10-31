import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
import random
import board
import busio
import adafruit_bno055
import RPi.GPIO as GPIO
import time

Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class BalanceBot:
    def __init__(self):
        # Hardware setup
        self.i2c = busio.I2C(board.SCL, board.SDA)
        self.imu = adafruit_bno055.BNO055_I2C(self.i2c)
        
        # Reaction wheel motor setup (adjust pins as needed)
        self.motor_pin = 18
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.motor_pin, GPIO.OUT)
        self.pwm = GPIO.PWM(self.motor_pin, 1000)  # 1000Hz frequency
        self.pwm.start(0)
        
        # RL parameters
        self.state_size = 4  # [angle, angular_velocity, wheel_velocity, wheel_position]
        self.action_size = 9  # Different PWM values (-100 to 100 in steps)
        self.memory = deque(maxlen=10000)
        self.batch_size = 64
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        
        # Neural Networks
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQN(self.state_size, self.action_size).to(self.device)
        self.target_net = DQN(self.state_size, self.action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters())
        
    def get_state(self):
        # Read IMU data
        angle = self.imu.euler[1]  # pitch angle
        angular_velocity = self.imu.gyro[1]  # pitch angular velocity
        wheel_velocity = self.imu.gyro[2]  # wheel velocity
        wheel_position = 0  # You'll need to implement wheel position tracking
        
        return np.array([angle, angular_velocity, wheel_velocity, wheel_position])
    
    def take_action(self, action):
        # Convert action index to PWM value (-100 to 100)
        pwm_value = (action - 4) * 25  # Maps actions 0-8 to -100 to 100
        
        # Apply PWM to motor
        if pwm_value >= 0:
            self.pwm.ChangeDutyCycle(pwm_value)
        else:
            # Implement reverse direction control here
            self.pwm.ChangeDutyCycle(abs(pwm_value))
    
    def calculate_reward(self, state):
        angle = state[0]
        angular_velocity = state[1]
        
        # Reward function parameters
        angle_threshold = 30  # degrees
        
        # Check if robot has fallen
        if abs(angle) > angle_threshold:
            return -100  # Large negative reward for falling
        
        # Reward for staying upright (closer to vertical = better)
        angle_reward = -abs(angle) / angle_threshold
        stability_reward = -abs(angular_velocity) * 0.1
        
        return angle_reward + stability_reward
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append(Experience(state, action, reward, next_state, done))
    
    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax().item()
    
    def learn(self):
        if len(self.memory) < self.batch_size:
            return
        
        # Sample batch of experiences
        batch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor([e.state for e in batch]).to(self.device)
        actions = torch.LongTensor([e.action for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e.reward for e in batch]).to(self.device)
        next_states = torch.FloatTensor([e.next_state for e in batch]).to(self.device)
        dones = torch.FloatTensor([e.done for e in batch]).to(self.device)
        
        # Compute Q values
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_net(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))
        
        # Compute loss and update
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def train(self, episodes=1000, max_steps=500):
        for episode in range(episodes):
            state = self.get_state()
            total_reward = 0
            
            for step in range(max_steps):
                action = self.choose_action(state)
                self.take_action(action)
                
                # Wait for action to take effect
                time.sleep(0.02)
                
                next_state = self.get_state()
                reward = self.calculate_reward(next_state)
                done = abs(next_state[0]) > 30  # Check if fallen
                
                self.remember(state, action, reward, next_state, done)
                self.learn()
                
                total_reward += reward
                state = next_state
                
                if done:
                    break
            
            # Update target network periodically
            if episode % 10 == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
            
            print(f"Episode: {episode}, Total Reward: {total_reward}, Epsilon: {self.epsilon}")
            
            # Save model periodically
            if episode % 100 == 0:
                torch.save(self.policy_net.state_dict(), f'balance_bot_model_{episode}.pth')
    
    def cleanup(self):
        self.pwm.stop()
        GPIO.cleanup()

if __name__ == "__main__":
    bot = BalanceBot()
    try:
        bot.train()
    finally:
        bot.cleanup()