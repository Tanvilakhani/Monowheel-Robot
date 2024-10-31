import numpy as np
import math
import random
from dataclasses import dataclass
import matplotlib.pyplot as plt
from typing import List, Tuple
import pickle

@dataclass
class PhysicalParams:
    g: float = 9.81  # gravity (m/s^2)
    m_wheel: float = 0.1  # wheel mass (kg)
    m_body: float = 0.3  # body mass (kg)
    r_wheel: float = 0.05  # wheel radius (m)
    h_body: float = 0.15  # body height (m)
    I_wheel: float = None  # wheel moment of inertia (kg*m^2)
    I_body: float = None  # body moment of inertia (kg*m^2)
    motor_torque_constant: float = 0.1  # Motor Kt (Nm/A)
    max_torque: float = 0.5  # Maximum motor torque (Nm)
    dt: float = 0.02  # simulation timestep (s)
    
    def __post_init__(self):
        # Calculate moments of inertia if not provided
        if self.I_wheel is None:
            self.I_wheel = 0.5 * self.m_wheel * self.r_wheel**2
        if self.I_body is None:
            self.I_body = self.m_body * self.h_body**2 / 12

class BalanceSimulator:
    def __init__(self, params: PhysicalParams = None):
        self.params = params if params else PhysicalParams()
        
    def simulate_step(self, state: np.ndarray, torque: float) -> np.ndarray:
        """
        Simulate one timestep of the system dynamics
        state: [angle, angular_velocity, wheel_omega, wheel_position]
        """
        theta, theta_dot, wheel_omega, wheel_pos = state
        
        # Calculate accelerations using physics equations
        # Simplified equations of motion for an inverted pendulum on a wheel
        theta_ddot = (self.params.m_body * self.params.h_body * 
                     self.params.g * math.sin(theta) - torque) / \
                    (self.params.I_body + self.params.m_body * 
                     self.params.h_body**2)
        
        wheel_alpha = torque / self.params.I_wheel
        
        # Euler integration
        theta_new = theta + theta_dot * self.params.dt
        theta_dot_new = theta_dot + theta_ddot * self.params.dt
        wheel_omega_new = wheel_omega + wheel_alpha * self.params.dt
        wheel_pos_new = (wheel_pos + wheel_omega * self.params.dt) % (2 * math.pi)
        
        return np.array([theta_new, theta_dot_new, wheel_omega_new, wheel_pos_new])

class DataGenerator:
    def __init__(self, simulator: BalanceSimulator):
        self.simulator = simulator
        self.training_data = []
        
    def generate_initial_state(self, difficulty: str) -> np.ndarray:
        """Generate initial states based on difficulty level"""
        if difficulty == "easy":
            angle = random.uniform(-0.1, 0.1)  # ±5.7 degrees
            ang_vel = random.uniform(-0.2, 0.2)
        elif difficulty == "medium":
            angle = random.uniform(-0.3, 0.3)  # ±17.2 degrees
            ang_vel = random.uniform(-0.5, 0.5)
        else:  # hard
            angle = random.uniform(-0.6, 0.6)  # ±34.4 degrees
            ang_vel = random.uniform(-1.0, 1.0)
            
        return np.array([
            angle,
            ang_vel,
            random.uniform(-2.0, 2.0),  # wheel_omega
            random.uniform(0, 2 * math.pi)  # wheel_position
        ])
    
    def generate_episode(self, difficulty: str, 
                        max_steps: int = 500) -> List[Tuple]:
        """Generate one episode of training data"""
        state = self.generate_initial_state(difficulty)
        episode_data = []
        
        for _ in range(max_steps):
            # Calculate "optimal" action based on LQR-like control
            angle = state[0]
            ang_vel = state[1]
            wheel_vel = state[2]
            
            # Simple PD control as baseline policy
            k_p = 2.0  # proportional gain
            k_d = 0.5  # derivative gain
            baseline_torque = -(k_p * angle + k_d * ang_vel)
            
            # Add exploration noise
            noise = random.uniform(-0.1, 0.1)
            torque = np.clip(baseline_torque + noise, 
                           -self.simulator.params.max_torque,
                           self.simulator.params.max_torque)
            
            # Convert torque to discrete action (-4 to 4)
            action = int(round(torque * 4 / self.simulator.params.max_torque))
            action = np.clip(action, -4, 4)
            
            # Simulate system dynamics
            next_state = self.simulator.simulate_step(state, torque)
            
            # Calculate reward
            reward = self.calculate_reward(next_state)
            
            # Check if episode is done
            done = abs(next_state[0]) > math.pi/3  # 60 degrees
            
            episode_data.append((state, action + 4, reward, next_state, done))
            
            if done:
                break
                
            state = next_state
            
        return episode_data
    
    def calculate_reward(self, state: np.ndarray) -> float:
        """Calculate reward for a given state"""
        angle = state[0]
        ang_vel = state[1]
        wheel_vel = state[2]
        
        angle_reward = math.cos(angle)
        stability_penalty = -0.1 * (abs(ang_vel) + 0.1 * abs(wheel_vel))
        efficiency_reward = -0.05 * wheel_vel**2
        
        reward = 2.0 * angle_reward + stability_penalty + efficiency_reward
        
        if abs(angle) > math.pi/3:
            reward = -10
            
        return reward
    
    def generate_dataset(self, num_episodes: int = 1000,
                        difficulties: List[str] = ['easy', 'medium', 'hard'],
                        visualize: bool = True) -> List[Tuple]:
        """Generate full training dataset with multiple difficulties"""
        all_data = []
        episode_lengths = []
        total_rewards = []
        
        for episode in range(num_episodes):
            difficulty = random.choice(difficulties)
            episode_data = self.generate_episode(difficulty)
            all_data.extend(episode_data)
            episode_lengths.append(len(episode_data))
            total_reward = sum(data[2] for data in episode_data)
            total_rewards.append(total_reward)
            
            if episode % 100 == 0:
                print(f"Generated episode {episode}/{num_episodes}")
                print(f"Average episode length: {np.mean(episode_lengths):.2f}")
                print(f"Average total reward: {np.mean(total_rewards):.2f}")
        
        if visualize:
            self.visualize_dataset(episode_lengths, total_rewards)
        
        return all_data
    
    def visualize_dataset(self, episode_lengths: List[int], 
                         total_rewards: List[float]):
        """Visualize dataset statistics"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Episode lengths histogram
        ax1.hist(episode_lengths, bins=30)
        ax1.set_title('Episode Lengths Distribution')
        ax1.set_xlabel('Steps')
        ax1.set_ylabel('Count')
        
        # Total rewards histogram
        ax2.hist(total_rewards, bins=30)
        ax2.set_title('Total Rewards Distribution')
        ax2.set_xlabel('Reward')
        ax2.set_ylabel('Count')
        
        plt.tight_layout()
        plt.savefig('simulation_stats.png')
        plt.close()

def save_dataset(data, filename='balance_bot_training_data.pkl'):
    """Save generated dataset to file"""
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
    print(f"Dataset saved to {filename}")

def load_dataset(filename='balance_bot_training_data.pkl'):
    """Load dataset from file"""
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data

if __name__ == "__main__":
    # Create simulator and data generator
    params = PhysicalParams(
        m_wheel=0.1,  # Adjust these parameters to match your robot
        m_body=0.3,
        r_wheel=0.05,
        h_body=0.15,
        max_torque=0.5
    )
    simulator = BalanceSimulator(params)
    generator = DataGenerator(simulator)
    
    # Generate dataset with different difficulties
    dataset = generator.generate_dataset(
        num_episodes=1000,
        difficulties=['easy', 'medium', 'hard'],
        visualize=True
    )
    
    # Save dataset
    save_dataset(dataset)
    
    # Print dataset statistics
    num_samples = len(dataset)
    avg_reward = np.mean([data[2] for data in dataset])
    print(f"Generated {num_samples} training samples")
    print(f"Average reward: {avg_reward:.2f}")