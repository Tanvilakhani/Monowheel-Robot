import gymnasium as gym 
from stable_baselines3 import DQN

import self_balacing_necklace
env = gym.make("SelfBalancingNecklace-v0", render_mode='human')

model = DQN(policy = "MlpPolicy", env = env, verbose = 1)
model.learn(total_timesteps = 100_000)