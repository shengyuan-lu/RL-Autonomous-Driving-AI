import gym
import carla_env
import cv2

from carla_env.data_collector import DataCollector
from carla_env.carla_env import CarlaEnv

env = CarlaEnv()

env = DataCollector(env, steps=1000, save_dir='./output')

env.reset()
done = False

try:
    while not done:
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        env.render(mode='human')
    
finally:
    env.close()