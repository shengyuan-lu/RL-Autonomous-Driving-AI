import gym
import carla_env
import cv2

from carla_env.data_collector import DataCollector
from carla_env.carla_env import CarlaEnv


env = DataCollector(env=None)

dataset = env._load_dataset(load_dir='./output/dataset_111.pkl')

# print the first 5 data
print(dataset['observations'][:5])
print(dataset['actions'][:5])
print(dataset['rewards'][:5])
print(dataset['dones'][:5])