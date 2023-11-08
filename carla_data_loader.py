import gym
import carla_env
import cv2

from carla_env.data_collector import DataCollector
from carla_env.carla_env import CarlaEnv


env = DataCollector(env=None)

dataset = env._load_dataset(load_dir='./output/dataset_377.pkl')

# print the first 5 data
print(dataset['observations'][:5])
print(dataset['actions'][:5])
print(dataset['rewards'][:5])
print(dataset['dones'][:5])

# show camera images
for obs in dataset['observations']:
    display_image = cv2.resize(obs, None, fx=1, fy=1)

    cv2.imshow("camera", display_image)
    cv2.waitKey(1)
