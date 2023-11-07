import gym
import carla_env
import cv2

from carla_env.data_collector import DataCollector
from carla_env.carla_env import CarlaEnv

env = CarlaEnv()

env = DataCollector(env, steps=200, save_dir='./output')

env.reset()
done = False

try:
    while not done:
        action = env.action_space.sample()

        observation, reward, done, info = env.step(action)

        if observation is not None and observation.size != 0:
            display_image = cv2.resize(observation, None, fx=1, fy=1)

            cv2.imshow("camera", display_image)
            cv2.waitKey(1)

    print(env._load_dataset(load_dir='./output/dataset_111.pkl'))
    
finally:
    env.close()