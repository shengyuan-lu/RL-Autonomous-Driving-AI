import gym
import cv2
from carla_env.carla_env import CarlaEnv

env = CarlaEnv()

num_episodes = 5

try:
    for episode in range(num_episodes):
        observation = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = env.action_space.sample()

            observation, reward, done, info = env.step(action)

            if observation is not None and observation.size != 0:
                display_image = cv2.resize(observation, None, fx=1, fy=1)

                cv2.imshow("camera", display_image)
                cv2.waitKey(1)

            total_reward += reward


            if done:
                print(total_reward)
                break
finally:
    env.close()

