
import gym
import cv2
from stable_baselines3 import PPO
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn
from carla_env.carla_env import CarlaEnv


# class CustomCNN(BaseFeaturesExtractor):
#     def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512):
#         super(CustomCNN, self).__init__(observation_space, features_dim)
#         # Define your CNN architecture here
#         self.cnn = nn.Sequential(
#             nn.Conv2d(3, 32, kernel_size=8, stride=4, padding=0),
#             nn.ReLU(),
#             nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
#             nn.ReLU(),
#             nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
#             nn.ReLU(),
#             nn.Flatten(),
#             nn.Linear(64 * 7 * 7, features_dim),
#             nn.ReLU()
#         )

#     def forward(self, observations):
#         return self.cnn(observations)
    

# policy_kwargs = dict(
#     features_extractor_class=CustomCNN,
#     features_extractor_kwargs=dict(features_dim=512),
# )

env = make_vec_env(lambda: CarlaEnv(), n_envs=1, seed=0)

model = PPO('CnnPolicy', env, verbose=1, tensorboard_log="./ppo_carla_tensorboard/")

try:
    total_timesteps = 100000
    log_interval = 1000
    for step in range(1, total_timesteps + 1):
        model.learn(total_timesteps=log_interval, reset_num_timesteps=False, tb_log_name="ppo_carla_tensorboard_3")
        env.render(mode='human')
        model.save(f"ppo_carla_model_checkpoint_{step * log_interval}")
finally:
    env.close()



