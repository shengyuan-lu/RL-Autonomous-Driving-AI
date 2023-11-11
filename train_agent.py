import os

import carla
import torch
from torch import nn

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv

from carla_env.carla_env_multi_obs import CarlaEnv
from utils.clean_actors import clean_actors

# connect to simulator
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)

# initialize world
world = client.get_world()
client.load_world('Town05')



# both fully connected and convolutional neural network 
# to handle telemetry data and camera images
class FC_CNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=512):
        super(FC_CNN, self).__init__(observation_space, features_dim)

        # extract the image shape
        self.image_shape = observation_space['camera'].shape

        # extract the telemetry shape
        self.tele_shape = observation_space['telemetry'].shape[0]

        # CNN layers to extract features from the camera image
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(self.image_shape[0], 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Flatten()
        )

        # Calculate the size of the flattened CNN features
        with torch.no_grad():
            self._cnn_output_dim = self._get_conv_output_dim(torch.zeros(1, *self.image_shape))

        # fully connected layer to combines CNN features with telemetry features
        self.fc_layers = nn.Sequential(
            nn.Linear(self._cnn_output_dim + self.tele_shape, features_dim),
            nn.ReLU(),
            nn.Linear(features_dim, features_dim),
            nn.ReLU()
        )

    def _get_conv_output_dim(self, shape):
        # calculate the output dimension of the CNN layers
        return self.cnn_layers(shape).data.view(1, -1).size(1)

    # forward pass of the network
    def forward(self, observation):
        # extract the image and telemetry data from the observation
        image, tele = observation['camera'], observation['telemetry']

        # pass the image through the CNN layers
        cnn_features = self.cnn_layers(image)

        # flatten the CNN features
        tele_features = tele.squeeze(1)

        # print(cnn_features.shape)
        # print(tele_features.shape)

        # concatenate CNN features and telemetry features
        combined_features = torch.cat((cnn_features, tele_features), dim=1)

        # pass the combined features through the fully connected layers
        return self.fc_layers(combined_features)


policy_kwargs = dict(
    features_extractor_class=FC_CNN,
    features_extractor_kwargs=dict(features_dim=512),
)


# train the existing model
def train_exist_model(model_path, total_timesteps=100000):

    env = lambda: CarlaEnv()
    env = DummyVecEnv([env])

    log_path = os.path.join('./Training/Logs')

    # extract the base model name from the model_path
    model_name = os.path.basename(os.path.dirname(model_path))

    ppo_path = os.path.join('./Training/Saved_Models', model_name)

    # load the model
    model = PPO.load(model_path, env=env, verbose=1, tensorboard_log=log_path)

    # evaluate the model
    eval_env = model.get_env()

    # save the best model
    eval_callback = EvalCallback(eval_env=eval_env, best_model_save_path=ppo_path,
                                 n_eval_episodes=8,
                                 eval_freq=5000, verbose=1,
                                 deterministic=True, render=False)

    model.learn(total_timesteps=total_timesteps, tb_log_name=model_name, callback=eval_callback,
                reset_num_timesteps=False)
    model.save(ppo_path)

# train the new model
def train_new_model(model_name, total_timesteps=100000):
    env = lambda: CarlaEnv()
    env = DummyVecEnv([env])

    log_path = os.path.join('./Training/Logs')
    ppo_path = os.path.join(f'./Training/Saved_Models/{model_name}_best')

    model = PPO('MultiInputPolicy', env, verbose=1, tensorboard_log=log_path)
    eval_env = model.get_env()
    eval_callback = EvalCallback(eval_env=eval_env, best_model_save_path=ppo_path,
                                 n_eval_episodes=8,
                                 eval_freq=5000, verbose=1,
                                 deterministic=True, render=False)
    model.learn(total_timesteps=total_timesteps, tb_log_name=model_name, callback=eval_callback,
                reset_num_timesteps=False)
    ppo_path = os.path.join(f'./Training/Saved_Models/{model_name}_final')
    model.save(ppo_path)


def eval_model(model_path):
    # evaluate the model
    env = lambda: CarlaEnv()
    env = DummyVecEnv([env])

    model = PPO.load(model_path, env=env, verbose=1)

    obs = env.reset()
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = env.step(action)
        env.render(mode='human')


if __name__ == '__main__':

    clean_actors()

    model_path = os.path.join('./Training/Saved_Models/PPO_highway_best/best_model')

    # train_new_model("PPO_highway", total_timesteps=100000)
    # train_exist_model(model_path, total_timesteps=100000)

    eval_model(model_path)


