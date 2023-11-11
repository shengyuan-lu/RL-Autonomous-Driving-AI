import os

import carla

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback

from stable_baselines3.common.vec_env import DummyVecEnv

from carla_env.carla_env_multi_obs import CarlaEnv

from utils.clean_actors import clean_actors

from agents.our_agents.FC_CNN_agent import FC_CNN


policy_kwargs = dict(
    features_extractor_class=FC_CNN,
    features_extractor_kwargs=dict(features_dim=512),
)


class ModelTrainer:

    def __init__(self, new_model_name, exist_model_name='', total_timesteps=100000):
        self.new_model_name = new_model_name
        self.exist_model_name = exist_model_name + '.zip'
        self.total_timesteps = total_timesteps

        self.set_model_and_log_paths()

        self.connect_to_simulator()



    def set_model_and_log_paths(self):

        model_path_base = "trained_models"
        log_path_base = "training_logs"

        if not os.path.exists(model_path_base):
            os.makedirs(model_path_base)

        if not os.path.exists(log_path_base):
            os.makedirs(log_path_base)

        model_version = 1

        while os.path.exists(f'{model_path_base}/{self.new_model_name}_{model_version}.zip'):
            model_version += 1

        self.model_best_name = f'{self.new_model_name}_best.zip'

        self.new_model_name = f'{self.new_model_name}_{model_version}'

        self.new_model_path = os.path.join('trained_models', self.new_model_name)

        self.model_best_path = os.path.join('trained_models', self.model_best_name)

        self.exist_model_path = os.path.join('trained_models', f'{self.exist_model_name}')

        self.log_path = os.path.join(log_path_base)

    def connect_to_simulator(self):

        print('Connecting to simulator...')

        try:
            # connect to simulator
            client = carla.Client('localhost', 2000)
            client.set_timeout(10.0)

            # initialize world
            world = client.get_world()
            client.load_world('Town05')
            print("Successfully connected to simulator")

        except:
            print("Failed to connect to simulator")

    # train the new model
    def train_new_model(self):

        env = lambda: CarlaEnv()
        env = DummyVecEnv([env])

        model = PPO('MultiInputPolicy', env, verbose=1, tensorboard_log=self.log_path)

        eval_env = model.get_env()

        eval_callback = EvalCallback(eval_env=eval_env, best_model_save_path=self.model_best_path,
                                     n_eval_episodes=8,
                                     eval_freq=5000, verbose=1,
                                     deterministic=True, render=False)

        model.learn(total_timesteps=self.total_timesteps, tb_log_name=self.new_model_name, callback=eval_callback,
                    reset_num_timesteps=False)

        model.save(self.new_model_path)


    # train the existing model
    def train_exist_model(self):

        env = lambda: CarlaEnv()
        env = DummyVecEnv([env])

        if os.path.exists(self.exist_model_path):

            # load the model
            model = PPO.load(self.exist_model_path, env=env, verbose=1, tensorboard_log=self.log_path)

            # evaluate the model
            eval_env = model.get_env()

            # save the best model
            eval_callback = EvalCallback(eval_env=eval_env, best_model_save_path=self.model_best_path,
                                         n_eval_episodes=8,
                                         eval_freq=5000, verbose=1,
                                         deterministic=True, render=False)

            model.learn(total_timesteps=self.total_timesteps, tb_log_name=self.new_model_name, callback=eval_callback,
                        reset_num_timesteps=False)

            model.save(self.new_model_path)

        else:
            print(f"There's no existing model {self.exist_model_name}")

    def train_model(self, train_new=True):

        if train_new:
            print(f"Training new model: {self.new_model_name}")
            self.train_new_model()

        else:
            print(f"Training on existing model: {self.exist_model_name}")
            print(f"Saving as a new model: {self.new_model_name}")
            self.train_exist_model()


if __name__ == '__main__':

    clean_actors()

    trainer = ModelTrainer(new_model_name='PPO_highway', exist_model_name='PPO_highway_1', total_timesteps=1000)

    trainer.train_model(train_new=False)

