import os
import carla
from carla_env.carla_env_multi_obs import CarlaEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from utils.clean_actors import clean_actors

class ModelEvaluator:

    def __init__(self, model_name):

        model_path_base = "trained_models"

        self.model_name = model_name
        self.model_path = os.path.join(model_path_base, f"{model_name}.zip")

        self.connect_to_simulator()

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


    def evaluate_model(self):

        print(f"Evaluating model: {self.model_name}.zip")

        if os.path.exists(self.model_path):

            # evaluate the model
            env = lambda: CarlaEnv()
            env = DummyVecEnv([env])

            model = PPO.load(self.model_path, env=env, verbose=1)

            obs = env.reset()

            while True:
                action, _states = model.predict(obs, deterministic=True)
                obs, rewards, dones, info = env.step(action)
                env.render(mode='human')

        else:
            print("Model does not exist")


if __name__ == '__main__':
    clean_actors()  # clean the environment

    model_name = "PPO_highway_4" # doesn't need to include .zip

    evaluator = ModelEvaluator(model_name=model_name)

    evaluator.evaluate_model()