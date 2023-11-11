from carla_data.carla_data_collector import DataCollector
from carla_env.carla_env import CarlaEnv


def test_carla_collector(save_dir='../dataset_output'):
    """
    Test the DataCollector class to see if it works
    """
    env = CarlaEnv()

    env = DataCollector(env, steps=1000, save_dir=save_dir)

    env.reset()
    done = False

    try:
        while not done:
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            env.render(mode='human')

    finally:
        env.close()


if __name__ == '__main__':
    print("Running carla_collector_test.py")
    test_carla_collector()