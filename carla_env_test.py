from carla_env.carla_env_multi_obs import CarlaEnv
from utils.clean_actors import clean_actors

def test_carla_env():
    """
    Test the CarlaEnv class to see if it works
    """
    clean_actors()

    env = CarlaEnv()

    num_episodes = 20

    try:
        for episode in range(num_episodes):
            obs = env.reset()
            done = False
            total_reward = 0

            while not done:
                action = env.action_space.sample()

                obs, reward, done, info = env.step(action)

                env.render(mode='human')
                total_reward += reward

                if done:
                    print(total_reward)
                    break
    finally:
        env.close()


if __name__ == '__main__':
    test_carla_env()
