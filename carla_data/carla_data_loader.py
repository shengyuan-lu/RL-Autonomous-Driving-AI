import os
import cv2

from carla_data.carla_data_collector import DataCollector

def run_carla_data_loader(dataset_name, dataset_directory='../dataset_output'):

    env = DataCollector(env=None)

    dataset_location = os.path.join(dataset_directory, dataset_name)

    # check if dataset_dir is a directory
    if os.path.exists(dataset_location) and os.path.isfile(dataset_location):

        # load the dataset
        print(f"Loading dataset: {dataset_location}")
        dataset = env._load_dataset(load_dir=dataset_location)

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

    else:
        print(f"Error: {dataset_location} is not a file or does not exist")

if __name__ == '__main__':
    print("Running carla_data_loader.py")

    dataset_name = 'dataset_70.pkl'

    run_carla_data_loader(dataset_name)