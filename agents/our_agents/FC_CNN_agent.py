import torch
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# both fully connected and convolutional neural network
# to handle telemetry data and camera images
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