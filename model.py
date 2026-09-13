import torch
import torch.nn as nn


class DQN(nn.Module):
    def __init__(self, num_actions, num_frames=4):
        super().__init__()
        # the convolution layer involves three hidden layers
        self.conv = nn.Sequential(
            nn.Conv2d(num_frames, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        # 84x84 input -> conv stack above -> 64 channels of 7x7 feature maps
        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions),
        )
        # the output is converted to 18 actions each with their Q-value

    def forward(self, x):
        # this is the normalization step
        x = x.float() / 255.0
        # calling the conv layer
        x = self.conv(x)
        # flattening (64, 7, 7) -> (3136, )
        x = x.flatten(start_dim=1)
        # passing the flattend into fully-connected network
        return self.fc(x)
