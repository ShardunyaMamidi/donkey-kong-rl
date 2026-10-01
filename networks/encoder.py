import torch.nn as nn


# conv trunk from the Nature DQN paper, shared by every algorithm's network
# (Q-network for DQN, actor-critic for PPO, ...) - each adds its own head on top
class NatureCNN(nn.Module):
    def __init__(self, num_frames=4):
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
        self.out_features = 64 * 7 * 7

    def forward(self, x):
        # this is the normalization step
        x = x.float() / 255.0
        # calling the conv layer
        x = self.conv(x)
        # flattening (64, 7, 7) -> (3136, )
        return x.flatten(start_dim=1)
