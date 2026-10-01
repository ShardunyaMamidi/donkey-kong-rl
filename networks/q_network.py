import torch.nn as nn

from networks.encoder import NatureCNN


# shared CNN encoder + fully-connected head outputting one Q-value per action
class QNetwork(nn.Module):
    def __init__(self, num_actions, num_frames=4):
        super().__init__()
        self.encoder = NatureCNN(num_frames)
        self.head = nn.Sequential(
            nn.Linear(self.encoder.out_features, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions),
        )
        # the output is converted to 18 actions each with their Q-value

    def forward(self, x):
        return self.head(self.encoder(x))
