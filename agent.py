import torch


class EpsilonGreedy:
    def __init__(self, epsilon, rng, device):
        self.epsilon = epsilon
        self.rng = rng
        self.device = device

    # exploration vs exploitation
    def choose_action(self, action_space, observation, net):
        explor_exploit_prob = self.rng.uniform(0, 1)

        # exploration
        if explor_exploit_prob < self.epsilon:
            action = action_space.sample()

        # exploitation
        else:
            # the purpose of unsqueeze is to add a 'batch dimension' of 1 at index 0 as the input of nn needs a batch
            obs_tensor = torch.from_numpy(observation).unsqueeze(0).to(self.device)
            # no_grad because we dont need gradient descent and backpropagation happenining here
            with torch.no_grad():
                q_values = net(obs_tensor)  # net is our neural network
            # action will have the index that has the highest q-value
            action = torch.argmax(q_values, dim=1).item()

        return action
