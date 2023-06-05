import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.1):
        super(NoisyLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.Tensor(out_features, in_features))
        self.register_buffer("weight_epsilon", torch.Tensor(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_sigma = nn.Parameter(torch.Tensor(out_features))
        self.register_buffer("bias_epsilon", torch.Tensor(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(
            self.std_init / math.sqrt(self.in_features)
        )
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(
            self.std_init / math.sqrt(self.out_features)
        )

    def reset_noise(self):
        epsilon_in = self.scale_noise(self.in_features)
        epsilon_out = self.scale_noise(self.out_features)

        # outer product
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x):
        return F.linear(
            x,
            self.weight_mu + self.weight_sigma * self.weight_epsilon,
            self.bias_mu + self.bias_sigma * self.bias_epsilon,
        )

    @staticmethod
    def scale_noise(size):
        x = torch.randn(size)

        return x.sign().mul(x.abs().sqrt())


class RainbowDQNNetwork(nn.Module):
    def __init__(self, input_dims, n_actions, atom_size, support, name, chkpt_dir):
        super(RainbowDQNNetwork, self).__init__()

        self.support = support
        self.n_actions = n_actions
        self.atom_size = atom_size

        #for saving our network
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)

        #convolutional part of network
        self.cnn = nn.Sequential(
            nn.Conv2d(input_dims[0], 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
        )

        # noisy linear part
        noisy_input_dims = self.conv_dims(input_dims)
        # advantage layer
        self.advantage_hidden_layer = NoisyLinear(noisy_input_dims, 512)
        self.advantage_layer = NoisyLinear(512, n_actions * atom_size)

        # value layer
        self.value_hidden_layer = NoisyLinear(noisy_input_dims, 512)
        self.value_layer = NoisyLinear(512, atom_size)

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def conv_dims(self, input_dims):
        base = torch.zeros(*input_dims)
        dims = self.cnn(base)
        return int(np.prod(dims.size()))

    def dist(self, x):
        feature = self.cnn(x)
        if len(feature.size()) == 4:
            feature_flat = feature.view(feature.size()[0], -1)
        else:
            feature_flat = feature.view(-1)
        adv_hid = F.relu(self.advantage_hidden_layer(feature_flat))
        val_hid = F.relu(self.value_hidden_layer(feature_flat))

        advantage = self.advantage_layer(adv_hid).view(-1, self.n_actions, self.atom_size)
        value = self.value_layer(val_hid).view(-1, 1, self.atom_size)
        q_atoms = value + advantage - advantage.mean(dim=1, keepdim=True)

        dist = F.softmax(q_atoms, dim=-1)
        dist = dist.clamp(min=1e-3)  # for avoiding nans

        return dist

    def forward(self, state):
        dist = self.dist(state)
        q = torch.sum(dist * self.support, dim=2)
        return q

    def reset_noise(self):
        self.advantage_hidden_layer.reset_noise()
        self.advantage_layer.reset_noise()
        self.value_hidden_layer.reset_noise()
        self.value_layer.reset_noise()

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(torch.load(self.checkpoint_file))