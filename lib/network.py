import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



class NoisyLinear(nn.Module):
    def __init__(self, input_dim, output_dim, std_init=0.4):
        super(NoisyLinear, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.FloatTensor(self.output_dim, self.input_dim))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(self.output_dim, self.input_dim))
        self.register_buffer('weight_epsilon', torch.FloatTensor(self.output_dim, self.input_dim))

        self.bias_mu = nn.Parameter(torch.FloatTensor(self.output_dim))
        self.bias_sigam = nn.Parameter(torch.FloatTensor(self.output_dim))
        self.register_buffer('bias_epsilon', torch.FloatTensor(self.output_dim))

        self.reset_parameter()
        self.reset_noise()

    def forward(self, input):
        if self.training:
            weight = self.weight_mu + self.weight_sigma.mul(self.weight_epsilon)
            bias = self.bias_mu + self.bias_sigam.mul(self.bias_epsilon)
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(input, weight, bias)

    def _scale_noise(self, size):
        noise = torch.randn(size)
        noise = noise.sign().mul(noise.abs().sqrt())
        return noise

    def reset_parameter(self):
        mu_range = 1. / np.sqrt(self.input_dim)

        self.weight_mu.detach().uniform_(-mu_range, mu_range)
        self.bias_mu.detach().uniform_(-mu_range, mu_range)

        self.weight_sigma.detach().fill_(self.std_init / np.sqrt(self.input_dim))
        self.bias_sigam.detach().fill_(self.std_init / np.sqrt(self.output_dim))

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.input_dim)
        epsilon_out = self._scale_noise(self.output_dim)

        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(self._scale_noise(self.output_dim))




class RainbowDQN(nn.Module):
    def __init__(self, input_dims, n_actions, atoms, vmin, vmax):
        super(RainbowDQN, self).__init__()

        self.input_dims = input_dims
        self.n_actions = n_actions
        self.atoms = atoms
        self.vmin = vmin
        self.vmax = vmax


        self.conv1 = nn.Conv2d(self.input_dims[0], 32, kernel_size = 8, stride = 4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size = 4, stride = 2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size = 3, stride = 1)

        conv_out = self.get_conv_out(self.input_dims)

        self.value_noisy1 = NoisyLinear(conv_out, 512) 
        self.value_noisy2 = NoisyLinear(512, self.atoms)

        self.adv_noisy1 = NoisyLinear(conv_out, 512)
        self.adv_noisy2 = NoisyLinear(512, self.n_actions * self.atoms)

    
    def get_conv_out(self, input_shape):
        dims = torch.zeros(1, *input_shape)
        dims = self.conv1(dims)
        dims = self.conv2(dims)
        dims = self.conv3(dims)
        return int(np.prod(dims.size()))

    
    def forward(self, dims):
        batch_size = dims.size(0)

        x = F.relu(self.conv1(dims))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        out = x.view(x.size(0), -1)

        value = F.relu(self.value_noisy1(out))
        value = self.value_noisy2(value)

        advantage = F.relu(self.adv_noisy1(out))
        advantage = self.adv_noisy2(advantage)

        value = value.view(batch_size, 1, self.atoms)
        advantage = advantage.view(batch_size, self.n_actions, self.atoms)

        dist = value + advantage - advantage.mean(1, keepdim = True)
        #dist = F.softmax(dist, 2)
        dist = F.softmax(dist.view(-1, self.atoms), 1).view(-1, self.n_actions, self.atoms)
        return dist

    
    def reset_noise(self):
        self.value_noisy1.reset_noise()
        self.value_noisy2.reset_noise()
        self.adv_noisy1.reset_noise()
        self.adv_noisy2.reset_noise()
