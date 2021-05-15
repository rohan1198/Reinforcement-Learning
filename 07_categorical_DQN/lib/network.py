import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np




class NoisyLinear(nn.Module):
    def __init__(self, input_dim, output_dim, std_init = 0.5, bias = True):
        super(NoisyLinear, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.std_init = std_init
        self.bias = bias

        self.weight_mu = nn.Parameter(torch.FloatTensor(self.output_dim, self.input_dim))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(self.output_dim, self.input_dim))
        self.register_buffer("weight_epsilon", torch.FloatTensor(self.output_dim, self.input_dim))

        if bias:
            self.bias_mu = nn.Parameter(torch.FloatTensor(self.output_dim))
            self.bias_sigma = nn.Parameter(torch.FloatTensor(self.output_dim))
            self.register_buffer("bias_epsilon", torch.FloatTensor(self.output_dim))

        self.reset_parameter()
        self.reset_noise()

    
    def reset_parameter(self):
        mu_range = 1 / np.sqrt(self.input_dim)

        self.weight_mu.detach().uniform_((-1)*mu_range, mu_range)
        self.bias_mu.detach().uniform_((-1)*mu_range, mu_range)

        self.weight_sigma.detach().fill_(self.std_init / np.sqrt(self.input_dim))
        self.bias_sigma.detach().fill_(self.std_init / np.sqrt(self.output_dim))


    def scale_noise(self, size):
        noise = torch.randn(size)
        noise = noise.sign().mul(noise.abs().sqrt())
        return noise

    
    def reset_noise(self):
        epsilon_in = self.scale_noise(self.input_dim)
        epsilon_out = self.scale_noise(self.output_dim)

        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(self.scale_noise(self.output_dim))


    def forward(self, input):
        weight = self.weight_mu + self.weight_sigma.mul(self.weight_epsilon)
        bias = self.bias
        
        if bias is not None:
            bias = self.bias_mu + self.bias_sigma.mul(self.bias_epsilon)
        
        return F.linear(input, weight, bias)



class CategoricalDQN(nn.Module):
    def __init__(self, input_dims, n_actions, atoms, vmin, vmax):
        super(CategoricalDQN, self).__init__()

        self.input_dims = input_dims
        self.n_actions = n_actions
        self.atoms = atoms
        self.vmin = vmin
        self.vmax = vmax

        self.conv1 = nn.Conv2d(self.input_dims[0], 32, kernel_size= 8, stride= 4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size= 4, stride= 2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size= 3, stride= 1)

        conv_out = self.get_conv_out(self.input_dims)

        self.noisy1 = NoisyLinear(conv_out, 512)
        self.noisy2 = NoisyLinear(512, self.n_actions * self.atoms)

    
    def get_conv_out(self, input_shape):
        dims = torch.zeros(1, *input_shape)
        dims = self.conv1(dims)
        dims = self.conv2(dims)
        dims = self.conv3(dims)
        return int(np.prod(dims.size()))

    
    def forward(self, dims):
        x = F.relu(self.conv1(dims))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        out = x.view(x.size(0), -1)

        x = F.relu(self.noisy1(out))
        x = self.noisy2(x)
        x = F.softmax(x.view(-1, self.atoms), 1).view(-1, self.n_actions, self.atoms)
        return x

    
    def reset_noise(self):
        self.noisy1.reset_noise()
        self.noisy2.reset_noise()


    