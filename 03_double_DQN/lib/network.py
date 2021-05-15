import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



class DDQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DDQN, self).__init__()

        self.input_shape = input_shape
        self.n_actions = n_actions

        self.conv1 = nn.Conv2d(self.input_shape[0], 32, kernel_size = 8, stride = 4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size = 4, stride = 2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size = 3, stride = 1)

        conv_out = self.get_conv_out(self.input_shape)

        self.fc1 = nn.Linear(conv_out, 512)
        self.fc2 = nn.Linear(512, n_actions)

    
    def get_conv_out(self, input_dims):
        tmp = torch.zeros(1, *input_dims)
        dims = self.conv1(tmp)
        dims = self.conv2(dims)
        dims = self.conv3(dims)
        return int(np.prod(dims.size()))

    
    def forward(self, obs):
        x = F.relu(self.conv1(obs))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x