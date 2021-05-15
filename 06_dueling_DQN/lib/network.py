import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



class DuelingDQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DuelingDQN, self).__init__()

        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size = 8, stride = 4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size = 4, stride = 2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size = 3, stride = 1)

        conv_out = self.get_conv_out(input_shape)

        self.fc_a1 = nn.Linear(conv_out, 512)
        self.fc_a2 = nn.Linear(512, n_actions)

        self.fc_v1 = nn.Linear(conv_out, 512)
        self.fc_v2 = nn.Linear(512, 1)

    
    def get_conv_out(self, input_dims):
        shape = torch.zeros(1, *input_dims)
        dims = self.conv1(shape)
        dims = self.conv2(dims)
        dims = self.conv3(dims)
        return int(np.prod(dims.size()))

    
    def forward(self, dims):
        x = F.relu(self.conv1(dims))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        out = x.view(x.size()[0], -1)

        adv = F.relu(self.fc_a1(out))
        adv = self.fc_a2(adv)

        val = F.relu(self.fc_v1(out))
        val = self.fc_v2(val)

        return val + (adv - adv.mean(dim = 1, keepdim = True))