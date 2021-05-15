import numpy as np
import collections
import torch



Experience = collections.namedtuple("Experience", field_names = ["state", "action", "reward", "next_state", "done"])



class Agent(object):
    def __init__(self, env, exp_buffer):
        self.env = env
        self.exp_buffer = exp_buffer
        self.reset()

    
    def reset(self):
        self.state = self.env.reset()
        self.total_reward = 0.0

    
    def play_step(self, net, epsilon, vmin, vmax, atoms, device):
        done_reward = None

        if np.random.random() < epsilon:
            action = self.env.action_space.sample()
        else:
            state_a = np.array([self.state], copy = False)
            state_v = torch.tensor(state_a).to(device)
            dist = net.forward(state_v)
            dist = dist.detach()
            space = torch.linspace(vmin, vmax, atoms).to(device)
            dist = dist.mul(space)
            action = dist.sum(2).max(1)[1].detach()[0].item()

        new_state, reward, done, _ = self.env.step(action)
        self.total_reward += reward

        self.exp_buffer.store(self.state, action, reward, new_state, done)
        self.state = new_state

        if done:
            done_reward = self.total_reward
            self.reset()
        
        return done_reward
