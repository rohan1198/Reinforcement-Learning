import numpy as np
import collections
import torch



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


    def play_step(self, net, epsilon = 0.05, device = "cpu"):
        done_reward = None

        if np.random.random() < epsilon:
            action = self.env.action_space.sample()
        else:
            state_a = np.array([self.state], copy = False)
            state_v = torch.tensor(state_a).to(device)
            q_vals_v = net(state_v)
            _, act_v = torch.max(q_vals_v, dim = 1)
            action = int(act_v.item())

        new_state, reward, done, _ = self.env.step(action)
        self.total_reward += reward

        self.exp_buffer.store(self.state, action, reward, new_state, done)
        self.state = new_state

        if done:
            done_reward = self.total_reward
            self.reset()
        
        return done_reward