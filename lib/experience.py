import numpy as np
import random
import collections



class ReplayBuffer(object):
    def __init__(self, capacity, n_step, gamma):
        self.capacity = capacity
        self.n_step = n_step
        self.gamma = gamma
        self.memory = collections.deque(maxlen = self.capacity)
        self.n_step_buffer = collections.deque(maxlen = self.n_step)


    def __len__(self):
        return len(self.memory)


    def get_n_step_info(self):
        reward, next_observation, done = self.n_step_buffer[-1][-3:]
        for _, _, rew, next_obs, do in reversed(list(self.n_step_buffer)[: -1]):
            reward = self.gamma * reward + rew
            next_observation, done = (next_obs, do) if do else (next_observation, done)
        return reward, next_observation, done


    def store(self, observation, action, reward, next_observation, done):
        observation = np.expand_dims(observation, 0)
        next_observation = np.expand_dims(next_observation, 0)


        self.n_step_buffer.append([observation, action, reward, next_observation, done])
        if len(self.n_step_buffer) < self.n_step:
            return
        reward, next_observation, done = self.get_n_step_info()
        observation, action = self.n_step_buffer[0][: 2]
        self.memory.append([observation, action, reward, next_observation, done])


    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        observation, action, reward, next_observation, done = zip(*batch)
        return np.concatenate(observation, 0), action, reward, np.concatenate(next_observation, 0), done
