import numpy as np
import collections
import random



class ReplayBuffer(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = collections.deque(maxlen = self.capacity)


    def __len__(self):
        return len(self.memory)

    
    def store(self, observation, action, reward, next_observation, done):
        observation = np.expand_dims(observation, 0)
        next_observation = np.expand_dims(next_observation, 0)

        self.memory.append([observation, action, reward, next_observation, done])

    
    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        observation, action, reward, next_observation, done = zip(*batch)

        return np.concatenate(observation, 0), action, reward, np.concatenate(next_observation, 0), done
