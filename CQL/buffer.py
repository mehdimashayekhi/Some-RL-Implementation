import numpy as np
import random
import torch
from collections import deque, namedtuple

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size, device):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.device = device
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        tmp = [e.state for e in experiences if e is not None and len(e.state)!=2]
        print(tmp)
        states = torch.from_numpy(np.stack([e.state for e in experiences if e is not None and len(e.state)!=2])).float().to(self.device)
        if states.shape != (self.batch_size, 4):
            states = torch.cat((states, torch.zeros((self.batch_size -states.shape[0] ), 4)), dim=0)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(self.device)
        if actions.shape != (self.batch_size, 1):
            actions = torch.cat((actions, torch.zeros(self.batch_size -states.shape[0], 1)), dim=0)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        if rewards.shape != (self.batch_size, 1):
            rewards = torch.cat((rewards, torch.zeros(self.batch_size -states.shape[0], 1)), dim=0)
        next_states = torch.from_numpy(np.stack([e.next_state for e in experiences if e is not None and len(e.state)!=2])).float().to(self.device)
        if next_states.shape != (self.batch_size, 4):
            next_states = torch.cat((next_states, torch.zeros(self.batch_size -next_states.shape[0], 4)), dim=0)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)
        if dones.shape != (self.batch_size, 1):
            dones = torch.cat((dones, torch.zeros(self.batch_size -states.shape[0], 1)), dim=0)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
