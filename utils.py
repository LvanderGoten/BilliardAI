from collections import deque
from random import sample
import numpy as np
import torch.nn as nn


class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.deque = deque(maxlen=buffer_size)

    def __len__(self):
        return len(self.deque)

    def add_transition(self, transition):
        assert len(transition) == 5, "Definition of transition [s, a, r, s', done]"
        self.deque.append(transition)

    def get_batch(self, batch_size):
        capacity = len(self)
        if batch_size >= capacity:
            if capacity == 0:
                raise StopIteration("ReplayBuffer is empty!")
            else:
                # Return everything in buffer
                ix = range(capacity)
        else:
            # Use random elements to break temporal correlation
            ix = sample(population=range(len(self)), k=batch_size)

        # Retrieve elements (list of tuples)
        state_batch, action_batch, reward_batch, new_state_batch, done_batch = zip(*[self.deque[i] for i in ix])

        # Convert to NumPy arrays
        state_batch = np.array(state_batch)
        action_batch = np.array(action_batch)
        reward_batch = np.array(reward_batch).astype(np.float32)
        new_state_batch = np.array(new_state_batch)
        done_batch = np.array(done_batch).astype(np.float32)

        return state_batch, action_batch, reward_batch, new_state_batch, done_batch


def compute_output_shape(h_in, kernel_size, stride):
    return int((h_in - kernel_size)/stride + 1)


def resolve_activations(s):
    if s.startswith("torch.nn."):
        return getattr(nn, s.replace("torch.nn.", ""))
    else:
        raise ValueError("Only activations starting with 'torch.nn' are acceptable!")
