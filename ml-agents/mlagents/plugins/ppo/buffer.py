import numpy as np

class RolloutBuffer:
    """
    Buffer which contains information of the trajectory of a single agent
    Saves a list of actions, states (observations), rewards, log probabilities and 
    episode state at each timestep.
    """
    def __init__(self):

        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
        self.values = []
    

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]
        del self.values[:]

    def extend(self, buffer : 'RolloutBuffer'):
        self.actions.extend(buffer.actions)
        self.states.extend(buffer.states)
        self.logprobs.extend(buffer.logprobs)
        self.rewards.extend(buffer.rewards)
        self.is_terminals.extend(buffer.is_terminals)
        self.values.extend(buffer.values)

    def to_numpy(self):
        """
        converts actions, states, rewards, and logprobs to numpy arrays
        doesnt convert is terminal (bool list)
        """
        actions = np.array(self.actions, dtype=np.float32)
        states = np.array(self.states, dtype=np.float32)
        rewards = np.array(self.rewards, dtype=np.float32)
        logprobs = np.array(self.logprobs, dtype=np.float32)
        is_terminals = np.array(self.is_terminals, dtype=np.float32)
        values = np.array(self.values, dtype = np.float32)

        return actions, states, rewards, logprobs, is_terminals, values

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, index):
        sliced_buffer = RolloutBuffer()

        sliced_buffer.actions = self.actions[index]
        sliced_buffer.states = self.states[index]
        sliced_buffer.rewards = self.rewards[index]
        sliced_buffer.logprobs = self.logprobs[index]
        sliced_buffer.is_terminals = self.is_terminals[index]
        sliced_buffer.values = self.values[index]

        return sliced_buffer 
