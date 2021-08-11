from mlagents.torch_utils import torch, default_device
import numpy as np

# taken from openai baselines:
# https://github.com/openai/baselines/blob/master/baselines/common/running_mean_std.py

class RunningMeanStd(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = torch.zeros(shape).float()
        self.var = torch.ones(shape).float()
        self.count = epsilon

    def update(self, x):
        """
        Update the mean and var with new data
        x : 
        """
        if(x.shape[0]>1):
            batch_mean = torch.mean(x.detach(), axis=0)
            batch_var = torch.var(x.detach(), axis=0)
            batch_count = x.shape[0]
            self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count)
        self.var = torch.where( self.var < torch.as_tensor(1e-5).float(),  torch.as_tensor(1.).float(), self.var)

    def set_mean_var(self, mean, var):
        self.mean = mean
        self.var = var

def update_mean_var_count_from_moments(mean, var, count, batch_mean, batch_var, batch_count):
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + torch.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count