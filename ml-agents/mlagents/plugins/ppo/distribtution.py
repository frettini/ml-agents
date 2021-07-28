import math
from mlagents.torch_utils.torch import torch

EPSILON = 1e-7  # Small value to avoid divide by zero

# taken from mlagent.trainers.torch
class GaussianDistInstance:
    def __init__(self, mean, std):
        super().__init__()
        self.mean = mean
        self.std = std

    def sample(self):
        sample = self.mean + torch.randn_like(self.mean) @ self.std
        return sample

    def log_prob(self, value):
        # one dimensional log probability
        # var = self.std ** 2
        # log_scale = torch.log(self.std + EPSILON)
        # return (
        #     -((value - self.mean) ** 2) / (2 * var + EPSILON)
        #     - log_scale
        #     - math.log(math.sqrt(2 * math.pi))
        # )
        
        ivar = torch.diag(1/(torch.diag(self.std)**2 + EPSILON))
        torch.log(1/torch.det(((self.std.double())**2 + EPSILON)))
        temp1 = 0.5 * self.std.shape[0] *  math.log(2*math.pi)
        # temp2 = 0.5 * torch.log(1/torch.det(ivar.double())).float()
        # temp2 = 0.5 * torch.log(torch.det((self.std.double()**2 + EPSILON))).float()
        temp2 = 0.5 * torch.sum(torch.log(torch.diag(self.std)**2 + EPSILON))
        temp3 = 0.5 * torch.sum(((value - self.mean) @ (ivar)) * (value - self.mean), dim=1)

        return (-temp1 - temp2 - temp3)

    def pdf(self, value):
        log_prob = self.log_prob(value)
        return torch.exp(log_prob)

    def entropy(self):
        return torch.mean(
            0.5 * torch.log(2 * math.pi * math.e * self.std ** 2 + EPSILON),
            dim=1,
            keepdim=True,
        )  # Use equivalent behavior to TF
        

    def exported_model_output(self):
        return self.sample()


class CategoricalDistInstance:
    def __init__(self, logits):
        super().__init__()
        self.logits = logits
        self.probs = torch.softmax(self.logits, dim=-1)

    def sample(self):
        return torch.multinomial(self.probs, 1)

    def pdf(self, value):
        # This function is equivalent to torch.diag(self.probs.T[value.flatten().long()]),
        # but torch.diag is not supported by ONNX export.
        idx = torch.arange(start=0, end=len(value)).unsqueeze(-1)
        return torch.gather(
            self.probs.permute(1, 0)[value.flatten().long()], -1, idx
        ).squeeze(-1)

    def log_prob(self, value):
        return torch.log(self.pdf(value) + EPSILON)

    def all_log_prob(self):
        return torch.log(self.probs + EPSILON)

    def entropy(self):
        return -torch.sum(
            self.probs * torch.log(self.probs + EPSILON), dim=-1
        ).unsqueeze(-1)

    def exported_model_output(self):
        return self.sample()

if __name__=="__main__":
    
    mean = torch.zeros((10,2))
    mean[:,1] = 1
    std = torch.rand_like(mean)

    sample = torch.rand((10,2))

    gauss = GaussianDistInstance(mean, std)
    print("sampling : ", gauss.sample())
    print("log prob : ", gauss.log_prob(sample))