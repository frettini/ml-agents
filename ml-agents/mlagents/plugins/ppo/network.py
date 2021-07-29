from mlagents.torch_utils import torch, default_device

# dictionary of various activation function that are specified in the options
activation_dict = {"tanh":torch.nn.Tanh(), "sigmoid":torch.nn.Sigmoid(),
                   "relu":torch.nn.ReLU(), "leakyrelu":torch.nn.LeakyReLU(),
                   "identity":torch.nn.Identity(), "softmax":torch.nn.Softmax()}

class MLPNet(torch.nn.Module):
    def __init__(self, input_dim:int,
                 output_dim:int,
                 hidden_dim:int=64,
                 num_layers:int=1, 
                 mid_activation = torch.nn.Tanh(),
                 last_activation=torch.nn.Tanh()):

        super(MLPNet, self).__init__()
        
        self.layers = torch.nn.ModuleList()
        
        seq = []
        seq.append(torch.nn.Linear(input_dim, hidden_dim))
        seq.append(mid_activation)

        for i in range(num_layers):
            seq.append(torch.nn.Linear(hidden_dim, hidden_dim))
            seq.append(mid_activation)
        
        seq.append(torch.nn.Linear(hidden_dim, output_dim))
        seq.append(last_activation)

        self.model = torch.nn.Sequential(*seq)

        
    def forward(self, input):
        output = self.model.forward(input)
        return output


class Discriminator(torch.nn.Module):
    def __init__(self, options):
        """
        Discriminator which drives the adversarial loss of the reinforcement learning process
        The discriminator takes two frames features, concatenated together, as input
        """
        super(Discriminator, self).__init__()

        self.discrim = MLPNet(input_dim = options["input_dim_discrim"], 
                              output_dim= 1, 
                              num_layers=options["num_layers_discrim"],
                              hidden_dim=options["hidden_units_discrim"],
                              mid_activation=activation_dict[options["mid_activation_discrim"]],
                              last_activation=activation_dict[options["last_activation_discrim"]])

        self.real_label = 1
        self.false_label = -1

    def forward(self, input):
        output = self.discrim(input)
        return torch.sigmoid(output)

    def G_reward(self, input):
        """
        Calculate the generator reward for creating realistic data.
        The reward has the form  max[0,1-0.25(D-1)^2].

        :params input: 2D tensor [batch_size, input_dim] data coming from the simulation
        :returns reward: the reward for being able to fool the discriminator
        """
        # reward from discriminator :
        temp = 1-0.25*(self.forward(input)-1)**2
        reward = torch.maximum(temp, torch.zeros_like(temp))
        return reward

    def D_loss(self, real_input, fake_input):
        """
        Calculate the discriminator loss, which has the form E_m[log(D)] - E_pi[log(1-D)]
        with m the adversarial dataset and pi the current policy 1 for dataset and -1 for policy.

        :params real_input: 2D tensor [batch_size, input_dim] data coming from the adversarial dataset
        :params fake_input: 2D tensor [batch_size, input_dim] data coming from the simulation
        
        :returns d_loss: The discriminator output
        """
        curr_batch_size = real_input.shape[0]
        label = torch.full((curr_batch_size,), self.real_label, dtype=torch.float, device=default_device())
        
        # get the classification of real input and compare to labels
        real_estimation = self.forward(real_input) 
        loss_real = self.criterion_gan(real_estimation,label)
        
        # do the same with the generator's poses
        label.fill_(self.fake_label)
        fake_estimation = self.forward(fake_input.float().detach())
        loss_fake = self.criterion_gan(fake_estimation, label)
        
        # calculate overall loss 
        d_loss = loss_real + loss_fake
        return d_loss
        
    def optimize(self, real_input, fake_input):
        """
        Calculate the discriminator and generators losses. 
        Update the discriminator before using it to compute the generator's loss
        """
        self.zero_grad()
        d_loss = self.D_loss(real_input, fake_input)
        d_loss.backward()
        d_loss.step()

        g_loss = self.G_reward(fake_input)
        return g_loss




if __name__ == "__main__":
    actor = MLPNet(40,10,20,last_activation=torch.nn.Identity())
    res = actor.forward(torch.rand((40,),dtype=torch.float32))
    print(res)