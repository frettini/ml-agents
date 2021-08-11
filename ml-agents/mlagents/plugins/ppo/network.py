from mlagents.torch_utils import torch, default_device
from mlagents.plugins.ppo.running_mean import RunningMeanStd
import mlagents.plugins.utils.logger as log

# dictionary of various activation function that are specified in the options
activation_dict = {"tanh":torch.nn.Tanh(), "sigmoid":torch.nn.Sigmoid(),
                   "relu":torch.nn.ReLU(), "leakyrelu":torch.nn.LeakyReLU(),
                   "identity":torch.nn.Identity(), "softmax":torch.nn.Softmax()}

class MLPNet(torch.nn.Module):
    def __init__(self, input_dim:int,
                 output_dim:int,
                 hidden_dim:list=[64],
                 num_layers:int=1, 
                 mid_activation = torch.nn.Tanh(),
                 last_activation=torch.nn.Tanh()):

        super(MLPNet, self).__init__()
        
        self.layers = torch.nn.ModuleList()
        
        seq = []

        input_list = [input_dim] + hidden_dim
        output_list = hidden_dim + [output_dim] 

        for i in range(len(input_list)):

            seq.append(torch.nn.Linear(input_list[i], output_list[i]))

            if i == len(input_list)-1:
                seq.append(last_activation)
            else:    
                seq.append(mid_activation)

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

        self.optimizer = torch.optim.Adam(self.discrim.parameters(), options["lr_discrim"], weight_decay=0.0005)

        self.real_label = 1
        self.fake_label = -1
        self.grad_pernalty_factor = options["grad_penalty_factor"]

        self.cumul_d_loss = 0
        self.cumul_g_reward = 0
        self.cumul_grad_penalty = 0

        self.criterion_gan = torch.nn.MSELoss()

        self.running_mean_std_real = RunningMeanStd(shape=options["input_dim_discrim"])
        self.running_mean_std_fake = RunningMeanStd(shape=options["input_dim_discrim"])

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

        self.running_mean_std_fake.update(input)
        input = (input - self.running_mean_std_fake.mean)/self.running_mean_std_fake.var

        # reward from discriminator :
        temp = 1-0.25*(self.forward(input)-1)**2
        reward = torch.maximum(temp, torch.zeros_like(temp))
        self.cumul_g_reward += reward
        return reward

    def D_loss(self, real_input, fake_input):
        """
        Calculate the discriminator loss, which has the form E_m[log(D)] - E_pi[log(1-D)]
        with m the adversarial dataset and pi the current policy 1 for dataset and -1 for policy.

        :params real_input: 2D tensor [batch_size, input_dim] data coming from the adversarial dataset
        :params fake_input: 2D tensor [batch_size, input_dim] data coming from the simulation
        
        :returns d_loss: The discriminator output
        """

        # normalize input 
        self.running_mean_std_real.update(real_input)
        self.running_mean_std_fake.update(fake_input)
        fake_input = (fake_input - self.running_mean_std_real.mean)/self.running_mean_std_real.var
        fake_input = (fake_input - self.running_mean_std_fake.mean)/self.running_mean_std_fake.var

        # generate label vector which contains 1 or -1 for LSGAN
        curr_batch_size = real_input.shape[0]
        label = torch.full((curr_batch_size,), self.real_label, dtype=torch.float, device=default_device())
        
        # get the classification of real input and compare to labels
        real_estimation = self.forward(real_input).squeeze()
        loss_real = self.criterion_gan(real_estimation,label)
        
        # gradient penalty 
        discriminator_gradient =torch.autograd.grad(loss_real, self.discrim.parameters(), retain_graph=True, create_graph=True)
        grad_norm = 0
        for grad in discriminator_gradient:
            grad_norm += grad.pow(2).sum()
        # grad_norm = grad_norm.sqrt()
        loss_real += (self.grad_pernalty_factor/2) * grad_norm
        loss_real.backward()

        self.cumul_grad_penalty += grad_norm.detach()

        # do the same with the generator's poses
        label.fill_(self.fake_label)
        fake_estimation = self.forward(fake_input.float().detach()).squeeze()
        loss_fake = self.criterion_gan(fake_estimation, label)
        loss_fake.backward()
        
        self.cumul_d_loss += loss_real.detach() + loss_fake.detach() + grad_norm.detach()

    def optimize(self, real_input, fake_input):
        """
        Calculate the discriminator and generators losses. 
        Update the discriminator before using it to compute the generator's loss
        """
        self.zero_grad()
        self.D_loss(real_input, fake_input)
        self.optimizer.step()

    def save(self, checkpoint_path):

        save_model_dict = {'model_state_dict' : self.discrim.state_dict(),
                    'optimizer_state_dict' : self.optimizer.state_dict(),
                    'running_mean_fake' : self.running_mean_std_fake.mean,
                    'running_std_fake' : self.running_mean_std_fake.var,
                    'running_mean_real' : self.running_mean_std_real.mean,
                    'running_std_real' : self.running_mean_std_real.var,
                    }

        torch.save(save_model_dict, checkpoint_path)

    def load(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)

        self.discrim.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.running_mean_std_fake.mean = checkpoint['running_mean_fake']
        self.running_mean_std_fake.var = checkpoint['running_std_fake']
        self.running_mean_std_real.mean = checkpoint['running_mean_real']
        self.running_mean_std_real.var = checkpoint['running_std_real']


        



if __name__ == "__main__":
    actor = MLPNet(40,10,[30,20],2,last_activation=torch.nn.Identity())
    res = actor.forward(torch.rand((40,),dtype=torch.float32))
    print(res)