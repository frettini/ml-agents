from mlagents.plugins.ppo import running_mean
import numpy as np
import random

from mlagents.torch_utils import torch, default_device
from mlagents_envs.environment import ActionTuple

from mlagents.plugins.ppo.distribtution import GaussianDistInstance, CategoricalDistInstance
from mlagents.plugins.ppo.network import MLPNet
from mlagents.plugins.ppo.buffer import RolloutBuffer
from mlagents.plugins.ppo.running_mean import RunningMeanStd

import mlagents.plugins.utils.logger as log

# dictionary of various activation function that are specified in the options
activation_dict = {"tanh":torch.nn.Tanh(), "sigmoid":torch.nn.Sigmoid(),
                   "relu":torch.nn.ReLU(), "leakyrelu":torch.nn.LeakyReLU(),
                   "identity":torch.nn.Identity(), "softmax":torch.nn.Softmax()}


class ActorCritic(torch.nn.Module):
    def __init__(self, behavior_specs, options, running_mean_std=None):
        
        """
        Actor Critic module
        Contains the actor and critic model, enables to sample an action from
        the actions distribution, and evaluate the state values and log probabilities
        
        :params state_dim: int, observation size
        :params action_dim: int, action size
        :params has_continuous_action_space: bool, determine the architecture of the MLPs and distributions
        :params action_std_init: float, initial value of the action 
        """

        super(ActorCritic, self).__init__()

        self.options = options
        self.has_continuous_action_space = options["continuous_space"]

        behavior_name = list(behavior_specs)[0]
        self.behavior_spec = behavior_specs[behavior_name]
        self.action_spec = behavior_specs[behavior_name].action_spec

        if options["rotation_repr"] == "6D":
            state_dim = self.behavior_spec.observation_specs[0].shape[0] + 44
        else:
            state_dim = self.behavior_spec.observation_specs[0].shape[0]
        action_dim = self.action_spec.continuous_size

        self.device = default_device()

        if self.has_continuous_action_space:
            self.action_dim = action_dim
            self.action_var = torch.full((action_dim,), options["action_std"]**2 ).to(self.device)

        # setup the actor which will process the observations and output a distribution
        if self.has_continuous_action_space :
            self.actor = MLPNet(state_dim, action_dim,
                                hidden_dim = options["hidden_units_actor"],
                                num_layers = options["num_layers_actor"],
                                mid_activation=activation_dict[options["mid_activation_actor"]],
                                last_activation=activation_dict[options["last_activation_actor"]]
                                )
        else:
            self.actor = MLPNet(state_dim, action_dim, 
                                hidden_dim = options["hidden_units_actor"], 
                                num_layers = options["num_layers_actor"],
                                mid_activation=activation_dict[options["mid_activation_actor"]],
                                last_activation=activation_dict[options["last_activation_actor"]])

        # setup the critic which takes the observations and outputs the Value
        self.critic = MLPNet( state_dim, 1, hidden_dim = options["hidden_units_critic"], 
                              num_layers = options["num_layers_critic"], 
                              mid_activation=activation_dict[options["last_activation_critic"]],
                              last_activation=activation_dict[options["last_activation_critic"]])

        # initialize the actor and critics weights using orthogonal initialization
        self.initialize_weights()

        if options["normalize"] is True:
            if running_mean_std is None:
                self.running_mean_std = RunningMeanStd(shape=state_dim)
            else:
                self.running_mean_std = running_mean_std

        # ML Agent naming conventions for importing into Unity
        # ------------------------------------------------------------------------------------------

        self.m_size = 0
        self.export_memory_size = self.m_size

        self.MODEL_EXPORT_VERSION = 3
        self.version_number = torch.nn.Parameter(
            torch.Tensor([self.MODEL_EXPORT_VERSION]), requires_grad=False
        )
        self.is_continuous_int_deprecated = torch.nn.Parameter(
            torch.Tensor([int(self.action_spec.is_continuous())]), requires_grad=False
        )
        self.continuous_act_size_vector = torch.nn.Parameter(
            torch.Tensor([int(self.action_spec.continuous_size)]), requires_grad=False
        )
        self.discrete_act_size_vector = torch.nn.Parameter(
            torch.Tensor([self.action_spec.discrete_branches]), requires_grad=False
        )
        self.act_size_vector_deprecated = torch.nn.Parameter(
            torch.Tensor(
                [
                    self.action_spec.continuous_size
                    + sum(self.action_spec.discrete_branches)
                ]
            ),
            requires_grad=False,
        )
        
        self.memory_size_vector = torch.nn.Parameter(
            torch.Tensor([int(0)]), requires_grad=False
        )
        # --------------------------------------------------------------------------------------
        
    def initialize_weights(self):
        print("Initialize Actor Weights")
        layer_number = 0
        for i, layer_info in enumerate(self.actor.model.named_parameters()):
            name, param = layer_info

            if 'weight' in name:
                if layer_number == self.options["num_layers_actor"]:
                    torch.nn.init.orthogonal_(param, gain=np.sqrt(0.01))
                else:
                    torch.nn.init.orthogonal_(param, gain=np.sqrt(2))

            elif 'bias' in name:
                layer_number += 1
                torch.nn.init.zeros_(param)

        print("Initialize Critic Weights")
        # initialize critic weights
        layer_number = 0
        for i,layer_info in enumerate(self.critic.model.named_parameters()):
            name, param = layer_info

            if 'weight' in name:
                if layer_number == self.options["num_layers_critic"]:
                    torch.nn.init.orthogonal_(param, gain=np.sqrt(0.01))
                else:
                    torch.nn.init.orthogonal_(param, gain=np.sqrt(2))

            elif 'bias' in name:
                layer_number += 1
                torch.nn.init.zeros_(param)
    
    def set_action_std(self, new_action_std):

        if self.has_continuous_action_space:
            self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(self.device)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling ActorCritic::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def forward(self, input, dummy_mask=None, dummy_memory=None):
        """
        Forward specifically built for ONNX export for ml-agent 
        """
        if isinstance(input, list):
            input = input[0]
        continuous_action, _ , _ = self.act(input)

        export_out = [self.version_number, self.memory_size_vector, continuous_action, self.continuous_act_size_vector]
        return tuple(export_out)
    
    def act(self, state):
        """
        Sample an action from an observation. 
        Generate a distribution using the output of actor as mean, and self.action_var has variance
        :params state: observations passed to the actor to generate the distribution and thus action
        :params update_norm: bool used to determine if the running mean is to be updated
        :returns action: [batch_size, action_dim] tensor of actions sampled from distribution
        :returns action_logprob: [batch_size]
        """

        if self.options["normalize"] is True:
            self.running_mean_std.update(state)
            state = (state - self.running_mean_std.mean)/self.running_mean_std.var

        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            cov_mat = torch.diag(self.action_var)#.unsqueeze(dim=0)
            # dist = torch.distributions.MultivariateNormal(action_mean, cov_mat)
            dist = GaussianDistInstance(action_mean, cov_mat)
            
        else:
            action_probs = self.actor(state)
            # dist = torch.distributions.Categorical(action_probs)
            dist = CategoricalDistInstance(action_probs)

        action = dist.sample()
        action_logprob = dist.log_prob(action)
        state_values = self.critic(state)
        
        return action.detach(), action_logprob.detach(), state_values.detach()

    def evaluate(self, state, action):
        """
        Evaluate the entropy of each observation, and the log probs of each 
        action using the most recent policy. 

        :params state: tensor - observations to generate the distribution
        :params action: tensor - actions from which the logprobs are calculated
        :returns action_logprobs: log probability of the input action 
        :returns state_values: values at given states
        :returns dist_entropy: returns the entroyp of the distribution
        """

        if self.options["normalize"] is True:
            state = (state - self.running_mean_std.mean)/self.running_mean_std.var

        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            
            # action_var = self.action_var.expand_as(action_mean)
            # cov_mat = torch.diag_embed(action_var).to(self.device)
            cov_mat = torch.diag(self.action_var)
            dist = GaussianDistInstance(action_mean, cov_mat)
            # dist = torch.distributions.MultivariateNormal(action_mean, cov_mat)
            
            # For Single Action Environments.
            if self.action_dim == 1:
                action = action.reshape(-1, self.action_dim)

        else:
            action_probs = self.actor(state)
            # dist = torch.distributions.Categorical(action_probs)
            dist = CategoricalDistInstance(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)
        
        return action_logprobs, state_values, dist_entropy


class PPO:
    def __init__(self, behaviour_spec, options):
        """
        Proximal Policy Optimization class 
        Gathers Trajectories, and updates the PPO optimizer. (Might consider a new structure).
        The Class uses many options that can be found in the options.json file
        
        :params behaviour_spec: information about the environment it lies in
        :params options: dictionary holding parameters about the optimizer and policy
        """

        self.options = options
        self.device = torch.device(default_device())
        self.has_continuous_action_space = options["continuous_space"]

        if self.has_continuous_action_space:
            self.action_std = options["action_std"]

        self.gamma = options["gamma"]
        self.lmbda = options["lambda"]
        self.eps_clip = options["eps_clip"]
        self.K_epochs = options["K_epochs"]
        
        # Initialize a running mean std that will be shared between the actorcritics
        behaviour_name = list(behaviour_spec)[0]
        running_mean_std = None #RunningMeanStd(shape=behaviour_spec[behaviour_name].observation_specs[0].shape[0])

        self.policy : ActorCritic = ActorCritic(behaviour_spec, options, running_mean_std=running_mean_std).to(self.device)
        self.optimizer = torch.optim.Adam([
                        {'params': self.policy.actor.parameters(), 'lr': options["lr_actor"], 'weight_decay':0.0005},
                        {'params': self.policy.critic.parameters(), 'lr': options["lr_critic"]}
                    ])
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=options["scheduler_step"], gamma=options["scheduler_gamma"])
            
        # keep track of the previous policy to generate the ratio
        self.policy_old : ActorCritic = ActorCritic(behaviour_spec, options, running_mean_std=running_mean_std).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        # buffer of observations
        self.buffer = RolloutBuffer()

        self.MseLoss = torch.nn.MSELoss()

        self.cumulated_training_steps = 0
        self.last_decay_action_check = 0

    def set_action_std(self, new_action_std):
        
        if self.has_continuous_action_space:
            self.action_std = new_action_std
            self.policy.set_action_std(new_action_std)
            self.policy_old.set_action_std(new_action_std)
        
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling PPO::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def decay_action_std(self, action_std_decay_rate, min_action_std):
        
        """
        Apply the decay rate to the action distribution standard deviation.
        :params action_std_decay_rate: amount by which the std is reduced
        :params min_action_std: minimum possible std  
        """

        if self.has_continuous_action_space:
            self.action_std = self.action_std - action_std_decay_rate
            self.action_std = round(self.action_std, 4)
            if (self.action_std <= min_action_std):
                self.action_std = min_action_std
                print("setting actor output action_std to min_action_std : ", self.action_std)
            else:
                print("setting actor output action_std to : ", self.action_std)
            self.set_action_std(self.action_std)

        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling PPO::decay_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def batch_select_action(self, state):
        
        if self.has_continuous_action_space:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(self.device)
                action, logprobs, values = self.policy.act(state)

            return action, logprobs, values
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(self.device)
                action, logprobs, values = self.policy.act(state)

            return action, logprobs, values

    def batch_update(self):

        buffer_length = len(self.buffer)
        
        num_batches = (buffer_length // self.options["batch_size"]) 
        batch_indices = list(range(num_batches))
        cumul_entropy = cumul_loss_policy = cumul_loss_value = cumul_values = cumul_returns = 0

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            random.shuffle(batch_indices)

            for ind in batch_indices:

                batch_buffer = self.buffer[batch_indices[ind]*self.options["batch_size"] : (batch_indices[ind]+1)*self.options["batch_size"]]
                
                # convert list to tensor
                old_states = torch.squeeze(torch.stack(batch_buffer.states, dim=0)).detach().to(self.device)
                old_actions = torch.squeeze(torch.stack(batch_buffer.actions, dim=0)).detach().to(self.device)
                old_logprobs = torch.squeeze(torch.stack(batch_buffer.logprobs, dim=0)).detach().to(self.device)
                old_state_values = torch.squeeze(torch.stack(batch_buffer.values, dim=0)).detach().to(self.device)

                # Evaluating old actions and values
                logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
                # _, old_state_values, _ = self.policy_old.evaluate(old_states, old_actions)

                # match state_values tensor dimensions with rewards tensor
                state_values = torch.squeeze(state_values)
                # old_state_values = torch.squeeze(old_state_values)

                if self.options["advantage"] == "montecarlo":
                    advantages, returns = self.montecarlo_return(state_values, batch_buffer)
                elif self.options["advantage"] == "gae":
                    advantages, returns = self.GAE_Return(state_values, batch_buffer)
                else:
                    raise Exception('[PPO] Advantage method not implemented')

                # Finding the ratio (pi_theta / pi_theta__old)
                ratios = torch.exp(logprobs - old_logprobs.detach())

                # Finding Surrogate Loss
                 
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

                # value_loss1 = self.MseLoss(state_values, returns)
                # value_loss2 = torch.maximum(torch.minimum(state_values, old_state_values + self.eps_clip), old_state_values-self.eps_clip)
                # value_loss3 = 0.5*torch.maximum(value_loss1, self.MseLoss(value_loss2, returns))
                clipped_value = old_state_values + (state_values - old_state_values).clamp(min=-self.eps_clip,
                                                                              max=self.eps_clip)
                value_loss1 = torch.max((state_values - returns) ** 2, (clipped_value - returns) ** 2)
                value_loss3 = 0.5 * value_loss1.mean()

                # value_loss3 = 0.5*self.MseLoss(state_values, returns)

                # final loss of clipped objective PPO
                loss = -torch.min(surr1, surr2) + value_loss3 - 0.01*dist_entropy

                # take gradient step
                self.optimizer.zero_grad()
                loss.mean().backward()
                torch.nn.utils.clip_grad_norm_(list(self.policy.actor.parameters()) + list(self.policy.critic.parameters()), max_norm=0.5)
                self.optimizer.step()
                
                cumul_returns += torch.mean(returns).detach().item()
                cumul_values +=  torch.mean(state_values).detach().item()
                cumul_entropy += torch.mean(dist_entropy).detach().item()
                cumul_loss_policy += torch.mean(torch.min(surr1, surr2)).detach().item()
                cumul_loss_value += value_loss3.detach().item()

        # Log Information
        N = len(batch_indices*self.K_epochs)
        log.writer.add_scalar("Policy/Returns", cumul_returns/N, self.cumulated_training_steps)
        log.writer.add_scalar("Policy/Value Estimate", cumul_values/N, self.cumulated_training_steps)
        log.writer.add_scalar("Policy/Entropy", cumul_entropy/N, self.cumulated_training_steps)
        log.writer.add_scalar("Losses/Policy", cumul_loss_policy/N, self.cumulated_training_steps)
        log.writer.add_scalar("Losses/Value",  cumul_loss_value/N, self.cumulated_training_steps)
        log.writer.add_scalar("Policy/Learning Rate", self.scheduler.get_last_lr()[0], self.cumulated_training_steps)
        log.writer.add_scalar("Policy/Action Std", self.action_std, self.cumulated_training_steps)
        cumul_entropy = cumul_loss_policy = cumul_loss_value = cumul_values = cumul_returns = 0
                    
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()
        self.scheduler.step()

    def GAE_Return(self, state_values, batch_buffer:RolloutBuffer):

        returns = torch.zeros(len(batch_buffer.rewards))
        gae = 0

        # Is_terminals indicates wether an episode is finished or not
        # inverting yields a mask
        masks = [not term for term in batch_buffer.is_terminals]
        values = torch.zeros((state_values.shape[0]+1))
        values[:state_values.shape[0]] = state_values

        for i in reversed(range(len(batch_buffer.rewards))):
            
            delta = batch_buffer.rewards[i] + self.gamma * values[i + 1] * masks[i] - values[i]
            gae = delta + self.gamma * self.lmbda * masks[i] * gae
            returns[i] = gae + values[i]

        # returns = torch.tensor(returns).float()
        # returns = (returns - returns.mean()) / (returns.std() + 1e-7)
        adv = returns - values[:-1]
        adv = adv - adv.mean() / (adv.std() + 1e-7)
        adv = torch.clip(adv, -self.options["clip_norm_adv"], self.options["clip_norm_adv"])

        return adv, returns

    def montecarlo_return(self, state_values, batch_buffer):
        """
        Calculate return using MonteCarlo Method
        """
        rewards = []
        discounted_reward = 0
        # reverse method
        for reward, is_terminal in zip(reversed(batch_buffer.rewards), reversed(batch_buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        # rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        advantages = rewards - state_values.detach()  
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)
        advantages = torch.clip(advantages, -self.options["clip_norm_adv"], self.options["clip_norm_adv"])
        return advantages, rewards

    def save(self, checkpoint_path):

        save_model_dict = {'action_std' : self.action_std,
                          'cumulated_training_steps' : self.cumulated_training_steps,
                          'policy_state_dict' : self.policy.state_dict(),
                          'old_policy_state_dict' : self.policy_old.state_dict(),
                          'optimizer_state_dict' : self.optimizer.state_dict(),
                          'scheduler' : self.scheduler.state_dict(),
                          'running_mean' : self.policy.running_mean_std.mean,
                          'running_std' : self.policy.running_mean_std.var}

        torch.save(save_model_dict, checkpoint_path)

    def load(self, checkpoint_path):
        
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        
        self.action_std = checkpoint['action_std']
        self.cumulated_training_steps = checkpoint['cumulated_training_steps']
        self.policy_old.load_state_dict(checkpoint['old_policy_state_dict'])
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.policy.running_mean_std.mean = checkpoint['running_mean']
        self.policy.running_mean_std.std = checkpoint['running_std']

        self.set_action_std(self.action_std)

        return self.cumulated_training_steps
        
        
       
if __name__ == "__main__":
    import numpy as np
    from mlagents.trainers.settings import TorchSettings
    from mlagents.torch_utils import set_torch_config

    set_torch_config(TorchSettings(device='cpu'))
    print("device :" ,default_device())

    np.random.seed(0)
    torch.manual_seed(0)

    obs_dim = 8
    action_dim = 2
    lr_actor = 0.0003   
    gamma = 0.99
    lr_critic = 0.001 
    K_epochs = 80
    eps_clip = 0.2
    
    has_continuous_action_space = True

    ppo_agent = PPO(obs_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space)
    
    buffer = RolloutBuffer()
    buffer.actions.extend(torch.rand(100,2))
    buffer.states.extend(torch.rand(100,8))
    buffer.rewards.extend(list(range(100)))
    buffer.logprobs.extend(torch.rand(100))
    buffer.is_terminals.extend(list(range(100)))

    batch_size = 25
    buffer_size = len(buffer)
    
    num_batches = (buffer_size // batch_size) 
    batch_indices = list(range(num_batches))
    random.shuffle(batch_indices)

    ppo_agent.batch_update(buffer, batch_size)