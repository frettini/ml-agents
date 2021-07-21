import numpy as np
import random

from mlagents.torch_utils import torch, default_device
from mlagents.plugins.ppo.distribtution import GaussianDistInstance, CategoricalDistInstance
from mlagents.plugins.ppo.network import MLPNet
from mlagents.plugins.ppo.buffer import RolloutBuffer
from mlagents_envs.environment import ActionTuple

import mlagents.plugins.utils.logger as log


class ActorCritic(torch.nn.Module):
    def __init__(self, behavior_specs, has_continuous_action_space, action_std_init):
        
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

        self.has_continuous_action_space = has_continuous_action_space

        behavior_name = list(behavior_specs)[0]
        self.behavior_spec = behavior_specs[behavior_name]
        self.action_spec = behavior_specs[behavior_name].action_spec

        state_dim = self.behavior_spec.observation_specs[0].shape[0]
        action_dim = len(self.action_spec)

        self.device = default_device()
        print(self.device)
        if has_continuous_action_space:
            self.action_dim = action_dim
            self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(self.device)

        # actor
        if has_continuous_action_space :

            self.actor = MLPNet(state_dim, action_dim)
        else:
            self.actor = MLPNet(state_dim, action_dim,last_activation = torch.nn.Softmax(dim=-1))

        self.critic = MLPNet(state_dim,1,last_activation=torch.nn.Identity())

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
        continuous_action, _ = self.act(input)

        export_out = [self.version_number, self.memory_size_vector, continuous_action, self.continuous_act_size_vector]
        return tuple(export_out)
    
    def act(self, state):
        """
        Sample an action from an observation. 
        Generate a distribution using the output of actor as mean, and self.action_var has variance
        :params state: observations passed to the actor to generate the distribution and thus action
        """
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
        
        return action.detach(), action_logprob.detach()

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
    def __init__(self, options, behaviour_spec):
        """
        Proximal Policy Optimization class 
        
        :params state_dim:
        :params action_dim:
        :params lr_actor:
        :params lr_critic:
        :params gamma:
        :params K_epochs:
        :params eps_clip:
        :params has_continuous_action_space:
        :params action_std_init:
        """

        self.options = options
        self.device = torch.device(default_device())
        self.has_continuous_action_space = options["continuous_space"]

        if self.has_continuous_action_space:
            self.action_std = options["action_std"]

        self.gamma = options["gamma"]
        self.eps_clip = options["eps_clip"]
        self.K_epochs = options["K_epochs"]
        
        self.policy = ActorCritic(behaviour_spec, self.has_continuous_action_space, options["action_std"]).to(self.device)
        self.optimizer = torch.optim.Adam([
                        {'params': self.policy.actor.parameters(), 'lr': options["lr_actor"]},
                        {'params': self.policy.critic.parameters(), 'lr': options["lr_critic"]}
                    ])
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=options["scheduler_gamma"])
            
        # keep track of the previous policy to generate the ratio
        self.policy_old = ActorCritic(behaviour_spec, self.has_continuous_action_space, options["action_std"]).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        # dictionary of buffers which maps an agent to a trajectory
        self.agent_buffers = {}
        self.buffer = RolloutBuffer()
        self.buffer_size = options["buffer_size"]
        self.cumulated_training_steps = 0

        self.MseLoss = torch.nn.MSELoss()

        self.cumulated_reward = 0
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
                action, logprobs = self.policy_old.act(state)

            return action, logprobs
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(self.device)
                action, logprobs = self.policy_old.act(state)

            return action, logprobs

    def get_trajectory(self, env):
        """
        Computes the trajectory of the environment for every agent in the scene. 
        Stores the trajectory in RolloutBuffers.
        """
        cumulated_steps = 0
        step = 0
        behavior_name = list(env.behavior_specs)[0]

        while cumulated_steps < self.buffer_size:
            decision_steps, terminal_steps = env.get_steps(behavior_name)

            for agent_ind in terminal_steps:

                # create a RolloutBuffer for any new agent 
                if agent_ind not in self.agent_buffers.keys():
                    self.agent_buffers[agent_ind] = RolloutBuffer()
                    self.agent_buffers[agent_ind].clear()
        
                index = terminal_steps.agent_id_to_index[agent_ind]

                # collect for the terminal step too
                obs = terminal_steps.obs[0] # [num_agent, obs_dim]
                action, act_logprob = self.batch_select_action(obs) #[num_agent, act_dim]

                obs = torch.tensor(obs, requires_grad=False).float().to(self.device)

                # append the information corresponding to each agent
                self.agent_buffers[agent_ind].rewards.append(terminal_steps[agent_ind].reward)
                self.agent_buffers[agent_ind].is_terminals.append(True)
                self.agent_buffers[agent_ind].states.append(obs[index,:])
                self.agent_buffers[agent_ind].actions.append(action[index,:])
                self.agent_buffers[agent_ind].logprobs.append(act_logprob[index])

                self.cumulated_reward +=  terminal_steps[agent_ind].reward
            
            # get next action for all observations 
            obs = decision_steps.obs[0] # [num_agent, obs_dim]
            action, act_logprob = self.batch_select_action(obs) #[num_agent, act_dim]
            

            for agent_ind in decision_steps:
                # create a RolloutBuffer for any new agent 
                if agent_ind not in self.agent_buffers.keys():
                    self.agent_buffers[agent_ind] = RolloutBuffer()
                    self.agent_buffers[agent_ind].clear()

                index = decision_steps.agent_id_to_index[agent_ind]
                obs = torch.tensor(obs, requires_grad=False).float().to(self.device)

                # append the information corresponding to each agent
                self.agent_buffers[agent_ind].rewards.append(decision_steps[agent_ind].reward)
                self.agent_buffers[agent_ind].is_terminals.append(False)
                self.agent_buffers[agent_ind].states.append(obs[index,:])
                self.agent_buffers[agent_ind].actions.append(action[index,:])
                self.agent_buffers[agent_ind].logprobs.append(act_logprob[index])

                self.cumulated_reward +=  decision_steps[agent_ind].reward

            action_tuple = ActionTuple()
            action_tuple.add_continuous(action.detach().cpu().numpy())

            env.set_actions(behavior_name, action_tuple)
            env.step()
            
            if step % self.options["print_freq"] == 0:
                if len(decision_steps) > 0:
                    reward = np.mean(decision_steps.reward)
                elif len(terminal_steps) > 0:
                    reward = np.mean(terminal_steps.reward)
                else: reward = 'None'

                print("step: {} \t cumulated steps: {} \t reward: {} ".format(step, cumulated_steps, np.mean(reward)))

            curr_decay_action_action = (self.cumulated_training_steps+cumulated_steps) % self.options["action_std_decay_freq"]
            # if continuous action space; then decay action std of ouput action distribution
            if self.has_continuous_action_space and curr_decay_action_action < self.last_decay_action_check:
                self.decay_action_std(self.options["action_std_decay_rate"], self.options["min_action_std"])
            
            self.last_decay_action_check = curr_decay_action_action

            cumulated_steps += len(decision_steps)
            step += 1

        self.cumulated_training_steps += self.options["buffer_size"]

        # take all our buffers and concatenated them 
        self.buffer = RolloutBuffer()
        for key in self.agent_buffers.keys():
            self.buffer.extend(self.agent_buffers[key])
            self.agent_buffers[key].clear()

    def update(self):
        """
        Update the policy using clipped gradient policy
        """
        # Monte Carlo estimate of returns
        # Rewards to go calculation
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(buffer.rewards), reversed(buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(buffer.states, dim=0)).detach().to(self.device)
        old_actions = torch.squeeze(torch.stack(buffer.actions, dim=0)).detach().to(self.device)
        old_logprobs = torch.squeeze(torch.stack(buffer.logprobs, dim=0)).detach().to(self.device)

        
        # Optimize policy for K epochs
        for _ in range(self.K_epochs):

            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)
            
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            advantages = rewards - state_values.detach()   
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards) - 0.01*dist_entropy
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        buffer.clear()

    def batch_update(self):

        buffer_length = len(self.buffer)
        
        num_batches = (buffer_length // self.options["batch_size"]) 
        batch_indices = list(range(num_batches))
        cumul_entropy = cumul_loss_policy = cumul_loss_value = cumul_values = 0

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):

            random.shuffle(batch_indices)

            for ind in batch_indices:

                batch_buffer = self.buffer[batch_indices[ind]*self.options["batch_size"] : (batch_indices[ind]+1)*self.options["batch_size"]]
            
                # Monte Carlo estimate of returns
                # Rewards to go calculation
                rewards = []
                discounted_reward = 0
                for reward, is_terminal in zip(reversed(batch_buffer.rewards), reversed(batch_buffer.is_terminals)):
                    if is_terminal:
                        discounted_reward = 0
                    discounted_reward = reward + (self.gamma * discounted_reward)
                    rewards.insert(0, discounted_reward)
                    
                # Normalizing the rewards
                rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
                rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

                # convert list to tensor
                old_states = torch.squeeze(torch.stack(batch_buffer.states, dim=0)).detach().to(self.device)
                old_actions = torch.squeeze(torch.stack(batch_buffer.actions, dim=0)).detach().to(self.device)
                old_logprobs = torch.squeeze(torch.stack(batch_buffer.logprobs, dim=0)).detach().to(self.device)

                # Evaluating old actions and values
                logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

                # match state_values tensor dimensions with rewards tensor
                state_values = torch.squeeze(state_values)
                
                # Finding the ratio (pi_theta / pi_theta__old)
                ratios = torch.exp(logprobs - old_logprobs.detach())

                # Finding Surrogate Loss
                advantages = rewards - state_values.detach()   
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

                # final loss of clipped objective PPO
                loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards) - 0.01*dist_entropy

                # take gradient step
                self.optimizer.zero_grad()
                loss.mean().backward()
                self.optimizer.step()
                
                cumul_values +=  torch.mean(state_values).item()
                cumul_entropy += torch.mean(dist_entropy).item()
                cumul_loss_policy += torch.mean(torch.min(surr1, surr2)).item()
                cumul_loss_value += 0.5*self.MseLoss(state_values, rewards).item()

        # Log Information
        N = len(batch_indices*self.K_epochs)
        log.writer.add_scalar("Policy/Value Estimate", cumul_values/N, self.cumulated_training_steps)
        log.writer.add_scalar("Policy/Entropy", cumul_entropy/N, self.cumulated_training_steps)
        log.writer.add_scalar("Losses/Policy", cumul_loss_policy/N, self.cumulated_training_steps)
        log.writer.add_scalar("Losses/Value",  cumul_loss_value/N, self.cumulated_training_steps)
        log.writer.add_scalar("Policy/Learning Rate", self.scheduler.get_last_lr()[0], self.cumulated_training_steps)
        log.writer.add_scalar("Policy/Action Std", self.action_std, self.cumulated_training_steps)
        cumul_entropy = cumul_loss_policy = cumul_loss_value = cumul_values= 0
                    
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()
        self.scheduler.step()

    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        
        
       
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