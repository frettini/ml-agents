import numpy as np
from mlagents.torch_utils import torch, default_device

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.environment import ActionTuple

from mlagents.plugins.ppo.PPO import PPO
from mlagents.plugins.ppo.buffer import RolloutBuffer
from mlagents.plugins.ppo.network import Discriminator
from mlagents.plugins.dataset.dataset import TemporalMotionData, SkeletonInfo

import mlagents.plugins.utils.logger as log


class AMPTainer():
    def __init__(self, env, options, motion_path,
                 side_channels, model_path):
        """
        The Adversarial Motion Prior Trainer uses motion data to drive the learning
        of a RL policy. 
        :params side_channels: dictionary of side channels
        """
        self.options = options
        self.model_path = model_path
        self.device = default_device()
        self.env = env
        
        if side_channels is None:
            raise("Side Channel should at least contain the Skeleton side channel")

        # extract initial skeleton information from side_channel        
        init_info = side_channels["skeleton"].get_info()
        self.unity_init_rotations = torch.tensor(init_info[:22*4]).float().reshape(22,4)
        self.unity_init_positions = torch.tensor(init_info[22*4:]).float().reshape(22,3)

        # TODO : ensure that the frame rate of simulation is the same as the one from the adversarial dataset

        # INIT PPO AGENT
        self.ppo_agent = PPO(env.behavior_specs, options)

        # LOAD DATA FROM MOTION CAPTURE DATASET
        # Setup LaFan dataset and skeleton information
        lafan_dataset = TemporalMotionData(motion_path, recalculate_mean_var=True, normalize_data=True, xyz='zxy', rot_order=None)
        self.skdata = lafan_dataset.skdata

        # INIT DISCRIMINATOR
        self.discrim = Discriminator(options)

        # INIT TRAJECTORY RETRIEVAL
        # dictionary of buffers which maps an agent to a trajectory
        self.agent_buffers = {}
        self.buffer = RolloutBuffer()
        self.buffer_size = options["buffer_size"]
        self.cumulated_training_steps = 0
        self.cumulated_reward = 0
        self.last_decay_action_check = 0


        pass

    def train(self):
        """
        Main training loop for the reinforcement learning policy. 
        Responsible for gathering trajectories, calculating adversarial loss, and
        and updating the policy. 
        """

        # MAIN PPO LOOP
        epoch = 0
        while(self.ppo_agent.cumulated_training_steps < self.options["max_training_steps"]):

            print("Ep {} \t Training step {} : ".format(epoch, self.ppo_agent.cumulated_training_steps))

            # step through the simulation and gather the trajectory
            self.get_trajectory()

            # get the style reward from the discriminator
            

            # update the policy using the collected buffer
            self.ppo_agent.batch_update()

            print("Ep {} finished with cumulated reward {} \t average reward {}".format(epoch, self.cumulated_reward, self.cumulated_reward/self.options["buffer_size"]))
            log.writer.add_scalar("Train/Cumulated_Reward",self.cumulated_reward, self.cumulated_training_steps)
            log.writer.add_scalar("Train/Average_Reward",self.cumulated_reward/self.options["buffer_size"], self.cumulated_training_steps)
            
            self.cumulated_reward = 0

            epoch += 1
            
            if(epoch % 5 == 0):
                self.ppo_agent.save(self.model_path + "LafanLine_ep_{}.pt".format(epoch))
        pass

    def get_trajectory(self):
        """
        Computes the trajectory of the environment for every agent in the scene. 
        Stores the trajectory in RolloutBuffers.
        """
        cumulated_steps = 0
        step = 0
        behavior_name = list(self.env.behavior_specs)[0]

        while cumulated_steps < self.buffer_size:
            decision_steps, terminal_steps = self.env.get_steps(behavior_name)

            # retrieve information from agents in their terminal step
            for agent_ind in terminal_steps:
                # create a RolloutBuffer for any new agent 
                if agent_ind not in self.agent_buffers.keys():
                    self.agent_buffers[agent_ind] = RolloutBuffer()
                    self.agent_buffers[agent_ind].clear()
        
                index = terminal_steps.agent_id_to_index[agent_ind]

                # collect for the terminal step too
                obs = terminal_steps.obs[0] # [num_agent, obs_dim]
                action, act_logprob = self.ppo_agent.batch_select_action(obs) #[num_agent, act_dim]

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
            action, act_logprob = self.ppo_agent.batch_select_action(obs) #[num_agent, act_dim]
            
            # retriave information from agents who require a decision
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

            # generate action, pass it to the environment and step the simulation
            action_tuple = ActionTuple()
            action_tuple.add_continuous(action.detach().cpu().numpy())

            self.env.set_actions(behavior_name, action_tuple)
            self.env.step()
            
            if step % self.options["print_freq"] == 0:
                num = 0
                reward  = 0
                if len(decision_steps) > 0:
                    num += len(decision_steps)
                    reward += np.sum(decision_steps.reward)
                elif len(terminal_steps) > 0:
                    num += len(terminal_steps)
                    reward += np.sum(terminal_steps.reward)
                else: reward = 'None'

                print("step: {} \t cumulated steps: {} \t reward: {} ".format(step, cumulated_steps, reward/num))

            curr_decay_action_action = (self.cumulated_training_steps+cumulated_steps) % self.options["action_std_decay_freq"]
            # if continuous action space; then decay action std of ouput action distribution
            if self.options["continuous_space"] and curr_decay_action_action < self.last_decay_action_check:
                self.ppo_agent.decay_action_std(self.options["action_std_decay_rate"], self.options["min_action_std"])
            
            self.last_decay_action_check = curr_decay_action_action

            cumulated_steps += len(decision_steps)
            step += 1

        self.cumulated_training_steps += self.options["buffer_size"]
        self.ppo_agent.cumulated_training_steps = self.cumulated_training_steps

        # take all our buffers and concatenated them into a large buffer on which update will be performed
        self.ppo_agent.buffer = RolloutBuffer()
        for key in self.agent_buffers.keys():
            self.ppo_agent.buffer.extend(self.agent_buffers[key])
            self.agent_buffers[key].clear()

    def compute_style_reward(self):
        # get pair of observations from the buffer and concatenate them 

        # get pair of observations from the dataset and concatenate them

        # construct batches of the same size 

        # run otpimize and get the 

        
        pass

        