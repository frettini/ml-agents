from mlagents_envs import side_channel
import numpy as np
from mlagents.torch_utils import torch, default_device

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.environment import ActionTuple

from mlagents.plugins.ppo.PPO import PPO
from mlagents.plugins.ppo.buffer import RolloutBuffer
from mlagents.plugins.ppo.network import Discriminator
from mlagents.plugins.dataset.dataset import TemporalMotionData, SkeletonInfo
from mlagents.plugins.bvh_utils.lafan_utils import get_pos_info_from_raw, get_batch_velo2

import mlagents.plugins.utils.logger as log
from mlagents.plugins.bvh_utils.visualize import skeletons_plot, motion_animation

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
            raise("Side Channel should at least contain the Skeleton and Engine side channels")
        side_channels["engine"].set_configuration_parameters(width=350, height=300)

        # extract initial skeleton information from side_channel  
        # TODO : remove hardcode 22 joints      
        init_info = side_channels["skeleton"].get_info()
        self.unity_init_rotations = torch.tensor(init_info[:22*4]).float().reshape(22,4)
        self.unity_init_positions = torch.tensor(init_info[22*4:]).float().reshape(22,3)

        # TODO : ensure that the frame rate of simulation is the same as the one from the adversarial dataset

        # INIT PPO AGENT
        self.ppo_agent = PPO(env.behavior_specs, options)

        # LOAD DATA FROM MOTION CAPTURE DATASET
        # Setup LaFan dataset and skeleton information
        self.adv_dataset = TemporalMotionData(motion_path, recalculate_mean_var=True, normalize_data=True, xyz='xyz', rot_order=None)
        self.skdata : SkeletonInfo = self.adv_dataset.skdata
        self.scale = 100

        # INIT DISCRIMINATOR
        self.discrim = Discriminator(options)
        self.features_per_joints=13

        # INIT TRAJECTORY RETRIEVAL
        # dictionary of buffers which maps an agent to a trajectory
        self.agent_buffers = {}
        self.buffer = RolloutBuffer()
        self.buffer_size = options["buffer_size"]
        self.cumulated_training_steps = 0
        self.cumulated_reward = 0
        self.last_decay_action_check = 0

        self.cumul_style_reward = 0
        self.cumul_goal_reward = 0

    def train(self):
        """
        Main training loop for the reinforcement learning policy. 
        Responsible for gathering trajectories, calculating adversarial loss, and
        and updating the policy. 
        """

        # MAIN AMP LOOP
        epoch = 0
        while(self.ppo_agent.cumulated_training_steps < self.options["max_training_steps"]):

            print("Ep {} \t Training step {} : ".format(epoch, self.ppo_agent.cumulated_training_steps))

            # step through the simulation and gather the trajectory
            self.get_trajectory()

            # get the style reward from the discriminator
            self.compute_style_reward()

            # update the discriminator 
            self.discrim_update()

            # update the policy using the collected buffer
            self.ppo_agent.batch_update()

            print("Ep {} finished with cumulated reward {} \t average reward {}".format(epoch, self.cumulated_reward, self.cumulated_reward/self.options["buffer_size"]))
            # log.writer.add_scalar("Reward/Cumulated_Reward",self.cumulated_reward, self.cumulated_training_steps)
            # log.writer.add_scalar("Reward/Average_Reward",self.cumulated_reward/self.options["buffer_size"], self.cumulated_training_steps)
            
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
        action_freq = self.options["action_std_decay_freq"]
        continuous_space  = self.options["continuous_space"]

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

            curr_decay_action_action = (self.cumulated_training_steps+cumulated_steps) % action_freq
            # if continuous action space; then decay action std of ouput action distribution
            if continuous_space and curr_decay_action_action < self.last_decay_action_check:
                self.ppo_agent.decay_action_std(self.options["action_std_decay_rate"], self.options["min_action_std"])
            
            self.last_decay_action_check = curr_decay_action_action

            cumulated_steps += len(decision_steps)
            step += 1

        self.cumulated_training_steps += self.options["buffer_size"]
        self.ppo_agent.cumulated_training_steps = self.cumulated_training_steps
        log.log_step = self.cumulated_training_steps

        # take all our buffers and concatenated them into a large buffer on which update will be performed
        self.ppo_agent.buffer = RolloutBuffer()
        for key in self.agent_buffers.keys():
            self.ppo_agent.buffer.extend(self.agent_buffers[key])
            self.agent_buffers[key].clear()

    def compute_style_reward(self):
        """

        """
        # get pair of observations from the buffer and concatenate them 
        batch_size = self.options["batch_size_discrim"]
        num_iter = len(self.ppo_agent.buffer) // batch_size

        # iterate through the entire buffer in batches
        for i in range(num_iter):
            start_ind  =  i*batch_size
            end_ind = (i+1)*(batch_size) + 1

            # check that we dont overflow
            if end_ind >= self.options["buffer_size"]:

                # if we do, reduce the batch size to the len of the buffer
                buffer_batch = self.ppo_agent.buffer.states[start_ind:end_ind-1]
                # also take care of the last frame (double the last observation)
                buffer_batch.append(self.ppo_agent.buffer.states[end_ind-2])
            else:
                buffer_batch = self.ppo_agent.buffer.states[start_ind:end_ind]
            
            buffer_batch = torch.vstack(buffer_batch)

            # get the discriminator input from buffer
            discrim_input  = self.buffer_to_discrim(buffer_batch)

            # call the discriminator G_reward
            style_reward = self.discrim(discrim_input)
            style_reward = torch.squeeze(style_reward)

            # add the reward to the buffer.reward (remember to add the factors)
            goal_reward = torch.tensor(self.ppo_agent.buffer.rewards[start_ind:end_ind-1]).float()
            reward =  self.options["goal_factor"] * goal_reward + self.options["style_factor"] * style_reward
            self.ppo_agent.buffer.rewards[start_ind:end_ind-1] = reward.cpu().detach().tolist()
            
            self.cumul_style_reward += torch.mean(style_reward).detach()
            self.cumul_goal_reward += torch.mean(goal_reward).detach()
            self.cumulated_reward += torch.mean(reward).detach()

        log.writer.add_scalar("Reward/Style_Reward", self.cumul_style_reward/num_iter, self.cumulated_training_steps)
        log.writer.add_scalar("Reward/Goal_Reward", self.cumul_goal_reward/num_iter, self.cumulated_training_steps)
        log.writer.add_scalar("Reward/Average_Reward", self.cumulated_reward/num_iter, self.cumulated_training_steps)
        log.writer.add_scalar("Reward/Cumulated_Reward", self.cumul_goal_reward, self.cumulated_training_steps)
        self.cumul_style_reward = 0
        self.cumul_goal_reward = 0

    def discrim_update(self):
        """
        
        """
        batch_size = self.options["batch_size_discrim"] 
        window_size = self.options["window_size"]
        n_batches = self.options["buffer_size"] // batch_size

        # for loop or not?
        for i in range(self.options["K_discrim"]):
            # sample batch size from the motion dataset
            n_wind = (batch_size+1)//(window_size//2) + 1 # +1 to account for next state
            # rand_ind = np.random.randint(0,len(self.adv_dataset) - n_wind)
            rand_ind = np.random.randint(12,162- n_wind) # TODO: remove hardcoded values
            adv_motion = self.adv_dataset[rand_ind:rand_ind+n_wind]


            # extract the motion information, make sure its local, and compute the forward kinematics
            # MAKE SURE TO APPLY THE ROTATION OFFSET
            real_input = self.adversarial_to_discrim(adv_motion, batch_size)

            # sample batch size from the buffer 
            # extract the information from it and concatenate in a single vector 
            # get the frames in batch size + 1 to account for the last next frame
            ind = i % (n_batches-1)
            start_ind  =  ind*batch_size
            end_ind = (ind+1)*(batch_size) + 1
            buffer_batch = self.ppo_agent.buffer.states[start_ind:end_ind]
            buffer_batch = torch.vstack(buffer_batch)
            fake_input = self.buffer_to_discrim(buffer_batch)

            # pass in the data to the discriminator so that it can update itself
            self.discrim.optimize(real_input.float(), fake_input.float())

        log.writer.add_scalar("Losses/Discriminator", self.discrim.cumul_d_loss/self.options["K_discrim"], self.cumulated_training_steps)
        log.writer.add_scalar("Losses/Grad_Penalty", self.discrim.cumul_grad_penalty/self.options["K_discrim"], self.cumulated_training_steps)
        self.discrim.cumul_d_loss = 0
        self.discrim.cumul_grad_penalty = 0

    def buffer_to_discrim(self, batch):
        """
        Extract rotation, position and velocity of current and next state and 
        shape it to as an input for the discriminator
        """
        
        batch_size = self.options["batch_size_discrim"]

        # extract the observations needed for the discriminator 
        # extract only the observation corresponding to the joints
        joint_features = batch[:,:-6]
        joint_features = joint_features.reshape(batch_size+1,-1, self.features_per_joints)

        u_velocity = joint_features[:,:,:3]
        angular_vel = joint_features[:,:,3:6]
        positions = joint_features[:,:,6:9]
        rotations = joint_features[:,:,9:]

        local_positions = positions[:] - positions[:,0:1,:]
        
        velocity = torch.zeros_like(positions)
    
        # get velocity by doing central difference
        for i in range(positions.shape[0]-2):
            # print(((positions[i+2] - positions[i])/(frametime*2)).shape)
            velocity[i+1, ...] = (positions[i+2] - positions[i])/(self.skdata.frametime*2)

        # boundary cases 
        velocity[0] = (positions[1] - positions[0])/self.skdata.frametime
        velocity[-1] = (positions[-1] - positions[-2])/self.skdata.frametime

        velocity += u_velocity[:,0:1,:]
        #velocity = get_batch_velo2(local_positions, self.skdata.frametime)
        
        # skeletons_plot([local_positions.cpu().detach()], [self.skdata.edges], ['g'], limits=limits, return_plot=False)

        # limits = [[-1,1],[-1,1],[-1,1]]
        # pos_from_vel = torch.zeros_like(local_positions)
        # pos_from_vel[0] = local_positions[0]
        # for i in range(1,pos_from_vel.shape[0]):
        #     if torch.abs(velocity.sum(dim=[1,2])[i]) < 2:
        #         pos_from_vel[i,:] = local_positions[i]
        #     else:
        #         local_vel= velocity -velocity[:,0:1]
        #         pos_from_vel[i,:] = pos_from_vel[i-1] + local_vel[i] * self.skdata.frametime
        # anim = motion_animation([pos_from_vel.cpu().detach(), local_positions.cpu().detach()], [self.skdata.edges, self.skdata.edges], ['g', 'b'], limits)

        # HTML(anim.to_jshtml())

        # stack them vertically in one big vector 
        feature_stack = torch.cat((velocity.reshape(batch_size+1,-1),local_positions.reshape(batch_size+1,-1)), dim=1)
        # stack the current state and next state in a single buffer
        discrim_input = torch.cat((feature_stack[:-1,:], feature_stack[1:,:]), dim=1)

        return discrim_input

    def adversarial_to_discrim(self, batch, batch_size):

        batch = self.adv_dataset.denormalize(batch)
        
        adv_offsets = self.skdata.offsets.clone()/self.scale
        rotation_offset = torch.tensor([  0., 0., 1., 0. ])#Quaternions.from_euler(np.array([0,0,0]), 'xyz').qs)
        adv_data = get_pos_info_from_raw(batch, self.skdata, adv_offsets, self.options, norm_rot=False, rotation_offset=rotation_offset)
        adv_pos_global, adv_pos_local, adv_hips_rot, adv_hips_velo, adv_vel, adv_rot_local = adv_data

        # adv_pos_local = adv_pos_local.reshape(-1, adv_pos_local.shape[-2], adv_pos_local.shape[-1])
        adv_pos_local = self.adv_dataset.reconstruct_from_windows(adv_pos_local)

        adv_pos_local = adv_pos_local.reshape(adv_pos_local.shape[0], -1)
        adv_pos_local = adv_pos_local[:batch_size+1]

        adv_vel[:,:,1:,:] += adv_vel[:,:,0:1,:] 
        
        # instead of reshaping it that way, we want to disentangle the windows
        adv_vel = self.adv_dataset.reconstruct_from_windows(adv_vel)

        adv_vel = adv_vel.reshape(adv_vel.shape[0], -1)
        adv_vel = adv_vel[:batch_size+1]

        # adv_vel = adv_vel[:batch_size+1]
        # adv_rot_local = adv_rot_local[:batch_size+1]

        # velocity, position, and rotation
        feature_stack = torch.cat((adv_vel, adv_pos_local), dim=1)
        # stack the current state and next state in a single buffer
        discrim_input = torch.cat((feature_stack[:-1,:], feature_stack[1:,:]), dim=1)

        return discrim_input
