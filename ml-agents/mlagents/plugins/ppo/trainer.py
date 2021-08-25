import numpy as np
from mlagents.torch_utils import torch, default_device

from mlagents_envs.environment import ActionTuple

from mlagents.plugins.ppo.PPO import PPO
from mlagents.plugins.ppo.buffer import RolloutBuffer
from mlagents.plugins.ppo.network import Discriminator
from mlagents.plugins.dataset.dataset import UnityMotionDataset, SkeletonInfo

import mlagents.plugins.utils.logger as log
from mlagents.plugins.utils.memory import memReport, cpuStats
from mlagents.plugins.bvh_utils.visualize import skeletons_plot, motion_animation
import mlagents.plugins.bvh_utils.lafan_utils as utils

class AMPTainer():
    def __init__(self, env, options, motion_path,
                 side_channels, model_path, load_path=None):
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



        # INIT PPO AGENT
        self.ppo_agent = PPO(env.behavior_specs, options)

        # LOAD DATA FROM MOTION CAPTURE DATASET
        # Setup LaFan dataset and skeleton information
        self.adv_dataset = UnityMotionDataset(motion_path, side_channels["skeleton"], options=options)
        self.skdata : SkeletonInfo = self.adv_dataset.skdata
        self.scale = 100

        # INIT DISCRIMINATOR
        self.discrim = Discriminator(options, self.ppo_agent.policy.running_mean_std, self.adv_dataset)
        self.features_per_joints=13


        # INIT TRAJECTORY RETRIEVAL
        # dictionary of buffers which maps an agent to a trajectory
        self.agent_buffers = {}
        self.buffer = RolloutBuffer()
        self.buffer_size = options["buffer_size"]
        self.cumulated_training_steps = 0
        self.cumulated_reward = 0
        self.last_decay_action_check = 0

        self.discrim_buffer = DiscrimBuffer(self.options)

        self.cumul_style_reward = 0
        self.cumul_goal_reward = 0

        # INIT LOADED MODEL (if loadpath is not None)
        if load_path is not None:
            self.load(load_path)

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
            
            if(epoch % self.options["save_freq"] == 0):
                self.save(epoch)
                # self.ppo_agent.save(self.model_path + "LafanLine_ep_{}.pt".format(epoch))
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
                obs = torch.tensor(obs, requires_grad=False).float().to(self.device)
                if self.options["rotation_repr"] == "6D": obs = self.quaternion_state_to_6D(obs)

                action, act_logprob, state_values = self.ppo_agent.batch_select_action(obs) #[num_agent, act_dim]

                # append the information corresponding to each agent
                self.agent_buffers[agent_ind].rewards.append(terminal_steps[agent_ind].reward)
                self.agent_buffers[agent_ind].is_terminals.append(True)
                self.agent_buffers[agent_ind].states.append(obs[index,:])
                self.agent_buffers[agent_ind].actions.append(action[index,:])
                self.agent_buffers[agent_ind].logprobs.append(act_logprob[index])
                self.agent_buffers[agent_ind].values.append(state_values[index])
            
            # get next action for all observations 
            obs = decision_steps.obs[0] # [num_agent, obs_dim]
            obs = torch.tensor(obs, requires_grad=False).float().to(self.device)
            if self.options["rotation_repr"] == "6D": obs = self.quaternion_state_to_6D(obs)

            action, act_logprob, state_values = self.ppo_agent.batch_select_action(obs) #[num_agent, act_dim]
            
            # retriave information from agents who require a decision
            for agent_ind in decision_steps:
                # create a RolloutBuffer for any new agent 
                if agent_ind not in self.agent_buffers.keys():
                    self.agent_buffers[agent_ind] = RolloutBuffer()
                    self.agent_buffers[agent_ind].clear()

                index = decision_steps.agent_id_to_index[agent_ind]
                

                # append the information corresponding to each agent
                self.agent_buffers[agent_ind].rewards.append(decision_steps[agent_ind].reward)
                self.agent_buffers[agent_ind].is_terminals.append(False)
                self.agent_buffers[agent_ind].states.append(obs[index,:])
                self.agent_buffers[agent_ind].actions.append(action[index,:])
                self.agent_buffers[agent_ind].logprobs.append(act_logprob[index])
                self.agent_buffers[agent_ind].values.append(state_values[index])

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
        Compute the style reward which is obtained from running the discriminator using trajectory data. 
        The resulting reward is added to the goal reward to be later used in the policy update. 
        """
        # first add the full trajectory to the discriminator buffer:

        # if we do, reduce the batch size to the len of the buffer
        buffer_batch = self.ppo_agent.buffer.states[:self.options["buffer_size"]]
        buffer_batch = torch.vstack(buffer_batch)
        buffer_batch_curr = buffer_batch[:-1]
        buffer_batch_next = buffer_batch[1:]
        buffer_batch = (buffer_batch_curr, buffer_batch_next)
        discrim_input  = self.buffer_to_discrim(buffer_batch)
        self.discrim_buffer.add(discrim_input)

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
            buffer_batch_curr = buffer_batch[:-1]
            buffer_batch_next = buffer_batch[1:]
            buffer_batch = (buffer_batch_curr, buffer_batch_next)

            # get the discriminator input from buffer
            discrim_input  = self.buffer_to_discrim(buffer_batch)
            # discrim_input = (discrim_input - self.discrim_buffer.mean)/self.discrim_buffer.var

            # call the discriminator G_reward
            style_reward = self.discrim.G_reward(discrim_input)
            style_reward = torch.squeeze(style_reward)

            # add the reward to the buffer.reward (remember to add the factors)
            goal_reward = torch.tensor(self.ppo_agent.buffer.rewards[start_ind:end_ind-1]).float()
            reward =  self.options["goal_factor"] * goal_reward + self.options["style_factor"] * style_reward

            reward = reward * torch.tensor([not f for f in self.ppo_agent.buffer.is_terminals[start_ind:end_ind-1]])

            # reward = torch.where(torch.logical_and(goal_reward < -0.99, goal_reward > -1.01), -1., reward.double()).float()
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
        Update the discriminator by getting data from the adversarial dataset and trajectory buffer. 
        """
        batch_size = self.options["batch_size_discrim"] 
        window_size = self.options["window_size"]
        n_batches = self.options["buffer_size"] // batch_size

        # for loop or not?
        for i in range(self.options["K_discrim"]):

            # sample batch size from the motion dataset
            #rand_ind = np.random.randint(80, 1300 - (batch_size+1)) # TODO: remove hardcoded values
            # adv_motion = self.adv_dataset[rand_ind:rand_ind+batch_size+1]

            rand_ind = np.random.randint(0,275, batch_size)
            rand_ind[rand_ind < 175] += 75
            rand_ind[rand_ind > 175] += 175

            adv_motion_curr = self.adv_dataset[rand_ind]
            adv_motion_next = self.adv_dataset[rand_ind+1]
            adv_motion = (adv_motion_curr, adv_motion_next)

            # extract the motion information, make sure its local, and compute the forward kinematics
            # MAKE SURE TO APPLY THE ROTATION OFFSET
            real_input = self.adversarial_to_discrim(adv_motion, batch_size)

            # sample batch size from the buffer 
            # extract the information from it and concatenate in a single vector 
            # get the frames in batch size + 1 to account for the last next frame
            # ind = i % (n_batches-1)
            # start_ind  =  ind*batch_size
            # end_ind = (ind+1)*(batch_size) + 1
            # rand_ind = np.random.randint(0, len(self.ppo_agent.buffer.states)-1, batch_size)

            # buffer_batch_curr = torch.vstack([self.ppo_agent.buffer.states[f] for f in rand_ind])
            # buffer_batch_next = torch.vstack([self.ppo_agent.buffer.states[f] for f in (rand_ind+1)])
            # buffer_batch = (buffer_batch_curr, buffer_batch_next)
            # fake_input = self.buffer_to_discrim(buffer_batch)

            # add to buffer, then randomly sample from buffer
            # self.discrim_buffer.add(fake_input)
            rand_ind = np.random.randint(0, self.discrim_buffer.max_ind, batch_size)
            fake_input = self.discrim_buffer[rand_ind]
            # fake_input = (fake_input - self.discrim_buffer.mean)/self.discrim_buffer.var

            # pass in the data to the discriminator so that it can update itself
            self.discrim.optimize(real_input.float(), fake_input.float())

        log.writer.add_scalar("Losses/Discriminator", self.discrim.cumul_d_loss/self.options["K_discrim"], self.cumulated_training_steps)
        log.writer.add_scalar("Losses/Discriminator_Real", self.discrim.cumul_d_real_loss/self.options["K_discrim"], self.cumulated_training_steps)
        log.writer.add_scalar("Losses/Discriminator_Fake", self.discrim.cumul_d_fake_loss/self.options["K_discrim"], self.cumulated_training_steps)
        log.writer.add_scalar("Losses/Grad_Penalty", self.discrim.cumul_grad_penalty/self.options["K_discrim"], self.cumulated_training_steps)
        self.discrim.cumul_d_loss = 0
        self.discrim.cumul_d_real_loss = 0
        self.discrim.cumul_d_fake_loss = 0
        self.discrim.cumul_grad_penalty = 0

    def buffer_to_discrim(self, batch):
        """
        Extract rotation, position and velocity of current and next state and 
        shape it to as an input for the discriminator

        :params batch: [batch_size+1, obs_dim] tensor of features collected in the trajectory
        :returns discrim_input: [batch_size, discrim_input_size] tensor used in the discriminator as "fake data" 
        """
        
        batch_size = batch[0].shape[0]#self.options["batch_size_discrim"]
        feature_stacks = []

        if self.options["rotation_repr"] == "6D":
            features_per_joints = self.features_per_joints + 2
        else:
            features_per_joints = self.features_per_joints

        for i in range(2):
            # extract the observations needed for the discriminator 
            # extract only the observation corresponding to the joints
            joint_features = batch[i][:,:-6]
            joint_features = joint_features.reshape(batch_size,-1, features_per_joints)

            velocity = joint_features[:,:,:3].clone()
            # angular_vel = joint_features[:,:,3:6].clone()
            positions = joint_features[:,:,6:9].clone()
            rotations = joint_features[:,:,9:].clone()

            local_positions = positions[:] - positions[:,0:1,:]

            if self.options["rotation_repr"] == "quater":
                temp = rotations[:,:,3].clone()
                temp2 = rotations[:,:,:3].clone()
                rotations[:,:,1:] = temp2
                rotations[:,:,0] = temp
                rotations[:,0,:] = torch.tensor([1.,0.,0.,0.]).float()

            ee_pos = local_positions[:,self.skdata.ee_id,:].reshape(batch_size,-1)

            # velocity and angular velocity calculation from positions and rotations, this is okay because it is sequential
            # velocity = utils.get_velocity(local_positions, self.skdata.frametime)

            # stack them vertically in one big vector 
            # feature_stack = torch.cat((local_positions.reshape(batch_size+1,-1), rotations.reshape(batch_size+1,-1)), dim=1)
            # feature_stack = torch.cat((local_positions.reshape(batch_size+1,-1), rotations.reshape(batch_size+1,-1), velocity.reshape(batch_size+1,-1)), dim=1)
            feature_stacks.append(torch.cat((ee_pos, rotations.reshape(batch_size,-1), velocity.reshape(batch_size,-1)), dim=1))

        # stack the current state and next state in a single buffer
        discrim_input = torch.cat((feature_stacks[0], feature_stacks[1]), dim=1)

        return discrim_input

    def adversarial_to_discrim(self, batch, batch_size):
        """
        format the adversarial data to be usable by the discriminator.

        :params batch: tuple containing the current frames batch and next frame batch. 
        Those batches are themselves tuples containing a local positions and rotation tensor.  
        :params batch_size: int specifying the size of the discriminator batch 
        :returns discrim_input: [batch_size, discrim_input_size] tensor of "real data" 
        """
        curr_batch, next_batch = batch
        
        # velocity, position, and rotation
        curr_ee_pos = curr_batch[0][:,self.skdata.ee_id,:].reshape(batch_size, -1)
        next_ee_pos = next_batch[0][:,self.skdata.ee_id,:].reshape(batch_size, -1)

        if self.options["rotation_repr"] == "quater":
            curr_rotation = curr_batch[1].reshape(batch_size,-1)
            next_rotation = next_batch[1].reshape(batch_size,-1)
        elif self.options["rotation_repr"] == "6D":

            curr_rotation = curr_batch[1].reshape(batch_size, -1)
            next_rotation = next_batch[1].reshape(batch_size, -1)


        # curr_feature_stack = torch.cat((curr_batch[0].reshape(batch_size, -1), curr_batch[1].reshape(batch_size,-1), curr_batch[2].reshape(batch_size,-1)), dim=1)
        # next_feature_stack = torch.cat((next_batch[0].reshape(batch_size, -1), next_batch[1].reshape(batch_size,-1), next_batch[2].reshape(batch_size,-1)), dim=1)
        curr_feature_stack = torch.cat((curr_ee_pos, curr_rotation.reshape(batch_size,-1), curr_batch[2].reshape(batch_size,-1)), dim=1)
        next_feature_stack = torch.cat((next_ee_pos, next_rotation.reshape(batch_size,-1), next_batch[2].reshape(batch_size,-1)), dim=1)
     

        # stack the current state and next state in a single buffer
        discrim_input = torch.cat((curr_feature_stack, next_feature_stack), dim=1)

        return discrim_input

    def quaternion_state_to_6D(self, states):
        """
        state is a 2D tensor, [n_agents, state_dim]
        return a 2D tensor of size [n_agents, state_dim + 6D dim]
        """
        # extract the observations needed for the discriminator 
        # extract only the observation corresponding to the joints
        
        if states.shape[0] == 0:
            return torch.empty((0,states.shape[1]+self.skdata.num_joints*2))

        joint_features = states[:,:-6]
        joint_features = joint_features.reshape(states.shape[0],-1, self.features_per_joints)

        u_velocity = joint_features[:,:,:3].clone()
        angular_vel = joint_features[:,:,3:6].clone()
        positions = joint_features[:,:,6:9].clone()
        rotations = joint_features[:,:,9:].clone()

        temp = rotations[:,:,3].clone()
        temp2 = rotations[:,:,:3].clone()
        rotations[:,:,1:] = temp2
        rotations[:,:,0] = temp
        rotations[:,0,:] = torch.tensor([1.,0.,0.,0.]).float()

        # compute the rotation matrix, rotation_mat is of size [n_agents, n_joints, 3,3]
        rotation_mat = utils.compute_rotation_matrix_from_quaternion(rotations)
        rotation6D = torch.cat((rotation_mat[:,:,0,:], rotation_mat[:,:,1,:]), dim=2)

        new_states = torch.cat((u_velocity , angular_vel, positions, rotation6D), dim=2)
        new_states = new_states.reshape(states.shape[0], -1)
        new_states = torch.cat((new_states, states[:,-6:]), dim=1)

        return new_states


    def save(self, epoch):
        self.ppo_agent.save(self.model_path + "LafanLine_ep_{}_AC.tar".format(epoch))
        self.discrim.save(self.model_path + "LafanLine_ep_{}_discrim.tar".format(epoch), self.discrim_buffer)

    def load(self, load_path):
        cumul = self.ppo_agent.load(load_path + '_AC.tar')
        buffer = self.discrim.load(load_path + '_discrim.tar')

        self.cumulated_training_steps = cumul
        self.discrim_buffer.load(buffer)


class DiscrimBuffer():
    def __init__(self, options):
        self.max_size = options['buffer_max_size']
        self.n_features = options['input_dim_discrim']
        self.index = 0
        self.max_ind = 0
        self.buffer = torch.zeros((self.max_size, self.n_features))
        self.mean = torch.zeros((self.n_features))
        self.var = torch.ones((self.n_features))

    def add(self, input):
        batch_size = input.shape[0]
        
        start_ind = 0
        end_ind = 0

        if self.index + batch_size > self.max_size - 1 :
            ind = (self.max_size - 1) - (self.index)
            self.buffer[self.index : self.max_size-1] = input[:ind]
            input = input[ind:]
            start_ind = 0
            end_ind = batch_size - ind 
        else:
            start_ind = self.index
            end_ind = self.index+batch_size

        self.buffer[start_ind : end_ind] = input
        self.index = end_ind


        self.max_ind += batch_size
        if self.max_ind > self.max_size-1:
            self.max_ind = self.max_size-1

        self.update_mean_var()

    def update_mean_var(self):
        self.mean = torch.mean(self.buffer[:self.max_ind], dim=0)
        self.var = torch.var(self.buffer[:self.max_ind], dim=0)
        self.var = torch.where( self.var.double() < 1e-5, 1., self.var.double()).float()
        self.var = self.var ** (1/2)


    def __getitem__(self,index):
        return self.buffer[index]

    def load(self, buffer):
        self.buffer = buffer
        self.index = 0
        self.max_ind = self.max_size

# limits = [[-1,1],[-1,1],[-1,1]]
# skeletons_plot([local_positions[0].cpu().detach()], [self.skdata.edges], ['g'], limits=limits, return_plot=False)
# skeletons_plot([local_positions0[0].cpu().detach(),pos_from_vel[0].cpu().detach(),local_positions1[0].cpu().detach()],
#                [self.skdata.edges,self.skdata.edges,self.skdata.edges], 
#                ['b','r','g'], limits=limits, return_plot=False)

# offsets = self.skdata.offsets.clone()
# offsets = offsets.reshape(1,22,3)
# offsets = offsets.repeat(rotations.shape[0],1,1)

# _, pos_from_rot = utils.quat_fk(rotations, offsets, self.skdata.parents)
# skeletons_plot([pos_from_rot[0].cpu().detach()], [self.skdata.edges], ['g'], limits=limits, return_plot=False)
# skeletons_plot([local_positions[0].cpu().detach()], [self.skdata.edges,self.skdata.edges], ['g', 'b'], limits=limits, return_plot=False)

# anim = motion_animation([pos_from_rot[0].cpu().detach(), local_positions[0].cpu().detach()], [self.skdata.edges, self.skdata.edges], ['g', 'b'], limits)
# HTML(anim.to_jshtml())


# # get velocity by doing central difference
# for i in range(positions.shape[0]-2):
#     # print(((positions[i+2] - positions[i])/(frametime*2)).shape)
#     velocity[i+1, ...] = (positions[i+2] - positions[i])/(self.skdata.frametime*2)

# # boundary cases 
# velocity[0] = (positions[1] - positions[0])/self.skdata.frametime
# velocity[-1] = (positions[-1] - positions[-2])/self.skdata.frametime

# velocity += u_velocity[:,0:1,:]
# velocity = get_batch_velo2(local_positions, self.skdata.frametime)


# pos_from_vel = local_positions0 + velocity0 * 0.05
# pos_from_vel = utils.get_global_position_from_velocity(torch.zeros((50,22,3)), velocity, 0.01*5, local_positions)

# adv_offsets = self.skdata.offsets.clone()/self.scale
# rotation_offset = torch.tensor([  0., 0., 1., 0. ])#Quaternions.from_euler(np.array([0,0,0]), 'xyz').qs)
# adv_data = get_pos_info_from_raw(batch, self.skdata, adv_offsets, self.options, norm_rot=False, rotation_offset=rotation_offset)
# adv_pos_global, adv_pos_local, adv_hips_rot, adv_hips_velo, adv_vel, adv_rot_local = adv_data

# adv_pos_local = adv_pos_local.reshape(-1, adv_pos_local.shape[-2], adv_pos_local.shape[-1])
# adv_pos_local = self.adv_dataset.reconstruct_from_windows(adv_pos_local)

# adv_pos_local = adv_pos_local.reshape(adv_pos_local.shape[0], -1)
# adv_pos_local = adv_pos_local[:batch_size+1]

# adv_vel[:,:,1:,:] += adv_vel[:,:,0:1,:] 

# instead of reshaping it that way, we want to disentangle the windows
# adv_vel = self.adv_dataset.reconstruct_from_windows(adv_vel)

# adv_vel = adv_vel.reshape(adv_vel.shape[0], -1)
# adv_vel = adv_vel[:batch_size+1]

# adv_vel = adv_vel[:batch_size+1]
# adv_rot_local = adv_rot_local[:batch_size+1]