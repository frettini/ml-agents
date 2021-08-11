from mlagents.torch_utils import torch, default_device
import random

from mlagents.plugins.dataset.dataset import  UnityMotionDataset, SkeletonInfo
from mlagents.plugins.skeleton_aware_op.autoencoder_temporal import StaticEncoder, AE, Encoder, Decoder
from mlagents.plugins.skeleton_aware_op.discriminator_temporal import Discriminator
from mlagents.plugins.skeleton_aware_op.loss import calc_chain_velo_loss, calc_ee_loss
from mlagents.plugins.bvh_utils import lafan_utils as utils
from mlagents.plugins.bvh_utils import BVH_mod as BVH

from mlagents.plugins.bvh_utils.visualize import skeletons_plot, motion_animation
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import art3d
import numpy as np


class Sk_Trainer():

    def __init__(self, options, input_motion_path, adv_motion_path):
        
        self.options = options
        self.device = default_device()

        # Load Input and Adversarial Dataset
        self.input_dataset = UnityMotionDataset(input_motion_path)
        self.adv_dataset = UnityMotionDataset(adv_motion_path)
        self.skdata_input = self.input_dataset.skdata
        self.skdata_adv = self.adv_dataset.skdata
        self.input_limits = None
        self.adv_limits = None

        # Initialize the main functions
        self.discriminator = Discriminator(self.skdata_adv.edges, options).to(self.device)

        # Retargetter initialization
        self.encoder_sim = Encoder(self.skdata_input.edges, options)
        # use the initialization of encoder_data to get the correct pooling lists
        self.encoder_data = Encoder(self.skdata_adv.edges, options) 
        self.decoder_data = Decoder(self.encoder_data, options)

        self.static_encoder_sim = StaticEncoder(self.skdata_input.edges, options).to(self.device)
        self.static_encoder_data = StaticEncoder(self.skdata_adv.edges, options).to(self.device)

        # Optimizers concatenate parameters used for pose generation
        gen_parameters = list(self.retargetter.encoder_sim.parameters()) + list(self.retargetter.decoder_data.parameters()) \
            + list(self.retargetter.static_encoder_sim.parameters()) + list(self.retargetter.static_encoder_data.parameters())

        self.gen_optimizer = torch.optim.Adam(gen_parameters, lr=self.options["sk_g_lr"], betas=(0.9, 0.999))
        self.discrim_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=self.options["sk_d_lr"], betas=(0.9, 0.999))
        
        # scheduler = torch.optim.lr_scheduler.StepLR(gen_optimizer, 10, 0.9)
        # cyclic_decay = lambda x : max(0.0001,0.4*(np.cos(lr_freq*x)+1.2)*(lr_decay**x))
        # scheduler = torch.optim.lr_scheduler.LambdaLR(gen_optimizer, lr_lambda=cyclic_decay)

        self.criterion_gan = torch.nn.MSELoss()
        self.criterion_ee = torch.nn.MSELoss()
        self.criterion_root_velocity = torch.nn.MSELoss()
        self.criterion_velo = torch.nn.MSELoss()

        self.G_cumul = self.G_loss_adv_cumul = self.G_loss_ee_cumul = self.G_loss_velo_cumul = self.G_loss_glob_cumul = 0
        self.D_cumul = self.D_real_cumul = self.D_fake_cumul = 0

    def train(self):
        
        for ep in range(self.options["sk_K_epochs"]):
            print('Epoch : ', ep)

            # do the job of a dataloader (shuffle and load only once each data point)
            # by hand because our dataset class cannot comply with the dataloader format.
            # get length of effective dataset
            if self.input_limits is not None:
                length = self.input_limits[1] - self.input_limits[0]
            else:
                length = len(self.input_dataset)
                
            # get the number of windows in the dataset (with overlap thus //2)
            n_windows = length // (self.options["window_size"]//2)
            # generate a list of indices and shuffle them
            wind_indices = np.array(list(range(n_windows)), dtype=np.int32) * (self.options["window_size"]//2)
            random.shuffle(wind_indices)
            # determine how many loops can be done in batches with the available indices list
            n_loop = len(wind_indices) // self.options["sk_batch_size"]

            for i in range(n_loop):
                
                # get start and end indices of each batch, then load data 
                start_inds = wind_indices[i*self.options["sk_batch_size"], (i+1)*self.options["sk_batch_size"]] 
                end_inds = start_inds + self.options["window_size"]

                # reshape motion to [batch_size, window_size, n_joints*channel_size]
                motion = self.input_dataset.to_skdata(start_inds, end_inds)
                curr_batch_size = motion.shape[0]

                # get a frame from the "real" data 
                randinds = np.random.randint(0,len(self.adv_dataset) - self.options["window_size"], curr_batch_size)
                real_motion = self.adv_dataset.to_skdata(randinds, start_inds + self.options["window_size"])

                # perform retargeting
                output_motion = self.retarget(motion)

                # extract global position, root velocity,  
                # TODO: Changed get_pos_info_from_raw definition to take in offsets as a 2d data [num_joints, 3]
                fake_data = utils.get_pos_info_from_raw(output_motion, self.skdata_input, self.options, norm_rot=True)
                fake_pos, fake_pos_local, fake_glob, fake_glob_velo, fake_vel = fake_data

                real_data = utils.get_pos_info_from_raw(real_motion, self.skdata_adv, self.options, norm_rot=True)
                real_pos, real_pos_local, real_glob, real_glob_velo, real_vel = real_data

                motion_data = utils.get_pos_info_from_raw(motion, self.skdata_input, self.options, norm_rot=False )
                input_pos, input_pos_local, input_glob, input_glob_velo, input_vel = motion_data

                # the GAN update code is mostly taken from :
                #https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html#loss-functions-and-optimizers
                # update discriminator 
                self.update_discriminator(real_pos, fake_pos)
                
                # UPDATE GENERATOR NETWORK
                # auto_encoder.zero_grad()
                self.update_generator(motion_data, fake_data)

                if(i % 10 == 0):
                    print("Batch : {}, Discriminator loss: {}, Generator loss : {}".format(i, self.D_cumul, self.G_cumul))

    def retarget(self, motion):
        """
        Retarget the motion from window frames from one window to another 
        """

        # motion should have shape [batch_size, window_size, n_joints*channel_size]

        # first get the offsets from the static encoder 
        # deep_offsets = static_encoder(torch.tensor(anim.offsets[np.newaxis, :,:]).float())
        deep_offsets_sim = self.static_encoder_sim(self.skdata_input.offsets.reshape(1, self.skdata_input.offsets.shape[0], -1))
        deep_offsets_output = self.static_encoder_data(self.skdata_output.offsets.reshape(1, self.skdata_output.offsets.shape[0], -1))

        self.input_dataset.normalize(motion)
        latent = self.encoder_sim(motion, deep_offsets_sim)
        res = self.decoder_data(latent, deep_offsets_output)
        res = self.adv_dataset.denormalize(res)

        return res

    def update_generator(self, input_data, output_data):
        """
        Compute the full generator loss (adv and other) and its gradients and apply them 
        to the autoencoder and static encoders.
        """
        
        # TODO: pass all the required information for updating the generator

        # UPDATE GENERATOR NETWORK
        # auto_encoder.zero_grad()
        self.encoder_sim.zero_grad()
        self.decoder_data.zero_grad()
        self.static_encoder_sim.zero_grad()
        self.static_encoder_data.zero_grad()

        input_pos, input_pos_local, input_root_rotation, input_glob_velo, input_vel = input_data
        output_pos, output_pos_local, output_root_rotation, output_glob_velo, output_vel = output_data

        curr_batch_size = input_pos.shape[0]
        
        # Adversial Loss, use real labels to maximize log(D(G(x))) instead of log(1-D(G(x)))
        loss_adv = self.discriminator.G_loss(output_pos)

        # Gobal, Velocity and End Effector Losses
        loss_rot = self.criterion_global_rotation(output_root_rotation, input_root_rotation)

        # Get the loss that tries to match the total velocity of every corresponding chain        
        # loss_chain_velo = calc_chain_velo_loss(real_vel, fake_vel, chain_indices, 
        #                                        criterion_velo, normalize_velo = False)
        loss_glob_velo = self.criterion_velo(output_glob_velo/self.skdata_adv.height, input_glob_velo/self.skdata_input.height)
        loss_ee = self.calc_ee_loss(input_vel, input_pos_local, output_vel, output_pos_local, self.criterion_ee)
        

        # combine all the losses together
        loss_gen =  loss_adv*self.options["sk_adv_factor"] + loss_ee*self.options["sk_ee_factor"] + \
        loss_rot*self.options["sk_rot_factor"] + loss_glob_velo*self.options["sk_glob_velo_factor"]
        
        # update gradient
        loss_gen.backward()
        self.gen_optimizer.step()
        
        # record losses 
        self.G_cumul += loss_gen
        
        self.G_loss_adv_cumul += loss_adv
        self.G_loss_ee_cumul  += loss_ee
        self.G_loss_velo_cumul+= loss_glob_velo
        self.G_loss_glob_cumul+= loss_rot

    def update_discriminator(self, real_pos, fake_pos):
        """
        Compute the discriminator loss and its gradient, and apply it to the discriminator
        """
        
        curr_batch_size = real_pos.shape[0]
        label = torch.full((curr_batch_size,), self.real_label, dtype=torch.float, device=default_device())
        
        self.discriminator.zero_grad()

        # forward pass
        output = self.discriminator.forward(real_pos.float()).view(-1)
        loss_real = self.criterion_gan(output,label)
        loss_real.backward()
        
        # do the same with the generated position
        label.fill_(self.fake_label)
        output = self.forward(fake_pos.float().detach()).view(-1)
        loss_fake = self.criterion_gan(output, label)
        loss_fake.backward()

        loss_discrim = loss_real + loss_fake

        self.discrim_optimizer.step()

        # record losses 
        self.D_cumul += loss_discrim
        self.D_real_cumul += loss_real
        self.D_fake_cumul  += loss_fake
    
    def calc_ee_loss(self, real_vel, real_pos, fake_vel, fake_pos):
        """
        Calculate the end effector loss using the fake velocity and position (local or global)
        """
        # [batch, frame, n_end_effector, 6]
        ee_false = self.skdata_input.ee_id[self.skdata_input.ee_id != 0]
        ee_real = self.skdata_adv.ee_id[self.skdata_adv.ee_id != 0]

        # print(fake_pos[:,:,0:1,:].shape)
        fake_pos_loc = fake_pos - fake_pos[:,:,0:1,:]
        real_pos_loc = real_pos - real_pos[:,:,0:1,:]

        fake_ee = torch.cat((fake_vel[:,:,ee_false,:], fake_pos_loc[:,:,ee_false,:]), dim=2)
        real_ee = torch.cat((real_vel[:,:,ee_real,:], real_pos_loc[:,:,ee_real,:]), dim=2)

        real_length = self.skdata_adv.ee_length.reshape(1,1,self.skdata_adv.ee_length.shape[0],1)
        fake_length = self.skdata_input.ee_length.reshape(1,1,self.skdata_input.ee_length.shape[0],1)

        # print(torch.repeat_interleave(real_length,2, dim=2))
        real_length = torch.repeat_interleave(real_length,2, dim=2)
        fake_length = torch.repeat_interleave(fake_length,2, dim=2)


        # loss_ee_velo = criterion_ee(fake_joint_vel[:,:,ee_id,:], real_joint_vel[:,:,ee_id,:])
        loss_ee_velo = self.criterion_ee(fake_ee/fake_length, real_ee/real_length)

        return loss_ee_velo