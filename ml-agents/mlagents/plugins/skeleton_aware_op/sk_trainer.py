from mlagents.torch_utils import torch, default_device
import random

from mlagents.plugins.dataset.dataset import  UnityMotionDataset
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

class Retargetter():
    def __init__(self, options, skdata_input, skdata_output, discriminator:Discriminator):
        
        self.options = options
        self.skdata_input = skdata_input
        self.skdata_output = skdata_output

        self.encoder_sim = Encoder(skdata_input.edges)
        # use the initialization of encoder_data to get the correct pooling lists
        # its forward is not used
        self.encoder_data = Encoder(skdata_output.edges) 
        self.decoder_data = Decoder(self.encoder_data)

        self.static_encoder_sim = StaticEncoder(skdata_input.edges).to(device)
        self.static_encoder_data = StaticEncoder(skdata_output.edges).to(device)

        self.discriminator = discriminator

        self.criterion_gan = torch.nn.MSELoss()
        self.criterion_ee = torch.nn.MSELoss()
        self.criterion_root_velocity = torch.nn.MSELoss()
        self.criterion_velo = torch.nn.MSELoss()

    def retarget(self, motion):
        """
        Retarget the motion from window frames from one window to another 
        """

        # motion should have shape [batch_size, window_size, n_joints*channel_size]
        


        # first get the offsets from the static encoder 
        # deep_offsets = static_encoder(torch.tensor(anim.offsets[np.newaxis, :,:]).float())
        deep_offsets_sim = self.static_encoder_sim(self.skdata_input.offsets.reshape(1, self.skdata_input.offsets.shape[0], -1))
        deep_offsets_output = self.static_encoder_data(self.skdata_output.offsets.reshape(1, self.skdata_output.offsets.shape[0], -1))

        latent = self.encoder_sim(motion, deep_offsets_sim)
        res = self.decoder_data(latent, deep_offsets_output)

        return res

    def update(self, input_motion, output_motion):
        
        # TODO: pass all the required information for updating the generator

        # UPDATE GENERATOR NETWORK
        # auto_encoder.zero_grad()
        self.encoder_sim.zero_grad()
        self.decoder_data.zero_grad()
        self.static_encoder_sim.zero_grad()
        self.static_encoder_data.zero_grad()

        curr_batch_size = input_motion.shape[0]
        
        # Adversial Loss, use real labels to maximize log(D(G(x))) instead of log(1-D(G(x)))
        loss_adv = self.discriminator.G_loss(output_motion)

        # Gobal, Velocity and End Effector Losses
        loss_glob = self.criterion_global_rotation(output_root_rotation, input_root_rotation)

        # Get the loss that tries to match the total velocity of every corresponding chain        
        # loss_chain_velo = calc_chain_velo_loss(real_vel, fake_vel, chain_indices, 
        #                                        criterion_velo, normalize_velo = False)
        loss_chain_velo = criterion_velo(fake_glob_velo/skdata_data.height, input_glob_velo/skdata_sim.height)
        loss_ee = calc_ee_loss(input_vel, input_pos_local, fake_vel, fake_pos_local,
                            criterion_ee, skdata_sim, skdata_data)
        

        # combine all the losses together
        loss_gen =  loss_adv*mult_adv + loss_ee*mult_ee + \
        loss_glob*mult_glob + loss_chain_velo*mult_chain_velo
        
        # update gradient
        loss_gen.backward()
        gen_optimizer.step()
        
        # record losses 
        D_cumul += loss_discrim
        G_cumul += loss_gen
        
        G_loss_adv_cumul += loss_adv
        G_loss_ee_cumul  += loss_ee
        G_loss_velo_cumul+= loss_chain_velo
        G_loss_glob_cumul+= loss_glob




class Sk_Trainer():

    def __init__(self, options, input_motion_path, adv_motion_path):
        
        self.options = options
        
        # Load Input and Adversarial Dataset
        self.input_dataset = UnityMotionDataset(input_motion_path)
        self.adv_dataset = UnityMotionDataset(adv_motion_path)
        self.skdata_input = self.input_dataset.skdata
        self.skdata_adv = self.adv_dataset.skdata
        self.input_limits = None
        self.adv_limits = None

        # Initialize the main functions
        self.retargetter = Retargetter()
        self.discriminator = Discriminator(self.skdata_adv.edges).to(device)



        # Optimizers
        # concatenate parameters used for pose generation
        gen_parameters = list(self.retargetter.encoder_sim.parameters()) + list(self.retargetter.decoder_data.parameters()) \
            + list(self.retargetter.static_encoder_sim.parameters()) + list(self.retargetter.static_encoder_data.parameters())

        self.gen_optimizer = torch.optim.Adam(gen_parameters, lr=self.options["sk_g_lr"], betas=(0.9, 0.999))
        self.discrim_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=self.options["sk_d_lr"], betas=(0.9, 0.999))

        

        # scheduler = torch.optim.lr_scheduler.StepLR(gen_optimizer, 10, 0.9)
        # cyclic_decay = lambda x : max(0.0001,0.4*(np.cos(lr_freq*x)+1.2)*(lr_decay**x))
        # scheduler = torch.optim.lr_scheduler.LambdaLR(gen_optimizer, lr_lambda=cyclic_decay)


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
                self.input_dataset.normalize(motion)
                output_motion = self.retargetter.retarget(motion)
                res = self.adv_dataset.denormalize(res)

                # extract global position, root velocity,  
                # TODO: Changed get_pos_info_from_raw definition to take in offsets as a 2d data [num_joints, 3]
                fake_data = utils.get_pos_info_from_raw(res, self.skdata_input, self.options, norm_rot=True)
                fake_pos, fake_pos_local, fake_glob, fake_glob_velo, fake_vel = fake_data

                real_data = utils.get_pos_info_from_raw(real_motion, self.skdata_adv, self.options, norm_rot=True)
                real_pos, real_pos_local, real_glob, real_glob_velo, real_vel = real_data

                # motion_data = utils.get_pos_info_from_raw(motion, skdata_sim, offsets_sim, options, norm_rot=False )
                # input_pos, input_pos_local, input_glob, input_glob_velo, input_vel = motion_data

                # the GAN update code is mostly taken from :
                #https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html#loss-functions-and-optimizers
                

                # UPDATE DISCRIMINATOR :
                # first do real data batch then fake data
                
                # start adverserial training after a set number of epochs
                loss_real, loss_fake, loss_discrim = self.discriminator.D_loss(real_pos_local, fake_pos_local, False)
                self.discrim_optimizer.step()
                
                # UPDATE GENERATOR NETWORK
                # auto_encoder.zero_grad()
                encoder_sim.zero_grad()
                decoder_data.zero_grad()
                static_encoder_sim.zero_grad()
                static_encoder_data.zero_grad()
                
                # Adversial Loss 
                loss_adv = 0
                # use real labels to maximize log(D(G(x))) instead of log(1-D(G(x)))
                label = torch.full((curr_batch_size,), real_label, dtype=torch.float, device=device)
                label.fill_(real_label)
                # use updated discriminator
                output = discriminator(fake_pos_local.float()).view(-1)
                # Calculate the generator's loss and gradients
                loss_adv = criterion_gan(output,label)
                    
                # Gobal, Velocity and End Effector Losses
                loss_glob = criterion_glob(fake_glob, input_glob)

                # Get the loss that tries to match the total velocity of every corresponding chain        
                # loss_chain_velo = calc_chain_velo_loss(real_vel, fake_vel, chain_indices, 
                #                                        criterion_velo, normalize_velo = False)
                loss_chain_velo = criterion_velo(fake_glob_velo/skdata_data.height, input_glob_velo/skdata_sim.height)
                loss_ee = calc_ee_loss(input_vel, input_pos_local, fake_vel, fake_pos_local,
                                    criterion_ee, skdata_sim, skdata_data)
                

                # combine all the losses together
                loss_gen =  loss_adv*mult_adv + loss_ee*mult_ee + \
                loss_glob*mult_glob + loss_chain_velo*mult_chain_velo
                
                # update gradient
                loss_gen.backward()
                gen_optimizer.step()
                
                # record losses 
                D_cumul += loss_discrim
                G_cumul += loss_gen
                
                G_loss_adv_cumul += loss_adv
                G_loss_ee_cumul  += loss_ee
                G_loss_velo_cumul+= loss_chain_velo
                G_loss_glob_cumul+= loss_glob

                if(i % 10 == 0):
                    print("Batch : {}, Discriminator loss: {}, Generator loss : {}".format(i, loss_discrim, loss_gen))


    