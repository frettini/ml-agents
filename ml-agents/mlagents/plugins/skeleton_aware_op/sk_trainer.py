from mlagents.torch_utils import torch, default_device
import random

from mlagents.plugins.dataset.dataset import  Motion_Dataset
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
    def __init__(self):
        self.encoder_sim = Encoder(skdata_sim.edges)
        # use the initialization of encoder_data to get the correct pooling lists
        # its forward is not used
        self.encoder_data = Encoder(skdata_data.edges) 
        self.decoder_data = Decoder(encoder_data)

        self.static_encoder_sim = StaticEncoder(skdata_sim.edges).to(device)
        self.static_encoder_data = StaticEncoder(skdata_data.edges).to(device)
        self.discriminator = Discriminator(skdata_data.edges).to(device)

    def retarget(self, motion):

        # reshape motion to [batch_size, window_size, n_joints*channel_size]
        motion = motion.reshape(motion.shape[0], motion.shape[-2], motion.shape[-1]).float()
        curr_batch_size = motion.shape[0]

        # get a frame from the "real" data 
        randinds = np.random.randint(0,len(adv_dataset), curr_batch_size)
        real_motion = adv_dataset[randinds]

        offsets_sim = skdata_sim.offsets.reshape(1, 1, skdata_sim.offsets.shape[0], skdata_sim.offsets.shape[1])
        offsets_sim = offsets_sim.repeat(curr_batch_size, options['window_size'], 1, 1)

        offsets_data = skdata_data.offsets.reshape(1, 1, skdata_data.offsets.shape[0], skdata_data.offsets.shape[1])
        offsets_data = offsets_data.repeat(curr_batch_size, options['window_size'], 1, 1)

        #----------------------------------------------------
        # GENERATE AUTO ENCODED POSE
        #----------------------------------------------------

        # first get the offsets from the static encoder 
        # deep_offsets = static_encoder(torch.tensor(anim.offsets[np.newaxis, :,:]).float())
        deep_offsets_sim = static_encoder_sim(skdata_sim.offsets.reshape(1, skdata_sim.offsets.shape[0], -1))
        deep_offsets_data = static_encoder_data(skdata_data.offsets.reshape(1, skdata_data.offsets.shape[0], -1))

        latent = self.encoder_sim(motion, deep_offsets_sim)
        res = self.decoder_data(latent, deep_offsets_data)

        # denormalize data to get correct forward kinematics 
        if isNormalize is True:
            res = adv_dataset.denormalize(res)
            real_motion = adv_dataset.denormalize(real_motion)
            motion = input_dataset.denormalize(motion)


class Sk_Trainer():

    def __init__(self, options, input_motion_path, adv_motion_path):
        
        self.options = options

        self.retargetter = Retargetter()
        
        # Load Input and Adversarial Dataset
        self.input_dataset = Motion_Dataset(input_motion_path)
        self.adv_dataset = Motion_Dataset(adv_motion_path)
        self.input_limits = None
        self.adv_limits = None

        # this might be the only loss we can use since we don't 
        # have a direct reference to a motion
        self.criterion_gan = torch.nn.MSELoss()
        self.criterion_ee = torch.nn.MSELoss()
        self.criterion_glob = torch.nn.MSELoss()
        self.criterion_velo = torch.nn.MSELoss()

        # Optimizers
        # concatenate parameters used for pose generation
        gen_parameters = list(encoder_sim.parameters()) + list(decoder_data.parameters()) \
            + list(static_encoder_sim.parameters()) + list(static_encoder_data.parameters())

        self.gen_optimizer = torch.optim.Adam(gen_parameters, lr=learning_rate, betas=(0.9, 0.999))
        self.discrim_optimizer = torch.optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.9, 0.999))

        self.real_label = 1.0
        self.fake_label = 0.0

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
                motion = motion.reshape(motion.shape[0], motion.shape[-2], motion.shape[-1]).float()
                curr_batch_size = motion.shape[0]

                # get a frame from the "real" data 
                randinds = np.random.randint(0,len(adv_dataset), curr_batch_size)
                real_motion = adv_dataset[randinds]

                offsets_sim = skdata_sim.offsets.reshape(1, 1, skdata_sim.offsets.shape[0], skdata_sim.offsets.shape[1])
                offsets_sim = offsets_sim.repeat(curr_batch_size, options['window_size'], 1, 1)

                offsets_data = skdata_data.offsets.reshape(1, 1, skdata_data.offsets.shape[0], skdata_data.offsets.shape[1])
                offsets_data = offsets_data.repeat(curr_batch_size, options['window_size'], 1, 1)

                #----------------------------------------------------
                # GENERATE AUTO ENCODED POSE
                #----------------------------------------------------

                # first get the offsets from the static encoder 
                # deep_offsets = static_encoder(torch.tensor(anim.offsets[np.newaxis, :,:]).float())
                deep_offsets_sim = static_encoder_sim(skdata_sim.offsets.reshape(1, skdata_sim.offsets.shape[0], -1))
                deep_offsets_data = static_encoder_data(skdata_data.offsets.reshape(1, skdata_data.offsets.shape[0], -1))


                # latent, res = auto_encoder(motion, deep_offsets)
                


                # TODO: Changed get_pos_info_from_raw definition to take in offsets as a 2d data [num_joints, 3]
                # need to adjust that
                fake_data = lafan_utils.get_pos_info_from_raw(res, skdata_data, offsets_data, options, norm_rot=True )
                fake_pos, fake_pos_local, fake_glob, fake_glob_velo, fake_vel = fake_data

                motion_data = lafan_utils.get_pos_info_from_raw(motion, skdata_sim, offsets_sim, options, norm_rot=False )
                input_pos, input_pos_local, input_glob, input_glob_velo, input_vel = motion_data

                real_data = lafan_utils.get_pos_info_from_raw(real_motion, skdata_data, offsets_data, options, norm_rot=True )
                real_pos, real_pos_local, real_glob, real_glob_velo, real_vel = real_data

                # the GAN update code is mostly taken from :
                #https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html#loss-functions-and-optimizers
                

                # UPDATE DISCRIMINATOR :
                # first do real data batch then fake data
                
                # start adverserial training after a set number of epochs
                loss_real, loss_fake, loss_discrim = discriminator.D_loss(real_pos_local, fake_pos_local, False)
                discrim_optimizer.step()
                
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


    def retarget(self, input):
        """
        Responsible of taking a valid motion for a given skeleton and reproducing this motion 
        on a different motion.
        """
        pass