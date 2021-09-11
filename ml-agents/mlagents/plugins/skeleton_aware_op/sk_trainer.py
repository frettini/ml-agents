from mlagents.torch_utils import torch, default_device
import random

from mlagents.plugins.dataset.dataset import  UnityMotionDataset, SkeletonInfo
from mlagents.plugins.skeleton_aware_op.autoencoder_temporal import StaticEncoder, Encoder, Decoder
from mlagents.plugins.skeleton_aware_op.discriminator_temporal import Discriminator
from mlagents.plugins.bvh_utils import lafan_utils as utils
import mlagents.plugins.utils.logger as log 

from mlagents.plugins.bvh_utils.visualize import skeletons_plot, motion_animation

import matplotlib.pyplot as plt
import numpy as np

class Retargetter():
    def __init__(self, options, skdata_input, skdata_output):
        
        self.device = default_device()

        self.skdata_input = skdata_input
        self.skdata_output = skdata_output

        # Retargetter initialization
        self.encoder_sim = Encoder(self.skdata_input.edges, options)
        # use the initialization of encoder_data to get the correct pooling lists
        self.encoder_data = Encoder(self.skdata_output.edges, options) 
        self.decoder_data = Decoder(self.encoder_data, options)

        self.static_encoder_sim = StaticEncoder(self.skdata_input.edges, options).to(self.device)
        self.static_encoder_data = StaticEncoder(self.skdata_output.edges, options).to(self.device)

    def retarget(self, input_motion):
        """
        Retarget the motion from window frames from one window to another. 
        This function assumes that the data is already normalized, and expects the user 
        to denormalize the output data. This is done so that the retargetting function 
        doesn't have to hold the datasets.
        :params input_motion: [batch_size, window_size, n_joints*channel_size] tensor 
        :returns res: [batch_size, window_size, n_joints*channel_size] tensor of the retargetted rotations
        """

        # first get the offsets from the static encoder 
        deep_offsets_sim = self.static_encoder_sim(self.skdata_input.offsets.reshape(1, self.skdata_input.offsets.shape[0], -1))
        deep_offsets_output = self.static_encoder_data(self.skdata_output.offsets.reshape(1, self.skdata_output.offsets.shape[0], -1))

        motion = input_motion.permute(0,2,1)
        latent = self.encoder_sim(motion, deep_offsets_sim)
        res = self.decoder_data(latent, deep_offsets_output)
        res = res.permute(0,2,1)

        return res

    def save(self, model_path):
        state_dict = {'encoder_sim' : self.encoder_sim.state_dict(),
                      'encoder_data': self.encoder_data.state_dict(),
                      'decoder_data': self.decoder_data.state_dict(),
                      'static_encoder_sim' : self.static_encoder_sim.state_dict(),
                      'static_encoder_data': self.static_encoder_data.state_dict()}
        torch.save(state_dict, model_path)

    def load(self, model_path):
        state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
        self.encoder_sim.load_state_dict(state_dict['encoder_sim'])
        self.encoder_data.load_state_dict(state_dict['encoder_data'])
        self.decoder_data.load_state_dict(state_dict['decoder_data'])
        self.static_encoder_sim.load_state_dict(state_dict['static_encoder_sim'])
        self.static_encoder_data.load_state_dict(state_dict['static_encoder_data'])


class Sk_Trainer():

    def __init__(self, options, input_motion_path, adv_motion_path):
        
        self.options = options
        self.device = default_device()

        # Load Input and Adversarial Dataset
        self.input_dataset = UnityMotionDataset(input_motion_path, options=options)
        self.adv_dataset = UnityMotionDataset(adv_motion_path, options=options)
        self.skdata_input = self.input_dataset.skdata
        self.skdata_adv = self.adv_dataset.skdata
        self.input_limits = None
        self.adv_limits = None

        # Initialize the Discriminator and Retargetter
        self.discriminator = Discriminator(self.skdata_adv.edges, options).to(self.device)
        self.retargetter = Retargetter(options, self.skdata_input, self.skdata_adv)

        # Optimizers concatenate parameters used for pose generation
        gen_parameters = list(self.retargetter.encoder_sim.parameters()) + list(self.retargetter.decoder_data.parameters()) \
            + list(self.retargetter.static_encoder_sim.parameters()) + list(self.retargetter.static_encoder_data.parameters())

        self.gen_optimizer = torch.optim.Adam(gen_parameters, lr=self.options["sk_g_lr"], betas=(0.9, 0.999))
        self.discrim_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=self.options["sk_d_lr"], betas=(0.9, 0.999))
        
        # scheduler = torch.optim.lr_scheduler.StepLR(gen_optimizer, 10, 0.9)
        # cyclic_decay = lambda x : max(0.0001,0.4*(np.cos(lr_freq*x)+1.2)*(lr_decay**x))
        # scheduler = torch.optim.lr_scheduler.LambdaLR(gen_optimizer, lr_lambda=cyclic_decay)

        self.real_label = 1
        self.fake_label = 0

        self.criterion_gan = torch.nn.MSELoss()
        self.criterion_ee = torch.nn.MSELoss()
        self.criterion_root_velocity = torch.nn.MSELoss()
        self.criterion_velo = torch.nn.MSELoss()
        self.criterion_global_rotation = torch.nn.MSELoss()

        self.G_cumul = self.G_loss_adv_cumul = self.G_loss_ee_cumul = self.G_loss_velo_cumul = self.G_loss_glob_cumul = 0
        self.D_cumul = self.D_real_cumul = self.D_fake_cumul = 0

    def train(self, print_skeleton=False):
        """
        Main train loop for the retargetting function.
        """

        # do the job of a dataloader (shuffle and load only once each data point)
        # by hand because our dataset class cannot comply with the dataloader format.
        # get length of effective dataset
        if self.input_limits is not None:
            length = self.input_limits[1] - self.input_limits[0]
        else:
            length = len(self.input_dataset)
            
        # get the number of windows in the dataset (with overlap thus //2), generate shuffled list 
        n_windows = length // (self.options["window_size"]//2) - 1
        wind_indices = np.array(list(range(n_windows)), dtype=np.int32) * (self.options["window_size"]//2)
        random.shuffle(wind_indices)
        
        # separate the data in train and test batches
        ind_split = int(len(wind_indices) * 0.8)
        train_indices  = wind_indices[:ind_split]
        test_indices = wind_indices[ind_split:]
        n_loop = len(train_indices) // self.options["sk_batch_size"]

        for ep in range(self.options["sk_K_epochs"]):
            print('Epoch : ', ep)

            # at every epoch shuffle the batches around
            random.shuffle(train_indices)

            for i in range(n_loop):
                
                # get start and end indices of each batch, then load data 
                start_inds = train_indices[i*self.options["sk_batch_size"]:(i+1)*self.options["sk_batch_size"]] 
                end_inds = start_inds + self.options["window_size"]

                # reshape motion to [batch_size, window_size, n_joints*channel_size]
                motion = self.input_dataset.to_skaware(start_inds, end_inds)
                curr_batch_size = motion.shape[0]

                # add noise for study
                noise = torch.rand_like(motion) * self.options["noise_std"] + self.options["noise_mean"]
                motion = motion + noise

                # get a frame from the "real" data , avoid the data with weird gates
                randinds = np.random.randint(0,4000 - self.options["window_size"], curr_batch_size)
                real_motion = self.adv_dataset.to_skaware(randinds, randinds + self.options["window_size"])

                fake_motion_data, real_motion_data, input_motion_data = self.retarget(motion, real_motion)

                # the GAN update code is mostly taken from :
                #https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html#loss-functions-and-optimizers
                self.update_discriminator(real_motion_data[1], fake_motion_data[1], isTest = False)
                self.update_generator(input_motion_data, fake_motion_data, isTest = False)

                if(i % 20 == 0):
                    print("Batch : {}, Discriminator loss: {}, Generator loss : {}".format(i, self.D_cumul, self.G_cumul))

        
            N = n_loop
            log.writer.add_scalar('Train/Discriminator/D_loss', self.D_cumul/N, ep)
            log.writer.add_scalar('Train/Discriminator/D_real', self.D_real_cumul/N, ep)
            log.writer.add_scalar('Train/Discriminator/D_fake', self.D_fake_cumul/N, ep)

            log.writer.add_scalar('Train/AE/G_loss',    self.G_cumul/N, ep)
            log.writer.add_scalar('Train/AE/adv_loss',  self.G_loss_adv_cumul/N, ep)
            log.writer.add_scalar('Train/AE/ee_loss',   self.G_loss_ee_cumul/N, ep)
            log.writer.add_scalar('Train/AE/velo_loss', self.G_loss_velo_cumul/N, ep)
            log.writer.add_scalar('Train/AE/glob_loss', self.G_loss_glob_cumul/N, ep)
            
            self.G_cumul = self.G_loss_adv_cumul = self.G_loss_ee_cumul  = self.G_loss_velo_cumul = self.G_loss_glob_cumul = 0
            self.D_cumul = self.D_real_cumul = self.D_fake_cumul = 0   

            # Record test losses
            with torch.no_grad():
                # Test
                # get some random samples from both the test and train data and get the loss and print results
                test_size = len(test_indices)
                # test_input_inds = np.random.randint(0,len(self.input_dataset) - self.options["window_size"], test_size)
                test_adv_inds = np.random.randint(0,len(self.adv_dataset) - self.options["window_size"], test_size)
                motion = self.input_dataset.to_skaware(test_indices, test_indices + self.options["window_size"])
                real_motion = self.adv_dataset.to_skaware(test_adv_inds, test_adv_inds + self.options["window_size"])


                fake_motion_data, real_motion_data, input_motion_data = self.retarget(motion, real_motion)
                self.update_discriminator(real_motion_data[1], fake_motion_data[1], isTest = True)
                self.update_generator(input_motion_data, fake_motion_data, isTest = True)

                if print_skeleton is True and ep%10 == 0:
                    batch_ind = np.random.randint(0, test_size)
                    frame_ind = np.random.randint(0, self.options["window_size"])
                    # skeletons_plot([input_motion_data[1][batch_ind,frame_ind], fake_motion_data[1][batch_ind,frame_ind]],
                    #                 [self.skdata_input.edges, self.skdata_adv.edges], 
                    #                 colors_list=['g','r','b'], )
                    skeletons_plot([input_motion_data[1][batch_ind,frame_ind]],
                                    [self.skdata_input.edges], 
                                    colors_list=['g'] )
                    skeletons_plot([fake_motion_data[1][batch_ind,frame_ind]],
                                    [self.skdata_adv.edges], 
                                    colors_list=['r'], )
                    # skeletons_plot([real_motion_data[1][batch_ind,frame_ind]],
                    #                 [self.skdata_adv.edges], 
                    #                 colors_list=['b'], )

                N = 1
                log.writer.add_scalar('Test/Discriminator/D_loss', self.D_cumul/N, ep)
                log.writer.add_scalar('Test/Discriminator/D_real', self.D_real_cumul/N, ep)
                log.writer.add_scalar('Test/Discriminator/D_fake', self.D_fake_cumul/N, ep)

                log.writer.add_scalar('Test/AE/G_loss',    self.G_cumul/N, ep)
                log.writer.add_scalar('Test/AE/adv_loss',  self.G_loss_adv_cumul/N, ep)
                log.writer.add_scalar('Test/AE/ee_loss',   self.G_loss_ee_cumul/N, ep)
                log.writer.add_scalar('Test/AE/velo_loss', self.G_loss_velo_cumul/N, ep)
                log.writer.add_scalar('Test/AE/glob_loss', self.G_loss_glob_cumul/N, ep)
            
                self.G_cumul = self.G_loss_adv_cumul = self.G_loss_ee_cumul  = self.G_loss_velo_cumul = self.G_loss_glob_cumul = 0
                self.D_cumul = self.D_real_cumul = self.D_fake_cumul = 0   

    def retarget(self, input_motion, real_motion):
        """
        Wrapper around the retargetter retargeting function which takes care of all the data processing
        needed to then compute the losses
        """

        # perform retargeting
        input_not_norm = input_motion.clone()
        self.input_dataset.normalize(skaware_data = input_motion)
        output_motion = self.retargetter.retarget(input_motion)
        self.adv_dataset.denormalize(skaware_data = output_motion)

        # extract global position, root velocity,  
        fake_data = utils.get_pos_info_from_raw(output_motion, self.skdata_adv, self.options, norm_rot=True)
        real_data = utils.get_pos_info_from_raw(real_motion, self.skdata_adv, self.options, norm_rot=False)
        input_motion_data = utils.get_pos_info_from_raw(input_not_norm, self.skdata_input, self.options, norm_rot=False )

        

        return fake_data, real_data, input_motion_data

    def update_generator(self, input_data, output_data, isTest = False):
        """
        Compute the full generator loss (adv and other) and its gradients and apply them 
        to the autoencoder and static encoders.
        """
        
        # TODO: pass all the required information for updating the generator

        # UPDATE GENERATOR NETWORK
        if not isTest:
            self.gen_optimizer.zero_grad()

        input_pos, input_pos_local, input_root_rotation, input_glob_velo, input_vel, _ = input_data
        output_pos, output_pos_local, output_root_rotation, output_glob_velo, output_vel, _ = output_data

        curr_batch_size = input_pos.shape[0]
        rot_offset = torch.tensor(self.options["input_offset"]).float()
        glob_rot_offset = rot_offset.reshape(1,1,1,4).repeat(curr_batch_size, self.options['window_size'], 1, 1)
        pos_rot_offset = rot_offset.reshape(1,1,1,4).repeat(curr_batch_size, self.options['window_size'], self.skdata_input.num_joints, 1)
        
        input_glob_velo = utils.quat_mul_vec(glob_rot_offset, input_glob_velo) # glob velocity
        input_pos_local = utils.quat_mul_vec(pos_rot_offset, input_pos_local) # local position


        curr_batch_size = input_pos.shape[0]
        # skeletons_plot([input_pos[0,0]], [self.skdata_input.edges], ['b'])
        # Adversial Loss, use real labels to maximize log(D(G(x))) instead of log(1-D(G(x)))
        loss_adv = self.discriminator.G_loss(output_pos_local)

        # Gobal, Velocity and End Effector Losses
        loss_rot = self.criterion_global_rotation(output_root_rotation, input_root_rotation)

        # Get the loss that tries to match the total velocity of every corresponding chain        
        loss_glob_velo = self.criterion_velo(output_glob_velo/self.skdata_adv.height, input_glob_velo/self.skdata_input.height)
        loss_ee = self.calc_ee_loss(input_vel, input_pos_local, output_vel, output_pos_local)
        

        # combine all the losses together
        loss_gen =  loss_adv*self.options["sk_adv_factor"] + loss_ee*self.options["sk_ee_factor"] + \
        loss_rot*self.options["sk_rot_factor"] + loss_glob_velo*self.options["sk_glob_velo_factor"]
        
        if not isTest:
            # update gradient
            loss_gen.backward()
            self.gen_optimizer.step()
        
        # record losses 
        self.G_cumul += loss_gen.detach().item()
        
        self.G_loss_adv_cumul += loss_adv.detach().item()
        self.G_loss_ee_cumul  += loss_ee.detach().item()
        self.G_loss_velo_cumul+= loss_glob_velo.detach().item()
        self.G_loss_glob_cumul+= loss_rot.detach().item()
        
    def update_discriminator(self, real_pos, fake_pos, isTest = False):
        """
        Compute the discriminator loss and its gradient, and apply it to the discriminator
        """
        
        curr_batch_size = real_pos.shape[0]
        label = torch.full((curr_batch_size,), self.real_label, dtype=torch.float, device=default_device())

        rot_offset = torch.tensor(self.options["input_offset"]).float()
        pos_rot_offset = rot_offset.reshape(1,1,1,4).repeat(curr_batch_size, self.options['window_size'], self.skdata_input.num_joints, 1)

        # pos_rot_offset = rot_offset.reshape(1,1,1,4).repeat(curr_batch_size, self.options['window_size'], self.skdata_adv.num_joints, 1)
        # fake_pos = utils.quat_mul_vec(pos_rot_offset, fake_pos) # local position
        
        if not isTest:
            self.discriminator.zero_grad()

        # forward pass
        output = self.discriminator.forward(real_pos.float()).view(-1)
        loss_real = self.criterion_gan(output,label)
        if not isTest:
            loss_real.backward()
        
        # do the same with the generated position
        label.fill_(self.fake_label)
        output = self.discriminator.forward(fake_pos.float().detach()).view(-1)
        loss_fake = self.criterion_gan(output, label)
        if not isTest:
            loss_fake.backward()

        loss_discrim = loss_real + loss_fake

        if not isTest:
            self.discrim_optimizer.step()

        # record losses 
        self.D_cumul += loss_discrim.detach().item()
        self.D_real_cumul += loss_real.detach().item()
        self.D_fake_cumul  += loss_fake.detach().item()
    
    def calc_ee_loss(self, real_vel, real_pos, fake_vel, fake_pos):
        """
        Calculate the end effector loss using the fake velocity and position (local or global)
        """
        # [batch, frame, n_end_effector, 6]
        ee_false = self.skdata_adv.ee_id[self.skdata_adv.ee_id != 0]
        ee_real = self.skdata_input.ee_id[self.skdata_input.ee_id != 0]

        # print(fake_pos[:,:,0:1,:].shape)
        fake_pos_loc = fake_pos - fake_pos[:,:,0:1,:]
        real_pos_loc = real_pos - real_pos[:,:,0:1,:]

        # fake_ee = fake_pos_loc[:,:,ee_false,:]
        # real_ee = real_pos_loc[:,:,ee_real,:]

        real_vel = utils.get_batch_velo2(real_pos, self.skdata_adv.frametime)
        fake_vel = utils.get_batch_velo2(fake_pos, self.skdata_input.frametime)

        fake_ee = torch.cat((fake_vel[:,:,ee_false,:], fake_pos_loc[:,:,ee_false,:]), dim=2)
        real_ee = torch.cat((real_vel[:,:,ee_real,:], real_pos_loc[:,:,ee_real,:]), dim=2)
        
        # fake_ee = torch.cat((fake_vel, fake_pos_loc), dim=2)
        # real_ee = torch.cat((real_vel, real_pos_loc), dim=2)

        real_length = self.skdata_input.ee_length.reshape(1,1,self.skdata_input.ee_length.shape[0],1)
        fake_length = self.skdata_adv.ee_length.reshape(1,1,self.skdata_adv.ee_length.shape[0],1)

        # print(torch.repeat_interleave(real_length,2, dim=2))
        real_length = torch.repeat_interleave(real_length,2, dim=2)
        fake_length = torch.repeat_interleave(fake_length,2, dim=2)


        # loss_ee_velo = criterion_ee(fake_joint_vel[:,:,ee_id,:], real_joint_vel[:,:,ee_id,:])
        loss_ee_velo = self.criterion_ee(fake_ee/fake_length, real_ee/real_length)
        # loss_ee_velo = self.criterion_ee(fake_ee, real_ee)

        return loss_ee_velo