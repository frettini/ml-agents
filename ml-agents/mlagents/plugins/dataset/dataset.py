import os
from re import sub

from mlagents.torch_utils import torch, default_device
import numpy as np

from mlagents.plugins.bvh_utils import BVH_mod as BVH
from mlagents.plugins.bvh_utils.Quaternions import Quaternions 
import mlagents.plugins.bvh_utils.lafan_utils as utils
from mlagents.plugins.skeleton_aware_op.options import get_options 


class TemporalMotionData(torch.utils.data.Dataset):

    def __init__(self, input_path:str, recalculate_mean_var:bool = False,
                 normalize_data:bool = False, subsample=False, xyz='xyz',
                 rot_order = 'xyz', device:str='cpu'):
        
        options = get_options()
        self.window_size = options['window_size']
        torch.device(device)

        self.skdata = SkeletonInfo(input_path, subsample=subsample, xyz=xyz)

        input_files = []
        for file in os.listdir(input_path):
            if file.endswith('.bvh'):
                input_files.append(file)
        
        input_motions = []
        self.normalize_data = normalize_data
        self.compute_edge = True

        # cycle through all files in the directory
        # add the motion to the list 
        for i, file in enumerate(input_files):

            full_path = os.path.join(input_path, file)
            anim, names, frametime = BVH.load(full_path, need_quater = True, order=rot_order)
            
            # anim rotation shape : [motion_lenght, num_joints, 3]
            # anim position shape : [motion_lenght, num_joints, 3]

            tmp_pos = anim.positions.copy()
            tmp_rot = anim.rotations.qs.copy()

            # reorder to fit the frame of reference
            for ind, letter in enumerate(xyz):
                
                if letter == 'x':
                    anim.positions[..., ind] = tmp_pos[..., 0]
                    anim.rotations.qs[..., ind+1] = tmp_rot[..., 1]
                elif letter == 'y':
                    anim.positions[..., ind] = tmp_pos[..., 1]
                    anim.rotations.qs[..., ind+1] = tmp_rot[..., 2]
                elif letter == 'z':
                    anim.positions[..., ind] = tmp_pos[..., 2]
                    anim.rotations.qs[..., ind+1] = tmp_rot[..., 3]

            glob_position = torch.tensor(anim.positions[:, 0, :]).float()
            rotations = torch.tensor(anim.rotations.qs).float()

            # if modified, remove the offset using the first global rotation
            if xyz != 'xyz':
                rotate = rotations[0,0,:]
                rotate[0] = -rotate[0]
                rotations[:,0:1,:] = utils.quat_mul(rotate, rotations[:,0:1,:])
                rotations[0,0,:] = torch.tensor([1,0,0,0]).float()

            num_frames = rotations.shape[0]
            num_joints = len(anim.parents)

            # reshape rotations so that we can add the global position
            index = []
            for e in self.skdata.edges:
                index.append(e[0])
            rotations = rotations[:,index, :]

            # get the global velocity:
            glob_velocity = utils.get_velocity(glob_position, frametime)

            # concatenate the rotations, global pos and a padding together [frames, joints*channel_base]
            rotations_cat = rotations.reshape(num_frames, -1)
            input_motion = torch.cat([rotations_cat, glob_velocity.reshape(num_frames, -1), torch.zeros((num_frames,1))], axis=1)
            
            if subsample is True:
                input_motion = input_motion[::2, :]

            # print("[DATASET] input data : ", input_motion.shape)
            # cut the motion into motion chunks (size set in options)
            windowed_motion = self.get_windows(input_motion)


            # print("[DATASET] windowed data : ", windowed_motion.shape)
            input_motions.append(windowed_motion)

        self.input_motions = torch.cat(input_motions, dim=0)
        self.input_motions = self.input_motions.permute(0,2,1)
        print("[DATASET] final data shape : ", self.input_motions.shape)

        if(recalculate_mean_var):
            self.save_mean_var(input_path)
        else:
            print("[DATASET] load mean var")
            mean_var_file_path = os.path.join(input_path, 'mean_var.npy')
            mean_var = np.load(mean_var_file_path)
            self.mean_var = torch.tensor(mean_var)

    def __len__(self):
        return self.input_motions.shape[0]

    def __getitem__(self, index):

        # motion shape [[batch_size,] joint, frame]
        motion = self.input_motions[index]
        
        # normalize data
        if self.normalize_data is True:
            motion = (motion-self.mean_var[np.newaxis, 0,:,:])/self.mean_var[np.newaxis, 1,:,:]
            motion[:,-1,:] = 0
        
        return motion

    def get_windows(self, input_motion):

        # calculate the number of windows in a motion clip 
        # with half window overlap
        motion_len = input_motion.shape[0]
        n_windows = motion_len//self.window_size * 2 -1
        windowed_motion = torch.zeros((n_windows, self.window_size, input_motion.shape[-1])).double()
        
        ind = 0
        for i in range(n_windows):
            windowed_motion[i,:,:] = input_motion[ind:ind+self.window_size, :] 
            ind += self.window_size//2

        return windowed_motion

    def denormalize(self, data):
        """ 
        Denormalize data using the mean and variance of the dataset
        data - [batch_size, J*4, frame]
        """
        return data*self.mean_var[np.newaxis,1,:, :]+self.mean_var[np.newaxis, 0,:,:]

    def save_mean_var(self, input_path):
        """
        Save the mean and variance of the input motion into a npy file
        """
        
        input_concatenated = self.input_motions.permute(0,2,1)
        input_concatenated = input_concatenated.reshape(-1, input_concatenated.shape[-1])
        print("[DATASET] input concatenated shape: ", input_concatenated.shape)

        mean = torch.mean(input_concatenated, dim = 0, keepdim = True)
        var = torch.var(input_concatenated, dim = 0, keepdim = True)
        
        # replace any 0s by 1s to avoid division by 0
        var = torch.where( var < 1e-5, 1., var)
        var = var ** (1/2)

        print("[DATASET] mean shape: ", mean.shape)
        print("[DATASET] var shape: ", var.shape)

        # save it in a single file with structure [[mean],[var]] shape [2, J*4]
        self.mean_var = torch.cat([mean,var], dim=0)
        self.mean_var = self.mean_var[..., np.newaxis]

        print("[DATASET] mean var shape :", self.mean_var.shape)

        mean_var_file_path = os.path.join(input_path, 'mean_var.npy')
        np.save(mean_var_file_path, self.mean_var.detach().cpu().numpy(), allow_pickle=True)

    def reconstruct_from_windows(self, input_motion):
        """
        Reconstruct the motion from a set of overlapping windows (the same overlap as 
        they are constructed with). 
        :params input_motion: tensor [num_batch, num_windows, num_joints, channel_size]
        """
        overlap = self.window_size//2
        init_motion = input_motion[0,:overlap]
        output = input_motion[1:,overlap:].reshape(-1, input_motion.shape[-2], input_motion.shape[-1])
        output = torch.cat((init_motion,output), dim=0)
        return output


class UnityMotionDataset(torch.utils.data.Dataset):
    """
    This dataset reads csv files generated by recording positions and rotations within Unity.
    It is used to load data for the reinforcement learning policy as well as the skeleton aware 
    motion retargetting by using the to_skdata() function. 
    """

    def __init__(self, input_path:str, skdata_sidechannel, frame_boundaries=None, device:str='cpu'):
        options = get_options()
        self.window_size = options['window_size']
        self.channel_base = options["channel_base"]
        torch.device(device)

        # get the skeleton information
        self.skdata = SkeletonInfo(skchannel = skdata_sidechannel)

        # find all txt files in the dataset
        input_files = []
        for file in os.listdir(input_path):
            if file.endswith('.txt'):
                input_files.append(input_path + file)

        rotations = []
        positions = []
        velocity = []

        n_features_per_joints = 10

        # open and parse the csv files
        for file in input_files:

            with open(file) as f:
                lines = f.readlines()
            
            n_lines = len(lines)
            n_features = len(lines[0].split(','))
            features = torch.zeros((n_lines,n_features)).float()

            for i, line in enumerate(lines):
                line = line
                line_array = [float(f) for f in line.split(',')]
                features[i] = torch.tensor(line_array).float()
            features  = features.reshape(features.shape[0], self.skdata.num_joints, n_features_per_joints)

            # extract and process features: remove position and hip rotation offset            
            temp_pos = features[:,:,:3]
            temp_pos = temp_pos - temp_pos[:,0:1,:]
            temp_rot = features[:,:,3:7]
            temp_rot[:,0,:] = torch.tensor([1.,0.,0.,0.]).float()
            temp_vel = features[:,:,7:]

            positions.append(temp_pos)
            rotations.append(temp_rot)
            velocity.append(temp_vel)
    
        self.positions = torch.cat(positions, dim=0)
        self.rotations = torch.cat(rotations, dim=0)
        self.velocity  = torch.cat(velocity,  dim=0) #utils.get_velocity(self.positions, self.skdata.frametime)

        # calculate the statistics
        self.positions_mean = torch.mean(self.positions, dim=0)
        self.positions_var = torch.var(self.positions, dim=0)
        self.positions_var = torch.where( self.positions_var.double() < 1e-5, 1., self.positions_var.double()).float()
        self.positions_var = self.positions_var ** (1/2)

        self.rotations_mean = torch.mean(self.rotations, dim=0)
        self.rotations_var = torch.var(self.rotations, dim=0)
        self.rotations_var = torch.where( self.rotations_var.double() < 1e-5, 1., self.rotations_var.double()).float()
        self.rotations_var = self.rotations_var ** (1/2)

        self.velocity_mean = torch.mean(self.velocity, dim=0)
        self.velocity_var = torch.var(self.velocity, dim=0)
        self.velocity_var = torch.where( self.velocity_var.double() < 1e-5, 1., self.velocity_var.double()).float()
        self.velocity_var = self.velocity_var ** (1/2)

    def __getitem__(self, index):
        """
        return positions and rotation (in that order) with shape [n_frames, n_joints, 3/4]
        """
        positions = self.positions[index] 
        rotations = self.rotations[index]
        velocity = self.velocity[index]
        return positions, rotations, velocity

    def __len__(self):
        return self.positions.shape[0]

    def to_skdata(self, start_inds, end_inds):
        """
        Provide data to comply to the shape needed for the skeleton aware code. 
        That means shape [batch_size, window_size, n_joints*channel_size]
        :params start_inds, end_inds: lists/ arrays of indices which represent the start and the end 
        of each windows 
        """

        positions = []
        rotations = []
        n_windows = len(start_inds)

        # extract the positions and rotations windows
        for i in range(len(start_inds)):
            pos = self.positions[start_inds[i]:end_inds[i]]
            rot = self.rotations[start_inds[i]:end_inds[i]]
            positions.append(pos)
            rotations.append(rot)

        # concatenate them in a single tensor of shape [n_windows, n_frames, n_joints, 3/4]
        positions = torch.cat(positions, dim=0)
        rotations = torch.cat(rotations, dim=0)

        self.normalize(positions,rotations)

        # reshape rotations so that we provide the parent rotation (edge rotation)
        index = []
        for e in self.skdata.edges:
            index.append(e[0])
        rotations = rotations[:,index, :]
        # concatenate the rotations, global pos and a padding together [n_windows, n_frames, joints*channel_base]
        rotations_cat = rotations.reshape(n_windows, self.window_size, self.skdata.num_joints * self.channel_base)

        # get the global velocity:
        velocity = utils.get_batch_velo2(positions, frametime)
        root_velocity = velocity[:,:,0,:]

        motion_data = torch.cat([rotations_cat, root_velocity.reshape(n_windows, self.window_size, -1), torch.zeros((n_windows, self.window_size,1))], axis=1)
    
    def normalize(self, positions = None, rotations=None):
        """
        Normalize the data passed. This is an inplace function which changes the variables passed
        to the function. 
        """

        if positions is not None:
            positions = (positions - self.positions_mean) / self.positions_var

        if rotations is not None:
            rotations = (rotations - self.rotations_mean) / self.rotations_var

    def denormalize(self, positions = None, rotations = None):
        """
        Normalize the data passed. This is an inplace function which changes the variables passed
        to the function. 
        """
        if positions is not None:
            positions = positions * self.positions_var + self.positions_mean

        if rotations is not None:
            rotations = rotations * self.rotations_var + self.rotations_mean


class SkeletonInfo:
    def __init__(self, input_path=None, skchannel=None, subsample=False, xyz = 'xyz'):
        """
        Returns information about the skeleton topology using an example motion file 
        specified by file_path. 
        Args:
        input_path   - string, path to directory containing motion examples.

        Returns:
        edges       - [n_joints-1, 2] array of indices of joints which constitute edges \n
        offsets     - [n_joints, 3] tensor of offsets for each corresponding joint \n
        ee_id       - [n_effectors] list of end effector indices \n
        chain_indices - [n_chain] list of list of indices contained in a sequential chain \n
        parents     - list of parents indices of each joint (-1 if no parent) \n
        frametime   - float , time elapsed between two frames \n
        """

        if input_path is not None:
            self.load_from_file(input_path, xyz)
        elif skchannel is not None:
            self.load_from_channel(skchannel)
        else:
            raise Exception('Pass either the input path or the skchannel to SkeletonInfo')

        # extract end effectors for autoencoder loss and large nodes 
        # TODO: make the loss tolerant to different shapes and sizes 
        self.edges = utils.build_edges(self.parents)
        ee_id, j_count = torch.unique(self.edges.reshape(-1), return_counts = True)
        j =  torch.logical_or(j_count == 1, j_count > 2) # end effectors and multi_chain nodes 
        # j =  j_count == 1 # only end effectors 
        self.ee_id = ee_id[j]

        # This algorithm works for skeletons with same topology but different number of joints
        # and proportions. This means that there are an equivalent number of chains.
        # We extract the joint indices of each chain here, which will be used for some losses.
        self.chain_indices = utils.build_chain_list(self.edges)
        
        # get height (heighest point minus lowest)
        self.height = utils.get_height(self.parents, self.offsets)

        offset_norms = torch.linalg.norm(self.offsets,dim=1)
        offset_norms[0] = 0 # remove global pos

        # get the length of each chain (from base to end)
        self.ee_length = torch.zeros(len(self.chain_indices)).float()
        for ind, chains in enumerate(self.chain_indices):
            self.ee_length[ind] = torch.sum(offset_norms[chains])

    def load_from_file(self, input_path, xyz='xyz'):

        file_list = os.listdir(input_path)
        for file in file_list:
            if file.endswith('.bvh'):
                file_path = os.path.join(input_path, file)
                break 

        anim, names, frametime = BVH.load(file_path, need_quater = False)

        # generate edge list
        self.num_joints = len(anim.parents)
        self.parents = anim.parents
        self.frametime = frametime

        offsets     = anim.offsets.copy()
        for ind, letter in enumerate(xyz):
            if letter == 'x':
                offsets[:,ind]     = anim.offsets[:,0]
            elif letter == 'y':
                offsets[:,ind]     = anim.offsets[:,1]
            elif letter == 'z':
                offsets[:,ind]     = anim.offsets[:,2]
    
        # generate skeleton offsets, shape [J,3]       
        self.offsets = torch.tensor(offsets).float()
        self.offsets[0,:] = torch.zeros(3).float()

    def load_from_channel(self, skchannel):
        # extract joints, init rotations, offsets, parents and frametime
        self.num_joints = int(skchannel.msg[0])
        init_rotations = torch.tensor(skchannel.msg[1:1+self.num_joints*4]).float().reshape(self.num_joints,4)
        self.offsets = torch.tensor(skchannel.msg[ 1+self.num_joints*4 : 1+self.num_joints*4 + self.num_joints*3 ]).float().reshape(self.num_joints,3)
        self.parents = [int(f) for f in skchannel.msg[ 1+self.num_joints*4 + self.num_joints*3 : -1 ]]
        self.frametime = skchannel.msg[ -1 ]
        
    

if __name__ == "__main__":

    import matplotlib.pyplot as plt
    from mlagents.plugins.bvh_utils.visualize import skeletons_plot, motion_animation
    from mlagents.plugins.bvh_utils import lafan_utils as utils

    from mlagents.plugins.dataset.skeleton_side_channel import Skeleton_SideChannel 
    from mlagents_envs.environment import UnityEnvironment
        
    # Setup Unity Environment + sidechannel
    skeleton_sidechannel = Skeleton_SideChannel()

    try:
        env.close()
    except:
        pass

    print("Waiting for environment to boot")
    # filename = None enables to communicate directly with the unity editor
    env = UnityEnvironment(file_name=None, seed=1, side_channels=[skeleton_sidechannel])
    env.reset()

    input_path = "C:\\Users\\nicol\\Work\\Master\\dissertation\\ml-agents\\colab\\data\\"
    motion_data = UnityMotionDataset(input_path, skeleton_sidechannel)
        
    print("motion_data loaded")
    print("data length:", len(motion_data))

    pos, rot = motion_data[10:20]

    fig,ax = skeletons_plot([pos[0].cpu().detach()], [motion_data.skdata.edges], colors_list=['b'], limits=[[-1.,1.],[-1.,1.],[-1.,1.]], return_plot=True)
    plt.plot()