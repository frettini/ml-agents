import os
from re import sub

from mlagents.torch_utils import torch, default_device
import numpy as np

from mlagents.plugins.bvh_utils import BVH_mod as BVH
from mlagents.plugins.bvh_utils.Quaternions import Quaternions 
from mlagents.plugins.bvh_utils.lafan_utils import get_velocity, build_edges, build_chain_list, get_height
from mlagents.plugins.bvh_utils.lafan_utils import quat_mul, quat_mul_vec, quat_inv
from mlagents.plugins.skeleton_aware_op.options import get_options 


class MotionData(torch.utils.data.Dataset):

    def __init__(self, input_path, recalculate_mean_var = False, normalize_data = True, device='cpu'):
        """
        input_path - str containing the path of the data [e.g. './data/Aj']

        This dataset first concatenates all motions to produce a mean and var estimate of the motion
        for normalization.
        """
        torch.device(device)

        input_files = []
        for file in os.listdir(input_path):
            if file.endswith('.bvh'):
                input_files.append(file)
        
        input_motions = []

        self.normalize_data = normalize_data

        # cycle through all files in the directory
        # add the motion to the list 
        for i, file in enumerate(input_files):
            
            full_path = os.path.join(input_path, file)
            anim, names, frametime = BVH.load(full_path, need_quater = False)
            
            # anim rotation shape : [motion_lenght, num_joints, 3]
            # anim position shape : [motion_lenght, num_joints, 3]
            # convert euler rotations to quaternions and concatenate them
            rotations = anim.rotations[:,:,:]
            glob_position = torch.tensor(anim.positions[:, 0, :])
            rotations = torch.tensor(Quaternions.from_euler(np.radians(rotations)).qs)

            num_frames = rotations.shape[0]
            num_joints = len(anim.parents)
            edges = np.zeros((num_joints-1, 2), dtype=int)

            # generate edge list
            count = 0
            for i, parent in enumerate(anim.parents):
                # check if parent is existent
                if parent != -1:
                    edges[count, 0] = parent
                    edges[count, 1] = i
                    count += 1

            # reshape rotations so that we can add the global position
            index = []
            for e in edges:
                index.append(e[0])
            rotations = rotations[:,index, :]

            # get the global velocity:
            glob_velocity = get_velocity(glob_position, frametime)

            # concatenate the rotations, global pos and a padding together
            rotations_cat = rotations.reshape(num_frames, -1)
            input_motion = torch.cat([rotations_cat, glob_velocity.reshape(num_frames, -1), torch.zeros((num_frames,1))], axis=1)
            # print("input motion name : {}, shape : {}".format(file, input_motion.shape))

            input_motions.append(input_motion)

        self.input_motions = torch.cat(input_motions, dim=0)
        print("[DATASET] input motion shape: ", self.input_motions.shape)

        # save and get the mean and variance of the input motion
        if recalculate_mean_var is True:
            self.save_mean_var(input_path)
        else:
            print("[DATASET] load mean var")
            mean_var_file_path = os.path.join(input_path, 'mean_var.npy')
            mean_var = np.load(mean_var_file_path)
            self.mean_var = torch.tensor(mean_var)
        
        print("[DATASET] mean var shape: ", self.mean_var.shape)

        # If the number of input motions is odd, remove the last motion from the list
        if(self.input_motions.shape[0]%2 != 0):
            self.input_motions = self.input_motions[:-1,:]

        # reshape motion to be in pairs of two [num_pairs, 2, J*4]
        self.input_motions = self.input_motions.reshape(self.input_motions.shape[0]//2,2,self.input_motions.shape[1])
        print("[DATASET] input motion reshape: ", self.input_motions.shape)
        
    def __len__(self):
        return self.input_motions.shape[0]

    def __getitem__(self,index):

        motion = self.input_motions[index]
        motion = motion.reshape(-1,motion.shape[-1])

        # normalize data
        if self.normalize_data is True:
            motion = (motion-self.mean_var[0])/self.mean_var[1]
            motion[:,-1] = 0
        
        return motion

    def denormalize(self, data):
        """ 
        Denormalize data using the mean and variance of the dataset
        data - [batch_size, J*4]
        """
        return data*self.mean_var[1]+self.mean_var[0]

    def save_mean_var(self, input_path):
        """
        Save the mean and variance of the input motion into a npy file
        """
        mean = torch.mean(self.input_motions, dim = 0, keepdim = True)
        var = torch.var(self.input_motions, dim = 0, keepdim = True)
        
        # replace any 0s by 1s to avoid division by 0
        var = torch.where( var < 1e-5, 1., var)
        var = var ** (1/2)

        print("[DATASET] mean shape: ", mean.shape)
        print("[DATASET] var shape: ", var.shape)

        # save it in a single file with structure [[mean],[var]] shape [2, J*4]
        self.mean_var = torch.cat([mean,var], dim=0)

        mean_var_file_path = os.path.join(input_path, 'mean_var.npy')
        np.save(mean_var_file_path, self.mean_var.detach().cpu().numpy(), allow_pickle=True)

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
                # rotate[0] = -rotate[0] # inverse rotation
                rotations[:,0:1,:] = quat_mul(quat_inv(rotate), rotations[:,0:1,:])
                rotations[0,0,:] = torch.tensor([1,0,0,0]).float()

            num_frames = rotations.shape[0]
            num_joints = len(anim.parents)
            # edges = np.zeros((num_joints-1, 2), dtype=int)  

            # # generate edge list
            # count = 0
            # for i, parent in enumerate(anim.parents):
            #     # check if parent is existent
            #     if parent != -1:
            #         edges[count, 0] = parent
            #         edges[count, 1] = i
            #         count += 1

            # reshape rotations so that we can add the global position
            index = []
            for e in self.skdata.edges:
                index.append(e[0])
            rotations = rotations[:,index, :]

            # get the global velocity:
            glob_velocity = get_velocity(glob_position, frametime)

            if xyz != 'xyz':
                rotate_r = rotate.clone().reshape(1,-1)
                rotate_r = rotate_r.repeat(glob_velocity.shape[0],1)
                glob_velocity = quat_mul_vec(quat_inv(rotate_r), glob_velocity[:,:])

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

class SkeletonInfo:
    def __init__(self, input_path, subsample=False, xyz = 'xyz'):
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

        file_list = os.listdir(input_path)
        for file in file_list:
            if file.endswith('.bvh'):
                file_path = os.path.join(input_path, file)
                break 

        anim, names, frametime = BVH.load(file_path, need_quater = False)

        # generate edge list
        self.num_joints = len(anim.parents)
        self.edges = build_edges(anim.parents)
        self.parents = anim.parents
        self.frametime = frametime

        if subsample is True: 
            self.frametime *= 2

        # extract end effectors for autoencoder loss and large nodes 
        # TODO: make the loss tolerant to different shapes and sizes 
        ee_id, j_count = torch.unique(self.edges.reshape(-1), return_counts = True)
        j =  torch.logical_or(j_count == 1, j_count > 2) # end effectors and multi_chain nodes 
        # j =  j_count == 1 # only end effectors 
        self.ee_id = ee_id[j]

        # This algorithm works for skeletons with same topology but different number of joints
        # and proportions. This means that there are an equivalent number of chains.
        # We extract the joint indices of each chain here, which will be used for some losses.
        self.chain_indices = build_chain_list(self.edges)

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
        # get height (heighest point minus lowest)
        self.height = get_height(self.parents, self.offsets)

        offset_norms = torch.linalg.norm(self.offsets,dim=1)
        offset_norms[0] = 0 # remove global pos

        # get the length of each chain (from base to end)
        self.ee_length = torch.zeros(len(self.chain_indices)).float()
        for ind, chains in enumerate(self.chain_indices):
            self.ee_length[ind] = torch.sum(offset_norms[chains])

    
if __name__ == "__main__":

    from mlagents.plugins.bvh_utils.visualize import skeleton_plot, two_skeleton_plot
    from mlagents.plugins.bvh_utils import lafan_utils

    # skeleton information : topology and offsets 
    file_path = "./data/LaFan/Train/walk1_subject1.bvh"
    anim, names, frametime = BVH.load(file_path, need_quater = False)

    num_joints = len(anim.parents)
    edges = torch.zeros((num_joints-1, 2), dtype=int, requires_grad=False)

    # generate edge list
    count = 0
    for i, parent in enumerate(anim.parents):
        # check if parent is existent
        if parent != -1:
            edges[count, 0] = parent 
            edges[count, 1] = i
            count += 1

    # generate skeleton offsets, shape [1,J,3]       
    offsets = torch.tensor(np.tile(anim.offsets[np.newaxis,...],(2,16,1,1))).float()

    input_path = './data/LaFan/Train/'
    input_path = './data/LaFan/Train/'
    # data = MotionData(input_path, recalculate_mean_var=False, normalize_data=False, device='cuda:0')
    temporaldata = TemporalMotionData(input_path, recalculate_mean_var=False, normalize_data=False)
    temporaldata_norm = TemporalMotionData(input_path, recalculate_mean_var=False, normalize_data=True)

    # batch_size = 2
    # some_motion = data[20]
    randind = np.random.randint(0, len(temporaldata)-2)
    motion = temporaldata[randind:randind+2]
    motion_norm = temporaldata_norm[randind:randind+2]
    # print(motion_norm[1,:,0])
    motion_denorm = temporaldata_norm.denormalize(motion_norm)
    # print(motion[1,:,0])
    # print(motion[0,:,0] - motion_denorm[0,:,0])

    
    motion = motion.permute(0,2,1).reshape(2,16,-1,4)
    motion_denorm = motion_denorm.permute(0,2,1).reshape(2,16,-1,4)
    
    # res has the shape [batch_size, (rotations+glob_pos+1)]
    motion_rotation = motion[:,:,:-1,:]
    motion_denorm_rotation = motion_denorm[:,:,:-1,:]
    
    # apply fk on generated pos, this will be the 'fake' data in the discriminator
    _, real_pos = lafan_utils.quat_fk(motion_rotation,
                                        offsets,
                                        anim.parents)
    _, real_pos_denorm = lafan_utils.quat_fk(motion_denorm_rotation,
                                        offsets,
                                        anim.parents)

    limits = [[100,200],[500,600],[0,100]]
    # skeleton_plot(real_pos[0,0,:,:].cpu().detach(), edges, 'b')
    # skeleton_plot(real_pos_denorm[0,0,:,:].cpu().detach(), edges, 'b',limits=limits)
    two_skeleton_plot(real_pos[0,0,:,:].cpu().detach(), real_pos[0,1,:,:].cpu().detach(),
                    edges, edges, 'b', 'r' )

        
