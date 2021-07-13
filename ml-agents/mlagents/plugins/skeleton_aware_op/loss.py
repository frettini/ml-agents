from mlagents.plugins.skeleton_aware_op.dataset import SkeletonInfo
from mlagents.torch_utils import torch, default_device

def calc_chain_velo_loss(real_joint_vel, fake_joint_vel, chain_indices, criterion_velo, normalize_velo = False):
    """
    return the loss of the mean velocity between each chain
    """
    # Get the loss that tries to match the total velocity of every corresponding chain 
    loss_chain_velo = 0

    for ind in range(len(chain_indices)):
        # sum all the velocities together 
        real_joint_vel_sum = torch.sum(real_joint_vel[:,:,chain_indices[ind],:], dim=2)
        fake_joint_vel_sum = torch.sum(fake_joint_vel[:,:,chain_indices[ind],:], dim=2) # TODO: change to the input skeleton

        if normalize_velo is True:
            real_joint_vel_sum = torch.nn.functional.normalize(real_joint_vel_sum, dim=2)
            fake_joint_vel_sum = torch.nn.functional.normalize(fake_joint_vel_sum, dim=2)

        loss_chain_velo += criterion_velo(fake_joint_vel_sum, real_joint_vel_sum)

    loss_chain_velo /= len(chain_indices)

    return loss_chain_velo


def calc_ee_loss(real_vel, real_pos, fake_vel, fake_pos, criterion_ee, skdata_real : SkeletonInfo, skdata_fake : SkeletonInfo):
    """
    Calculate the end effector loss using the fake velocity and position (local or global)
    """
    # [batch, frame, n_end_effector, 6]
    ee_false = skdata_fake.ee_id[skdata_fake.ee_id != 0]
    ee_real = skdata_real.ee_id[skdata_real.ee_id != 0]

    # print(fake_pos[:,:,0:1,:].shape)
    fake_pos_loc = fake_pos - fake_pos[:,:,0:1,:]
    real_pos_loc = real_pos - real_pos[:,:,0:1,:]

    fake_ee = torch.cat((fake_vel[:,:,ee_false,:], fake_pos_loc[:,:,ee_false,:]), dim=2)
    real_ee = torch.cat((real_vel[:,:,ee_real,:], real_pos_loc[:,:,ee_real,:]), dim=2)

    real_length = skdata_real.ee_length.reshape(1,1,skdata_real.ee_length.shape[0],1)
    fake_length = skdata_fake.ee_length.reshape(1,1,skdata_fake.ee_length.shape[0],1)

    # print(torch.repeat_interleave(real_length,2, dim=2))
    real_length = torch.repeat_interleave(real_length,2, dim=2)
    fake_length = torch.repeat_interleave(fake_length,2, dim=2)


    # loss_ee_velo = criterion_ee(fake_joint_vel[:,:,ee_id,:], real_joint_vel[:,:,ee_id,:])
    loss_ee_velo = criterion_ee(fake_ee/fake_length, real_ee/real_length)

    return loss_ee_velo