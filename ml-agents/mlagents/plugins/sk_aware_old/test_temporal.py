from torch.functional import norm
import numpy as np

from mlagents.torch_utils import torch
from autoencoder_temporal import Encoder, Decoder, AE, StaticEncoder
from discriminator_temporal import Discriminator
from skeleton_temporal import SkeletonConv, SkeletonPool, SkeletonUnpool, SkeletonLinear
from mlagents.plugins.bvh_utils.visualize import skeleton_plot
from mlagents.plugins.bvh_utils import BVH_mod as BVH
from mlagents.plugins.bvh_utils import lafan_utils
from dataset import TemporalMotionData
from Kinematics import ForwardKinematics

if __name__ == "__main__":
    # get data 
    input_path = './data/Aj'
    data = TemporalMotionData(input_path, normalize_data=False)
    motion = data[0:2]
    print('input motion: ', motion.shape)
    # init edges and offset
    # skeleton information : topology and offsets 
    file_path = "./data/Aj/walking.bvh"
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
            
    # extract end effectors for autoencoder loss
    # TODO: make the loss tolerant to different shapes and sizes 
    ee_id, j_count = torch.unique(edges.reshape(-1), return_counts = True)
    ee_id = ee_id[j_count==1]

    # generate skeleton offsets, shape [1,J,3]       
    offset_o = torch.tensor(anim.offsets[np.newaxis, :,:]).float()

    encoder = Encoder(edges)
    decoder = Decoder(encoder)
    auto_encoder = AE(edges)
    static_encoder = StaticEncoder(edges)
    discrim = Discriminator(edges)

    offsets = static_encoder.forward(torch.tensor(anim.offsets[np.newaxis, :,:]).float())
    for offset in offsets:
        print(offset.shape)

    enc_output = encoder.forward(motion, offsets)
    dec_output = decoder.forward(enc_output, offsets)
    latent,results = auto_encoder.forward(motion, offsets)

    print('enc_output : \n', enc_output.shape)
    print('dec_output : \n', dec_output.shape)
    print('results : \n', results.shape )

    fk = ForwardKinematics(edges)
    fake_pos = fk.forward_from_raw(results[:,:-1,:], offset_o)
    # real_pos = fk.forward_from_raw(motion[:,:-1,:], offset_o)

    results = results.permute(0,2,1).reshape(2,16,23,4)
    motion_reshaped = motion.permute(0,2,1).reshape(2,16,23,4)

    print(results.shape)
    print(motion_reshaped.shape)

    _, real_pos = lafan_utils.quat_fk(motion_reshaped[:,:,:-1,:],
                                     torch.tensor(np.tile(anim.offsets[np.newaxis,...], (2,16,1,1))).float(),
                                     anim.parents)
    _, fake_pos = lafan_utils.quat_fk(results[:,:,:-1,:],
                                     torch.tensor(np.tile(anim.offsets[np.newaxis,...], (2,16,1,1))).float(),
                                     anim.parents)

    print(fake_pos.shape)
    # skeleton_plot(fake_pos[0,0,:,:].cpu().detach(), edges, 'b')
    
    print('normalize : ', torch.nn.functional.normalize(results[:,:,:-1,:], dim=3).shape)

    out_discrim = discrim.forward(fake_pos)

    print(out_discrim)
