import matplotlib.pyplot as plt
from matplotlib import collections as mc
from mpl_toolkits.mplot3d import art3d     
import numpy as np

from mlagents.torch_utils import torch, default_device
from skeleton import SkeletonConv, SkeletonPool, SkeletonUnpool, SkeletonLinear
from autoencoder import Encoder, StaticEncoder, Decoder, AutoEncoder
from mlagents.plugins.bvh_utils import BVH_mod as BVH
from mlagents.plugins.bvh_utils.Quaternions import Quaternions 
from discriminator import Discriminator
from Kinematics import ForwardKinematics
from mlagents.plugins.bvh_utils import lafan_utils

# TODO: add a feature vector size variable

def skeleton_plot(points, edges, color):

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    # order is x,z,y
    xzy = [0,2,1]
    points[:,2] = -points[:,2]
    ax.scatter(points[:,0],points[:,2],points[:,1] ,c=color, marker='o')

    lines = np.zeros((len(edges), 2, 3))

    for i, e in enumerate(edges):
        point1 = points[e[0],:]
        point2 = points[e[1],:]
        lines[i,0,:] = point1[xzy]
        lines[i,1,:] = point2[xzy]

        # print(x,y,z)
        # ax.plot(x,y,z)
    lc = art3d.Line3DCollection(lines, linewidths=2)
    ax.add_collection(lc)
    ax.set_zlim(-20,100)
    ax.set_xlim(-50,50)
    ax.set_ylim(-40,40)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()


if __name__=="__main__":
    file_path = "./data/Aj/walking.bvh"
    # file_path2 = "./data/Aj/walking.bvh"
    anim, names, frametime = BVH.load(file_path, need_quater = False)

    batch_size = 1

    # anim rotation shape : [motion_lenght, num_joints, 3]
    # convert euler rotations to quaternions and concatenate them
    rotations = anim.rotations[:batch_size,:,:]
    glob_position = anim.positions[:batch_size, 0, :]
    rotations = Quaternions.from_euler(np.radians(rotations)).qs

    print("rotation shape : {}, glob pos shape : {}".format(rotations.shape, glob_position.shape))

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


    # need to invert the edge column order for it to work in pooling
    # edges[:,[0,1]] = edges[:,[1,0]]

    # generate edge matrix by going through the edges and adding a one accordingly
    edge_mat = np.eye(num_joints, dtype=int)
    for edge in edges:
        edge_mat[edge, edge[::-1]] = 1

    print(edges.shape)

    # reshape rotations so that we can add the index
    index = []
    for e in edges:
        print(e[0])
        index.append(e[0])
    rotations = rotations[:,index, :]

    
    print(rotations.shape)

    # concatenate the rotations and positions + one 0 for padding
    # rotations_cat = np.concatenate(rotations,axis=1)
    rotations_cat = rotations.reshape(batch_size, -1)

    print(rotations_cat.shape)

    rotations_cat = np.concatenate([rotations_cat, glob_position.reshape(batch_size, -1), np.zeros((batch_size,1))], axis=1)

    print(rotations_cat.shape)

    # create a neighbour list
    neighbour_list = []
    for i in range(num_joints):
        neighbour_list.append(np.where(edge_mat[i] == 1)[0]) 

    # Generate test layers of skeleton operations
    conv_layer = SkeletonConv(neighbour_list, 23*4, 23*8, 23, bias=True, add_offset=False, in_offset_channel=0)
    conv_offset_layer = SkeletonConv(neighbour_list, 23*4, 23*8, 23, bias=True, add_offset=True, in_offset_channel=3)
    pool_layer = SkeletonPool( edges, 'mean', 8, last_pool=False)
    unpool_layer = SkeletonUnpool( pool_layer.pooling_list, 8)
    linear_layer = SkeletonLinear( neighbour_list, 23*3, 23*8)

    rotations_cat = torch.tensor(rotations_cat, requires_grad=False).float()
    print("[INPUT] input shape :", rotations_cat.shape)
    result = conv_layer.forward(rotations_cat)
    print("[CONV] conv shape :",result.shape)

    result_off = conv_layer.forward(rotations_cat)
    print("[CONV] conv offset shape :",result.shape)

    result = pool_layer.forward(result)
    print("[POOL] pool shape :",result.shape)

    offset_concat  = np.concatenate(anim.offsets, axis=0)[np.newaxis,:] 
    linear_res = linear_layer.forward(torch.tensor(offset_concat).float()) 
    print("[LINEAR] offset shape : ",offset_concat.shape)


    result = unpool_layer.forward(result)
    print("[UNPOOL] unpool shape :",result.shape)

    # test encoders and decoders
    staticEncoder = StaticEncoder(edges)
    offsets = staticEncoder.forward(torch.tensor(anim.offsets[np.newaxis, :,:]).float())
    for offset in offsets:
        print(offset.shape)

    encoder = Encoder(edges)
    enc_output = encoder.forward(rotations_cat, offsets)
    print("[ENCODER] encoder output shape: ", enc_output.shape)

    decoder = Decoder(encoder)
    dec_output = decoder.forward(enc_output, offsets)
    print("[DECODER] decoder shape:", dec_output.shape)

    autoencoder = AutoEncoder(edges)
    latent,results = autoencoder.forward(rotations_cat, offsets)
    print("[AUTO] latent shape : {}, result shape: {}".format(latent.shape, results.shape))
    
    print(results[:,:-1, np.newaxis].shape)
    print(np.tile(anim.offsets[np.newaxis,...], (2,1,1)).shape)
    args=0
    # fk = ForwardKinematics(args,edges)
    # fake_pos = fk.forward_from_raw(results[:,:-1, np.newaxis],torch.tensor(np.tile(anim.offsets[np.newaxis,...], (2,1,1))).float())

    # print("[FK] fake pos result shape : ", fake_pos.shape)

    # real_pos = fk.forward_from_raw(rotations_cat[:,:-1, np.newaxis],torch.tensor(np.tile(anim.offsets[np.newaxis,...], (2,1,1))).float())
    
    # skeleton_plot(real_pos[1].cpu(),edges)

    # rotations_cat.numpy()
    print(results.reshape(batch_size,-1,4).shape)

    print(results[:,:-4].reshape(batch_size,-1,4).shape)
    
    # LAFAN KINEMATICS
    _, real_pos = lafan_utils.quat_fk(rotations_cat[:,:-4].reshape(batch_size,-1,4),
                                     torch.tensor(np.tile(anim.offsets, (batch_size,1,1))).float(),
                                     anim.parents)
    _, fake_pos = lafan_utils.quat_fk(results[:,:-4].reshape(batch_size,-1,4),
                                      torch.tensor(np.tile(anim.offsets, (batch_size,1,1))).float(),
                                      anim.parents)


    # print(real_pos)
    skeleton_plot(real_pos[0].cpu().detach(),edges,'b')

    print(fake_pos.shape)
    discrim = Discriminator(edges)
    discrim_out = discrim.forward(fake_pos)
    print("[DISCRIM] discriminator output shape: ", discrim_out.shape)
    print("[DISCRIM] Output : ", discrim_out)