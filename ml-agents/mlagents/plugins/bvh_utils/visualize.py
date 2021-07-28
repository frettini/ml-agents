from mlagents.plugins.bvh_utils import lafan_utils 
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from matplotlib import collections as mc
from mpl_toolkits.mplot3d import art3d     
import mpl_toolkits.mplot3d.axes3d as p3

import numpy as np

def skeleton_plot(points, edges, color, limits=None, save_filename = None):

    # if limits is None:
        # limits=[[-50,50],[-40,40],[-20,100]]

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
    lc = art3d.Line3DCollection(lines, linewidths=2, color=color)
    ax.add_collection(lc)

    if limits is not None:
        ax.set_xlim(limits[0][0],limits[0][1])
        ax.set_ylim(limits[1][0],limits[1][1])
        ax.set_zlim(limits[2][0],limits[2][1])

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    if save_filename is not None :
        plt.savefig(save_filename, bbox_inches="tight",transparent=False)

    plt.show()


def two_skeleton_plot(points_sk1, points_sk2, edges_sk1, edges_sk2, color1, color2, limits=None, save_filename = None):

    # if limits is None:
        # limits=[[-50,50],[-40,40],[-20,100]]

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    # order is x,z,y
    xzy = [0,2,1]
    points_sk1[:,2] = -points_sk1[:,2] # flip axis
    points_sk2[:,2] = -points_sk2[:,2] # flip axis

    ax.scatter(points_sk1[:,0],points_sk1[:,2],points_sk1[:,1] ,c=color1, marker='o')
    ax.scatter(points_sk2[:,0],points_sk2[:,2],points_sk2[:,1] ,c=color2, marker='o')

    lines1 = np.zeros((len(edges_sk1), 2, 3))
    lines2 = np.zeros((len(edges_sk2), 2, 3))

    for i, e in enumerate(edges_sk1):
        point1 = points_sk1[e[0],:]
        point2 = points_sk1[e[1],:]
        lines1[i,0,:] = point1[xzy]
        lines1[i,1,:] = point2[xzy]

    for i, e in enumerate(edges_sk2):
        point1 = points_sk2[e[0],:]
        point2 = points_sk2[e[1],:]
        lines2[i,0,:] = point1[xzy]
        lines2[i,1,:] = point2[xzy]  

    lc1 = art3d.Line3DCollection(lines1, linewidths=2, color=color1)
    lc2 = art3d.Line3DCollection(lines2, linewidths=2, color=color2)
    ax.add_collection(lc1)
    ax.add_collection(lc2)

    if limits is not None:
        ax.set_xlim(limits[0][0],limits[0][1])
        ax.set_ylim(limits[1][0],limits[1][1])
        ax.set_zlim(limits[2][0],limits[2][1])

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    if save_filename is not None :
        plt.savefig(save_filename, bbox_inches="tight",transparent=False)

    plt.show()

def skeletons_plot(points_list, edges_list, colors_list, limits=None, save_filename = None, return_plot=False):
    """
    Plots a graph of point in 3D. 
    This function can plot as many skeleton as needed by having points, edges, and
    colors in lists.
    
    :params points: list of 2D array ([num_joints, 3]) of position of each joint
    :params edges: list of 2D array ([num_joints-1, 2]) of indices of connected joints
    :params colors: list of character that specify the color of each skeleton
    :params limits: 2D list/array ([3,2]) of axis limits 
    """

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    for points, edges, color in zip(points_list, edges_list, colors_list):

        ax.scatter(points[:,0],points[:,1],points[:,2] ,c=color, marker='o')

        lines = np.zeros((len(edges), 2, 3))

        for i, e in enumerate(edges):
            point1 = points[e[0],:]
            point2 = points[e[1],:]
            lines[i,0,:] = point1
            lines[i,1,:] = point2

        lc = art3d.Line3DCollection(lines, linewidths=2, color=color)
        ax.add_collection(lc)

    if limits is not None:
        ax.set_xlim(limits[0][0],limits[0][1])
        ax.set_ylim(limits[1][0],limits[1][1])
        ax.set_zlim(limits[2][0],limits[2][1])

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    if save_filename is not None :
        plt.savefig(save_filename, bbox_inches="tight",transparent=False)

    if return_plot is True:
        return fig, ax
    else:
        plt.show()


def motion_animation(motion_pos1, motion_pos2, edges1, edges2, limits=None):

    fig = plt.figure()
    ax = p3.Axes3D(fig)

    motion_pos1[:,:,2] = -motion_pos1[:,:,2] # flip axis
    motion_pos2[:,:,2] = -motion_pos2[:,:,2] # flip axis
    lines1 = np.zeros((len(edges1), 2, 3))
    lines2 = np.zeros((len(edges2), 2, 3))

    xzy = [0,2,1]
    for i, e in enumerate(edges1):
        point1 = motion_pos1[0, e[0],:]
        point2 = motion_pos1[0, e[1],:]
        lines1[i,0,:] = point1[xzy]
        lines1[i,1,:] = point2[xzy]

    for i, e in enumerate(edges2):
        point1 = motion_pos2[0, e[0],:]
        point2 = motion_pos2[0, e[1],:]
        lines2[i,0,:] = point1[xzy]
        lines2[i,1,:] = point2[xzy]

    lc1 = art3d.Line3DCollection(lines1, linewidths=2, color='g')
    lc2 = art3d.Line3DCollection(lines2, linewidths=2, color='b')
    ax.add_collection3d(lc1)
    ax.add_collection3d(lc2)

    if limits is not None:
        ax.set_xlim(limits[0][0],limits[0][1])
        ax.set_ylim(limits[1][0],limits[1][1])
        ax.set_zlim(limits[2][0],limits[2][1])

    def update_lines(num):

        segments1 = np.zeros((len(edges1), 2, 3))
        segments2 = np.zeros((len(edges2), 2, 3))

        xzy = [0,2,1]
        for i, e in enumerate(edges1):
            point1 = motion_pos1[num, e[0],:]
            point2 = motion_pos1[num, e[1],:]
            segments1[i,0,:] = point1[xzy]
            segments1[i,1,:] = point2[xzy]

        for i, e in enumerate(edges2):
            point1 = motion_pos2[num, e[0],:]
            point2 = motion_pos2[num, e[1],:]
            segments2[i,0,:] = point1[xzy]
            segments2[i,1,:] = point2[xzy]

        lc1.set_segments(segments1)
        lc2.set_segments(segments2)


    anim = animation.FuncAnimation(fig, update_lines, 16, 
                               interval=30, blit=False)
    
    
    return anim



if __name__ == "__main__":

    from mlagents.plugins.skeleton_aware_op.dataset import TemporalMotionData, get_skeleton_info
    from mlagents.torch_utils import torch, default_device

    input_path =  "../skeleton_aware_op/data/LaFan/Train/"
    dataset = TemporalMotionData(input_path)
    edges, offset , ee_id, chain_list, parents, frametime = get_skeleton_info(input_path)

    motion_data = dataset[100].permute(1,0)
    motion_data = motion_data.reshape(motion_data.shape[0], -1, 4)

    rotations = torch.tensor(motion_data[:,:-1,:]).float()
    velocity = torch.tensor(motion_data[:,-1,:-1]).float()
    offsets = torch.unsqueeze(offset, dim=0)

    offsets = offsets.repeat(motion_data.shape[0],1,1)

    _, positions = lafan_utils.quat_fk(rotations, offsets, parents)

    print(offsets.shape)
    print(positions.shape)
    print(rotations.shape)
    print(velocity.shape)

    _, global_pos = lafan_utils.get_global_position_from_velocity(torch.tensor([0,0,0]), velocity, frametime, positions)

    limits = [[100,200],[500,600],[0,100]]
    motion_animation(global_pos.cpu().detach(), global_pos.cpu().detach(), edges,edges, limits)

