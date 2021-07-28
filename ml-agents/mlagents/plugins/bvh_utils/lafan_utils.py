import numpy as np
from mlagents.torch_utils import torch, default_device

def length(x, axis=-1, keepdims=True):
    """
    Computes vector norm along a tensor axis(axes)
    :param x: tensor
    :param axis: axis(axes) along which to compute the norm
    :param keepdims: indicates if the dimension(s) on axis should be kept
    :return: The length or vector of lengths.
    """
    lgth = torch.sqrt(torch.sum(x * x, dim=axis, keepdims=keepdims))
    return lgth


def normalize(x, axis=-1, eps=1e-8):
    """
    Normalizes a tensor over some axis (axes)
    :param x: data tensor
    :param axis: axis(axes) along which to compute the norm
    :param eps: epsilon to prevent numerical instabilities
    :return: The normalized tensor
    """
    res = x / (length(x, axis=axis) + eps)
    return res


def quat_normalize(x, eps=1e-8):
    """
    Normalizes a quaternion tensor
    :param x: data tensor
    :param eps: epsilon to prevent numerical instabilities
    :return: The normalized quaternions tensor
    """
    res = normalize(x, eps=eps)
    return res


def angle_axis_to_quat(angle, axis):
    """
    Converts from and angle-axis representation to a quaternion representation
    :param angle: angles tensor
    :param axis: axis tensor
    :return: quaternion tensor
    """
    c = np.cos(angle / 2.0)[..., np.newaxis]
    s = np.sin(angle / 2.0)[..., np.newaxis]
    q = np.concatenate([c, s * axis], axis=-1)
    return q


def euler_to_quat(e, order='zyx'):
    """
    Converts from an euler representation to a quaternion representation
    :param e: euler tensor
    :param order: order of euler rotations
    :return: quaternion tensor
    """
    axis = {
        'x': np.asarray([1, 0, 0], dtype=np.float32),
        'y': np.asarray([0, 1, 0], dtype=np.float32),
        'z': np.asarray([0, 0, 1], dtype=np.float32)}

    q0 = angle_axis_to_quat(e[..., 0], axis[order[0]])
    q1 = angle_axis_to_quat(e[..., 1], axis[order[1]])
    q2 = angle_axis_to_quat(e[..., 2], axis[order[2]])

    return quat_mul(q0, quat_mul(q1, q2))


def quat_inv(q):
    """
    Inverts a tensor of quaternions
    :param q: quaternion tensor
    :return: tensor of inverted quaternions
    """
    res = torch.tensor([1, -1, -1, -1]).float() * q
    return res


def quat_fk(lrot, lpos, parents):
    """
    Performs Forward Kinematics (FK) on local quaternions and local positions to retrieve global representations
    :param lrot: tensor of local quaternions with shape (..., Nb of joints, 4)
    :param lpos: tensor of local positions with shape (..., Nb of joints, 3)
    :param parents: list of parents indices
    :return: tuple of tensors of global quaternion, global positions
    """
    gp, gr = [lpos[..., :1, :]], [lrot[..., :1, :]]
    for i in range(1, len(parents)):
        gp.append(quat_mul_vec(gr[parents[i]], lpos[..., i:i+1, :]) + gp[parents[i]])
        gr.append(quat_mul    (gr[parents[i]], lrot[..., i:i+1, :]))

    res = torch.cat(gr, dim=-2), torch.cat(gp, dim=-2)
    return res


def quat_ik(grot, gpos, parents):
    """
    Performs Inverse Kinematics (IK) on global quaternions and global positions to retrieve local representations
    :param grot: tensor of global quaternions with shape (..., Nb of joints, 4)
    :param gpos: tensor of global positions with shape (..., Nb of joints, 3)
    :param parents: list of parents indices
    :return: tuple of tensors of local quaternion, local positions
    """
    res = [
        torch.cat([
            grot[..., :1, :],
            quat_mul(quat_inv(grot[..., parents[1:], :]), grot[..., 1:, :]),
        ], dim=-2),
        torch.cat([
            gpos[..., :1, :],
            quat_mul_vec(
                quat_inv(grot[..., parents[1:], :]),
                gpos[..., 1:, :] - gpos[..., parents[1:], :]),
        ], dim=-2)
    ]

    return res


def quat_mul(x, y):
    """
    Performs quaternion multiplication on arrays of quaternions
    :param x: tensor of quaternions of shape (..., Nb of joints, 4)
    :param y: tensor of quaternions of shape (..., Nb of joints, 4)
    :return: The resulting quaternions
    """
    x0, x1, x2, x3 = x[..., 0:1], x[..., 1:2], x[..., 2:3], x[..., 3:4]
    y0, y1, y2, y3 = y[..., 0:1], y[..., 1:2], y[..., 2:3], y[..., 3:4]

    res = torch.cat([
        y0 * x0 - y1 * x1 - y2 * x2 - y3 * x3,
        y0 * x1 + y1 * x0 - y2 * x3 + y3 * x2,
        y0 * x2 + y1 * x3 + y2 * x0 - y3 * x1,
        y0 * x3 - y1 * x2 + y2 * x1 + y3 * x0], dim=-1)

    return res


def quat_mul_vec(q, x):
    """
    Performs multiplication of an array of 3D vectors by an array of quaternions (rotation).
    :param q: tensor of quaternions of shape (..., Nb of joints, 4)
    :param x: tensor of vectors of shape (..., Nb of joints, 3)
    :return: the resulting array of rotated vectors
    """
    t = 2.0 * torch.cross(q[..., 1:].float(), x, dim=-1)
    res = x + q[..., 0][..., np.newaxis] * t + torch.cross(q[..., 1:].float(), t, dim=-1)

    return res


def quat_slerp(x, y, a):
    """
    Perfroms spherical linear interpolation (SLERP) between x and y, with proportion a
    :param x: quaternion tensor
    :param y: quaternion tensor
    :param a: indicator (between 0 and 1) of completion of the interpolation.
    :return: tensor of interpolation results
    """
    len = np.sum(x * y, axis=-1)

    neg = len < 0.0
    len[neg] = -len[neg]
    y[neg] = -y[neg]

    a = np.zeros_like(x[..., 0]) + a
    amount0 = np.zeros(a.shape)
    amount1 = np.zeros(a.shape)

    linear = (1.0 - len) < 0.01
    omegas = np.arccos(len[~linear])
    sinoms = np.sin(omegas)

    amount0[linear] = 1.0 - a[linear]
    amount0[~linear] = np.sin((1.0 - a[~linear]) * omegas) / sinoms

    amount1[linear] = a[linear]
    amount1[~linear] = np.sin(a[~linear] * omegas) / sinoms
    res = amount0[..., np.newaxis] * x + amount1[..., np.newaxis] * y

    return res


def quat_between(x, y):
    """
    Quaternion rotations between two 3D-vector arrays
    :param x: tensor of 3D vectors
    :param y: tensor of 3D vetcors
    :return: tensor of quaternions
    """
    res = torch.cat([
        torch.sqrt(torch.sum(x * x, dim=-1) * torch.sum(y * y, dim=-1))[..., np.newaxis] +
        torch.sum(x * y, dim=-1)[..., np.newaxis],
        # torch.sqrt((length(x)**2) * (length(y)**2)) + torch.sum(x * y, dim=-1)[..., np.newaxis],
        torch.cross(x, y, dim=-1)], dim=-1)
    return res


def interpolate_local(lcl_r_mb, lcl_q_mb, n_past, n_future):
    """
    Performs interpolation between 2 frames of an animation sequence.
    The 2 frames are indirectly specified through n_past and n_future.
    SLERP is performed on the quaternions
    LERP is performed on the root's positions.
    :param lcl_r_mb:  Local/Global root positions (B, T, 1, 3)
    :param lcl_q_mb:  Local quaternions (B, T, J, 4)
    :param n_past:    Number of frames of past context
    :param n_future:  Number of frames of future context
    :return: Interpolated root and quats
    """
    # Extract last past frame and target frame
    start_lcl_r_mb = lcl_r_mb[:, n_past - 1, :, :][:, None, :, :]  # (B, 1, J, 3)
    end_lcl_r_mb = lcl_r_mb[:, -n_future, :, :][:, None, :, :]

    start_lcl_q_mb = lcl_q_mb[:, n_past - 1, :, :]
    end_lcl_q_mb = lcl_q_mb[:, -n_future, :, :]

    # LERP Local Positions:
    n_trans = lcl_r_mb.shape[1] - (n_past + n_future)
    interp_ws = np.linspace(0.0, 1.0, num=n_trans + 2, dtype=np.float32)
    offset = end_lcl_r_mb - start_lcl_r_mb

    const_trans    = np.tile(start_lcl_r_mb, [1, n_trans + 2, 1, 1])
    inter_lcl_r_mb = const_trans + (interp_ws)[None, :, None, None] * offset

    # SLERP Local Quats:
    interp_ws = np.linspace(0.0, 1.0, num=n_trans + 2, dtype=np.float32)
    inter_lcl_q_mb = np.stack(
        [(quat_normalize(quat_slerp(quat_normalize(start_lcl_q_mb), quat_normalize(end_lcl_q_mb), w))) for w in
         interp_ws], axis=1)

    return inter_lcl_r_mb, inter_lcl_q_mb


def remove_quat_discontinuities(rotations):
    """
    Removing quat discontinuities on the time dimension (removing flips)
    :param rotations: Array of quaternions of shape (T, J, 4)
    :return: The processed array without quaternion inversion.
    """
    rots_inv = -rotations

    for i in range(1, rotations.shape[0]):
        # Compare dot products
        replace_mask = np.sum(rotations[i - 1: i] * rotations[i: i + 1], axis=-1) < np.sum(
            rotations[i - 1: i] * rots_inv[i: i + 1], axis=-1)
        replace_mask = replace_mask[..., np.newaxis]
        rotations[i] = replace_mask * rots_inv[i] + (1.0 - replace_mask) * rotations[i]

    return rotations


# Orient the data according to the las past keframe
def rotate_at_frame(X, Q, parents, n_past=10):
    """
    Re-orients the animation data according to the last frame of past context.
    :param X: tensor of local positions of shape (Batchsize, Timesteps, Joints, 3)
    :param Q: tensor of local quaternions (Batchsize, Timesteps, Joints, 4)
    :param parents: list of parents' indices
    :param n_past: number of frames in the past context
    :return: The rotated positions X and quaternions Q
    """
    # Get global quats and global poses (FK)
    global_q, global_x = quat_fk(Q, X, parents)

    key_glob_Q = global_q[:, n_past - 1: n_past, 0:1, :]  # (B, 1, 1, 4)
    forward = np.array([1, 0, 1])[np.newaxis, np.newaxis, np.newaxis, :] \
                 * quat_mul_vec(key_glob_Q, np.array([0, 1, 0])[np.newaxis, np.newaxis, np.newaxis, :])
    forward = normalize(forward)
    yrot = quat_normalize(quat_between(np.array([1, 0, 0]), forward))
    new_glob_Q = quat_mul(quat_inv(yrot), global_q)
    new_glob_X = quat_mul_vec(quat_inv(yrot), global_x)

    # back to local quat-pos
    Q, X = quat_ik(new_glob_Q, new_glob_X, parents)

    return X, Q


def extract_feet_contacts(pos, lfoot_idx, rfoot_idx, velfactor=0.02):
    """
    Extracts binary tensors of feet contacts
    :param pos: tensor of global positions of shape (Timesteps, Joints, 3)
    :param lfoot_idx: indices list of left foot joints
    :param rfoot_idx: indices list of right foot joints
    :param velfactor: velocity threshold to consider a joint moving or not
    :return: binary tensors of left foot contacts and right foot contacts
    """
    lfoot_xyz = (pos[1:, lfoot_idx, :] - pos[:-1, lfoot_idx, :]) ** 2
    contacts_l = (np.sum(lfoot_xyz, axis=-1) < velfactor)

    rfoot_xyz = (pos[1:, rfoot_idx, :] - pos[:-1, rfoot_idx, :]) ** 2
    contacts_r = (np.sum(rfoot_xyz, axis=-1) < velfactor)

    # Duplicate the last frame for shape consistency
    contacts_l = np.concatenate([contacts_l, contacts_l[-1:]], axis=0)
    contacts_r = np.concatenate([contacts_r, contacts_r[-1:]], axis=0)

    return contacts_l, contacts_r

# ----------------------------------------
# MY OWN UTILS HERE
# ----------------------------------------

def get_velocity(positions, frametime):
    """
    Calculate the velocity given a set of positions and the frametime between two frames
    :param position: [n_frames, [positions]] tensor containing the positions, first dim should be number of frames
    :param frametime: float specifying the time elapsed between two frames
    :return: [n_frames, [positions]] tensor containing the calculated velocities
    """
    velocity = torch.zeros_like(positions)
    
    # get velocity by doing central difference
    for i in range(positions.shape[0]-2):
        # print(((positions[i+2] - positions[i])/(frametime*2)).shape)
        velocity[i+1, ...] = (positions[i+2] - positions[i])/(frametime*2)
    
    # boundary cases 
    velocity[0] = (positions[1] - positions[0])/frametime
    velocity[-1] = (positions[-1] - positions[-2])/frametime

    return velocity

def get_batch_velo(real_pos, fake_pos, frametime):
    """
    return a tensor same size that the input with the velocity
    :params real_pos: [batch_size, window_size, [positions]] tensor of real position
    :params fake_pos: [batch_size, window_size, [positions]] tensor of real position
    :params frametime: float time elapsed between two frames
    :return real_joint_vel: real velocity of same size as real_pose
    :return fake_joint_vel: fake velocity of same size as fake_pose
    """
    
    # get the velocity at every joint
    real_joint_vel = torch.zeros_like(real_pos)
    fake_joint_vel = torch.zeros_like(fake_pos)

    for ind in range(real_pos.shape[0]):
        real_joint_vel[ind] = get_velocity(real_pos[ind], frametime)

    for ind in range(fake_pos.shape[0]):
        fake_joint_vel[ind] = get_velocity(fake_pos[ind], frametime)

    return real_joint_vel, fake_joint_vel

def get_batch_velo2(input_pos, frametime):
    """
    return a tensor same size that the input with the velocity
    :params input_pos: [batch_size, window_size, [positions]] tensor of real position
    :params frametime: float time elapsed between two frames
    :return input_joint_vel: real velocity of same size as real_pose
    """
    
    # get the velocity at every joint
    input_joint_vel = torch.zeros_like(input_pos)

    for ind in range(input_pos.shape[0]):
        input_joint_vel[ind] = get_velocity(input_pos[ind], frametime)

    return input_joint_vel

def get_global_position_from_velocity(init_position, velocity, frametime, positions = None):
    """
    Get the global position from velocity and initial position. Another set of positions 
    can be provided to which the calculated global positions will be added.
    :params init_position: [frame_size, 3] tensor initial global pos
    :params velocity: [frame_size, 3] tenosr global velocity
    :params frametime: float, time elapsed between frames
    :params positions: [frame_size, [positions]] tensor of position to which global pos is applied
    :returns glob_positions: [frame_size, 3] velocity of global positions
    :returns positions: initial vector with added global pos
    """
    glob_positions = torch.zeros_like(velocity)
    prev_pos = init_position

    for i in range(1,velocity.shape[0]):
        glob_positions[i-1,:] = prev_pos
        prev_pos = prev_pos + velocity[i,:] * frametime
    
    if positions is not None:
        positions += glob_positions.reshape(glob_positions.shape[0],1,-1)
        return glob_positions, positions
    
    return glob_positions

def get_angular_velocity(rotations, frametime):
    pass

def build_edges(parents):
    """
    Build edge array from parent transform information
    :param parents: list of indices indicating the parent joints (-1 means no parent)
    :return edges: [num_edges, 2] tensor of indices for each edge
    """
    num_joints = len(parents)
    edges = torch.zeros((num_joints-1, 2), dtype=int, requires_grad=False)

    count = 0
    for i, parent in enumerate(parents):
        # check if parent is existent
        if parent != -1:
            edges[count, 0] = parent 
            edges[count, 1] = i
            count += 1
    return edges

def build_chain_list(edges):
    """
    Build list of kinematic chains using the edges. 
    :param edges: [num_edges, 2] containing indices to connected joints
    :return: list of lists containing joints that for a chain
    """

    # get valence of each node in the graph
    degree = np.zeros(edges.shape[0]+1)
    for edge in edges:
        degree[edge[0]] += 1
        degree[edge[1]] += 1

    chain_indices = []

    # extract chains from edges
    def find_chains(j, seq, degree, edges):
        if degree[j] > 2 and j != 0:
            chain_indices.append(seq)
            seq = [j.item()]

        if degree[j] == 1:
            # print(j)
            seq.append(j.item())

            chain_indices.append(seq)
            return

        for idx, edge in enumerate(edges):
            if edge[0] == j:
                find_chains(edge[1], seq + [edge[1].item()], degree, edges)

    find_chains(0, [0], degree, edges)

    for ind, chain in enumerate(chain_indices):
        chain_indices[ind] = list(dict.fromkeys(chain))

    return chain_indices
    
def get_height(parents, offsets):
    """
    Get the height of a character calculated by finding the highest and lowest point
    in the skeleton. 
    :params parents:
    :params offsets:
    :returns height: 
    """
    low = high = 0

    joint_num = len(parents)
    def dfs(i, pos):
        nonlocal low
        nonlocal high
        low = min(low, pos[-1])
        high = max(high, pos[-1])

        for j in range(joint_num):
            if parents[j] == i:
                dfs(j, pos + offsets[j])

    dfs(0, torch.tensor([0, 0, 0]))

    return high - low

def get_pos_info_from_raw(input_data : torch.Tensor, skdata, offsets, options, norm_rot=False, rotation_offset=None):
    """
    :params input data: [batch_size, (rotations+glob_pos+1), window_size] \n
    :params skdata: SkeletonInfo object containing edge, and more information about skeleton 
    :params options: dictionary with options
    :params offsets: 2D tensor [num_joints, 3] containing the initial offsets of the skeleton 
    :params norm_rot: boolean, wether to nromalize the rotations extracted from the input_data (e.g. output of decoder)
    :rotation_offset: add a rotation offset to the local space to fit the correct rotation (1D tensor)

    return - 
    :return global_position:
    :return local_position:
    :return global_rotation:
    :return global_velocity:
    :return local_velocity:
    """ 
    curr_batch_size = input_data.shape[0]

    # transform res shape from [batch_size, (rotations+glob_pos+1), window_size]
    # to [batch_size, window_size, n_joints,4]
    input_data = input_data.permute(0,2,1).reshape(curr_batch_size, options['window_size'], -1, options['channel_base'])
    # and offsets to [batch_size, window_size, n_joints, 3]
    offsets = offsets.reshape(1, 1, skdata.offsets.shape[0], skdata.offsets.shape[1])
    offsets = offsets.repeat(curr_batch_size, options['window_size'], 1, 1)

    # extract rotation and velocity from raw
    rotation = input_data[:,:,:-1,:]
    velocity_global  = input_data[:,:,-1:,:-1]

    if norm_rot is True:
        # the current output rotation is not necessarily a normalized quaternion
        rotation = torch.nn.functional.normalize(rotation, dim=3)

    rotation_global = rotation[:,:,0,:].reshape(curr_batch_size, options['window_size'], -1)
    rotation_local = torch.clone(rotation)
    rotation_local[:,:,0,:] = torch.tensor([1,0,0,0]).float()
    
    if rotation_offset is not None:
        rotation_offset = rotation_offset.reshape(1,1,4)
        rotation_offset = rotation_offset.repeat(rotation_local.shape[0],rotation_local.shape[1],1)
        print(rotation_offset.shape)
        print(rotation_local[:,:,0,:].shape)
        rotation_local[:,:,0,:] = utils.quat_mul(rotation_offset.float(), rotation_local[:,:,0,:])

    # extract position information
    _, position_global = utils.quat_fk(rotation, offsets, skdata.parents)
    _, position_local = utils.quat_fk(rotation_local, offsets, skdata.parents)
    velocity_local = utils.get_batch_velo2(position_local, skdata.frametime)

    return position_global, position_local, rotation_global, velocity_global, velocity_local