from mlagents.torch_utils import torch, default_device
import time

from mlagents.plugins.skeleton_aware_op.dataset import  SkeletonInfo, TemporalMotionData
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
        pass

    def train(self):
        pass

    def retarget(self, input):
        """
        Responsible of taking a valid input and generating a valid output
        """
        pass