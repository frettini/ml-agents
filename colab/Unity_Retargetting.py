import os
import time

from mlagents.torch_utils import torch, set_torch_config, default_device
from mlagents.trainers.settings import TorchSettings

from mlagents.plugins.skeleton_aware_op.sk_trainer import Sk_Trainer
from options import get_options

if __name__ == "__main__":

    # initialize torch to be cpu only
    # usually faster for reinforcement learning problems
    set_torch_config(TorchSettings(device='cpu'))
    options = get_options('/config/LaFanWalker.json')
    print("Default device : ", default_device())

    # t = time.localtime()
    # prt_time = time.strftime("%d_%m_%Y-%H_%M", t)

    # input_name = "Lafan"
    # output_name = "Lafan"
    # # log path
    # log_path = "/skamr/beans/discrim_buffer/"
    # log_dir = log_path + '{}_to_{}_{}/'.format(input_name, output_name, prt_time)
    # log_dir = os.path.dirname(os.path.abspath(__file__)) + log_dir
    # log_dir = os.path.join(os.getcwd(),log_dir)
    
    motion_path = os.path.dirname(os.path.abspath(__file__)) + "/data/"

    trainer = Sk_Trainer(options, motion_path, motion_path)

    trainer.train()