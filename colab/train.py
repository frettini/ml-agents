import time
import os

from mlagents_envs.registry import default_registry
from mlagents.trainers.settings import TorchSettings
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel

from mlagents.torch_utils import default_device, set_torch_config
from mlagents.plugins.dataset.skeleton_side_channel import Skeleton_SideChannel
from mlagents.plugins.ppo.trainer import AMPTainer

import mlagents.plugins.utils.logger as log
from options import get_options

def paths_setup(model_path, log_path, behaviour_name):
    t = time.localtime()
    prt_time = time.strftime("%d_%m_%Y-%H_%M", t)
    behaviour_name = str.split(behaviour_name, '?')[0]

    # log path
    log_dir = log_path + '{}_{}/'.format(behaviour_name, prt_time)
    log_dir = os.path.dirname(os.path.abspath(__file__)) + log_dir
    log_dir = os.path.join(os.getcwd(),log_dir)

     # model save path
    model_dir = model_path + '{}_{}/'.format(behaviour_name, prt_time)
    model_dir = os.path.dirname(os.path.abspath(__file__)) + model_dir
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir.replace('/','\\'))

    print("Logging at : ", log_dir)
    print("Model saving at : ", model_dir)

    return  model_dir, log_dir


if __name__ == "__main__":

    # initialize torch to be cpu only
    # usually faster for reinforcement learning problems
    set_torch_config(TorchSettings(device='cpu'))
    options = get_options('/config/LaFanWalker.json')
    print("Default device : ", default_device())

    # filename = None enables to communicate directly with the unity editor
    env_file = "/myenv/" 
    env_file = os.path.dirname(os.path.abspath(__file__)) + env_file
    env_file = None
    if env_file is None:
        print("Env file is null, press play on the Editor to start the training.")

    # initialize the environment
    sk_side_channel = Skeleton_SideChannel()
    eng_side_channel = EngineConfigurationChannel()
    side_channels = {"skeleton":sk_side_channel, "engine":eng_side_channel}
    env = UnityEnvironment(file_name=env_file, seed=1, side_channels=[sk_side_channel, eng_side_channel])
    env.reset()

    # initialize model save path and logging path 
    model_path = "/models/"
    log_path = "/runs/beans/addedhiprotation/"
    model_dir, log_dir = paths_setup(model_path, log_path, list(env.behavior_specs)[0])
    log.init(log_dir)

    # log the hyperparameters for future reference
    log.writer.add_text("Hyperparameters", str(options))

    # initialize the trainer, which in turn initializes the policies, discrim etc.. 
    # motion_path = "C:/Users/nicol/Work/Master/dissertation/ml-agents/colab/data/"
    motion_path = os.path.dirname(os.path.abspath(__file__)) + "/data/"
    trainer = AMPTainer(env,options,motion_path,side_channels, model_dir)
    
    # start the training loop
    trainer.train()

    env.close()