# Some standard imports
import os
import numpy as np

from mlagents.torch_utils.torch import torch, default_device, set_torch_config
from mlagents_envs.registry import default_registry
from mlagents.trainers.settings import TorchSettings
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents.plugins.dataset.skeleton_side_channel import Skeleton_SideChannel
from mlagents.plugins.ppo.running_mean import RunningMeanStd

from mlagents.plugins.ppo.PPO import ActorCritic
from options import get_options

from mlagents.trainers.torch.model_serialization import ModelSerializer


set_torch_config(TorchSettings(device='cpu'))
print(default_device())
options = get_options("config/LaFanWalkerLegs.json")

script_dir = os.path.dirname(__file__)
model_path = '/models/LaFanLineLegs_08_09_2021-00_15/LafanLine_ep_200_AC.tar'
export_path = "/models/export/"

model_path = script_dir + model_path
export_path = script_dir + export_path

print(script_dir)
print(model_path)
print(export_path)

if not os.path.isdir(export_path):
    os.mkdir(export_path)

export_path = export_path + str.split(model_path,'/')[-1]
export_path = export_path.replace('.tar', '.onnx')


print(export_path)
# filename = None enables to communicate directly with the unity editor
env_file = "/envs/legs/" 
env_file = os.path.dirname(os.path.abspath(__file__)) + env_file
# env_file = None
if env_file is None:
    print("Env file is null, press play on the Editor to start the training.")

# initialize the environment
sk_side_channel = Skeleton_SideChannel()
eng_side_channel = EngineConfigurationChannel()
side_channels = {"skeleton":sk_side_channel, "engine":eng_side_channel}
env = UnityEnvironment(file_name=env_file, seed=1, side_channels=[sk_side_channel, eng_side_channel])
env.reset()

# env_id = '3DBall'
# env = default_registry[env_id].make()
# env.reset()

# get environment informations such as state and action dimensions
behavior_name = list(env.behavior_specs)[0]
spec = env.behavior_specs[behavior_name]
state_dim = spec.observation_specs[0].shape[0]
action_dim = len(spec.action_spec)
has_continuous_action_space = options["continuous_space"]
running_mean_std = RunningMeanStd(shape=state_dim)

# initialize a PPO agent
policy = ActorCritic(env.behavior_specs, options, running_mean_std)
checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)

action_std = checkpoint['action_std']
policy.load_state_dict(checkpoint['policy_state_dict'])
policy.running_mean_std.mean = checkpoint['running_mean']
policy.running_mean_std.std = checkpoint['running_std']
policy.set_action_std(action_std)

decision_steps, terminal_steps = env.get_steps(behavior_name)
x = torch.rand((len(decision_steps), state_dim)).float()

# Export using mlagent onnx serializer, which formats the onnx correctly
# for unity import
mod_serializer = ModelSerializer(policy)
mod_serializer.export_policy_model(export_path)