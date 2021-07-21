# Some standard imports
import os
import numpy as np

from mlagents.torch_utils.torch import torch, default_device, set_torch_config
from mlagents_envs.registry import default_registry
from mlagents.trainers.settings import TorchSettings

from mlagents.plugins.ppo.PPO import ActorCritic
from options import get_options

from mlagents.trainers.torch.model_serialization import ModelSerializer


set_torch_config(TorchSettings(device='cpu'))
print(default_device())
options = get_options()

script_dir = os.path.dirname(__file__)
model_path = '/models/3DBall_21_07_2021-10_08/3Dball_complete.pt'
export_path = "/models/export/"

model_path = script_dir + model_path
export_path = script_dir + export_path

print(script_dir)
print(model_path)
print(export_path)

if not os.path.isdir(export_path):
    os.mkdir(export_path)

export_path = export_path + str.split(model_path,'/')[-1]
export_path = export_path.replace('.pt', '.onnx')


print(export_path)

env_id = '3DBall'
env = default_registry[env_id].make()
env.reset()

# get environment informations such as state and action dimensions
behavior_name = list(env.behavior_specs)[0]
spec = env.behavior_specs[behavior_name]
state_dim = spec.observation_specs[0].shape[0]
action_dim = len(spec.action_spec)
has_continuous_action_space = options["continuous_space"]

# initialize a PPO agent
policy = ActorCritic(env.behavior_specs, has_continuous_action_space, options["action_std"])
policy.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))

decision_steps, terminal_steps = env.get_steps(behavior_name)
x = torch.rand((len(decision_steps), state_dim)).float()

# Export using mlagent onnx serializer, which formats the onnx correctly
# for unity import
mod_serializer = ModelSerializer(policy)
mod_serializer.export_policy_model(export_path)