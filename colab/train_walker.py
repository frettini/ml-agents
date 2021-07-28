import time
import os

from mlagents_envs.registry import default_registry
from mlagents.trainers.settings import TorchSettings
from mlagents_envs.environment import UnityEnvironment

from mlagents.torch_utils import default_device, set_torch_config
from mlagents.plugins.ppo.PPO import PPO

import mlagents.plugins.utils.logger as log
from options import get_options


# initialize logger which then becomes global to the rest of the project
t = time.localtime()
log_dir = 'runs/beans/lafan_{}/'.format(time.strftime("%d_%m_%Y-%H_%M", t))
log_dir = os.path.join(os.getcwd(),log_dir)
print("Logging at : ", log_dir)
log.init(log_dir)


# initialize torch to be cpu only
# usually faster for reinforcement learning problems
set_torch_config(TorchSettings(device='cpu'))
options = get_options('LaFanWalker.json')
log.writer.add_text("Hyperparameters", str(options))
print("Default device : ", default_device())


# model save path
save_path = '/models/Walker_{}/'.format(time.strftime("%d_%m_%Y-%H_%M", t))
save_path = os.path.dirname(__file__) + save_path
if not os.path.isdir(save_path):
    os.mkdir(save_path)


# filename = None enables to communicate directly with the unity editor
env_file = None
if env_file is None:
    print("Env file is null, press play on the Editor to start the training.")
# initialize the environment
env = UnityEnvironment(file_name=None, seed=1, side_channels=[])
env.reset()


# initialize a PPO agent
ppo_agent = PPO(env.behavior_specs, options)

# MAIN PPO LOOP
epoch = 0
while(ppo_agent.cumulated_training_steps < options["max_training_steps"]):

    print("Ep {} :".format(epoch))

    # step through the simulation and gather the trajectory
    ppo_agent.get_trajectory(env)

    # update the policy using the collected buffer
    ppo_agent.batch_update()

    print("Ep {} finished with cumulated reward {} \t average reward {}".format(epoch, ppo_agent.cumulated_reward, ppo_agent.cumulated_reward/options["buffer_size"]))
    log.writer.add_scalar("Train/Cumulated_Reward",ppo_agent.cumulated_reward, ppo_agent.cumulated_training_steps)
    log.writer.add_scalar("Train/Average_Reward",ppo_agent.cumulated_reward/options["buffer_size"], ppo_agent.cumulated_training_steps)
    
    ppo_agent.cumulated_reward = 0

    epoch += 1
    
    if(epoch % 5 == 0):
        ppo_agent.save(save_path + "LafanWalker_ep_{}.pt".format(epoch))

ppo_agent.save(save_path + "LafanWalker_complete.pt")
env.close()