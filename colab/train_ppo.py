from datetime import datetime
import time

import numpy as np
import mlagents
from mlagents_envs.registry import default_registry
from mlagents_envs.environment import ActionTuple
from mlagents.trainers.settings import TorchSettings

from mlagents.torch_utils import torch, default_device, set_torch_config
from mlagents.plugins.ppo.PPO import PPO, RolloutBuffer

from torch.utils.tensorboard import SummaryWriter


set_torch_config(TorchSettings(device='cpu'))
print(default_device())

env_id = '3DBall'

try:
    env.close()
except:
    pass

env = default_registry[env_id].make()
env.reset()

has_continuous_action_space = True  # continuous action space; else discrete

max_ep_len = 1000                   # max timesteps in one episode
max_training_timesteps = int(3e6)   # break training loop if timeteps > max_training_timesteps

print_freq = max_ep_len /10        # print avg reward in the interval (in num timesteps)
log_freq = max_ep_len * 2           # log avg reward in the interval (in num timesteps)
save_model_freq = int(1e5)          # save model frequency (in num timesteps)

action_std = 0.6                    # starting std for action distribution (Multivariate Normal)
action_std_decay_rate = 0.05        # linearly decay action_std (action_std = action_std - action_std_decay_rate)
min_action_std = 0.1                # minimum action_std (stop decay after action_std <= min_action_std)
action_std_decay_freq = int(2.5e5)  # action_std decay frequency (in num timesteps)

update_timestep = max_ep_len * 4      # update policy every n timesteps
K_epochs = 80               # update policy for K epochs in one PPO update

eps_clip = 0.2          # clip parameter for PPO
gamma = 0.99            # discount factor

lr_actor = 0.0003       # learning rate for actor network
lr_critic = 0.001       # learning rate for critic network

random_seed = 0         # set random seed if required (0 = no random seed)


def train(env, options):

    env.reset()
    
    # get the first behavior name and its spec
    behavior_name = list(env.behavior_specs)[0]
    spec = env.behavior_specs[behavior_name]

    # state space dimension
    state_dim = spec.observation_specs[0].shape[0]

    # action space dimension
    action_dim = len(spec.action_spec)

    # setup logger
    t = time.localtime()
    log_dir = './runs/beans/first_logs_{}/'.format(time.strftime("%d_%M_%Y-%H_%M", t))
    writer = SummaryWriter(log_dir)


    run_num_pretrained = 0      #### change this to prevent overwriting weights in same env_name folder

    hyperparam_log = """
    max training timesteps : {}, max timesteps per episode : {}, model saving frequency : {}, state space dimension : {}, action space dimension : {}, action_std : {}, action decay rate {}, min action std : {}, decay frequency : {}, PPO update freq : {}, PPO K epochs : {}, Epsilon clip : {}, gamma : {}, Lr actor : {}, Lr Critic : {}
    """.format(max_training_timesteps, max_ep_len, save_model_freq, state_dim, action_dim,  action_std, action_std_decay_rate, min_action_std, action_std_decay_freq, update_timestep, K_epochs, eps_clip, gamma, lr_actor, lr_critic)


    # initialize a PPO agent
    ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std)


    # track total training time
    start_time = datetime.now().replace(microsecond=0)

    # printing and logging variables
    print_running_reward = 0
    print_running_episodes = 1

    log_running_reward = 0
    log_running_episodes = 0

    time_step = 0
    i_episode = 0
    
    # training loop
    while time_step <= max_training_timesteps:

        state = env.reset()
        current_ep_reward = 0
        print("time step:", time_step)

        #TODO: initialize with empty state
        for t in range(1, max_ep_len+1):
            
            decision_steps, terminal_steps = env.get_steps(behavior_name)
    
            if 0 in terminal_steps.agent_id:
                done = True

                index = terminal_steps.agent_id_to_index[0]
                reward = terminal_steps[index].reward
                ppo_agent.buffer.rewards.append(reward)
                ppo_agent.buffer.is_terminals.append(done)

                obs = terminal_steps[index].obs[0]
                action = ppo_agent.select_action(obs)

            env.set_actions(behavior_name, spec.action_spec.empty_action(len(decision_steps)))

            if 0 in decision_steps.agent_id:
                done = False

                index = decision_steps.agent_id_to_index[0]
                reward = decision_steps[index].reward
                ppo_agent.buffer.rewards.append(reward)
                ppo_agent.buffer.is_terminals.append(done)

                # get action
                obs = decision_steps[index].obs[0]
                action = ppo_agent.select_action(obs)
                action_tuple = ActionTuple()
                action_tuple.add_continuous(action[np.newaxis,:])

                env.set_action_for_agent(behavior_name, 0, action_tuple)


            env.step()

            time_step +=1
            current_ep_reward += reward

            # update PPO agent
            if time_step % update_timestep == 0:
                print("Update PPO")
                ppo_agent.update()

            # if continuous action space; then decay action std of ouput action distribution
            if has_continuous_action_space and time_step % action_std_decay_freq == 0:
                ppo_agent.decay_action_std(action_std_decay_rate, min_action_std)

            # log in logging file
            if time_step % log_freq == 0:

                # log average reward till last episode
                log_avg_reward = log_running_reward / log_running_episodes
                log_avg_reward = round(log_avg_reward, 4)

                log_running_reward = 0
                log_running_episodes = 0

            # printing average reward
            if time_step % print_freq == 0:

                # print average reward till last episode
                print_avg_reward = print_running_reward #/ print_running_episodes
                print_avg_reward = round(print_avg_reward, 2)

                print("Episode : {} \t Timestep : {} \t Current Ep Reward : {} \t Average Reward : {}".format(i_episode, time_step, current_ep_reward, print_running_reward/print_running_episodes))

                print_running_reward = 0
                print_running_episodes = 0


        print_running_reward += current_ep_reward
        print_running_episodes += 1

        log_running_reward += current_ep_reward
        log_running_episodes += 1

        i_episode += 1

    # env.close()


    # print total training time
    end_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    print("Finished training at (GMT) : ", end_time)
    print("Total training time  : ", end_time - start_time)
  

train(env, 0)