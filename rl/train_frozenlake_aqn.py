import gym

from rl.schedule import LinearExploration, LinearSchedule
from rl.models.AQN import AQN
from envs.MfccFrozenlake import MfccFrozenlake
from envs.AudioFrozenlake import AudioFrozenlake

from configs.train_frozenlake_aqn import config

"""
Use deep Q network for the Atari game. Please report the final result.
Feel free to change the configurations (in the configs/ folder). 
If so, please report your hyperparameters.

You'll find the results, log and video recordings of your agent every 250k under
the corresponding file in the results folder. A good way to monitor the progress
of the training is to use Tensorboard. The starter code writes summaries of different
variables.

To launch tensorboard, open a Terminal window and run 
tensorboard --logdir=results/
Then, connect remotely to 
address-ip-of-the-server:6006 
6006 is the default port used by tensorboard.
"""


def main(run_name):
    # Initialize configuration
    run_config = config(run_name)

    # make env
    env = gym.make(run_config.env_name)
    env = AudioFrozenlake(env)
    env = MfccFrozenlake(env, num_mfcc=run_config.num_mfcc)

    # exploration strategy
    exp_schedule = LinearExploration(env, run_config.eps_begin, run_config.eps_end, run_config.eps_nsteps)

    # learning rate schedule
    lr_schedule = LinearSchedule(run_config.lr_begin, run_config.lr_end, run_config.lr_nsteps)

    # train model
    model = AQN(env, run_config)
    model.run(exp_schedule, lr_schedule)
