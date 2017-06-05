from admin.config import project_config
from configs.train_frozenlake_aqn import config
from envs.MfccFrozenlake import MfccFrozenlake
from rl.models.AQN import AQN
from rl.utils.schedule import LinearExploration, LinearSchedule
import logging
import os

logger = logging.getLogger(__name__)

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


def make_split_envs(run_config):
    train_env, val_env, test_env = MfccFrozenlake.make_train_val_test_envs(
        run_config.env_name, num_mfcc=run_config.num_mfcc,
        use_synthesized=run_config.audio_clip_mode)

    logger.info('Train env has %d raw samples' % train_env.n_samples)
    logger.info('Val env has %d raw samples' % val_env.n_samples)
    logger.info('Test env has %d raw samples' % test_env.n_samples)

    return {
        'train': train_env,
        'val': val_env,
        'test': test_env
    }


def train_frozenlake_aqn(run_name):
    run_config = config(run_name)
    envs = make_split_envs(run_config)
    exp_schedule = LinearExploration(envs['train'], run_config.eps_begin, run_config.eps_end, run_config.eps_nsteps)
    lr_schedule = LinearSchedule(run_config.lr_begin, run_config.lr_end, run_config.lr_nsteps)
    aqn_model = AQN(run_config, envs=envs, mode='train')
    aqn_model.run(exp_schedule, lr_schedule)


def test_frozenlake_aqn(restore_run_name, env_to_test='test'):
    assert env_to_test in ('train', 'val', 'test')
    run_config = config(restore_run_name)

    # Create the train,val,test envs in standard or synthesized mode
    envs = make_split_envs(run_config)

    aqn_model = AQN(run_config, envs=envs, logger=logger, mode='test')

    # Restore weights from latest checkpoint
    aqn_model.restore()

    aqn_model.evaluate(envs[env_to_test], num_episodes=5, max_episode_steps=10)
