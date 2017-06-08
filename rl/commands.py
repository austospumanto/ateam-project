from admin.config import project_config
from configs.train_frozenlake_aqn import config
from rl.models.AQN import AQN
from rl.utils.schedule import LinearExploration, LinearSchedule
import logging
import tensorflow as tf

from envs.MfccFrozenlake import MfccFrozenlake
from speech.models.StateRecognizer import StateRecognizer
from speech.models.CTCModel import CTCModel
from rl.combined.AsrQlearnAgent import AsrQlearnAgent
from rl.combined.AsrQlearnAgent import Config as AsrQConfig

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


def make_split_envs(run_config, demo=False):
    use_synthesized = True if run_config.audio_clip_mode == 'synthesized' else False
    train_env, val_env, test_env = MfccFrozenlake.make_train_val_test_envs(
        run_config.env_name, num_mfcc=run_config.num_mfcc,
        use_synthesized=use_synthesized, demo=demo)

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


def test_frozenlake_aqn(restore_run_name, env_to_test, demo, num_episodes):
    assert env_to_test in ('train', 'val', 'test')
    run_config = config(restore_run_name)

    # Create the train,val,test envs in standard or synthesized mode
    envs = make_split_envs(run_config, demo=demo)
    aqn_model = AQN(run_config, envs=envs, logger=logger, mode='test')
    aqn_model.restore()
    aqn_model.evaluate(envs[env_to_test],
                       num_episodes=num_episodes,
                       max_episode_steps=300, demo=demo)


def test_asr_qagent(restore_run_name, name_env_to_eval, demo=False, train_with_asr=False, num_episodes=100):
    run_config = AsrQConfig()
    envs = make_split_envs(run_config, demo=False)
    valid_env_states = list(range(envs['train'].nS))

    with tf.Session() as sess:
        ctc_model = CTCModel(sess, restore_run_name)
        state_recognizer = StateRecognizer(valid_env_states, ctc_model, verbose=False)
        asr_qagent = AsrQlearnAgent(envs, state_recognizer)

        asr_qagent.train(train_with_asr)
        env_to_eval = envs[name_env_to_eval]
        env_to_eval.demo = demo
        asr_qagent.evaulate(env_to_eval, num_trials=num_episodes, verbose=True)
