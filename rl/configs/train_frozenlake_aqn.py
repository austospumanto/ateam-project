import datetime
import os

from admin.config import project_config

class config(object):
    def __init__(self, run_name):
        # env config
        self.render_train     = False
        self.render_test      = False
        self.env_name         = 'Deterministic-4x4-FrozenLake-v0'
        self.overwrite_render = True
        self.record           = True
        self.high             = 255.

        # output config
        self.run_dir = '%s-%s' % (datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'), run_name)
        self.output_path  = os.path.join(
            project_config.base_dir, 'results', self.run_dir
        )
        self.model_output = os.path.join(self.output_path, 'model.weights/')
        self.log_path     = os.path.join(self.output_path, 'log.txt')
        self.plot_output  = os.path.join(self.output_path, 'scores.png')
        self.record_path  = os.path.join(self.output_path, 'monitor/')
        self.config_path  = os.path.join(self.output_path, 'config.txt')
        self.qn_src_path  = os.path.join(project_config.models_dir, 'QN.py')
        self.qn_dst_path  = os.path.join(self.output_path, 'QN.py')
        self.dqn_src_path = os.path.join(project_config.models_dir, 'DQN.py')
        self.dqn_dst_path = os.path.join(self.output_path, 'DQN.py')
        self.aqn_src_path = os.path.join(project_config.models_dir, 'AQN.py')
        self.aqn_dst_path = os.path.join(self.output_path, 'AQN.py')

        # model and training config
        num_episodes_test = 20
        max_steps_test    = 200
        grad_clip         = True
        clip_val          = 10
        saving_freq       = 5000
        log_freq          = 50
        eval_freq         = 1000
        record_freq       = 1000
        soft_epsilon      = 0.00  # Set this to 0 so no random actions during testing
        clip_q            = False

        # nature paper hyper params
        nsteps_train       = 200000
        batch_size         = 64
        buffer_size        = 10000
        target_update_freq = 100
        gamma              = 0.98
        learning_freq      = 1
        state_history      = 1
        skip_frame         = 1
        lr_begin           = 0.00025
        lr_end             = 0.00001
        lr_nsteps          = nsteps_train/2
        eps_begin          = 1
        eps_end            = 0.1
        eps_nsteps         = 10000
        learning_start     = 1000

        # for mfcc derivation
        num_mfcc           = 13
        num_digits         = 11  # 11 (TIDIGITS - 0-9 + oh) + 1 (blank) = 12

        # for the Neural Net
        num_hidden         = 64
        num_layers         = 1
