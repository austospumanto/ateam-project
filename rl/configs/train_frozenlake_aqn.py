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
        self.run_name = run_name
        self.run_dir = '%s-%s' % (datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'), run_name)
        self.output_path  = os.path.join(
            project_config.base_dir, 'results', self.run_dir
        )
        self.model_output_path = os.path.join(self.output_path, 'model.weights/')
        self.log_path     = os.path.join(self.output_path, 'log.txt')
        self.plot_output  = os.path.join(self.output_path, 'scores.png')
        self.record_path  = os.path.join(self.output_path, 'monitor/')
        self.config_path  = os.path.join(self.output_path, 'train_frozenlake_aqn_config.txt')
        self.qn_src_path  = os.path.join(project_config.models_dir, 'QN.py')
        self.qn_dst_path  = os.path.join(self.output_path, 'QN.py')
        self.dqn_src_path = os.path.join(project_config.models_dir, 'DQN.py')
        self.dqn_dst_path = os.path.join(self.output_path, 'DQN.py')
        self.aqn_src_path = os.path.join(project_config.models_dir, 'AQN.py')
        self.aqn_dst_path = os.path.join(self.output_path, 'AQN.py')
        self.project_cfg_src_path = os.path.join(project_config.base_dir, 'admin', 'config.py')
        self.project_cfg_dst_path = os.path.join('project_config.py')

        # model and training config
        self.num_episodes_test = 30
        self.max_steps_test    = 30
        self.grad_clip         = True
        self.clip_val          = 10
        self.saving_freq       = 2500
        self.log_freq          = 100
        self.eval_freq         = 1000
        self.record_freq       = 1000
        self.soft_epsilon      = 0.00  # Set this to 0 so no random actions during testing
        self.clip_q            = False

        # nature paper hyper params
        self.nsteps_train       = 1000000
        self.batch_size         = 128
        self.buffer_size        = 10000
        self.target_update_freq = 25
        self.gamma              = 0.95
        self.learning_freq      = 4
        self.state_history      = 1
        self.skip_frame         = 1
        self.lr_begin           = 0.00025
        self.lr_end             = 0.000001
        self.lr_nsteps          = self.nsteps_train/5
        self.eps_begin          = 1.0
        self.eps_end            = 0.05
        self.eps_nsteps         = 10000
        self.learning_start     = 1000
        self.l2_lambda          = 1e-12

        # for mfcc derivation
        self.num_mfcc           = 13
        self.num_digits         = 11  # 11 (TIDIGITS - 0-9 + oh) + 1 (blank) = 12

        # for the Neural Net
        self.n_hidden_rnn       = 128
        self.n_hidden_fc        = 64
        self.n_layers_rnn         = 1

        # For the MfccFrozenLake environment
        self.audio_clip_mode = 'standard'  # oneof('standard', 'synthesized')
        if 'synth' in run_name:
            self.audio_clip_mode = 'synthesized'
