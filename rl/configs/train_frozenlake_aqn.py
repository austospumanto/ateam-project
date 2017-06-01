import datetime
import os

from admin.config import project_config

class config(object):
    def __init__(self, run_name):
        # env config
        self.render_train     = False
        self.render_test      = False
        self.env_name         = 'Stochastic-4x4-FrozenLake-v0'
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
        self.num_episodes_test = 40
        self.max_steps_test    = 300
        self.grad_clip         = True
        self.clip_val          = 5
        self.saving_freq       = 5000
        self.log_freq          = 50
        self.eval_freq         = 1000
        self.record_freq       = 1000
        self.soft_epsilon      = 0.00  # Set this to 0 so no random actions during testing
        self.clip_q            = False

        # nature paper hyper params
        self.nsteps_train       = 2000000
        self.batch_size         = 64
        self.buffer_size        = 10000
        self.target_update_freq = 25
        self.gamma              = 0.98
        self.learning_freq      = 1
        self.state_history      = 1
        self.skip_frame         = 1
        self.lr_begin           = 0.00025
        self.lr_end             = 0.000005
        self.lr_nsteps          = self.nsteps_train/2
        self.eps_begin          = 1.0
        self.eps_end            = 0.05
        self.eps_nsteps         = 20000
        self.learning_start     = 1000

        # for mfcc derivation
        self.num_mfcc           = 13
        self.num_digits         = 11  # 11 (TIDIGITS - 0-9 + oh) + 1 (blank) = 12

        # for the Neural Net
        self.n_hidden_rnn       = 32
        self.n_hidden_fc        = 64
        self.n_layers_rnn         = 2
