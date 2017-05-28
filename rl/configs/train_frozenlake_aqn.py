from admin.config import project_config
import os


class config():
    # env config
    render_train     = False
    render_test      = False
    env_name         = 'Deterministic-4x4-FrozenLake-v0'
    overwrite_render = True
    record           = True
    high             = 255.

    # output config
    output_path  = os.path.join(
        project_config.base_dir,
        'results/train_frozenlake_aqn/'
    )
    model_output = output_path + 'model.weights/'
    log_path     = output_path + 'log.txt'
    plot_output  = output_path + 'scores.png'
    record_path  = output_path + 'monitor/'

    # model and training config
    num_episodes_test = 100
    grad_clip         = True
    clip_val          = 10
    saving_freq       = 25000
    log_freq          = 50
    eval_freq         = 1000
    record_freq       = 1000
    soft_epsilon      = 0.05

    # nature paper hyper params
    nsteps_train       = 50000
    batch_size         = 32
    buffer_size        = 10000
    target_update_freq = 1000
    gamma              = 0.99
    learning_freq      = 1
    state_history      = 1
    skip_frame         = 1
    lr_begin           = 0.000025
    lr_end             = 0.000005
    lr_nsteps          = nsteps_train/2
    eps_begin          = 1
    eps_end            = 0.1
    eps_nsteps         = 10000
    learning_start     = 1000

    # for mfcc derivation
    num_mfcc           = 13
    num_digits         = 12  # 11 (TIDIGITS - 0-9 + oh) + 1 (blank) = 12
    num_hidden         = 128
