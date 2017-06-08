import os
from logging_tools import setup_logging

__all__ = ['project_config']


class ProjectConfig(object):
    def __init__(self, *args, **kw):
        self.project_root = os.path.dirname(os.path.dirname(__file__))
        self.LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO').upper()
        self.base_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
        self.rl_dir = os.path.join(self.base_dir, 'rl')
        self.models_dir = os.path.join(self.rl_dir, 'models')
        self.saved_runs_dir = os.path.join(self.base_dir, 'saved-runs')
        self.speech_results_dir = os.path.join(self.base_dir, 'speech', 'results')
        self.val_ratio = 0.2

        # Tensorflow
        self.tensorboard_logdir = os.environ.get('TENSORBOARD_LOGDIR', os.path.join(self.base_dir, 'results'))


def bool_param(name, default):
    try:
        param = os.environ[name]
    except KeyError:
        return default
    return param.lower() in ('true', 'on', 'yes')


def int_param(name, default):
    try:
        param = os.environ[name]
    except KeyError:
        return default
    return int(param)


def float_param(name, default):
    try:
        param = os.environ[name]
    except KeyError:
        return default
    return float(param)


project_config = ProjectConfig()
setup_logging()
