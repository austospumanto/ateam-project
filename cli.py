#!/usr/bin/env python

import os
import random
import dotenv
import logging

logger = logging.getLogger(__name__)

# Load environment variables from `base/.env` if that file exists
# NOTE: There should be no .env file in production. Environment variables are
#       injected in another way in production.
dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
if os.path.exists(dotenv_path):
    print('Loading environment variables from {}'.format(dotenv_path))
    dotenv.load_dotenv(dotenv_path)


import fire

from rl import Q
from data.tidigits import tidigits_db
from rl import commands as rl_commands
from speech import commands as speech_commands
from envs.lake_envs import *


class Ateam(object):
    def vanilla_example(self):
        Q.vanilla_example()

    def shallow_q_network(self):
        Q.shallow_q_network()

    def shallow_q_network_with_asr(self):
        Q.shallow_q_network_with_asr()

    def process_tidigits(self, sequence_len=None):
        sequence_len = int(sequence_len) if sequence_len else None
        tidigits_db.process_data(sequence_len)

    def get_split_fl_dataset(self):
        tidigits_db.get_split_fl_dataset()

    def train_and_test_with_asr(self):
        Q.train_and_test_with_asr()

    def test_with_asr(self):
        Q.test_with_asr()



    def train_aqn(self, run_name):
        rl_commands.train_frozenlake_aqn(run_name)

    def transfer_train_aqn(self, run_name, restore_run_name):
        rl_commands.transfer_train_frozenlake_aqn(run_name, restore_run_name)

    def test_aqn(self, run_name, env_to_test='test', demo=False, num_episodes=100):
        rl_commands.test_frozenlake_aqn(run_name, env_to_test, demo, num_episodes)



    def train_ctc(self, run_name):
        speech_commands.train_ctcmodel(run_name)

    def transfer_train_ctc(self, ctc_run_name, other_run_name):
        speech_commands.transfer_train_ctcmodel(ctc_run_name, other_run_name)

    def resume_train_ctc(self, run_name):
        speech_commands.resume_train_ctcmodel(run_name)


    def test_asr_qagent(self, restore_run_name, train_subset, env_to_test='test',
                        demo=False, train_with_asr=False, num_episodes=100):
        rl_commands.test_asr_qagent(restore_run_name, train_subset, env_to_test, demo, train_with_asr, num_episodes)



if __name__ == "__main__":
    random.seed(42)
    fire.Fire(Ateam)
