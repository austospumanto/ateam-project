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
from rl import commands
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
        commands.train_frozenlake_aqn(run_name)

    def test_aqn(self, run_name):
        commands.test_frozenlake_aqn(run_name)



if __name__ == "__main__":
    random.seed(42)
    fire.Fire(Ateam)
