#!/usr/bin/env python

import os
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
from data import process_tidigits
from rl import train_frozenlake_aqn
from envs.lake_envs import *


class Ateam(object):
    def vanilla_example(self):
        Q.vanilla_example()

    def shallow_q_network(self):
        Q.shallow_q_network()

    def shallow_q_network_with_asr(self):
        Q.shallow_q_network_with_asr()

    def process_tidigits(self):
        process_tidigits.process_data()

    def train_and_test_with_asr(self):
        Q.train_and_test_with_asr()

    def test_with_asr(self):
        Q.test_with_asr()

    def train_frozenlake_aqn(self):
        train_frozenlake_aqn.main()


fire.Fire(Ateam)
