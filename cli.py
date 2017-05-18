#!/usr/bin/env python

import fire

from rl import Q
from data import process_tidigits


class Ateam(object):
    def vanilla_example(self):
        Q.vanilla_example()

    def base_asr_example(self):
        Q.pretrained_asr_example()

    def process_tidigits(self):
        process_tidigits.process_data()

    def train_and_test_with_asr(self):
        Q.train_and_test_with_asr()

    def test_with_asr(self):
        Q.test_with_asr()


if __name__ == '__main__':
    fire.Fire(Ateam)
