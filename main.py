#!/usr/bin/env python

import fire

from rl import Q


class Ateam(object):
    def vanilla_example(self):
        Q.vanilla_example()


if __name__ == '__main__':
    fire.Fire(Ateam)
