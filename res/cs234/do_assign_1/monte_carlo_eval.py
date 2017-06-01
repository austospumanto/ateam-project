import numpy as np
import random
import gym
import time
import matplotlib as mpl

mpl.use('TkAgg')

import matplotlib.pyplot as plt
from argparse import ArgumentParser
from lake_envs import *
from tqdm import tqdm


num_iters = 100000


def _get_start_policy(policy):
    # [0: left, 1: down, 2: right, 3: up]
    start_policy = None
    if policy == 1:
        start_policy = {
            0: 1,
            1: 0,
            2: 0,
            3: 0,
            4: 1,
            5: 0,
            6: 3,
            7: 0,
            8: 2,
            9: 2,
            10: 1,
            11: 1,
            12: 2,
            13: 2,
            14: 2,
            15: 2
        }
    elif policy == 2:
        start_policy = {
            0: 2,
            1: 2,
            2: 1,
            3: 0,
            4: 3,
            5: 2,
            6: 1,
            7: 0,
            8: 2,
            9: 2,
            10: 1,
            11: 0,
            12: 2,
            13: 2,
            14: 2,
            15: 2
        }
    else:
        start_policy = {
            0: 0,
            1: 3,
            2: 3,
            3: 3,
            4: 0,
            5: 0,
            6: 0,
            7: 0,
            8: 3,
            9: 1,
            10: 0,
            11: 0,
            12: 0,
            13: 2,
            14: 1,
            15: 0
        }
    return start_policy


def _monte_carlo_eval(env, start_policy, wer):
    V = np.zeros((env.nS))
    average_R = 0.0
    average_steps = 0.0
    policy = _get_start_policy(start_policy)
    states_seen = []

    num_seen = {}
    for i in range(env.nS):
        num_seen[i] = 0

    for _ in tqdm(range(num_iters)):
        done = False
        steps = 0
        episode_reward = 0.0
        states_seen[:] = []
        state = env.reset()
        states_seen.append(state)

        while not done:
            action = 0

            if random.random() < wer:
                action = random.randint(0, env.nA - 1)
            else:
                action = policy[state]

            new_state, reward, done, _ = env.step(action)
            episode_reward += reward
            steps += 1

            if new_state not in states_seen:
                states_seen.append(new_state)

            state = new_state

        for j in range(len(states_seen)):
            num_seen[states_seen[j]] += 1
            old_return = V[states_seen[j]]
            new_return = old_return * (num_seen[states_seen[j]] - 1) + episode_reward
            new_return /= num_seen[states_seen[j]]
            V[states_seen[j]] = new_return

        average_R += episode_reward
        average_steps += steps

    average_R /= num_iters
    average_steps /= num_iters

    return V, average_R, average_steps


def grid_print(V):
    print "%f, %f, %f, %f" % (V[0], V[1], V[2], V[3])
    print "%f, %f, %f, %f" % (V[4], V[5], V[6], V[7])
    print "%f, %f, %f, %f" % (V[8], V[9], V[10], V[11])
    print "%f, %f, %f, %f" % (V[12], V[13], V[14], V[15])


def main(start_policy):
    env = gym.make('Stochastic-4x4-FrozenLake-v0')
    wer_list = np.linspace(0.0, 0.03, num=20)
    Q_max_list = np.zeros(20)
    average_R_list = np.zeros(20)
    average_steps_list = np.zeros(20)

    for i in tqdm(range(len(wer_list))):
        error = 1 - ((1 - wer_list[i]) * (1 - wer_list[i]))
        V, average_R, average_steps = _monte_carlo_eval(env, start_policy, error)
        Q_max_list[i] = V[14]
        average_R_list[i] = average_R
        average_steps_list[i] = average_steps

    plt.figure()
    plt.plot(wer_list, Q_max_list)
    plt.xlabel("WER")
    plt.ylabel("Q_max")
    plt.savefig("Qmax.png")

    plt.figure()
    plt.plot(wer_list, average_R_list)
    plt.xlabel("WER")
    plt.ylabel("Average R")
    plt.savefig("AvgR.png")

    plt.figure()
    plt.plot(wer_list, average_steps_list)
    plt.xlabel("WER")
    plt.ylabel("Average steps")
    plt.savefig("AvgSteps.png")

    # V, average_R, average_steps = _monte_carlo_eval(env, start_policy, wer=0.003)

    # grid_print(V)
    # print "Average reward: %f" % average_R
    # print "Average steps: %f" % average_steps


if __name__ == '__main__':
    parser = ArgumentParser(description="monte_carlo_learning -n NUM")
    parser.add_argument('-n', '--num', help="Policy to start (1 = bottom left, 2 = top right)")
    sysargs = parser.parse_args()

    main(sysargs.num)
