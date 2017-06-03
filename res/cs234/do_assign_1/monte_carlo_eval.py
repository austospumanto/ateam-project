import numpy as np
import random
import gym
import time
import matplotlib as mpl
import collections

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
            0: 3,
            1: 3,
            2: 1,
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


def _monte_carlo_eval(env, start_policy, wer, eps=0.0, gamma=1.0):
    V = np.zeros(env.nS)
    average_R = 0.0
    average_steps = 0.0
    policy = _get_start_policy(start_policy)
    num_seen = np.zeros(env.nS)

    for _ in tqdm(range(num_iters)):
        done = False
        steps = 0.0
        episode_reward = 0.0
        state = env.reset()
        steps_state_seen = collections.defaultdict(list)
        steps_state_seen[state].append(0)

        while not done:
            # Simulate incorrect state prediction
            if random.random() < wer:
                # Choose from valid states
                guessed_state = random.choice([0, 1, 2, 3, 4, 6, 8, 9, 10, 13, 14])
            else:
                guessed_state = state

            # Simulate epsilon greedy action
            if random.random() < eps:
                action = random.randint(0, env.nA - 1)
            else:
                action = policy[guessed_state]

            new_state, reward, done, _ = env.step(action)
            episode_reward += reward
            steps += 1.0
            state = new_state

            if state not in steps_state_seen:
                steps_state_seen[state].append(steps)

        for s, steps_seen in steps_state_seen.items():
            num_seen[s] += 1.0
            steps_to_reward = steps - np.average(steps_seen)
            discounted_reward = (gamma ** (steps_to_reward - 1)) * episode_reward
            V[s] = (V[s] * (num_seen[s] - 1) + discounted_reward) / num_seen[s]

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

    # V, average_R, average_steps = _monte_carlo_eval(env, start_policy, 0.006, eps=0.0, gamma=1.0)
    # grid_print(V)
    # print 'average_R=%f' % average_R
    # print 'average_steps=%f' % average_steps
    # exit()

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
