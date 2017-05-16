### Episode model free learning using Q-learning and SARSA

# Do not change the arguments and output types of any of the functions provided! You may debug in Main and elsewhere.

import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import random
import gym
import time
from lake_envs import *

def learn_Q_QLearning(env, num_episodes=5000, gamma=0.95, lr=0.95, e=0.8, decay_rate=0.999):
    """Learn state-action values using the Q-learning algorithm with epsilon-greedy exploration strategy.
    Update Q at the end of every episode.

    Parameters
    ----------
    env: gym.core.Environment
        Environment to compute Q function for. Must have nS, nA, and P as
        attributes.
    num_episodes: int 
        Number of episodes of training.
    gamma: float
        Discount factor. Number in range [0, 1)
    learning_rate: float
        Learning rate. Number in range [0, 1)
    e: float
        Epsilon value used in the epsilon-greedy method. 
    decay_rate: float
        Rate at which epsilon falls. Number in range [0, 1)

    Returns
    -------
    np.array
        An array of shape [env.nS x env.nA] representing state, action values
    """

    ############################
    # YOUR IMPLEMENTATION HERE #
    ############################

    Q = np.zeros((env.nS, env.nA))
    action_list = []
    reward_list = []
    state_list = []
    epsilon = e

    average_rewards = []
    average_rewards.append(0.0)

    for i in range(num_episodes):
        state = env.reset()
        action_list[:] = []
        reward_list[:] = []
        state_list[:] = []

        state_list.append(state)
        done = False
        reward = 0.0

        while not done:
            eps_check = random.random()
            action = 0

            if eps_check < epsilon:
                action = random.randint(0, env.nA - 1)
            else:
                action = np.argmax(Q[state])

            new_state, reward, done, _ = env.step(action)
            action_list.append(action)
            reward_list.append(reward)
            state_list.append(new_state)
            state = new_state

        for j in range(len(action_list)):
            Qsamp = reward_list[j]
            V_s_prime = np.amax(Q[state_list[j+1]])
            Qsamp += (gamma * V_s_prime)

            old_Q = Q[state_list[j]][action_list[j]]
            Q[state_list[j]][action_list[j]] = ((1 - lr) * old_Q) + (lr * Qsamp)

        if i < 1000:
            prev_reward = average_rewards[i] * i
            new_reward = (prev_reward + reward) / (i + 1)
            average_rewards.append(new_reward)

        epsilon *= decay_rate

    """plt.plot(average_rewards)
    plt.ylabel('Average reward')
    plt.xlabel('Episodes')
    plt.show()"""

    return Q

def learn_Q_SARSA(env, num_episodes=10000, gamma=0.95, lr=0.9, e=0.8, decay_rate=0.999):
    """Learn state-action values using the SARSA algorithm with epsilon-greedy exploration strategy
    Update Q at the end of every episode.

    Parameters
    ----------
    env: gym.core.Environment
        Environment to compute Q function for. Must have nS, nA, and P as
        attributes.
    num_episodes: int 
        Number of episodes of training.
    gamma: float
        Discount factor. Number in range [0, 1)
    learning_rate: float
        Learning rate. Number in range [0, 1)
    e: float
        Epsilon value used in the epsilon-greedy method. 
    decay_rate: float
        Rate at which epsilon falls. Number in range [0, 1)

    Returns
    -------
    np.array
        An array of shape [env.nS x env.nA] representing state-action values
    """

    ############################
    # YOUR IMPLEMENTATION HERE #
    ############################
    Q = np.zeros((env.nS, env.nA))
    epsilon = e

    for i in range(num_episodes):
        state = env.reset()
        done = False
        eps_check = random.random()
        action = 0
        # reward = 0.0

        if eps_check < epsilon:
            action = random.randint(0, env.nA - 1)
        else:
            action = np.argmax(Q[state])

        while not done:
            new_state, reward, done, _ = env.step(action)

            eps_check = random.random()
            new_action = 0

            if eps_check < epsilon:
                new_action = random.randint(0, env.nA - 1)
            else:
                new_action = np.argmax(Q[new_state])

            old_Q = Q[state][action]
            Q[state][action] = ((1 - lr) * old_Q) + (lr * (reward + (gamma * Q[new_state][new_action])))

            state = new_state
            action = new_action

        epsilon *= decay_rate


    return Q

def render_single_Q(env, Q):
    """Renders Q function once on environment. Watch your agent play!

        Parameters
        ----------
        env: gym.core.Environment
            Environment to play Q function on. Must have nS, nA, and P as
            attributes.
        Q: np.array of shape [env.nS x env.nA]
            state-action values.
    """

    episode_reward = 0
    state = env.reset()
    done = False
    while not done:
        # env.render()
        # time.sleep(0.5) # Seconds between frames. Modify as you wish.
        action = np.argmax(Q[state])
        state, reward, done, _ = env.step(action)
        episode_reward += reward

    print "Episode reward: %f" % episode_reward
    return episode_reward

# Feel free to run your own debug code in main!
def main():
    env = gym.make('Stochastic-4x4-FrozenLake-v0')
    Q = learn_Q_QLearning(env)
    # Q = learn_Q_SARSA(env)

    # render_single_Q(env, Q)

    cum_reward = 0.0
    for i in range(100):
        reward = render_single_Q(env, Q)
        cum_reward += reward

    avg_reward = cum_reward / 100
    print "Average over 100 trials: %f" % avg_reward

if __name__ == '__main__':
        main()
