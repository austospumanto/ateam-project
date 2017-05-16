# Episode model free learning using Q-learning and SARSA

# Do not change the arguments and output types of any of the functions provided! You may debug in Main and elsewhere.

from __future__ import print_function
import tqdm
import numpy as np
import random
import gym
import time
from lake_envs import *

import matplotlib as mil
mil.use('TkAgg')
import matplotlib.pyplot as plt


# Part 5a
def learn_Q_QLearning(env, num_episodes=15000, gamma=0.98, lr=0.08, e=0.5, decay_rate=0.9999, episode_scores=None):
    """
        Learn state-action values using the Q-learning algorithm with epsilon-greedy exploration strategy.
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
    Q = np.zeros((env.nS, env.nA))
    for episode_idx in tqdm.tqdm(range(num_episodes)):
        # Choose a random starting state
        cur_state = env.reset()

        # Data structure for storing (s, a, r, s') tuples
        sars = []

        # Start the episode. The episode ends when we reach a terminal state (i.e. "done is True")
        done = False
        episode_reward = 0.0
        while not done:
            # Choose an action "epsilon-greedily" (where epsilon is the var "e")
            action = _choose_egreedy_action(env, cur_state, Q, e)

            # Use env's transition probs to "choose" next state
            next_state, reward, done, _ = _ASR(env.step(action))  # env.step(action)

            sars.append((cur_state, action, reward, next_state))

            # Move to the next state
            cur_state = next_state

            episode_reward += reward

        # If we're running this as part of 5c, then record the scores
        if episode_scores is not None:
            # NOTE: Here I am simply recording 0 or 1 (the undiscounted score)
            episode_scores[episode_idx] = episode_reward

        # Update Q after episode ends
        for cur_state, action, reward, next_state in sars:
            # Get optimal value of next state (i.e. assume we act greedily from the next state onwards)
            V_opt_ns = np.max(Q[next_state])

            # Calculate Q_samp_sa (i.e. "What was Q[s][a] for this particular sample/event")
            Q_samp_sa = reward + gamma * V_opt_ns

            # Update our overall estimate of Q[s][a]
            Q[cur_state][action] = (1 - lr) * Q[cur_state][action] + lr * Q_samp_sa

        # Decay the randomness of our action selection (i.e. increase greediness)
        e *= decay_rate

    return Q
    ############################


def _choose_egreedy_action(env, s, Q, e):
    """
        Given a Q function (Q), the environment (env), and the current state (s),
        choose a random action with probability e and the optimal action with
        probability 1-e

        Returns
        -------
        int
            The index of the chosen action
    """
    be_greedy = bool((1.0 - e) > random.random())
    if be_greedy:
        # If greedy, randomly choose from among the best actions
        a = np.argmax(Q[s])  # random.choice([a for a, q_val in enumerate(Q[s]) if q_val == np.max(Q[s])])
    else:
        a = random.randint(0, env.nA - 1)
    return a


# Part 5b
def learn_Q_SARSA(env, num_episodes=15000, gamma=0.99, lr=0.05, e=0.4, decay_rate=0.9999):
    """
        Learn state-action values using the SARSA algorithm with epsilon-greedy exploration strategy
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
    Q = np.zeros((env.nS, env.nA))
    for episode_idx in tqdm.tqdm(range(num_episodes)):
        # Choose a random starting state
        cur_state = env.reset()

        # Choose an action "epsilon-greedily" (where epsilon is the var "e")
        cur_action = _choose_egreedy_action(env, cur_state, Q, e)

        # Start the episode. The episode ends when we reach a terminal state (i.e. "done is True")
        done = False
        while not done:
            # Use env's transition probs to "choose" next state
            next_state, reward, done, _ = env.step(cur_action)

            # Choose next action "epsilon-greedily" (where epsilon is the var "e")
            next_action = _choose_egreedy_action(env, next_state, Q, e)

            # Calculate Q_samp_sp_ap (i.e. "What is Q[s][a] for next action from next state")
            Q_samp_sp_ap = reward + gamma * Q[next_state][next_action]

            # Update our overall estimate of Q[s][a]
            Q[cur_state][cur_action] = (1 - lr) * Q[cur_state][cur_action] + lr * Q_samp_sp_ap

            cur_state, cur_action = next_state, next_action

        # Decay the randomness of our action selection (i.e. increase greediness)
        e *= decay_rate
    return Q
    ############################


def render_single_Q(env, Q):
    """
        Renders Q function once on environment. Watch your agent play!

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
        env.render()
        time.sleep(0.5)  # Seconds between frames. Modify as you wish.
        action = np.argmax(Q[state])
        state, reward, done, _ = env.step(action)
        episode_reward += reward

    print("Episode reward: %f" % episode_reward)


def _run_trial_Q(env, Q):
    """
        Runs Q function once on environment and returns the reward.

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
        action = np.argmax(Q[state])
        state, reward, done, _ = env.step(action)
        episode_reward += reward
    return episode_reward


# Testing the Q-learning agent to see if its average score is >0.78
def part5a():
    env = gym.make('Stochastic-4x4-FrozenLake-v0')
    Q = learn_Q_QLearning(env)

    # Print policy
    policy = np.argmax(Q, axis=1)
    print('Policy: %r' % policy)

    print_avg_score(env, Q)


# Testing the SARSA-learning agent to see if its average score is >0.78
def part5b():
    env = gym.make('Stochastic-4x4-FrozenLake-v0')
    Q = learn_Q_SARSA(env)

    # Print policy
    policy = np.argmax(Q, axis=1)
    print('Policy: %r' % policy)

    print_avg_score(env, Q)


def print_avg_score(env, Q):
    # Average episode rewards over trials
    num_trials = 100
    episode_rewards = [_run_trial_Q(env, Q) for _ in range(num_trials)]
    avg_reward = np.average(episode_rewards)
    print('Averge episode score/reward: %.3f' % avg_reward)


def part5c():
    env = gym.make('Stochastic-4x4-FrozenLake-v0')
    num_runs = 9
    sqrt_num_runs = int(num_runs ** 0.5)
    num_episodes = 1000
    fig, axarr = plt.subplots(sqrt_num_runs, sqrt_num_runs, sharex=True, sharey=True)
    for i in range(num_runs):
        episode_scores = np.zeros(num_episodes)

        # Implicitly fills the episode_scores array - no need to store result of call
        Q = learn_Q_QLearning(env, lr=0.14, e=0.99, decay_rate=0.996, episode_scores=episode_scores, num_episodes=num_episodes)

        # Print policy
        policy = np.argmax(Q, axis=1)
        print('Policy: %r' % policy)
        print_avg_score(env, Q)

        running_avg_scores = []
        for episode_num in range(1, num_episodes + 1):
            running_avg_score = np.sum(episode_scores[:episode_num]) / episode_num
            running_avg_scores.append(running_avg_score)

        x = list(range(1, num_episodes + 1))

        row_idx = int(i / sqrt_num_runs)
        col_idx = i % sqrt_num_runs
        ax = axarr[row_idx, col_idx]
        ax.plot(x, running_avg_scores)

        if row_idx == sqrt_num_runs - 1:
            ax.set_xlabel('Episode Number')
        if col_idx == 0:
            ax.set_ylabel('Running Average Score')
    plt.suptitle('Running Average Score Of Q-Learning Agent \nOver First 1000 Training Episodes \n(over %s independent runs)' % num_runs)
    plt.savefig('pics/part5c.png', bbox_inches='tight')
    plt.show()


# Feel free to run your own debug code in main!
def main():
    env = gym.make('Stochastic-4x4-FrozenLake-v0')
    Q = learn_Q_QLearning(env)
    Q = learn_Q_SARSA(env)
    render_single_Q(env, Q)


if __name__ == '__main__':
    # main()
    # part5a()
    part5b()
    # part5c()
