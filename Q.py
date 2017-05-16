import tqdm
import gym
import numpy as np
import random
import time
from lake_envs import *


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
            next_state, reward, done, _ = env.step(action)

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


# Functions for testing
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

    print "Episode reward: %f" % episode_reward


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


def print_avg_score(env, Q):
    # Average episode rewards over trials
    num_trials = 100
    episode_rewards = [_run_trial_Q(env, Q) for _ in range(num_trials)]
    avg_reward = np.average(episode_rewards)
    print 'Averge episode score/reward: %.3f' % avg_reward


def main():
    env = gym.make('Stochastic-4x4-FrozenLake-v0')
    Q = learn_Q_QLearning(env)
    print_avg_score(env, Q)
    render_single_Q(env, Q)
