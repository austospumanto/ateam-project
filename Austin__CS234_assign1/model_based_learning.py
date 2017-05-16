# Episodic Model Based Learning using Maximum Likelihood Estimate of the Environment

# Do not change the arguments and output types of any of the functions provided! You may debug in Main and elsewhere.

from __future__ import print_function
import numpy as np
import random
import tqdm
import gym
import time
from lake_envs import *
from vi_and_pi import value_iteration
import matplotlib as mil
mil.use('TkAgg')
import matplotlib.pyplot as plt


def initialize_P(nS, nA):
    """
        Initializes a uniformly random model of the environment with 0 rewards.

        Parameters
        ----------
        nS: int
            Number of states
        nA: int
            Number of actions

        Returns
        -------
        P: np.array of shape [nS x nA x nS x 4] where items are tuples representing transition information
            P[state][action] is a list of (prob, next_state, reward, done) tuples.
    """
    P = [[[(1.0/nS, i, 0, False) for i in range(nS)] for _ in range(nA)] for _ in range(nS)]
    return P


def initialize_counts(nS, nA):
    """
        Initializes a counts array.

        Parameters
        ----------
        nS: int
            Number of states
        nA: int
            Number of actions

        Returns
        -------
        counts: np.array of shape [nS x nA x nS]
            counts[state][action][next_state] is the number of times that doing "action" at state "state" transitioned to "next_state"
    """
    counts = [[[0 for _ in range(nS)] for _ in range(nA)] for _ in range(nS)]
    return counts


def initialize_rewards(nS, nA):
    """
        Initializes a rewards array. Values represent running averages.

        Parameters
        ----------
        nS: int
            Number of states
        nA: int
            Number of actions

        Returns
        -------
        rewards: array of shape [nS x nA x nS]
            counts[state][action][next_state] is the running average of rewards of doing "action" at "state" transtioned to "next_state"
    """
    rewards = [
        [
            [0 for _ in range(nS)]
            for _ in range(nA)
        ]
        for _ in range(nS)
    ]
    return rewards


def counts_and_rewards_to_P(counts, rewards):
    """
        Converts counts and rewards arrays to a P array consistent with the Gym environment data structure for a model of the environment.
        Use this function to convert your counts and rewards arrays to a P that you can use in value iteration.

        Parameters
        ----------
        counts: array of shape [nS x nA x nS]
            counts[state][action][next_state] is the number of times that doing "action" at state "state" transitioned to "next_state"
        rewards: array of shape [nS x nA x nS]
            counts[state][action][next_state] is the running average of rewards of doing "action" at "state" transtioned to "next_state"

        Returns
        -------
        P: np.array of shape [nS x nA x nS x 4] where items are tuples representing transition information
            P[state][action] is a list of (prob, next_state, reward, done) tuples.
    """
    nS = len(counts)
    nA = len(counts[0])
    P = [[[] for _ in range(nA)] for _ in range(nS)]

    for state in range(nS):
        for action in range(nA):
            if sum(counts[state][action]) != 0:
                for next_state in range(nS):
                    if counts[state][action][next_state] != 0:
                        prob = float(counts[state][action][next_state]) / float(sum(counts[state][action]))
                        reward = rewards[state][action][next_state]
                        P[state][action].append((prob, next_state, reward, False))
            else:
                prob = 1.0 / float(nS)
                for next_state in range(nS):
                    P[state][action].append((prob, next_state, 0, False))

    return P


def update_mdp_model_with_history(counts, rewards, history):
    """
        Given a history of an entire episode, update the count and rewards arrays

        Parameters
        ----------
        counts: array of shape [nS x nA x nS]
            counts[state][action][next_state] is the number of times that doing "action" at state "state" transitioned to "next_state"
        rewards: array of shape [nS x nA x nS]
            counts[state][action][next_state] is the running average of rewards of doing "action" at "state" transtioned to "next_state"
        history:
            a list of [state, action, reward, next_state, done]
    """
    ############################
    for state, action, reward, next_state, done in history:
        if int(state) == 15 and int(next_state) == 0:
            import pdb; pdb.set_trace()
        cur_count, cur_reward = counts[state][action][next_state], rewards[state][action][next_state]

        # Calcualte running average of reward
        rewards[state][action][next_state] = (cur_reward * cur_count + reward) / (cur_count + 1.0)

        # Increment count
        counts[state][action][next_state] += 1

        # HINT: For terminal states, we define that the probability of any action returning the state to itself is 1 (with zero reward)
        # Make sure you record this information in your counts array by updating the counts for this accordingly for your
        # value iteration to work.
        if done:
            # NOTE: We assume here that the counts for (next_state, a, s) for all s != next_state
            #       are all 0 at this point
            for a in range(len(counts[next_state])):
                counts[next_state][a][next_state] = 1
                rewards[next_state][a][next_state] = 0
    ############################

    return counts, rewards


def learn_with_mdp_model(env, num_episodes=5000, gamma=0.95, e=0.9, decay_rate=0.996, episode_scores=None):
    """
        Build a model of the environment and use value iteration to learn a policy. In the next episode, play with the new
        policy using epsilon-greedy exploration.

        Your model of the environment should be based on updating counts and rewards arrays. The counts array counts the number
        of times that "state" with "action" led to "next_state", and the 8ewards array is the running average of rewards for
        going from at "state" with "action" leading to "next_state".

        For a single episode, create a list called "history" with all the experience
        from that episode, then update the "counts" and "rewards" arrays using the function "update_mdp_model_with_history".

        You may then call the prewritten function "counts_and_rewards_to_P" to convert your counts and rewards arrays to
        an environment data structure P consistent with the Gym environment's one. You may then call on value_iteration(P, nS, nA)
        to get a policy.

        Parameters
        ----------
        env: gym.core.Environment
            Environment to compute Q function for. Must have nS, nA, and P as
            attributes.
        num_episodes: int
            Number of episodes of training.
        gamma: float
            Discount factor. Number in range [0, 1)
        e: float
            Epsilon value used in the epsilon-greedy method.
        decay_rate: float
            Rate at which epsilon falls. Number in range [0, 1)

        Returns
        -------
        policy: np.array
            An array of shape [env.nS] representing the action to take at a given state.
    """

    P = initialize_P(env.nS, env.nA)
    counts = initialize_counts(env.nS, env.nA)
    rewards = initialize_rewards(env.nS, env.nA)

    ############################
    for episode_idx in tqdm.tqdm(range(num_episodes)):
        # Choose a random starting state
        cur_state = env.reset()

        # Calculate a policy for this episode using what we know about the env so far (P)
        _, policy = value_iteration(P, env.nS, env.nA, gamma=gamma, max_iteration=100, verbose=False)

        # Start the episode. The episode ends when we reach a terminal state (i.e. "done is True")
        done = False
        history = []
        episode_reward = 0.0
        while not done:
            # Choose an action "epsilon-greedily" (where epsilon is the var "e")
            action = _choose_egreedy_action(env, cur_state, policy, e)

            # Use env's transition probs to "choose" next state
            next_state, reward, done, _ = env.step(action)

            # Record this step in the history
            history.append((cur_state, action, reward, next_state, done))

            # Move to next state
            cur_state = next_state

            episode_reward += reward

        counts, rewards = update_mdp_model_with_history(counts, rewards, history)
        P = counts_and_rewards_to_P(counts, rewards)

        # If we're running this as part of 5d, then record the scores
        if episode_scores is not None:
            # NOTE: Here I am simply recording 0 or 1 (the undiscounted score).
            episode_scores[episode_idx] = episode_reward

        # Decay the randomness of our action selection (i.e. increase greediness)
        e *= decay_rate

    _, policy = value_iteration(P, env.nS, env.nA, gamma=gamma, max_iteration=100, verbose=False)
    ############################

    return policy


def _choose_egreedy_action(env, s, policy, e):
    """
        Given a policy (policy), the environment (env), and the current state (s),
        choose a random action with probability e and the policy's action with
        probability 1-e

        Returns
        -------
        int
            The index of the chosen action
    """
    be_greedy = bool((1.0 - e) > random.random())
    if be_greedy:
        # If greedy, choose the action that the policy dictates
        a = policy[s]
    else:
        a = random.randint(0, env.nA - 1)
    return a


def render_single(env, policy):
    """
        Renders policy once on environment. Watch your agent play!

        Parameters
        ----------
        env: gym.core.Environment
            Environment to play on. Must have nS, nA, and P as
            attributes.
        Policy: np.array of shape [env.nS]
            The action to take at a given state
    """

    episode_reward = 0
    state = env.reset()
    done = False
    while not done:
        env.render()
        time.sleep(0.5)  # Seconds between frames. Modify as you wish.
        action = policy[state]
        state, reward, done, _ = env.step(action)
        episode_reward += reward

    print("Episode reward: %f" % episode_reward)


def _run_trial_policy(env, policy):
    """
        Runs policy once on environment and returns the episode's reward.

        Parameters
        ----------
        env: gym.core.Environment
            Environment to play on. Must have nS, nA, and P as
            attributes.
        Policy: np.array of shape [env.nS]
            The action to take at a given state
    """

    episode_reward = 0
    state = env.reset()
    done = False
    while not done:
        action = policy[state]
        state, reward, done, _ = env.step(action)
        episode_reward += reward
    return episode_reward


def print_avg_score(env, policy):
    # Average episode rewards over trials
    num_trials = 100
    episode_rewards = [_run_trial_policy(env, policy) for _ in range(num_trials)]
    avg_reward = np.average(episode_rewards)
    print('Averge episode score/reward: %.3f' % avg_reward)


def part5d_1():
    env = gym.make('Stochastic-4x4-FrozenLake-v0')
    policy = learn_with_mdp_model(env)

    # Print policy
    print('Policy: %r' % policy)
    print_avg_score(env, policy)


def part5d_2():
    env = gym.make('Stochastic-4x4-FrozenLake-v0')
    num_runs = 9
    sqrt_num_runs = int(num_runs ** 0.5)
    num_episodes = 1000
    fig, axarr = plt.subplots(sqrt_num_runs, sqrt_num_runs, sharex=True, sharey=True)
    for i in range(num_runs):
        episode_scores = np.zeros(num_episodes)

        # Implicitly fills the episode_scores array - no need to store result of call
        policy = learn_with_mdp_model(env, num_episodes, episode_scores=episode_scores)

        # Print policy and avg score over 100 trials
        print('Policy: %r' % policy)
        print_avg_score(env, policy)

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
    plt.suptitle('Running Average Score Of Model-Based-Learning Agent \nOver First 1000 Training Episodes \n(over %s independent runs)' % num_runs)
    plt.savefig('pics/part5d.png', bbox_inches='tight')
    plt.show()

# Feel free to run your own debug code in main!
def main():
    env = gym.make('Stochastic-4x4-FrozenLake-v0')
    policy = learn_with_mdp_model(env)
    render_single(env, policy)


if __name__ == '__main__':
    # main()
    part5d_1()
    part5d_2()
