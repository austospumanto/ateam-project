# MDP Value Iteration and Policy Iteratoin
# You might not need to use all parameters

from __future__ import print_function
import numpy as np
import gym
import time
from lake_envs import *


np.set_printoptions(precision=3)


# Part 4a
def policy_iteration(P, nS, nA, gamma=0.9, max_iteration=20, tol=1e-3, verbose=True):
    """
        Runs policy iteration.

        You should use the policy_evaluation and policy_improvement methods to
        implement this method.

        Parameters
        ----------
        P: dictionary
            It is from gym.core.Environment
            P[state][action] is tuples with (probability, nextstate, reward, terminal)
        nS: int
            number of states
        nA: int
            number of actions
        gamma: float
            Discount factor. Number in range [0, 1)
        max_iteration: int
            The maximum number of iterations to run before stopping. Feel free to change it.
        tol: float
            Determines when value function has converged.
        Returns:
        ----------
        value function: np.ndarray
        policy: np.ndarray
    """

    V_old = np.zeros(nS)
    policy_old = np.zeros(nS, dtype=int)

    ############################
    for i in range(max_iteration):
        V_old = policy_evaluation(P, nS, nA, policy_old, gamma)
        policy_new = policy_improvement(P, nS, nA, V_old, policy_old, gamma)
        V_new = policy_evaluation(P, nS, nA, policy_new, gamma)

        # Stop if the policy has converged
        if i != 0 and np.linalg.norm(V_new - V_old, ord=np.inf) <= tol:
            if verbose:
                print('Exited policy iteration after %s iterations' % str(i + 1))
            break

        V_old, policy_old = V_new, policy_new

    V, policy = V_new, policy_new
    ############################

    return V, policy


# Part 4b
def value_iteration(P, nS, nA, gamma=0.9, max_iteration=20, tol=1e-3, verbose=True):
    """
        Learn value function and policy by using value iteration method for a given
        gamma and environment.

        Parameters:
        ----------
        P: dictionary
            It is from gym.core.Environment
            P[state][action] is tuples with (probability, nextstate, reward, terminal)
        nS: int
            number of states
        nA: int
            number of actions
        gamma: float
            Discount factor. Number in range [0, 1)
        max_iteration: int
            The maximum number of iterations to run before stopping. Feel free to change it.
        tol: float
            Determines when value function has converged.
        Returns:
        ----------
        value function: np.ndarray
        policy: np.ndarray
    """

    V_old = np.zeros(nS)

    ############################
    for i in range(max_iteration):
        Q = _calculate_Q(P, nS, nA, V_old, gamma)
        V_new = np.max(Q, axis=1)
        # Stop if the value function has converged
        if np.linalg.norm(V_old - V_new, ord=np.inf) <= tol:
            if verbose:
                print('Exited value iteration after %s iterations' % str(i + 1))
            break
        V_old = V_new
    V = V_new
    policy = policy_improvement(P, nS, nA, V, None, gamma=0.9)
    ############################

    return V_old, policy


# Part 4a
def policy_evaluation(P, nS, nA, policy, gamma=0.9, max_iteration=1000, tol=1e-3):
    """
        Evaluate the value function from a given policy.

        Parameters
        ----------
        P: dictionary
            It is from gym.core.Environment
            P[state][action] is tuples with (probability, nextstate, reward, terminal)
        nS: int
            number of states
        nA: int
            number of actions
        gamma: float
            Discount factor. Number in range [0, 1)
        policy: np.array
            The policy to evaluate. Maps states to actions.
        max_iteration: int
            The maximum number of iterations to run before stopping. Feel free to change it.
        tol: float
            Determines when value function has converged.

        Returns
        -------
        value function: np.ndarray
            The value function from the given policy.
    """

    ############################
    V_old = np.zeros(nS)
    for _ in range(max_iteration):
        V_new = np.zeros(nS)
        for state_idx in range(nS):
            action_idx = policy[state_idx]
            R = 0.0
            future_reward = 0.0
            for probability, nextstate_idx, reward, _ in P[state_idx][action_idx]:
                R += probability * reward
                future_reward += probability * V_old[nextstate_idx]
            discounted_future_reward = gamma * future_reward
            V_new[state_idx] = R + discounted_future_reward
        if np.linalg.norm(V_new - V_old, ord=np.inf) <= tol:
            break
        V_old = V_new
    return V_new
    ############################


# Part 4a
def policy_improvement(P, nS, nA, value_from_policy, policy, gamma=0.9):
    """
        Given the value function from policy improve the policy.

        Parameters
        ----------
        P: dictionary
            It is from gym.core.Environment
            P[state][action] is tuples with (probability, nextstate, reward, terminal)
        nS: int
            number of states
        nA: int
            number of actions
        gamma: float
            Discount factor. Number in range [0, 1)
        value_from_policy: np.ndarray
            The value calculated from the policy
        policy: np.array
            The previous policy.

        Returns
        -------
        new policy: np.ndarray
            An array of integers. Each integer is the optimal action to take
            in that state according to the environment dynamics and the
            given value function.
    """

    ############################
    Q = _calculate_Q(P, nS, nA, value_from_policy, gamma)
    new_policy = np.argmax(Q, axis=1)
    assert new_policy.size == nS
    return new_policy
    ############################


def _calculate_Q(P, nS, nA, V, gamma=0.9):
    """
        Given the value function from the policy, compute

        Parameters
        ----------
        P: dictionary
            It is from gym.core.Environment
            P[state][action] is tuples with (probability, nextstate, reward, terminal)
        nS: int
            number of states
        nA: int
            number of actions
        gamma: float
            Discount factor. Number in range [0, 1)
        value_from_policy: np.ndarray
            The value calculated from the policy
        policy: np.array
            The previous policy.

        Returns
        -------
        new policy: np.ndarray
            An array of integers. Each integer is the optimal action to take
            in that state according to the environment dynamics and the
            given value function.
    """
    Q = np.zeros((nS, nA))
    for state_idx in range(nS):
        for action_idx in range(nA):
            R = 0.0
            future_reward = 0.0
            for probability, nextstate_idx, reward, _ in P[state_idx][action_idx]:
                R += probability * reward
                future_reward += probability * V[nextstate_idx]
            discounted_future_reward = gamma * future_reward
            Q[state_idx][action_idx] = R + discounted_future_reward
    return Q


def example(env):
    """
        Show an example of gym

        Parameters
        ----------
        env: gym.core.Environment
            Environment to play on. Must have nS, nA, and P as
            attributes.
    """

    env.seed(0); 
    from gym.spaces import prng; prng.seed(10) # for print the location
    # Generate the episode
    ob = env.reset()
    for t in range(100):
        env.render()
        a = env.action_space.sample()
        ob, rew, done, _ = env.step(a)
        if done:
            break
    assert done
    env.render()


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
    ob = env.reset()
    for t in range(100):
        env.render()
        time.sleep(0.5) # Seconds between frames. Modify as you wish.
        a = policy[ob]
        ob, rew, done, _ = env.step(a)
        episode_reward += rew
        if done:
            break
    assert done
    env.render()
    print("Episode reward: %f" % episode_reward)


# Part 4c
def part4c():
    print('Part 4c\n--------')
    env_names = ('Deterministic-4x4-FrozenLake-v0', 'Stochastic-4x4-FrozenLake-v0')
    for env_name in env_names[1:]:
        env = gym.make(env_name)
        # print(env.__doc__)
        # print("Here is an example of state, action, reward, and next state")
        # example(env)

        V_vi, p_vi = value_iteration(env.P, env.nS, env.nA, gamma=0.9, max_iteration=80, tol=1e-3)
        V_pi, p_pi = policy_iteration(env.P, env.nS, env.nA, gamma=0.9, max_iteration=80, tol=1e-3)

        print('Environment: "%s"' % env_name)
        print('------------------------------')
        print('Policy Iteration')
        print('  Optimal Value Function: %r' % V_pi)
        print('  Optimal Policy:         %r' % p_pi)
        print('Value Iteration')
        print('  Optimal Value Function: %r' % V_vi)
        print('  Optimal Policy:         %r' % p_vi)
        print('\n##########\n##########\n\n')
        render_single(env, p_pi)
        print('\n\n\n\n')
        render_single(env, p_vi)


# Feel free to run your own debug code in main!
# Play around with these hyperparameters.
if __name__ == "__main__":
    part4c()
