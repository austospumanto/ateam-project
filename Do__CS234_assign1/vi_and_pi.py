### MDP Value Iteration and Policy Iteratoin
# You might not need to use all parameters

import numpy as np
import random
import gym
import time
from lake_envs import *

np.set_printoptions(precision=3)


def value_iteration(P, nS, nA, gamma=0.9, max_iteration=20, tol=1e-3):
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
    V = np.zeros(nS)
    policy = np.zeros(nS, dtype=int)
    ############################
    # YOUR IMPLEMENTATION HERE #
    ############################
    counter = 0
    V_new = np.zeros(nS)

    # Loop to get value convergence
    while counter < max_iteration:
        counter += 1

        for i in range(nS):
            best_value = 0.0
            for j in range(nA):
                possibilities = P[i][j]
                comp_value = 0.0
                for k in range(len(possibilities)):
                    prob_action = possibilities[k]
                    stochastic_value = prob_action[0] * (prob_action[2] + (gamma * V[prob_action[1]]))
                    comp_value += stochastic_value
                if comp_value > best_value:
                    best_value = comp_value
            V_new[i] = best_value

        diff = np.amax(np.absolute(V - V_new))
        # Break out of value improvement if that maximum difference is less than tolerance
        if diff < tol: break
        V = np.copy(V_new)

    V = np.copy(V_new)

    # Extract optimal policy

    for i in range(nS):
        best_value = 0.0
        best_action = 0
        for j in range(nA):
            comp_value = 0.0
            possibilities = P[i][j]
            for k in range(len(possibilities)):
                prob_action = possibilities[k]
                stochastic_value = prob_action[0] * (prob_action[2] + (gamma * V[prob_action[1]]))
                comp_value += stochastic_value
            if comp_value > best_value:
                best_value = comp_value
                best_action = j
        policy[i] = best_action

    return V, policy


def policy_evaluation(P, nS, nA, policy, gamma=0.9, max_iteration=1000, tol=1e-3):
    """Evaluate the value function from a given policy.

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
    # YOUR IMPLEMENTATION HERE #
    ############################
    V = np.zeros(nS)
    counter = 0

    while counter < max_iteration:
        counter += 1
        V_new = np.copy(V)
        # Iterating through every state in value vector
        for i in range(nS):
            possibilities = P[i][policy[i]]
            comp_value = 0.0
            for j in range(len(possibilities)):
                prob_action = possibilities[j]
                # Sum of probability of next state * (reward + gamma * value_new_state)
                stochastic_value = prob_action[0] * (prob_action[2] + (gamma * V[prob_action[1]]))
                comp_value += stochastic_value
            V_new[i] = comp_value

        # Find maximum absolute elementwise difference between the old and new value vectors

        diff = np.amax(np.absolute(V - V_new))
        # Break out of value improvement if that maximum difference is less than tolerance
        if diff < tol: return V_new
        V = np.copy(V_new)

    return V


def policy_improvement(P, nS, nA, value_from_policy, policy, gamma=0.9):
    """Given the value function from policy improve the policy.

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
    # YOUR IMPLEMENTATION HERE #
    ############################
    Q_matrix = np.array([np.zeros(nS), np.zeros(nS), np.zeros(nS), np.zeros(nS)])

    for i in range(nA):
        for j in range(nS):
            possibilities = P[j][i]
            comp_value = 0.0
            for k in range(len(possibilities)):
                prob_action = possibilities[k]
                stochastic_value = prob_action[0] * (prob_action[2] + (gamma * value_from_policy[prob_action[1]]))
                comp_value += stochastic_value
            Q_matrix[i][j] = comp_value

    new_policy = np.argmax(Q_matrix, axis=0)

    return new_policy


def policy_iteration(P, nS, nA, gamma=0.9, max_iteration=20, tol=1e-3):
    """Runs policy iteration.

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
    V = np.zeros(nS)
    policy = np.zeros(nS, dtype=int)
    ############################
    # YOUR IMPLEMENTATION HERE #
    ############################

    # Randomly initialize policies
    for i in range(nS):
        policy[i] = random.randint(0, nA - 1)

    counter = 0

    while counter < max_iteration:
        counter += 1
        V_pi = policy_evaluation(P, nS, nA, policy, gamma)
        new_policy = policy_improvement(P, nS, nA, V_pi, policy, gamma)

        if np.array_equal(policy, new_policy): break

        V = np.copy(V_pi)
        policy = np.copy(new_policy)

    return V, policy


def example(env):
    """Show an example of gym
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
    env.render();

def render_single(env, policy):
    """Renders policy once on environment. Watch your agent play!

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
    env.render();
    print "Episode reward: %f" % episode_reward


# Feel free to run your own debug code in main!
# Play around with these hyperparameters.
if __name__ == "__main__":
    env = gym.make("Deterministic-4x4-FrozenLake-v0")
    # print env.__doc__
    # print "Here is an example of state, action, reward, and next state"
    # example(env)
    V_vi, p_vi = value_iteration(env.P, env.nS, env.nA, gamma=0.9, max_iteration=20, tol=1e-3)
    V_pi, p_pi = policy_iteration(env.P, env.nS, env.nA, gamma=0.9, max_iteration=20, tol=1e-3)

    print "Value iteration (deterministic):"
    print p_vi
    # render_single(env, p_vi)
    # print "Converged in %d" % c_vi
    print "\n"

    print "Policy iteration (deterministic):"
    print p_pi
    # render_single(env, p_pi)
    # print "Converged in %d" % c_pi

    envs = gym.make("Stochastic-4x4-FrozenLake-v0")

    Vs_vi, ps_vi = value_iteration(envs.P, envs.nS, envs.nA, gamma=0.9, max_iteration=20, tol=1e-3)
    Vs_pi, ps_pi = policy_iteration(envs.P, envs.nS, envs.nA, gamma=0.9, max_iteration=20, tol=1e-3)

    print "Value iteration (stochastic):"
    print ps_vi
    # render_single(envs, ps_vi)
    # print "Converged in %d" % cs_vi
    print "\n"

    print "Policy iteration (stochastic):"
    print ps_pi
    # render_single(envs, ps_pi)
    # print "Converged in %d" % cs_pi

