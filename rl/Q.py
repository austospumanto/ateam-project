import random
import time
import gym
import numpy as np
import tensorflow as tf
import tqdm

# This registers the various FrozenLake maps by ID with Gym
from envs.lake_envs import *

from speech.model import CTCModel
from envs import envs
from rl.StateRecognizer import StateRecognizer
from speech.digits import DigitsRecognizer


def qlearning(env, num_episodes=15000, gamma=0.98, lr=0.08, e=0.5, decay_rate=0.9999,
              episode_scores=None):
    """
        Learn state-action values using the Q-learning algorithm with epsilon-greedy exploration
        strategy.
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
            # Get optimal value of next state (i.e. assume we act greedily from the next state
            # onwards)
            V_opt_ns = np.max(Q[next_state])

            # Calculate Q_samp_sa (i.e. "What was Q[s][a] for this particular sample/event")
            Q_samp_sa = reward + gamma * V_opt_ns

            # Update our overall estimate of Q[s][a]
            Q[cur_state][action] = (1 - lr) * Q[cur_state][action] + lr * Q_samp_sa

        # Decay the randomness of our action selection (i.e. increase greediness)
        e *= decay_rate

    return Q


def qlearning_pretrained_asr(env_asr, state_recognizer, num_episodes=15000, gamma=0.98,
                             lr=0.08, e=0.5, decay_rate=0.9999, episode_scores=None):
    Q = np.zeros((env_asr.nS, env_asr.nA))
    for episode_idx in tqdm.tqdm(range(num_episodes)):
        # Choose a random starting state
        cur_state = env_asr.reset()

        # Data structure for storing (s, a, r, s') tuples and
        sars = []

        # Data structure for

        # Start the episode. The episode ends when we reach a terminal state (i.e. "done is True")
        done = False
        episode_reward = 0.0
        while not done:
            # Choose an action "epsilon-greedily" (where epsilon is the var "e")
            action = _choose_egreedy_action(env_asr, cur_state, Q, e)

            # Use env's transition probs to "choose" next state
            # NOTE: We throw away the first return val (next_state) and last return val (info)
            #       because using these would be cheating
            _, next_state_features, reward, done, _ = env_asr.step(action)

            # Translate the auditory features of the next state to the state's index via ASR
            next_state = int(state_recognizer.recognize(next_state_features))

            # TODO: Should we hard-code the Goal state as being state 15? Should it know what
            #      state it was in at the time of episode termination?

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
            # Get optimal value of next state (i.e. assume we act greedily from the next state
            # onwards)
            V_opt_ns = np.max(Q[next_state])

            # Calculate Q_samp_sa (i.e. "What was Q[s][a] for this particular sample/event")
            Q_samp_sa = reward + gamma * V_opt_ns

            # Update our overall estimate of Q[s][a]
            Q[cur_state][action] = (1 - lr) * Q[cur_state][action] + lr * Q_samp_sa

        # Decay the randomness of our action selection (i.e. increase greediness)
        e *= decay_rate

    return Q


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
        a = np.argmax(Q[s])
        # random.choice([a for a, q_val in enumerate(Q[s]) if q_val == np.max(Q[s])])
    else:
        a = random.randint(0, env.nA - 1)
    return a


# Functions for testing
def render_single_q(env, Q, state_recognizer=None, verbose=False):
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
        if state_recognizer:
            _, state_features, reward, done, _ = env.step(action)
            state = int(state_recognizer.recognize(state_features, verbose=verbose))
        else:
            state, reward, done, _ = env.step(action)
        episode_reward += reward

    print "Episode reward: %f" % episode_reward


def _run_trial_q(env, Q, state_recognizer=None, verbose=False):
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
        if state_recognizer:
            actual_state, state_features, reward, done, _ = env.step(action)
            state = int(state_recognizer.recognize(state_features, verbose=True))
        else:
            state, reward, done, _ = env.step(action)
        if verbose:
            print 'Actual state: %d' % actual_state
        episode_reward += reward
    if verbose:
        print 'Trial reward: %d\n' % episode_reward
    return episode_reward


def print_avg_score(env, Q, state_recognizer=None):
    # Average episode rewards over trials
    num_trials = 100
    episode_rewards = [_run_trial_q(env, Q, state_recognizer) for _ in range(num_trials)]
    avg_reward = np.average(episode_rewards)
    print 'Averge episode score/reward: %.3f' % avg_reward


def vanilla_example():
    env = gym.make('Stochastic-4x4-FrozenLake-v0')
    Q = qlearning(env)
    print_avg_score(env, Q)
    render_single_q(env, Q)


def train_and_test_with_asr():
    env_asr = envs.MfccFrozenlake(gym.make('Stochastic-4x4-FrozenLake-v0'))

    with tf.Session() as sess:
        model = CTCModel()
        ckpt = tf.train.get_checkpoint_state("cs224s/viggy_assign3/saved_models")
        v2_path = ckpt.model_checkpoint_path + ".index" if ckpt else ""

        if ckpt and (tf.gfile.Exists(ckpt.model_checkpoint_path) or tf.gfile.Exists(v2_path)):
            model.saver.restore(sess, ckpt.model_checkpoint_path)
            print "Restored save properly."

        digits_recognizer = DigitsRecognizer(model, sess)
        state_recognizer = StateRecognizer(env_asr, digits_recognizer)
        Q = qlearning_pretrained_asr(env_asr, state_recognizer)

        print_avg_score(env_asr, Q, state_recognizer)
        render_single_q(env_asr, Q, state_recognizer)


def test_with_asr():
    env = gym.make('Stochastic-4x4-FrozenLake-v0')
    env_asr = envs.MfccFrozenlake(env)

    with tf.Session() as sess:
        model = CTCModel()
        ckpt = tf.train.get_checkpoint_state("cs224s/viggy_assign3/saved_models")
        v2_path = ckpt.model_checkpoint_path + ".index" if ckpt else ""

        if ckpt and (tf.gfile.Exists(ckpt.model_checkpoint_path) or tf.gfile.Exists(v2_path)):
            model.saver.restore(sess, ckpt.model_checkpoint_path)
            print "Restored save properly."

        digits_recognizer = DigitsRecognizer(model, sess)
        state_recognizer = StateRecognizer(env_asr, digits_recognizer)
        Q = qlearning(env)

        # print_avg_score(env_asr, Q, state_recognizer, verbose=True)
        render_single_q(env_asr, Q, state_recognizer, verbose=True)
