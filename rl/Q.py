import logging
import random
import time

import gym
import numpy as np
import tensorflow as tf
import tqdm

from admin.config import project_config
from envs import MfccFrozenlake
from speech.models.DigitsRecognizer import DigitsRecognizer
from speech.models.StateRecognizer import StateRecognizer
from speech.models.CTCModel import CTCModel

# This registers the various FrozenLake maps by ID with Gym

logger = logging.getLogger(__name__)


def qlearning(env, num_episodes=15000, gamma=0.98, lr=0.08, e=0.5, decay_rate=0.9999, episode_scores=None):
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
        lr: float
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

        # If we'res running this as part of 5c, then record the scores
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


def shallow_qlearning(env, sess, model, num_episodes=15000, gamma=0.98, e=0.5, decay_rate=0.9999,
                      episode_scores=None):
    """
        Learn state-action values using the Q-learning algorithm with function approximation
        via a TensorFlow-specified shallow neural network with the epsilon-greedy exploration strategy.
        Update Q and NN parameters at the end of every episode.

        Parameters
        ----------
        env: gym.core.Environment
            Environment to compute Q function for. Must have nS, nA, and P as
            attributes.
        num_episodes: int
            Number of episodes of training.
        gamma: float
            Discount factor. Number in range [0, 1)
        lr: float
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
    inputs = model['inputs']
    action = model['action']
    Q_out = model['Q_out']
    next_Q = model['next_Q']
    W = model['W']
    gdo = model['gdo']

    rewards = []

    sess.run(tf.global_variables_initializer())
    for episode_idx in tqdm.tqdm(range(num_episodes)):
        curr_state = env.reset()
        done = False
        episode_reward = 0.0
        steps_taken = 0
        while not done:
            steps_taken += 1

            # Choose an action "epsilon-greedily" (where epsilon is the var "e")
            if isinstance(curr_state, int):
                curr_input_vector = _one_hot_state_vector(curr_state, env.nS)
            else:
                curr_input_vector = curr_state
            action_array, Q = sess.run([action, Q_out], feed_dict={inputs: curr_input_vector})
            if random.random() < e:
                action_array[0] = env.action_space.sample()

            # Use env's transition probs to "choose" next state
            next_state, reward, done, _ = env.step(action_array[0])

            if isinstance(next_state, int):
                next_input_vector = _one_hot_state_vector(next_state, env.nS)
            else:
                next_input_vector = next_state

            Q_1 = sess.run(Q_out, feed_dict={inputs: next_input_vector})
            target_Q = Q
            target_Q[0, action_array[0]] = reward + gamma * np.max(Q_1)

            sess.run([gdo, W], feed_dict={inputs: curr_input_vector, next_Q: target_Q})
            # Move to the next state
            curr_state = next_state
            episode_reward += reward

            # Decay the randomness of our action selection (i.e. increase greediness)
            e *= decay_rate
        rewards.append(episode_reward)
    percent_success = float(len([reward for reward in rewards if reward > 0.0])) / float(num_episodes)
    logger.info("Percent of succesful episodes: " + str(sum(rewards) / num_episodes * 100) + "%")


def qlearning_pretrained_asr(env_asr, state_recognizer, num_episodes=15000, gamma=0.98,
                             lr=0.08, e=0.5, decay_rate=0.9999, episode_scores=None):
    Q = np.zeros((env_asr.nS, env_asr.nA))
    for episode_idx in tqdm.tqdm(range(num_episodes)):
        # Choose a random starting state
        cur_state = int(state_recognizer.recognize(env_asr.reset()))

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
            next_state_features, reward, done, _ = env_asr.step(action)

            # Translate the auditory features of the next state to the state's index via ASR
            next_state = int(state_recognizer.recognize(next_state_features))

            # TODO: Should we hard-code the Goal state as being state 15? Should it know what
            #      state it was in at the time of episode termination?

            sars.append((cur_state, action, reward, next_state))

            # Move to the next state
            cur_state = next_state

            episode_reward += reward

        # If we'res running this as part of 5c, then record the scores
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


def _one_hot_state_vector(s, nS):
    return np.identity(nS)[s:s + 1]


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
    if state_recognizer:
        state = int(state_recognizer.recognize(env.reset()))
    else:
        state = env.reset()
    done = False
    while not done:
        env.render()
        time.sleep(0.5)  # Seconds between frames. Modify as you wish.
        action = np.argmax(Q[state])
        if state_recognizer:
            state_features, reward, done, _ = env.step(action)
            state = int(state_recognizer.recognize(state_features, verbose=verbose))
        else:
            state, reward, done, _ = env.step(action)
        episode_reward += reward

    logger.info("Episode reward: %f" % episode_reward)


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
    if state_recognizer:
        state = int(state_recognizer.recognize(env.reset()))
    else:
        state = env.reset()
    done = False
    while not done:
        action = np.argmax(Q[state])
        if state_recognizer:
            state_features, reward, done, _ = env.step(action)
            state = int(state_recognizer.recognize(state_features, verbose=verbose))
        else:
            state, reward, done, _ = env.step(action)
        episode_reward += reward
    if verbose:
        logger.info('Trial reward: %d\n' % episode_reward)
    return episode_reward


def _run_trial_network_q(env, sess, model):
    episode_reward = 0
    curr_state = env.reset()
    done = False
    action = model['action']
    inputs = model['inputs']
    while not done:
        if isinstance(curr_state, int):
            input_vector = _one_hot_state_vector(curr_state, env.nS)
        else:
            input_vector = curr_state
        action_array = sess.run([action], feed_dict={inputs: input_vector})
        next_state, reward, done, _ = env.step(action_array[0][0])
        episode_reward += reward
        curr_state = next_state
    return episode_reward


def print_avg_score(env, use_network=False, Q=None, sess=None, model=None, state_recognizer=None, verbose=False):
    # Average episode rewards over trials
    num_trials = 100
    episode_rewards = []
    for _ in tqdm.tqdm(xrange(num_trials)):
        if use_network:
            episode_rewards.append(_run_trial_network_q(env, sess, model))
        else:
            episode_rewards.append(_run_trial_q(env, Q, state_recognizer, verbose=verbose))
    avg_reward = np.average(episode_rewards)
    logger.info('Averge episode score/reward: %.3f' % avg_reward)


def vanilla_example():
    env = gym.make('Stochastic-4x4-FrozenLake-v0')
    Q = qlearning(env)
    print_avg_score(env, Q)
    render_single_q(env, Q)


def shallow_q_network():
    with tf.Session() as sess:
        env = gym.make('Stochastic-4x4-FrozenLake-v0')
        inputs = tf.placeholder(shape=[1, env.nS], dtype=tf.float32)
        W = tf.Variable(tf.random_uniform([env.nS, env.nA], 0, 0.01))
        Q_out = tf.matmul(inputs, W)
        action = tf.argmax(Q_out, 1)

        # Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
        next_Q = tf.placeholder(shape=[1, env.nA], dtype=tf.float32)
        loss = tf.reduce_sum(tf.square(next_Q - Q_out))
        gdo = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss)
        model = {'inputs': inputs, 'W': W, 'Q_out': Q_out, 'action': action, 'next_Q': next_Q, 'loss': loss, 'gdo': gdo}

        fw = tf.summary.FileWriter(project_config.tensorboard_logdir, sess.graph)

        shallow_qlearning(env, sess, model, num_episodes=2000)
        print_avg_score(env, use_network=True, sess=sess, model=model)


def shallow_q_network_with_asr():
    with tf.Session() as sess:
        env_asr = envs.MfccFrozenlake(gym.make('Stochastic-4x4-FrozenLake-v0'))
        inputs = tf.placeholder(shape=[None, env_asr.num_mfcc], dtype=tf.float32)
        W = tf.Variable(tf.random_uniform([env_asr.num_mfcc, env_asr.nA], 0, 0.01))
        Q_out = tf.matmul(inputs, W)
        action = tf.argmax(Q_out, 1)

        # Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
        next_Q = tf.placeholder(shape=[None, env_asr.nA], dtype=tf.float32)
        loss = tf.reduce_sum(tf.square(next_Q - Q_out))
        gdo = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss)
        model = {'inputs': inputs, 'W': W, 'Q_out': Q_out, 'action': action, 'next_Q': next_Q, 'loss': loss, 'gdo': gdo}

        fw = tf.summary.FileWriter(project_config.tensorboard_logdir, sess.graph)

        shallow_qlearning(env_asr, sess, model, num_episodes=5000)
        print_avg_score(env_asr, use_network=True, sess=sess, model=model)


def train_and_test_with_asr():
    env_asr = MfccFrozenlake.MfccFrozenlake(gym.make('Stochastic-4x4-FrozenLake-v0'))

    with tf.Session() as sess:
        model = CTCModel()
        ckpt = tf.train.get_checkpoint_state("res/cs224s/viggy_assign3/saved_models")
        v2_path = ckpt.model_checkpoint_path + ".index" if ckpt else ""

        if ckpt and (tf.gfile.Exists(ckpt.model_checkpoint_path) or tf.gfile.Exists(v2_path)):
            model.saver.restore(sess, ckpt.model_checkpoint_path)
            logger.info("Restored save properly.")

        digits_recognizer = DigitsRecognizer(model, sess)
        state_recognizer = StateRecognizer(env_asr, digits_recognizer)
        Q = qlearning_pretrained_asr(env_asr, state_recognizer)

        print_avg_score(env_asr, Q=Q, state_recognizer=state_recognizer)
        render_single_q(env_asr, Q=Q, state_recognizer=state_recognizer)


def test_with_asr():
    env = gym.make('Stochastic-4x4-FrozenLake-v0')
    env_asr = MfccFrozenlake.MfccFrozenlake(env)

    with tf.Session() as sess:
        model = CTCModel()
        ckpt = tf.train.get_checkpoint_state("res/cs224s/viggy_assign3/saved_models")
        v2_path = ckpt.model_checkpoint_path + ".index" if ckpt else ""

        if ckpt and (tf.gfile.Exists(ckpt.model_checkpoint_path) or tf.gfile.Exists(v2_path)):
            model.saver.restore(sess, ckpt.model_checkpoint_path)
            logger.info("Restored save properly.")

        digits_recognizer = DigitsRecognizer(model, sess)
        state_recognizer = StateRecognizer(env_asr, digits_recognizer)
        Q = qlearning(env)

        print_avg_score(env_asr, Q=Q, state_recognizer=state_recognizer, verbose=False)
        render_single_q(env_asr, Q=Q, state_recognizer=state_recognizer, verbose=True)
