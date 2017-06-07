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


class Config(object):
    num_episodes = 15000
    gamma = 0.98
    lr = 0.08
    e = 0.5
    decay_rate = 0.9999


class AsrQlearnAgent(object):
    def __init__(self, envs, state_recognizer, run_config):
        self.train_env = envs['train']
        self.val_env = envs['val']
        if 'test' in envs:
            self.test_env = envs['test']

        self.num_episodes = run_config.num_episodes
        self.gamma = run_config.gamma
        self.lr = run_config.lr
        self.e = run_config.e
        self.decay_rate = run_config.decay_rate
        self.Q = np.zeros((self.train_env.nS, self.train_env.nA))
        self.state_recognizer = state_recognizer

    def train(self):
        Q = self.Q
        state_recognizer = self.state_recognizer
        env, num_episodes, gamma, lr, e, decay_rate = \
            self.train_env, self.num_episodes, self.gamma, self.lr, self.e, self.decay_rate
        episode_scores = np.zeros(num_episodes)
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
                action = self.choose_egreedy_action(env, Q, cur_state, e)

                # Use env's transition probs to "choose" next state
                next_state, reward, done, _ = env.step(action)

                next_state = int(state_recognizer.recognize(next_state_features))

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

        self.Q = Q
        return Q

    def evaulate(self, mode, num_trials=100, verbose=False):
        assert mode in ('train', 'val', 'test')
        env = None
        if mode == 'train':
            env = self.train_env
        elif mode == 'val':
            env = self.val_env
        elif mode == 'test':
            env = self.test_env

        num_trials = 100
        episode_rewards = []
        for _ in tqdm.tqdm(xrange(num_trials)):
            episode_rewards.append(run_trial(env, verbose=verbose))
        avg_reward = np.average(episode_rewards)
        logger.info('Averge episode score/reward: %.3f' % avg_reward)

    def run_trial(self, env, verbose=False):
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
        state = int(self.state_recognizer.recognize(env.reset()))
        done = False
        while not done:
            action = np.argmax(Q[state])
            state_features, reward, done, _ = env.step(action)
            state = int(self.state_recognizer.recognize(state_features, verbose=verbose))
            episode_reward += reward
        if verbose:
            logger.info('Trial reward: %d\n' % episode_reward)
        return episode_reward

    def render_episode(self, env, verbose=False):
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
        state = int(state_recognizer.recognize(env.reset()))
        done = False
        while not done:
            env.render()
            time.sleep(0.5)  # Seconds between frames. Modify as you wish.
            action = np.argmax(Q[state])
            state_features, reward, done, _ = env.step(action)
            state = int(self.state_recognizer.recognize(state_features, verbose=verbose))
            episode_reward += reward

        logger.info("Episode reward: %f" % episode_reward)

    @staticmethod
    def choose_egreedy_action(env, Q, s, e):
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
