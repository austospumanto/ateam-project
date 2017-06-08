import logging
import random
import time

import numpy as np
import tqdm

# This registers the various FrozenLake maps by ID with Gym

logger = logging.getLogger(__name__)


class Config(object):
    num_episodes = 15000
    gamma = 0.95
    lr = 0.15
    e = 1.0
    decay_rate = 0.999
    audio_clip_mode = 'standard'
    num_mfcc = 13
    env_name = 'Deterministic-4x4-FrozenLake-v0'


class AsrQlearnAgent(object):
    a_to_d = ['left', 'down', 'right', 'up']

    def __init__(self, envs, state_recognizer):
        self.train_env = envs['train']
        self.val_env = envs['val']
        if 'test' in envs:
            self.test_env = envs['test']

        self.num_episodes = Config.num_episodes
        self.gamma = Config.gamma
        self.lr = Config.lr
        self.e = Config.e
        self.decay_rate = Config.decay_rate
        self.Q = np.zeros((self.train_env.nS, self.train_env.nA))
        self.state_recognizer = state_recognizer

    def policy(self, state):
        return int(np.argmax(self.Q[state]))

    def train(self, train_with_asr):
        Q = self.Q
        state_recognizer = self.state_recognizer
        env, num_episodes, gamma, lr, e, decay_rate = \
            self.train_env, self.num_episodes, self.gamma, self.lr, self.e, self.decay_rate
        episode_scores = np.zeros(num_episodes)
        for episode_idx in tqdm.tqdm(range(num_episodes)):
            # Choose a random starting state
            init_state_features = env.reset()
            if train_with_asr:
                cur_state = int(state_recognizer.recognize(init_state_features))
            else:
                cur_state = 0

            # Data structure for storing (s, a, r, s') tuples
            sars = []

            # Start the episode. The episode ends when we reach a terminal state (i.e. "done is True")
            done = False
            episode_reward = 0.0
            info = None
            steps_taken = 0
            max_steps = 15
            while not done and steps_taken < max_steps:
                # if train_with_asr and (info or steps_taken == 0):
                #     actual_state = 0 if steps_taken == 0 else info['state']
                #     policy_s = self.a_to_d[self.policy(actual_state)]
                #     logger.info('steps_taken=%d   state=%d   policy=%s' % (steps_taken, actual_state, policy_s))

                # Choose an action "epsilon-greedily" (where epsilon is the var "e")
                action = self.choose_egreedy_action(env, Q, cur_state, e)

                # Use env's transition probs to "choose" next state
                next_state_features, reward, done, info = env.step(action)
                steps_taken += 1

                if train_with_asr:
                    next_state = int(state_recognizer.recognize(next_state_features))
                else:
                    next_state = info['state']

                sars.append((cur_state, action, reward, next_state))

                # Move to the next state
                cur_state = next_state

                episode_reward += reward
                if train_with_asr and done:
                    wl = 'WIN' if reward > 0 else 'LOSS'
                    logger.info('steps_taken=%d %s' % (steps_taken, wl))
                    break

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

    def evaulate(self, env_to_eval, num_trials=100, verbose=False):
        episode_rewards = [self.run_trial(env_to_eval, verbose=verbose) for _ in xrange(num_trials)]
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
        """
        episode_reward = 0
        state = self.state_recognizer.recognize(env.reset())
        done = False
        steps_taken = 0
        info = None
        while not done:
            if info or steps_taken == 0:
                actual_state = 0 if steps_taken == 0 else info['state']
                policy_s = self.a_to_d[self.policy(actual_state)]
                logger.info('steps_taken=%d   state=%d   policy=%s' % (steps_taken, actual_state, policy_s))
            action = np.argmax(self.Q[state])
            state_features, reward, done, info = env.step(action)
            steps_taken += 1
            state = self.state_recognizer.recognize(state_features)
            episode_reward += reward
            if done:
                wl = 'WIN' if reward > 0 else 'LOSS'
                logger.info('steps_taken=%d %s' % (steps_taken, wl))
                break
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
        """

        episode_reward = 0
        state = int(self.state_recognizer.recognize(env.reset()))
        done = False
        while not done:
            env.render()
            time.sleep(0.5)  # Seconds between frames. Modify as you wish.
            action = np.argmax(self.Q[state])
            state_features, reward, done, _ = env.step(action)
            state = int(self.state_recognizer.recognize(state_features))
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
