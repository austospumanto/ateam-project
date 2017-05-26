import tensorflow as tf
from rl.models.DQN import DQN
from tensorflow.contrib import layers
import numpy as np


class AQN(DQN):
    def add_placeholders_op(self):
        """
        Adds placeholders to the graph

        These placeholders are used as inputs by the rest of the model building and will be fed
        data during training.  Note that when "None" is in a placeholder's shape, it's flexible
        (so we can use different batch sizes without rebuilding the model
        """

        ##############################################################
        """
        TODO: add placeholders:
              Remember that we stack 4 consecutive frames together, ending up with an input of shape
              (timesteps, 13, 4).
               - self.s: batch of states, type = float32
                         shape = (batch_size, audio timesteps, audio features, config.state_history)
               - self.a: batch of actions, type = int32
                         shape = (batch_size)
               - self.r: batch of rewards, type = float32
                         shape = (batch_size)
               - self.sp: batch of next states, type = float32
                         shape = (batch_size, audio timesteps, audio features, config.state_history)
               - self.done_mask: batch of done, type = bool
                         shape = (batch_size)
                         note that this placeholder contains bool = True only if we are done in 
                         the relevant transition
               - self.lr: learning rate, type = float32

        (Don't change the variable names!)

        HINT: variables from config are accessible with self.config.variable_name
              Also, you may want to use a dynamic dimension for the batch dimension.
              Check the use of None for tensorflow placeholders.

              you can also use the state_shape computed above.
        """
        ##############################################################
        ################YOUR CODE HERE (6-15 lines) ##################
        # TODO: Make sure we don't pass in the last 4 audio samples - only the current audio sample
        # ???: We changed this from uint8 to float32 - will this cause issues?
        s_shape = [None, None, self.config.num_mfcc]
        self.s = tf.placeholder(tf.float32, shape=s_shape, name='s')

        sl_shape = (None)
        self.sl = tf.placeholder(tf.int32, sl_shape, 'sl')

        a_shape = [None]
        self.a = tf.placeholder(tf.int32, shape=a_shape, name='a')

        r_shape = [None]
        self.r = tf.placeholder(tf.float32, shape=r_shape, name='r')

        # ???: We changed this from uint8 to float32 - will this cause issues?
        sp_shape = [None, None, self.config.num_mfcc]
        self.sp = tf.placeholder(tf.float32, shape=sp_shape, name='sp')

        slp_shape = (None)
        self.slp = tf.placeholder(tf.int32, slp_shape, 'slp')

        done_mask_shape = [None]
        self.done_mask = tf.placeholder(tf.bool, shape=done_mask_shape, name='done_mask')

        lr_shape = []
        self.lr = tf.placeholder(tf.float32, shape=lr_shape, name='lr')
        ##############################################################
        ######################## END YOUR CODE #######################

    def get_q_values_op(self, state, seq_len, scope, reuse=False):
        """
        Returns Q values for all actions

        Args:
            state: (tf tensor) 
                shape = (batch_size, img height, img width, nchannels)
            scope: (string) scope name, that specifies if target network or not
            reuse: (bool) reuse of variables in the scope

        Returns:
            out: (tf tensor) of shape = (batch_size, num_actions)
            :param seq_len: 
        """
        # this information might be useful
        num_actions = self.env.action_space.n
        ##############################################################
        """
        TODO: implement the computation of Q values like in the paper
                https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf
                https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf

              you may find the section "model architecture" of the appendix of the 
              nature paper particulary useful.

              store your result in out of shape = (batch_size, num_actions)

        HINT: you may find tensorflow.contrib.layers useful (imported)
              make sure to understand the use of the scope param

              you can use any other methods from tensorflow
              you are not allowed to import extra packages (like keras,
              lasagne, cafe, etc.)

        """
        ##############################################################
        ################ YOUR CODE HERE - 10-15 lines ################
        ### YOUR CODE HERE (~10-15 lines)
        cell = tf.contrib.rnn.GRUCell(self.config.num_hidden, reuse=reuse)

        # f is of shape [batch_s, max_timesteps, num_hidden]
        f, last_state = tf.nn.dynamic_rnn(cell, state,
                                          sequence_length=seq_len,
                                          dtype=tf.float32)
        with tf.variable_scope(scope):
            b = tf.get_variable("b", shape=(self.config.num_digits,),
                                initializer=tf.zeros_initializer())
            W = tf.get_variable("W", shape=(self.config.num_hidden, self.config.num_digits),
                                initializer=tf.contrib.layers.xavier_initializer())
            new_shape = [-1, tf.shape(f)[2]]
            matmul_and_add = tf.matmul(tf.reshape(f, new_shape), W) + b

            # logits is of shape [batch_s, max_timesteps, num_classes]
            new_logits_shape = [-1, tf.shape(f)[1], self.config.num_digits]
            logits = tf.reshape(matmul_and_add, new_logits_shape)
            logits_flattened = layers.flatten(logits)

            # TODO: Fix this section so the Tensorflow compiler doesn't complain about
            #       the last dimension of logits_flattened being None or replace this section
            #       with something else
            out = layers.fully_connected(
                inputs=logits_flattened,
                num_outputs=num_actions,
                activation_fn=None,
                reuse=reuse
            )
        ##############################################################
        ######################## END YOUR CODE #######################
        return out

    def build(self):
        """
        Build model by adding all necessary variables
        """
        # add placeholders
        self.add_placeholders_op()

        # compute Q values of state
        s = self.s  # self.process_state(self.s)
        sl = self.sl
        self.q = self.get_q_values_op(s, sl, scope="q", reuse=False)

        # compute Q values of next state
        sp = self.sp  # self.process_state(self.sp)
        slp = self.slp
        self.target_q = self.get_q_values_op(sp, slp, scope="target_q", reuse=False)

        # add update operator for target network
        self.add_update_target_op("q", "target_q")

        # add square loss
        self.add_loss_op(self.q, self.target_q)

        # add optmizer for the main networks
        self.add_optimizer_op("q")

    def update_step(self, t, replay_buffer, lr):
        """
        Performs an update of parameters by sampling from replay_buffer

        Args:
            t: number of iteration (episode and move)
            replay_buffer: ReplayBuffer instance .sample() gives batches
            lr: (float) learning rate
        Returns:
            loss: (Q - Q_target)^2
        """
        # TODO: Do sequence length padding similar to that in CS224S HW3
        #       so that all inputs have same shape (.., SAME_NUM_TIMESTAMPS, 13)
        # ???: Is the above TODO necessary?
        s_batch, a_batch, r_batch, sp_batch, done_mask_batch = \
            replay_buffer.sample(self.config.batch_size)

        # ???: Should all inputs have same num timestamps?
        sl_batch = np.array([s.shape[1] for s in s_batch])
        slp_batch = np.array([sp.shape[1] for sp in sp_batch])

        fd = {
            # inputs
            self.s: s_batch,
            self.sl: sl_batch,
            self.a: a_batch,
            self.r: r_batch,
            self.sp: sp_batch,
            self.slp: slp_batch,
            self.done_mask: done_mask_batch,
            self.lr: lr,
            # extra info
            self.avg_reward_placeholder: self.avg_reward,
            self.max_reward_placeholder: self.max_reward,
            self.std_reward_placeholder: self.std_reward,
            self.avg_q_placeholder: self.avg_q,
            self.max_q_placeholder: self.max_q,
            self.std_q_placeholder: self.std_q,
            self.eval_reward_placeholder: self.eval_reward,
        }

        loss_eval, grad_norm_eval, summary, _ = self.sess.run(
            [self.loss, self.grad_norm, self.merged, self.train_op], feed_dict=fd
        )

        # tensorboard stuff
        self.file_writer.add_summary(summary, t)

        return loss_eval, grad_norm_eval

    def get_best_action(self, state):
        """
        Return best action (used during testing/evaluation)

        Args:
            state: 4 consecutive observations from gym
        Returns:
            action: (int)
            action_values: (np array) q values for all actions
        """
        # Feed in a batch of size 1 to get a single best action for this state
        seq_len = state.shape[0]
        action_values = self.sess.run(self.q, feed_dict={self.s: [state], self.sl: [seq_len]})[0]
        return np.argmax(action_values), action_values
