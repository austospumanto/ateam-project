import tensorflow as tf
from rl.models.DQN import DQN


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
               - self.s: batch of states, type = uint8
                         shape = (batch_size, audio timesteps, audio features, config.state_history)
               - self.a: batch of actions, type = int32
                         shape = (batch_size)
               - self.r: batch of rewards, type = float32
                         shape = (batch_size)
               - self.sp: batch of next states, type = uint8
                         shape = (batch_size, img height, img width, nchannels x config.state_history)
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
        # ???: We changed this from uint8 to float32 - will this cause issues?
        s_shape = [None, None, self.config.num_mfcc, self.config.state_history]
        self.s = tf.placeholder(tf.float32, shape=s_shape, name='s')

        a_shape = [None]
        self.a = tf.placeholder(tf.int32, shape=a_shape, name='a')

        r_shape = [None]
        self.r = tf.placeholder(tf.float32, shape=r_shape, name='r')

        # ???: We changed this from uint8 to float32 - will this cause issues?
        sp_shape = [None, None, self.config.num_mfcc, self.config.state_history]
        self.sp = tf.placeholder(tf.float32, shape=sp_shape, name='sp')

        done_mask_shape = [None]
        self.done_mask = tf.placeholder(tf.bool, shape=done_mask_shape, name='done_mask')

        lr_shape = []
        self.lr = tf.placeholder(tf.float32, shape=lr_shape, name='lr')
        ##############################################################
        ######################## END YOUR CODE #######################

    def get_q_values_op(self, state, scope, reuse=False):
        """
        Returns Q values for all actions

        Args:
            state: (tf tensor) 
                shape = (batch_size, img height, img width, nchannels)
            scope: (string) scope name, that specifies if target network or not
            reuse: (bool) reuse of variables in the scope

        Returns:
            out: (tf tensor) of shape = (batch_size, num_actions)
        """
        # this information might be useful
        num_actions = self.env.action_space.n
        out = state
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
        # Initialize GRU
        cell = tf.contrib.rnn.GRUCell(self.config.num_hidden)
        f, last_state = tf.nn.dynamic_rnn(cell, self.inputs_placeholder,
                                          sequence_length=self.seq_lens_placeholder,
                                          dtype=tf.float32)
        with tf.variable_scope(scope):
            b = tf.get_variable("b", shape=(self.config.num_digits,),
                                initializer=tf.zeros_initializer())
            W = tf.get_variable("W", shape=(self.config.num_hidden, self.config.num_digits),
                                initializer=tf.contrib.layers.xavier_initializer())
            new_shape = [-1, tf.shape(f)[2]]
            matmul_and_add = tf.matmul(tf.reshape(f, new_shape), W) + b
            logits = tf.reshape(matmul_and_add, [-1, tf.shape(f)[1], self.config.num_digits])
        ### END YOUR CODE



        with tf.variable_scope(scope):
            conv1 = layers.convolution2d(
                inputs=state,
                num_outputs=32,
                kernel_size=(8, 8),
                stride=4,
                activation_fn=tf.nn.relu,
                reuse=reuse
            )
            conv2 = layers.convolution2d(
                inputs=conv1,
                num_outputs=64,
                kernel_size=(4, 4),
                stride=2,
                activation_fn=tf.nn.relu,
                reuse=reuse
            )
            conv3 = layers.convolution2d(
                inputs=conv2,
                num_outputs=64,
                kernel_size=(3, 3),
                stride=1,
                activation_fn=tf.nn.relu,
                reuse=reuse
            )
            flattened_conv3 = layers.flatten(conv3)
            fc = layers.fully_connected(
                inputs=flattened_conv3,
                num_outputs=512,
                activation_fn=tf.nn.relu,
                reuse=reuse
            )
            out = layers.fully_connected(
                inputs=fc,
                num_outputs=num_actions,
                activation_fn=None,
                reuse=reuse
            )


        ##############################################################
        ######################## END YOUR CODE #######################
        return out