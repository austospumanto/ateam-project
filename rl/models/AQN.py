import tensorflow as tf
from rl.models.DQN import DQN
from tensorflow.contrib import layers
import numpy as np
from admin.config import project_config
import os
import glob


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

        sl_shape = [None]
        self.sl = tf.placeholder(tf.int32, sl_shape, 'sl')

        a_shape = [None]
        self.a = tf.placeholder(tf.int32, shape=a_shape, name='a')

        r_shape = [None]
        self.r = tf.placeholder(tf.float32, shape=r_shape, name='r')

        # ???: We changed this from uint8 to float32 - will this cause issues?
        sp_shape = [None, None, self.config.num_mfcc]
        self.sp = tf.placeholder(tf.float32, shape=sp_shape, name='sp')

        slp_shape = [None]
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
        num_actions = self.train_env.action_space.n
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
        cells = [
            tf.contrib.rnn.GRUCell(self.config.n_hidden_rnn, reuse=reuse)
            for _ in range(self.config.n_layers_rnn)
        ]
        multi_layer_cell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=False)

        with tf.variable_scope(scope):
            # f is of shape [batch_s, max_timesteps, num_hidden]
            rnn_outputs, last_states = tf.nn.dynamic_rnn(
                multi_layer_cell,
                state,
                sequence_length=seq_len,
                dtype=tf.float32
            )

            if self.config.n_hidden_fc:
                fc = layers.fully_connected(
                    inputs=last_states,
                    num_outputs=self.config.n_hidden_fc,
                    activation_fn=tf.nn.relu,
                    reuse=reuse,
                    weights_initializer=layers.variance_scaling_initializer()
                )

            logits_input = fc if self.config.n_hidden_fc else last_states

            logits = layers.fully_connected(
                inputs=logits_input,
                num_outputs=num_actions,
                activation_fn=None,
                reuse=reuse,
                weights_initializer=layers.variance_scaling_initializer()
            )
        ##############################################################
        ######################## END YOUR CODE #######################
        return logits

    def build(self):
        """
        Build model by adding all necessary variables
        """

        # compute Q values of state
        if self.mode == 'train':
            self.add_placeholders_op()
            s = self.s  # self.process_state(self.s)
            sl = self.sl
            self.q = self.get_q_values_op(s, sl, scope="q", reuse=False)

            sp = self.sp  # self.process_state(self.sp)
            slp = self.slp
            self.target_q = self.get_q_values_op(sp, slp, scope="target_q", reuse=False)

            # add update operator for target network
            self.add_update_target_op("q", "target_q")

            # add square loss
            self.add_loss_op(self.q, self.target_q)

            # add optmizer for the main networks
            self.add_optimizer_op("q")
        elif self.mode == 'test':
            self.s = tf.get_default_graph().get_tensor_by_name('s:0')
            self.sl = tf.get_default_graph().get_tensor_by_name('sl:0')
            self.q = tf.get_default_graph().get_tensor_by_name('q/fully_connected_1/BiasAdd:0')

            # NOTE: This might be useful if we wanted to continue training
            # self.sp = tf.get_default_graph().get_tensor_by_name('sp:0')
            # self.slp = tf.get_default_graph().get_tensor_by_name('slp:0')
            # self.target_q = tf.get_default_graph().get_tensor_by_name('target_q/fully_connected_1/BiasAdd:0')

    def restore(self):
        saved_run_dir = os.path.join(project_config.saved_runs_dir, self.config.run_name)
        model_weights_dir = os.path.join(saved_run_dir, 'model.weights')
        assert os.path.exists(model_weights_dir)
        saved_model_metas = glob.glob(os.path.join(model_weights_dir, '*.meta'))
        assert len(saved_model_metas) > 0
        ckpt_iters = sorted(
            [int(os.path.basename(fp).split('.')[0][1:]) for fp in saved_model_metas])
        self.logger.info('Found model checkpoints for iterations: ' + repr(ckpt_iters))
        latest_ckpt_iter = ckpt_iters[-1]
        latest_model_meta_path = [smm for smm in saved_model_metas
                                  if str(latest_ckpt_iter) in smm][0]

        # We load meta graph and restore weights
        self.logger.info('Loading model meta information from "%s"' % latest_model_meta_path)
        self.saver = tf.train.import_meta_graph(latest_model_meta_path)
        if not hasattr(self, 'sess'):
            self.sess = tf.Session()
        self.saver.restore(self.sess, tf.train.latest_checkpoint(model_weights_dir))
        self.logger.info([v.name for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)])
        self.build()

    def restore_from_ctc(self, restore_run_name):
        # Initialize the vars here so we can write over them
        self.initialize()

        # Write over the GRU cell variables with the values from the trained CTC Model
        saved_run_dir = os.path.join(project_config.speech_results_dir, restore_run_name)
        model_weights_dir = os.path.join(saved_run_dir, 'model.weights')
        assert os.path.exists(model_weights_dir)

        restore_var_names = [u'q/rnn/multi_rnn_cell/cell_0/gru_cell/gates/weights:0',
                             u'q/rnn/multi_rnn_cell/cell_0/gru_cell/gates/biases:0',
                             u'q/rnn/multi_rnn_cell/cell_0/gru_cell/candidate/weights:0',
                             u'q/rnn/multi_rnn_cell/cell_0/gru_cell/candidate/biases:0']

        pre_restore_norms = [
            np.linalg.norm(self.sess.run(tf.get_default_graph().get_tensor_by_name(var_name)))
            for var_name in restore_var_names
        ]

        # Restore all GRU vars from CTCModel to the AQN
        aqn_vars_to_restore = [tf.get_default_graph().get_tensor_by_name(vn) for vn in restore_var_names]
        saver = tf.train.Saver(aqn_vars_to_restore)
        saver.restore(self.sess, tf.train.latest_checkpoint(model_weights_dir))

        # Check to make sure the restored variables were not written over during
        # initialization of other vars
        post_restore_norms = [
            np.linalg.norm(self.sess.run(tf.get_default_graph().get_tensor_by_name(var_name)))
            for var_name in restore_var_names
        ]
        self.logger.info('Pre-restore norms' + repr(pre_restore_norms))
        self.logger.info('Post-restore norms' + repr(post_restore_norms))
        assert all([pre_norm != post_norm for pre_norm, post_norm in zip(pre_restore_norms, post_restore_norms)])
        self.initial_restored_var_norms = post_restore_norms

        # This is done in self.initialize() but we must do it again because we changed the weights
        self.sess.run(self.update_target_op)

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
        s_batch, sl_batch, a_batch, r_batch, sp_batch, slp_batch, done_mask_batch = \
            replay_buffer.sample(self.config.batch_size)

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

        if self.freeze_pretrained:
            restore_var_names = [u'q/rnn/multi_rnn_cell/cell_0/gru_cell/gates/weights:0',
                                 u'q/rnn/multi_rnn_cell/cell_0/gru_cell/gates/biases:0',
                                 u'q/rnn/multi_rnn_cell/cell_0/gru_cell/candidate/weights:0',
                                 u'q/rnn/multi_rnn_cell/cell_0/gru_cell/candidate/biases:0']
            restore_var_norms = [
                np.linalg.norm(self.sess.run(tf.get_default_graph().get_tensor_by_name(var_name)))
                for var_name in restore_var_names
            ]
            assert restore_var_norms == self.initial_restored_var_norms

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