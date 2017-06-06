import os
import shutil
import numpy as np
import glob
import tensorflow as tf
from admin.config import project_config

from QN import QN


class DQN(QN):
    """
    Abstract class for Deep Q Learning
    """
    def add_placeholders_op(self):
        raise NotImplementedError

    def get_q_values_op(self, *args, **kwargs):
        """
        set Q values, of shape = (batch_size, num_actions)
        """
        raise NotImplementedError

    def add_update_target_op(self, q_scope, target_q_scope):
        """
        update_target_op will be called periodically 
        to copy Q network weights to target Q network

        Remember that in DQN, we maintain two identical Q networks with
        2 different set of weights. In tensorflow, we distinguish them
        with two different scopes. One for the target network, one for the
        regular network. If you're not familiar with the scope mechanism
        in tensorflow, read the docs
        https://www.tensorflow.org/programmers_guide/variable_scope

        Periodically, we need to update all the weights of the Q network 
        and assign them with the values from the regular network. Thus,
        what we need to do is to build a tf op, that, when called, will 
        assign all variables in the target network scope with the values of 
        the corresponding variables of the regular network scope.

        Args:
            q_scope: (string) name of the scope of variables for q
            target_q_scope: (string) name of the scope of variables
                        for the target network
        """
        ##############################################################
        """
        TODO: add an operator self.update_target_op that assigns variables
            from target_q_scope with the values of the corresponding var 
            in q_scope

        HINT: you may find the following functions useful:
            - tf.get_collection
            - tf.assign
            - tf.group

        (be sure that you set self.update_target_op)
        """
        ##############################################################
        ################### YOUR CODE HERE - 5-10 lines #############
        q_vals = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=q_scope)
        target_q_vals = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=target_q_scope)
        q_vals.sort(key=lambda v: v.name)
        target_q_vals.sort(key=lambda v: v.name)
        assign_ops = [tf.assign(target_q_val, q_val) for q_val, target_q_val in zip(q_vals, target_q_vals)]
        self.update_target_op = tf.group(*assign_ops)
        ##############################################################
        ######################## END YOUR CODE #######################

    def add_loss_op(self, q, target_q):
        """
        Sets the loss of a batch, self.loss is a scalar

        Args:
            q: (tf tensor) shape = (batch_size, num_actions)
            target_q: (tf tensor) shape = (batch_size, num_actions)
        """
        # you may need this variable
        num_actions = self.train_env.action_space.n

        ##############################################################
        """
        TODO: The loss for an example is defined as:
                Q_samp(s) = r if done
                          = r + gamma * max_a' Q_target(s', a')
                loss = (Q_samp(s) - Q(s, a))^2 

              You need to compute the average of the loss over the minibatch
              and store the resulting scalar into self.loss

        HINT: - config variables are accessible through self.config
              - you can access placeholders like self.a (for actions)
                self.r (rewards) or self.done_mask for instance
             - you may find the following functions useful
                    - tf.cast
                    - tf.reduce_max / reduce_sum
                    - tf.one_hot
                    - ...

        (be sure that you set self.loss)
        """
        ##############################################################
        ##################### YOUR CODE HERE - 4-5 lines #############
        # Shape is (batch_size,)
        done_mask = tf.cast(tf.logical_not(self.done_mask), tf.float32)

        # Shape is (batch_size,)
        target_q_star = tf.reduce_max(target_q, axis=1)

        # Shape is (batch_size,)
        Q_samp_s = tf.add(
            self.r,
            tf.multiply(
                done_mask,
                tf.multiply(
                    self.config.gamma,
                    target_q_star
                )
            )
        )

        # Shape is (batch_size, num_actions)
        action_mask = tf.one_hot(self.a, depth=num_actions, dtype=tf.float32)

        # Shape is (batch_size,)
        Q_sa = tf.reduce_sum(tf.multiply(q, action_mask), axis=1)

        # Shape is ()
        loss = tf.reduce_mean(
            tf.square(
                tf.subtract(
                    Q_samp_s,
                    Q_sa
                )
            )
        )

        # L2 Regularization
        l2_cost = 0.0
        for variable in tf.trainable_variables():
            name = variable.name
            shape = variable.get_shape().as_list()
            # Avoid biases in L2 loss
            if len(shape) != 1 and "biases" not in name:
                l2_cost += tf.nn.l2_loss(variable)

        self.loss = loss + self.config.l2_lambda * l2_cost
        ##############################################################
        ######################## END YOUR CODE #######################

    def add_optimizer_op(self, scope):
        """
        Set self.train_op and self.grad_norm
        """

        ##############################################################
        """
        TODO: 1. get Adam Optimizer (remember that we defined self.lr in the placeholders
                section)
              2. compute grads wrt to variables in scope for self.loss
              3. clip the grads by norm with self.config.clip_val if self.config.grad_clip
                is True
              4. apply the gradients and store the train op in self.train_op
               (sess.run(train_op) must update the variables)
              5. compute the global norm of the gradients and store this scalar
                in self.grad_norm

        HINT: you may find the following functinos useful
            - tf.get_collection
            - optimizer.compute_gradients
            - tf.clip_by_norm
            - optimizer.apply_gradients
            - tf.global_norm

             you can access config variable by writing self.config.variable_name

        (be sure that you set self.train_op and self.grad_norm)
        """
        ##############################################################
        #################### YOUR CODE HERE - 8-12 lines #############
        # Get the Adam Optimizer
        adam_opt = tf.train.AdamOptimizer(learning_rate=self.lr, name='Adam')

        # Get all trainable values in the graph for this scope
        scope_train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)

        # Compute gradients with respect to variables in scope for self.loss
        grads_and_vars = adam_opt.compute_gradients(self.loss, scope_train_vars)

        # Clip the grads by norm
        if self.config.grad_clip:
            grads_and_vars = [
                (tf.clip_by_norm(grad, self.config.clip_val), var)
                for grad, var in grads_and_vars
            ]

        # Apply the gradients (training) and store the training op
        self.train_op = adam_opt.apply_gradients(grads_and_vars)

        # Compute the global norm of the gradients and store the scalar
        self.grad_norm = tf.global_norm([grad for grad, _ in grads_and_vars])
        ##############################################################
        ######################## END YOUR CODE #######################

    def process_state(self, state):
        # TODO: Is this where we go from state int to MFCCs?
        """
        Processing of state

        State placeholders are tf.uint8 for fast transfer to GPU
        Need to cast it to float32 for the rest of the tf graph.

        Args:
            state: node of tf graph of shape = (batch_size, height, width, nchannels)
                    of type tf.uint8.
                    if , values are between 0 and 255 -> 0 and 1
        """
        state = tf.cast(state, tf.float32)
        state /= self.config.high

        return state

    def build(self):
        """
        Build model by adding all necessary variables
        """
        # add placeholders
        self.add_placeholders_op()

        # compute Q values of state
        s = self.process_state(self.s)
        self.q = self.get_q_values_op(s, scope="q", reuse=False)

        # compute Q values of next state
        sp = self.process_state(self.sp)
        self.target_q = self.get_q_values_op(sp, scope="target_q", reuse=False)

        # add update operator for target network
        self.add_update_target_op("q", "target_q")

        # add square loss
        self.add_loss_op(self.q, self.target_q)

        # add optmizer for the main networks
        self.add_optimizer_op("q")

    def initialize(self):
        """
        Assumes the graph has been constructed
        Creates a tf Session and run initializer of variables
        """
        # Dumps all configuration to self.config.config_path
        with open(self.config.config_path, 'wb') as configfile:
            # Dump config and current copy of AQN.py to disk
            for key, value in self.config.__dict__.iteritems():
                if not key.startswith('__') and not key.endswith('__'):
                    configfile.write('%s=%s\n' % (str(key), str(value)))
        
        # Saves a copy of *QN.py at the time of run to self.config_*qn_dst_path
        shutil.copy2(self.config.qn_src_path, self.config.qn_dst_path)
        shutil.copy2(self.config.dqn_src_path, self.config.dqn_dst_path)
        shutil.copy2(self.config.aqn_src_path, self.config.aqn_dst_path)
        shutil.copy2(self.config.project_cfg_src_path, self.config.project_cfg_dst_path)

        # create tf session
        self.sess = tf.Session()

        # tensorboard stuff
        self.add_summary()

        # initiliaze all variables
        init = tf.global_variables_initializer()
        self.sess.run(init)

        # synchronise q and target_q networks
        self.sess.run(self.update_target_op)

        # for saving networks weights
        self.saver = tf.train.Saver()
       
    def add_summary(self):
        """
        Tensorboard stuff
        """
        # extra placeholders to log stuff from python
        self.avg_reward_placeholder = tf.placeholder(tf.float32, shape=(), name="avg_reward")
        self.max_reward_placeholder = tf.placeholder(tf.float32, shape=(), name="max_reward")
        self.std_reward_placeholder = tf.placeholder(tf.float32, shape=(), name="std_reward")

        self.avg_q_placeholder  = tf.placeholder(tf.float32, shape=(), name="avg_q")
        self.max_q_placeholder  = tf.placeholder(tf.float32, shape=(), name="max_q")
        self.std_q_placeholder  = tf.placeholder(tf.float32, shape=(), name="std_q")

        self.eval_reward_placeholder = tf.placeholder(tf.float32, shape=(), name="eval_reward")

        # add placeholders from the graph
        tf.summary.scalar("loss", self.loss)
        tf.summary.scalar("grads norm", self.grad_norm)

        # extra summaries from python -> placeholders
        tf.summary.scalar("Avg Reward", self.avg_reward_placeholder)
        tf.summary.scalar("Max Reward", self.max_reward_placeholder)
        tf.summary.scalar("Std Reward", self.std_reward_placeholder)

        tf.summary.scalar("Avg Q", self.avg_q_placeholder)
        tf.summary.scalar("Max Q", self.max_q_placeholder)
        tf.summary.scalar("Std Q", self.std_q_placeholder)

        tf.summary.scalar("Eval Reward", self.eval_reward_placeholder)
            
        # logging
        self.merged = tf.summary.merge_all()
        self.file_writer = tf.summary.FileWriter(self.config.output_path, self.sess.graph)

    def save(self, global_step=None):
        """
        Saves session
        """
        if not os.path.exists(self.config.model_output_path):
            os.makedirs(self.config.model_output_path)

        self.saver.save(self.sess, self.config.model_output_path, global_step=global_step)

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
        self.saver.restore(self.sess, tf.train.latest_checkpoint(model_weights_dir))
        # self.build()
        # self.sess.run(tf.global_variables_initializer())
        # self.sess.run(self.update_target_op)

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
        action_values = self.sess.run(self.q, feed_dict={self.s: [state]})[0]
        return np.argmax(action_values), action_values

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

        s_batch, a_batch, r_batch, sp_batch, done_mask_batch = replay_buffer.sample(
            self.config.batch_size)

        fd = {
            # inputs
            self.s: s_batch,
            self.a: a_batch,
            self.r: r_batch,
            self.sp: sp_batch,
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

    def update_target_params(self):
        """
        Update parametes of Q' with parameters of Q
        """
        self.sess.run(self.update_target_op)
