# Compatibility imports
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from speech.models.model_utils import *
import math
import tensorflow as tf
from admin.config import project_config
import os
import shutil
import datetime

# uncomment this line to suppress Tensorflow warnings
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class Config:
    """Holds model hyperparams and data information.

    The config class is used to store various hyperparameters and dataset
    information parameters. Model objects are passed a Config() object at
    instantiation.
    """
    context_size = 0
    num_mfcc_features = 13
    num_final_features = num_mfcc_features * (2 * context_size + 1)

    batch_size = 64
    num_classes = 11   # 11 (TIDIGITS - 0-9 + oh) + 1 (blank) = 12
    num_hidden = 128
    num_layers_rnn = 1

    num_epochs = 500
    l2_lambda = 0.0000001
    lr = 5e-4

    tidigits_subset = 'fl'

    # Training vars
    save_every = 5
    log_every = 5

    def __init__(self, run_name):
        self.run_name = run_name
        self.results_path = os.path.join(project_config.base_dir, 'speech', 'results')
        self.run_results_path = os.path.join(
            self.results_path,
            '%s-%s' % (datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'), run_name))
        if not os.path.exists(self.run_results_path):
            os.makedirs(self.run_results_path)
        self.logs_dir = os.path.join(self.run_results_path, 'logs')
        self.model_outputs_path = os.path.join(self.run_results_path, 'model.weights/')

        ctcmodel_src_path = os.path.join(project_config.base_dir, 'speech', 'models', 'CTCModel.py')
        ctcmodel_dst_path = os.path.join(self.run_results_path, 'CTCModel.py')
        shutil.copy2(ctcmodel_src_path, ctcmodel_dst_path)

        project_cfg_src_path = os.path.join(project_config.base_dir, 'admin', 'config.py')
        project_cfg_dst_path = os.path.join(self.run_results_path, 'project_config.py')
        shutil.copy2(project_cfg_src_path, project_cfg_dst_path)

        commands_src_path = os.path.join(project_config.base_dir, 'speech', 'commands.py')
        commands_dst_path = os.path.join(self.run_results_path, 'commands.py')
        shutil.copy2(commands_src_path, commands_dst_path)


class CTCModel(object):
    """
    Implements a recursive neural network with a single hidden layer attached to CTC loss.
    This network will predict a sequence of TIDIGITS (e.g. z1039) for a given audio wav file.
    """
    def __init__(self):
        self.build()

    def add_placeholders(self):
        """Generates placeholder variables to represent the input tensors

        Adds following nodes to the computational graph:

        inputs_placeholder: Input placeholder tensor of shape (None, None, num_final_features), type tf.float32
        targets_placeholder: Sparse placeholder, type tf.int32. You don't need to specify shape dimension.
        seq_lens_placeholder: Sequence length placeholder tensor of shape (None), type tf.int32

        (Don't change the variable names)
        """
        self.inputs_placeholder = tf.placeholder(tf.float32, (None, None, Config.num_final_features))
        self.seq_lens_placeholder = tf.placeholder(tf.int32, (None,))
        self.targets_placeholder = tf.sparse_placeholder(tf.int32)

    def create_feed_dict(self, inputs_batch, targets_batch, seq_lens_batch):
        """Creates the feed_dict for the digit recognizer.

        Args:
            inputs_batch: A batch of input data.
            targets_batch: A batch of targets data.
            seq_lens_batch: A batch of seq_lens data.
        Returns:
            feed_dict: The feed dictionary mapping from placeholders to values.
        """
        feed_dict = {
            self.inputs_placeholder: inputs_batch,
            self.targets_placeholder: targets_batch,
            self.seq_lens_placeholder: seq_lens_batch
        }
        return feed_dict

    def add_prediction_op(self):
        """Applies a GRU RNN over the input data, then an affine layer projection. Steps to complete
        in this function:

        Remember:
            * Use the xavier initialization for matrices (W, but not b).
            * W should be shape [num_hidden, num_classes]. num_classes for our dataset is 12
            * tf.contrib.rnn.GRUCell, tf.contrib.rnn.MultiRNNCell and tf.nn.dynamic_rnn are of interest
        """
        cells = [
            tf.contrib.rnn.GRUCell(Config.num_hidden)
            for _ in range(Config.num_layers_rnn)
        ]
        multi_layer_cell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=False)

        # Use this scope so we can do transfer learning from AQN
        with tf.variable_scope('q'):
            f, last_state = tf.nn.dynamic_rnn(multi_layer_cell, self.inputs_placeholder,
                                              sequence_length=self.seq_lens_placeholder,
                                              dtype=tf.float32)
        with tf.variable_scope("global"):
            b = tf.get_variable("b", shape=(Config.num_classes,),
                                initializer=tf.zeros_initializer())
            W = tf.get_variable("W", shape=(Config.num_hidden, Config.num_classes),
                                initializer=tf.contrib.layers.xavier_initializer())
            f_shape = tf.shape(f)
            new_shape = [-1, f_shape[2]]
            matmul_and_add = tf.matmul(tf.reshape(f, new_shape), W) + b
            self.logits = tf.reshape(matmul_and_add, [-1, f_shape[1], Config.num_classes])

    def add_decoder_and_wer_op(self):
        """Setup the decoder and add the word error rate calculations here.

        Tip: You will find tf.nn.ctc_beam_search_decoder and tf.edit_distance methods useful here.
        Also, report the mean WER over the batch in variable wer

        """
        decoded, log_probs = tf.nn.ctc_beam_search_decoder(
            self.logits,
            self.seq_lens_placeholder,
            merge_repeated=False
        )
        # top_paths = 1, so take first (and only) decoded sequence
        self.decoded_sequence = tf.cast(decoded[0], tf.int32)

        edit_dist = tf.edit_distance(
            self.decoded_sequence,
            self.targets_placeholder
        )
        self.wer = tf.reduce_mean(edit_dist)

        tf.summary.scalar("loss", self.loss)
        tf.summary.scalar("wer", self.wer)

    def add_loss_op(self):
        """Adds Ops for the loss function to the computational graph.

        - Use tf.nn.ctc_loss to calculate the CTC loss for each example in the batch. You'll need self.logits,
          self.targets_placeholder, self.seq_lens_placeholder for this. Set variable ctc_loss to
          the output of tf.nn.ctc_loss
        - You will need to first tf.transpose the data so that self.logits is shaped [max_timesteps, batch_s,
          num_classes].
        - Configure tf.nn.ctc_loss so that identical consecutive labels are allowed
        - Compute L2 regularization cost for all trainable variables. Use tf.nn.l2_loss(var).

        """
        # Transpose self.logits from here on out
        # See https://piazza.com/class/j0qbtwsddft6na?cid=352
        self.logits = tf.transpose(self.logits, perm=[1, 0, 2])
        ctc_loss = tf.nn.ctc_loss(
            self.targets_placeholder,
            self.logits,
            self.seq_lens_placeholder,
            preprocess_collapse_repeated=False,
            ctc_merge_repeated=False
        )

        l2_cost = 0.0
        for variable in tf.trainable_variables():
            name = variable.name
            shape = variable.get_shape().as_list()
            # Avoid biases in L2 loss
            if shape != [Config.num_classes] and "biases" not in name:
                l2_cost += tf.nn.l2_loss(variable)

        # Remove inf cost training examples (no path found, yet)
        loss_without_invalid_paths = tf.boolean_mask(ctc_loss, tf.less(ctc_loss, tf.constant(10000.)))
        self.num_valid_examples = tf.cast(tf.shape(loss_without_invalid_paths)[0], tf.int32)
        cost = tf.reduce_mean(loss_without_invalid_paths)

        self.loss = Config.l2_lambda * l2_cost + cost

    def add_training_op(self):
        """Sets up the training Ops.
        """
        self.optimizer = tf.train.AdamOptimizer(learning_rate=Config.lr).minimize(self.loss)

    def add_summary_op(self):
        self.merged_summary_op = tf.summary.merge_all()

    def build(self):
        # This actually builds the computational graph
        self.add_placeholders()
        self.add_prediction_op()
        self.add_loss_op()
        self.add_training_op()
        self.add_decoder_and_wer_op()
        self.add_summary_op()

    def train_on_batch(self, session, train_inputs_batch, train_targets_batch, train_seq_len_batch, train=True):
        feed = self.create_feed_dict(train_inputs_batch, train_targets_batch, train_seq_len_batch)
        batch_cost, wer, batch_num_valid_ex, summary = session.run(
            [self.loss, self.wer, self.num_valid_examples, self.merged_summary_op],
            feed)

        if math.isnan(batch_cost):  # basically all examples in this batch have been skipped
            return 0
        if train:
            _ = session.run([self.optimizer], feed)

        return batch_cost, wer, summary

    def print_results(self, session, train_inputs_batch, train_targets_batch, train_seq_len_batch):
        train_feed = self.create_feed_dict(train_inputs_batch, train_targets_batch, train_seq_len_batch)
        train_first_batch_preds = session.run(self.decoded_sequence, feed_dict=train_feed)
        compare_predicted_to_true(train_first_batch_preds, train_targets_batch)
