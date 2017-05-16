# Compatibility imports
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import argparse
import math
import random
import os
# uncomment this line to suppress Tensorflow warnings
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
from six.moves import xrange as range

from speech.model_utils import *
import pdb
from time import gmtime, strftime

class Config:
    """Holds model hyperparams and data information.

    The config class is used to store various hyperparameters and dataset
    information parameters. Model objects are passed a Config() object at
    instantiation.
    """
    context_size = 0
    num_mfcc_features = 13
    num_final_features = num_mfcc_features * (2 * context_size + 1)

    batch_size = 16
    num_classes = 12   # 11 (TIDIGITS - 0-9 + oh) + 1 (blank) = 12
    num_hidden = 128

    num_epochs = 50
    l2_lambda = 0.0000001
    lr = 1e-4

class CTCModel():
    """
    Implements a recursive neural network with a single hidden layer attached to CTC loss.
    This network will predict a sequence of TIDIGITS (e.g. z1039) for a given audio wav file.
    """

    def add_placeholders(self):
        """Generates placeholder variables to represent the input tensors

        These placeholders are used as inputs by the rest of the model building and will be fed
        data during training.  Note that when "None" is in a placeholder's shape, it's flexible
        (so we can use different batch sizes without rebuilding the model).

        Adds following nodes to the computational graph:

        inputs_placeholder: Input placeholder tensor of shape (None, None, num_final_features), type tf.float32
        targets_placeholder: Sparse placeholder, type tf.int32. You don't need to specify shape dimension.
        seq_lens_placeholder: Sequence length placeholder tensor of shape (None), type tf.int32

        TODO: Add these placeholders to self as the instance variables
            self.inputs_placeholder
            self.targets_placeholder
            self.seq_lens_placeholder

        HINTS:
            - Use tf.sparse_placeholder(tf.int32) for targets_placeholder. This is required by TF's ctc_loss op.
            - Inputs is of shape [batch_size, max_timesteps, num_final_features], but we allow flexible sizes for
              batch_size and max_timesteps (hence the shape definition as [None, None, num_final_features].

        (Don't change the variable names)
        """
        inputs_placeholder = None
        seq_lens_placeholder = None

        ### YOUR CODE HERE (~3 lines)
        inputs_placeholder = tf.placeholder(tf.float32, (None, None, Config.num_final_features))
        seq_lens_placeholder = tf.placeholder(tf.int32, (None))
        ### END YOUR CODE

        self.inputs_placeholder = inputs_placeholder
        self.seq_lens_placeholder = seq_lens_placeholder


    def create_feed_dict(self, inputs_batch, seq_lens_batch):
        """Creates the feed_dict for the digit recognizer.

        A feed_dict takes the form of:

        feed_dict = {
                <placeholder>: <tensor of values to be passed for placeholder>,
                ....
        }

        Hint: The keys for the feed_dict should be a subset of the placeholder
                    tensors created in add_placeholders.

        Args:
            inputs_batch: A batch of input data.
            targets_batch: A batch of targets data.
            seq_lens_batch: A batch of seq_lens data.
        Returns:
            feed_dict: The feed dictionary mapping from placeholders to values.
        """
        feed_dict = {}

        ### YOUR CODE HERE (~3-4 lines)
        feed_dict[self.inputs_placeholder] = inputs_batch
        feed_dict[self.seq_lens_placeholder] = seq_lens_batch
        ### END YOUR CODE

        return feed_dict

    def add_prediction_op(self):
        """Applies a GRU RNN over the input data, then an affine layer projection. Steps to complete
        in this function:

        - Roll over inputs_placeholder with GRUCell, producing a Tensor of shape [batch_s, max_timestep,
          num_hidden].
        - Apply a W * f + b transformation over the data, where f is each hidden layer feature. This
          should produce a Tensor of shape [batch_s, max_timesteps, num_classes]. Set this result to
          "logits".

        Remember:
            * Use the xavier initialization for matrices (W, but not b).
            * W should be shape [num_hidden, num_classes]. num_classes for our dataset is 12
            * tf.contrib.rnn.GRUCell, tf.contrib.rnn.MultiRNNCell and tf.nn.dynamic_rnn are of interest
        """

        logits = None

        ### YOUR CODE HERE (~10-15 lines)
        # Initialize GRU
        cell = tf.contrib.rnn.GRUCell(Config.num_hidden)
        f, last_state = tf.nn.dynamic_rnn(cell, self.inputs_placeholder,
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
            logits = tf.reshape(matmul_and_add, [-1, f_shape[1], Config.num_classes])
        ### END YOUR CODE
        self.logits = tf.transpose(logits, perm=[1, 0, 2])

    def add_decoder_and_wer_op(self):
        """Setup the decoder and add the word error rate calculations here.

        Tip: You will find tf.nn.ctc_beam_search_decoder and tf.edit_distance methods useful here.
        Also, report the mean WER over the batch in variable wer

        """
        decoded_sequence = None

        ### YOUR CODE HERE (~2-3 lines)
        decoded, log_probs = tf.nn.ctc_beam_search_decoder(
            self.logits,
            self.seq_lens_placeholder,
            merge_repeated=False
        )
        # top_paths = 1, so take first (and only) decoded sequence
        decoded_sequence = tf.cast(decoded[0], tf.int32)
        ### END YOUR CODE

        self.decoded_sequence = decoded_sequence

    # This actually builds the computational graph
    def build(self):
        self.add_placeholders()
        self.add_prediction_op()
        self.add_decoder_and_wer_op()

    def __init__(self):
        self.build()
        self.saver = tf.train.Saver()
