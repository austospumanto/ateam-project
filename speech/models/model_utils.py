from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange as range

import numpy as np
import tensorflow as tf
import cPickle as pickle
from data.tidigits import tidigits_db
import logging
logger = logging.getLogger(__name__)


def sparse_tuple_from(sequences, dtype=np.int32):
    """Create a sparse representention of x.
    Args:
        sequences: a list of lists of type dtype where each element is a sequence
    Returns:
        A tuple with (indices, values, shape)
    """
    indices = []
    values = []

    for n, seq in enumerate(sequences):
        indices.extend(zip([n] * len(seq), range(len(seq))))
        values.extend(seq)
    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1] + 1], dtype=np.int64)

    return indices, values, shape


def label_from_sparse_tensor(sparse_tensor):
    inv_index_mapping = {v: k for k, v in tidigits_db.index_mapping.items()}
    dense_tensor = tf.sparse_tensor_to_dense(sparse_tensor, default_value=-1).eval()
    label = "".join([inv_index_mapping[ch] for ch in dense_tensor[0] if ch != -1])
    label = label.replace('z', '0')
    label = label.replace('_', '')
    label = label.replace('o', '0')
    return label


def label_from_dense_tensor(dense_tensor):
    inv_index_mapping = {v: k for k, v in tidigits_db.index_mapping.items()}
    label = "".join([inv_index_mapping[ch] for ch in dense_tensor[0] if ch != -1])
    label = label.replace('z', '0')
    label = label.replace('_', '')
    label = label.replace('o', '0')
    return label


def pad_sequences(sequences, maxlen=None, dtype=np.float32,
                  padding='post', truncating='post', value=0.):
    """Pads each sequence to the same length: the length of the longest
    sequence.
        If maxlen is provided, any sequence longer than maxlen is truncated to
        maxlen. Truncation happens off either the beginning or the end
        (default) of the sequence. Supports post-padding (default) and
        pre-padding.

        Args:
            sequences: list of lists where each element is a sequence
            maxlen: int, maximum length
            dtype: type to cast the resulting sequence.
            padding: 'pre' or 'post', pad either before or after each sequence.
            truncating: 'pre' or 'post', remove values from sequences larger
            than maxlen either in the beginning or in the end of the sequence
            value: float, value to pad the sequences to the desired value.
        Returns
            x: numpy array with dimensions (number_of_sequences, maxlen)
            lengths: numpy array with the original sequence lengths
    """
    lengths = np.asarray([len(s) for s in sequences], dtype=np.int64)

    nb_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)

    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break

    x = (np.ones((nb_samples, maxlen) + sample_shape) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if len(s) == 0:
            continue  # empty list was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" not understood' % truncating)

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x, lengths


def compare_predicted_to_true(preds, trues_tup):
    inv_index_mapping = {v: k for k, v in tidigits_db.index_mapping.items()}

    trues = tf.sparse_tensor_to_dense(tf.SparseTensor(indices=trues_tup[0], values=trues_tup[1], dense_shape=trues_tup[2]), default_value=-1).eval()

    for true, pred in zip(trues, preds):
        predicted_label = "".join([inv_index_mapping[ch] for ch in pred if ch != -1])
        true_label = "".join([inv_index_mapping[ch] for ch in true if ch != -1])

        logger.info("Predicted: {}\n   Actual: {}\n".format(predicted_label, true_label))


def ctc_preds_to_labels(preds):
    inv_index_mapping = {v: k for k, v in tidigits_db.index_mapping.items()}
    preds = tf.sparse_tensor_to_dense(preds, default_value=-1).eval()
    pred_labels = [
        "".join([inv_index_mapping[ch] for ch in pred if ch != -1])
        for pred in preds
    ]
    return pred_labels


def load_dataset(dataset_path):
    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)
    return dataset


def make_batches(dataset, batch_size):
    examples = []
    sequences = []
    seqlens = []

    l1, l2, l3 = dataset

    for i in range(0, len(l1), batch_size):
        examples_batch = l1[i:i + batch_size]
        labels_batch = sparse_tuple_from(l2[i:i + batch_size])
        seqlens_batch = l3[i:i + batch_size]

        examples.append(examples_batch)
        sequences.append(labels_batch)
        seqlens.append(seqlens_batch)

    return examples, sequences, seqlens
