import tensorflow as tf
from speech.models.CTCModel import CTCModel, Config
import random
from speech.models.model_utils import make_batches, pad_sequences
import numpy as np
import math
import logging
from data.tidigits import tidigits_db
from data.DigitsSample import DigitsSample
import tqdm
from admin.config import project_config
import os

logger = logging.getLogger(__name__)


def train_ctcmodel(run_name):
    run_config = Config(run_name)

    if run_config.tidigits_subset == 'fl':
        datasets = tidigits_db.get_split_fl_dataset()
    elif run_config.tidigits_subset == 'non-fl':
        datasets = tidigits_db.get_split_non_fl_dataset()
    else:
        datasets = tidigits_db.get_split_all_dataset()
    for dset_type, sample_ids in datasets.items():
        if dset_type == 'test':
            continue
        dset = []
        for sample_id in tqdm.tqdm(sample_ids):
            sample = DigitsSample(sample_id)
            features = sample.to_mfccs(run_config.num_mfcc_features)
            seqlen = features.shape[0]
            label = tuple(tidigits_db.index_mapping[ch] for ch in sample.digits)
            dset.append((features, label, seqlen))
        datasets[dset_type] = zip(*dset)

    # Make minibatches
    train_feature_minibatches, train_labels_minibatches, train_seqlens_minibatches = \
        make_batches(datasets['train'], batch_size=run_config.batch_size)
    val_feature_minibatches, val_labels_minibatches, val_seqlens_minibatches = \
        make_batches(datasets['val'], batch_size=run_config.batch_size)

    def pad_all_batches(batch_feature_array):
        for batch_num in range(len(batch_feature_array)):
            batch_feature_array[batch_num] = pad_sequences(batch_feature_array[batch_num])[0]
        return batch_feature_array

    train_feature_minibatches = pad_all_batches(train_feature_minibatches)
    val_feature_minibatches = pad_all_batches(val_feature_minibatches)

    num_examples = np.sum([mb_idx.shape[0] for mb_idx in train_feature_minibatches])
    num_batches_per_epoch = int(math.ceil(num_examples / run_config.batch_size))
    with tf.Graph().as_default():
        with tf.Session() as session:
            ctc_model = CTCModel()
            saver = tf.train.Saver(tf.trainable_variables())
            session.run(tf.global_variables_initializer())
            train_writer = tf.summary.FileWriter(run_config.run_results_path, session.graph)

            step_ii = 0
            for curr_epoch in range(run_config.num_epochs):
                total_train_cost = total_train_wer = 0

                for mb_idx in random.sample(range(num_batches_per_epoch), num_batches_per_epoch):
                    cur_batch_size = len(train_seqlens_minibatches[mb_idx])

                    batch_cost, batch_ler, summary = ctc_model.train_on_batch(
                        session,
                        train_feature_minibatches[mb_idx],
                        train_labels_minibatches[mb_idx],
                        train_seqlens_minibatches[mb_idx],
                        train=True)
                    total_train_cost += batch_cost * cur_batch_size
                    total_train_wer += batch_ler * cur_batch_size

                    train_writer.add_summary(summary, step_ii)
                    step_ii += 1

                train_cost = total_train_cost / num_examples
                train_wer = total_train_wer / num_examples

                val_batch_cost, val_batch_ler, _ = ctc_model.train_on_batch(
                    session,
                    val_feature_minibatches[0],
                    val_labels_minibatches[0],
                    val_seqlens_minibatches[0],
                    train=False)

                log_msg = "Epoch {}/{}, train_cost = {:.3f}, train_ed = {:.3f}, " \
                          "val_cost = {:.3f}, val_ed = {:.3f}"
                logger.info(
                    log_msg.format(curr_epoch + 1, Config.num_epochs, train_cost,
                                   train_wer, val_batch_cost, val_batch_ler))

                if run_config.log_every is not None and \
                   (curr_epoch + 1) % run_config.log_every == 0:
                    batch_ii = random.randint(0, len(train_feature_minibatches) - 1)
                    ctc_model.print_results(session,
                                            train_feature_minibatches[batch_ii],
                                            train_labels_minibatches[batch_ii],
                                            train_seqlens_minibatches[batch_ii])

                if run_config.save_every is not None and \
                   run_config.model_outputs_path is not None and \
                   (curr_epoch + 1) % run_config.save_every == 0:
                    if not os.path.exists(run_config.model_outputs_path):
                        os.makedirs(run_config.model_outputs_path)
                    saver.save(session, run_config.model_outputs_path, global_step=curr_epoch + 1)


def transfer_train_ctcmodel(ctc_run_name, restore_run_name):
    run_config = Config(ctc_run_name)

    datasets = tidigits_db.get_split_fl_dataset()
    for dset_type, sample_ids in datasets.items():
        if dset_type == 'test':
            continue
        dset = []
        for sample_id in tqdm.tqdm(sample_ids):
            sample = DigitsSample(sample_id)
            features = sample.to_mfccs(run_config.num_mfcc_features)
            seqlen = features.shape[0]
            label = tuple(tidigits_db.index_mapping[ch] for ch in sample.digits)
            dset.append((features, label, seqlen))
        datasets[dset_type] = zip(*dset)

    # Make minibatches
    train_feature_minibatches, train_labels_minibatches, train_seqlens_minibatches = \
        make_batches(datasets['train'], batch_size=run_config.batch_size)
    val_feature_minibatches, val_labels_minibatches, val_seqlens_minibatches = \
        make_batches(datasets['val'], batch_size=run_config.batch_size)

    def pad_all_batches(batch_feature_array):
        for batch_num in range(len(batch_feature_array)):
            batch_feature_array[batch_num] = pad_sequences(batch_feature_array[batch_num])[0]
        return batch_feature_array

    train_feature_minibatches = pad_all_batches(train_feature_minibatches)
    val_feature_minibatches = pad_all_batches(val_feature_minibatches)

    num_examples = np.sum([mb_idx.shape[0] for mb_idx in train_feature_minibatches])
    num_batches_per_epoch = int(math.ceil(num_examples / run_config.batch_size))
    with tf.Graph().as_default():
        with tf.Session() as session:
            saved_run_dir = os.path.join(project_config.saved_runs_dir, restore_run_name)
            model_weights_dir = os.path.join(saved_run_dir, 'model.weights')
            assert os.path.exists(model_weights_dir)

            ctc_model = CTCModel()

            # # NOTE: Could do it this way and skip the init after restoration,
            # #       but want to double check that GRU weights were actually
            # #       restored, so will do specific initialization of non-GRU vars after restoration
            # # Initialize all vars in CTCModel graph - will re-init the GRU vars with AQN weights
            # session.run(tf.global_variables_initializer())

            # Restore all GRU vars from AQN to the CTCModel
            aqn_var_names = [u'q/rnn/multi_rnn_cell/cell_0/gru_cell/gates/weights:0',
                             u'q/rnn/multi_rnn_cell/cell_0/gru_cell/gates/biases:0',
                             u'q/rnn/multi_rnn_cell/cell_0/gru_cell/candidate/weights:0',
                             u'q/rnn/multi_rnn_cell/cell_0/gru_cell/candidate/biases:0']
            ctc_aqn_vars = [tf.get_default_graph().get_tensor_by_name(vn) for vn in aqn_var_names]
            saver = tf.train.Saver(ctc_aqn_vars)
            saver.restore(session, tf.train.latest_checkpoint(model_weights_dir))

            pre_init_norms = [
                np.linalg.norm(session.run(tf.get_default_graph().get_tensor_by_name(var_name)))
                for var_name in aqn_var_names
            ]

            # Specifically initialize variables that we did not restore
            vars_to_init = [
                var for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
                if var.name not in aqn_var_names
            ]
            logger.info([var.name for var in vars_to_init])
            session.run(tf.variables_initializer(vars_to_init))

            # Check to make sure the restored variables were not written over during
            # initialization of other vars
            post_init_norms = [
                np.linalg.norm(session.run(tf.get_default_graph().get_tensor_by_name(var_name)))
                for var_name in aqn_var_names
            ]
            logger.info('Pre-init GRU norms' + repr(pre_init_norms))
            logger.info('Post-init GRU norms' + repr(post_init_norms))
            assert pre_init_norms == post_init_norms

            train_writer = tf.summary.FileWriter(run_config.run_results_path, session.graph)

            step_ii = 0
            for curr_epoch in range(run_config.num_epochs):
                total_train_cost = total_train_wer = 0

                for mb_idx in random.sample(range(num_batches_per_epoch), num_batches_per_epoch):
                    cur_batch_size = len(train_seqlens_minibatches[mb_idx])

                    batch_cost, batch_ler, summary = ctc_model.train_on_batch(
                        session,
                        train_feature_minibatches[mb_idx],
                        train_labels_minibatches[mb_idx],
                        train_seqlens_minibatches[mb_idx],
                        train=True)
                    total_train_cost += batch_cost * cur_batch_size
                    total_train_wer += batch_ler * cur_batch_size

                    train_writer.add_summary(summary, step_ii)
                    step_ii += 1

                train_cost = total_train_cost / num_examples
                train_wer = total_train_wer / num_examples

                val_batch_cost, val_batch_ler, _ = ctc_model.train_on_batch(
                    session,
                    val_feature_minibatches[0],
                    val_labels_minibatches[0],
                    val_seqlens_minibatches[0],
                    train=False)

                log_msg = "Epoch {}/{}, train_cost = {:.3f}, train_ed = {:.3f}, " \
                          "val_cost = {:.3f}, val_ed = {:.3f}"
                logger.info(
                    log_msg.format(curr_epoch + 1, Config.num_epochs, train_cost,
                                   train_wer, val_batch_cost, val_batch_ler))

                if run_config.log_every is not None and \
                   (curr_epoch + 1) % run_config.log_every == 0:
                    batch_ii = 0
                    ctc_model.print_results(session,
                                            train_feature_minibatches[batch_ii],
                                            train_labels_minibatches[batch_ii],
                                            train_seqlens_minibatches[batch_ii])

                if run_config.save_every is not None and \
                   run_config.model_outputs_path is not None and \
                   (curr_epoch + 1) % run_config.save_every == 0:
                    if not os.path.exists(run_config.model_outputs_path):
                        os.makedirs(run_config.model_outputs_path)
                    saver.save(session, run_config.model_outputs_path, global_step=curr_epoch + 1)