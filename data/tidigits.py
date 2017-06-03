from admin.config import project_config

import pickle
import random
import sqlite3
import collections
import itertools

import numpy as np
import python_speech_features as psf
import scikits.audiolab
from sklearn.model_selection import StratifiedShuffleSplit

DB_NAME = 'data/tidigits.db'
TIDIGITS_PATH = 'data/LDC93S10_TIDIGITS/%s/tidigits'
CDS = ['CD4_1_1', 'CD4_2_1', 'CD4_3_1']

conn = sqlite3.connect(DB_NAME)
conn.row_factory = sqlite3.Row


class DigitsSample(object):
    def __init__(self, sample_id):
        self.row = get_digits_audio_sample_row_by_id(sample_id)
        raw_audio = scikits.audiolab.Sndfile(self.path, 'r')
        self.audio = raw_audio.read_frames(raw_audio.nframes)
        self._mfccs = {}

    @property
    def id(self):
        return self.row['id']

    @property
    def path(self):
        return self.row['path']

    @property
    def digits(self):
        return str(self.row['digits'])

    def to_mfccs(self, num_mfcc=13):
        if num_mfcc not in self._mfccs:
            # Cache the mfccs for this value of num_mfcc so we
            # don't have to derive them again
            self._mfccs[num_mfcc] = self.derive_mfccs(num_mfcc)
        return self._mfccs[num_mfcc]

    def derive_mfccs(self, num_mfcc):
        return psf.mfcc(self.audio, numcep=num_mfcc)


class DigitsSampleCollection(object):
    def __init__(self, digits_sample_ids):
        self.digits_to_samples = collections.defaultdict(list)
        for sample_id in digits_sample_ids:
            digits_audio_sample = DigitsSample(sample_id)
            self.digits_to_samples[digits_audio_sample.digits] = digits_audio_sample
        self.digits_to_samples = dict(self.digits_to_samples)

    def get_samples(self, digits=None, desired_length=2):
        if digits:
            _clean_digits = clean_digits(digits, desired_length)
            return self.digits_to_samples[_clean_digits]
        else:
            return itertools.chain.from_iterable(self.digits_to_samples.values())

    def choose_random_sample(self, digits=None, desired_length=2):
        samples = self.get_samples(digits, desired_length)
        return random.choice(samples)


def get_digits_audio_sample_row_by_id(sample_id):
    audio_sample_rows = fetchall_for_query(
        ('select * from tidigits where id = ?;', (sample_id,)))
    assert len(audio_sample_rows) == 1
    audio_sample_row = audio_sample_rows[0]
    assert audio_sample_row['id'] == sample_id
    return audio_sample_row


def fetchall_for_query(query_args):
    c = conn.cursor()
    c.execute(*query_args)
    rows = c.fetchall()
    return rows


def clean_digits(desired_digits, desired_length):
    desired_digits = str(desired_digits)
    # Must have desired length, or else we prepend zeros
    if len(desired_digits) != desired_length:
        prepend = 'z' * (desired_length - len(desired_digits))
        desired_digits = prepend + desired_digits
    # By convention, ZERO-ONE is labeled z1 in TIDIGITS
    desired_digits = desired_digits.replace('0', 'z')
    return desired_digits


def get_random_digits_audio_sample_with_valid_id(desired_digits, valid_ids=None, desired_length=2):
    desired_digits = clean_digits(desired_digits, desired_length)
    rows = fetchall_for_query(
        ('select * from tidigits where digits = ?;', (desired_digits,)))
    if valid_ids:
        rows = [row for row in rows if row['id'] in valid_ids]
    return random.choice(rows)


def get_tidigits_to_index_mapping():
    return {"z": 0, "o": 10, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "_": 11}


def get_split_dataset():
    # Part 1 - fetch test_ids directly from database (where usage = 'test')
    test_rows = fetchall_for_query((
        "select id, digits from tidigits where usage = 'test' and \
         (digits like ? or digits = ? or digits = ? or digits = ? \
         or digits = ? or digits = ? or digits = ?);",
        ('z%', '1z', '11', '12', '13', '14', '15')
    ))
    test_ids = [int(row['id']) for row in test_rows]

    # Part 2a - get all non-test data from database (where usage = 'train')
    train_rows = fetchall_for_query((
        "select id, digits from tidigits where usage = 'train' and \
         (digits like ? or digits = ? or digits = ? or digits = ? \
         or digits = ? or digits = ? or digits = ?);",
        ('z%', '1z', '11', '12', '13', '14', '15')
    ))

    # Part 2b - split non-test data into train and val splits, based on speaker_type
    speaker_info = [row['digits'] for row in train_rows]
    sss = StratifiedShuffleSplit(n_splits=1, test_size=project_config.val_ratio,
                                 random_state=42)
    splits = sss.split(np.zeros(len(speaker_info)), speaker_info)
    train_indices, val_indices = list(splits)[0]  # Only one split

    # Part 2c - use split indices to retrieve actual ids
    train_indices = list(train_indices)
    val_indices = list(val_indices)
    train_ids = [row['id'] for i, row in enumerate(train_rows) if i in train_indices]
    val_ids = [row['id'] for i, row in enumerate(train_rows) if i in val_indices]

    # Part 3 - assert that we split correctly
    assert len(set(train_ids) & set(val_ids) & set(test_ids)) == 0
    total_num_samples = fetchall_for_query((
        "select count(*) as total from tidigits where \
         (digits like ? or digits = ? or digits = ? or digits = ? \
         or digits = ? or digits = ? or digits = ?);",
        ('z%', '1z', '11', '12', '13', '14', '15')
    ))[0]['total']
    assert len(train_ids) + len(val_ids) + len(test_ids) == total_num_samples
    return {
        'train': train_ids,
        'val': val_ids,
        'test': test_ids
    }


def dump_data():
    train_examples = []
    train_sequences = []
    train_seqlens = []
    test_examples = []
    test_sequences = []
    test_seqlens = []
    c = conn.cursor()
    c.execute('select path, digits from tidigits where digits like ? or digits like ?;', ('z%', '1%'))
    rows = c.fetchall()
    random.shuffle(rows)
    print len(rows)
    cutoff = int(len(rows) * 0.8)
    for row in rows[:cutoff]:
        example, sequence, seq_len = get_mfcc_data(row['path'], row['digits'])
        train_examples.append(example)
        train_sequences.append(sequence)
        train_seqlens.append(seq_len)
    for row in rows[cutoff:]:
        example, sequence, seq_len = get_mfcc_data(row['path'], row['digits'])
        test_examples.append(example)
        test_sequences.append(sequence)
        test_seqlens.append(seq_len)
    train_dataset = (train_examples, train_sequences, train_seqlens)
    test_dataset = (test_examples, test_sequences, test_seqlens)
    pickle.dump(train_dataset, open('data/train.dat', 'wb'))
    pickle.dump(test_dataset, open('data/test.dat', 'wb'))


def get_mfcc_data(path, digits):
    index_mapping = get_tidigits_to_index_mapping()
    f = scikits.audiolab.Sndfile(path, 'r')
    signal = f.read_frames(f.nframes)
    example = psf.mfcc(signal)
    sequence = [index_mapping[ch] for ch in digits]
    seq_len = example.shape[0]
    return example, sequence, seq_len


if __name__ == "__main__":
    dump_data()
