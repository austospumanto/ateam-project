from admin.config import project_config

import pickle
import random
import sqlite3

import numpy as np
import python_speech_features as psf
import scikits.audiolab
from sklearn.model_selection import StratifiedShuffleSplit

DB_NAME = 'data/tidigits.db'
TIDIGITS_PATH = 'data/LDC93S10_TIDIGITS/%s/tidigits'
CDS = ['CD4_1_1', 'CD4_2_1', 'CD4_3_1']

conn = sqlite3.connect(DB_NAME)
conn.row_factory = sqlite3.Row

def get_audio_file_path(desired_digits, desired_length=2, usage='train'):
    # Must have desired length, or else we prepend zeros
    if len(desired_digits) != desired_length:
        prepend = 'z' * (desired_length - len(desired_digits))
        desired_digits = prepend + desired_digits
    # By convention, ZERO-ONE is labeled z1 in TIDIGITS
    desired_digits = desired_digits.replace('0', 'z')
    c = conn.cursor()
    c.execute('select * from tidigits where digits = ? and usage = ?;', (desired_digits, usage))
    rows = c.fetchall()
    choice = random.choice(rows)
    return choice['path']

def get_tidigits_to_index_mapping():
    return {"z": 0, "o": 10, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "_": 11}

def get_split_dataset():
    c = conn.cursor()
    # Part 1 - fetch test_ids directly from database (where usage = 'test')
    c.execute("select id from tidigits where usage = 'test' and \
              (digits like ? or digits = ? or digits = ? or digits = ? \
               or digits = ? or digits = ? or digits = ?);",
              ('z%', '1z', '11', '12', '13', '14', '15'))
    test_ids = [int(row['id']) for row in c.fetchall()]

    # Part 2a - get all non-test data from database (where usage = 'train')
    c.execute("select id, digits from tidigits where usage = 'train' \
               and (digits like ? or digits = ? or digits = ? or digits = ? \
                    or digits = ? or digits = ? or digits = ?);",
              ('z%', '1z', '11', '12', '13', '14', '15'))
    rows = c.fetchall()

    # Part 2b - split non-test data into train and val splits, based on speaker_type
    speaker_info = [row['digits'] for row in rows]
    sss = StratifiedShuffleSplit(n_splits=1, test_size=project_config.val_ratio,
                                 random_state=42)
    splits = sss.split(np.zeros(len(speaker_info)), speaker_info)
    train_indices, val_indices = list(splits)[0]  # Only one split

    # Part 2c - use split indices to retrieve actual ids
    train_indices = list(train_indices)
    val_indices = list(val_indices)
    train_ids = [row['id'] for i, row in enumerate(rows) if i in train_indices]
    val_ids = [row['id'] for i, row in enumerate(rows) if i in val_indices]

    # Part 3 - assert that we split correctly
    assert len(set(train_ids) & set(val_ids) & set(test_ids)) == 0
    result = c.execute("select count(*) as total from tidigits where \
                           (digits like ? or digits = ? or digits = ? or digits = ? \
                            or digits = ? or digits = ? or digits = ?);",
                           ('z%', '1z', '11', '12', '13', '14', '15')).fetchone()
    assert len(train_ids) + len(val_ids) + len(test_ids) == result['total']
    return train_ids, val_ids, test_ids

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
