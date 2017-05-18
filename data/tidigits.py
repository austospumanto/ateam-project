import pickle
import random
import sqlite3

import python_speech_features as psf
import scikits.audiolab

DB_NAME = 'data/tidigits.db'
TIDIGITS_PATH = 'data/LDC93S10_TIDIGITS/%s/tidigits'
CDS = ['CD4_1_1', 'CD4_2_1', 'CD4_3_1']

conn = sqlite3.connect(DB_NAME)
conn.row_factory = sqlite3.Row

def get_audio_file_path(desired_digits, desired_length=2):
    # Must have desired length, or else we prepend zeros
    if len(desired_digits) != desired_length:
        prepend = 'z' * (desired_length - len(desired_digits))
        desired_digits = prepend + desired_digits
    # By convention, ZERO-ONE is labeled z1 in TIDIGITS
    desired_digits = desired_digits.replace('0', 'z')
    c = conn.cursor()
    c.execute('select * from tidigits where digits = ?;', (desired_digits,))
    rows = c.fetchall()
    choice = random.choice(rows)
    print choice
    return choice['path']

def get_tidigits_to_index_mapping():
    return {"z": 0, "o": 10, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "_": 11}

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
