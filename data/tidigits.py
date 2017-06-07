from admin.config import project_config

import os
import random
import sqlite3
import logging
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

logger = logging.getLogger(__name__)
__all__ = ['tidigits_db']


# noinspection PyTypeChecker,SqlDialectInspection
class TidigitsDatabase(object):
    RELATIVE_DB_PATH = 'data/tidigits.db'
    RELATIVE_TIDIGITS_PATH = 'data/LDC93S10_TIDIGITS/%s/tidigits'
    CDS = ['CD4_1_1', 'CD4_2_1', 'CD4_3_1']
    index_mapping = {"z": 0, "o": 10, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "_": 11}

    # For retrieving digit samples used in FrozenLake
    fl_digits_query_template = \
        'select * from tidigits where \
         %s \
         (digits = ? or digits = ? or digits = ? or digits = ? or \
         digits = ? or digits = ? or digits = ? or digits = ? or \
         digits = ? or digits = ? or digits = ? or digits = ? or \
         digits = ? or digits = ? or digits = ? or digits = ? or \
         digits = ? or digits = ? or digits = ? or digits = ? or \
         digits = ? or digits = ? or digits = ? or digits = ? or \
         digits = ? or digits = ?);'
    fl_digits_query_args = ('z', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                            'zz', 'z1', 'z2', 'z3', 'z4', 'z5', 'z6', 'z7', 'z8', 'z9',
                            '1z', '11', '12', '13', '14', '15')

    # For retrieving all digit samples
    all_digits_query_template = 'select * from tidigits %s;'
    all_digits_query_args = tuple()

    create_table_query = \
        'create table tidigits (\
         id integer primary key,\
         speaker_id text,\
         speaker_type text,\
         usage text,\
         production text,\
         digits text,\
         length integer,\
         path text);'

    def __init__(self):
        self.conn = sqlite3.connect(self.RELATIVE_DB_PATH)
        self.conn.row_factory = sqlite3.Row

    def fetchall(self, query_args):
        c = self.conn.cursor()
        c.execute(*query_args)
        rows = c.fetchall()
        return rows

    def create_table(self):
        c = self.conn.cursor()
        c.execute(self.create_table_query)

    def insert_into_table(self, insert_tuple):
        c = self.conn.cursor()
        c.execute('insert into tidigits values (?, ?, ?, ?, ?, ?, ?, ?);', insert_tuple)
        self.conn.commit()

    def fetch_fl_digits(self, usage=None):
        template_filler = ""
        if usage == 'test':
            template_filler = "usage = 'test' and "
        elif usage == 'train':
            template_filler = "usage = 'train' and "
        fl_digits_query = self.fl_digits_query_template % template_filler
        return self.fetchall((
            fl_digits_query,
            self.fl_digits_query_args
        ))

    def fetch_non_fl_digits(self, usage=None):
        template_filler = ""
        if usage == 'test':
            template_filler = "usage = 'test' and "
        elif usage == 'train':
            template_filler = "usage = 'train' and "
        fl_digits_query = self.fl_digits_query_template % template_filler
        fl_digits_rows = self.fetchall((
            fl_digits_query,
            self.fl_digits_query_args
        ))
        fl_digits_ids = set([r['id'] for r in fl_digits_rows])
        all_digits_rows = self.fetch_all_digits(usage)
        non_fl_digits_rows = [r for r in all_digits_rows if r['id'] not in fl_digits_ids]
        assert len(non_fl_digits_rows) + len(fl_digits_rows) == len(all_digits_rows)
        assert len(non_fl_digits_rows) < len(all_digits_rows)
        return non_fl_digits_rows


    def fetch_all_digits(self, usage=None):
        template_filler = ""
        if usage == 'test':
            template_filler = "where usage = 'test'"
        elif usage == 'train':
            template_filler = "where usage = 'train'"
        fl_digits_query = self.all_digits_query_template % template_filler
        return self.fetchall((
            fl_digits_query,
            self.all_digits_query_args
        ))

    def get_digits_audio_sample_row_by_id(self, sample_id):
        audio_sample_rows = self.fetchall(
            ('select * from tidigits where id = ?;', (sample_id,))
        )
        assert len(audio_sample_rows) == 1
        audio_sample_row = audio_sample_rows[0]
        assert audio_sample_row['id'] == sample_id
        return audio_sample_row

    def get_digits_audio_sample_row_by_path(self, sample_path):
        audio_sample_rows = self.fetchall(
            ('select * from tidigits where path = ?;', (sample_path,))
        )
        assert len(audio_sample_rows) == 1
        audio_sample_row = audio_sample_rows[0]
        assert audio_sample_row['path'] == sample_path
        return audio_sample_row

    def sample_row_exists(self, sample_path):
        try:
            self.get_digits_audio_sample_row_by_path(sample_path)
            return True
        except AssertionError:
            return False

    def get_random_digits_audio_sample_with_valid_id(self, desired_digits, valid_ids=None, desired_length=2):
        desired_digits = clean_digits(desired_digits, desired_length)
        rows = self.fetchall(
            ('select * from tidigits where digits = ?;', (desired_digits,))
        )
        if valid_ids:
            rows = [row for row in rows if row['id'] in valid_ids]
        return random.choice(rows)

    def process_data(self, sequence_len=2, verbose=True):
        try:
            self.create_table()
        except sqlite3.OperationalError:
            # Tablea already created
            pass

        num_inserted = 0
        for cd in self.CDS:
            cd_dir = self.RELATIVE_TIDIGITS_PATH % cd
            for root, _, files in os.walk(cd_dir):
                for f in files:
                    if not f.lower().endswith('.wav'):
                        continue
                    path = os.path.join(root, f)
                    relpath = os.path.relpath(path, cd_dir)
                    usage, speaker_type, speaker_id, filename = relpath.split('/')
                    stripped_filename = filename[:-4]  # strip out .wav extension
                    digits = stripped_filename[:-1]  # z1z36
                    production = stripped_filename[-1]  # a or b
                    if not (sequence_len and len(digits) != sequence_len) and not self.sample_row_exists(path):
                        # Let primary key take care of ID
                        insert_tuple = (None, speaker_id, speaker_type, usage, production, digits, len(digits), path)
                        self.insert_into_table(insert_tuple)
                        if verbose:
                            logger.info('Inserted tuple into tidigits: %r' % (insert_tuple,))
                        num_inserted += 1
        logger.info('Done with TidigitsDatabase.process_data(sequence_len=%r). %d inserted' % (sequence_len, num_inserted))

    def get_split_fl_dataset(self):
        return self.get_split_dataset(self.fetch_fl_digits, split_by='digits')

    def get_split_non_fl_dataset(self):
        return self.get_split_dataset(self.fetch_non_fl_digits, split_by='speaker_type')

    def get_split_all_dataset(self):
        return self.get_split_dataset(self.fetch_all_digits, split_by='speaker_type')

    @classmethod
    def filter_out_oh_digits(cls, rows):
        return [row for row in rows if 'o' not in row['digits']]

    @classmethod
    def get_split_dataset(cls, fetch_digits_fxn, split_by):
        all_rows = fetch_digits_fxn(usage=None)
        all_rows = cls.filter_out_oh_digits(all_rows)
        total_num_fl_samples = len(all_rows)

        # Part 1 - fetch test_ids directly from database (where usage = 'test')
        test_rows = fetch_digits_fxn(usage='test')
        test_rows = cls.filter_out_oh_digits(test_rows)
        test_ids = [int(row['id']) for row in test_rows]
        logger.info('Num usage=\'test\' rows: %d' % len(test_rows))

        # Part 2a - get all non-test data from database (where usage = 'train')
        train_rows = fetch_digits_fxn(usage='train')
        train_rows = cls.filter_out_oh_digits(train_rows)
        train_rows.sort(key=lambda r: r['id'])
        logger.info('Num usage=\'train\' rows: %d' % len(train_rows))

        logger.info('Digits "classes" collected: %r' % list(set((r['digits'] for r in train_rows))))

        # Part 2b - split non-test data into train and val splits, based on speaker_type
        speaker_info = [row[split_by] for row in train_rows]
        sss = StratifiedShuffleSplit(n_splits=1, test_size=project_config.val_ratio, random_state=42)
        splits = sss.split(np.zeros(len(speaker_info)), speaker_info)
        train_indices, val_indices = map(list, list(splits)[0])  # Only one split

        # Part 2c - use split indices to retrieve actual ids
        train_ids = [row['id'] for i, row in enumerate(train_rows) if i in train_indices]
        val_ids = [row['id'] for i, row in enumerate(train_rows) if i in val_indices]

        # Part 3 - assert that we split correctly
        assert len(set(train_ids) & set(val_ids) & set(test_ids)) == 0
        assert len(train_ids) + len(val_ids) + len(test_ids) == total_num_fl_samples
        data_split = {
            'train': train_ids,
            'val': val_ids,
            'test': test_ids
        }
        logger.info('Data split stats: ' + str({k: len(v) for k, v in data_split.items()}))
        return data_split

    # DEPRECATED
    # ----------
    # def dump_data(self):
    #     train_examples = []
    #     train_sequences = []
    #     train_seqlens = []
    #     test_examples = []
    #     test_sequences = []
    #     test_seqlens = []
    #     rows = self.fetchall(('select path, digits from tidigits where digits like ? or digits like ?;', ('z%', '1%')))
    #     random.shuffle(rows)
    #     print len(rows)
    #     cutoff = int(len(rows) * 0.8)
    #     for row in rows[:cutoff]:
    #         example, sequence, seq_len = self.get_mfcc_data(row['path'], row['digits'])
    #         train_examples.append(example)
    #         train_sequences.append(sequence)
    #         train_seqlens.append(seq_len)
    #     for row in rows[cutoff:]:
    #         example, sequence, seq_len = self.get_mfcc_data(row['path'], row['digits'])
    #         test_examples.append(example)
    #         test_sequences.append(sequence)
    #         test_seqlens.append(seq_len)
    #     train_dataset = (train_examples, train_sequences, train_seqlens)
    #     test_dataset = (test_examples, test_sequences, test_seqlens)
    #     pickle.dump(train_dataset, open('data/train.dat', 'wb'))
    #     pickle.dump(test_dataset, open('data/test.dat', 'wb'))
    #
    # @classmethod
    # def get_mfcc_data(cls, path, digits):
    #     index_mapping = cls.index_mapping
    #     f = scikits.audiolab.Sndfile(path, 'r')
    #     signal = f.read_frames(f.nframes)
    #     example = psf.mfcc(signal)
    #     sequence = [index_mapping[ch] for ch in digits]
    #     seq_len = example.shape[0]
    #     return example, sequence, seq_len


def clean_digits(desired_digits, desired_length):
    desired_digits = str(desired_digits)
    assert len(desired_digits) <= desired_length
    # Must have desired length, or else we prepend zeros
    if len(desired_digits) != desired_length:
        prepend = 'z' * (desired_length - len(desired_digits))
        desired_digits = prepend + desired_digits
    # By convention, ZERO-ONE is labeled z1 in TIDIGITS
    desired_digits = desired_digits.replace('0', 'z')
    return desired_digits


tidigits_db = TidigitsDatabase()
