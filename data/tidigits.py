import random
import sqlite3

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
