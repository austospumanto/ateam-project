import os
import sqlite3

from tidigits import TIDIGITS_PATH, CDS, DB_NAME


def process_data(sequence_len=2):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    # Create table
    c.execute('create table tidigits (\
               id integer primary key,\
               speaker_id text,\
               speaker_type text,\
               usage text,\
               production text,\
               digits text,\
               path text);')
    for cd in CDS:
        cd_dir = TIDIGITS_PATH % cd
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
                if len(digits) == sequence_len:
                    # Let primary key take care of ID
                    insert_tuple = (None, speaker_id, speaker_type, usage, production, digits, path)
                    print insert_tuple
                    c.execute('insert into tidigits values (?, ?, ?, ?, ?, ?, ?);', insert_tuple)
                    conn.commit()
    print 'DONE!'
    conn.close()


if __name__ == '__main__':
    process_data()






