from admin.config import project_config
from data.tidigits import tidigits_db, clean_digits

import random
import collections
import itertools

import numpy as np
import python_speech_features as psf
import scikits.audiolab


class DigitsSynthesizedSample(object):
    def __init__(self, samples):
        self.samples = samples
        self.audio = np.concatenate(tuple(sample.audio for sample in samples))
        self._mfccs = {}

    def to_mfccs(self, num_mfcc=13):
        if num_mfcc not in self._mfccs:
            # Cache the mfccs for this value of num_mfcc so we
            # don't have to derive them again
            self._mfccs[num_mfcc] = self._to_mfccs(num_mfcc)
        return self._mfccs[num_mfcc]

    def _to_mfccs(self, num_mfcc):
        return np.concatenate(
            tuple(sample.to_mfccs(num_mfcc=num_mfcc) for sample in self.samples))


class DigitsSample(object):
    def __init__(self, sample_id):
        self.row = tidigits_db.get_digits_audio_sample_row_by_id(sample_id)
        self.audio = self.raw_audio.read_frames(self.raw_audio.nframes)
        self._mfccs = {}

    @property
    def id(self):
        return self.row['id']

    @property
    def path(self):
        return self.row['path']

    @property
    def raw_audio(self):
        return scikits.audiolab.Sndfile(self.path, 'r')

    @property
    def digits(self):
        return str(self.row['digits'])

    def to_mfccs(self, num_mfcc=13):
        if num_mfcc not in self._mfccs:
            # Cache the mfccs for this value of num_mfcc so we
            # don't have to derive them again
            self._mfccs[num_mfcc] = self._to_mfccs(num_mfcc)
        return self._mfccs[num_mfcc]

    def _to_mfccs(self, num_mfcc):
        return psf.mfcc(self.audio, numcep=num_mfcc)


class DigitsSampleCollection(object):
    def __init__(self, digits_sample_ids):
        self.digits_to_samples = collections.defaultdict(list)
        for sample_id in digits_sample_ids:
            digits_audio_sample = DigitsSample(sample_id)
            self.digits_to_samples[digits_audio_sample.digits].append(digits_audio_sample)
        self.digits_to_samples = dict(self.digits_to_samples)

    def get_samples(self, digits=None, desired_length=2):
        if digits:
            _clean_digits = clean_digits(digits, desired_length)
            return self.digits_to_samples[_clean_digits]
        else:
            return list(itertools.chain.from_iterable(self.digits_to_samples.values()))

    def choose_random_sample(self, digits=None, desired_length=2):
        if project_config.audio_clip_mode == 'standard':
            samples = self.get_samples(digits, desired_length)
            return random.choice(samples)
        elif project_config.audio_clip_mode == 'synthesized':
            samples = []
            for digit in clean_digits(digits, desired_length):
                options = self.get_samples(digit, desired_length=1)
                samples.append(random.choice(options))
            return DigitsSynthesizedSample(samples)
