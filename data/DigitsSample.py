from data.tidigits import tidigits_db, clean_digits

import random
import collections
import itertools

import numpy as np
import python_speech_features as psf
import scikits.audiolab
import vlc
import sys


class SynthesizedDigitsSample(object):
    def __init__(self, samples):
        self.samples = samples
        self.audio = np.concatenate(tuple(sample.audio for sample in samples))
        self._mfccs = {}

    def play(self):
        for sample in self.samples:
            sample.play()

    @property
    def sequence_length(self):
        return sum((samp.sequence_length for samp in self.samples))

    @property
    def digits(self):
        return ''.join((s.digits for s in self.samples))

    @property
    def length(self):
        return sum((sample.length for sample in self.samples))

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

    def play(self):
        p = vlc.MediaPlayer(self.path)
        p.play()
        while p.get_state() != vlc.State.Ended:
            import time
            time.sleep(0.01)

    @property
    def sequence_length(self):
        return len(self.audio)

    @property
    def id(self):
        return self.row['id']

    @property
    def path(self):
        return self.row['path']

    @property
    def length(self):
        return self.row['length']

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
        self.digits_sample_ids = digits_sample_ids
        self.digits_to_samples = collections.defaultdict(list)
        for sample_id in digits_sample_ids:
            digits_audio_sample = DigitsSample(sample_id)
            self.digits_to_samples[digits_audio_sample.digits].append(digits_audio_sample)
        self.digits_to_samples = dict(self.digits_to_samples)

    @property
    def size(self):
        return len(self.digits_sample_ids)

    def get_samples(self, desired_digits=None, desired_length=2):
        if desired_digits is not None:
            _clean_digits = clean_digits(desired_digits, desired_length)
            return self.digits_to_samples[_clean_digits]
        else:
            return list(itertools.chain.from_iterable(self.digits_to_samples.values()))

    def choose_random_sample(self, digits=None, desired_length=2):
        samples = self.get_samples(digits, desired_length)
        return random.choice(samples)


class SynthesizedDigitsSampleCollection(object):
    def __init__(self, digits_sample_ids):
        self.digits_sample_ids = digits_sample_ids
        self.digits_to_samples = collections.defaultdict(list)
        for sample_id in digits_sample_ids:
            digits_audio_sample = DigitsSample(sample_id)
            self.digits_to_samples[digits_audio_sample.digits].append(digits_audio_sample)
        self.digits_to_samples = dict(self.digits_to_samples)

    @property
    def size(self):
        return len(self.digits_sample_ids)

    def get_samples(self, desired_digits=None, desired_length=2):
        if desired_digits is not None:
            _clean_digits = clean_digits(desired_digits, desired_length)
            return self.digits_to_samples[_clean_digits]
        else:
            return list(itertools.chain.from_iterable(self.digits_to_samples.values()))

    def choose_random_sample(self, digits=None, desired_length=2):
        # In the case of frozenlake, fetch two digits audio samples
        # each with length 1
        samples = [
            random.choice(self.get_samples(digit, desired_length=1))
            for digit in clean_digits(digits, desired_length)
        ]
        synth_digits_sample = SynthesizedDigitsSample(samples)
        assert synth_digits_sample.length == desired_length
        assert synth_digits_sample.digits == clean_digits(digits, desired_length)
        return synth_digits_sample
