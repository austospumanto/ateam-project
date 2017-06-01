import numpy as np
import python_speech_features as psf
from speech.model_utils import label_from_sparse_tensor
from data.tidigits import get_audio_file_path
import scikits.audiolab


class DigitsRecognizer(object):
    def __init__(self, model, sess):
        self._model = model
        self._sess = sess

    def recognize(self, features, raw=False):
        """
        Translates MFCC features/raw audio of spoken digits to a string of the digits. ASR

        :param features: The MFCC features of an audio sample of a person saying a string of digits
        :param raw: If False, we are using MFCC features.
        :return: digit
        """
        seq_len = np.array([features.shape[0]])
        input_features = np.expand_dims(features, axis=0)
        input_feed = self._model.create_feed_dict(input_features, seq_len)
        digits = label_from_sparse_tensor(
            self._sess.run(self._model.decoded_sequence, input_feed)
        )
        return digits


class DigitsSpeaker(object):
    def speak(self, state, usage='train'):
        """
        Convert state (string of digits) to MFCC features/raw audio representation. TTS

        :param state: String of numeric characters representing state.
        :param raw: If False, we are using MFCC features.
        :return:
        """
        state = state if isinstance(state, str) else str(state)
        audio_file_path = get_audio_file_path(state, usage=usage)
        f = scikits.audiolab.Sndfile(audio_file_path, 'r')
        audio_signal = f.read_frames(f.nframes)
        return audio_signal


class MfccDeriver(object):
    def __init__(self, num_mfcc=13):
        self._num_mfcc = num_mfcc

    def derive(self, audio_signal):
        return psf.mfcc(audio_signal, numcep=self._num_mfcc)
