import numpy as np
import librosa
import python_speech_features as psf
import scipy.io.wavfile as wav
import tensorflow as tf

from speech.model import CTCModel
from speech.model_utils import label_from_sparse_tensor
from data.tidigits import get_audio_file_path


class DigitsRecognizer(object):
    # TODO: Do
    def recognize(self, features, model, sess, raw=False):
        """
        Translates MFCC features/raw audio of spoken digits to a string of the digits. ASR

        :param features: The MFCC features of an audio sample of a person saying a string of digits
        :param raw: If False, we are using MFCC features.
        :return: digit
        """
        sh = np.array([features.shape[0]])
        print 'yy', sh
        input_feed = model.create_feed_dict(np.expand_dims(features, axis=0), sh)  # Hard-coding sequence length as 2
        digits = label_from_sparse_tensor(sess.run(model.decoded_sequence, input_feed))
        print digits
        return int(digits)


class DigitsSpeaker(object):
    # TODO: Viggy
    def speak(self, state, raw=False):
        """
        Convert state (string of digits) to MFCC features/raw audio representation. TTS

        :param state: String of numeric characters representing state.
        :param raw: If False, we are using MFCC features.
        :return:
        """
        audio_file_path = get_audio_file_path(state)
        # (rate, sig) = wav.read(audio_file_path)
        (signal, sample_rate) = librosa.load(audio_file_path)
        if raw:
            return signal
        mfcc = psf.mfcc(signal, sample_rate)
        print 'xxx', mfcc.shape
        return mfcc
