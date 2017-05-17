import librosa
import tensorflow as tf

from speech.model import CTCModel
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
        input_feed = model.create_feed_dict(features, 2)  # Hard-coding sequence length as 2
        digits = sess.run(model.decoded_sequence, input_feed)
        return digits


class DigitsSpeaker(object):
    # TODO: Viggy
    def speak(self, state, raw=False):
        """
        Convert state (string of digits) to MFCC features/raw audio representation. TTS

        :param state: String of numeric characters representing state.
        :param raw: If False, we are using MFCC features.
        :return:
        """
        y, sr = librosa.load(get_audio_file_path(state))
        if raw:
            return y
        return librosa.mfcc(y, sr, n_mfcc=13)
