from speech.model import CTCModel
import tensorflow as tf

class DigitsRecognizer(object):
    # TODO Do
    def recognize(self, features, raw=False):
        """
        Translates MFCC features/raw audio of spoken digits to a string of the digits. ASR

        :param features: The MFCC features of an audio sample of a person saying a string of digits
        :param raw: If False, we are using MFCC features.
        :return: digit
        """
        digits = []

        with tf.Session() as sess:
            model = CTCModel()
            ckpt = tf.train.get_checkpoint_state("cs224s/viggy_assign3/saved_models")
            v2_path = ckpt.model_checkpoint_path + ".index" if ckpt else ""

            if ckpt and (tf.gfile.Exists(ckpt.model_checkpoint_path) or tf.gfile.Exists(v2_path)):
                model.saver.restore(sess, ckpt.model_checkpoint_path)
                print "Restored save properly."

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
        pass
