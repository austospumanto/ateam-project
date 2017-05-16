

class DigitsRecognizer(object):
    # TODO Viggy
    def recognize(self, features, raw=False):
        """
        Translates MFCC features/raw audio of spoken digits to a string of the digits. ASR
        
        :param features: The MFCC features of an audio sample of a person saying a string of digits
        :param raw: If False, we are using MFCC features.
        :return: digit
        """
        pass


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