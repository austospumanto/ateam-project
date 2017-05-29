import gym
from speech.digits import MfccDeriver


class MfccFrozenlake(gym.Wrapper):
    """
    NOTE: You have to wrap this around an env wrapped in AudioFrozenlake class
          (see _reset and _step methods for why)
    """
    def __init__(self, env, num_mfcc=13):
        super(MfccFrozenlake, self).__init__(env)
        self._mfcc_deriver = MfccDeriver(num_mfcc)

    # Private method
    def _convert_audio_to_mfccs(self, audio_signal):
        return self._mfcc_deriver.derive(audio_signal)

    # Wrapper overload
    def _reset(self):
        return self._convert_audio_to_mfccs(self.env.reset())

    # Wrapper overload
    def _step(self, action):
        next_state_audio, reward, done, info = self.env.step(action)

        # Convert audio to mfccs
        next_state_mfccs = self._convert_audio_to_mfccs(next_state_audio)

        return next_state_mfccs, reward, done, info
