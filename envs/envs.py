from speech.digits import DigitsSpeaker


class MfccFrozenlake(object):
    def __init__(self, env):
        self._env = env
        self._digits_speaker = DigitsSpeaker()

    def step(self, action, raw=False):
        next_state, reward, done, info = self._env.step(action)
        next_state_features = self._digits_speaker.speak(str(next_state), raw=raw)
        return next_state, next_state_features, reward, done, info

    def __getattr__(self, name):
        return getattr(self._env, name)
