from speech.digits import DigitsSpeaker


class MfccFrozenlake(object):
    def __init__(self, env, raw=False, num_mfcc=13):
        self._env = env
        self._digits_speaker = DigitsSpeaker()
        self.raw = raw
        self.num_mfcc = num_mfcc
        self.nA = env.nA

    def speak_state(self, state):
        return self._digits_speaker.speak(str(state), raw=self.raw, mfcc=self.num_mfcc)

    def reset(self):
        return self.speak_state(0)

    def step(self, action):
        next_state, reward, done, info = self._env.step(action)
        next_state_features = self.speak_state(next_state)
        return next_state_features, reward, done, info

    def __getattr__(self, name):
        return getattr(self._env, name)
