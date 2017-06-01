from speech.digits import DigitsSpeaker
import gym


class AudioFrozenlake(gym.Wrapper):
    def __init__(self, env, usage='train'):
        super(AudioFrozenlake, self).__init__(env)
        self._digits_speaker = DigitsSpeaker()
        self._usage = usage;

    # Private method
    def _speak_state(self, state):
        return self._digits_speaker.speak(state, usage=self._usage)

    # Wrapper overload
    def _reset(self):
        return self._speak_state(self.env.reset())

    # Wrapper overload
    def _step(self, action):
        next_state, reward, done, info = self.env.step(action)

        # So we have a way to access the ground truth state
        info['state'] = next_state

        # Convert state int to pure audio
        next_state_audio = self._speak_state(next_state)

        return next_state_audio, reward, done, info
