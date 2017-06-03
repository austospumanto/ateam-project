from data.tidigits import DigitsSampleCollection
from data import tidigits
import gym


class AudioFrozenlake(gym.Wrapper):
    """
    NOTE: Wrap this env around a raw FrozenLake-v0 type env
    """
    def __init__(self, env, audio_sample_ids, usage):
        super(AudioFrozenlake, self).__init__(env)
        self._audio_sample_ids = audio_sample_ids
        self.digits_sample_collection = DigitsSampleCollection(audio_sample_ids)
        self.usage = usage

    # Private method
    def _speak_state_as_audio(self, state):
        digits_sample = self.digits_sample_collection.choose_random_sample(state)
        return digits_sample.audio

    # Wrapper overload
    def _reset(self):
        return self._speak_state_as_audio(self.env.reset())

    # Wrapper overload
    def _step(self, action):
        next_state, reward, done, info = self.env.step(action)

        # So we have a way to access the ground truth state
        info['state'] = next_state

        # Convert state int to pure audio
        next_state_audio = self._speak_state_as_audio(next_state)

        return next_state_audio, reward, done, info

    @classmethod
    def make_train_val_test_envs(cls, base_env_name, data_splits=None):
        data_splits = data_splits or tidigits.get_split_dataset()
        train_env, val_env, test_env = [
            AudioFrozenlake(gym.make(base_env_name), data_splits[usage], usage)
            for usage in ('train', 'test', 'val')
        ]
        return train_env, val_env, test_env
