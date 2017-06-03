import gym
from data import tidigits
from data.tidigits import DigitsSampleCollection


class MfccFrozenlake(gym.Wrapper):
    """
    NOTE: Wrap this env around a raw FrozenLake-v0 type env
    """
    def __init__(self, env, audio_sample_ids, usage, num_mfcc=13):
        super(MfccFrozenlake, self).__init__(env)
        self.num_mfcc = num_mfcc
        self._audio_sample_ids = audio_sample_ids
        self.digits_sample_collection = DigitsSampleCollection(audio_sample_ids)
        self.usage = usage

    # Private method
    def _speak_state_as_mfccs(self, state):
        digits_sample = self.digits_sample_collection.choose_random_sample(state)
        return digits_sample.to_mfccs(self.num_mfcc)

    # Wrapper overload
    def _reset(self):
        return self._speak_state_as_mfccs(self.env.reset())

    # Wrapper overload
    def _step(self, action):
        next_state, reward, done, info = self.env.step(action)

        # So we have a way to access the ground truth state
        info['state'] = next_state

        # Convert audio to mfccs
        next_state_mfccs = self._speak_state_as_mfccs(next_state)

        return next_state_mfccs, reward, done, info

    @classmethod
    def make_train_val_test_envs(cls, base_env_name, data_splits=None, num_mfcc=13):
        data_splits = data_splits or tidigits.get_split_dataset()
        train_env, val_env, test_env = [
            MfccFrozenlake(gym.make(base_env_name), data_splits[usage], usage, num_mfcc=num_mfcc)
            for usage in ('train', 'test', 'val')
        ]
        return train_env, val_env, test_env
