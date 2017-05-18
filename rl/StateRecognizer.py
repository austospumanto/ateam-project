import random


class StateRecognizer(object):
    def __init__(self, env, recognizer):
        self._env = env
        self._recognizer = recognizer

    def recognize(self, *a, **kw):
        verbose = kw.get('verbose')
        state = self._recognizer.recognize(*a, **kw)
        if not state.isdigit() or int(state) not in self._env.P:
            rand_state = random.choice(self._env.P.keys())
            if verbose:
                print 'Recognized state %s is not a valid state for this environment.' % state + \
                      'Returning random env state: %d' % rand_state
            state = str(rand_state)
        else:
            if verbose:
                print 'Recognized state = %s' % state
        return int(state)
