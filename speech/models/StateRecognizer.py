import random
import logging

logger = logging.getLogger(__name__)


class StateRecognizer(object):
    def __init__(self, valid_states, ctc_model, verbose=False):
        self._valid_states = [int(state) for state in valid_states]
        self._ctc_model = ctc_model
        self._verbose = verbose

    def recognize(self, state_features):
        state = self._ctc_model.classify_mfccs(state_features)
        if not state.isdigit() or int(state) not in self._valid_states:
            rand_state = random.choice(self._valid_states)
            if self._verbose:
                logger.info('Recognized state %s is not a valid state for this environment.' % state + \
                            'Returning random env state: %d' % rand_state)
            state = str(rand_state)
        else:
            if self._verbose:
                logger.info('Recognized state = %s' % state)
        return int(state)
