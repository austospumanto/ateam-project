from __future__ import print_function
import numpy as np
import random
from utils.test_env import EnvTest


class LinearSchedule(object):
    def __init__(self, eps_begin, eps_end, nsteps):
        """
        Args:
            eps_begin: initial exploration
            eps_end: end exploration
            nsteps: number of steps between the two values of eps
        """
        self.epsilon        = eps_begin
        self.eps_begin      = eps_begin
        self.eps_end        = eps_end
        self.nsteps         = nsteps


    def update(self, t):
        """
        Updates epsilon

        Args:
            t: (int) nth frames
        """
        ##############################################################
        """
        TODO: modify self.epsilon such that 
               for t = 0, self.epsilon = self.eps_begin
               for t = self.nsteps, self.epsilon = self.eps_end
               linear decay between the two

              self.epsilon should never go under self.eps_end
        """
        ##############################################################
        ################ YOUR CODE HERE - 3-4 lines ################## 
        self.epsilon = ((self.eps_end - self.eps_begin) / float(self.nsteps)) * t + self.eps_begin
        self.epsilon = max(self.epsilon, self.eps_end)
        ##############################################################
        ######################## END YOUR CODE ############## ########


class LinearExploration(LinearSchedule):
    def __init__(self, env, eps_begin, eps_end, nsteps):
        """
        Args:
            env: gym environment
            eps_begin: initial exploration
            eps_end: end exploration
            nsteps: number of steps between the two values of eps
        """
        self.env = env
        super(LinearExploration, self).__init__(eps_begin, eps_end, nsteps)


    def get_action(self, best_action):
        """
        Returns a random action with prob epsilon, otherwise return the best_action

        Args:
            best_action: (int) best action according some policy
        Returns:
            an action
        """
        ##############################################################
        """
        TODO: with probability self.epsilon, return a random action
               else, return best_action

               you can access the environment stored in self.env
               and epsilon with self.epsilon
        """
        ##############################################################
        ################ YOUR CODE HERE - 4-5 lines ##################
        if self.epsilon > random.random():
            return self.env.action_space.sample()
        else:
            return best_action
        ##############################################################
        ######################## END YOUR CODE ############## ########



def test1():
    env = EnvTest((5, 5, 1))
    exp_strat = LinearExploration(env, 1, 0, 10)
    
    found_diff = False
    for i in range(10):
        rnd_act = exp_strat.get_action(0)
        if rnd_act != 0 and rnd_act is not None:
            found_diff = True

    assert found_diff, "Test 1 failed."
    print("Test1: ok")


def test2():
    env = EnvTest((5, 5, 1))
    exp_strat = LinearExploration(env, 1, 0, 10)
    exp_strat.update(5)
    assert exp_strat.epsilon == 0.5, "Test 2 failed"
    print("Test2: ok")


def test3():
    env = EnvTest((5, 5, 1))
    exp_strat = LinearExploration(env, 1, 0.5, 10)
    exp_strat.update(20)
    assert exp_strat.epsilon == 0.5, "Test 3 failed"
    print("Test3: ok")


def your_test():
    """
    Use this to implement your own tests
    """
    # Define hyperparams
    eps_begin = 1
    eps_end = 0.1
    nsteps = 10

    # Set up the environment and model
    env = EnvTest((5, 5, 1))
    exp_strat = LinearExploration(env, eps_begin, eps_end, nsteps)

    # Initialize the environment
    cur_state_obs_space = env.reset()
    env.render()
    done = False
    tot_reward = 0.0

    # Train/explore
    while not done:
        exp_strat.update(env.num_iters)
        best_action = np.argmax(cur_state_obs_space[env.cur_state, :, 0])
        action = exp_strat.get_action(best_action)
        next_state_obs_space, reward, done, _ = env.step(action)
        env.render()

        # Update reward, and observation space (state automatically updated)
        tot_reward += reward
        # ???: How update cur_state_obs_space? Do we need to?
        cur_state_obs_space = next_state_obs_space
    
    print('Total reward: %s' % tot_reward)


if __name__ == "__main__":
    test1()
    test2()
    test3()
    your_test()
