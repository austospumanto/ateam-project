from __future__ import print_function
import random


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
