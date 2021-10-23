import numpy as np
from numpy.random import random_sample
from scipy.special import softmax

from .Player import Player

class SimpleNeuralNet(Player):

    """A complete neural network with a single hidden layer and"""

    def __init__(self, id, n):
        """
        The number of nodes in the hidden layer
        """
        Player.__init__(self, id)
        self._n = n
        self._input = random_sample((0,))

        # Perceptron
        self._p = lambda x: 1  / (1 - np.exp(-x))
        self._dp = lambda x: np.exp(-x) / (1 - np.exp(-x))**2

        # Biases and activation levels
        self._hidden = {'bias': random_sample((1, self._n)), 'activation': np.zeros((1, self._n))}
        self._output = {'bias': random_sample((self._n, 0)), 'activation': np.zeros(0)}

        # Weights
        self._Wi = random_sample((0, self._n))
        self._Wo = random_sample((self._n, 0))

        self._variables = []
        self._actions = []

        self._last_score = None
        self._last_action = None


    def select_move(self, observation):
        actions = observation['moves']
        for action in actions:
            if action not in self._actions:
                self._add_action(action)
        variables = observation['observation']
        for var in variables:
            if var['name'] not in self._variables:
                assert isinstance(var['value'], float) or isinstance(var['value'],int), 'Only accepting atomic numbers for SimpleNeuralNet'
                self._add_variable(var['name'])

        ...
        cdf = np.cumsum(softmax(self._output['activation']))
        r = random_sample()
        move = 0
        while r > cdf[move]:
            move += 1
        return self._actions[move]


    def _add_action(self, action):
        """Adds a new action type to the to network

        :action: TODO
        :returns: TODO

        """
        pass

    def _add_variable(self, var):
        """Adds an input variable to the network

        :var: TODO
        :returns: TODO

        """
        pass


