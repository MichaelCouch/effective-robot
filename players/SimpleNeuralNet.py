import numpy as np
from numpy.random import random_sample
from scipy.special import softmax

from .Player import Player

class SimpleNeuralNet(Player):

    """A complete neural network with a single hidden layer and"""

    def __init__(self, id, n, discount=0.9, learning_rate=0.1):
        """
        The number of nodes in the hidden layer
        """
        Player.__init__(self, id)
        self._n = n
        self._discount = discount
        self._learning_rate = learning_rate

        self._input = random_sample((0, 1))

        # Perceptrons
        self._p1 = lambda x: 1  / (1 - np.exp(-x))
        self._dp1 = lambda x: np.exp(-x) / (1 - np.exp(-x))**2
        self._p2 = lambda x: 1  / (1 - np.exp(-x))
        self._dp2 = lambda x: np.exp(-x) / (1 - np.exp(-x))**2

        # Biases and activation levels
        self._hidden = {'bias': random_sample((self._n, 1)), 'activation': np.zeros((self._n, 1))}
        self._output = {'bias': random_sample((0, 1)), 'activation': np.zeros((0, 1))}

        # Weights
        self._Wi = random_sample((self._n, 0))
        self._Wo = random_sample((0, self._n))

        self._variables = []
        self._actions = []

        self._last_score = None
        self._last_action = None

    def select_move(self, observation):
        # Update known state data
        for var in observation['observations']:
            if var['name'] in self._variables:
                self._input[self._variables.index(var['name']),1] = var['value']

        if self._last_score and self._last_action:
            ## Update the neural network based on our new score
            previous_state_action_value = self._output['activations'][self._actions.index(self._last_action)]
            new_score = observation['scores'][self._id]
            points_awarded = self._last_score = new_score

            # Update the state
            self._update_activations()
            current_state_value = np.max(self._output['activations'])

            error = points_awarded - (
                previous_state_action_value - self._discount * current_state_value
            )
            update_vector = np.zeros(self._output['activations'].shape)
            update_vector[self._actions.index(self._last_action),1] = error

            # Update network weights and biases by gradient descent

            dWi = (
                self._dp2(self._Wo.dot(self._hidden['activation']) - self._output['bias'])
                * self._dp1(self._Wi.dot(self._input) - self._hidden['bias'])
                * self._Wo * self._input
            )

            dWo = (
                self._dp2(self._Wo.dot(self._hidden['activation']) - self._output['bias'])
                * self._hidden['activation']
            )

            dhb = - (
                self._dp2(self._Wo.dot(self._hidden['activation']) - self._output['bias'])
                * self._dp1(self._Wi.dot(self._input) - self._hidden['bias'])
                * self._Wo
            )

            dob = - (
                self._dp2(self._Wo.dot(self._hidden['activation']) - self._output['bias'])
            )

            self._Wi += self._learning_rate * dWi.dot(update_vector)
            self._Wo += self._learning_rate * dWo.dot(update_vector)
            self._hidden['bias'] += self._learning_rate * dhb.dot(update_vector)
            self._output['bias'] += self._learning_rate * dob.dot(update_vector)
            

        ## Add data of any never-before-seen moves and state data
        actions = observation['moves']
        for action in actions:
            if action not in self._actions:
                self._add_action(action)
        variables = observation['observation']
        for var in variables:
            if var['name'] not in self._variables:
                assert isinstance(var['value'], float) or isinstance(var['value'],int), 'Only accepting atomic numbers for SimpleNeuralNet'
                self._add_variable(var)

        ## Select the next move to make
        self._update_activations()
        # Not all actions we know about are valid so restrict
        valid = [action in actions for action in self._actions]
        moves = [action for action in self._actions if action in actions]

        # Use activation levels to pick a move randomly
        cdf = np.cumsum(softmax(self._output['activation'][valid]))
        r = random_sample()
        move = 0
        while r > cdf[move]:
            move += 1

        self._last_score = observation['scores'][self._id]
        self._last_action = self._actions[move]
        return self._last_action


    def _add_action(self, action):
        """Adds a new action type to the to network

        :action: A move
        :returns: None

        """
        self._actions.append(action)
        self._Wo = np.r_[self._Wo, random_sample((1, self._n))]
        self._output['bias'] = np.r_[self._output['bias'], random_sample((1, 1))]
        self._output['activation'] = np.r_[self._output['activation'], np.zeros((1, 1))]
        self._update_activations()

    def _add_variable(self, var):
        """Adds an input variable to the network

        :var: TODO
        :returns: TODO

        """
        self._variables.append(var['name'])
        # Assuming all variable values are atomic numbers
        self._input = np.r_[self._input, np.array([[var['value']]])]
        self._Wi = np.c_[self._Wi, random_sample((self._n, 1))]
        self._update_activations()

    def _update_activations(self):
        """Update the neuron activation levels based on current weights, 
        biases, and input

        :returns: None

        """
        # Update from start to end of neural network
        self._hidden['activation'] = self._p1(
            self._Wi.dot(self._input) \
            - self._hidden['bias'] \
        )

        self._output['activation'] = self._p2(
            self._Wo.dot(self._hidden['activation']) \
            - self._output['bias'] \
        )

