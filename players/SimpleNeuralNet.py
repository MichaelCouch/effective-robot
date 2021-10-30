import numpy as np
from numpy.random import random_sample
from scipy.special import softmax

from .Player import Player

class SimpleNeuralNet(Player):

    """A complete neural network with a single hidden layer and"""

    def __init__(self, id, n, discount=0.9, learning_rate=0.1, visualize=False):
        """
        The number of nodes in the hidden layer
        """
        Player.__init__(self, id)
        self._n = n
        self._discount = discount
        self._learning_rate = learning_rate
        self._visualize = visualize
        if self._visualize:
            self._start_image_window()

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

        self._last_score = 0
        self._last_action = None

    def _update_network(self, observation):
        """ Update the neural network based on the new score we see
        """
        ## Update the neural network based on our new score
        previous_state_action_value = self._output['activation'][self._actions.index(self._last_action)]
        new_score = observation['scores'][self.id]
        points_awarded = self._last_score = new_score

        # Update the state
        self._update_activations()
        current_state_value = np.max(self._output['activation'])

        error = points_awarded - (
            previous_state_action_value - self._discount * current_state_value
        )
        update_vector = np.zeros(self._output['activation'].shape)
        update_vector[self._actions.index(self._last_action), 0] = error

        # Update network weights and biases by gradient descent
        dp2 = self._dp2(self._Wo.dot(self._hidden['activation']) - self._output['bias'])
        dp1 = self._dp1(self._Wi.dot(self._input) - self._hidden['bias'])

        dWo = np.tensordot(dp2, self._hidden['activation'], axes=(1,1))
        dWi = np.tensordot(np.transpose(dp2 * self._Wo) * dp1, self._input, axes=0)
        dob = - dp2
        dhb = - np.transpose(dp2 * self._Wo) * dp1

        self._Wi += self._learning_rate * np.tensordot(dWi, update_vector,axes=([1,3], [0,1]))
        self._hidden['bias'] += self._learning_rate * np.tensordot(dhb, update_vector, axes=(1,0))
        self._Wo += self._learning_rate * dWo * update_vector
        self._output['bias'] += self._learning_rate * dob * update_vector

    def _select_valid_move(self, observation):
        """Use activation levels to randomly select a valid move to make.

        :obsercation: TODO
        :returns: a valid moce

        """
        # Not all actions we know about are valid so restrict
        valid = [action in observation['moves'] for action in self._actions]
        moves = [action for action in self._actions if action in observation['moves']]

        # Use activation levels to pick a move randomly
        cdf = np.cumsum(softmax(self._output['activation'][valid]))
        r = random_sample()
        move = 0
        while r > cdf[move]:
            move += 1

        return self._actions[move]

    def _start_image_window(self):
        """Initialize a pyglet window"""
        import pyglet
        from pyglet import shapes, clock
        self._width =  640
        self._height = 480
        self._window = pyglet.window.Window(self._width, self._height)
        self._batch = pyglet.graphics.Batch()
        pyglet.app.run()


    def _update_image(self):
        """Update the image
        :returns: None

        """
        input_x = self._width * 1 / 8
        hidden_x = 640 * 1 / 2
        action_x = self._width * 7 / 8

        for i, var in enumerate(self._input):
            shapes.Circle(input_x, self._height * (i+1)/(len(self._input) + 1),5,
                          color = (155,155,155), batch=self._batch)
        for i, var in enumerate(self._hidden['activation']):
            shapes.Circle(hidden_x, self._height * (i+1)/(len(self._hidden['activation']) + 1),5,
                          color = (155,155,155), batch=self._batch)
        for i, var in enumerate(self._output['activation']):
            shapes.Circle(action_x, self._height * (i+1)/(len(self._hidden['output']) + 1),5,
                          color = (155,155,155), batch=self._batch)
        self._window.clear()
        self._batch.draw()


    def select_move(self, observation):

        # Update known state data
        for var in observation['observation']:
            if var['name'] in self._variables:
                self._input[self._variables.index(var['name']), 0] = var['value']

        ## Update the neural network based on our new score
        if self._last_score and self._last_action:
            self._update_network(observation)

        ## Add data of any never-before-seen moves and state data
        for action in observation['moves']:
            if action not in self._actions:
                self._add_action(action)
        for var in observation['observation']:
            if var['name'] not in self._variables:
                assert isinstance(var['value'], float) or isinstance(var['value'],int), 'Only accepting atomic numbers for SimpleNeuralNet'
                self._add_variable(var)

        ## Select the next move to make
        self._update_activations()
        if self._visualize:
            self._update_image()
        if observation['moves'] == ['GAME_OVER']:
            next_action = 'GAME_OVER'
            self._last_score = 0
            self._last_action = None
        else:
            next_action = self._select_valid_move(observation)

            self._last_score = observation['scores'][self.id]
            self._last_action = next_action
        print(list(zip(*[self._output['activation'], self._actions])))
        return next_action


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

