#!/usr/bin/env python3


class Player(object):

    """A class representing an abstract player.
    Players observe the state of the game, then select a move to make"""

    def __init__(self, id):
        self.id = id

    def select_move(self, observation):
        ...


