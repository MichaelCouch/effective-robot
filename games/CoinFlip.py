#!/usr/bin/env python3

from .Game import Game
from random import random


class CoinFlip(Game):

    """A game where a weighted coin is flipped"""

    def __init__(self, players, coin_bias=0.5):
        """
        :player: list of players for the game
        :coin_bias: between 0 and 1.

        """
        Game.__init__(self, players)

        self._bias = coin_bias
        self._turn_order = [player_id for player_id in self._players]
        self._next_player = 0

    def get_player_observation(self, player_id):
        """Provide the observation of the state of the game to the play

        :player_id: ID of the player
        :returns: dict

        """
        return {
            "moves": ['h', 't'],
            "observation": [],
            "scores": {
                player_id: player_data['score']
                for player_id, player_data in self._players.items()
            },
            "game_over": False
        }

    def get_next_player(self):
        """Returns the player_id of the player who's turn it is next
        :returns: player_id

        """
        return self._turn_order[self._next_player]

    def make_move(self, player_id, guess):
        """player_id selects heads (h) or tails (t), and the coin is flipped.

        :player_id: a player id
        :guess: 'h' or 't'
        :returns: None

        """
        flip = 'h' if random() < self._bias else 't'
        if flip == guess:
            self._players[player_id]['score'] += 1

        self._next_player = (self._next_player + 1) % len(self._players)
