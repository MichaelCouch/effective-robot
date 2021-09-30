#!/usr/bin/env python3


class Game(object):

    """A generic game class. A game has players, scores """

    def __init__(self, players):
        """
        :players: list of Players

        """
        self._players = self._initialize_players(players)
        self._game_state = None

    def _initialize_players(self, players):
        """Set up a data structure containing the players in the game,
        and their current scores.

        :players: list of plauers
        :returns: dictionary

        """
        return {
            player.id: {
                 "player": player,
                 "score": 0,
            } for player in players
        }

    def take_turn(self, player_id):
        """Allow a player to take their turn in the game.
        :returns: None

        """
        observation = self.get_player_observation(player_id)

        player = self._players[player_id]

        move = player.select_move(observation)

        self.make_move(player_id, move)

    def make_move(self, player_id, move):
        """The move is made in the game

        :player_id: id of the player
        :move: move selected to be made
        :returns: None

        """
        pass

    def get_next_player(self):
        """Returns the player_id of the next player
        :returns: id

        """
        pass

    def get_player_observation(self, player_id):
        """Return the observation of the game that player_id makes

        result = {
            "moves": [...],
            "observation": [
                {'name': name of variable,
                 'type': type of data - continuous, discrete, ordinal, categorical
                 'value': value of data (atomic, or numpy array)
                }, ...
            ],
            'scores': {player_id: score, ...}
            "game_over": False
        }

        :player_id: id of the player
        :returns: dict of legal moves, and an observation of the game state

        """
        pass

