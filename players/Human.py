#!/usr/bin/env python3

from .Player import Player


class Human(Player):
    """A class implementing a human player manually taking turns
    """

    def __init__(self, id):
        Player.__init__(self, id)

    def select_move(self, observation):
        """Allow the human player to select a move for the game

        :observation: Observation of the game
        :returns: move listed in observations

        """
        print(f"\n\nIt's human player {self.id}'s turn.")
        print("Scores are:")
        for player_id, score in observation['scores'].items():
            print(f"{player_id}: {score}")

        print("The game looks like")
        print(observation['observation'])

        return self.read_move(observation['moves'])

    def read_move(self, moves):
        """Get the next player move, with checking

        :moves: list of strings
        :returns: element in moves

        """
        valid_move = False
        while not valid_move:
            move = input(f"Please enter a valid move ({moves}): ")
            if move in moves:
                return move
