#!/usr/bin/env python3


from .Player import Player
import random


class RandomPlayer(Player):

    """A class representing a random player.
    A random player who always make the move randomly.
    A random player can still observe the state of the game
    (however this does not impact their decicion for next move),
    then select a move to make.
    """

    def __init__(self, id):
        Player.__init__(self, id)

    def select_move(self, observation):
        """Allow the random player to select a move for the game

        :observation: Observation of the game
        :moves: moves options in the game
        :returns: move listed in observations

        """
        moves = observation['moves']
        print(f"\n\nIt's random player {self.id}'s turn.")
        print("Scores are:")
        for player_id, score in observation["scores"].items():
            print(f"{player_id}: {score}")

        print("The game looks like")
        print(observation["observation"])
        move = random.choice(moves)
        print(f"Our random player {self.id} guessed {move}")
        return move
