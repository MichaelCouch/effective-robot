#!/usr/bin/env python3


from .Player import Player
import random


class Randomplayer(Player):

    """A class representing a random player.
        A random player who always guess randomly with the flipped coin game. 
        A random player can still observe the state of the game (however this does not impact their decicion for next move), then select a move to make"""

    def __init__(self, id):
        Player.__init__(self, id)

    
    def select_move(self, observation):
        """Allow the random player to select a move for the game

        :observation: Observation of the game
        :returns: move listed in observations

        """

        
        print(f"\n\nIt's random player {self.id}'s turn.")
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
        
        move = random.choice(moves)
        print(f"Our random player {self.id} guessed {move} ")
        return move

