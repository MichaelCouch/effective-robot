#!/usr/bin/env python3

from time import sleep
from games import CoinFlip
from games import CartPole
from players import Human
from players import RandomPlayer
from players import SimpleNeuralNet

players = [
    SimpleNeuralNet('C3PO', n=2, visualize=True),
    # Human("Lily"),
    # Human("Michael"),
    RandomPlayer("Lilith")
]


playing = True
n = 0
while playing:
    game = CartPole(players)
    game.run_game()
    #if input('Play again?') != 'y':
    #    playing=False
    sleep(1000/1000)
    n+=1
print(end_scores)
