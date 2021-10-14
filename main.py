#!/usr/bin/env python3

from games import CoinFlip
from games import CartPole
from players import Human
from players import RandomPlayer

players = [Human("Lily"), Human("Michael"), RandomPlayer("Lilith")]

game = CartPole(players)

end_scores = game.run_game()
print(end_scores)
