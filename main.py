#!/usr/bin/env python3

from games import CoinFlip
from players import Human 
from players import Randomplayer

players = [Human('Lily'), Human('Michael'), Randomplayer('Lilith')]

game = CoinFlip(players)

end_scores = game.run_game()
print(end_scores)
