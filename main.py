#!/usr/bin/env python3

from games import CoinFlip
from players import Human

players = [Human('Lily'), Human('Michael')]

game = CoinFlip(players)

end_scores = game.run_game()
print(end_scores)
