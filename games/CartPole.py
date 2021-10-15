from numpy import random, sin, cos, array, pi, floor
from .Game import Game


class CartPole(Game):

    """
    Cartpole

    Defines the cartpole game. A cart sits on a track, with a pole
    standing vertically. The AI can nudge the cart left and right down the
    track and is rewarded each timestep for keeping the pole upright

           |
          /
        __|__
    ===o=====o===

    """

    def __init__(self, players, pos=None, vel=None, angle=None, m=None,
                 time_step=None):
        """Initializes the cartpole game """
        Game.__init__(self, players)
        # position of the centre of the cart
        self._pos = 0 if pos is None else pos
        # the velocity of the cart
        self._vel = 0 if vel is None else vel
        # angle of the pole relative to vertical in radian
        self._angle = 0 if angle is None else angle
        # angular velocity of the pole relative in radian
        self._ang_vel = (random.rand()-0.5) / 100
        # the mass of the pole
        self._m = 1 if m is None else m
        self._game_over = False
        self._time_step = 0.1 if time_step is None else time_step
        # It's a single player game
        self._player_id = list(self._players.keys())[0]
        self._pole_owner = list(self._players.keys())[1] + "'s " if len(self._players) > 1 else 'the '
 

    def get_cur_state(self):
        return array([self._pos, self._vel, self._angle])

    def get_player_observation(self, player_id):
        """Provide the observation of the state of the game to the play

        :player_id: ID of the player
        :returns: dict

        """
        return {
            "moves": ['l', 'r', 'o'],
            "observation": CartPoleObservation(
                self._pos, self._vel, self._angle, self._ang_vel,
                self._time_step, self._m, self._game_over),
            "scores": {
                player_id: player_data['score']
                for player_id, player_data in self._players.items()
            },
            "game_over": self._game_over
        }

    def get_next_player(self):
        return self._player_id

    def make_move(self, player_id, move):
        if move == 'l':
            d = -1
        elif move == 'r':
            d = +1
        elif move == 'o':
            d = 0

        pos = self._pos + self._vel * self._time_step
        vel = self._vel + d * self._time_step
        angle = self._angle + self._ang_vel * self._time_step
        ang_vel = self._ang_vel + self._time_step * (
            - d / self._m * cos(self._angle)
            + sin(self._angle)
            - 2 * sin(self._angle)**3 / cos(self._angle) * (self._ang_vel)**2
        )

        self._pos, self._vel, self._angle, self._ang_vel = pos, vel, angle, ang_vel

        if abs(self._angle) > pi/2:
            self._game_over = True

        self._players[player_id]['score'] += 0 if self._game_over else self._time_step

        if self._game_over:
           score = self._players[player_id]['score']
           print("The pole has fallen.")
           print(f"Player {player_id} kept {self._pole_owner} pole up for {round(score,1)} seconds!")


class CartPoleObservation():
    """ An observation object. Should have an __iter__ method,
    and a __str__ method """

    def __init__(self, pos, vel, angle, ang_vel, time_step, m, game_over):

        """As per CartPole

        :pos:
        :vel: TODO
        :angle: TODO
        :ang_vel: TODO
        :time_step: TODO
        :m: TODO

        """
        self._pos = pos
        self._vel = vel
        self._angle = angle
        self._ang_vel = ang_vel
        self._time_step = time_step
        self._m = m
        self._game_over = game_over
        self._data = [
            {'name': 'position', 'type': 'continuous', 'value': self._pos},
            {'name': 'velocity', 'type': 'continuous', 'value': self._vel},
            {'name': 'angle', 'type': 'continuous', 'value': self._angle},
            {'name': 'ang_vel', 'type': 'continuous', 'value': self._ang_vel},
            {'name': 'time_step', 'type': 'continuous', 'value': self._time_step},
            # This is kinda optional - do we want the AI to know
            # how heavy the object is?
            {'name': 'm', 'type': 'continuous', 'value': self._m},
        ]

    def __iter__(self):
        """ Defines, what happens with "for item in self" """
        self.n = 0
        return self

    def __next__(self):
        """ Defines, what happens with "for item in self" """
        if self.n < len(self._data):
            return self._data[self.n]
        else:
            raise StopIteration

    def __str__(self):
        """ Defines what print(self) means """
        top = "\n"
        track = "===o=====o===\n"

        poles = [
            "             \n"
            "    __       \n"
            "    __\__    \n",

            "    _        \n"
            "     \       \n"
            "    __\__    \n",

            "    \        \n"
            "     \       \n"
            "    __\__    \n",

            "     \       \n"
            "      \      \n"
            "    __|__    \n",

            "     |       \n"
            "      \      \n"
            "    __|__    \n",

            "      \      \n"
            "      |      \n"
            "    __|__    \n",

            "      /      \n"
            "      |      \n"
            "    __|__    \n",

            "       |     \n"
            "      /      \n"
            "    __|__    \n",


            "       /     \n"
            "      /      \n"
            "    __|__    \n",

            "        /    \n"
            "       /     \n"
            "    __/__    \n",

            "        _    \n"
            "       /     \n"
            "    __/__    \n",

            "             \n"
            "       __    \n"
            "    __/__    \n"
        ]
        pole = poles[
            max(min(int(
                floor((self._angle / pi + 1 / 2) * (len(poles)))
            ), len(poles)), 0)
        ]

        result = top + pole + track
        x, v, a, z = (round(self._pos, 2),
                      round(self._vel, 2),
                      round(self._angle * 180 / pi),
                      round(self._ang_vel * 180 / pi))
        result += "\n"
        result += f"\nThe cart is at position {x} travelling at speed {v},"
        result += f"\nthe pole is at an angle {a} degrees and with"
        result += f"\nangular velocity {z} degrees per second."
        if self._game_over:
            result += "\nThe pole has fallen over!"
        return result
