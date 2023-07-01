from enum import Enum

#class for which screen to be on
class GameState(Enum):
    QUIT = -1
    TITLE = 0
    GAME_BOARD = 1
    SOLVER = 2
