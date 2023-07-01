from enum import Enum

#class for which board action to perform
class BoardAction(Enum):
    NOTHING = 0
    SOLVE = 1
    RESET = 2
    NEWGAME = 3
    HOME = 4
    GENERATE = 5
