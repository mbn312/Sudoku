import IPython
from collections import deque
import sys
import time
import math
import pygame
import pygame.freetype
import random
import copy
from pygame.sprite import Sprite
from pygame.rect import Rect
from enum import Enum
from pygame.sprite import RenderUpdates

#####
from boardaction import BoardAction
from gamestate import GameState
from problem import Problem
#####


#RGB Codes for colors used
WHITE = (255,255,255)
BLACK = (0,0,0)
MAPLE = (241, 195, 142)
RED = (255,102,102)
LIGHT_RED = (255,153,153)
GREEN = (102,255,102)
LIGHT_GREEN = (204,255,153)
YELLOW = (255,255,204)
GRAY = (192,192,192)
DARK_GRAY = (64,64,64)
LIGHT_GRAY = (240,240,240)

COUNTER = 0

class Controls:

    def __init__(self,screen,width=400,height=700,x_margin=900,y_margin=50,buttons=[],start=0):
        self.screen = screen
        self.buttons = buttons
        self.width = width
        self.height = height
        self.x_margin = x_margin
        self.y_margin = y_margin
        self.start = start
        self.pause = 0

        
    def update_buttons(self,btns):
        self.buttons = btns

    def start_timer(self):
        self.start = time.time()
        self.pause = 0

    def pause_timer(self):
        self.pause = round(time.time() - self.start)

    def get_timer(self):
        timer = ""
        if self.pause == 0:
            total = round(time.time() - self.start)
        else:
            total = self.pause

        times = []
        increments = [1,60,60,24,365,100]
        show = False
        z = 1
        for inc in increments:
            z *= inc

        while len(increments) != 0:
            times.append(total//z)
            total = total % z
            z = z//increments.pop()

        while len(times) != 0:
            t = times.pop(0)
            if timer == "":
                if t != 0:
                    if (t // 10) == 0:
                        timer += "0" + str(t)
                    else:
                        timer += str(t)
            else:
                if (t // 10) == 0:
                    timer += ":0" + str(t)
                else:
                    timer += ":" + str(t)

        if len(timer) == 0:
            timer = "00:00"
        elif 0 < len(timer) < 5:
            timer = "00:" + timer

        timer = "Time Elapsed: " + timer

        return timer

    def draw(self,mouse_up):
        board_action = BoardAction.NOTHING

        pygame.draw.rect(self.screen, LIGHT_GRAY, [self.x_margin,self.y_margin,self.width,self.height])

        pygame.draw.line(self.screen, BLACK, (self.x_margin, self.y_margin), (self.x_margin, self.height+self.y_margin), 2)
        pygame.draw.line(self.screen, BLACK, (self.x_margin, self.height + self.y_margin), (self.width + self.x_margin, self.height+self.y_margin), 2)
        pygame.draw.line(self.screen, BLACK, (self.width + self.x_margin, self.y_margin), (self.x_margin, self.y_margin), 2)
        pygame.draw.line(self.screen, BLACK, (self.width + self.x_margin, self.y_margin), (self.width + self.x_margin, self.height+self.y_margin), 2)

        text = create_text("Sudoku",self.height/10,BLACK,LIGHT_GRAY)
        self.screen.blit(text, (self.x_margin + (self.width/2)-(text.get_width()/2),self.y_margin*2))
        pygame.draw.line(self.screen, BLACK, (self.x_margin,self.y_margin*4), (self.x_margin + self.width, self.y_margin*4), 1)

        for btn in self.buttons:
            btn.draw(self.screen)
            btn_action = btn.update(pygame.mouse.get_pos(),mouse_up)
            if btn_action is not None:
                board_action = btn_action
        
        if self.start != 0:
            timer = self.get_timer()
            text = create_text(timer,self.width/(len(timer)/2),BLACK,LIGHT_GRAY,False)
            pygame.draw.line(self.screen, BLACK, (self.x_margin,self.screen.get_height() - self.y_margin*4), (self.x_margin + self.width, self.screen.get_height() - self.y_margin*4), 1)
            self.screen.blit(text, (self.x_margin + (self.width/2)-(text.get_width()/2),self.screen.get_height() - self.y_margin*2.75))

        return board_action

class Game:

    def __init__(self,screen,bg_color=WHITE,selected=None,rows=9,cols=9,width=700,height=700,x_margin=100,y_margin=50,solutions=0,left=81,ended=False):
        self.ended = ended
        self.selected = selected
        self.select_input = None
        self.rows = rows
        self.cols = cols
        self.width = width
        self.height = height
        self.x_margin = x_margin
        self.y_margin = y_margin
        self.screen = screen
        self.solutions = solutions
        self.bg_color = bg_color
        self.original_board = [[0 for i in range(9)] for x in range(9)]
        self.solution_board = [[0 for i in range(9)] for x in range(9)]
        self.board = [[0 for i in range(9)] for x in range(9)]
        self.squares = [[Square(screen,(width/cols),(height/rows),x_margin,y_margin,row,col) for row in range(rows)] for col in range(cols)]
        self.problem = SudokuBoardImpliedFill({}, False)
        self.won = False
        self.solver = True
    
    #sets whether or not the game has ended
    def end(self,val):
        self.ended = val

    #sets all of its squares equal to the its boards values
    def set_all_squares(self):
        for row in range(self.rows):
            for col in range(self.cols):
                val = self.board[row][col]
                if val == 0:
                    val = None
                self.squares[row][col].set_value(val)
                self.squares[row][col].set_valid(None)
                self.squares[row][col].set_selected(False)
        

    #sets a single square and board value
    def set_square(self,val):
        pos = self.select_input
        if val is None:
            self.board[pos[0]][pos[1]] = 0
            self.problem.contents.pop((pos[0]+1,pos[1]+1),None)
        else:
            self.board[pos[0]][pos[1]] = val
            self.problem.contents[(pos[0]+1,pos[1]+1)] = val
        self.squares[pos[0]][pos[1]].set_value(val)
        
        if val is not None:
            for x in range(9):
                self.squares[x][pos[1]].set_temp_value(val,val_set=False)
                self.squares[pos[0]][x].set_temp_value(val,val_set=False)

            x = 3 * (pos[0] // 3)
            y = 3 * (pos[1] // 3)
            for r in range(3):
                for c in range(3):
                    self.squares[x+r][y+c].set_temp_value(val,val_set=False)
                    
        if self.problem.goal_test():
            self.won = True

    def set_problem(self):
        self.problem = SudokuBoardImpliedFill({}, False)
        for row in range(self.rows):
            for col in range(self.cols):
                if self.board[row][col] != 0:
                    self.problem.contents[(row+1,col+1)] = self.board[row][col]

    def clear_temp_values(self):
        for r in range(self.rows):
            for c in range(self.cols):
                self.squares[r][c].set_temp_value(0)

    #checks to see if any squares are invalid
    def check_invalid(self):
        for r in range(self.rows):
            for c in range(self.cols):
                if self.squares[r][c].valid is not None and not self.squares[r][c].valid:
                    return True
        return False

    #sets which square is selected
    def select_square(self,pos):
        if self.selected is not None and pos is not None and self.selected == pos:
            self.squares[pos[0]][pos[1]].set_input_state(True)
            self.select_input = pos
        else:
            if self.selected is not None:
                cur = self.selected
                self.squares[cur[0]][cur[1]].set_selected(False)
                self.squares[cur[0]][cur[1]].set_input_state(False)
                self.squares[cur[0]][cur[1]].set_temp_input(None)
            self.selected = pos
            if pos is not None:
                self.squares[pos[0]][pos[1]].set_selected(True)
                if self.solver:
                    self.select_input = pos
                    self.squares[pos[0]][pos[1]].set_input_state(True)

    #returns the position of where the users mouseclick was on the board
    def click(self, mouse_pos):
        pos = (int((mouse_pos[1] - self.y_margin)/(self.height/9)),int((mouse_pos[0] - self.x_margin)/(self.width/9)))
        return pos

    #creates a random new board
    def new_board(self):
        self.original_board,self.solution_board = create_board()
        self.board = copy.deepcopy(self.original_board)
        self.set_all_squares()
        self.selected = None
        self.clear_temp_values()
        self.set_problem()
        self.solver = False

    #clears all the values from the board
    def clear_board(self):
        self.problem = SudokuBoardImpliedFill({}, False)
        self.original_board = [[0 for i in range(9)] for x in range(9)]
        self.solution_board = [[0 for i in range(9)] for x in range(9)]
        self.board = [[0 for i in range(9)] for x in range(9)]
        self.squares = [[Square(self.screen,(self.width/self.cols),(self.height/self.rows),self.x_margin,self.y_margin,row,col) for row in range(self.rows)] for col in range(self.cols)]
        self.selected = None
        self.clear_temp_values()

    #resets the board to the original version
    def reset_board(self):
        self.board = copy.deepcopy(self.original_board)
        self.set_all_squares()
        self.selected = None
        self.clear_temp_values()
        self.set_problem()

    #checks to see if all the spots are filled to determine if the winner has won
    def check_win(self):
        if self.won:
            self.win()
            self.ended = True
            return True
        else:
            return False

    #sets all the squares as valid
    def win(self):
        for r in range(self.rows):
            for c in range(self.cols):
                self.squares[r][c].set_valid(True)

    #sets all the squares that are wrong to invalid
    def lose(self):
        for r in range(self.rows):
            for c in range(self.cols):
                if self.squares[r][c].value != self.solution_board[r][c]:
                    self.squares[r][c].set_valid(False)

    #draws the board
    def draw(self):

        space = self.width/9
        pygame.draw.rect(self.screen, WHITE, [self.x_margin,self.y_margin,self.width,self.height])

        for row in range(self.rows):
            for col in range(self.cols):
                self.squares[row][col].draw()

        for i in range(self.rows + 1):
            if i % 3 == 0:
                pygame.draw.line(self.screen, BLACK, (self.x_margin, i*space+self.y_margin), (self.width +self.x_margin, i*space+self.y_margin), 4)
                pygame.draw.line(self.screen, BLACK, (i * space+self.x_margin, self.y_margin), (i * space+self.x_margin, self.height+self.y_margin), 4)
            else:
                pygame.draw.line(self.screen, BLACK, (self.x_margin, i*space+self.y_margin), (self.width+self.x_margin, i*space+self.y_margin), 1)
                pygame.draw.line(self.screen, BLACK, (i * space+self.x_margin, self.y_margin), (i * space+self.x_margin, self.height+self.y_margin), 1)


    def elimination_solve(self,verbose=False):
        done = False
        level = 0
        x = 0
        while (not done and not self.problem.has_conflict):
            lst = self.problem.find_implied_fills(verbose)
            if len(lst) != 0:
                level = 1
                x = 0
                self.elim_solver_set(lst)
            else:
                lst = self.problem.find_row_fills(verbose)
                if len(lst) != 0:
                    level = 2
                    x = 0
                    self.elim_solver_set(lst)
                else:
                    lst = self.problem.find_col_fills(verbose)
                    if len(lst) != 0:
                        level = 3
                        x = 0
                        self.elim_solver_set(lst)
                    else:
                        lst = self.problem.find_block_fills(verbose)
                        if len(lst) != 0:
                            level = 4
                            x = 0
                            self.elim_solver_set(lst)
                        else:
                            if level < 5:
                                level = 5
                                self.problem.find_pointed_pairs(verbose)
                            else:
                                if level < 6:
                                    level = 6
                                    self.problem.find_naked_pairs(verbose)
                                else:
                                    if level < 7:
                                        level = 7
                                        self.problem.find_hidden_pairs(verbose)
                                    else:
                                        if level < 8:
                                            level = 8
                                            self.problem.find_pointed_triples(verbose)
                                        else:
                                            if level < 9:
                                                level = 9
                                                self.problem.find_naked_triples(verbose)
                                            else:
                                                if level < 10:
                                                    level = 10
                                                    self.problem.find_hidden_triples(verbose)
                                                else:
                                                    if x != 1:
                                                        level = 0
                                                        x += 1
                                                    else:
                                                        done = True

    def elim_solver_set(self,lst):
        for (i,j,k) in lst:
            self.board[i-1][j-1] = k                
            self.squares[i-1][j-1].set_value(k)
            self.squares[i-1][j-1].set_valid(True)
            self.problem.contents[(i,j)] = k
            self.draw()
            pygame.display.update()
            pygame.time.delay(10)

    def iterative_deepening_search(self,prob):

        for depth in range(sys.maxsize):
            result = self.depth_limited_search(prob,depth)

            if result != 'cutoff':
                return result

    def depth_limited_search(self,problem,limit=50):
        
        def recursive_dls(node, problem, limit):
            if problem.goal_test(node.state):
                for (i,j) in node.state.contents:
                    if (i,j) not in self.problem.contents.keys():
                        k = node.state.contents[(i,j)]
                        self.squares[i-1][j-1].set_value(k)
                        self.squares[i-1][j-1].set_valid(True)
                        self.draw()
                        pygame.display.update()
                        pygame.time.delay(10)
                return node
            elif limit == 0:
                return 'cutoff'
            else:
                cutoff_occurred = False
                for child in node.expand(problem):
                    pygame.display.update()
                    result = recursive_dls(child, problem, limit - 1)
                    if result == 'cutoff':
                        cutoff_occurred = True
                    elif result is not None:
                        return result

                return 'cutoff' if cutoff_occurred else None

    # Body of depth_limited_search:
        return recursive_dls(Node(problem.initial),problem,limit)

    def solve(self):
        self.elimination_solve()
        if not self.problem.goal_test():
            sudoku = SudokuProblemClassImpliedFill(self.problem.contents)
            result = self.iterative_deepening_search(sudoku)
            self.end(True)
        return True

    def generate_candidates(self):
        self.clear_temp_values()
        candidates = self.problem.get_candidates()
        for (i,j) in candidates:
            for k in candidates[(i,j)]:
                self.squares[i-1][j-1].set_temp_value(k,True)

#class for each of the games squares
class Square:

    def __init__(self,screen,width,height,x_margin,y_margin,row,col,value=None,selected=False,input_state=False,temp_input=None,valid=None):
        self.screen = screen
        self.width = width
        self.height = height
        self.x_margin = x_margin
        self.y_margin = y_margin
        self.row = row
        self.col = col
        self.value = value
        self.temp = [False for i in range(9)]
        self.selected = selected
        self.valid = valid
        self.input_state = input_state
        self.temp_input = temp_input

    #sets the value of the square
    def set_value(self,val):
        self.value = val

    #sets whether or not the square is selected
    def set_selected(self,val):
        self.selected = val

    def set_input_state(self,val):
        self.input_state = val

    #sets the temporary value of the square
    def set_temp_value(self,val,generate=False,val_set=None):
        if 1 <= val <= 9:
            if val_set is not None:
                self.temp[val-1] = val_set
            elif not generate and self.temp[val-1]:
                self.temp[val-1] = False
            else:
                self.temp[val-1] = True
        else:
            self.temp = [False for i in range(9)]

    def set_temp_input(self,val):
        self.temp_input = val

    
    #sets whether or not the square is valid
    def set_valid(self,val):
        self.valid = val

    #draws the square
    def draw(self):

        pos = (self.width * self.row, self.width * self.col)

        if self.valid is not None:
            if self.valid:
                pygame.draw.rect(self.screen,GREEN,(pos[0]+self.x_margin,pos[1]+self.y_margin,self.width+1,self.width+1))
                bg_rgb = GREEN
            else:
                pygame.draw.rect(self.screen,RED,(pos[0]+self.x_margin,pos[1]+self.y_margin,self.width+1,self.width+1))
                bg_rgb = RED
        elif self.input_state:
            pygame.draw.rect(self.screen,LIGHT_GREEN,(pos[0]+self.x_margin,pos[1]+self.y_margin,self.width+1,self.width+1))
            bg_rgb = LIGHT_GREEN
        elif self.selected:
            pygame.draw.rect(self.screen,YELLOW,(pos[0]+self.x_margin,pos[1]+self.y_margin,self.width+1,self.width+1))
            bg_rgb = YELLOW
        else:
            bg_rgb = WHITE

        if self.value is not None:
            text = create_text(str(self.value),(self.width * 0.54),BLACK,bg_rgb,False)
            self.screen.blit(text, (self.x_margin + pos[0] + ((self.width - text.get_width())/2), self.y_margin + pos[1] + ((self.width- text.get_height())/2)))   
        elif self.temp_input is not None:
            text = create_text(str(self.temp_input),(self.width * 0.54),GRAY,bg_rgb,False)
            self.screen.blit(text, (self.x_margin + pos[0] + ((self.width - text.get_width())/2), self.y_margin + pos[1] + ((self.width- text.get_height())/2)))   
        elif any(self.temp):
            for val in range(9):
                if self.temp[val]:
                    x = (1/6) + ((val % 3) * (1/3))
                    y = (1/6) + ((val // 3) * (1/3))
                    text = create_text(str(val+1),(self.width * (0.54/2)),GRAY,bg_rgb,False)
                    self.screen.blit(text, (self.x_margin + pos[0] + ((self.width - text.get_width())*x), self.y_margin + pos[1] + ((self.width- text.get_height())*y))) 


#function checks to see if a number is valid for a certain position
def is_valid(board, row, col, check):

    #checks to see if any spots in the same row have the same value
    for c in range(9):
        if board[row][c] == check and c != col:
            return False

    #checks to see if any spots in the same column have the same value
    for r in range(9):
        if (board[r][col] == check and r != row):
            return False

    #checks to see if any spots in the same box have the same value
    x = 3 * (col // 3)
    y = 3 * (row // 3)
    for r in range(3):
        for c in range(3):
            if board[y+r][x+c] == check and (y+r, x+c) != (row, col):
                return False

    return True

#generates a random sudoku board
def create_board():

    global COUNTER

    #blank is the desired amount of blank spots for the puzzle
    blank = 60
    #fail is the maximum amount of failures allowed
    fail = 5
    valid = False

    #creates a random valid sudoku solution
    while not valid:
        board = [[0 for i in range(9)] for x in range(9)]
        #calls on the solve function to fill in the grid and make sure that it is valid
        valid = solve_board(board, False)

    solution = copy.deepcopy(board)
    #removes spaces until wanted number of blank spots or maximum amount of failures is reached
    while (fail > 0 and blank > 0):
        COUNTER = 0
        test_board = copy.deepcopy(board)
        #randomly chooses a filled in spot to remove
        while True:
            rand = random.randint(0, 80)
            rem = (rand//9, rand%9)
            if test_board[rem[0]][rem[1]] != 0:
                test_board[rem[0]][rem[1]] = 0
                break
        #tests to see if board with the random spot removed has only one solution
        solve_board(test_board, True)

        #if it only has one solution it will keep the change or else it will count as a failure
        if COUNTER == 1:
            board[rem[0]][rem[1]] = 0
            blank -= 1
        else:
            fail -= 1

    #returns the created board
    return board,solution

#returns whether the board is solvable, solves the board, or gets the number of solutions
def solve_board(board, count):
    idx = (0, 0)
    while board[idx[0]][idx[1]] != 0: #loops through the board looking for an empty spot
        if idx[1] + 1 > 8:
            idx = (idx[0] + 1, 0)
        else:
            idx = (idx[0], idx[1] + 1)

        if idx == (9, 0): #if the board is filled, returns that the board is solvable
            return True

    num_list = [i for i in range(1, 10)]
    random.shuffle(num_list) #shuffles the domain to generate a random puzzle
    for check in num_list:
        if is_valid(board, idx[0], idx[1], check): #checks to see if a number is valid for that spot
            board[idx[0]][idx[1]] = check
            #recursively calls the function to see if the number leads to a solution
            if solve_board(board, count):
                if count:
                    #if looking for number of solutions, it will increment counter by 1
                    global COUNTER
                    COUNTER += 1
                else:
                    return True #returns that the board is solvable

            board[idx[0]][idx[1]] = 0

    return False #returns that the board is not solvable if it isn't

def read_sudoku_problem(filename):
    state = {}
    with open(filename, 'r') as file:
        row_id = 1
        for rows in file:
            rows = rows.strip()
            cont_list = [char for char in rows]
            for (col_id, row_contents) in enumerate(cont_list):
                row_contents = row_contents.strip()
                if '1' <= row_contents <= '9':
                    state[(row_id, col_id+1)] = int(row_contents)
            row_id = row_id + 1
        file.close()
    return state


# Search tree node class.
class Node:
    def __init__(self, state, parent=None, action=None, path_cost=0):
        self.state = state
        self.parent = parent
        self.action = action
        self.path_cost = path_cost
        if parent:
            self.depth = parent.depth + 1
        else: 
            self.depth = 0

    def __repr__(self):
        return "<Node {}>".format(self.state)

    def __lt__(self, node):
        return self.state < node.state

    def expand(self, problem):
        """List the nodes reachable in one step from this node."""
        return [self.child_node(problem, action)
                for action in problem.actions(self.state)]

    def child_node(self, problem, action):
        next_state = problem.result(self.state, action)
        next_node = Node(next_state, self, action,
                    problem.path_cost(self.path_cost, self.state,
                                      action, next_state))
        return next_node

    def solution(self):
        """Return the sequence of actions to go from the root to this node."""
        return [node.action for node in self.path()[1:]]

    def path(self):
        """Return a list of nodes forming the path from the root to this node."""
        node, path_back = self, []
        while node:
            path_back.append(node)
            node = node.parent
        return list(reversed(path_back))

    # We want for a queue of nodes in breadth_first_graph_search or
    # astar_search to have no duplicated states, so we treat nodes
    # with the same state as equal. [Problem: this may not be what you
    # want in other contexts.]

    def __eq__(self, other):
        return isinstance(other, Node) and self.state == other.state

# search algorithms

def depth_limited_search(problem, limit=50):
    """[Figure 3.17]"""

    def recursive_dls(node, problem, limit):
        if problem.goal_test(node.state):
            return node
        elif limit == 0:
            return 'cutoff'
        else:
            cutoff_occurred = False
            for child in node.expand(problem):
                result = recursive_dls(child, problem, limit - 1)
                if result == 'cutoff':
                    cutoff_occurred = True
                elif result is not None:
                    return result
            return 'cutoff' if cutoff_occurred else None

    # Body of depth_limited_search:
    return recursive_dls(Node(problem.initial), problem, limit)


def iterative_deepening_search(problem):
    """[Figure 3.18]"""
    for depth in range(sys.maxsize):
        result = depth_limited_search(problem, depth)
        if result != 'cutoff':
            return result

class SudokuBoard:
    
    def __init__(self, filled_cells_dict):
        self.contents = filled_cells_dict
        
    def pretty_print(self):
        state = self.contents
        blk_sep = '|' + '-'*9 + '+' + '-'*9 +  '+' +  '-'*9  + '|'
        print(blk_sep)
        for row_id in range(1,10): 
            # Iterate through each column
            row_str = '|'
            for col_id in range(1,10):
                # If row is not empty
                if (row_id, col_id) in state:
                    row_str = row_str + ' '+str(state[(row_id, col_id)]) + ' '
                else:
                    row_str = row_str + '   '
                if col_id % 3 == 0:
                    row_str = row_str + '|'
            print(row_str)
            if row_id %3 == 0:
                print(blk_sep)
    
    def get_numbers_for_row(self, j):
        assert(j >= 1 and j <= 9)
        state = self.contents
        row_nums = [state[(j,k)] 
                    for k in range(1,10)
                    if (j,k) in state.keys() ]
        return row_nums
    
    def get_numbers_for_col(self, j):
        assert(j >= 1 and j <= 9)
        state = self.contents
        col_nums = [state[(k,j)] 
                    for k in range(1,10)
                    if (k,j) in state.keys() ]
        return col_nums
    
    def get_numbers_for_block(self, blk_x, blk_y):
        assert( 1 <= blk_x <= 3)
        assert( 1 <= blk_y <= 3)
        state = self.contents
        row_nums = [ state[(j,k)] 
                    for j in range(blk_x*3-2, blk_x*3+1)
                    for k in range(blk_y*3-2, blk_y*3+1)
                    if (j,k) in state.keys() ]
        return row_nums
    
    def has_repeated_entries(self, lst_of_numbers):
        if len(lst_of_numbers) == len(set(lst_of_numbers)):
            return False
        else:
            return True

    def is_valid(self, verbose=False):

        for x in range(1,10):
            row_nums = self.get_numbers_for_row(x)
            col_nums = self.get_numbers_for_col(x)
            if self.has_repeated_entries(row_nums) or self.has_repeated_entries(col_nums):
                return False
            
        for x in range(1,4):
            for y in range(1,4):
                block_nums = self.get_numbers_for_block(x,y)
                if self.has_repeated_entries(block_nums):
                    return False
        
        return True

    def get_block_number(self, i, j):
        return ((i-1)//3 + 1, (j-1)//3 + 1)
    
    def get_possible_fills(self):
        state = self.contents
        lst_of_unfilled_cells = [ (i,j) 
                           for i in range(1, 10) 
                           for j in range(1, 10) 
                           if (i,j) not in state.keys()]
        valid_actions = []
        
        for (i,j) in lst_of_unfilled_cells:
            row_nums = self.get_numbers_for_row(i)
            col_nums = self.get_numbers_for_col(j)
            (x,y) = self.get_block_number(i,j)
            block_nums = self.get_numbers_for_block(x,y)
            invalid_nums = row_nums + col_nums + block_nums
            for k in range(1,10):
                valid = True
                z = 0
                while valid and z < len(invalid_nums):
                    if k == invalid_nums[z]:
                        valid = False
                    z += 1
                if valid:
                    valid_actions.append((i,j,k))       
        return valid_actions
    
    def fill_up(self, i, j, k):
        assert((i,j) not in self.contents.keys())
        new_state = self.contents.copy()
        assert (1 <= i <= 9 and 1 <= j <= 9 and 1 <= k <= 9)
        new_state[(i,j)] = k
        return SudokuBoard(new_state)
    
    def goal_test(self):
        state = self.contents
        if not self.is_valid():
            return False
        
        for i in range(1,10):
            for j in range(1,10):
                if (i,j) not in state.keys():
                    return False
        return True

    def __hash__(self):
        cells_lst = [(i,j,k) for ((i,j),k) in self.contents.items()]
        return hash(frozenset(cells_lst))

class SudokuBoardImpliedFill(SudokuBoard):
    
    def __init__(self, filled_cells_dict, do_implied_fills=True):
        super().__init__(filled_cells_dict)
        self.has_conflict = False
        self.pairs = []
        self.triples = []
        self.quads = []
        if do_implied_fills:
            self.do_all_implied_fills(False)
        
    def get_block_number(self, i, j):
        return ((i-1)//3 + 1, (j-1)//3 + 1)

    def get_candidates(self):
        candidates = {}
        state = self.contents
        implied_fill_list = []
        unfilled_cells = [(i,j) 
                          for i in range(1,10) 
                          for j in range(1,10) 
                          if (i,j) not in state.keys()]

        for (i,j) in unfilled_cells:
            candidates[(i,j)] = []
            row_nums = self.get_numbers_for_row(i)
            col_nums = self.get_numbers_for_col(j)
            (x,y) = self.get_block_number(i,j)
            block_nums = self.get_numbers_for_block(x,y)
            invalid_nums = row_nums + col_nums + block_nums
            for pair in self.pairs:
                if (i,j) != pair[3] and (i,j) != pair[4]:
                    if pair[0] == "row" and pair[1] == i:
                        invalid_nums += pair[2]
                    elif pair[0] == "col" and pair[1] == j:
                        invalid_nums += pair[2]
                    elif pair[0] == "block" and pair[1] == (x,y):
                        invalid_nums += pair[2]
            for triple in self.triples:
                if (i,j) != triple[3] and (i,j) != triple[4] and (i,j) != triple[5]:
                    if triple[0] == "row" and triple[1] == i:
                        invalid_nums += triple[2]
                    elif triple[0] == "col" and triple[1] == j:
                        invalid_nums += triple[2]
                    elif triple[0] == "block" and triple[1] == (x,y):
                        invalid_nums += triple[2]
            for quad in self.quads:
                if (i,j) != quad[3] and (i,j) != quad[4] and (i,j) != quad[5] and (i,j) != quad[6]:
                    if quad[0] == "row" and quad[1] == i:
                        invalid_nums += quad[2]
                    elif quad[0] == "col" and quad[1] == j:
                        invalid_nums += quad[2]
                    elif quad[0] == "block" and quad[1] == (x,y):
                        invalid_nums += quad[2]    
            for k in range(1,10):
                valid = True
                z = 0
                while valid and z < len(invalid_nums):
                    if k == invalid_nums[z]:
                        valid = False
                    z += 1
                if valid:
                    candidates[(i,j)].append(k)

        return candidates
         
    def find_implied_fills(self, verbose=False):
        
        state = self.contents
        implied_fill_list = []
        unfilled_cells = [(i,j) 
                          for i in range(1,10) 
                          for j in range(1,10) 
                          if (i,j) not in state.keys()]

        for (i,j) in unfilled_cells:
            valid_nums = []
            row_nums = self.get_numbers_for_row(i)
            col_nums = self.get_numbers_for_col(j)
            (x,y) = self.get_block_number(i,j)
            block_nums = self.get_numbers_for_block(x,y)
            invalid_nums = row_nums + col_nums + block_nums
            for pair in self.pairs:
                if (i,j) != pair[3] and (i,j) != pair[4]:
                    if pair[0] == "row" and pair[1] == i:
                        invalid_nums += pair[2]
                    elif pair[0] == "col" and pair[1] == j:
                        invalid_nums += pair[2]
                    elif pair[0] == "block" and pair[1] == (x,y):
                        invalid_nums += pair[2]
            for triple in self.triples:
                if (i,j) != triple[3] and (i,j) != triple[4] and (i,j) != triple[5]:
                    if triple[0] == "row" and triple[1] == i:
                        invalid_nums += triple[2]
                    elif triple[0] == "col" and triple[1] == j:
                        invalid_nums += triple[2]
                    elif triple[0] == "block" and triple[1] == (x,y):
                        invalid_nums += triple[2]
            for quad in self.quads:
                if (i,j) != quad[3] and (i,j) != quad[4] and (i,j) != quad[5] and (i,j) != quad[6]:
                    if quad[0] == "row" and quad[1] == i:
                        invalid_nums += quad[2]
                    elif quad[0] == "col" and quad[1] == j:
                        invalid_nums += quad[2]
                    elif quad[0] == "block" and quad[1] == (x,y):
                        invalid_nums += quad[2]    
            for k in range(1,10):
                valid = True
                z = 0
                while valid and z < len(invalid_nums):
                    if k == invalid_nums[z]:
                        valid = False
                    z += 1
                if valid:
                    valid_nums.append(k)
            if len(valid_nums) == 0:
                self.has_conflict = True
            elif len(valid_nums) == 1:
                implied_fill_list.append((i,j,valid_nums[0]))
        return implied_fill_list
    
    def do_all_implied_fills(self, verbose=False):
        done = False
        state = self.contents
        while (not done and not self.has_conflict):
            lst = self.find_implied_fills(verbose)
            if len(lst) == 0:
                done = True
            else:
                for (i,j,k) in lst:
                    state[(i,j)] = k
        return

    def is_valid(self, verbose=False):
        return not(self.has_conflict) and super().is_valid(verbose)
    
    def fill_up(self, i, j, k):

        new_state = self.contents.copy()
        assert (1 <= i <= 9 and 1 <= j <= 9 and 1 <= k <= 9)
        new_state[(i,j)] = k
        return SudokuBoardImpliedFill(new_state)
    
    def find_row_fills(self,verbose=False):
        state = self.contents
        implied_fill_list = []

        for i in range(1,10):
            row_nums = self.get_numbers_for_row(i)
            rnon_valid = []
        
            for pair in self.pairs:
                if pair[0] == "row" and pair[1] == i:
                    rnon_valid.append((pair[2],(pair[3],pair[4]),"pair"))
                    
            for triple in self.triples:
                if triple[0] == "row" and triple[1] == i:
                    rnon_valid.append((triple[2],(triple[3],triple[4],triple[5]),"triple"))
              
            for quad in self.quads:
                if quad[0] == "row" and quad[1] == i:
                    rnon_valid.append((quad[2],(quad[3],quad[4],quad[5],quad[6]),"quad"))
                    
            for k in range(1,10):
                filled = True
                valid_cells = []
                row_nums.append(k)
                if not self.has_repeated_entries(row_nums):
                    for j in range(1,10):
                        if self.pair_check(rnon_valid,(i,j),k) and (i,j) not in state.keys():
                            cnon_valid = []
                            
                            for pair in self.pairs:
                                if pair[0] == "col" and pair[1] == j:
                                    cnon_valid.append((pair[2],(pair[3],pair[4]),"pair"))

                            for triple in self.triples:
                                if triple[0] == "col" and triple[1] == j:
                                    cnon_valid.append((triple[2],(triple[3],triple[4],triple[5]),"triple"))
                                    
                            for quad in self.quads:
                                if quad[0] == "col" and quad[1] == j:
                                    cnon_valid.append((quad[2],(quad[3],quad[4],quad[5],quad[6]),"quad"))
                                    
                            if self.pair_check(cnon_valid,(i,j),k):
                                filled = False
                                col_nums = self.get_numbers_for_col(j)
                                col_nums.append(k)
                                if not self.has_repeated_entries(col_nums):
                                    bnon_valid = []
                                    (x,y) = self.get_block_number(i,j)
                                    
                                    for pair in self.pairs:
                                        if pair[0] == "block" and pair[1] == (x,y):
                                            bnon_valid.append((pair[2],(pair[3],pair[4]),"pair"))

                                    for triple in self.triples:
                                        if triple[0] == "block" and triple[1] == (x,y):
                                            bnon_valid.append((triple[2],(triple[3],triple[4],triple[5]),"triple"))
                                          
                                    for quad in self.quads:
                                        if quad[0] == "block" and quad[1] == (x,y):
                                            bnon_valid.append((quad[2],(quad[3],quad[4],quad[5],quad[6]),"quad"))
                                            
                                    if self.pair_check(bnon_valid,(i,j),k):
                                        block_nums = self.get_numbers_for_block(x,y)
                                        block_nums.append(k)
                                        if not self.has_repeated_entries(block_nums):
                                            valid_cells.append((i,j,k))
                                    
                if len(valid_cells) == 0 and not filled:
                    self.has_conflict = True
                elif len(valid_cells) == 1:
                    implied_fill_list.append(valid_cells[0])
                row_nums.pop()
        return implied_fill_list
    
    def find_col_fills(self,verbose=False):
        state = self.contents
        implied_fill_list = []
                
        for j in range(1,10):
            col_nums = self.get_numbers_for_col(j)
            cnon_valid = []
        
            for pair in self.pairs:
                if pair[0] == "col" and pair[1] == j:
                    cnon_valid.append((pair[2],(pair[3],pair[4]),"pair"))
                    
            for triple in self.triples:
                if triple[0] == "col" and triple[1] == j:
                    cnon_valid.append((triple[2],(triple[3],triple[4],triple[5]),"triple"))
              
            for quad in self.quads:
                if quad[0] == "col" and quad[1] == j:
                    cnon_valid.append((quad[2],(quad[3],quad[4],quad[5],quad[6]),"quad"))
                    
            for k in range(1,10):
                filled = True
                valid_cells = []
                col_nums.append(k)
                if not self.has_repeated_entries(col_nums):
                    for i in range(1,10):
                        if self.pair_check(cnon_valid,(i,j),k) and (i,j) not in state.keys():
                            rnon_valid = []
                            
                            for pair in self.pairs:
                                if pair[0] == "row" and pair[1] == i:
                                    rnon_valid.append((pair[2],(pair[3],pair[4]),"pair"))

                            for triple in self.triples:
                                if triple[0] == "row" and triple[1] == i:
                                    rnon_valid.append((triple[2],(triple[3],triple[4],triple[5]),"triple"))
                              
                            for quad in self.quads:
                                if quad[0] == "row" and quad[1] == i:
                                    rnon_valid.append((quad[2],(quad[3],quad[4],quad[5],quad[6]),"quad"))
                                    
                            if self.pair_check(rnon_valid,(i,j),k):
                                filled = False
                                row_nums = self.get_numbers_for_row(i)
                                row_nums.append(k)
                                if not self.has_repeated_entries(row_nums):
                                    bnon_valid = []
                                    (x,y) = self.get_block_number(i,j)
                                    
                                    for pair in self.pairs:
                                        if pair[0] == "block" and pair[1] == (x,y):
                                            bnon_valid.append((pair[2],(pair[3],pair[4]),"pair"))

                                    for triple in self.triples:
                                        if triple[0] == "block" and triple[1] == (x,y):
                                            bnon_valid.append((triple[2],(triple[3],triple[4],triple[5]),"triple"))
                        
                                    for quad in self.quads:
                                        if quad[0] == "block" and quad[1] == (x,y):
                                            bnon_valid.append((quad[2],(quad[3],quad[4],quad[5],quad[6]),"quad"))
                                            
                                    if self.pair_check(bnon_valid,(i,j),k):
                                        block_nums = self.get_numbers_for_block(x,y)
                                        block_nums.append(k)
                                        if not self.has_repeated_entries(block_nums):
                                            valid_cells.append((i,j,k))
                                    
                if len(valid_cells) == 0 and not filled:
                    self.has_conflict = True
                elif len(valid_cells) == 1:
                    implied_fill_list.append(valid_cells[0])
                col_nums.pop()
        return implied_fill_list       
    
    def find_block_fills(self,verbose=False):
        state = self.contents
        implied_fill_list = []
                
        for x in range(1,4):
            for y in range(1,4):
                block_nums = self.get_numbers_for_block(x,y)
                bnon_valid = []

                for pair in self.pairs:
                    if pair[0] == "block" and pair[1] == (x,y):
                        bnon_valid.append((pair[2],(pair[3],pair[4]),"pair"))
                
                for triple in self.triples:
                    if triple[0] == "block" and triple[1] == (x,y):
                        bnon_valid.append((triple[2],(triple[3],triple[4],triple[5]),"triple"))

                for quad in self.quads:
                    if quad[0] == "block" and quad[1] == (x,y):
                        bnon_valid.append((quad[2],(quad[3],quad[4],quad[5],quad[6]),"quad"))
                        
                for k in range(1,10):
                    filled = True
                    valid_cells = []
                    block_nums.append(k)
                    if not self.has_repeated_entries(block_nums):
                        for r in range(3):
                            i = (x*3)-r
                            row_nums = self.get_numbers_for_row(i)
                            row_nums.append(k)
                            if not self.has_repeated_entries(row_nums):
                                for c in range(3):
                                    j = (y*3)-c
                                    if self.pair_check(bnon_valid,(i,j),k) and (i,j) not in state.keys():
                                        non_valid = []
                                        for pair in self.pairs:
                                            if pair[0] == "row" and pair[1] == i:
                                                non_valid.append((pair[2],(pair[3],pair[4]),"pair"))
                                            elif pair[0] == "col" and pair[1] == j:
                                                non_valid.append((pair[2],(pair[3],pair[4]),"pair"))
                                        for triple in self.triples:
                                            if triple[0] == "row" and triple[1] == i:
                                                non_valid.append((triple[2],(triple[3],triple[4],triple[5]),"triple"))
                                            elif triple[0] == "col" and triple[1] == j:
                                                non_valid.append((triple[2],(triple[3],triple[4],triple[5]),"triple"))
                                        for quad in self.quads:
                                            if quad[0] == "row" and quad[1] == i:
                                                non_valid.append((quad[2],(quad[3],quad[4],quad[5],quad[6]),"quad"))
                                            elif quad[0] == "col" and quad[1] == j:
                                                non_valid.append((quad[2],(quad[3],quad[4],quad[5],quad[6]),"quad"))  
                                                
                                        if self.pair_check(non_valid,(i,j),k):
                                            filled = False
                                            col_nums = self.get_numbers_for_col(j)
                                            col_nums.append(k)
                                            if not self.has_repeated_entries(col_nums):
                                                valid_cells.append((i,j,k))
                    if len(valid_cells) == 0 and not filled:
                        self.has_conflict = True
                    elif len(valid_cells) == 1:
                        implied_fill_list.append(valid_cells[0])
                    block_nums.pop()
                
        return implied_fill_list
    
    def pair_check(self,lst,pos,k):
        for x in lst:
            if x[2] == "pair":
                if k == x[0][0] or k == x[0][1]:
                    if pos == x[1][0] or pos == x[1][1]:
                        return self.triple_check(lst,pos,k)
                    else:
                        return False      
        return self.triple_check(lst,pos,k)
    
    def triple_check(self,lst,pos,k):
        for x in lst:
            if x[2] == "triple":
                if k == x[0][0] or k == x[0][1] or k == x[0][2]:
                    if pos == x[1][0] or pos == x[1][1] or pos == x[1][2]:
                        return self.quad_check(lst,pos,k)
                    else:
                        return False
        return self.quad_check(lst,pos,k)
    
    def quad_check(self,lst,pos,k):
        for x in lst:
            if x[2] == "quad":
                if k == x[0][0] or k == x[0][1] or k == x[0][2] or k == x[0][3]:
                    if pos == x[1][0] or pos == x[1][1] or pos == x[1][2] or pos == x[1][3]:
                        return True
                    else:
                        return False
        return True
    
    def check_dupe(self, lst, elmt):
        for x in lst:
            if elmt == x:
                return True
        return False
    
    def find_naked_pairs(self,verbose=False):
        state = self.contents
        implied_fill_list = []
        unfilled_cells = [(i,j) 
                          for i in range(1,10) 
                          for j in range(1,10) 
                          if (i,j) not in state.keys()]
        has_pair = []
        np = []
        for (i,j) in unfilled_cells:
            p = []
            row_nums = self.get_numbers_for_row(i)
            col_nums = self.get_numbers_for_col(j)
            (x,y) = self.get_block_number(i,j)
            block_nums = self.get_numbers_for_block(x,y)
            invalid_nums = row_nums + col_nums + block_nums
            for pair in self.pairs:
                if (i,j) != pair[3] and (i,j) != pair[4]:
                    if pair[0] == "row" and pair[1] == i:
                        invalid_nums += pair[2]
                    elif pair[0] == "col" and pair[1] == j:
                        invalid_nums += pair[2]
                    elif pair[0] == "block" and pair[1] == (x,y):
                        invalid_nums += pair[2]
            for triple in self.triples:
                if (i,j) != triple[3] and (i,j) != triple[4] and (i,j) != triple[5]:
                    if triple[0] == "row" and triple[1] == i:
                        invalid_nums += triple[2]
                    elif triple[0] == "col" and triple[1] == j:
                        invalid_nums += triple[2]
                    elif triple[0] == "block" and triple[1] == (x,y):
                        invalid_nums += triple[2]
            for quad in self.quads:
                if (i,j) != quad[3] and (i,j) != quad[4] and (i,j) != quad[5] and (i,j) != quad[6]:
                    if quad[0] == "row" and quad[1] == i:
                        invalid_nums += quad[2]
                    elif quad[0] == "col" and quad[1] == j:
                        invalid_nums += quad[2]
                    elif quad[0] == "block" and quad[1] == (x,y):
                        invalid_nums += quad[2]   
                        
            for k in range(1,10):
                valid = True
                z = 0
                while valid and z < len(invalid_nums):
                    if k == invalid_nums[z]:
                        valid = False
                    z += 1
                if valid:
                    p.append(k)
                    
            if len(p) == 2:
                b = (x,y)
                for c in has_pair:
                    if p == c[0]:
                        if i == c[1]:
                            np.append(("row",i,p,(i,j),(c[1],c[2])))           
                        elif j == c[2]:
                            np.append(("col",j,p,(i,j),(c[1],c[2])))
                        elif b == c[3]:
                            np.append(("block",b,p,(i,j),(c[1],c[2])))
                has_pair.append((p,i,j,b))
        for pair in np:
            if not self.check_dupe(self.pairs,pair):
                self.pairs.append(pair)
        return
    
    def find_hidden_pairs(self,verbose=False):
        state = self.contents

        for i in range(1,10):
            row_nums = self.get_numbers_for_row(i)
            non_valid = []
        
            for pair in self.pairs:
                if pair[0] == "row" and pair[1] == i:
                    non_valid.append((pair[2],(pair[3],pair[4]),"pair"))
                    
            for triple in self.triples:
                if triple[0] == "row" and triple[1] == i:
                    non_valid.append((triple[2],(triple[3],triple[4],triple[5]),"triple"))
                    
            for quad in self.quads:
                if quad[0] == "row" and quad[1] == i:
                    non_valid.append((quad[2],(quad[3],quad[4],quad[5],quad[6]),"quad"))
                    
            p = []        
            for k in range(1,10):
                filled = True
                valid_cells = []
                row_nums.append(k)
                if not self.has_repeated_entries(row_nums):
                    for j in range(1,10):
                        if self.pair_check(non_valid,(i,j),k) and (i,j) not in state.keys():
                            filled = False
                            col_nums = self.get_numbers_for_col(j)
                            col_nums.append(k)
                            if not self.has_repeated_entries(col_nums):
                                (x,y) = self.get_block_number(i,j)
                                block_nums = self.get_numbers_for_block(x,y)
                                block_nums.append(k)
                                if not self.has_repeated_entries(block_nums):
                                    valid_cells.append(((i,j),k))
                row_nums.pop()
                if len(valid_cells) == 2:
                    app = True
                    for z in p:
                        if z[0][0] == valid_cells[0][0] and z[1][0] == valid_cells[1][0]:
                            pair = ("row",i,[z[0][1],valid_cells[1][1]],z[0][0],z[1][0])
                            if not self.check_dupe(self.pairs,pair):
                                self.pairs.append(pair)
                            p.remove(z)
                            app = False
                    if app:
                        p.append(valid_cells)
                        
        for j in range(1,10):
            col_nums = self.get_numbers_for_col(j)
            non_valid = []
        
            for pair in self.pairs:
                if pair[0] == "col" and pair[1] == j:
                    non_valid.append((pair[2],(pair[3],pair[4]),"pair"))
                    
            for triple in self.triples:
                if triple[0] == "col" and triple[1] == j:
                    non_valid.append((triple[2],(triple[3],triple[4],triple[5]),"triple"))
                    
            for quad in self.quads:
                if quad[0] == "col" and quad[1] == j:
                    non_valid.append((quad[2],(quad[3],quad[4],quad[5],quad[6]),"quad"))
            p = []        
            for k in range(1,10):
                filled = True
                valid_cells = []
                col_nums.append(k)
                if not self.has_repeated_entries(col_nums):
                    for i in range(1,10):
                        if self.pair_check(non_valid,(i,j),k) and (i,j) not in state.keys():
                            filled = False
                            row_nums = self.get_numbers_for_row(i)
                            row_nums.append(k)
                            if not self.has_repeated_entries(row_nums):
                                (x,y) = self.get_block_number(i,j)
                                block_nums = self.get_numbers_for_block(x,y)
                                block_nums.append(k)
                                if not self.has_repeated_entries(block_nums):
                                    valid_cells.append(((i,j),k))
                col_nums.pop()
                if len(valid_cells) == 2:
                    app = True
                    for z in p:
                        if z[0][0] == valid_cells[0][0] and z[1][0] == valid_cells[1][0]:
                            pair = ("col",j,[z[0][1],valid_cells[1][1]],z[0][0],z[1][0])
                            if not self.check_dupe(self.pairs,pair):
                                self.pairs.append(pair)
                            p.remove(z)
                            app = False
                    if app:
                        p.append(valid_cells)
                
                        
        for x in range(1,4):
            for y in range(1,4):
                block_nums = self.get_numbers_for_block(x,y)
                non_valid = []

                for pair in self.pairs:
                    if pair[0] == "block" and pair[1] == (x,y):
                        non_valid.append((pair[2],(pair[3],pair[4]),"pair"))

                for triple in self.triples:
                    if triple[0] == "block" and pair[1] == (x,y):
                        non_valid.append((triple[2],(triple[3],triple[4],triple[5]),"triple"))
                        
                for quad in self.quads:
                    if quad[0] == "block" and quad[1] == (x,y):
                        non_valid.append((quad[2],(quad[3],quad[4],quad[5],quad[6]),"quad"))
                        
                p = []        
                for k in range(1,10):
                    filled = True
                    valid_cells = []
                    block_nums.append(k)
                    if not self.has_repeated_entries(block_nums):
                        for r in range(3):
                            i = (x*3)-r
                            row_nums = self.get_numbers_for_row(i)
                            row_nums.append(k)
                            if not self.has_repeated_entries(row_nums):
                                for c in range(3):
                                    j = (y*3)-c
                                    if self.pair_check(non_valid,(i,j),k) and (i,j) not in state.keys():
                                        false = False
                                        col_nums = self.get_numbers_for_col(j)
                                        col_nums.append(k)
                                        if not self.has_repeated_entries(col_nums):
                                            valid_cells.append(((i,j),k))
                    block_nums.pop()
                    if len(valid_cells) == 2:
                        app = True
                        for z in p:
                            if z[0][0] == valid_cells[0][0] and z[1][0] == valid_cells[1][0]:
                                pair = ("block",(x,y),[z[0][1],valid_cells[1][1]],z[0][0],z[1][0])
                                if not self.check_dupe(self.pairs,pair):
                                    self.pairs.append(pair)
                                p.remove(z)
                                app = False
                        if app:
                            p.append(valid_cells)
        return
    
    def find_pointed_pairs(self,verbose=False):
        state = self.contents
        
        for x in range(1,4):
            for y in range(1,4):
                block_nums = self.get_numbers_for_block(x,y)
                bnon_valid = []
                
                for pair in self.pairs:
                    if pair[0] == "block" and pair[1] == (x,y):
                        bnon_valid.append((pair[2],(pair[3],pair[4]),"pair"))
                for triple in self.triples:
                    if triple[0] == "block" and triple[1] == (x,y):
                        bnon_valid.append((triple[2],(triple[3],triple[4],triple[5]),"triple"))
                for quad in self.quads:
                    if quad[0] == "block" and quad[1] == (x,y):
                        bnon_valid.append((quad[2],(quad[3],quad[4],quad[5],quad[6]),"quad"))
                        
                p = []
                for k in range(1,10):
                    filled = True
                    valid_cells = []
                    block_nums.append(k)
                    if not self.has_repeated_entries(block_nums):
                        for r in range(3):
                            i = (x*3)-r
                            row_nums = self.get_numbers_for_row(i)
                            row_nums.append(k)
                            if not self.has_repeated_entries(row_nums):
                                for c in range(3):
                                    j = (y*3)-c
                                    if self.pair_check(bnon_valid,(i,j),k) and (i,j) not in state.keys():
                                        non_valid = []   
                                        for pair in self.pairs:
                                            if pair[0] == "row" and pair[1] == i:
                                                non_valid.append((pair[2],(pair[3],pair[4]),"pair"))
                                            elif pair[0] == "col" and pair[1] == j:
                                                non_valid.append((pair[2],(pair[3],pair[4]),"pair"))
                                        for triple in self.triples:
                                            if triple[0] == "row" and triple[1] == i:
                                                non_valid.append((triple[2],(triple[3],triple[4],triple[5]),"triple"))
                                            elif triple[0] == "col" and triple[1] == j:
                                                non_valid.append((triple[2],(triple[3],triple[4],triple[5]),"triple"))
                                        for quad in self.quads:
                                            if quad[0] == "row" and quad[1] == i:
                                                non_valid.append((quad[2],(quad[3],quad[4],quad[5],quad[6]),"quad"))
                                            elif quad[0] == "col" and quad[1] == j:
                                                non_valid.append((quad[2],(quad[3],quad[4],quad[5],quad[6]),"quad"))
                                        if self.pair_check(non_valid,(i,j),k):
                                            false = False
                                            col_nums = self.get_numbers_for_col(j)
                                            col_nums.append(k)
                                            if not self.has_repeated_entries(col_nums):
                                                valid_cells.append(((i,j),k))
                    block_nums.pop()
                    if len(valid_cells) == 2:
                        if valid_cells[0][0][0] == valid_cells[1][0][0]:
                            pair = ("row",valid_cells[0][0][0],[k,-1],valid_cells[0][0],valid_cells[1][0])
                            if not self.check_dupe(self.pairs,pair):
                                    self.pairs.append(pair)
                        elif valid_cells[0][0][1] == valid_cells[1][0][1]:
                            pair = ("col",valid_cells[0][0][1],[k,-1],valid_cells[0][0],valid_cells[1][0])
                            if not self.check_dupe(self.pairs,pair):
                                    self.pairs.append(pair)
        return
    
    def find_naked_triples(self,verbose=False):
        state = self.contents
        
        #for row
        for i in range(1,10):
            unfilled_cells_for_row = [(i,j) 
                  for j in range(1,10) 
                  if (i,j) not in state.keys()]
            if len(unfilled_cells_for_row) > 3:
                row_nums = self.get_numbers_for_row(i)
                possible = []

                for k in range(1,10):
                    if not self.check_dupe(row_nums,k):
                        possible.append(k)
                        
                for z in range(len(possible)-2):
                    for z2 in range(z+1,len(possible)-1):
                        for z3 in range(z2+1,len(possible)):
                            trip = []
                            check = [possible[z],possible[z2],possible[z3]]
                            for (i,j) in unfilled_cells_for_row:
                                valid_nums = []
                                col_nums = self.get_numbers_for_col(j)
                                (x,y) = self.get_block_number(i,j)
                                block_nums = self.get_numbers_for_block(x,y)
                                invalid_nums = row_nums + col_nums + block_nums
                                for pair in self.pairs:
                                    if (i,j) != pair[3] and (i,j) != pair[4]:
                                        if pair[0] == "row" and pair[1] == i:
                                            invalid_nums += pair[2]
                                        elif pair[0] == "col" and pair[1] == j:
                                            invalid_nums += pair[2]
                                        elif pair[0] == "block" and pair[1] == (x,y):
                                            invalid_nums += pair[2]
                                for triple in self.triples:
                                    if (i,j) != triple[3] and (i,j) != triple[4] and (i,j) != triple[5]:
                                        if triple[0] == "row" and triple[1] == i:
                                            invalid_nums += triple[2]
                                        elif triple[0] == "col" and triple[1] == j:
                                            invalid_nums += triple[2]
                                        elif triple[0] == "block" and triple[1] == (x,y):
                                            invalid_nums += triple[2]
                                for quad in self.quads:
                                    if (i,j) != quad[3] and (i,j) != quad[4] and (i,j) != quad[5] and (i,j) != quad[6]:
                                        if quad[0] == "row" and quad[1] == i:
                                            invalid_nums += quad[2]
                                        elif quad[0] == "col" and quad[1] == j:
                                            invalid_nums += quad[2]
                                        elif quad[0] == "block" and quad[1] == (x,y):
                                            invalid_nums += quad[2]  
                                for k in possible:
                                    if not self.check_dupe(invalid_nums,k):
                                        valid_nums.append(k)
                                if len(valid_nums) <= 3:
                                    valid = True
                                    for num in valid_nums:
                                        if not self.check_dupe(check,num):
                                            valid = False
                                    if valid:
                                        trip.append((i,j))
                            if len(trip) == 3:
                                triple = ["row",i,check,trip[0],trip[1],trip[2]]
                                if not self.check_dupe(self.triples,triple):
                                    self.triples.append(triple)

        
        #for col
        for j in range(1,10):
            unfilled_cells_for_col = [(i,j) 
                  for i in range(1,10) 
                  if (i,j) not in state.keys()]
            if len(unfilled_cells_for_col) > 3:
                col_nums = self.get_numbers_for_col(j)
                possible = []

                for k in range(1,10):
                    if not self.check_dupe(col_nums,k):
                        possible.append(k)

                
                for z in range(len(possible)-2):
                    for z2 in range(z+1,len(possible)-1):
                        for z3 in range(z2+1,len(possible)):
                            trip = []
                            check = [possible[z],possible[z2],possible[z3]]
                            for (i,j) in unfilled_cells_for_col:
                                valid_nums = []
                                row_nums = self.get_numbers_for_row(i)
                                (x,y) = self.get_block_number(i,j)
                                block_nums = self.get_numbers_for_block(x,y)
                                invalid_nums = row_nums + col_nums + block_nums
                                for pair in self.pairs:
                                    if (i,j) != pair[3] and (i,j) != pair[4]:
                                        if pair[0] == "row" and pair[1] == i:
                                            invalid_nums += pair[2]
                                        elif pair[0] == "col" and pair[1] == j:
                                            invalid_nums += pair[2]
                                        elif pair[0] == "block" and pair[1] == (x,y):
                                            invalid_nums += pair[2]
                                for triple in self.triples:
                                    if (i,j) != triple[3] and (i,j) != triple[4] and (i,j) != triple[5]:
                                        if triple[0] == "row" and triple[1] == i:
                                            invalid_nums += triple[2]
                                        elif triple[0] == "col" and triple[1] == j:
                                            invalid_nums += triple[2]
                                        elif triple[0] == "block" and triple[1] == (x,y):
                                            invalid_nums += triple[2]
                                for quad in self.quads:
                                    if (i,j) != quad[3] and (i,j) != quad[4] and (i,j) != quad[5] and (i,j) != quad[6]:
                                        if quad[0] == "row" and quad[1] == i:
                                            invalid_nums += quad[2]
                                        elif quad[0] == "col" and quad[1] == j:
                                            invalid_nums += quad[2]
                                        elif quad[0] == "block" and quad[1] == (x,y):
                                            invalid_nums += quad[2]   
                                for k in possible:
                                    if not self.check_dupe(invalid_nums,k):
                                        valid_nums.append(k)
      
                                if len(valid_nums) <= 3:
                                    valid = True
                                    for num in valid_nums:
                                        if not self.check_dupe(check,num):
                                            valid = False
                                    if valid:
                                        trip.append((i,j))
                            if len(trip) == 3:
                                triple = ["col",j,check,trip[0],trip[1],trip[2]]
                                if not self.check_dupe(self.triples,triple):
                                    self.triples.append(triple)

        
        #for block
        for x in range(1,4):
            for y in range(1,4):
                rows = [(x*3)-r for r in range(3)]
                cols = [(y*3)-c for c in range(3)]
                unfilled_cells_for_block = [(i,j) 
                      for i in rows
                      for j in cols
                      if (i,j) not in state.keys()]
                if len(unfilled_cells_for_block) > 3:
                    block_nums = self.get_numbers_for_block(x,y)
                    possible = []

                    for k in range(1,10):
                        if not self.check_dupe(block_nums,k):
                            possible.append(k)

                    
                    for z in range(len(possible)-2):
                        for z2 in range(z+1,len(possible)-1):
                            for z3 in range(z2+1,len(possible)):
                                trip = []
                                check = [possible[z],possible[z2],possible[z3]]
                                for (i,j) in unfilled_cells_for_block:
                                    valid_nums = []
                                    row_nums = self.get_numbers_for_row(i)
                                    col_nums = self.get_numbers_for_col(j)
                                    invalid_nums = row_nums + col_nums + block_nums
                                    for pair in self.pairs:
                                        if (i,j) != pair[3] and (i,j) != pair[4]:
                                            if pair[0] == "row" and pair[1] == i:
                                                invalid_nums += pair[2]
                                            elif pair[0] == "col" and pair[1] == j:
                                                invalid_nums += pair[2]
                                            elif pair[0] == "block" and pair[1] == (x,y):
                                                invalid_nums += pair[2]
                                    for triple in self.triples:
                                        if (i,j) != triple[3] and (i,j) != triple[4] and (i,j) != triple[5]:
                                            if triple[0] == "row" and triple[1] == i:
                                                invalid_nums += triple[2]
                                            elif triple[0] == "col" and triple[1] == j:
                                                invalid_nums += triple[2]
                                            elif triple[0] == "block" and triple[1] == (x,y):
                                                invalid_nums += triple[2]
                                    for quad in self.quads:
                                        if (i,j) != quad[3] and (i,j) != quad[4] and (i,j) != quad[5] and (i,j) != quad[6]:
                                            if quad[0] == "row" and quad[1] == i:
                                                invalid_nums += quad[2]
                                            elif quad[0] == "col" and quad[1] == j:
                                                invalid_nums += quad[2]
                                            elif quad[0] == "block" and quad[1] == (x,y):
                                                invalid_nums += quad[2]    
                                    for k in possible:
                                        if not self.check_dupe(invalid_nums,k):
                                            valid_nums.append(k)

                                    if len(valid_nums) <= 3:
                                        valid = True
                                        for num in valid_nums:
                                            if not self.check_dupe(check,num):
                                                valid = False
                                        if valid:
                                            trip.append((i,j))
                                if len(trip) == 3:
                                    triple = ["block",(x,y),check,trip[0],trip[1],trip[2]]
                                    if not self.check_dupe(self.triples,triple):
                                        self.triples.append(triple)
        
        return
    
    def find_pointed_triples(self,verbose=False):
        state = self.contents
        
        
        for i in range(1,10):
            row_nums = self.get_numbers_for_row(i)
            non_valid = []
        
            for pair in self.pairs:
                if pair[0] == "row" and pair[1] == i:
                    non_valid.append((pair[2],(pair[3],pair[4]),"pair"))
                    
            for triple in self.triples:
                if triple[0] == "row" and triple[1] == i:
                    non_valid.append((triple[2],(triple[3],triple[4],triple[5]),"triple"))
                    
            for quad in self.quads:
                if quad[0] == "row" and quad[1] == i:
                    non_valid.append((quad[2],(quad[3],quad[4],quad[5],quad[6]),"quad"))
                    
            for k in range(1,10):
                filled = True
                valid_cells = []
                row_nums.append(k)
                if not self.has_repeated_entries(row_nums):
                    for j in range(1,10):
                        if self.pair_check(non_valid,(i,j),k) and (i,j) not in state.keys():
                            filled = False
                            col_nums = self.get_numbers_for_col(j)
                            col_nums.append(k)
                            if not self.has_repeated_entries(col_nums):
                                (x,y) = self.get_block_number(i,j)
                                block_nums = self.get_numbers_for_block(x,y)
                                block_nums.append(k)
                                if not self.has_repeated_entries(block_nums):
                                    valid_cells.append(((i,j),k))
                row_nums.pop()
                if len(valid_cells) == 3:
                    b1 = self.get_block_number(valid_cells[0][0][0],valid_cells[0][0][1])
                    b2 = self.get_block_number(valid_cells[1][0][0],valid_cells[1][0][1])
                    b3 = self.get_block_number(valid_cells[2][0][0],valid_cells[2][0][1])
                    if b1 == b2 == b3:
                        triple = ("block",b1,[k,-1,-1],valid_cells[0][0],valid_cells[1][0],valid_cells[2][0])
                        if not self.check_dupe(self.triples,triple):
                            self.triples.append(triple)
        
        for j in range(1,10):
            col_nums = self.get_numbers_for_col(j)
            non_valid = []
        
            for pair in self.pairs:
                if pair[0] == "col" and pair[1] == j:
                    non_valid.append((pair[2],(pair[3],pair[4]),"pair"))
                    
            for triple in self.triples:
                if triple[0] == "col" and triple[1] == j:
                    non_valid.append((triple[2],(triple[3],triple[4],triple[5]),"triple"))
                    
            for quad in self.quads:
                if quad[0] == "col" and quad[1] == j:
                    non_valid.append((quad[2],(quad[3],quad[4],quad[5],quad[6]),"quad"))
            p = []        
            for k in range(1,10):
                filled = True
                valid_cells = []
                col_nums.append(k)
                if not self.has_repeated_entries(col_nums):
                    for i in range(1,10):
                        if self.pair_check(non_valid,(i,j),k) and (i,j) not in state.keys():
                            filled = False
                            row_nums = self.get_numbers_for_row(i)
                            row_nums.append(k)
                            if not self.has_repeated_entries(row_nums):
                                (x,y) = self.get_block_number(i,j)
                                block_nums = self.get_numbers_for_block(x,y)
                                block_nums.append(k)
                                if not self.has_repeated_entries(block_nums):
                                    valid_cells.append(((i,j),k))
                col_nums.pop()
                if len(valid_cells) == 3:
                    b1 = self.get_block_number(valid_cells[0][0][0],valid_cells[0][0][1])
                    b2 = self.get_block_number(valid_cells[1][0][0],valid_cells[1][0][1])
                    b3 = self.get_block_number(valid_cells[2][0][0],valid_cells[2][0][1])
                    if b1 == b2 == b3:
                        triple = ("block",b1,[k,-1,-1],valid_cells[0][0],valid_cells[1][0],valid_cells[2][0])
                        if not self.check_dupe(self.triples,triple):
                            self.triples.append(triple)
        
        for x in range(1,4):
            for y in range(1,4):
                block_nums = self.get_numbers_for_block(x,y)
                bnon_valid = []
                
                for pair in self.pairs:
                    if pair[0] == "block" and pair[1] == (x,y):
                        bnon_valid.append((pair[2],(pair[3],pair[4]),"pair"))
                for triple in self.triples:
                    if triple[0] == "block" and triple[1] == (x,y):
                        bnon_valid.append((triple[2],(triple[3],triple[4],triple[5]),"triple"))
                for quad in self.quads:
                    if quad[0] == "block" and quad[1] == (x,y):
                        bnon_valid.append((quad[2],(quad[3],quad[4],quad[5],quad[6]),"quad"))
                        
                p = []
                for k in range(1,10):
                    filled = True
                    valid_cells = []
                    block_nums.append(k)
                    if not self.has_repeated_entries(block_nums):
                        for r in range(3):
                            i = (x*3)-r
                            row_nums = self.get_numbers_for_row(i)
                            row_nums.append(k)
                            if not self.has_repeated_entries(row_nums):
                                for c in range(3):
                                    j = (y*3)-c
                                    if self.pair_check(bnon_valid,(i,j),k) and (i,j) not in state.keys():
                                        non_valid = []   
                                        for pair in self.pairs:
                                            if pair[0] == "row" and pair[1] == i:
                                                non_valid.append((pair[2],(pair[3],pair[4]),"pair"))
                                            elif pair[0] == "col" and pair[1] == j:
                                                non_valid.append((pair[2],(pair[3],pair[4]),"pair"))
                                        for triple in self.triples:
                                            if triple[0] == "row" and triple[1] == i:
                                                non_valid.append((triple[2],(triple[3],triple[4],triple[5]),"triple"))
                                            elif triple[0] == "col" and triple[1] == j:
                                                non_valid.append((triple[2],(triple[3],triple[4],triple[5]),"triple"))
                                        for quad in self.quads:
                                            if quad[0] == "row" and quad[1] == i:
                                                non_valid.append((quad[2],(quad[3],quad[4],quad[5],quad[6]),"quad"))
                                            elif quad[0] == "col" and quad[1] == j:
                                                non_valid.append((quad[2],(quad[3],quad[4],quad[5],quad[6]),"quad"))
                                        if self.pair_check(non_valid,(i,j),k):
                                            false = False
                                            col_nums = self.get_numbers_for_col(j)
                                            col_nums.append(k)
                                            if not self.has_repeated_entries(col_nums):
                                                valid_cells.append(((i,j),k))
                    block_nums.pop()

                    if len(valid_cells) == 3:
                        if valid_cells[0][0][0] == valid_cells[1][0][0] == valid_cells[2][0][0]:
                            triple = ("row",valid_cells[0][0][0],[k,-1,-1],valid_cells[0][0],valid_cells[1][0],valid_cells[2][0])
                            if not self.check_dupe(self.triples,triple):
                                self.triples.append(triple)
                        elif valid_cells[0][0][1] == valid_cells[1][0][1] == valid_cells[2][0][1]:
                            triple = ("col",valid_cells[0][0][1],[k,-1,-1],valid_cells[0][0],valid_cells[1][0],valid_cells[2][0])
                            if not self.check_dupe(self.triples,triple):
                                self.triples.append(triple)
                
        return
    
    def find_hidden_triples(self,verbose=False):
        state = self.contents

        for i in range(1,10):
            row_nums = self.get_numbers_for_row(i)
            non_valid = []
        
            for pair in self.pairs:
                if pair[0] == "row" and pair[1] == i:
                    non_valid.append((pair[2],(pair[3],pair[4]),"pair"))
                    
            for triple in self.triples:
                if triple[0] == "row" and triple[1] == i:
                    non_valid.append((triple[2],(triple[3],triple[4],triple[5]),"triple"))
                    
            for quad in self.quads:
                if quad[0] == "row" and quad[1] == i:
                    non_valid.append((quad[2],(quad[3],quad[4],quad[5],quad[6]),"quad"))
                    
            p = []
            t = []
            for k in range(1,10):
                filled = True
                valid_cells = []
                row_nums.append(k)
                if not self.has_repeated_entries(row_nums):
                    for j in range(1,10):
                        if self.pair_check(non_valid,(i,j),k) and (i,j) not in state.keys():
                            filled = False
                            col_nums = self.get_numbers_for_col(j)
                            col_nums.append(k)
                            if not self.has_repeated_entries(col_nums):
                                (x,y) = self.get_block_number(i,j)
                                block_nums = self.get_numbers_for_block(x,y)
                                block_nums.append(k)
                                if not self.has_repeated_entries(block_nums):
                                    valid_cells.append(((i,j),k))
                row_nums.pop()
                if len(valid_cells) == 3:
                    app = True
                    for c in t:
                        if c[1] == valid_cells[0][0] and c[2] == valid_cells[1][0] and c[3] == valid_cells[2][0]:
                            triple = ("row",i,c[0].append(k),c[1],c[2],c[3])
                            if not self.check_dupe(self.triples,triple):
                                self.triples.append(triple)
                            t.remove(c)
                            app = False
                        else:
                            for z in p:
                                if z[0][0] == valid_cells[0][0] and z[1][0] == valid_cells[1][0] and z[2][0] == valid_cells[2][0]:
                                    pair = ([z[0][1],valid_cells[1][1]],z[0][0],z[1][0],z[2][0])
                                    t.append(pair)
                                    p.remove(z)
                                    app = False
                    if app:
                        p.append(valid_cells)
            
        for j in range(1,10):
            col_nums = self.get_numbers_for_col(j)
            non_valid = []
        
            for pair in self.pairs:
                if pair[0] == "col" and pair[1] == j:
                    non_valid.append((pair[2],(pair[3],pair[4]),"pair"))
                    
            for triple in self.triples:
                if triple[0] == "col" and triple[1] == j:
                    non_valid.append((triple[2],(triple[3],triple[4],triple[5]),"triple"))
                    
            for quad in self.quads:
                if quad[0] == "col" and quad[1] == j:
                    non_valid.append((quad[2],(quad[3],quad[4],quad[5],quad[6]),"quad"))
            p = []        
            for k in range(1,10):
                filled = True
                valid_cells = []
                col_nums.append(k)
                if not self.has_repeated_entries(col_nums):
                    for i in range(1,10):
                        if self.pair_check(non_valid,(i,j),k) and (i,j) not in state.keys():
                            filled = False
                            row_nums = self.get_numbers_for_row(i)
                            row_nums.append(k)
                            if not self.has_repeated_entries(row_nums):
                                (x,y) = self.get_block_number(i,j)
                                block_nums = self.get_numbers_for_block(x,y)
                                block_nums.append(k)
                                if not self.has_repeated_entries(block_nums):
                                    valid_cells.append(((i,j),k))
                col_nums.pop()
                if len(valid_cells) == 3:
                    app = True
                    for c in t:
                        if c[1] == valid_cells[0][0] and c[2] == valid_cells[1][0] and c[3] == valid_cells[2][0]:
                            triple = ("col",j,c[0].append(k),c[1],c[2],c[3])
                            if not self.check_dupe(self.triples,triple):
                                self.triples.append(triple)
                            t.remove(c)
                            app = False
                        else:
                            for z in p:
                                if z[0][0] == valid_cells[0][0] and z[1][0] == valid_cells[1][0] and z[2][0] == valid_cells[2][0]:
                                    pair = ([z[0][1],valid_cells[1][1]],z[0][0],z[1][0],z[2][0])
                                    t.append(pair)
                                    p.remove(z)
                                    app = False
                    if app:
                        p.append(valid_cells)
                
                        
        for x in range(1,4):
            for y in range(1,4):
                block_nums = self.get_numbers_for_block(x,y)
                non_valid = []

                for pair in self.pairs:
                    if pair[0] == "block" and pair[1] == (x,y):
                        non_valid.append((pair[2],(pair[3],pair[4]),"pair"))

                for triple in self.triples:
                    if triple[0] == "block" and pair[1] == (x,y):
                        non_valid.append((triple[2],(triple[3],triple[4],triple[5]),"triple"))
                        
                for quad in self.quads:
                    if quad[0] == "block" and quad[1] == (x,y):
                        non_valid.append((quad[2],(quad[3],quad[4],quad[5],quad[6]),"quad"))
                        
                p = []        
                for k in range(1,10):
                    filled = True
                    valid_cells = []
                    block_nums.append(k)
                    if not self.has_repeated_entries(block_nums):
                        for r in range(3):
                            i = (x*3)-r
                            row_nums = self.get_numbers_for_row(i)
                            row_nums.append(k)
                            if not self.has_repeated_entries(row_nums):
                                for c in range(3):
                                    j = (y*3)-c
                                    if self.pair_check(non_valid,(i,j),k) and (i,j) not in state.keys():
                                        false = False
                                        col_nums = self.get_numbers_for_col(j)
                                        col_nums.append(k)
                                        if not self.has_repeated_entries(col_nums):
                                            valid_cells.append(((i,j),k))
                    block_nums.pop()
                    if len(valid_cells) == 3:
                        app = True
                        for c in t:
                            if c[1] == valid_cells[0][0] and c[2] == valid_cells[1][0] and c[3] == valid_cells[2][0]:
                                triple = ("block",(x,y),c[0].append(k),c[1],c[2],c[3])
                                if not self.check_dupe(self.triples,triple):
                                    self.triples.append(triple)
                                t.remove(c)
                                app = False
                            else:
                                for z in p:
                                    if z[0][0] == valid_cells[0][0] and z[1][0] == valid_cells[1][0] and z[2][0] == valid_cells[2][0]:
                                        pair = ([z[0][1],valid_cells[1][1]],z[0][0],z[1][0],z[2][0])
                                        t.append(pair)
                                        p.remove(z)
                                        app = False
                        if app:
                            p.append(valid_cells)
        return
    
    def solve(self,verbose=False):
        done = False
        state = self.contents
        level = 0
        x = 0
        while (not done and not self.has_conflict):
            lst = self.find_implied_fills(verbose)
            if len(lst) != 0:
                level = 1
                x = 0
                for (i,j,k) in lst:
                    state[(i,j)] = k
            else:
                lst = self.find_row_fills(verbose)
                if len(lst) != 0:
                    level = 2
                    x = 0
                    for (i,j,k) in lst:
                        state[(i,j)] = k
                else:
                    lst = self.find_col_fills(verbose)
                    if len(lst) != 0:
                        level = 3
                        x = 0
                        for (i,j,k) in lst:
                            state[(i,j)] = k
                    else:
                        lst = self.find_block_fills(verbose)
                        if len(lst) != 0:
                            level = 4
                            x = 0
                            for (i,j,k) in lst:
                                state[(i,j)] = k
                        else:
                            if level < 5:
                                level = 5
                                self.find_pointed_pairs(verbose)
                            else:
                                if level < 6:
                                    level = 6
                                    self.find_naked_pairs(verbose)
                                else:
                                    if level < 7:
                                        level = 7
                                        self.find_hidden_pairs(verbose)
                                    else:
                                        if level < 8:
                                            level = 8
                                            self.find_pointed_triples(verbose)                  
                                        else:
                                            if level < 9:
                                                level = 9
                                                self.find_naked_triples(verbose)
                                            else:
                                                if level < 10:
                                                    level = 10
                                                    self.find_hidden_triples(verbose)
                                                else:
                                                    if x != 1:
                                                        level = 0
                                                        x += 1
                                                    else:
                                                        done = True                                                 
        return
        

class SudokuProblemClassImpliedFill(Problem):
    
    # Note:
    # A state of the sudoku problem is given by a dictionary
    #  mapping coordinate (i,j) -> k
    #  in other words, at (i, j) we have the number k where
    #   1 <= i <= 9, 1 <= j <= 9 and k is between 1 and 9
    # If a coordinate has no number in it, it is simply omitted from 
    # the dictionary.
    
    def __init__(self, filled_initial_cells):
        """ Constructor: filled_initial_cells is a dictionary
                        specifying which cells are already filled.
        """
        super().__init__(SudokuBoardImpliedFill(filled_initial_cells))
  
    def actions(self, state):
        # Define all possible actions from the given state.    
        if not state.is_valid():
            return [] # no actions
        return state.get_possible_fills()
    
    
    def result(self, state, action):
        (i,j,k) = action
        return state.fill_up(i,j,k)
    
    def goal_test(self, state):
        return state.goal_test()

    def path_cost(self, c, state1, action, state2):
        return 1

#class for the player
class Player:
    def __init__(self, streak=0):
        self.streak = streak

    def update_streak(self,win):
        if win:
            self.streak += 1
        else:
            self.streak = 0

#creates a surface with text on it to add to GUI
def create_text(text, font_size, text_rgb, bg_rgb,bold=True):
    font = pygame.freetype.SysFont("Times New Roman", font_size, bold=bold)
    surface,_ = font.render(text=text, fgcolor=text_rgb, bgcolor=bg_rgb)
    return surface.convert_alpha()

#class for the all of the buttons in the program
class UIElement(Sprite):

    def __init__(self, center_pos, text, font_size, text_rgb, bg_rgb, hl_action = True, hl_rgb=None, action=None):

        super().__init__()

        self.mouse_over = False
        default_img = create_text(text, font_size, text_rgb, bg_rgb)

        if hl_action:
            if hl_rgb is not None:
                highlighted_img = create_text('* ' + text + ' *', font_size, hl_rgb, bg_rgb)
            else:
                highlighted_img = create_text('* ' + text + ' *', font_size, text_rgb, bg_rgb)
        else:
            highlighted_img = default_img

        self.images = [default_img, highlighted_img]
        self.rects = [default_img.get_rect(center = center_pos),
                     highlighted_img.get_rect(center = center_pos)]

        self.action = action
    
    @property
    def image(self):
        return self.images[1] if self.mouse_over else self.images[0]

    @property
    def rect(self):
        return self.rects[1] if self.mouse_over else self.rects[0]

    #sets whether or not the mouse is over the button and returns action if it is clicked
    def update(self, mouse_pos, mouse_up):
        if self.rect.collidepoint(mouse_pos):
            self.mouse_over = True
            if mouse_up:
                return self.action
        else:
            self.mouse_over = False
    
    #draws the button
    def draw(self, surface):
        surface.blit(self.image, self.rect)


#displays the title screen of the program
def title_screen(screen):

    title = UIElement(
        center_pos = (screen.get_width()/2,screen.get_height()/4),
        font_size = screen.get_height()/6,
        bg_rgb = MAPLE,
        text_rgb = WHITE,
        hl_action = False,
        text = "Sudoku"
    )

    play_btn = UIElement(
        center_pos = (screen.get_width()/2,(screen.get_height()/1.5)-(screen.get_height()/12)),
        font_size = screen.get_height()/20,
        bg_rgb = MAPLE,
        text_rgb = WHITE,
        text = "New Game",
        action = GameState.GAME_BOARD
    )

    solver_btn = UIElement(
        center_pos = (screen.get_width()/2,(screen.get_height()/1.5)),
        font_size = screen.get_height()/20,
        bg_rgb = MAPLE,
        text_rgb = WHITE,
        text = "Solver",
        action = GameState.SOLVER
    )

    quit_btn = UIElement(
        center_pos = (screen.get_width()/2,(screen.get_height()/1.5)+(screen.get_height()/12)),
        font_size = screen.get_height()/20,
        bg_rgb = MAPLE,
        text_rgb = WHITE,
        text = "Quit",
        action = GameState.QUIT
    )

    buttons = [title, play_btn, quit_btn, solver_btn]

    while True:
        mouse_up = False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                mouse_up = True

        screen.fill(MAPLE)
        
        for btn in buttons:
            btn_action = btn.update(pygame.mouse.get_pos(),mouse_up)
            if btn_action is not None:
                return btn_action
            btn.draw(screen)

        pygame.display.flip()

#displays the solver screen for the program
def solver_screen(screen):
    game = Game(screen)
    board_action = BoardAction.NOTHING
    step = game.height/7
    start = ((((screen.get_width()-(game.width + game.x_margin))/2) + (game.width + game.x_margin)),(2*game.y_margin)+(1.5*step))

    solve_btn = UIElement(
        center_pos = (start[0],start[1] + (3*step)),
        font_size = game.height/15,
        bg_rgb = LIGHT_GRAY,
        text_rgb = BLACK,
        text = "Solve",
        action = BoardAction.SOLVE
    )
    clear_btn = UIElement(
        center_pos = (start[0],start[1] + (step)),
        font_size = game.height/15,
        bg_rgb = LIGHT_GRAY,
        text_rgb = BLACK,
        text = "Clear",
        action = BoardAction.RESET
    )
    home_btn = UIElement(
        center_pos = start,
        font_size =game.height/15,
        bg_rgb = LIGHT_GRAY,
        text_rgb = BLACK,
        text = "Home",
        action = BoardAction.HOME
    )
    generate_btn = UIElement(
        center_pos = (start[0],start[1] + (2*step)),
        font_size = game.height/20,
        bg_rgb = LIGHT_GRAY,
        text_rgb = BLACK,
        text = "Generate Candidates",
        action = BoardAction.GENERATE
    )

    btns = [home_btn, clear_btn, solve_btn,generate_btn]

    control_panel = Controls(screen,buttons=btns)

    while True:

        if board_action == BoardAction.SOLVE:
            if not game.check_invalid():
                game.select_square(None)
                game.solve()
            board_action = BoardAction.NOTHING
        elif board_action == BoardAction.RESET:
            game.clear_board()
            board_action = BoardAction.NOTHING
        elif board_action == BoardAction.HOME:
            return GameState.TITLE
        elif board_action == BoardAction.GENERATE:
            game.generate_candidates()
            board_action = BoardAction.NOTHING

        mouse_up = False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                pos = pygame.mouse.get_pos()
                if pos[0] >= game.x_margin and pos[0] <= game.x_margin + game.width and pos[1] >=game.y_margin and pos[1] <= game.y_margin + game.height:
                    game.select_square(game.click(pos))
                else:
                    mouse_up = True
            elif event.type == pygame.KEYDOWN and game.selected is not None:        
                row = game.selected[0]
                col = game.selected[1]
                value = -1

                if event.key == pygame.K_1:
                    value = 1
                elif event.key == pygame.K_2:
                    value = 2
                elif event.key == pygame.K_3:
                    value = 3
                elif event.key == pygame.K_4:
                    value = 4
                elif event.key == pygame.K_5:
                    value = 5
                elif event.key == pygame.K_6:
                    value = 6
                elif event.key == pygame.K_7:
                    value = 7
                elif event.key == pygame.K_8:
                    value = 8
                elif event.key == pygame.K_9:
                    value = 9
                elif event.key == pygame.K_DELETE or event.key == pygame.K_SPACE or event.key == pygame.K_0:
                    value = None

                if value is None or value > 0:
                    if value is None or is_valid(game.board, row, col, value):
                        game.squares[row][col].set_valid(None)
                    else:
                        game.squares[row][col].set_valid(False)

                    game.set_square(value)

        screen.fill(MAPLE)
        
        board_action = control_panel.draw(mouse_up)  
        game.draw()
        pygame.display.flip()

#displays the game board screen for the program
def game_board_screen(screen):

    player = Player()
    game = Game(screen)
    game.new_board()
    control_panel = Controls(screen)
    board_action = BoardAction.NOTHING
    step = game.height/10
    start = ((((screen.get_width()-(game.width + game.x_margin))/2) + (game.width + game.x_margin)),(2*game.y_margin)+(2*step))
    
    solve_btn = UIElement(
        center_pos = (start[0],start[1] + (3*step)),
        font_size = game.height/20,
        bg_rgb = LIGHT_GRAY,
        text_rgb = BLACK,
        text = "Solve",
        action = BoardAction.SOLVE
    )
    reset_btn = UIElement(
        center_pos = (start[0],start[1] + (step)),
        font_size = game.height/20,
        bg_rgb = LIGHT_GRAY,
        text_rgb = BLACK,
        text = "Reset",
        action = BoardAction.RESET
    )
    home_btn = UIElement(
        center_pos = start,
        font_size = game.height/20,
        bg_rgb = LIGHT_GRAY,
        text_rgb = BLACK,
        text = "Home",
        action = BoardAction.HOME
    )
    newgame_btn = UIElement(
        center_pos = (start[0],start[1] + (2*step)),
        font_size = game.height/20,
        bg_rgb = LIGHT_GRAY,
        text_rgb = BLACK,
        text = "New Game",
        action = BoardAction.NEWGAME
    )
    generate_btn = UIElement(
        center_pos = (start[0],start[1] + (2*step)),
        font_size = game.height/20,
        bg_rgb = LIGHT_GRAY,
        text_rgb = BLACK,
        text = "Generate Candidates",
        action = BoardAction.GENERATE
    )

    buttons = [home_btn,solve_btn,reset_btn,generate_btn]
    control_panel.update_buttons(buttons)
    control_panel.start_timer()
    while True:

        if board_action == BoardAction.SOLVE:
            control_panel.pause_timer()
            game.select_square(None)
            game.solve()
            game.end(True)
            buttons = [newgame_btn,home_btn,reset_btn]
            control_panel.update_buttons(buttons)
            board_action = BoardAction.NOTHING
        elif board_action == BoardAction.RESET:
            game.end(False)
            game.reset_board()
            buttons = [home_btn,solve_btn,reset_btn,generate_btn]
            control_panel.update_buttons(buttons)
            control_panel.start_timer()
            board_action = BoardAction.NOTHING
        elif board_action == BoardAction.NEWGAME:
            game.new_board()
            buttons = [home_btn,solve_btn,reset_btn,generate_btn]
            control_panel.update_buttons(buttons)
            game.end(False)
            control_panel.start_timer()
            board_action = BoardAction.NOTHING
        elif board_action == BoardAction.HOME:
            control_panel.pause_timer()
            if player.streak > 0:
                print("Streak Ended at " + str(player.streak) + " wins")
            return GameState.TITLE
        elif board_action == BoardAction.GENERATE:
            game.generate_candidates()
            board_action = BoardAction.NOTHING

        mouse_up = False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                if player.streak > 0:
                    print("Streak Ended at " + str(player.streak) + " wins")
                pygame.quit()
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                pos = pygame.mouse.get_pos()
                if not game.ended and pos[0] >= game.x_margin and pos[0] <= game.x_margin + game.width and pos[1] >=game.y_margin and pos[1] <= game.y_margin + game.height:
                    game.select_square(game.click(pos))
                else:
                    mouse_up = True
            elif not game.ended and event.type == pygame.KEYDOWN and game.selected is not None:        
                row = game.selected[0]
                col = game.selected[1]
                if game.squares[row][col].value is None:
                    value = -1
                    cur = game.squares[row][col].temp_input

                    if event.key == pygame.K_1:
                        value = 1
                    elif event.key == pygame.K_2:
                        value = 2
                    elif event.key == pygame.K_3:
                        value = 3
                    elif event.key == pygame.K_4:
                        value = 4
                    elif event.key == pygame.K_5:
                        value = 5
                    elif event.key == pygame.K_6:
                        value = 6
                    elif event.key == pygame.K_7:
                        value = 7
                    elif event.key == pygame.K_8:
                        value = 8
                    elif event.key == pygame.K_9:
                        value = 9
                    elif event.key == pygame.K_DELETE or event.key == pygame.K_SPACE or event.key == pygame.K_0:
                        value = 0
                    elif event.key == pygame.K_RETURN and cur != 0:
                        value = 0
                        game.set_square(cur)
                        if game.solution_board[row][col] == cur:
                            game.squares[row][col].set_valid(True)
                            if game.check_win():
                                control_panel.pause_timer()
                                buttons = [newgame_btn,home_btn,reset_btn]
                                control_panel.update_buttons(buttons)
                                player.update_streak(True)
                                print("Current Streak = " + str(player.streak))
                        else:
                            game.lose()
                            game.end(True)
                            control_panel.pause_timer()
                            if player.streak > 0:
                                print("Streak Ended at " + str(player.streak) + " wins")
                            player.update_streak(False)
                            buttons = [newgame_btn,home_btn,reset_btn]
                            control_panel.update_buttons(buttons)
                    
                    if value >= 0:
                        if not game.squares[row][col].input_state:
                            game.squares[row][col].set_temp_value(value)
                        else:
                            game.squares[row][col].set_temp_input(value)

        screen.fill(MAPLE)
        
        board_action = control_panel.draw(mouse_up)
            
        game.draw()
        pygame.display.flip()

#runs the program
def main():

    #initializes pygame
    pygame.init()

    #creates screen with width of 800 and height of 600
    screen = pygame.display.set_mode((1400, 800))
    pygame.display.set_caption("Sudoku")

    #comment out two line below if you did not download the icon into the program directory
    icon = pygame.image.load('sudoku_icon.png')
    pygame.display.set_icon(icon)

    game_state = GameState.TITLE

    while True:
        if game_state == GameState.TITLE:
            game_state = title_screen(screen)
        elif game_state == GameState.GAME_BOARD:
            game_state = game_board_screen(screen)
        elif game_state == GameState.SOLVER:
            game_state = solver_screen(screen)
        elif game_state == GameState.QUIT:
            pygame.quit()
            return

main()
