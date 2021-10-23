# Sudoku

---
### Description: 
---

This Sudoku program has two modes, a game mode and a solver mode. The game mode creates a random, valid sudoku board using a backtracking algorithm while the solver mode has a blank board for the user to input values into. Both modes have a solving feature that first uses various elimination and solving techniques (described below) before using an iterative deepening search algorithm to brute force the rest of the puzzle. When playing in game mode, click on a square once to set temporary values and click twice, press desired number, and press enter to lock in an answer (square will be highlighted light green when in input mode). Solver uses the solving techniques in order displayed below, looping until it loops through twice without a change.

---
### Requirements:
---

- Python3
- Pygame
- If you did not download the sudoku_icon.png file into the project directory, please comment out or delete lines 2594 and 2595 (depicted below) from sudoku.py
>icon = pygame.image.load('sudoku_icon.png')
>pygame.display.set_icon(icon)

---
### Solving Techniques:
---

##### Implied Fill :
>The implied fill technique finds all the valid numbers for a cell, and if there is only one valid number, it is entered in.

##### Row Fill:
>The row fill technique finds all the valid cells for a number in a row, and if there is only one valid cell, it is entered in.

##### Column Fill:
>The column fill technique finds all the valid cells for a number in a column, and if there is only one valid cell, it is entered in.

##### Block Fill:
>The column fill technique finds all the valid cells for a number in a block, and if there is only one valid cell, it is entered in.

##### Pointing Pair:
>If a number is valid for only two cells in a block, then it must be the solution for one of these two cells. If these two cells belong to the same row or column, then this candidate can not be the solution in any other cell of the same row or column.

##### Naked Pair:
>If two cells in a row, column, or block contain exactly the same two candidates, then one of these candidates is the solution for one of these cells and the other candidate is the solution for the other cell.

##### Hidden Pair:
>If two candidates can be found in only the same two cells of a row, column, or block, then one of these candidates is the solution for one of these cells and the other candidate is the solution for the other cell.

##### Pointing Triple:
>If a candidate is present in only three cells of a block, then it must be the solution for one of these three cells. If these three cells belong to the same row or column, then this candidate can not be the solution in any other cell of the same row or column.

##### Naked Triple:
>If three cells in a row, column or block contain exactly the same three candidates or only subsets of these three candidates, then one of these candidates is the solution for the first of these cells, a second one is the solution for the second of these cells and the last candidate is the solution for the third cell.

##### Hidden Triple:
>If three candidates can be found in only the same three cells of a row, column or block, then one of these candidates is the solution for the first of these cells, a second one is the solution for the second cell and the last candidate is the solution for the third cell.


---