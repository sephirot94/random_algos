import heapq
import math
from typing import List

from leet.playing import TestSubway, TestOscarInterview
from leet.google import Google, Oscar
from collections import defaultdict, Counter, deque
from heap.heapster import HeapProblems
from dynamic import dynamic
from trees.graph import RobotPath, OscarGraph


def predictPartyVictory(senate: str) -> str:
    radiant = deque()
    dire = deque()
    n = len(senate)
    for i, party in enumerate(senate):
        if party == 'R':
            radiant.append(i)
        else:
            dire.append(i)

    while radiant and dire:
        r = radiant.popleft()
        d = dire.popleft()

        if r < d:
            radiant.append(r + n)
        else:
            dire.append(d + n)

    return "Radiant" if radiant else "Dire"

def findPoisonedDuration(timeSeries: list[int], duration: int) -> int:
    st = set()
    for i in timeSeries:
        for j in range(duration):
            st.add(i+j)
    return len(st)

PLAYER_1 = 1
PLAYER_2 = 2

class Connect4:

    def __init__(self, rows=6, cols=7):
        self.rows = rows
        self.cols = cols
        self.board = [[0 for _ in range(cols)] for _ in range(rows)]
        self.winner = None
        self.current_player = PLAYER_1

    def play(self, col: int) -> str:

        if col < 0 or col > self.cols:
            return f"Invalid column, please select a column between 0-{self.cols-1}"
        
        if self.winner:
            return f"Game Over! Player {self.winer} has already won"
        
        for row in reversed(range(self.rows)):
            if self.board[row][col] == 0:
                self.board[row][col] = self.current_player
                if self.check_winner(row, col):
                    self.winner = self.current_player
                    return f"Game Over! Player {self.winner} wins!"
                self.current_player = PLAYER_2 if self.current_player == PLAYER_1 else PLAYER_1
                return f"Player {self.current_player}'s turn"
        return "Column already full"
    
    def check_winner(self, row: int, col: int) -> bool:

        def count_tokens(delta_row: int, delta_col: int) -> int:
            r, c = row + delta_row, col + delta_col
            count = 0
            while 0 <= r < self.rows and 0 <= c < self.cols and self.board[r][c] == self.current_player:
                count += 1
                r += delta_row
                c += delta_col
            return count
        
        directions = [
            (0, 1),
            (1, 0),
            (1, 1),
            (1, -1),
        ]

        for dr, dc in directions:
            total = 1 + count_tokens(dr, dc) + count_tokens(-dr, -dc)
            if total >= 4:
                return True
        
        return False
    
    def print(self):
        for row in self.board:
            print(' '.join(str(cell) for cell in row))
        print("\n")
            

if __name__ == "__main__":
    game = Connect4()

    while not game.winner:
        game.print()
        try:
            col = int(input(f"Player {game.current_player}, please choose a column between 0-{game.cols-1}:"))
            turn = game.play(col)
            print(turn)
        except ValueError:
            print(f"Please insert a valid number between o-{game.cols-1}")
    game.print()