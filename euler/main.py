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
        self.current_player = PLAYER_1
        self.winner = None
        self.board = [[0 for _ in range(cols)] for _ in range(rows)]
    
    def play(self, col: int) -> str:
        if col < 0 or col > self.cols:
            return "Invalid column"

        if self.winner:
            return f"Game Over! Player {self.winner} has already won"
        
        for row in reversed(range(self.rows)):
            if self.board[row][col] == 0:
                self.board[row][col] = self.current_player
                if self.check_winner(row, col):
                    self.winner = self.current_player
                    return f"Player {self.winner} wins!"
                self.current_player = PLAYER_2 if self.current_player == PLAYER_1 else PLAYER_1
                return f"Player {self.current_player}'s turn"
        return "Column already full, try a different column"

    def check_winner(self, row: int, col: int) -> bool:

        def count_line(delta_row: int, delta_column: int) -> int:
            """Counts how many cells player has already filled in any direction"""
            r, c = row + delta_row, col + delta_column
            count = 0
            while 0 <= r < self.rows and 0 <= c < self.cols and self.board[r][c] == self.current_player:
                count += 1
                r += delta_row
                c += delta_column
            return count
        
        directions = [
            (0,1),
            (1,0),
            (1,1),
            (1,-1),
        ]

        for dr, dc in directions:
            total = 1 + count_line(dr, dc) + count_line(-dr, -dc)
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
            col = int(input(f"Player {game.current_player}, choose a column between 0-{game.cols-1}:"))
            result = game.play(col)
            print(result)
        except ValueError:
            print(f"Invalid column, please choose a number between 0-{game.cols-1}")
    game.print()