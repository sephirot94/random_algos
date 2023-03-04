class BackTracker:

    def solve(self, values: list, safe_up_to, size) -> list:
        """
        Finds a solution to a backtracking problem.
        :param values: a sequence of values to try, in order. For a map coloring problem, this may be a list of colors,
         such as ['red', 'green', 'yellow', 'purple']
        :param safe_up_to: a function with two arguments, solution and position, that returns whether the values
         assigned to slots 0..pos in the solution list, satisfy the problem constraints.
        :param size: the total number of “slots” you are trying to fill
        Return the solution as a list of values.
        """
        solution = [None] * size

        def extend_solution(position):
            for value in values:
                solution[position] = value
                if safe_up_to(solution, position):
                    if position >= size - 1 or extend_solution(position + 1):
                        return solution
            return None

        return extend_solution(0)

    # Call using
    def no_adjacencies(self, string, up_to_index):
        # See if the sequence filled from indices 0 to up_to_index, inclusive, is
        # free of any adjancent substrings. We'll have to try all subsequences of
        # length 1, 2, 3, up to half the length of the string. Return False as soon
        # as we find an equal adjacent pair.
        length = up_to_index + 1
        for j in range(1, length // 2 + 1):
            if (string[length - 2 * j:length - j] == string[length - j:length]):
                return False
        return True

class KnightProblem:

    def __init__(self):
        self.size = 8

    def isValid(self, i: int, j: int, board: list) -> bool:
        """
        Utility to check if i, j are valid indexes for N*N chessboard
        :param i: i index
        :param j: j index
        :param board: 2D array representing chessboard
        :return: boolean indicating if position is valid
        """
        if 0 <= i < self.size and 0 <=  j < self.size and board[i][j] == -1:
            return True
        return False

    def printSolution(self, board: list):
        """
        Utility function to print Chessboard matrix
        :param board: 2d Array indicating board
        """
        for i in range(self.size):
            for j in range(self.size):
                print(board[i][j], end=' ')
            print()

    def solver_util(self, board: list, curr_i: int, curr_j: int, move_i: list, move_j: list, pos: int) -> bool:
        """
        Recursive utility function to solve knight tour problem
        :param board: 2D array containing board
        :param curr_i: current i position
        :param curr_j: current j position
        :param move_i: list of i positions
        :param move_j: list of j positions
        :param pos: result in board
        :return: boolean indicating recursive solution
        """
        # Base case
        if pos == self.size**2:
            return True

        # Try all next moves from the current coordinate i,j
        for i in range(self.size):
            new_i = curr_i + move_i[i]
            new_j = curr_j + move_j[i]
            if self.isValid(new_i, new_j, board):
                board[new_i][new_j] = pos
                if self.solver_util(board, new_i, new_j, move_i, move_j, pos+1):
                    return True

                # Backtrack
                board[new_i][new_j] = -1
        return False

    def knight_problem(self) -> bool:
        """
        Given a N*N board with the Knight placed on the first block of an empty board. Moving according to the rules of
        chess knight must visit each square exactly once. Print the order of each the cell in which they are visited.
        It returns false if no complete tour is possible, otherwise return true and prints the tour.
        :return boolean indicating if knight problem can be solved
        """
        # O(8N^2) time complexity
        # Use backtracking
        board = [[-1] * self.size] * self.size
        # move_i and move_j define next move
        move_i = [2,1,-1,-2,-2,-1,1,2]
        move_j = [1,2,2,1,-1,-2,-2,-1]
        # Position knight on first block
        board[0][0] = 0
        # step counter for position
        pos = 1

        # check if solution exists
        if not self.solver_util(board, 0, 0, move_i, move_j, pos):
            print("Solution not found")
            return False
        else:
            self.printSolution(board)
            return True

class RatInMaze:

    def __init__(self, size: int):
        self.size = size

    def printsolution(self, sol):
        """
        Utility to print solution
        """
        for i in sol:
            for j in i:
                print(str(j) + " ", end="")

    def isValid(self, maze, i, j) -> bool:
        """
        Utility to check if position is valid
        """
        if 0 <= i < self.size and 0 <= j < self.size and maze[i][j] == 1:
            return True
        return False

    def solve(self, maze) -> bool:
        """
        Solve the Rat in the Maze problem using backtracking.
        A Maze is given as N*N binary matrix of blocks where source block is the upper left most block i.e., maze[0][0]
        and destination block is lower rightmost block i.e., maze[N-1][N-1]. A rat starts from source and has to reach
        the destination. The rat can move only in two directions: forward and down. In the maze matrix, 0 means the
        block is a dead end and 1 means the block can be used in the path from source to destination.
        Note that this is a simple version of the typical Maze problem. For example, a more complex version can be that
        the rat can move in 4 directions and a more complex version can be with a limited number of moves
        """
        sol = [[0]*self.size]*self.size
        if not self.recursive_helper(maze, 0, 0, sol):
            print("Solution not exists")
            return False

        self.printsolution(sol)
        return True

    def recursive_helper(self, maze, i, j, sol) -> bool:
        """
        Recursive utility function to solve problem
        """
        # if i,j is goal, return True
        if i == self.size - 1 and j == self.size - 1 and maze[i][j] == 1:
            sol[i][j] = 1
            return True

        # Check if maze[i][j] is valid
        if self.isValid(maze, i, j):
            # check if the current block is already part of solution path
            if sol[i][j] == 1:
                return False

            # Mark i,j as port of solution path
            sol[i][j] = 1

            # Move forward in x axis
            if self.recursive_helper(maze, i+1, j, sol):
                return True

            # If moving horizontally didnt work, then move vertically
            if self.recursive_helper(maze, i, j+1, sol):
                return True

            # If moving vertically didnt work, then move back horizontally
            if self.recursive_helper(maze, i-1, j, sol):
                return True

            # If moving backwards in x axis didnt work, then move backwards vertically
            if self.recursive_helper(maze, i, j-1, sol):
                return True

            # If nothing worked, then backtrack and unmark i, j as part of solution
            sol[i][j] = 0
            return False

