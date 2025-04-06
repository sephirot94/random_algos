from collections import defaultdict
from datetime import datetime, timedelta
from typing import List
import math


class Play:

    def __init__(self):
        pass

    # @staticmethod
    # def is_balanced_parenthesis(word: str):
    #     """
    #     Given a string, returns true if it has balanced parenthesis
    #     :param word: input string
    #     :return: booleand indicating if string has valid parenthesis
    #     """
    #     stack = []
    #     for letter in word:
    #         if letter is "(" or letter is "[" or letter is "{":
    #             stack.append(letter)
    #         if letter is ")" or letter is "]" or letter is "}":
    #             prev = stack.pop()
    #             if letter is ")" and prev != "(":
    #                 return False
    #             if letter is "]" and prev != "]":
    #                 return False
    #             if letter is "}" and prev != "}":
    #                 return False
    #     return True

    def smallest_positive_integer_not_present(self, numbers: list[int]) -> int:
        """
        Returns smallest positive integer not present in input
        :param numbers: array containing positive integers
        :return: smallest positive integer not in array
        """

        # Create empty set to store numbers in array
        storage_set = set()
        for number in numbers:
            storage_set.add(number)
        size = len(numbers)
        for i in range(1, size + 1, 1):
            if i not in storage_set:
                return i
        return 1


    def max_product_subarray(self, arr: list[int]) -> int:
        """Given an integer array nums, find a subarray that has the largest product, and return the product"""
        if not arr:
            return 0
        ans = max_prod = min_prod = arr[0]

        for num in arr[1:]:
            max_temp = max_prod * num
            min_temp = min_prod * num

            max_prod = max(max_temp, min_temp, num)
            min_prod = min(max_temp, min_temp, num)

            ans = max(max_prod, ans)

        return ans



    def snake_can_pass(self, matrix: list[list[str]]) -> (list[int], list[int]):
        """
        Given a board with '0' and '+' determine if a snake can pass through an entire row or column. The snake can
        pass through any cell where there is a '0', and is blocked by the '+'. The board IS NOT necessarily a square
        Returns two collections in a tuple, the first element representing the rows and the second element representing the
        columns.
        """
        if not matrix:
            return [], []
        row_map = defaultdict(bool)
        column_map = defaultdict(bool)
        can_pass_rows, can_pass_columns = [], []
        for row_idx, row in enumerate(matrix):  # O(r*c)
            for col_idx, _ in enumerate(row):
                if matrix[row_idx][col_idx] == "+":
                    row_map[row_idx] = True
                    column_map[col_idx] = True

        for i in range(len(matrix)):  # O(r)
            if not column_map[i]:
                can_pass_columns.append(i)
            if not row_map[i]:
                can_pass_rows.append(i)

        return can_pass_rows, can_pass_columns


class RobotSimulator:
    """
    A robot on an infinite XY-plane starts at point (0, 0) facing north.
    The robot can receive a sequence of these three possible types of commands:
    -2: Turn left 90 degrees.
    -1: Turn right 90 degrees.
    1 <= k <= 9: Move forward k units, one unit at a time.
    Some of the grid squares are obstacles. The ith obstacle is at grid point obstacles[i] = (xi, yi).
    If the robot runs into an obstacle, then it will instead stay in its current location and move on to the next command.
    """

    def __init__(self, obstacles: List[List[int]]):
        self.obstacles = {(x[0], x[1]) for x in obstacles}
        self.orientation = "north"
        self.coordinate = (0,0)
        self.map = {
            "north": {
                "left": "west",
                "right": "east"
            },
            "south": {
                "left": "east",
                "right": "west"
            },
            "east": {
                "left": "north",
                "right": "south"
            },
            "west": {
                "left": "south",
                "right": "north"
            }
        }

    def turn_right(self):
        self.orientation = self.map[self.orientation]["right"]

    def turn_left(self):
        self.orientation = self.map[self.orientation]["left"]

    def validate(self, point: tuple[int, int]) -> bool:
        """Validate if next point in movement is an obstacle"""
        x, y = point
        if self.orientation == "north":
            y += 1
        elif self.orientation == "south":
            y -= 1
        elif self.orientation == "east":
            x += 1
        elif self.orientation == "west":
            x -= 1
        point = (x,y)
        return point not in self.obstacles

    def move(self, k: int):
        """Move robot k units in the direction it is facing"""
        x, y = self.coordinate
        for _ in range(k):
            if not self.validate((x,y)):
                break
            if self.orientation == "north":
                y += 1
            elif self.orientation == "south":
                y -= 1
            elif self.orientation == "east":
                x += 1
            elif self.orientation == "west":
                x -= 1
        self.coordinate = (x,y)

def robot_sim(commands: List[int], obstacles: List[List[int]]) -> int:
    """Return the maximum Euclidean distance that the robot ever gets from the origin squared"""
    robot = RobotSimulator(obstacles)
    m_distance = 0
    for command in commands:
        if command == -1:
            robot.turn_right()
        elif command == -2:
            robot.turn_left()
        else:
            robot.move(command)
        x, y = robot.coordinate
        m_distance = max(m_distance, math.sqrt(x**2 + y**2))
    return int(m_distance**2)

class Subway:
    """Handle subway checkin problem"""

    def __init__(self):
        self.tracker = defaultdict(lambda: None)

    def __str__(self):
        for name, time in self.tracker.items():
            print(f'{name} checked in at {time}')

    def check_in(self, name: str, time: datetime):
        if self.validate(name, time):
            self.tracker[name] = time
            return "Accepted"
        return "Rejected"

    def validate(self, name: str, time: datetime) -> bool:
        return not self.tracker[name] or self.tracker[name] + timedelta(minutes=5) <= time

class TestSubway:

    def __init__(self):
        self.subway = Subway()

    def test_subway(self):

        assert self.subway.check_in("John", datetime(2020, 1, 1, 1, 0)) == "Accepted"
        assert self.subway.check_in("Sally", datetime(2020, 1, 1, 1, 1)) == "Accepted"
        assert self.subway.check_in("Ed", datetime(2020, 1, 1, 1, 2)) == "Accepted"
        assert self.subway.check_in("John", datetime(2020, 1, 1, 1, 3)) == "Rejected"
        assert self.subway.check_in("Sally", datetime(2020, 1, 1, 1, 6)) == "Accepted"
        assert self.subway.check_in("Ed", datetime(2020, 1, 1, 1, 5)) == "Rejected"

class Visit:
    def __init__(self, cost: int=0):
        self.cost = cost

class Member:
    def __init__(self, visits: list[Visit]):
        self.visits = visits
        self.total_cost = sum(visit.cost for visit in self.visits)

class OscarInterview:

    def top_costly_members_threshold(self, members: List[Member], threshold: int) -> List[Member]:
        """Returns the smallest amount of members in which the sum of costs are at least the threshold, given as a percentage"""
        if not members:
            return []
        members = sorted(members, key=lambda member: member.total_cost, reverse=True)
        total_cost = sum(member.total_cost for member in members)
        curr_total_cost = 0
        res = []
        for member in members:
            curr_total_cost += member.total_cost
            res.append(member)
            if curr_total_cost * 100 / total_cost >= threshold:
                break
        return res

class TestOscarInterview:

    def __init__(self):
        visit1 = Visit(cost=100)
        visit2 = Visit(cost=200)
        visit3 = Visit(cost=300)
        visit4 = Visit(cost=400)
        visit5 = Visit(cost=500)

        member1 = Member(visits=[visit1, visit3, visit5])
        member2 = Member(visits=[visit2, visit4, visit5])
        member3 = Member(visits=[visit5])

        self.members = [member1, member2, member3]
        self.bl = OscarInterview()

    def test_costly_members_threshold(self):

        assert len(self.bl.top_costly_members_threshold(self.members, 90)) == 3
        assert len(self.bl.top_costly_members_threshold(self.members, 50)) == 2
        assert len(self.bl.top_costly_members_threshold(self.members, 25)) == 1

