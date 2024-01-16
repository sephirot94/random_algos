from leet.playing import Play
from leet.google import Google, Oscar
from collections import defaultdict, Counter, deque
from dynamic import dynamic
from trees.graph import RobotPath
import re


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

if __name__ == '__main__':
    houses = [1,2,3,1]
    graph = [
        [0,0,1,0],
        [0,1,0,0],
        [0,1,1,0],
        [1,0,0,0]
    ]
    robot = RobotPath(graph)
    print(robot.min_number_moves())
