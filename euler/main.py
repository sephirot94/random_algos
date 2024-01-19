import heapq
import math
from typing import List

from leet.playing import TestSubway
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

if __name__ == '__main__':
    ts = TestSubway()
    print(ts.test_subway())
    phrase = "this word is a word this is word this word"
    print(HeapProblems.top_k_words(phrase, 2))
    commands = [4,-1,4,-2,4]
    obstacles = [[2,4]]
    print(robot_sim(commands, obstacles))

