import heapq
from collections import deque


def smallestSubset(s):
    # Use a breakpoint in the code line below to debug your script.
    heapq.heapify(s)
    res = deque([])
    for i in range(1, len(s)):
        hq = heapq.nlargest(i, s)
        if sum(hq) > sum(heapq.nsmallest(len(s)-i, s)):
            while hq:
                res.append(hq.pop())
            return list(res)
    return s