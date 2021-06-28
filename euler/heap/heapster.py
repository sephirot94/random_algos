import heapq
from collections import deque


class KthLargest(object):
    def __init__(self, k: int, nums: list):
        heapq.heapify(nums)
        self.heap = nums
        self.k = k
        while len(self.heap) > self.k:
            heapq.heappop(self.heap)

    def add(self, val: int) -> int:
        heapq.heappush(self.heap, val)
        if len(self.heap) > self.k:
            heapq.heappop(self.heap)
        return self.heap[0]

    @staticmethod
    def KthLargestElementInArray(nums: list, k: int) -> int:
        """
        Given array of integers, return Kth largest element
        :param nums: array of integers to be searched
        :param k: Kth largest element searched
        :return: Kth largest element
        """
        heapq.heapify(nums)
        hq = heapq.nlargest(k, nums)
        return hq.pop()


def smallest_subset(s: list) -> list:
    """
    Given an array of integer, return the smallest subset where sum(subset) > sum(rest_of_nums)
    :param s: array of integers
    :return: smallest subset where sum is greater than rest elements sum
    """
    heapq.heapify(s)
    res = deque([])
    for i in range(1, len(s)):
        hq = heapq.nlargest(i, s)
        if sum(hq) > sum(heapq.nsmallest(len(s)-i, s)):
            while hq:
                res.append(hq.pop())
            return list(res)
    return s



