import heapq
import math
from collections import deque, defaultdict


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

def topKFrequentElementInArray(nums: list, k: int) -> list:
    """
    Given an array of nums and integer k, return k most frequent items in descending order of frequency
    :param nums: array of integers
    :param k: integer representing Kth most frequent elements
    :return: Kth most frequent elements
    """
    # Use hashmaps and priority queue (heap)
    d = defaultdict(lambda: 0)
    # Create map with occurences
    for num in nums:
        d[num] += 1
    heap = []
    # Create heap
    for value, count in d.items():
        heapq.heappush(heap, (count, value))
        if len(heap) > k:
            heapq.heappop(heap)

    resp = []
    while heap:
        resp.append(heapq.heappop(heap)[1])
    return resp

class KClosestPointToOrigin:

    def kClosestPointToOrigin(self, points: list, k: int) -> list:
        """
        Given an array of points (x,y) representing coordinates in 2D, return the K points that are closest to origin
        """
        d = {tuple(point): self.getDistance(point) for point in points}
        res = []
        for p, _ in heapq.nsmallest(k, d.items(), key=lambda p: p[1]):
            res.append(list(p))
        return res

    def getDistance(self, point: list) -> float:
        return math.sqrt((point[0])**2 + (point[1])**2)

# def topKFrequentWord(words: list, k: int) -> list:
#     """
#     Given a non-empty list of words, return the k most frequent elements.
#     """
#     d = defaultdict(lambda: 0)
#     for word in words:
#         d[word] += 1
#     heap = []
#     res = []
#
#     for word, count in d.items():
#         heapq.heappush(heap, (count, word))
#         if len(heap) > k:
#             heapq.heappop(heap)
#
#     while heap:
#         count, word = heapq.heappop(heap)
#         res.append(word) if d[word] < d[res[-1]] else
#     return res

def lengthOfLongestSubStringwithoutRepeating(s: str) -> int:
    """
    Given a string, return the size of the longest substring without repeating characters
    """
    # Base Case
    if not s:
        return 0

    curr = ""
    max_length = 0

    for letter in s:
        # check if letter exists in substring
        if letter in curr:
            # check for a new max length
            if len(curr) > max_length:
                max_length = len(curr)

            # Since we found a repeated character, we need to calculate a new starting point for the current string.
            # Remove everything up until the first occurence from the current string as soon as the second is found
            new = curr.find(letter) + 1
            curr = curr[new:]

        curr += letter

    # Check final letter in case is non repeating
    if len(curr) > max_length:
        max_length = len(curr)

    return max_length




