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

class CustomHeap:
    def __init__(self):
        self.array = []
        self.size = 0
        self.pos = []

    def newMinHeapNode(self, v, dist):
        minHeapNode = [v, dist]
        return minHeapNode

    def swapMinHeapNode(self, a, b):
        """
        A utility function to swap two nodes of min heap. Needed for min heapify
        """
        t = self.array[a]
        self.array[a] = self.array[b]
        self.array[b] = t

    def minHeapify(self, idx):
        """
        A standard function to heapify at given idx This function also updates position of nodes when they are swapped.
        Position is needed for decreaseKey()
        """
        smallest = idx
        left = 2 * idx + 1
        right = 2 * idx + 2

        if left < self.size and self.array[left][1] < \
                self.array[smallest][1]:
            smallest = left

        if right < self.size and self.array[right][1] < \
                self.array[smallest][1]:
            smallest = right

        # The nodes to be swapped in min heap
        # if idx is not smallest
        if smallest != idx:
            # Swap positions
            self.pos[self.array[smallest][0]] = idx
            self.pos[self.array[idx][0]] = smallest

            # Swap nodes
            self.swapMinHeapNode(smallest, idx)

            self.minHeapify(smallest)

    def extractMin(self):
        """
        Standard function to extract minimum node from heap
        """
        # Return NULL wif heap is empty
        if self.isEmpty():
            return

        # Store the root node
        root = self.array[0]

        # Replace root node with last node
        lastNode = self.array[self.size - 1]
        self.array[0] = lastNode

        # Update position of last node
        self.pos[lastNode[0]] = 0
        self.pos[root[0]] = self.size - 1

        # Reduce heap size and heapify root
        self.size -= 1
        self.minHeapify(0)

        return root

    def isEmpty(self):
        return self.size == 0

    def decreaseKey(self, v, dist):

        # Get the index of v in  heap array
        i = self.pos[v]

        # Get the node and update its dist value
        self.array[i][1] = dist

        # Travel up while the complete tree is not
        # hepified. This is a O(Logn) loop
        while i > 0 and self.array[i][1] < \
                self.array[(i - 1) / 2][1]:
            # Swap this node with its parent
            self.pos[self.array[i][0]] = (i - 1) / 2
            self.pos[self.array[(i - 1) / 2][0]] = i
            self.swapMinHeapNode(i, (i - 1) / 2)

            # move to parent index
            i = (i - 1) / 2;

    # A utility function to check if a given vertex
    # 'v' is in min heap or not
    def isInMinHeap(self, v):
        if self.pos[v] < self.size:
            return True
        return False

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




