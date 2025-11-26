import heapq
import math
from collections import deque, defaultdict


class HeapNode:
    def __init__(self, val: int, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


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


class RunningMedianCalculator:
    def running_median(self, arr):
        min_heap = []  # Represents the larger half of the elements
        max_heap = []  # Represents the smaller half of the elements

        for num in arr:
            self._add_number(num, min_heap, max_heap)
            yield self._calculate_median(min_heap, max_heap)

    def _add_number(self, num, min_heap, max_heap):
        if not max_heap or num < -max_heap[0]:
            heapq.heappush(max_heap, -num)
        else:
            heapq.heappush(min_heap, num)

        # Balance the heaps
        if len(max_heap) > len(min_heap) + 1:
            heapq.heappush(min_heap, -heapq.heappop(max_heap))
        elif len(min_heap) > len(max_heap):
            heapq.heappush(max_heap, -heapq.heappop(min_heap))

    def _calculate_median(self, min_heap, max_heap):
        if len(min_heap) == len(max_heap):
            return (min_heap[0] - max_heap[0]) / 2.0
        else:
            return -max_heap[0]


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
        return math.sqrt((point[0]) ** 2 + (point[1]) ** 2)


class MinHeap:
    """
    Heaps always have the following condition:
    Given an array, if node= arr[i] then left child of node is at  arr[2*i+1] and right child at arr[2*i+2]
    """

    def __init__(self, arr: list = None):
        if not arr:
            self.heap = []
            return
        start_idx = len(arr) // 2 - 1  # Last non-leaf node
        for i in range(start_idx, 0, -1):  # Reverse level order traversal of heap
            self.heapify(arr, i)

        self.heap = arr

    def swap(self, node1, node2):
        """
        Swaps two nodes in place
        """
        self.heap[node1], self.heap[node2] = self.heap[node2], self.heap[node1]

    def heapify(self, arr: list, idx: int):
        """
        Given an array, generate a min-heap. In a min-heap, smallest element is always at root.
        :param arr: list to heapify
        :param idx: index to heapify
        :return: heapify list
        """
        if not arr:
            return None

        smallest = idx
        left = 2 * idx + 1
        right = 2 * idx + 2

        if left < len(arr) and arr[left] < arr[smallest]:  # If left is smaller than smallest
            smallest = left

        if right < len(arr) and arr[right] < arr[smallest]:  # If right is smaller than smallest
            smallest = right

        if smallest != idx:  # If smallest value has changed
            arr[idx], arr[smallest] = arr[smallest], arr[idx]  # Switch items
            self.heapify(arr, smallest)  # Recursively call heapify on array

        return arr

    def peek_smallest(self) -> int:
        """
        Returns smallest element in the heap
        """
        return self.heap[0] if self.heap else None

    def pop_smallest(self) -> int:
        """
        Returns the smallest element and pops it from the array
        :return: Smallest element in heap
        """
        if not self.heap:
            return None
        resp = self.heap.pop(0)
        arr = self.heap
        for i in range(len(arr) // 2 - 1, 0, -1):
            arr = self.heapify(arr, i)

        self.heap = arr
        return resp

    def insert(self, value: int):
        """
        Inserts value in the heap
        :param value: value to be inserted
        """
        self.heap.append(value)

        last = len(self.heap) - 1

        while self.heap[last] < self.heap[last // 2]:  # While new node is smaller than parent
            self.swap(last, last // 2)  # Swap nodes
            last = last // 2  # Continue while loop


class MaxHeap:

    def __init__(self, arr: list = None):
        if not arr:
            self.heap = []
            return
        start_idx = len(arr) // 2 - 1
        for i in range(start_idx, 0, -1):
            self.heapify(arr, i)
        self.heap = arr

    def swap(self, node1: int, node2: int):
        self.heap[node1], self.heap[node2] = self.heap[node2], self.heap[node1]

    def heapify(self, arr: list, idx: int):
        if not arr:
            return None
        largest = idx
        left = 2 * idx + 1
        right = 2 * idx + 2

        if left < len(arr) and arr[left] > arr[largest]:
            largest = left

        if right < len(arr) and arr[right] > arr[largest]:
            largest = right

        if largest != idx:
            self.swap(idx, largest)
            self.heapify(arr, largest)

        return arr

    def peek_largest(self) -> int:
        """
        Returns the largest element in heap without popping
        :return: largest element in heap
        """
        if not self.heap:
            return None
        return self.heap[0]

    def pop_largest(self) -> int:
        """
        Pops and returns largest element, mutating the heap.
        :return: largest element in heap
        """
        if not self.heap:
            return None
        resp = self.heap.pop(0)
        self.heapify(self.heap, len(self.heap) - 1)
        return resp

    def insert(self, value: int):
        """
        Inserts a new value in the heap
        :param value: value to be inserted
        """
        self.heap.append(value)
        last = len(self.heap) - 1

        while self.heap[last] > self.heap[last // 2]:
            self.swap(last, last // 2)
            last = last // 2


class HeapProblems:
    @staticmethod
    def top_k_words(phrase: str, k: int) -> list[str]:
        words = phrase.split(' ')
        d = defaultdict(int)
        for word in words:  # O(n) Time and O(n) Space, with n = len(phrase)
            d[word] += 1

        heap = []
        for word, amount in d.items():  # O(m) Time and O(m) space, with m = number of unique words
            heap.append((amount, word))

        heapq.heapify(heap)

        return list(x[1] for x in heapq.nlargest(k, heap))

    @staticmethod
    def smallest_subset(s: list) -> list:
        """
        Given an array of integer, return the smallest subset where sum(subset) > sum(rest_of_nums)
        :param s: array of integers
        :return: smallest subset where sum is greater than rest elements sum
        """
        heapq.heapify(s)
        res = []
        for i in range(1, len(s)):
            hq = heapq.nlargest(i, s)
            if sum(hq) > sum(heapq.nsmallest(len(s) - i, s)):
                while hq:
                    res.append(hq.pop())
                return res
        return s

    @staticmethod
    def top_k_frequent_element_in_array(nums: list, k: int) -> list:
        """
        Given an array of nums and integer k, return k most frequent items in descending order of frequency
        :param nums: array of integers
        :param k: integer representing Kth most frequent elements
        :return: Kth most frequent elements
        """
        # Use hashmaps and priority queue (heap)
        d = defaultdict(int)
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

    def lengthOfLongestSubStringwithoutRepeating(self, s: str) -> int:
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
