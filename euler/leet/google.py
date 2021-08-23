import sys
from functools import reduce

from trees.tree import TreeNode
import collections


class CustomAlphabet:

    def __init__(self, alphabet: str):
        d = {}
        for i, char in enumerate(alphabet):
            d[char] = i
        self.alphabet = d

    def is_sorted(self, word: str) -> bool:
        if word == "":  # base case empty input
            return True

        for i in range(1, len(word)):  # iterate word
            if not self.alphabet.get(word[i], None) \
                    or not self.alphabet.get(word[i - 1]):  # skip character if not in alphabet
                continue
            else:
                if self.alphabet[word[i]] < self.alphabet[word[i - 1]]:
                    return False
        return True

    def is_sorted_with_array(self, arr: list) -> bool:
        for word in arr:
            is_sorted = self.is_sorted(word)
            if not is_sorted:
                return False
        return True


class GraphNode:
    def __init__(self, edge: int, neighbors: list):
        self.edge = edge
        self.neighbors = neighbors


class Node:
    def __init__(self, word="", depth=-1, children=None):
        self.word = word
        self.depth = depth

INT_MIN = -2**32
MTX_MAX = 20


class FindIslands:
    def __init__(self, matrix: list):
        self.matrix = matrix
        self.visited = set()

    def is_valid(self, i, j):
        return 0 <= i < len(self.matrix) \
               and 0 <= j < len(self.matrix[i])

    def dfs_util(self, i, j):
        """
        Runs DFS until a 0 (water) is found
        """
        self.visited.add((i, j))
        if self.matrix[i][j] == 0:  # If water is found
            return
        neighbors = [(i-1,j), (i+1, j), (i, j-1), (i, j+1)]
        for x, y in neighbors:  # Check all neighbors
            if (x, y) not in self.visited and self.is_valid(x, y):  # If not visited and valid
                self.dfs_util(x, y)  # Run DFS search on neighbor


    def find_islands(self) -> int:
        """
        Returns the number of islands in a 2D array conatining 1s and 0s.
        An island is a 1 with all adjacent nodes == 0.
        Adj nodes are up, left, right, and down. No diagonal neighbor considered.
        Assume all edges outside the grid are water.
        :param mtx: 2D array with 1s and 0s
        :return: number of islands present in matrix
        """
        islands = 0
        if not self.matrix:
            return islands
        for row in range(len(self.matrix)):
            for col in range(len(self.matrix[row])):
                if self.matrix[row][col] == 1 and (row, col) not in self.visited:
                    self.dfs_util(row, col)
                    islands +=1
        return islands


class Google:

    def distribute_bonuses(self, arr: list) -> list:
        """
        Returns lis of bonuses given an array of integers
        """

        if not arr:  # handle base case
            return []

        if len(arr) == 1:  # handle base case
            return arr
        resp = [1 for element in arr]  # create resp array of size len(arr)

        if arr[0] > arr[1]:  # handle first element
            resp[0] += 1

        if arr[-1] > arr[-2]:  # handle last element
            resp[-1] += 1

        for i in range(1, len(arr) - 1):  # iterate rest of array
            if arr[i] > arr[i - 1]:  # handle left neighbor
                resp[i] += 1
            if arr[i] > arr[i + 1]:  # handle right neighbor
                resp[i] += 1

        return resp

    def sum_bit_difference(self, arr: list) -> int:
        """
        Returns the sum of bit differences in all pairs that can be formed from array elements. Bit difference of a pair
        (x, y) is count of different bits at same positions in binary representations of x and y.
        For example, bit difference for 2 and 7 is 2. Binary representation of 2 is 010 and 7 is 111
        """
        if not arr:  # base check valid input
            return 0
        ans = 0  # initialize answer
        # solve this problem in O(n) time using the fact that all numbers are represented using 32 bits
        # The idea is to count differences at individual bit positions
        for i in range(32):  # traverse from 0 to 31 and count numbers with i’th bit set
            # There would be “n - count” numbers with i’th bit not set.
            # So count of differences at i’th bit would be “count * (n-count) * 2”.
            # the reason for this formula is as every pair having one element which has set bit at i’th position
            # and second element having unset bit at i’th position contributes exactly 1 to sum,
            # therefore total permutation count will be count * (n-count) and multiply by 2 is due to one more
            # repetition of all this type of pair as per given condition for making pair 1 <= i, j <= N.
            count = 0
            for j in range(len(arr)):
                if arr[j] and 1 << i:
                    count += 1

            ans += 2 * count * (len(arr)-count)

        return ans

    def longest_substring_with_k_unique_characters(self, s: str, k: int) -> int:
        """
        Returns the size of the longest substring with k unique characters
        """
        if s == "":
            return 0

        d = collections.defaultdict(lambda: 0)
        for char in s:  # create mapping of qty of characters
            d[char] += 1

        if len(d) < k:  # not enough unique characters
            return 0

        curr_start = curr_end = 0
        max_window_size = 1

        window_uniques = collections.defaultdict(lambda: 0)
        window_uniques[s[curr_start]] += 1

        for i in range(1, len(s)):
            window_uniques[s[i]] += 1  # add one count to the character
            curr_end += 1  # add one size to the window
            while len(window_uniques) > 3:  # if window is too big, recur and shrink
                window_uniques[s[curr_start]] -= 1
                if window_uniques[s[curr_start]] == 0:
                    window_uniques.pop(s[curr_start])
                curr_start += 1  # increment a pointer to the start of window

            if curr_end-curr_start+1 > max_window_size:  # if current window is bigger than max window so far
                max_window_size = curr_end-curr_start+1  # adding one since keeping track of pointers here

        return max_window_size

    def longest_increasing_subsequence(self, arr: list) -> int:
        """
        Returns the longest increasing subsequence in a list of integers
        """
        # Algorithm has O(nlogn) time complexity
        if not arr:  # base check
            return 0

        def find_ceil(arr: list, low: int, high: int, target: int):

            while high - low > 1:
                mid = low + (high - low) // 2
                if arr[mid] >= target:
                    high = mid
                else:
                    low = mid
            return high

        tail_tracker = [0 for i in range(len(arr)+1)]
        tail_tracker[0] = arr[0]  # add first element as tail and start recurring
        pointer = 1
        for i in range(1, len(arr)):  # start at 1 since we have already extracted first element
            if arr[i] < tail_tracker[0]:  # new smallest element found
                tail_tracker[0] = arr[i]  # replace smallest element
            elif arr[i] > tail_tracker[pointer-1]:  # if bigger, extend current subsequence
                tail_tracker[pointer] = arr[i]
                pointer += 1
            else:  # current element is candidate to end an existing subsequence
                tail_tracker[find_ceil(tail_tracker, -1, pointer-1, arr[i])] = arr[i]  # replace ceil in tracker

        return pointer



    def find_intersection_between_arrays(self, arr1: list, arr2: list):
        """
        Returns the intersection between two arrays.In other words, the elements present in both arrays
        """

        if not arr1 or not arr2:
            return []

        d = {}  # dictionary stores elements in first array
        resp = []
        for num in arr1:  # create dictionary
            d[num] = True

        for num in arr2:  # Check the intersection
            found = d.get(num, False)
            if found:
                resp.append(num)

        return resp

    def product_array_except_index(self, arr: list) -> list:
        """
        Returns a list containing at index the product of all the elements except the one at that given index
        in the original array. Division cannot be used in this case (cannot get total product and divide at each index).
        :param arr: list of integers. i.e. : [1,2,3,4,5]
        :return: list of integers with product of array except index. i.e. : [120, 60, 40, 30, 24]
        """
        resp = []
        for i, num in enumerate(arr):
            left_half = arr[:i]  # split array in half and omit element at index
            right_half = arr[i+1:]
            # since 1 is neutral in multiplication problems, use in case either half is empty
            left_val = reduce((lambda x, y: x*y), left_half) if left_half else 1
            right_val = reduce((lambda x, y: x * y), right_half) if right_half else 1
            resp.append(left_val*right_val)  # append the multiplication of both halves

        return resp

    def find_triplets_summing_zero(self, nums: list) -> list:
        """
        Given an integer array nums, return all the triplets [nums[i], nums[j], nums[k]] such that i != j, i != k,
        and j != k, and nums[i] + nums[j] + nums[k] == 0.
        Notice that the solution set must not contain duplicate triplets
        """
        nums.sort()  # O(nlogn)
        added = set()
        out = []
        for i in range(len(nums) - 1, -1, -1):  # reverse iteration
            last = nums[i]  # last evaluated
            start, end = 0, i - 1  # start from bottom until you reach previous element in array
            while start < end:  # inner loop from start to finish
                target = last + nums[start] + nums[end]  # searched value
                if target == 0:  # if we have found it
                    if (last, nums[start], nums[end]) not in added: out.append([last, nums[start], nums[end]])
                    added.add((last, nums[start], nums[end]))  # add triplet to set, avoid duplicates
                    start += 1  # continue with next element
                elif target > 0:  # if target is too big, then decrease last pointer (since sorted)
                    end -= 1
                else:  # if target is too small, then increase bottom pointer (since sorted)
                    start += 1
        return out

    def merge_list_numbers_into_ranges(self, arr: list) -> list:
        """
        Returns a list containing the ranges of all consecutive elements in a sorted array. for example:
        Input: [0, 1, 2, 5, 7, 8, 9, 9, 10, 11, 15]
        Output: ['0->2', '5->5', '7->11', '15->15']
        :param arr: list of sorted integers
        :return: list containing ranges
        """
        if not arr:
            return []
        resp = []
        prev = -1
        for i in range(len(arr)):
            if prev == -1:  # if searching prev, assign and continue
                prev = arr[i]
                continue
            if arr[i]-1 != arr[i-1] and arr[i] != arr[i-1]:  # If not consecutive and not equal
                s = f"{prev}->{arr[i-1]}"  # generate string
                resp.append(s)  # append to result
                prev = arr[i]  # reset prev for next search

        s = f"{prev}->{arr[len(arr)-1]}"  # handle last case
        resp.append(s)  # append to result

        return resp

    def reverse_words(self, s: str) -> str:
        """
        Returns the string with all its words reversed. White spaces are maintained
        :param s: string with words
        :return: string with reversed words
        """
        q = collections.deque([])
        stack = []
        res = ""
        for word in s.split(" "):  # form queue of reversed words to be appended to result
            w = ""
            # reverse the word
            for char in word:
                stack.append(char)
            while stack:
                char = stack.pop()
                w += char
            q.append(w)  # append reversed word to q

        while q:  # Iterate through queue and form res string
            word = q.popleft()
            res += f"{word} "

        if res:  # Clean last white space
            res = res[:-1]

        return res

    def room_scheduling(self, arr: list) -> int:
        """
        Returns the number of rooms required to hold meetings given a list of tuples (start, end)
        representing time intervals for lectures. The intervals may be overlapping.
        :param arr: list of tuples (start, end) with time intervals of meetings
        :return: min number of rooms required to hold all meetings
        """
        # Time complexity O(n*m) Where n = len(arr) and m = len(res)
        res = []

        for tuple in arr:  # O(n)
            new_room = True
            if not res:
                res.append([tuple])
                continue
            for scheduled in res:  # O(m)
                if scheduled[-1][1] <= tuple[0]:  # If latest scheduled meeting ends before tuple starts
                    scheduled.append(tuple)
                    new_room = False
                    break  # No need to continue looping the scheduled meetings
                else:
                    if scheduled[-1][0] >= tuple[1]:  # If latest scheduled meeting starts after current ends
                        scheduled.insert(0, tuple)
                        new_room = False
                        break
            if new_room:  # If i need a new room to hold te meeting
                res.append([tuple])

        return len(res)

    def running_median(self, arr: list) -> list:
        """
        Returns a list computing the running median, which is the median of the list with each new element.
        :param arr:
        :return:
        """

    def decompress_compress_string(self, arr: str) -> str:
        """
        Given a string in the format num[char] return the decompressed string resulting in num*char
        It can be given that we have nested strings like 3[2[a]b]
        :param arr: input string
        :return: decompressed string
        """
        closePos = {}
        st = []
        for i, c in enumerate(arr):
            if c == '[':
                st.append(i)
            elif c == ']':
                closePos[st.pop()] = i

        def dfs(l, r):
            num = 0
            ans = ""
            while l <= r:
                c = arr[l]
                if ord('0') <= ord(c) <= ord('9'):
                    num = num * 10 + ord(c) - ord('0')
                elif c == '[':
                    ans += num * dfs(l + 1, closePos[l] - 1)
                    num = 0
                    l = closePos[l]
                else:
                    ans += c
                l += 1
            return ans

        return dfs(0, len(arr) - 1)

    def find_binary_tree_max_sum(self, root: TreeNode) -> int:
        """
        Given a binary tree, find the max path sum. The path may start and end at any node in thre tree
        :param root: root of tree
        :return: max path sum
        """
        def recursive_helper(root: TreeNode):
            """
            Helper used for recursion
            """
            # Base Case
            if not root:
                return 0

            # l and r store maximum path sum going through left
            # and right child of root respectively
            l = recursive_helper(root.left)
            r = recursive_helper(root.right)

            # Max path for parent call of root. This path
            # must include at most one child of root
            max_single = max(max(l, r) + root.data, root.data)

            # Max top represents the sum when the node under
            # consideration is the root of the maxSum path and
            # no ancestor of root are there in max sum path
            max_top = max(max_single, l + r + root.data)

            # Static variable to store the changes
            # Store the maximum result
            recursive_helper.res = max(recursive_helper.res, max_top)

            return max_single

        recursive_helper.res = float("-inf")
        recursive_helper(root)
        return recursive_helper.res

    def find_max_path_between_leaves_binary_tree(self, root: TreeNode) -> int:
        """
        Given a binary tree in which each node element contains a number. Find the maximum possible sum from one leaf
        node to another. The maximum sum path may or may not go through root.
        """

        def recursive_helper(root, res):
            # Base Case
            if not root:
                return 0

            # Find maximumsum in left and righ subtree. Also
            # find maximum root to leaf sums in left and right
            # subtrees ans store them in ls and rs
            ls = recursive_helper(root.left, res)
            rs = recursive_helper(root.right, res)

            # If both left and right children exist
            if root.left and root.right:
                # update result if needed
                res[0] = max(res[0], ls + rs + root.data)

                # Return maximum possible value for root being
                # on one side
                return max(ls, rs) + root.data

            # If any of the two children is empty, return
            # root sum for root being on one side
            if not root.left:
                return rs + root.data
            else:
                return ls + root.data

        res = [INT_MIN]
        recursive_helper(root, res)
        return res[0]

    def count_paths_in_matrix(self, matrix: list) -> int:
        """
        Count the number of routes from start to finish moving down or right.
        """
        # Using memoization DP
        dp = [[]]
        def recursive_helper(matrix, n, m):
            if n < len(matrix) and m < len(matrix[0]):
                if n == len(matrix)-1 and len(matrix[0])-1:
                    dp[n][m] = 1
                    return dp[n][m]
                # Check dp array to see if we have found route for a particular cell. If result is not None, return it.
                if dp[n][m]:
                    return dp[n][m]
                # Store the sum of ways we can reach by right and bottom cell in the dp array and return result
                dp[n][m] = recursive_helper(matrix, n+1, m) + recursive_helper(matrix, n, m+1)
                return dp[n][m]
            return 0

        return recursive_helper(matrix, 0, 0)

    def word_ladder(self, beginWord: str, endWord:str, wordList: list) -> int:
        """
        Given two words, beginWord and endWord, and a dictionary wordList, return the number of words in the shortest
        transformation sequence from beginWord to endWord, or 0 if no such sequence exists.
        """
        beginNode = Node(beginWord, 1)
        queue = collections.deque()
        queue.append(beginNode)
        wordSet = set(wordList)

        while queue:
            curNode = queue.popleft()
            curWord, curDepth = curNode.word, curNode.depth
            # find children
            # iterate through wordList and find words that are off by 1 char
            wordList = list(wordSet)  # make copy so we can delete from wordSet while iterating
            for word in wordList:
                curIdx = 0
                numUnmatch = 0
                skipWord = False  # flag for whether we skip this word
                for char in word:
                    if curWord[curIdx] != char:
                        numUnmatch += 1
                        if numUnmatch == 2:
                            # we only want words that are off by 1 char
                            skipWord = True
                            break
                    curIdx += 1
                if skipWord:
                    # this word has more than one char difference, so skip it
                    skipWord = False
                    continue
                else:
                    # if it's endWord then we're done
                    if word == endWord:
                        return curDepth + 1
                    child = TreeNode(word, curDepth + 1)
                    queue.append(child)
                    # remove word from list
                    wordSet.remove(word)

        # not possible (e.g. the endWord isn't in list, or no path to endWord)
        return 0

    def longest_increasing_path_in_matrix(self, matrix: list):
        """
        Given a matrix of N rows and M columns. From m[i][j], we can move to m[i+1][j], if m[i+1][j] > m[i][j],
        or can move to m[i][j+1] if m[i][j+1] > m[i][j]. The task is print longest path length if we start from (0, 0).
        :param matrix: matrix of N rows and M columns
        """
        dp = [[-1 for i in range(MTX_MAX)] for i in range(MTX_MAX)]
        n = len(matrix)
        m = len(matrix[0])
        def recursive_helper(matrix, n, m, x, y):
            # If value not calculated yet.
            if dp[x][y] < 0:
                res = 0
                # If reach bottom left cell,
                # return 1.
                if (x == n - 1 and y == m - 1):
                    dp[x][y] = 1
                    return dp[x][y]

                # If reach the corner
                # of the matrix.
                if (x == n - 1 or y == m - 1):
                    result = 1

                # If value greater than below cell.
                if (x + 1 < n and matrix[x][y] < matrix[x + 1][y]):
                    result = 1 + recursive_helper(dp, matrix, n,
                                     m, x + 1, y)

                # If value greater than left cell.
                if (y + 1 < m and matrix[x][y] < matrix[x][y + 1]):
                    result = max(res, 1 + recursive_helper(dp, matrix, n,
                                                 m, x, y + 1))
                dp[x][y] = result
            return dp[x][y]
        return recursive_helper(matrix, n, m, 0, 0)

    def get_max_sum_leaves_to_root_path_binary_tree(self, root: TreeNode) -> int:
        """
        Given a Binary Tree, find the maximum sum path from a leaf to root
        """
        max_sum = 0
        target_leaf = None

        def print_path(root, target_leaf):
            """
            Print helper
            """

            # base case
            if not root:
                return False

            # return True if this node is the target_leaf
            # or target leaf is present in one of its
            # descendants
            if root == target_leaf or print_path(root.left, target_leaf) or print_path(root.right, target_leaf):
                print(root.val, end=" ")
                return True

            return False

        def recursive_helper(root: TreeNode, curr_sum: int):
            """
            This function Sets the target_leaf_ref to refer the leaf node of the maximum path sum. Also,
            returns the max_sum using max_sum_ref
            """
            if not root:
                return
            # Update current sum to hold sum of nodes on path from root to this node
            curr_sum += root.val
            # If this is a leaf node and path to this node had max sum so far, then make this node target_leaf
            if not root.left and not root.right:
                max_sum = curr_sum
                target_leaf = root

            # If this is not a leaf node, then recur down to find the target_leaf
            recursive_helper(root.left, curr_sum)
            recursive_helper(root.right, curr_sum)

        # Base case
        if not root:
            return 0

        recursive_helper(root, 0)
        print_path(root, target_leaf)
        return max_sum

    def find_min_number_coins(self, total: int) -> int:
        """
        Given a value V, if we want to make a change for V Rs, and we have an infinite supply of each of the
        denominations in Indian currency, i.e., we have an infinite supply of { 1, 2, 5, 10, 20, 50, 100, 500, 1000}
        valued coins/notes, what is the minimum number of coins and/or notes needed to make the change?
        """
        # All denominations of Indian Currency
        deno = [1, 2, 5, 10, 20, 50,
                100, 500, 1000]
        n = len(deno)

        # Initialize Result
        ans = []

        # Traverse through all denomination
        i = n - 1
        while (i >= 0):

            # Find denominations
            while (total >= deno[i]):
                total -= deno[i]
                ans.append(deno[i])

            i -= 1

        # Print result
        for i in range(len(ans)):
            print(ans[i], end=" ")

    def find_min_number_coins_dp(self, coins: list, total: int) -> int:
        """
        Given a value V, if we want to make change for V cents, and we have infinite supply of each of
        C = { C1, C2, .. , Cm} valued coins, what is the minimum number of coins to make the change?
        If it’s not possible to make change, print -1.
        """
        # O(mV) where m = len(coins) and V = total

        # table[i] will be storing the minimum number of coins required for i value. So table[total] will have result
        table = [0 for i in range(total + 1)]

        # Base case (If given value total is 0)
        table[0] = 0

        # Initialize all table values as Infinite
        for i in range(1, total + 1):
            table[i] = sys.maxsize

        # Compute minimum coins required
        # for all values from 1 to V
        for i in range(1, total + 1):

            # Go through all coins smaller than i
            for j in range(len(coins)):
                if coins[j] <= i:
                    sub_res = table[i - coins[j]]
                    if (sub_res != sys.maxsize and
                            sub_res + 1 < table[i]):
                        table[i] = sub_res + 1

        return table[total] if table[total] != sys.maxsize else -1

class urlShortener:
    """
    Use the integer id stored in the database and convert the integer to a character string that is at most 6 characters
    long. This problem can basically seen as a base conversion problem where we have a 10 digit input number and we want
    to convert it into a 6 character long string.
    Below is one important observation about possible characters in URL.
    A URL character can be one of the following
    1) A lower case alphabet [‘a’ to ‘z’], total 26 characters
    2) An upper case alphabet [‘A’ to ‘Z’], total 26 characters
    3) A digit [‘0’ to ‘9’], total 10 characters
    There are total 26 + 26 + 10 = 62 possible characters.
    So the task is to convert a decimal number to base 62 number.
    To get the original long URL, we need to get URL id in the database.
    The id can be obtained using base 62 to decimal conversion.
    """

    def idToShortURL(self, id):
        map = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        shortURL = ""

        # for each digit find the base 62
        while (id > 0):
            shortURL += map[id % 62]
            id //= 62

        # reversing the shortURL
        return shortURL[len(shortURL):: -1]

    def shortURLToId(self, shortURL):
        id = 0
        for i in shortURL:
            val_i = ord(i)
            if (val_i >= ord('a') and val_i <= ord('z')):
                id = id * 62 + val_i - ord('a')
            elif (val_i >= ord('A') and val_i <= ord('Z')):
                id = id * 62 + val_i - ord('Z') + 26
            else:
                id = id * 62 + val_i - ord('0') + 52
        return id

