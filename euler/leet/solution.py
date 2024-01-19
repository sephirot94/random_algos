import collections
import heapq
import math
import ast
import operator
import re
import functools
import sys
import bisect
from collections import defaultdict, deque
from itertools import permutations


class Solution:

    def reverse_integer(self, num: int) -> int:

        if num < -2 ** 31 or num > 2 ** 31 - 1:  # input is valid
            return 0

        ans = ""
        str_num = str(num)
        for i in range(len(str_num) - 1, -1, -1):  # iterate string backwards
            ans += str_num[i]

        return int(ans)

    def longest_common_prefix(self, words: list[str]) -> str:
        """
        Find the longest common prefix string amongst an array of strings. 
        If there is no common prefix, return an empty string
        """""
        if not words:
            return ""
        base = words[0]
        for i in range(len(base)):
            for word in words[1:]:
                if i == len(word) or word[i] != base[i]:
                    return base[:i]
        return base

    def celebrity_finder(self, matrix: list) -> int:
        """
        Returns an integer indicating person which is known to all but does not know anyone at party.
        Returns -1 if no celebrity is found at the party
        :param matrix: NXN matrix. Rows represent people at the party, and columns represent the people that the current
        row knows.
        """
        if not matrix:  # base check valid input
            return -1
        adj = defaultdict(set)
        for row in range(len(matrix)):  # traverse matrix and fill dictionary
            for column in range(len(matrix[row])):
                if matrix[row][column] == 1:  # row knows column
                    adj[row].add(column)

        for person in range(len(matrix)):
            if len(adj[person]) == 0:
                search = True
                for n_person in range(len(matrix)):
                    if n_person == person:  # avoid current candidate
                        continue
                    if person not in adj[n_person]:
                        search = False
                        break
                if search:
                    return person
        return -1

    def group_anagrams(self, arr: list[str]) -> list:
        """
        Returns a list with grouped anagrams (words made up of the same letters)
        """
        # Time complexity is O(NMlogM) Where N = len(arr) and M=len(word) for word in arr
        res = []
        d = collections.defaultdict(list)
        for word in arr:  # Generate a Hashmap storing all words that are anagrams
            d[str(sorted(word))].append(word)

        for group in d.values():
            res.append(group)

        return res

    def min_remove_to_valid_parenth(self, s: str) -> int:
        """
        Returns the minimum removals to be made for the string to contain valid set of parenthesis
        :param s: string containing parenthesis
        :return: integer with minimum removals that must be made for string to be valid
        """
        stack = []
        res = 0
        for char in s:
            if char == '(':
                stack.append(char)
            if char == ')':
                open = stack.pop() if stack else None
                if not open:
                    res += 1

        return res + len(stack)

    def is_valid_palindrome_with_one_del(self, s:str) -> bool:
        """
        Given a string s, return true if the s can be palindrome
        after deleting at most one character from it
        :param s:
        :return:
        """
        err = 0
        start, end = 0, len(s)-1
        while start < end:
            while s[start] != s[end]:
                if err == 0:  # First occurence, check moving left side
                    start += 1
                if err == 1:  # If it did not work, correct left side movement and move right side
                    start -= 1
                    end -= 1
                err += 1  # Worst case scenario, it will equal 2 after one delete operation
                if err > 2:  # Will be true if already deleted and still no palindrome
                    return False

            # Move pointers
            start += 1
            end -= 1
        return True

    def is_valid_palindrome(self, s: str) -> bool:
        """
        Given a string with different characters, considering only the alphanumeric ones,
        verify string is a valid palindrome
        :param s: string containing multiple characters
        :return: boolean indicating if it is a valid palindrome
        """
        pattern = re.compile('[^a-zA-Z0-9]')
        s = re.sub(pattern, '', s)
        for i in range(len(s)//2):
            if s[i].lower() != s[len(s)-i-1].lower():
                return False
        return True

    def two_sum(self, nums: list, target: int) -> list:
        """
        Given an array of integers and a target, return tuple with two values which sum equals target
        :param nums: array of integers
        :param target: target value being searched
        :return: tuple with two values. if not found, empty array.
        """
        diff_map = {}
        for i, num in enumerate(nums):
            diff = target - num
            if diff in diff_map:
                return [diff_map[diff], i]
            else:
                diff_map[num] = i
        return []

    def checkPerfectNumber(self, num):
        factors = []
        if num <= 0:
            return False
        for i in range(1, int(math.ceil(num ** 0.5))):
            if num % i == 0:
                factors.append(i)
                factors.append(num / i)
        sum = 0
        if factors:
            sum = functools.reduce(lambda x, y: x + y, factors) - num
            return sum == num

    def lengthOfLongestSubstring(self, s: str) -> int:
        d = {}
        start = 0
        res = 0

        for i, elem in enumerate(s, 1):
            val = d.get(elem, 0)

            if val > start:
                start = val

            if i - start > res:
                res = i - start

            d[elem] = i

        return res

    def minimumSwaps(self, arr: list) -> int:
        """
        Return minimum swaps required to sort an array
        :param arr: input array to be checked
        :return: min number of swaps required to sort the array
        """
        m = {}
        for i, n in enumerate(arr):
            idx = m.get(i+1, 0)
            if idx != 0:
                tmp = arr[i]
                arr[i] = arr[idx]
                arr[idx] = tmp
            else:
                m[n] = i
        return len(m)

    def is_balanced(self, s: str) -> bool:
        """
        Check wether a string has balanced () [] {}
        :param s: string to check
        :return: true if string is balanced, else false
        """
        stack = []

        for char in s:
            if char in ['(', '[', '{']:
                stack.append(char)
            if char in [')', ']', '}']:
                if len(stack) < 1:
                    return False
                c = stack.pop()
                if char == ')' and c != '(':
                    return False
                if char == ']' and c != '[':
                    return False
                if char == '}' and c != '{':
                    return False

        if len(stack) == 0:
            return True

        return False

    def flatlandSpaceStations(self, n, c):
        res = 0
        m = 0
        prev = 0
        for i in c:
            if i - prev > m:
                m = i - prev
            prev = i
        res = math.floor(m / 2)
        return res

    def addDigits(self, num):
        i = num % 9
        if i == 0:
            return 9
        return i

    def rotate(self, matrix):
        for j in range(0, len(matrix)):
            for i in range(j + 1, len(matrix[j])):
                tmp = matrix[i][j]
                matrix[i][j] = matrix[j][i]
                matrix[j][i] = tmp
            matrix[j].reverse()
        return matrix

    def findDuplicates(self, nums):
        outputs = []
        for n in nums:
            value = abs(n) - 1
            if nums[value] < 0:
                outputs.append(abs(n))
            else:
                nums[value] = nums[value] * -1
        return outputs

    def encode_string(self, s: str) -> str:
        if not s or len(s) == 0:
            return ''
        ctr = 0
        curr = ''
        output = ''
        for char in s:
            if curr == '':
                curr = char
                ctr+=1
                continue
            if char==curr:
                ctr+=1
            else:
                output += f"{ctr}{curr}"
                ctr = 1
                curr = char
        if ctr > 0:
            output+= f"{ctr}{curr}"
        return output

    def isPalindrome(self, x: int) -> bool:
        x = str(x)
        for i in range(len(x) // 2):
            if x[i] != x[len(x) - 1 - i]:
                return False
        return True

    def single_number(self, l: list) -> int:
        d = {}
        for num in l:
            val = d.get(num, 0)
            if val == 0:
                d[num] = 1
            else:
                d[num] = d[num] + 1

        for key in d:
            if d[key] == 1:
                return key

        return 0

    def sort_colors(self, l: list) -> list:
        """
        Given an array with 3 unique values (1,2,3) return the ordered list in place (Dutch National Flag Problem)

        :param l: list of unordered 3 unique values (1,2,3)

        :return: ordered list with 3 unique values (1,2,3)
        """
        high, mid, low = -1, 0, 0
        while (len(l)+high) != (mid-1):
            if l[mid] == 3:
                l[mid] = l[high]
                l[high] = 3
                high -= 1
            if l[mid] == 2:
                mid += 1
            if l[mid] == 1:
                l[mid] = l[low]
                l[low] = 1
                mid += 1
                low += 1

        return l

    def stock_max_profit(self, stocks: list) -> int:
        """
        Given an array of stock prices during a day, return the max gain that could have been made
        buying lowest price and selling at highest.
        :param stocks: list containing stock prices
        :return: max gain of the day
        """
        min_price = stocks[0]
        max_profit = 0
        for stock in stocks[1:]:
            max_profit = max(max_profit, stock - min_price)
            min_price = min(min_price, stock)
        return max_profit

    def stock_max_profit_2(self, prices: list) -> int:
        """
        You are given an integer array prices where prices[i] is the price of a given stock on the ith day.
        On each day, you may decide to buy and/or sell the stock. You can only hold at most one share of the stock at
        any time. However, you can buy it then immediately sell it on the same day.
        Find and return the maximum profit you can achieve.
        :return: max gain of the day
        """
        profit_from_price_gain = 0
        for idx in range(len(prices) - 1):
            if prices[idx] < prices[idx + 1]:
                profit_from_price_gain += (prices[idx + 1] - prices[idx])

        return profit_from_price_gain

    def stock_max_profit_3(self, prices: list) -> int:
        """
        You are given an array prices where prices[i] is the price of a given stock on the ith day.
        Find the maximum profit you can achieve. You may complete at most two transactions.
        Note: You may not engage in multiple transactions simultaneously
        (i.e., you must sell the stock before you buy again).
        :return: max gain of the day
        """
        buy, sell = [float("inf")] * 2, [0] * 2
        for x in prices:
            for i in range(2):
                if i:
                    buy[i] = min(buy[i], x - sell[i - 1])
                else:
                    buy[i] = min(buy[i], x)
                sell[i] = max(sell[i], x - buy[i])
        return sell[-1]

    def stock_max_profit_4(self, prices: list, k: int) -> int:
        """
        You are given an array prices where prices[i] is the price of a given stock on the ith day.
        Find the maximum profit you can achieve. You may complete at most K transactions.
        Note: You may not engage in multiple transactions simultaneously
        (i.e., you must sell the stock before you buy again).
        :return: max gain of the day
        """
        if not prices or k <= 0:
            return 0
        buy, sell = [float("inf")] * k, [0] * k
        for x in prices:
            for i in range(2):
                if i:
                    buy[i] = min(buy[i], x - sell[i - 1])
                else:
                    buy[i] = min(buy[i], x)
                sell[i] = max(sell[i], x - buy[i])
        return sell[-1]

    def stock_max_profit_5(self, prices: list[int], fee: int) -> int:
        """
        Finds the maximum profit you can achieve given an array prices and an integer fee representing a transaction fee.
        You may complete as many transactions as you like, but you need to pay the transaction fee for each transaction.
        You may not engage in multiple transactions simultaneously (i.e., you must sell the stock before you buy again).
        The transaction fee is only charged once for each stock purchase and sale.
        """
        s = len(prices)
        if s <= 1: return 0
        buy, sell = -prices[0], 0  # Negative for max comparison in first iteration
        for price in prices[1:]:
            buy = max(buy, sell-price)
            sell = max(sell, price + buy - fee)


    def check_array_non_decreasing(self, l: list) -> bool:
        """
        Given an array of integers, return if it is possible to make it a non-decreasing array by modifying at most 1 element.
        We define an array is non-decreasing if array[i] <= array[i + 1] holds for every i (1 <= i < n)

        :param l: array of integers
        :return: boolean indicating if it can be made non_decreasing
        """
        used = False
        for i in range(1, len(l)):
            if l[i] < l[i-1]:
                if used:
                    return False
                used = True

        return True

    def check_pythagorean_triplet(self, l: list) -> bool:
        """
        Given a list of integers, check if exists a pythagorean triplet in array.
        A pythagorean triplet is 3 variables a, b, c where a**2 + b**2 = c**2
        :param l: list of integers
        :return: boolean indicating whether pythagorean triplet exists in list
        """
        maximum = 0
        # Find max element
        for num in l:
            maximum = max(maximum, num)

        # hash array
        h = [0] * (maximum+1)
        # Increase the count of array elements in hash table
        for num in l:
            h[num] += 1

        for i in range(1, maximum+1):
            if h[i] == 0:
                continue

            for j in range(1, maximum+1):
                # If a and b are same and there is only one a
                # or if there is no b in original array
                if (i == j and h[i] == 1) or h[j] == 0:
                    continue
                val = int(math.sqrt(i*i+j*j))

                # If C is not square
                if val*val != (i*i+j*j):
                    continue

                # If C > max
                if val > maximum:
                    continue

                # If there is c in the original array, then True
                if h[val]:
                    return True

        return False

    def ransom_note_with_words(self, magazine: str, note: str) -> bool:
        """
        Given a string of magazine and a note to be written,
        return whether is possible to write the note using the words in magazine.
        No concatenation of substrings allowed to create words from magazine.
        The words are case sensitive ('hello'!='Hello')
        :param magazine: string containing words in magazine
        :param note: string containing note to be written
        :return: boolean indicating if it is possible to write note with words in magazine
        """
        d = defaultdict(lambda: None)
        mag = magazine.split(" ")
        note = note.split(" ")
        for word in mag:
            exists = d[word]
            if exists:
                d[word] += 1
            else:
                d[word] = 1
        for word in note:
            exists = d[word]
            if not exists or exists == 0:
                return False
            else:
                d[word] -= 1
        return True

    def min_deletions_in_arr_to_make_adj_elem_non_decreasing(self, arr: list) -> int:
        """
        Function for finding minimum deletions so that the array becomes non-decreasing
        and the difference between adjacent elements is also non-decreasing
        :param arr: input array of integers
        :return: minimum deletions needed to make array non-decreasing with non-decreasing difference between adjacent elements
        """
        # Time complexity: O(N*M+N**2), where M is the maximum element in A,
        # Space: O(N*M)
        max = max(arr)
        size = len(arr)
        # initialize the dp table
        # set all values to 0
        # pref[i][j] will contain min(dp[i][0], dp[i][1], ...dp[i][j])
        dp = []
        pref = []
        for i in range(size):
            for j in range(max):
                dp[i][j] = 0
                pref[i][j] = 0

        # Find the max valid size set
        # possible and then substract its size
        # from size to get min number of dels
        for i in range(size):
            # when selecting only the current element and
            # deleting all elements from 0 to i-1 inclusive
            dp[i][0] = 1

            for j in reversed(range(i-1)):
                if arr[i] > arr[j]:
                    diff = arr[i] - arr[j]
                    # we can get min(dp[j][0], .. dp[j])
                    # from pref array
                    dp[i] = max(dp[i], pref[j]+1)
            # construct the prefix array for this element
            pref[i][0] = dp[i][0]
            for j in range(1, max):
                pref[i][j] = max(dp[i][j], pref[i][j-1])
        # take the max set size from dp[N-1][0] to dp[N-1][MAX]
        maxSize = -1
        for i in range(max):
            maxSize = max(maxSize, dp[size-1][i])
        return size - maxSize

    def kbars_visible_from_left(self, n: int, k: int) -> int:
        """
        Given a number K and N bars of height 1 to N, the task is to find the number of ways
        to arranve the N bars such that oly K bars are visible from the left
        :param n: N bars of 1 to N height (height = n)
        :param k: number of bars that should be visible from left once reordered bars
        :return: number of ways such that only K bars are visible from the left
        """
        # Using DP and memoization, efficient solution beats recursive
        # O(NK) Time and O(NK) Space
        dp = [[-1] * (k+1)] * (n+1)

        def recursive_helper(n, k):
            """
            Recursive helper closure
            :return: recursion handler
            """
            # If subproblem has been calculated, return
            if dp[n][k] != -1:
                return dp[n][k]
            # If not ascending order
            if n == k:
                dp[n][k] = 1
                return dp[n][k]

            if k == 1:
                ans = 1
                for i in range(1, n):
                    ans *= i
                dp[n][k] = ans
                return dp[n][k]

            # recursion
            dp[n][k] = recursive_helper(n-1, k-1) + (n-1) * recursive_helper(n-1, k)
            return dp[n][k]

        return recursive_helper(n, k)

    def first_last_index_of_element_in_sorted_arr(self, arr: list, target: int) -> list:
        """
        Given a sorted array of integers, find the lowest and highest index of a given integer
        :param arr: list of integers
        :param target: integer searched
        :return: lowest and highest index of target element
        """

        def binary_search_iterative_helper(arr: list, target: int, low: int, high: int, findFirst) -> int:
            """
            Given a sorted array of integers, search for an element using binary search
            :param arr: list of integers
            :param target: element to be searched
            :param low: bottom of list
            :param target: top of list
            :return: target
            """
            while True:
                if high < low:
                    return -1

                mid = low + (high - low) // 2
                if findFirst:
                    if (mid == 0 or target > arr[mid - 1]) and arr[mid] == target:
                        return mid
                    if target > arr[mid]:
                        low = mid + 1
                    else:
                        high = mid - 1
                else:
                    if (mid == len(arr) - 1 or target < arr[mid + 1]) and arr[mid] == target:
                        return mid
                    elif target < arr[mid]:
                        high = mid - 1
                    else:
                        low = mid + 1

        first = binary_search_iterative_helper(arr, 0, len(arr)-1, target, True)
        last = binary_search_iterative_helper(arr, 0, len(arr)-1, target, False)
        return [first, last]

    def binary_search_iterative(self, arr: list, target: int, low: int, high: int) -> int:
        """
        Given a sorted array of integers, search for an element using binary search
        :param arr: list of integers
        :param target: element to be searched
        :param low: bottom of list
        :param target: top of list
        :return: target
        """
        while True:
            if high < low:
                return -1

            mid = low + (high-low) // 2

            if (mid == len(arr)-1 or target < arr[mid+1]) and arr[mid] == target:
                return mid
            elif target < arr[mid]:
                high = mid - 1
            else:
                low = mid + 1

    @staticmethod
    def check_possible_path_2d_matrix(arr: list) -> bool:
        """
        Given a 2D array (m*n), check if there is any path from top left to bottom right. In the matrix, -1 = blockage
        and 0 = path cell (can go through).
        Must get from (0,0) to (m, n).
        :param arr: 2D array to be searched
        :return: boolean indicating if path exists
        """
        # O(m*n) Time and O(1) Space
        if not arr:
            return False
        # mark the cell (0,0) as 1
        arr[0][0] = 1
        # traverse first row  until blocked
        for i in range(len(arr[0])):
            if arr[0][i] == -1:
                break
            arr[0][i] = 1
        # traverse the first column until blocked
        for i in range(len(arr)):
            if arr[i][0] == -1:
                break
            arr[i][0] = 1
        # traverse matrix
        for i in range(1, len(arr)):
            for j in range(1, len(arr[i])):
                if arr[i][j] != -1:
                    if arr[i-1][j] == 1 or arr[i][j-1] == 1:
                        arr [i][j] = 1
        return arr[-1][-1] == 1

    @staticmethod
    def word_search(arr: list, target: str) -> bool:
        """
        Given a 2D array of characters, and a target string. Return whether or not the word target word exists
        in the matrix. Unlike a standard word search, the word must be either going left-to-right,
        or top-to-bottom in the matrix.
        :param arr: 2D array of characters
        :param target: target string searched in array
        :return: boolean indicating if target exists in 2d array
        """
        # Base cases
        if len(arr) == 0 or not target:
            return False
        if len(arr) < len(target):
            return False
        q = deque([])
        s = ""
        # traverse matrix
        for i in range(len(arr)):
            if s == target:
                return s == target
            while len(q) > 0:
                tuple = q.popleft()
                i, j = tuple[0], tuple[1]
                s += arr[i][j]
                if i + 1 < len(arr):
                    if arr[i+1][j] == target[len(s)]:
                        q.append((i+1, j))
                    else:
                        s = ''

            # Handle case of success vertically
            if s == target:
                return True

            for j in range(len(arr[i])):
                char = arr[i][j]
                # if find letter searched
                if char == target[len(s)]:
                    # sum character to final string
                    s += char
                    # horizontal check is handled by for loop
                    if j+1 < len(arr[i]) and arr[i][j+1] == target[len(s)]:
                        continue
                    # check vertically
                    elif i+1 < len(arr) and arr[i+1][j] == target[len(s)]:
                        # here we use queue to store the vertical values and assess them first
                        q.append((i+1, j))
                    elif s == target:
                        # This will catch everything edge cases
                        return s == target
                    else:  # if neither vertical or horizontal, reset string
                        s = ''
        return s == target

    @staticmethod
    def count_number_ways_reach_destination_maze(maze: list) -> int:
        """
        Given a 2D array representing a maze with obstacles. count number of paths to reach rightmost-bottommost cell
        from topmost-leftmost cell. A cell in given maze has value -1 if it is a blockage or dead end, else 0.
        From a given cell, we are allowed to move to cells (i+1, j) and (i, j+1) only.
        :param maze: 2D array representing maze
        :return: number ways to reach destination maze
        """
        # O(N*M) Time complexity N and M number of rows and columns
        # The idea is to modify the given grid[][] so that grid[i][j] contains count of paths to reach
        # (i, j) from (0, 0) if (i, j) is not a blockage, else grid[i][j] remains -1.
        # Base case
        # if not array or last element is blocked
        if len(maze) == 0 or maze[len(maze)-1][len(maze)-1] == -1:
            return 0
        # If start is blocked, return
        if maze[0][0] == -1:
            return 0

        for i in range(len(maze)):
            if maze[i][0] == 0:
                maze[i][0] = 1

            # If we encounter a blocked cell in leftmost row, there is no way of visiting any cell directly below it
            else:
                break

        for i in range(1, len(maze[0])):
            if maze[0][i] == 0:
                maze[0][i] = 1

            # If we encounter a blocked cell in bottommost row, there is no way of visiting any cell directly below it
            else:
                break

        # if a cell is -1, ignore it. Else, recursively compute count value maze[i][j]
        for i in range(1, len(maze)):
            for j in range(1, len(maze[i])):
                # If blockage is found, ignore the cell
                if maze[i][j] == -1:
                    continue

                # If we can reach maze[i][j] from maze[i-1][j] then increment count
                if maze[i-1][j] > 0:
                    maze[i][j] += maze[i-1][j]

                # If we can reach maze[i][j] from maze[i][j-1] then increment count
                if maze[i][j-1] > 0:
                    maze[i][j] += maze[i][j-1]

        return maze[len(maze)-1][len(maze)-1]

    @staticmethod
    def binomial_coefficient_constant_space(n: int, k: int) -> int:
        """
        Calculate C(n,k) where
        C(n,k) = n! / (n-k)! * k!
        :param n: integer n parameter in C(n,k) formula
        :param k: integer k parameter in C(n,k) formula
        :return: C(n,k) value for given tuple
        """
        # Because C(n,k) = C(n,n-k)
        if k > n - k:
            k = n - k
        res = 1
        # Calculate iteration of products
        # [n * (n-1) *---* (n-k + 1)] / [k * (k-1) *----* 1]
        for i in range(k):  # O(k) complexity
            res = res * (n - i)
            res = res / (i + 1)
        return res

    @staticmethod
    def binomial_coefficient_recursive(n: int, k: int) -> int:
        """
        Calculate C(n,k) recursively given following formula:
        C(n, k) = C(n-1, k-1) + C(n-1, k)
        C(n, 0) = C(n, n) = 1
        :param n: n parameter value in C(n,k) formula
        :param k: k parameter value in C(n,k) formula
        :return: C(n,k) for given parameters
        """
        if k > n:
            return 0
        if k == 0 or k == n:
            return 1
        return Solution.binomial_coefficient_recursive(n-1, k-1) + Solution.binomial_coefficient_recursive(n-1, k)

    @staticmethod
    def binomial_coefficient_dynamic(n: int, k: int) -> int:
        """
        Calculate C(n,k) iteratively using DP and following formula:
        C(n, k) = C(n-1, k-1) + C(n-1, k)
        C(n, 0) = C(n, n) = 1
        :param n: n parameter value in C(n,k) formula
        :param k: k parameter value in C(n,k) formula
        :return: C(n,k) for given parameters
        """
        map = [[0 for x in range(k+1)] for x in range(n+1)]

        for i in range(n+1):  # O(n*k) Space and Time
            for j in range(min(i, k)+1):
                # Base case
                if j == 0 or j == i:
                    map[i][j] = 1
                else:
                    # Use previous calculated values in map
                    map[i][j] = map[i-1][j-1] + map[i-1][j]

        return map[n][k]

    @staticmethod
    def catalan_number_recursive(n: int) -> int:
        """
        Return Nth catalan number (Catalan Series)
        :param n: Sn number in Series (Nth catalan number)
        :return: Nth number in catalan series

        O(n***) -> exponential equivalent to nth catalan number
        """
        if n <= 1:
            return 1

        # Catalan(n) is the sum
        # of catalan(i) * catalan(n-i-1)
        res = 0
        for i in range(n):
            res += Solution.catalan_number_recursive(i) * Solution.catalan_number_recursive(n-i-1)
        return res

    @staticmethod
    def catalan_number_dynamic(n: int) -> int:
        """
        Return Nth catalan number (Catalan Series)
        :param n: Sn number in Series (Nth catalan number)
        :return: Nth number in catalan series

        O(n**2) -> exponential
        """
        if n <= 1:
            return 1

        dCatalan = []
        dCatalan[0] = 1
        dCatalan[1] = 1
        for i in range(2, n+1):
            for j in range(i):
                dCatalan[i] += dCatalan[j] * dCatalan[i-j-1]
        return dCatalan[n]

    @staticmethod
    def catalan_number_binomial(n: int) -> int:
        """
        Calculate catalan number using binomial coefficient
        O(n) time complexity
        :param n: Nth catalan number in series
        :return: Value of Nth catalan number
        """
        val = Solution.binomial_coefficient_dynamic(2*n,n)
        return val / (n+1)

    @staticmethod
    def common_substring_between_two_strings(s1: str, s2: str) -> bool:
        """
        Given two strings, determine if they share at least one common substring.
        Substring can be as small as single character
        :param s1: first string
        :param s2: second string
        :return: boolean indicating if they share a common substring
        """
        d = defaultdict(lambda: None)
        for char in s1:
            if d[char]:
                d[char] += 1
            else:
                d[char] = 1
        for char in s2:
            if d[char]:
                return True

        return False

    @staticmethod
    def min_edit_distance(s1: str, s2: str) -> int:
        """
        Given two strings, determine the edit distance between them.
        The edit distance is defined as the minimum number of edits (insertion, deletion, or substitution)
        needed to change one string to the other
        :param s1: first string
        :param s2: second string
        :return:  minimum number of edits to change s1 into s2 and viceversa
        """
        size1, size2 = len(s1), len(s2)
        dp = [[0 for x in range(size2+1)] for x in range(size1+1)]
        for i in range(size1+1):
            for j in range(size2+1):
                if i == 0:
                    dp[i][j]=j
                elif j == 0:
                    dp[i][j] = i
                elif s1[i-1] == s2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(dp[i][j-1], dp[i-1][j], dp[i-1][j-1])
        return dp[size1][size2]

    @staticmethod
    def evaluate_math_expr(expr: str) -> int:
        operators = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.USub: operator.neg
        }

        def eval_(node):
            if isinstance(node, ast.Num):  # Case of number
                return node.n
            if isinstance(node, ast.BinOp):  # Binary operator <left> <operator> <right>
                return operators[type(node.op)](eval_(node.left), eval_(node.right))
            if isinstance(node, ast.UnaryOp): # Unary operator. e.g. -1
                return operators[type(node.op)](eval_(node.operand))
            else:
                raise TypeError(node)

        return eval_(ast.parse(expr, mode='eval').body)

    @staticmethod
    def staircase_with_two_steps(n: int) -> int:
        """
        Given a positive integer N, representing the number of steps in a staircase,
        you can either climb 1 or 2 steps at a time.
        Calculate the number of unique ways to climb the stairs
        :param n: Positive integer representing number of steps in staricase
        :return: integer representing unique ways to climb the stairs
        """
        # We can use Fibonacci sequence here because number of unique ways to climb the stairs for
        # Xn is given by Xn = X(n-1) + X(n-2)
        # We use DP to optimize the solution. Basic Fibonacci sequence using DP will suffice
        dp = [1,1]
        for i in range(2, n+1):
            dp.append(dp[i-1] + dp[i-2])
        return dp[n]

    @staticmethod
    def staircase_with_three_steps(n: int) -> int:
        """
        Given a positive integer N, representing the number of steps in a staircase,
        you can either climb 1, 2 or 3 steps at a time.
        Calculate the number of unique ways to climb the stairs
        :param n: Positive integer representing number of steps in staricase
        :return: integer representing unique ways to climb the stairs
        """
        # We can use Fibonacci sequence here because number of unique ways to climb the stairs for
        # Xn is given by Xn = X(n-1) + X(n-2)
        # We use DP to optimize the solution. Basic Fibonacci sequence using DP will suffice
        dp = [1,1,2]
        for i in range(3, n + 1):
            dp.append(dp[i - 1] + dp[i - 2] + dp[i-3])
        return dp[n]

    @staticmethod
    def min_cost_climbing_stairs(cost: list) -> int:
        """
        Given an array of costs of each step in a staircase.
        Once you pay the cost, you can either climb one or two steps.
        You can either start form the step with index 0 or the step with index 1
        :param cost: array of i steps with cost to take the step
        :return: min cost to climb the staircase
        """
        # Base cases
        if not cost:
            return None
        size = len(cost)
        if size == 1:
            return cost[0]
        if size == 2:
            return min(cost[0], cost[1])

        # Generate empty tail
        cost.append(0)
        for i in range(2, size + 1):
            cost[i] = cost[i] + min(cost[i - 1], cost[i - 2])

        return min(cost[size - 1], cost[size - 2])

    @staticmethod
    def isSubsequence(arr: list, sub: list) -> bool:
        """
        Given two array of non negative integers,
        determine whether the second is a sub sequence of the first one
        :param arr: first array of non negative integers
        :param sub: second array of non negative integeres
        :return: boolean indicating if second array is sub sequence of first one
        """
        # O(n) Time and O(1) Space
        if not arr and not sub:
            return True
        firstIdx = 0
        secondIdx = 0
        while firstIdx < len(arr) and secondIdx < len(sub):
            if arr[firstIdx] == sub[secondIdx]:
                secondIdx += 1
            firstIdx += 1
        return secondIdx == len(sub)

    @staticmethod
    def remove_last_occurrence_of_word(s: str, w: str) -> str:
        """
        Given two strings S and W of sizes N and M respectively,
        the task is to remove the last occurrence of W from S.
        If there is no occurrence of W in S, print S as it is
        :param s: whole string to be searched
        :param w: word to be deleted
        :return: string without last occurence of word
        """
        # O(N*M) Time and O(1) Space
        s = [i for i in s]
        w = [i for i in w]
        n = len(s)
        m = len(w)
        # if word is of greater size than string
        if m > n:
            return s
        # Iterate while i >= 0
        for i in range(n-m, -1, -1):
            flag = 0
            for j in range(m):
                # ff s[j + 1] != w[j], mark flag true and break
                if s[j + i] != w[j]:
                    flag = 1
                    break
            # If occurrence has been found
            if flag == 0:
                # Delete the subover the range [i, i+M]
                for j in range(i, n - m):
                    s[j] = s[j + m]

                # Resize the S
                s = s[:n - m]
                break
        return "".join(s)

    @staticmethod
    def max_length_upper_boundary_placing_rectangles_horizontally_or_vertically(n: int, v: list) -> int:
        """
        Given a vector of pairs,  V[] denoting the width and height of N rectangles numbered from 1 to N,
        these rectangles are placed in contact with the horizontal axis and are adjacent from left to right in numerical order.
        The task is to find the maximum length of the upper boundary formed by placing each of the rectangles
        either horizontally or vertically.
        :param n: integer representing number of rectangles
        :param v: list with rectangles (each rectangle is a set (length, height)
        :return: max length of upper boundary
        """
        # Stores the intermediate
        # transition states
        dp = [[0 for i in range(2)] for j in range(n)]

        # Place the first rectangle
        # horizontally
        dp[0][0] = v[0][0]

        # Place the first rectangle
        # vertically
        dp[0][1] = v[0][1]

        for i in range(1, n):
            # Place horizontally
            dp[i][0] = v[i][0]

            # Stores the difference in height of
            # current and previous rectangle
            height1 = abs(v[i - 1][1] - v[i][1])
            height2 = abs(v[i - 1][0] - v[i][1])

            # Take maximum out of two options
            dp[i][0] += max(height1 + dp[i - 1][0], height2 + dp[i - 1][1])

            # Place Vertically
            dp[i][1] = v[i][1]

            # Stores the difference in height of
            # current and previous rectangle
            vertical1 = abs(v[i][0] - v[i - 1][1]);
            vertical2 = abs(v[i][0] - v[i - 1][1]);

            # Take maximum out two options
            dp[i][1] += max(vertical1 + dp[i - 1][0], vertical2 + dp[i - 1][1])

        # Print maximum of horizontal or vertical
        # alignment of the last rectangle
        return max(dp[n - 1][0], dp[n - 1][1]) - 1

    @staticmethod
    def max_subset_sum_divisible_at_most_k_elements(arr: list, n: int, k: int, d: int) -> int:
        """
        Given an array A[] of size N, and two numbers K and D, the task is to calculate the maximum subset-sum
        divisible by D possible by taking at most K elements from A.
        :param arr: array of integers
        :param n: size of array
        :param k: max size of subset
        :param d: divisor of subset sum
        :return: max sum of subset with at most k elements divisible by d
        """
        # Use dp to store the maximum sum possible if j elements are taken till the ith index and its modulo D is p
        # O(NKD) Time and O(NKD) Space
        dp = [[[-1 for _ in range(d+1)] for _ in range(k+1)] for _ in range(n+1)]
        for i in range(n+1):
            curr = arr[i-1]
            mod = curr % d
            dp[i] = dp[i-1]
            for j in range(k+1):
                dp[i][j][mod] = max(dp[i][j][mod], curr)
                for m in range(d):
                    if dp[i-j][j-1][m] != -1:
                        dp[i][j][(m+mod) % d] = max(dp[i][j][(m + mod) % d], dp[i - 1][j - 1][m] + curr)
        if dp[n][k][0] == -1:
            return 0
        return dp[n][k][0]

    @staticmethod
    def max_subsequence_sum_without_3_consecutive(arr: list, n: int) -> int:
        """
        Given an array A[] of N positive numbers, the task is to find the maximum sum that can be formed
        which has no three consecutive elements present.
        :param arr: list of positive integers
        :param n: size of array
        :return: max sum without three consecutive elements
        """
        if not arr:
            return 0
        # Using dp, O(n) Time and O(1) Space
        if n == 1:
            return arr[0]
        if n == 2:
            return sum(arr)
        # var to store sum up to i - 3
        third = arr[0]
        # var to store sum up to i - 2
        second = third + arr[1]
        # var to store sum up to i - 1
        first = max(second, arr[1] + arr[2])

        # var to store sum
        s = max(max(third, second), first)

        for i in range(3, n):
            # find the maximum subsequence sum up to index i
            s = max(max(first, second + arr[i]), third + arr[i] + arr[i - 1])
            # update first, second and third
            third = second
            second = first
            first = s

        return s

    @staticmethod
    def divide_chocolate_bar(n: int, m: int, arr: list) -> int:
        """
        Given a 2d array, arr[][] and a piece of the chocolate bar of dimension N × M,
        the task is to find the minimum possible sum of the area of invalid pieces by dividing the chocolate bar into
        one or more pieces where a chocolate piece is called invalid
        if the dimension of that piece does not match any given pair.

        Note: A chocolate piece can be cut vertically or horizontally (perpendicular to its sides),
        such that it is divided into two pieces and the dimension in the given vector is not ordered
        i.e. for a pair (x, y) in the given vector both dimensions (x, y) and (y, x) are considered valid.
        :param n: length of chocolate bar
        :param m: height of chocolate bar
        :param arr: 2D array containing
        :return: minimum possible sum of the area of invalid pieces
        """
        # https://www.geeksforgeeks.org/divide-chocolate-bar-into-pieces-minimizing-the-area-of-invalid-pieces/
        return None

    @staticmethod
    def knapsack_whole(maxW: int, wt: list, val: list, n: int) -> int:
        """
        Given weights and values of n items, put these items in a knapsack of capacity W to get the maximum total value
        in the knapsack. In other words, given two integer arrays val[0..n-1] and wt[0..n-1]
        which represent values and weights associated with n items respectively. Also given an integer W which
        represents knapsack capacity, find out the maximum value subset of val[] such that sum of the weights of this
        subset is smaller than or equal to W. You cannot break an item, either pick the complete item or don’t pick it
        :param maxW: Maximum supported by knapsack
        :param wt: array containing weight of each item
        :param val: array containing values of each item
        :param n: size of arrays
        :return: maximum sum of values available for that knapsack
        """
        # Time Complexity: O(N*W)
        # Auxiliary Space: O(W)
        dp = [0] * (maxW + 1)  # Create dp array
        for i in range(1, n+1):
            for j in range(maxW, 0, -1):  # reverse so that we also have data of previous computation
                if wt[i-1] <= j:
                    dp[j] = max(dp[j], dp[j-wt[i-1]]+val[i-1])  # find max value
        return dp[maxW]  # return max value of knapsak

    @staticmethod
    def maxSlidingWindow(nums: list, k: int) -> list:
        """
        given an array of integers nums, there is a sliding window of size k which is moving from the very left of the
        array to the very right. You can only see the k numbers in the window. Each time the sliding window moves right
        by one position.
        :param nums: array of integers
        :param k: size of sliding windows
        :return: array cointaining the max number of each window
        """
        h, resp = [], []
        for i in range(k):
            heapq.heappush(h, (-nums[i], i))
        resp.append(-h[0][0])
        for i in range(k, len(nums)):
            heapq.heappush(h, (-nums[i], i))
            while h and (i - h[0][1]) >= k:
                heapq.heappop(h)
            resp.append((-h[0][0]))
        return resp

    @staticmethod
    def travelling_salesman_problem(graph, v, currPos, n, count, cost):
        """
        Given a set of cities and distance between every pair of cities, the problem is to find the shortest possible
        route that visits every city exactly once and returns to the starting point.
        """
        # If last node is reached and it has
        # a link to the starting node i.e
        # the source then keep the minimum
        # value out of the total cost of
        # traversal and "ans"
        # Finally return to check for
        # more possible values
        answer = []
        if count == n and graph[currPos][0]:
            answer.append(cost + graph[currPos][0])
            return

        # BACKTRACKING STEP
        # Loop to traverse the adjacency list
        # of currPos node and increasing the count
        # by 1 and cost by graph[currPos][i] value
        for i in range(n):
            if not v[i] and graph[currPos][i]:
                # Mark as visited
                v[i] = True
                graph.travelling_salesman_problem(graph, v, i, n, count + 1,
                                                  cost + graph[currPos][i])

                # Mark ith node as unvisited
                v[i] = False

    @staticmethod
    def rank_teams(votes: list) -> str:
        if not votes:
            return ''
        # If only one vote or only one team
        if len(votes) == 1 or len(votes[0]) == 1:
            return votes[0]

        d = defaultdict(lambda: [])
        for i, c in enumerate(sorted(votes[0])):
            d[c] = i
            d[i] = c
        rank = [[i] + [0] * len(votes[0]) for i in range(len(votes[0]))]
        for vote in votes:
            for i, c in enumerate(vote):
                rank[d[c]][i+1] -= 1
        rank.sort(key=lambda row: row[1:])
        return "".join(d[r[0]] for r in rank)

    @staticmethod
    def minimumMountainRemovals(nums: list) -> int:
        """
        Given an integer array nums, return the minimum number of elements to remove to make nums a mountain array.
        """
        # O(NlogN) Time and O(N) Space
        def recursive_helper(nums):
            """
            Return length of LIS (excluding x) ending at x
            """
            dp = [10**10] * (len(nums)+1)
            lens = [0] * len(nums)
            for i, elem in enumerate(nums):
                lens[i] = bisect.bisect_left(dp, elem) + 1
                dp[lens[i]-1] = elem
            return lens

        left, right = recursive_helper(nums), recursive_helper(nums[::-1])[::-1]
        ans, n = 0, len(nums)
        for i in range(n):
            if left[i] >= 2 and right[i] >= 2:
                ans = max(ans, left[i] + right[i] - 1)
        return n - ans

    @staticmethod
    def ways_cut_pizza(pizza: list, k: int) -> int:
        """
        Given a rectangular pizza represented as a rows x cols matrix containing the following characters:
        'A' (an apple) and '.' (empty cell) and given the integer k.
        You have to cut the pizza into k pieces using k-1 cuts.
        For each cut you choose the direction: vertical or horizontal, then you choose a cut position at the cell
        boundary and cut the pizza into two pieces. If you cut the pizza vertically, give the left part of the pizza
        to a person. If you cut the pizza horizontally, give the upper part of the pizza to a person.
        Give the last piece of pizza to the last person.
        Return the number of ways of cutting the pizza such that each piece contains at least one apple.
        Since the answer can be a huge number, return this modulo 10^9 + 7.
        """
        m, n = len(pizza), len(pizza[0])
        prefix = [[0]*(n+1) for _ in range(m+1)] # prefix array
        for i in range(m):
            for j in range(n):
                prefix[i+1][j+1] = prefix[i][j+1] + prefix[i+1][j] - prefix[i][j]
                if pizza[i][j] == "A":
                    prefix[i+1][j+1] += 1

        def helper(i, j, k):
            """
            Return number of ways of cutting pizza[i:][j:] for k people
            """
            if i == m or j == n:
                return 0
            apples = prefix[-1][-1] - prefix[-1][j] - prefix[i][-1] + prefix[i][j]
            if apples < k+1:
                return 0
            if k == 0:
                return 1
            ans = 0
            for ii in range (i, m):
                if prefix[ii + 1][-1] - prefix[ii + 1][j] - prefix[i][-1] + prefix[i][j]:
                    ans += helper(ii + 1, j, k - 1)
            for jj in range(j, n):
                if prefix[-1][jj + 1] - prefix[-1][j] - prefix[i][jj + 1] + prefix[i][j]:
                    ans += helper(i, jj + 1, k - 1)
            return ans % 1_000_000_007

        return helper(0,0,k-1)

    @staticmethod
    def minimum_window_substring(s: str, t: str) -> str:
        """
        Given two strings s and t of lengths m and n respectively, return the minimum window substring of s such that
        every character in t (including duplicates) is included in the window.
        If there is no such substring, return the empty string ""
        """
        return ''

    @staticmethod
    def lenLongestGibSubseq(arr):
        """
        Given a strictly increasing array arr of positive integers forming a sequence, return the length of the longest
        Fibonacci-like subsequence of arr. If one does not exist, return 0.
        """
        # O(N**2 log(M))
        size = len(arr)
        lastElem = arr[-1]

        def recursive_helper(a, index, last2, last):
            if index >= size or last2 > lastElem:
                return 0
            if a<2:
                return max(1+recursive_helper(a+1, index+1, last2+arr[index], arr[index]), recursive_helper(a, index+1, last2, last))
            else:
                pos = bisect.bisect_left(arr, last2, lo=index)
                if pos < size and arr[pos] == last2:
                    return (1+recursive_helper(a+1, pos+1, last+arr[pos], arr[pos]))
                else:
                    return 0

        a = recursive_helper(0, 0, 0, 0)
        if a < 3:
            return 0
        else:
            return a

    @staticmethod
    def image_overlap(A: list, B: list) -> int:
        """
        You are given two images img1 and img2 both of size n x n, represented as binary, square matrices of the
        same size. (A binary matrix has only 0s and 1s as values.) We translate one image however we choose
        (sliding it left, right, up, or down any number of units), and place it on top of the other image.
        After, the overlap of this translation is the number of positions that have a 1 in both images.
        """
        A_points, B_points, d = [], [], defaultdict(int)

        # Filter points having 1 for each matrix respectively
        for i in range(len(A)):
            for j in range(len(A[0])):
                if A[i][j]:
                    A_points.append((i,j))
                if B[i][j]:
                    B_points.append((i,j))

        # For every point in filtered A, calculate the linear transformation vector with all points of filtered B
        # count the number of the pairs that have the same transformation vector

        for r_a, c_a in A_points:
            for r_b, c_b in B_points:
                d[(r_b - r_a, c_b - c_a)] += 1

        return max(d.values() or [0])

    @staticmethod
    def max_number_lectures(arrival, duration):
        """
        Given an array containing time of arrival and another array containing duration of stay,
        determine how many lectures can be made in single room without two occuring at same time
        :param arrival: array containing time of arrival for each participant
        :param duration: array containing duration of each participant's lecture
        :return: max number of lectures that can occur in a single day
        """
        # O(nlog(n)) Time
        ans = 0

        # Sorting of meeting according to their finish time.
        zipped = zip(arrival, duration)
        zipped = list(zipped)
        zipped.sort(key=lambda x: x[0] + x[1])  # O(nlog(n))

        # Initially select first meeting
        ans += 1
        # time_limit to check whether new
        # meeting can be conducted or not.
        time_limit = zipped[0][0] + zipped[0][1]

        # Check for all meeting whether it
        # can be selected or not.
        for i in range(1, len(arrival)):
            if zipped[i][0] > time_limit:
                ans += 1
                time_limit = zipped[i][0] + zipped[i][1]

        return ans

    @staticmethod
    def evaluate_bracket_pairs(s: str, knowledge: list) -> str:
        """
        Given a string with the format '(name) is mike' and a list containing tuples meant for a key value pair in hashm
        where [['name', mike']], return the evaluation of the string 'mike is mike' resulting from checking key inside
        parentheses. If key is not found, print ? instead. i.e. '? is mike'
        :param s: string to evaluate
        :param knowledge: known key pairs
        :return: evaluated string
        """
        # Create dictionary
        d = {}
        for pair in knowledge:  # O(k) where k = len(knowledge)
            d[pair[0]] = pair[1]

        output = ''
        stack = []
        use_stack = False
        for i in range(len(s)):
            if s[i] == "(":
                use_stack = True
            elif s[i] == ")":
                key = ''.join(stack)
                if key in d:
                    output += d[key]
                else:
                    output += "?"
                stack = []
                use_stack = False
            else:
                if use_stack:
                    stack.append(s[i])
                else:
                    output += s[i]
        return output

    def min_product_subset_in_array(self, arr: list) -> int:
        """
        Given an array a, we have to find the minimum product possible with the subset of elements present in the array.
        The minimum product can be a single element also.
        :param arr:
        :param n:
        :return:
        """
        # O(n) Time and O(1) Space
        if not arr:
            return None

        n = len(arr)
        if (n == 1):
            return arr[0]

        # Find count of negative numbers,
        # count of zeros, maximum valued
        # negative number, minimum valued
        # positive number and product
        # of non-zero numbers
        max_neg = float('-inf')
        min_pos = float('inf')
        count_neg = 0
        count_zero = 0
        prod = 1
        for i in range(0, n):

            # If number is 0, we don't
            # multiply it with product.
            if arr[i] == 0:
                count_zero += 1
                continue

            # Count negatives and keep
            # track of maximum valued
            # negative.
            if arr[i] < 0:
                count_neg += 1
                max_neg = max(max_neg, arr[i])

            # Track minimum positive
            # number of array
            if arr[i] > 0:
                min_pos = min(min_pos, arr[i])

            prod = prod * arr[i]

        # If there are all zeros or no negative number present
        if count_zero == n or (count_neg == 0 and count_zero > 0):
            return 0

        # If there are all positive
        if count_neg == 0:
            return min_pos

        # If there are even number of
        # negative numbers and count_neg
        # not 0
        if ((count_neg & 1) == 0 and
                count_neg != 0):
            # Otherwise result is product of
            # all non-zeros divided by
            # maximum valued negative.
            prod = int(prod / max_neg)

        return prod

    def max_product_subset_in_array(self, arr: list) -> int:
        """
        Given an array a, we have to find maximum product possible with the subset of elements present in the array.
        The maximum product can be single element also.
        """
        # O(n) Time and O(1) Space
        n = len(arr)
        # Find count of negative numbers, count
        # of zeros, negative number
        # with least absolute value
        # and product of non-zero numbers
        max_neg = -999999999999
        count_neg = 0
        count_zero = 0
        prod = 1
        for i in range(n):

            # If number is 0, we don't
            # multiply it with product.
            if arr[i] == 0:
                count_zero += 1
                continue

            # Count negatives and keep
            # track of negative number
            # with least absolute value.
            if arr[i] < 0:
                count_neg += 1
                max_neg = max(max_neg, arr[i])

            prod = prod * arr[i]

        # If there are all zeros
        if count_zero == n:
            return 0

        # If there are odd number of
        # negative numbers
        if count_neg & 1:

            # Exceptional case: There is only
            # negative and all other are zeros
            if (count_neg == 1 and count_zero > 0 and
                    count_zero + count_neg == n):
                return 0

            # Otherwise result is product of
            # all non-zeros divided
            # by negative number
            # with least absolute value
            prod = int(prod / max_neg)

        return prod

    def isLongPressedName(self, name: str, typed: str) -> bool:
        # cover the case: that the name and typed are equal, e.g., "laiden", "laiden"
        if name == typed:
            return True

        # Note that it is necessary to have both strings ending the same. E.g., "alex", "alexxr"

        if typed[-1] != name[-1]:
            return False

        # Index of the name char in string typed, the very first time iteration it's always zero
        idx = -1

        # Memorize the previous char in string name, starting from the first char in string name
        prev = name[0]

        # Iterate the string name. Make the decision on the fly.
        for char in name:
            if (char in typed):
                idx = typed.index(char)
                if idx > 0:
                    for i in range(0, idx):
                        if typed[i] != prev:
                            return False
                typed = typed[(idx + 1):]
                prev = char
            else:
                return False
        return True


class TopVotedCandidate:
    def __init__(self, persons: list, times: list):
        self.time_winning = defaultdict(lambda: -1)
        vote_count = defaultdict(int)
        curr_max, curr_win = 0, -1
        for p, t in zip(persons, times):
            vote_count[p] += 1
            if vote_count[p] >= curr_max:
                curr_max, curr_win = vote_count[p], p
            self.time_winning[t] = curr_win

    def q(self, t: int) -> int:
        return self.time_winning[self.times[bisect.bisect_right(self.times, t)-1]]
