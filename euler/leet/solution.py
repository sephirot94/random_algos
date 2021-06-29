import math
import ast
import operator
import re
import functools as fct
from collections import defaultdict

class Solution:

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
            sum = fct.reduce(lambda x, y: x + y, factors) - num
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

    def sort_nums(self, l: list) -> list:
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
        buy = 0
        sell = 0
        max_profit = -1
        search_buy_candidate = True
        for i, stock in enumerate(stocks):
            if i+1 == len(stocks):
                break
            sell = stocks[i+1]
            if search_buy_candidate:
                buy = stock
            if sell < buy:
                search_buy_candidate = True
                continue
            else:
                temp = sell - buy
                if temp > max_profit:
                    max_profit = temp
                    search_buy_candidate = False


        return max_profit

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

        return edits

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
    def staircase(n: int) -> int:
        """
        Given a positive integer N, representing the number of steps in a staircase,
        you can either climb 1 or 2 steps at a time.
        Calculate the number of unique ways to climb the stairs
        :param n: Positive integer representing number of steps in staricase
        :return: integer representing unique ways to climb the stairs
        """
        



