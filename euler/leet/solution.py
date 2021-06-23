import math
import functools as fct

class Solution:

    def two_sum(self, nums, target):
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
        if num <= 0: return False
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

    def isBalanced(self, s: str) -> bool:
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


