from collections import defaultdict


class Play:

    def __init__(self):
        pass

    # @staticmethod
    # def is_balanced_parenthesis(word: str):
    #     """
    #     Given a string, returns true if it has balanced parenthesis
    #     :param word: input string
    #     :return: booleand indicating if string has valid parenthesis
    #     """
    #     stack = []
    #     for letter in word:
    #         if letter is "(" or letter is "[" or letter is "{":
    #             stack.append(letter)
    #         if letter is ")" or letter is "]" or letter is "}":
    #             prev = stack.pop()
    #             if letter is ")" and prev != "(":
    #                 return False
    #             if letter is "]" and prev != "]":
    #                 return False
    #             if letter is "}" and prev != "}":
    #                 return False
    #     return True

    def smallest_positive_integer_not_present(self, numbers: list[int]) -> int:
        """
        Returns smallest positive integer not present in input
        :param numbers: array containing positive integers
        :return: smallest positive integer not in array
        """

        # Create empty set to store numbers in array
        storage_set = set()
        for number in numbers:
            storage_set.add(number)
        size = len(numbers)
        for i in range(1, size + 1, 1):
            if i not in storage_set:
                return i
        return 1


    def max_product_subarray(self, arr: list[int]) -> int:
        """Given an integer array nums, find a subarray that has the largest product, and return the product"""
        if not arr:
            return 0
        ans = max_prod = min_prod = arr[0]

        for num in arr[1:]:
            max_temp = max_prod * num
            min_temp = min_prod * num

            max_prod = max(max_temp, min_temp, num)
            min_prod = min(max_temp, min_temp, num)

            ans = max(max_prod, ans)

        return ans



    def snake_can_pass(self, matrix: list[list[str]]) -> (list[int], list[int]):
        """
        Given a board with '0' and '+' determine if a snake can pass through an entire row or column. The snake can
        pass through any cell where there is a '0', and is blocked by the '+'. The board IS NOT necessarily a square
        Returns two collections in a tuple, the first element representing the rows and the second element representing the
        columns.
        """
        if not matrix:
            return [], []
        row_map = defaultdict(bool)
        column_map = defaultdict(bool)
        can_pass_rows, can_pass_columns = [], []
        for row_idx, row in enumerate(matrix):  # O(r*c)
            for col_idx, _ in enumerate(row):
                if matrix[row_idx][col_idx] == "+":
                    row_map[row_idx] = True
                    column_map[col_idx] = True

        for i in range(len(matrix)):  # O(r)
            if not column_map[i]:
                can_pass_columns.append(i)
            if not row_map[i]:
                can_pass_rows.append(i)

        return can_pass_rows, can_pass_columns
