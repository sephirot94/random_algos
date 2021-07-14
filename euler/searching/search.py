import math

class BinarySearch:
    # Time Complexity: O(logn)
    def binarySearch(self, arr, l, r, x):
        """
        Returns index of x in arr if present, else -1
        """
        # Check base case
        if r >= l:

            mid = l + (r - l) // 2

            # If element is present at the middle itself
            if arr[mid] == x:
                return mid

            # If element is smaller than mid, then it
            # can only be present in left subarray
            elif arr[mid] > x:
                return self.binarySearch(arr, l, mid - 1, x)

            # Else the element can only be present
            # in right subarray
            else:
                return self.binarySearch(arr, mid + 1, r, x)

        else:
            # Element is not present in the array
            return -1

class JumpingSearch:
    # Time: O(√n)
    def jumpSearch(self, arr, x, n):

        # Finding block size to be jumped
        step = math.sqrt(n)

        # Finding the block where element is
        # present (if it is present)
        prev = 0
        while arr[int(min(step, n) - 1)] < x:
            prev = step
            step += math.sqrt(n)
            if prev >= n:
                return -1

        # Doing a linear search for x in
        # block beginning with prev.
        while arr[int(prev)] < x:
            prev += 1

            # If we reached next block or end
            # of array, element is not present.
            if prev == min(step, n):
                return -1

        # If element is found
        if arr[int(prev)] == x:
            return prev

        return -1

class InterpolationSearch:
    # Time: O(logn)
    def interpolationSearch(self, arr, lo, hi, x):
        """
        Given a sorted array of n uniformly distributed values arr[], write a function to search for a particular
        element x in the array.
        """
        # Since array is sorted, an element present
        # in array must be in range defined by corner
        if (lo <= hi and x >= arr[lo] and x <= arr[hi]):

            # Probing the position with keeping
            # uniform distribution in mind.
            pos = lo + ((hi - lo) // (arr[hi] - arr[lo]) *
                        (x - arr[lo]))

            # Condition of target found
            if arr[pos] == x:
                return pos

            # If x is larger, x is in right subarray
            if arr[pos] < x:
                return self.nterpolationSearch(arr, pos + 1,
                                           hi, x)

            # If x is smaller, x is in left subarray
            if arr[pos] > x:
                return self.interpolationSearch(arr, lo,
                                           pos - 1, x)
        return -1

class ExponentialSearch:
    # Time: O(logn)
    def binarySearch(self, arr, l, r, x):
        """
        A recurssive binary search function returns location  of x in given array arr[l..r] is present, otherwise -1
        """
        if r >= l:
            mid = l + (r - l) / 2

            # If the element is present at
            # the middle itself
            if arr[mid] == x:
                return mid

            # If the element is smaller than mid,
            # then it can only be present in the
            # left subarray
            if arr[mid] > x:
                return self.binarySearch(arr, l,
                                    mid - 1, x)

            # Else he element can only be
            # present in the right
            return self.binarySearch(arr, mid + 1, r, x)

        # We reach here if the element is not present
        return -1

    def exponentialSearch(self, arr, n, x):
        """
        Returns the position of first occurrence of x in array
        """
        # IF x is present at first
        # location itself
        if arr[0] == x:
            return 0

        # Find range for binary search
        # j by repeated doubling
        i = 1
        while i < n and arr[i] <= x:
            i = i * 2

        # Call binary search for the found range
        return self.binarySearch(arr, i / 2, min(i, n - 1), x)