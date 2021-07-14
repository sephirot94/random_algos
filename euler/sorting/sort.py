class QuickSort:
    # Best Case: O(nlogn)
    # Worst Case: O(n)
    def partition(self, arr, low, high):
        i = (low - 1)  # index of smaller element
        pivot = arr[high]  # pivot

        for j in range(low, high):

            # If current element is smaller than or
            # equal to pivot
            if arr[j] <= pivot:
                # increment index of smaller element
                i = i + 1
                arr[i], arr[j] = arr[j], arr[i]

        arr[i + 1], arr[high] = arr[high], arr[i + 1]
        return (i + 1)

    def quickSort(self, arr, low, high):
        if low < high:
            # pi is partitioning index, arr[p] is now
            # at right place
            pi = self.partition(arr, low, high)

            # Separately sort elements before
            # partition and after partition
            self.quickSort(arr, low, pi - 1)
            self.quickSort(arr, pi + 1, high)

class MergeSort:
    # Time Complexity : O(nlogn)
    def merge(self, Arr, start, mid, end):
        """
        merge function take two intervals one from start to mid second from mid+1, to end and merge them in sorted order
        :param Arr: array of integer type
        :param start: starting index of current interval
        :param mid: mid index od current interval
        :param end: end index of current interval
        """
        # create a temp array
        temp = [0] * (end - start + 1)

        # crawlers for both intervals and for temp
        i, j, k = start, mid + 1, 0

        # traverse both lists and in each iteration add smaller of both elements in temp
        while (i <= mid and j <= end):
            if (Arr[i] <= Arr[j]):
                temp[k] = Arr[i]
                k += 1;
                i += 1
            else:
                temp[k] = Arr[j]
                k += 1;
                j += 1

        # add elements left in the first interval
        while (i <= mid):
            temp[k] = Arr[i]
            k += 1;
            i += 1

        # add elements left in the second interval
        while (j <= end):
            temp[k] = Arr[j]
            k += 1;
            j += 1

        # copy temp to original interval
        for i in range(start, end + 1):
            Arr[i] = temp[i - start]

    def mergeSort(self, Arr, start, end):
        if (start < end):
            mid = (start + end) / 2
        self.mergeSort(Arr, start, mid)
        self.mergeSort(Arr, mid + 1, end)
        self.merge(Arr, start, mid, end)

class InsertionSort:
    # Best Case: O(n)
    # Worst Case: O(n**2)
    def insertionSort(self, arr):
        # Traverse through 1 to len(arr)
        for i in range(1, len(arr)):
            key = arr[i]
            # Move elements of arr[0..i-1], that are
            # greater than key, to one position ahead
            # of their current position
            j = i - 1
            while j >= 0 and key < arr[j]:
                arr[j + 1] = arr[j]
                j -= 1
            arr[j + 1] = key

class SelectionSort:
    # Best Case: O(n)
    # Worst Case: O(n**2)
    def findMinIndex(self, arr: list, start: int) -> int:
        min_index = start

        start += 1

        while start < len(arr):
            if arr[start] < arr[min_index]:
                min_index = start

            start += 1

        return min_index

    def selectionSort(self, arr):
        i = 0

        while i < len(arr):
            min_index = self.findMinIndex(arr, i)
            if i != min_index:
                arr[i], arr[min_index] = arr[min_index], arr[i]

            i += 1
