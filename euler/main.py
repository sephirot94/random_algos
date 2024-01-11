from leet.playing import Play
from leet.google import Google
from collections import defaultdict, Counter
import re


def reverse_words(words: str) -> str:
    """Reverses the words of a given string and returns it. Words are separated by a whitespace"""
    word_list = words.split(" ")
    word_list = [word_list[i] for i in range(len(word_list)-1, -1, -1)]
    return " ".join(word_list)


def unique_occurrences(arr: list[int]) -> bool:
    comp = set()
    d = defaultdict(int)
    for num in arr:
        d[num] += 1
    for num in d.values():
        if num in comp:
            return False
        comp.add(num)
    return True


def common_chars(words: list[str]) -> list[str]:
    counter_list = [Counter(word) for word in words]
    ans = []
    base = counter_list[0]
    for k in base.keys():
        m = base[k]
        for counter in counter_list[1:]:
            m = min(m, counter.get(k, 0))
        ans += [k] * m
    return ans


def max_vowels(s: str, k: int) -> int:
    """Return the amount of vowels in the substring of length K inside S which has the most vowels"""
    m = 0
    for i in range(len(s) - k):
        vowels = re.findall(r'[aeiouAEIOU]', s[i:i + k+1])
        m = max(m, len(vowels))
    return m


def max_area(height: list[int]) -> int:
    """
    You are given an integer array height of length n. There are n vertical lines drawn such that the two endpoints of
    the ith line are (i, 0) and (i, height[i]).
    Find two lines that together with the x-axis form a container, such that the container contains the most water.
    Return the maximum amount of water a container can store.
    """
    def calculate_area(b: int, h: int) -> int:
        return b * h

    l, r = 0, len(height) - 1
    m = 0
    while l != r:
        m = max(m, calculate_area(r - l, min(height[l], height[r])))
        if height[l + 1] >= height[r - 1]:
            l += 1
        else:
            r -= 1
    return m


def max_average_subarray(nums: list[int], k: int) -> float:
    """Returns the maximum average of a subarray of length K"""
    if len(nums) < k or k == 0:
        return 0
    if len(nums) == k:
        return avg(nums)
    m = 0
    for i in range(len(nums)-k):
        m = max(avg(nums[i:i+k]), m)
    return m


def intersection_three_sorted_arrays(arr1: list, arr2: list, arr3: list) -> list:
    """Given three sorted arrays, return an array containing the intersection of the three arrays, including repeated"""
    if not arr1:
        return []
    ans = []
    for i, elem in enumerate(arr1):
        try:
            if elem == arr2[i] and elem == arr3[i]:
                ans.append(elem)
        except IndexError:
            break
    return ans


if __name__ == '__main__':
    arr = [1,8,6,2,5,4,8,3,7]
    s = "weallloveyou"
    # words = ["bella", "roller", "label"]
    arr1 = [1,2,3,4,5,1,2]
    arr2 = [1,1,2,2,3,5]
    arr3 = [1,5,4,2,3,1,2,5,3]
    # print(intersection_three_sorted_arrays(sorted(arr1), sorted(arr2), sorted(arr3)))
    # print(max_area(arr))
    print(max_vowels(s, 7))