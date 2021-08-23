import collections
import math
from collections import defaultdict
from trees.graph import CheckStronglyConnected
from trees.binary_search import BST
from leet.google import Google, FindIslands, CustomAlphabet
from heap.heapster import MinHeap
from leet.solution import Solution
from dynamic.dynamic import compute_series
from searching.pattern import KMPPatternSearch, FiniteAutomataPatternSearching, AhoCorasickPatternSearching

# Press the green button in the gutter to run the script.
from trees.tree import TreeNode, Tree

if __name__ == '__main__':
    ggl = Google()
    sol = Solution()
    # print(sol.celebrity_finder([[0,1,0], [0,0,0], [0,1,0]]))

    print(sol.reverse_integer(534))
    print(sol.reverse_integer(2**32))