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


   arr = [{'num': 2, 'char': 'a'}, {'num': 1, 'char': 'c'}, {'num': 3, 'char': 'b'}]
   # print(sorted(arr, key=lambda d: d['char'], reverse=True))
   g = Google()

   root = TreeNode('*',
                   left=TreeNode("+",
                                 left=TreeNode(3),
                                 right=TreeNode(2)),
                   right=TreeNode("+",
                                  left=TreeNode(4),
                                  right=TreeNode(5))
                   )
   t = Tree(root)
   print(1<<0)





