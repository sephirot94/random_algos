import math
from collections import defaultdict
from trees.graph import CheckStronglyConnected
from trees.binary_search import BST
from leet.google import Google, FindIslands
from heap.heapster import MinHeap
from leet.solution import Solution
from dynamic.dynamic import compute_series
from searching.pattern import KMPPatternSearch, FiniteAutomataPatternSearching, AhoCorasickPatternSearching

# Press the green button in the gutter to run the script.
from trees.tree import TreeNode

if __name__ == '__main__':

   # aho = AhoCorasickPatternSearching()
   # print(aho.search("aab", "aaaabaaabaaaaabaabaaabaaaaabbbaaaab"))

   # mtx = [[0,1,1,0], [1,0,0,1], [0,1,1,0], [1,0,0,0]]
   # # mh = MinHeap(arr)
   ggl = Google()
   root= TreeNode(5,
                  left=TreeNode(6,
                                left=TreeNode(2,
                                              left=TreeNode(1),
                                              right=TreeNode(3))),
                  right=TreeNode(8,
                                 left=TreeNode(7),
                                 right=TreeNode(9)))
   bst = BST(root)
   print(bst.largest_bst_in_binary_tree(root).__str__())
   # sol = Solution()
   # island_finder = FindIslands(mtx)
   # print(island_finder.find_islands())
   # print(ggl.product_array_except_index([1,2,3,4,5]))

   # g = CheckStronglyConnected(5)
   # g.add_edge(1, 0)
   # g.add_edge(0, 2)
   # g.add_edge(2, 1)
   # g.add_edge(0, 3)
   # g.add_edge(3, 4)
   # print(g.find_all_strongly_connected())
   # print(g.check_strongly_connected())
