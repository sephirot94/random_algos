import math
from collections import defaultdict
from euler.trees.graph import CheckStronglyConnected
from euler.leet.google import Google, FindIslands
from euler.heap.heapster import MinHeap
from euler.leet.solution import Solution
from euler.dynamic.dynamic import compute_series
from euler.searching.pattern import KMPPatternSearch, FiniteAutomataPatternSearching, AhoCorasickPatternSearching

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

   aho = AhoCorasickPatternSearching()
   print(aho.search("aab", "aaaabaaabaaaaabaabaaabaaaaabbbaaaab"))


   # arr = [1,4,2,5,3,6,9,7]
   # mtx = [[0,1,1,0], [1,0,0,1], [0,1,1,0], [1,0,0,0]]
   # # mh = MinHeap(arr)
   # ggl = Google()
   # sol = Solution()
   # island_finder = FindIslands(mtx)
   # print(island_finder.find_islands())
   # print(compute_series(100))

   # g = CheckStronglyConnected(5)
   # g.add_edge(1, 0)
   # g.add_edge(0, 2)
   # g.add_edge(2, 1)
   # g.add_edge(0, 3)
   # g.add_edge(3, 4)
   # print(g.find_all_strongly_connected())
   # print(g.check_strongly_connected())
