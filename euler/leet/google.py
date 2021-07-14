from trees.tree import TreeNode
import collections

class Node:
    def __init__(self, word="", depth=-1, children=None):
        self.word = word
        self.depth = depth

INT_MIN = -2**32
MTX_MAX = 20

class Google:

    def decompress_compress_string(self, arr: str) -> str:
        """
        Given a string in the format num[char] return the decompressed string resulting in num*char
        It can be given that we have nested strings like 3[2[a]b]
        :param arr: input string
        :return: decompressed string
        """
        closePos = {}
        st = []
        for i, c in enumerate(arr):
            if c == '[':
                st.append(i)
            elif c == ']':
                closePos[st.pop()] = i

        def dfs(l, r):
            num = 0
            ans = ""
            while l <= r:
                c = arr[l]
                if ord('0') <= ord(c) <= ord('9'):
                    num = num * 10 + ord(c) - ord('0')
                elif c == '[':
                    ans += num * dfs(l + 1, closePos[l] - 1)
                    num = 0
                    l = closePos[l]
                else:
                    ans += c
                l += 1
            return ans

        return dfs(0, len(arr) - 1)

    def find_binary_tree_max_sum(self, root: TreeNode) -> int:
        """
        Given a binary tree, find the max path sum. The path may start and end at any node in thre tree
        :param root: root of tree
        :return: max path sum
        """
        def recursive_helper(root: TreeNode):
            """
            Helper used for recursion
            """
            # Base Case
            if not root:
                return 0

            # l and r store maximum path sum going through left
            # and right child of root respectively
            l = recursive_helper(root.left)
            r = recursive_helper(root.right)

            # Max path for parent call of root. This path
            # must include at most one child of root
            max_single = max(max(l, r) + root.data, root.data)

            # Max top represents the sum when the node under
            # consideration is the root of the maxSum path and
            # no ancestor of root are there in max sum path
            max_top = max(max_single, l + r + root.data)

            # Static variable to store the changes
            # Store the maximum result
            recursive_helper.res = max(recursive_helper.res, max_top)

            return max_single

        recursive_helper.res = float("-inf")
        recursive_helper(root)
        return recursive_helper.res

    def find_max_path_between_leaves_binary_tree(self, root: TreeNode) -> int:
        """
        Given a binary tree in which each node element contains a number. Find the maximum possible sum from one leaf
        node to another. The maximum sum path may or may not go through root.
        """

        def recursive_helper(root, res):
            # Base Case
            if not root:
                return 0

            # Find maximumsum in left and righ subtree. Also
            # find maximum root to leaf sums in left and right
            # subtrees ans store them in ls and rs
            ls = recursive_helper(root.left, res)
            rs = recursive_helper(root.right, res)

            # If both left and right children exist
            if root.left and root.right:
                # update result if needed
                res[0] = max(res[0], ls + rs + root.data)

                # Return maximum possible value for root being
                # on one side
                return max(ls, rs) + root.data

            # If any of the two children is empty, return
            # root sum for root being on one side
            if not root.left:
                return rs + root.data
            else:
                return ls + root.data

        res = [INT_MIN]
        recursive_helper(root, res)
        return res[0]

    def count_paths_in_matrix(self, matrix: list) -> int:
        """
        Count the number of routes from start to finish moving down or right.
        """
        # Using memoization DP
        dp = [[]]
        def recursive_helper(matrix, n, m):
            if n < len(matrix) and m < len(matrix[0]):
                if n == len(matrix)-1 and len(matrix[0])-1:
                    dp[n][m] = 1
                    return dp[n][m]
                # Check dp array to see if we have found route for a particular cell. If result is not None, return it.
                if dp[n][m]:
                    return dp[n][m]
                # Store the sum of ways we can reach by right and bottom cell in the dp array and return result
                dp[n][m] = recursive_helper(matrix, n+1, m) + recursive_helper(matrix, n, m+1)
                return dp[n][m]
            return 0

        return recursive_helper(matrix, 0, 0)

    def word_ladder(self, beginWord: str, endWord:str, wordList: list) -> int:
        """
        Given two words, beginWord and endWord, and a dictionary wordList, return the number of words in the shortest
        transformation sequence from beginWord to endWord, or 0 if no such sequence exists.
        """
        beginNode = Node(beginWord, 1)
        queue = collections.deque()
        queue.append(beginNode)
        wordSet = set(wordList)

        while queue:
            curNode = queue.popleft()
            curWord, curDepth = curNode.word, curNode.depth
            # find children
            # iterate through wordList and find words that are off by 1 char
            wordList = list(wordSet)  # make copy so we can delete from wordSet while iterating
            for word in wordList:
                curIdx = 0
                numUnmatch = 0
                skipWord = False  # flag for whether we skip this word
                for char in word:
                    if curWord[curIdx] != char:
                        numUnmatch += 1
                        if numUnmatch == 2:
                            # we only want words that are off by 1 char
                            skipWord = True
                            break
                    curIdx += 1
                if skipWord:
                    # this word has more than one char difference, so skip it
                    skipWord = False
                    continue
                else:
                    # if it's endWord then we're done
                    if word == endWord:
                        return curDepth + 1
                    child = TreeNode(word, curDepth + 1)
                    queue.append(child)
                    # remove word from list
                    wordSet.remove(word)

        # not possible (e.g. the endWord isn't in list, or no path to endWord)
        return 0

    def longest_increasing_path_in_matrix(self, matrix: list):
        """
        Given a matrix of N rows and M columns. From m[i][j], we can move to m[i+1][j], if m[i+1][j] > m[i][j],
        or can move to m[i][j+1] if m[i][j+1] > m[i][j]. The task is print longest path length if we start from (0, 0).
        :param matrix: matrix of N rows and M columns
        """
        dp = [[-1 for i in range(MTX_MAX)] for i in range(MTX_MAX)]
        n = len(matrix)
        m = len(matrix[0])
        def recursive_helper(matrix, n, m, x, y):
            # If value not calculated yet.
            if dp[x][y] < 0:
                res = 0
                # If reach bottom left cell,
                # return 1.
                if (x == n - 1 and y == m - 1):
                    dp[x][y] = 1
                    return dp[x][y]

                # If reach the corner
                # of the matrix.
                if (x == n - 1 or y == m - 1):
                    result = 1

                # If value greater than below cell.
                if (x + 1 < n and matrix[x][y] < matrix[x + 1][y]):
                    result = 1 + recursive_helper(dp, matrix, n,
                                     m, x + 1, y)

                # If value greater than left cell.
                if (y + 1 < m and matrix[x][y] < matrix[x][y + 1]):
                    result = max(res, 1 + recursive_helper(dp, matrix, n,
                                                 m, x, y + 1))
                dp[x][y] = result
            return dp[x][y]
        return recursive_helper(matrix, n, m, 0, 0)

    def get_max_sum_leaves_to_root_path_binary_tree(self, root: TreeNode) -> int:
        """
        Given a Binary Tree, find the maximum sum path from a leaf to root
        """
        max_sum = 0
        target_leaf = None

        def print_path(root, target_leaf):
            """
            Print helper
            """

            # base case
            if not root:
                return False

            # return True if this node is the target_leaf
            # or target leaf is present in one of its
            # descendants
            if root == target_leaf or print_path(root.left, target_leaf) or print_path(root.right, target_leaf):
                print(root.val, end=" ")
                return True

            return False

        def recursive_helper(root: TreeNode, curr_sum: int):
            """
            This function Sets the target_leaf_ref to refer the leaf node of the maximum path sum. Also,
            returns the max_sum using max_sum_ref
            """
            if not root:
                return
            # Update current sum to hold sum of nodes on path from root to this node
            curr_sum += root.val
            # If this is a leaf node and path to this node had max sum so far, then make this node target_leaf
            if not root.left and not root.right:
                max_sum = curr_sum
                target_leaf = root

            # If this is not a leaf node, then recur down to find the target_leaf
            recursive_helper(root.left, curr_sum)
            recursive_helper(root.right, curr_sum)

        # Base case
        if not root:
            return 0

        recursive_helper(root, 0)
        print_path(root, target_leaf)
        return max_sum
