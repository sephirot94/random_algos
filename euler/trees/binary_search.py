import sys
from collections import deque

from trees.tree import Tree, TreeNode

# NOTES
# Inorder traversal of BST always produces sorted output
#

class BST(Tree):

    def __init__(self, root: TreeNode):
        self.root = root

    def search_value(self, value: int) -> TreeNode:
        """
        Given a value, return the node with such value.
        If not found, return None.
        :param value: value to be searched in tree
        :return: Node containing searched value
        """
        curr = self.root
        while curr:
            if curr.val == value:
                return curr
            elif curr.val < value:
                curr = curr.right
            elif curr.val > value:
                curr = curr.left
        return None

    def insert_value_recursive(self, value: int) -> TreeNode:
        """
        Insert a new node containing given value recursively
        :param value: value of node to be inserted
        :return: root value of new tree
        """
        def recursive_helper(root, value):
            if not root:
                return TreeNode(value)
            else:
                if root.val == value:
                    return root
                elif root.val < value:
                    root.right = recursive_helper(root.right, value)
                else:
                    root.left = recursive_helper(root.left, value)

        return recursive_helper(self.root, value)

    def insert_value_iterative(self, value: int):
        """
        Insert a new node containing a given value iteratively
        :param value: value of node to be inserted
        """
        newNode = TreeNode(value)
        stack = [self.root]
        curr = None
        while stack:
            node = stack.pop()
            curr = node
            if node.val < value:
                if node.right:
                    stack.append(node.right)
            if node.val > value:
                if node.left:
                    stack.append(node.left)

        if curr.val > value:
            curr.left = newNode
        if curr.val < value:
            curr.right = newNode

        return

    def delete_value(self, value: int):
        """
        Delete the node containing the value if already exists
        :param value: value of node to be deleted
        """
        def recursive_helper(node: TreeNode, value: int) -> TreeNode:
            """
            Recursive helper closure used in delete_value
            :param node: current node in recursion
            :param value: value being deleted
            :return: Recursive iteration inorder search of valuee
            """
            if not node:
                return None
            if value < node.val:
                node.left = recursive_helper(node.left, value)
            if value > node.val:
                node.right = recursive_helper(node.right, value)
            if value == node.val:
                # Node with one or no children
                if not node.left:
                    temp = node.right
                    node = None
                    return temp
                elif not node.right:
                    temp = node.left
                    node = None
                    return temp
                # Node with two children
                # Get Smallest element in right subtree
                temp = BST.min_value(node.right)
                node.val = temp.val
                node.right = recursive_helper(node.right, temp.val)

            return node

        return recursive_helper(self.root, value)

    def find_ceil_floor(self, k: int) -> (int, int):
        """
        Given an integer k and a binary search tree, find the floor and ceil of k.
        Floor Value Node: Node with the greatest data lesser than or equal to the key value.
        Ceil Value Node: Node with the smallest data larger than or equal to the key value.
        If either does not exist, then print them as None.
        :param k: int value to find floor and ceil of
        :return: tuple containing floor and ceiling of given k
        """
        # O(n) time and O(1) space solution
        if not self.root:
            return None, None
        floor, ceil = 0, 0
        curr = self.root
        while curr:
            if curr.val == k:
                ceil = curr.val
                floor = curr.val
                break
            if curr.val < k:
                floor = curr.val
                curr = curr.right
            else:
                ceil = curr.val
                curr = curr.left
        return floor, ceil

    def get_floor(self, key: int) -> int:
        """
        Given a Binary Search Tree and a number x, find floor of x in the given BST.
        Floor Value Node: Node with the greatest data lesser than or equal to the key value.
        :param key: integer owner of floor
        :return: floor node of integer searched
        """
        if not self.root:
            return -1
        curr = self.root
        floor = 0
        while curr:
            if curr.val == key:
                return curr.val
            if curr.val > key:
                curr = curr.left
            else:
                floor = curr.val
                curr = curr.right
        return floor

    def get_ceil(self, key: int) -> int:
        """
        Given a binary tree and a key(node) value, find the ceil value for that particular key value.
        Ceil Value Node: node with the smallest data larger than or equal to the key value.
        :param key: integer owner of ceil
        :return: ceil node value of key searched
        """
        if not self.root:
            return -1
        curr = self.root
        ceil = -1
        while curr:
            if curr.val == key:
                return curr.val
            if curr.val < key:
                curr = curr.right
            else:
                ceil = curr.val
                curr = curr.left
        return ceil

    def convert_bst_into_binary(self) -> TreeNode:
        """
        Given a binary search tree, convert it to a binary tree
        convert it to a Binary Tree such that every key of the original BST
        is changed to key plus sum of all greater keys in BST.
        :return: root node of new binary tree
        """
        # traverse in order reversed and you will find in descending order all nodes
        # O(n) time and O(1) space
        if self.root is None:
            return None
        curr = self.root
        stack = []
        s = 0
        while True:
            if curr:
                stack.append(curr)
                curr = curr.right
            elif stack:
                curr = stack.pop()
                s = s + curr.val
                curr.val = s
                curr = curr.left
            else:
                break
        return self.root

    def invert_bst_in_place(self) -> TreeNode:
        """
        Invert the binary tree in place.
        That is, all left children should become right children,
        and all right children should become left children.
        :return: root node of inverted tree
        """
        def invert_recursive_helper(root: TreeNode) -> TreeNode:
            """
            Recursive helper closure to invert tree in place
            :param root: root node of tree to invert
            :return: root node of inverted tree
            """
            if not root:
                return None
            tmp = root.left
            root.left = root.right
            root.right = tmp
            invert_recursive_helper(root.left)
            invert_recursive_helper(root.right)
            return root

        if not self.root:
            return None

        return invert_recursive_helper(self.root)

    def minDiffInBST(self, root: TreeNode) -> int:
        """
        Given the root of a Binary Search Tree (BST), return the minimum difference
        between the values of any two different nodes in the tree.
        :param root: root node of tree
        :return: minimum difference between two nodes
        """
        if not root:
            return None
        prev = None
        diff = sys.maxsize

        stack = []
        cur = root
        while cur or stack:
            while cur:
                stack.append(cur)
                cur = cur.left
            cur = stack.pop()
            if prev:
                diff = min(cur.val - prev, diff)
            prev = cur.val
            cur = cur.right
        return diff

    @staticmethod
    def make_BST_from_preorder_traversal(pre: list) -> TreeNode:
        """
        Given preorder traversal of a binary search tree, construct the BST.
        :param pre: preorder traversal of binary seearch tree
        :return: root of BST
        """
        # Base case
        if not pre:
            return None
        # create a stack and append first element of pre as root
        root = TreeNode(pre[0])
        s = [root]
        # Iterate through rest of the size-1 items of given preorder array
        for idx in range(1, len(pre)):
            temp = None
            # Keep on popping while the next value is greater than stack's top value
            while s and pre[idx] > s[-1]:  # this will peek the top element
                temp = s.pop()
            # Make this greater value as the right child and push it to the stack
            if temp:
                temp.right = TreeNode(pre[idx])
                s.append(temp.right)
        return root

    @staticmethod
    def min_value(node: TreeNode) -> TreeNode:
        """
        Return left-most value, or minimum
        :param node: root node to be traversed
        :return: minimum value in tree
        """
        curr = node
        while curr.left:
            curr = curr.left
        return curr

    @staticmethod
    def max_value(node: TreeNode) -> TreeNode:
        """
        Return right-most value, or maximum
        :param node: root node to be traversed
        :return: maximum value in tree
        """
        curr = node
        while curr.right:
            curr = curr.right
        return curr

    @staticmethod
    def unique_bst(n: int) -> int:
        """
        Function uses Catalan Number Series to find how many unique bsts can be made with n keys
        The number of binary search trees that will be formed with N keys can be calculated by simply evaluating the corresponding number in Catalan Number series.
        First few Catalan numbers for n = 0, 1, 2, 3, … are 1, 1, 2, 5, 14, 42, 132, 429, 1430, 4862, …
        Catalan numbers satisfy the following recursive formula:

        C_0=1 \ and \ C_{n+1}=\sum_{i=0}^{n}C_iC_{n-i} \ for \ n\geq 0;
        :param n: number of keys
        :return: number of unique bsts that can be made
        """
        n1, n2, s = 0, 0, 0
        # Base case
        if n <=1:
            return 1

        # find nth Catalan number
        for i in range(1, n+1):
            n1 = BST.unique_bst(i-1)
            n2 = BST.unique_bst(n-i)
            s += n1 + n2

        return s

    @staticmethod
    def isSameBST(tree1: TreeNode, tree2: TreeNode) -> bool:
        """
        Given two different trees' roots. Determine if they are same tree
        :param tree1: root node of first tree
        :param tree2: root node of second tree
        :return: boolean indicating if both trees are identical
        """
        first = Tree(tree1)
        second = Tree(tree2)

        fPre = first.pre_order_iterative_memory_improvement(first)
        sPre = second.pre_order_iterative_memory_improvement(second)

        return fPre == sPre

    def kthSmallest(self, k: int) -> int:
        """
        Gien the root of a binary search tree, and an integer k,
        return the kth smallest element in the tree
        :param k: integer representing the kth smallest element
        :return: value of kth smallest element in tree
        """
        # This can be solved with inorder
        # because inorder tree traversal of bst
        # will always give a sorted array,
        # hence search for k-1 index in array
        tree = Tree(self.root)
        inorderArr = tree.in_order_iterative()
        return inorderArr[k-1]

    def range_sum(self, low: int, high: int) -> int:
        """
        Given the root node of a binary search tree and two integers low and high, return the sum of values of all nodes
        with a value in the inclusive range [low, high].
        :param low: floor of range
        :param high: ceiling of range
        :return: sum of nodes with values between low and high
        """
        res = 0
        if not self.root:
            return res
        s = [self.root]
        while s:
            node = s.pop()
            if low <= node.val <= high:
                res += node.val
            if node.left:
                s.append(node.left)
            if node.right:
                s.append(node.right)
        return res
