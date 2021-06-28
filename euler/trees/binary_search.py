from trees.tree import Tree, TreeNode

# NOTES
# Inorder traversal of BST always produces sorted output
#

class BST(Tree):

    def __init__(self, root: TreeNode):
        self.root = root

    def __str__(self):
        print(f"Node: {self.root.val}")
        print()

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
