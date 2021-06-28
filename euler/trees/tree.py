from collections import deque


class TreeNode(object):
    def __init__(self, x, left=None, right=None):
        self.val = x
        self.left = left
        self.right = right

    def __str__(self):
        print(f"Value: {self.val}")
        print(f"Left: {self.left.val if self.left else ''}")
        print(f"Right: {self.right.val if self.right else ''}")


class Tree(TreeNode):
    def __init__(self, root: TreeNode):
        self.root = root

    def pre_order(self, root):
        """
        Recursive preorder tree traversal
        :param root: root node of tree
        :return:
        """
        return [root.val] + self.pre_order(root.left) + self.pre_order(root.right) if root else []

    def pre_order_iterative(self, root):
        """
        pre order tree traversal iterative. Uses O(N) time and space complexity
        :param root: root node of tree
        :return: pre order traversal of tree
        """
        if root is None:
            return
        stack = []
        resp = []
        stack.append(root)
        while len(stack) > 0:
            node = stack.pop()
            resp.append(node.val)
            if node.right:
                stack.append(node.right)
            if node.left:
                stack.append(node.left)
        return resp

    def pre_order_iterative_memory_improvement(self, root):
        """
        Just like a pre order iterative, this method uses O(H) space complexity, where H is the height of tree
        :param root: root node of tree
        :return: pre order traversal of tree
        """
        if root is None:
            return
        stack = []
        resp = []
        curr = root
        while len(stack) > 0 or curr:
            while curr:
                resp.append(curr.val)
                if curr.right:
                    stack.append(curr.right)
                curr = curr.left
            if len(stack) > 0:
                curr = stack.pop()
        return resp

    def in_order(self, root):
        """
        Recursively traverse a tree inorder
        :param root: root node of tree
        :return: inorder traversal of tree
        """
        return self.in_order(root.left) + [root.val] + self.in_order(root.right) if root else []

    def in_order_iterative(self):
        """
        Iteratively traverse a tree inorder
        :return: inorder traversal of tree
        """
        if self.root is None:
            return
        curr = self.root
        stack = []
        resp = []
        while True:
            if curr:
                stack.append(curr)
                curr = curr.left
            elif stack:
                curr = stack.pop()
                resp.append(curr.val)
                curr = curr.right
            else:
                break
        return resp

    def post_order(self, root):
        """
        Recursively traverse tree post order
        :param root: root node of tree
        :return: post order tree traversal
        """
        return self.post_order(root.left) + self.post_order(root.right) + [root.val] if root else []

    def post_order_iteratively(self):
        """
        Iteratively traverse tree post order
        :return: post order tree traversal
        """
        if self.root is None:
            return
        mainStack = []
        auxStack = []
        resp = []
        auxStack.append(self.root)
        while auxStack:
            node = auxStack.pop()
            mainStack.append(node.val)
            if node.left:
                auxStack.append(node.left)
            if node.right:
                auxStack.append(node.right)
        while mainStack:
            val = mainStack.pop()
            resp.append(val)
        return resp

    def breath_first_search_iterative(self):
        """
        Given a root of tree, return the traversal of tree breath-first.
        In other words, traverse each level of leaves in the following order:
        root -> root's children -> root's grandchildren -> ... -> root's Nth great grandchildren
        :return: bfs traversal of tree
        """
        queue = deque([self.root])
        resp = []
        while len(queue):
            node = queue.popleft()
            resp.append(node.val)
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        return resp

    def maximum_depth_of_tree(self) -> int:
        """
        Given a tree, return max depth
        :return: height or max depth of tree
        """
        if not self.root:
            return 0

        queue = deque([self.root])
        height = 0
        while len(queue):
            elements = len(queue)
            for i in range(elements):
                node = queue.popleft()
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            height+=1
        return height

    def minimum_depth_of_tree(self) -> int:
        """
        Given a binary tree, return it's minimum depth.
        The minimum depth is the number of nodes along the shortest path from the root node down to the nearest leaf node
        :return: integer with minimum depth of tree
        """
        if not self.root:
            return 0
        queue = deque([self.root])
        height = 1
        while queue:
            elements = len(queue)
            for i in range(elements):
                node = queue.popleft()
                if not node.left and not node.right:
                    return height
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            height+=1

        return height

    def level_order(self) -> list:
        """
        Given the root of a binary tree, return the level order traversal of its nodes' values.
        Example:
            Input: root = [3,9,20,null,null,15,7]
            Output: [[3],[9,20],[15,7]]
        :return: level order traversal of tree nodes
        """
        if not self.root:
            return []
        queue = deque([self.root])
        resp = []
        while queue:
            aux = []
            elements = len(queue)
            for i in range(elements):
                node = queue.popleft()
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
                aux.append(node.val)
            resp.append(aux)

        return resp

    def is_symmetric_recursive(self):
        """
        Given a tree, return a boolean indicating if it symmetric around its center
        :return: whether tree is symmetric
        """

        def is_symmetric_recursive_helper(root1: TreeNode = None, root2: TreeNode = None) -> bool:
            """
            Recursive closure function to check symmetric tree
            :param root1: used for recursive strategy
            :param root2: used for recursive strategy
            :return: whether a tree is symmetric
            """

            if not root1 and not root2:
                return True
            if root1 and root2:
                if root1.val == root2.val:
                    return(is_symmetric_recursive_helper(root1.left, root2.right)
                            and is_symmetric_recursive_helper(root1.right, root2.left))

        return is_symmetric_recursive_helper(self.root.left, self.root.right) if self.root else True

    def is_symmetric_iterative(self) -> bool:
        """
        Given a tree, return a boolean indicating if it symmetric around its center
        :return: whether tree is symmetric
        """
        if not self.root:
            return True
        if not self.root.left and not self.root.right:
            return True
        stack = []
        stack.append(self.root)
        stack.append(self.root)
        while stack:
            leftNode = stack.pop()
            rightNode = stack.pop()
            if leftNode.val != rightNode.val:
                return False
            if leftNode.left and rightNode.right:
                stack.append(leftNode.left)
                stack.append(rightNode.right)
            elif leftNode.left or rightNode.right:
                return False
            if leftNode.right and rightNode.left:
                stack.append(leftNode.right)
                stack.append(rightNode.left)
            elif leftNode.right or rightNode.left:
                return False
        return True

    def check_tree_is_BST(self) -> bool:
        """
        Given a binary tree, check whether it is a BST or not
        :param root: root node of tree
        :return: boolean indicating whether it is a BST
        """
        if not self.root:
            return False

        resp = self.in_order_iterative(self.root)

        for idx in range(1, len(resp)):
            if resp[idx] <= resp[idx - 1]:
                return False

        return True

    @staticmethod
    def leaf_similar_recursive(root1: TreeNode, root2: TreeNode) -> bool:
        """
        Given two roots, check whether leaves are similar.
        Leaves are nodes without any children.
        :param root1: root node of first tree
        :param root2: root node of second tree
        :return: boolean indicating whether both trees have similar leaves
        """
        def leavesRecursiveHelper(n, leaves):
            """
            Helper used for recursion
            :param n: node
            :param leaves: already found leaves array
            :return:
            """
            if not n:
                return
            if not n.left and not n.right:
                leaves.append(n.val)
            leavesRecursiveHelper(n.left, leaves)
            leavesRecursiveHelper(n.right, leaves)

        def leavesRecursive(n):
            """
            Recursion handler
            :param n: node
            :return: leaves of node
            """
            leaves = []
            leavesRecursiveHelper(n, leaves)
            return leaves

        l1 = leavesRecursive(root1)
        l2 = leavesRecursive(root2)
        return l1 == l2

    def leaves(self):
        """
        Given a tree, return it's leaves in an array
        :return: array containing tree leaves
        """
        stack = [self.root]
        leaves = []
        while len(stack) > 0:
            n = stack.pop()
            if n.right:
                stack.append(n.right)
            if n.left:
                stack.append(n.left)
            if not n.left and not n.right:
                leaves.append(n.val)
        return leaves