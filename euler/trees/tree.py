class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


class Tree(object):

    def leafSimilar(self, root1, root2):
        # l1 = self.leaves(root1)
        # l2 = self.leaves(root2)
        l1 = self.leavesRecursive(root1)
        l2 = self.leavesRecursive(root2)
        return l1 == l2

    def leaves(self, root):
        stack = []
        stack.append(root)
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

    def leavesRecursive(self, n):
        leaves = []
        self.leavesRecursiveHelper(n, leaves)
        return leaves

    def leavesRecursiveHelper(self, n, leaves):
        if not n:
            return
        if not n.left and not n.right:
            leaves.append(n.val)
        self.leavesRecursiveHelper(n.left, leaves)
        self.leavesRecursiveHelper(n.right, leaves)