import collections
from collections import deque


class TreeNode(object):
    def __init__(self, x, left=None, right=None, height: int=0):
        self.val = x
        self.left = left
        self.right = right
        self.height = height

    def __str__(self):
        print(f"Value: {self.val}")
        print(f"Left: {self.left.val if self.left else ''}")
        print(f"Right: {self.right.val if self.right else ''}")


class Tree:
    def __init__(self, root: TreeNode):
        self.root = root

    def __str__(self):
        """Prints tree in preorder traversal"""
        if not self.root:
            return
        stack = [self.root]
        while stack:
            curr = stack.pop()
            print(curr.val)
            stack.append(curr.right) if curr.right else None
            stack.append(curr.left) if curr.left else None

    def bt_values_at_height(self, height: int) -> list:
        """
        Returns list with values at given height
        """
        # O(n) time and space complexity
        if not self.root:  # base check valid input
            return []
        self.root.height = 0  # init height as 0 for root
        resp = []
        q = collections.deque([self.root])
        while q:  # BFS until q is empty or forced break
            node = q.popleft()
            lvl = node.height
            if lvl > height:  # if lvl has been passed and all elements in resp, break
                break
            if lvl == height:  # if node is at searched height
                resp.append(node.val)
            if node.left:
                node.left.height = node.height + 1  # init children height as curr+1
                q.append(node.left)
            if node.right:
                node.right.height = node.height + 1  # init children height as curr+1
                q.append(node.right)
        return resp

    def arithmetic_tree(self, root: TreeNode) -> int:
        """
        Returns the integer resulting in the solution of arithmetic expression given by tree
        """
        if not root:
            return 0

        if isinstance(root.val, int):  # check if value at node is integer
            return root.val

        elif root.val == "+":  # case + operator
            left = right = 0  # 0 is neutral in + and - operations
            if root.left:
                left = self.arithmetic_tree(root.left)
            if root.right:
                right = self.arithmetic_tree(root.right)
            return left + right
        elif root.val == "-":  # case - operator
            left = right = 0  # 0 is neutral in - operation
            if root.left:
                left = self.arithmetic_tree(root.left)
            if root.right:
                right = self.arithmetic_tree(root.right)
            return left - right
        elif root.val == "*":  # case * operator
            left = right = 1  # 1 is neutral in * operation
            if root.left:
                left = self.arithmetic_tree(root.left)
            if root.right:
                right = self.arithmetic_tree(root.right)
            return left * right
        elif root.val == "/":  # case / operator
            left = right = 1  # 1 is neutral in division
            if root.left:
                left = self.arithmetic_tree(root.left)
            if root.right:
                right = self.arithmetic_tree(root.right)
            return int(left//right)

    def count_number_unival_trees(self) -> int:
        """
        Returns number of unival trees present in tree. Unival trees are those with all elements being equal.
        A sub tree can be any node where all its children have same value as root.
        """
        def count_unival_recursive(root: TreeNode, count: list) -> bool:
            if not root:  # leaf nodes all are considered unival trees
                return True
            left = count_unival_recursive(root.left, count)
            right = count_unival_recursive(root.right, count)
            if not left or not right:  # if either children are false (not univals)
                return False
            if root.left and root.val != root.left.val:  # case left child is not same as root
                return False
            if root.right and root.val != root.right.val:  # case right child is not same as root
                return False

            # If we reach here, means all checks have passed.
            count[0] += 1
            return True

        if not self.root:
            return 0

        count = [0]
        count_unival_recursive(self.root, count)
        return count[0]

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
        stack = [root]
        resp = []
        while stack:
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
            return []
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

    def post_order_iterative(self):
        """
        Iteratively traverse tree post order
        :return: post order tree traversal
        """
        if self.root is None:
            return
        mainStack = []
        auxStack = [self.root]
        resp = []
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
        In other words, traverse each level of nodes in the following order:
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

    def make_tree_from_preorder_inorder(self, preorder: list, inorder: list) -> TreeNode:
        """
        Returns root node of tree formed from given preorder and inorder traversals.
        Assumes the traversals are correct.
        """
        def recursive_helper(d: dict, preorder: list, inorder: list) -> TreeNode:
            if not preorder or not inorder:
                return None
            root = TreeNode(preorder[0])
            preorder.pop(0)  # eliminate value from array
            root_inorder_idx = d.get(root.val, None)
            if root_inorder_idx:
                left_side = inorder[:root_inorder_idx] if root_inorder_idx > 0 else []
                right_side = inorder[root_inorder_idx+1:] if root_inorder_idx + 1 < len(inorder) else []
                root.left = recursive_helper(d, preorder, left_side)
                root.right = recursive_helper(d, preorder, right_side)
            return root

        d = {}  # hashmap will store elements and indexes of inorder list
        for i, num in enumerate(inorder):
            d[num] = i

        return recursive_helper(d, preorder, inorder)

    def make_bst_from_postorder_inorder(self, postorder: list, inorder: list) -> TreeNode:
        """
        Returns root node of tree constructed from given postorder and inorder    traversals. Assumes both traversals are correct
        """
        def recursive_construct_bst(start: int, end: int, postorder: list, inorder: list, d: dict) -> TreeNode:
            """Returns root node of tree (sub-tree). Used recursively as helper"""

            if start > end or not postorder:  # base case
                return None

            root_val = postorder[-1]
            root = TreeNode(root_val)
            postorder.pop()  # pop last element (already visited)

            if start == end:  # element will be leaf
                return root

            root_idx = d[root_val]
            root.right = recursive_construct_bst(root_idx+1, end, postorder, inorder, d)  # find right child first
            root.left = recursive_construct_bst(start, root_idx-1, postorder, inorder, d)  # since it is postorder

            return root

        d = {}
        for i, num in enumerate(inorder):  # construct dictionary
            d[num] = i

        return recursive_construct_bst(0, len(inorder)-1, postorder, inorder, d)

    def make_full_tree_from_preorder_postorder(self, preorder: list, postorder: list) -> TreeNode:
        """
        Returns the root node of tree created from given preorder and postorder traversals.
        Assumes traversals are correct.
        """
        def recursive_helper(start: int, end: int, preorder: list, postorder: list, d: dict):
            if not preorder or not postorder or start > end:
                return None

            root_val = preorder.pop(0)
            root = TreeNode(root_val)

            if not preorder or start == end:  # leaf, no need to recur
                return root

            next = preorder[0]  # root of next iteration
            next_idx = d[next]  # index of next root in postorder list
            if next_idx <= end:
                root.left = recursive_helper(start, next_idx, preorder, postorder, d)
                root.right = recursive_helper(next_idx+1, end-1, preorder, postorder, d)

            return root

        d = {}
        for i, num in enumerate(postorder):  # make dict tracking indexes of elements in postorder
            d[num] = i

        return recursive_helper(0, len(postorder)-1, preorder, postorder, d)

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
        while stack:
            n = stack.pop()
            if n.right:
                stack.append(n.right)
            if n.left:
                stack.append(n.left)
            if not n.left and not n.right:
                leaves.append(n.val)
        return leaves

    def recover_tree_from_pre_traversal(self, traversal: str) -> TreeNode:
        """
        We run a preorder depth-first search (DFS) on the root of a binary tree. At each node in this traversal,
        we output D dashes (where D is the depth of this node), then we output the value of this node.  If the depth of
        a node is D, the depth of its immediate child is D + 1.  The depth of the root node is 0. If a node has only one
        child, that child is guaranteed to be the left child. Given the output traversal of this traversal, recover the
        tree and return its root.
        :param traversal: string containing traversal with dashes
        :return: root node of recovered tree
        """
        # O(n) Time and Space
        def helper(tree_str, curr):
            if len(tree_str) == 0:
                return None
            # construct value of the current node
            node = ""
            i = 0
            for s in tree_str:
                if s.isdigit():
                    node += s
                else:
                    break
                i += 1
            node = TreeNode(int(node))
            # Evaluate the new level at which the next node will be attached
            # to do this iterate starting from - till int is encountered
            new_lvl = 0
            j = 0
            for s in tree_str[i:]:
                if s.isdigit():
                    break
                else:
                    new_lvl += 1

            # if the new level is greater than curr we have to recursively go to the next level
            if new_lvl > curr:
                r_tree_str, r_new_lvl, node.left = helper(tree_str[i+j:], curr+1)
            # this means the new_lvl is less than curr, so return back till curr < new_lvl
            else:
                return (tree_str[i+j:], new_lvl, node)

            # Now we have to deal with the right node
            if r_new_lvl > curr:
                r_tree_str, r_new_lvl, node.right = help(r_tree_str, curr+1)

            return (r_tree_str, r_new_lvl, node)

        _, _, root = helper(traversal, 0)
        return root


class BST:

    def __init__(self, root: TreeNode):
        self.root = root

    def in_order_traversal(self) -> list:
        if not self.root:
            return []

        stack = []
        resp = []
        curr = self.root

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

    def sorted_array_to_bst(self, arr: list) -> TreeNode:
        """
        Given a sorted array as input, return root node of BST formed from array
        """
        if not arr:
            return None

        mid = len(arr)//2
        left = arr[:mid]
        right = arr[mid+1:]

        node = TreeNode(arr[mid])
        node.left = self.sorted_array_to_bst(left)
        node.right = self.sorted_array_to_bst(right)

        return node

    def search(self, target: int) -> TreeNode:
        """
        Given a target value, find if a node exists in BST containing that value
        """
        if not self.root:
            return None

        curr = self.root

        while curr:
            if target == curr.val:
                return curr
            if target < curr.val:
                curr = curr.left
            if target > curr.val:
                curr = curr.right

        return None

    def insert(self, node: TreeNode):
        """
        Given a node, insert it in the BST
        """
        if not self.root:
            self.root = node
            return
        curr = self.root
        while curr:
            prev = curr
            if curr.val == node.val:
                return
            if curr.val < node.val:
                curr = curr.right
            if curr.val > node.val:
                curr = curr.left

        if node.val < prev.val:
            prev.left = node
        else:
            prev.right = node

        return

    def remove(self, node: TreeNode):
        """
        Given an element, remove it from the BST
        """
        if not self.root:
            return

        curr = self.root
        while curr:  # Traverse until we find node to be deleted
            if curr.val < node.val:
                prev = curr
                curr = curr.right
            if curr.val > node.val:
                prev = curr
                curr = curr.left
            else:
                break;

        if not curr:  # If node not found, return
            return

        if not curr.left and not curr.right:  # If node has no childs, delete
            if prev.val < curr.val:
                prev.right = None
            else:
                prev.left = None

        if not curr.left:  # If only child is right
            tmp = curr.right
            curr = None
            if prev.val < tmp.val:
                prev.right = tmp
            else:
                prev.left = tmp

        if not curr.right:  # If only child is left
            tmp = curr.left
            curr = None
            if prev.val < tmp.val:
                prev.right = tmp
            else:
                prev.left = tmp

        # If both children exist, find succesor
        succ_parent = curr
        succ = curr.right

        while succ.left:  # Find smallest element bigger than eliminated node
            succ_parent = succ
            succ = succ.left

        if succ_parent != curr:
            succ_parent.left = succ.right
        else:
            succ_parent.right = succ.right

        curr.val = succ.val
        return






class Trie:
    class Node:
        def __init__(self, char: str, isWord=False):
            self.char = char
            self.isWord = isWord
            self.children = {}

    def __init__(self):
        self.root = self.Node("*", isWord=True)

    def insert(self, word: str):
        """
        Insert new word in Trie data structure
        """
        curr = self.root  # start from root
        for letter in word:  # Iterate through characters of word
            if letter not in curr.children.keys():  # If it is a new character
                new_node = self.Node(letter)  # Create a new node with letter
                curr.children[letter] = new_node  # Add character as child of current node
            curr = curr.children[letter]  # Move pointer to next node
        curr.isWord = True  # After adding the word, assign boolean to last node

    def search(self, word: str) -> bool:
        """
        Search word inside Trie data structure
        """
        curr = self.root  # Start from root
        for letter in word:  # Iterate through the word
            if letter not in curr.children:  # If character is not a node, return false
                return False
            curr = curr.children[letter]  # Move pointer
        return curr.isWord  # return if last character represents a word in Trie DS

    def starts_with(self, prefix: str) -> bool:
        """
        Check if any word inside the Trie starts with given prefix
        """
        curr = self.root
        for char in prefix:
            if char not in curr.children:
                return False
            curr = curr.children[char]
        return True

    def remove(self, word: str):
        """
        Delete a word from Trie data structure
        """
        stack = []
        curr = self.root
        for letter in word:
            if letter not in curr.children:
                return
            curr = curr.children[letter]
            stack.append(curr)
        curr.isWord = False
        curr = stack.pop()
        while stack:
            node = stack.pop()
            if not bool(curr.children) and not curr.isWord:
                del node.children[curr.char]
                curr = node
            else:
                break
        return


class AhoCorasick:
    table = {}

    class Node:
        def __init__(self,id: int, value: any):
            self.id = id
            self.value = value
            self.next_states = {}
            self.fail_state = 0
            self.isFinal = False
            self.output = set()

        def __str__(self):
            print(f"State: {self.id} \n"
                  f"Value: {self.value} \n"
                  f"Next States: { self.next_states.keys()} \n"
                  f"Failure: {self.fail_state} \n")

            if self.isFinal:
                print(f"Output: {self.output}")

            for node in self.next_states.keys():
                s = self.next_states[node]
                s.__str__()

        def goto(self, key: int):
            return self.next_states.get(key)

        def add_output(self, key):
            self.output.add(key)

    def __init__(self):
        self.root = self.Node(0, None)
        self.table[0] = self.root
        self.id = 0

    def add_word(self, word: str):
        curr = self.root
        for letter in word:
            if not curr.next_states.get(letter):
                self.id += 1
                curr.next_states[letter] = self.Node(self.id, letter)
                self.table[self.id] = curr.next_states[letter]
            curr = curr.next_states[letter]

        curr.isFinal = True
        curr.output.add(letter)

    def set_failure(self):
        queue = deque()
        curr = self.root
        for key in self.root.next_states:
            queue.append(self.root.next_states[key])

        while queue:
            curr_dq = queue.popleft()
            for key in curr_dq.next_states:
                queue.append(curr_dq.next_states[key])
                tmp = curr_dq.next_states[key]
                id = curr_dq.fail_state
                val = tmp.value
                curr = self.table[id]

                while True:
                    if not curr.goto(val) and curr.id != 0:
                        new_id = curr.fail_state
                        curr = self.table[new_id]
                    else:
                        break
                    child =  curr.goto(val)
                    if child is None:
                        tmp.fail_state = curr.id
                    else:
                        tmp.fail_state = child.id

                    tmp.add_output(self.table[tmp.fail_state].output)

    def find_string(self, str):
        curr = self.root

        for key in str:
            while True:
                if curr.goto(key) is None and curr.id != 0:
                    curr = self.table[curr.fail_state]
                else:
                    child = curr.goto(key)
                    break
            if child:
                curr = child
                if child.output:
                    print(f"Id {child.id}, {child.output}")

    def display(self):
        self.root.__str__()

