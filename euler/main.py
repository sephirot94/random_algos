from heap.heapster import smallest_subset
from lists.LinkedList import LinkedList, Node
from trees.tree import TreeNode, Tree
from trees.binary_search import BST
from leet.solution import Solution
import math

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # head = Node(1, next=Node(2, next=Node(3, next=Node(4, next=Node(5)))))
    # head = Node(1, next=Node(1, next=Node(2, next=Node(1))))
    # ll = LinkedList()
    # n = ll.isPalindrome(head)
    sol = Solution()
    print(sol.is_valid_palindrome("ab_a"))
    # root = TreeNode('a', TreeNode('b', TreeNode('c'), TreeNode('d')), TreeNode('e', TreeNode('f'), TreeNode('g')))
    root = TreeNode(5, TreeNode(1), TreeNode(4, TreeNode(3), TreeNode(6)))
    tree = Tree(root)
    bst = BST(root)
    # print(sol.check_pythagorean_triplet([3,2,4,6]))
    # print(sol.check_array_non_decreasing([13,4,1]))
    # print(bst.insert_value_iterative(9).__str__())
    # print(sol.ransom_note_with_words("two times three is not four", "two times two is four"))
    # print(tree.check_tree_is_BST(root))
    # print(tree.pre_order_iterative_memory_improvement(tree.root))