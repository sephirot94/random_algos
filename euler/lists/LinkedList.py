class Node:
    def __init__(self, value: int, prev=None, next=None):
        self.value = value
        self.prev = prev
        self.next = next

class LinkedList:
    def __init__(self, head=None):
        self.head = head

    def __str__(self):
        curr = self.head
        while curr:
            print(curr.value)
            curr = curr.next

    # insertion method for the linked list

    def insertAtTail(self, data: int):
        newNode = Node(data)
        if self.head:
            current = self.head
            while current.next:
                current = current.next
            current.next = newNode
        else:
            self.head = newNode

    def insertAtPosition(self, data: int, position: int):
        newNode = Node(data)
        itr = 0
        current = self.head
        while itr<position:
            itr+=1
            current = current.next
        tmp = current.next
        current.next = newNode
        current.next.next = tmp

    def removeValue(self, node: Node, val: int):
        curr = node
        head = None
        if curr.value == val:
            head = curr.next

        head = curr
        prev = head
        while curr.next:
            if curr.value == val:
                prev.next = curr.next
                curr = curr.next
                break
            prev = curr
            curr = curr.next

        if curr.value == val:
            prev.next = None

        return head

    def oddEvenLinkedList(self, head: Node) -> Node:
        if not head or not head.next:
            return head
        odd, even = head, head.next

        oddHead = head
        evenHead = even

        while odd and even:
            odd.next = even.next
            if not odd.next:
                break
            even.next = odd.next.next
            odd = odd.next
            even = even.next

        odd.next = evenHead

        return oddHead

    @staticmethod
    def add_two_numbers(l1: Node, l2: Node) -> Node:
        head = None
        curr = None
        carry = 0
        while l1 or l2:
            tmp = 0

            if l1:
                tmp += l1.val
                l1 = l1.next
            if l2:
                tmp += l2.val
                l2 = l2.next

            tmp += carry

            if tmp > 9:
                carry = 1
                val = tmp % 10
            else:
                carry = 0
                val = tmp
            node = Node(val, next=None)
            if head:
                curr.next = node
                curr = node
            else:
                head = node
                curr = node
        if carry == 1:
            node = Node(1, next=None)
            curr.next = node

        return head

    @staticmethod
    def list2LinkedList(l: list):
        curr = None
        head = None
        for x in l:
            if not head:
                head = Node(x)
                curr = head
            else:
                tmp = curr
                curr.next(Node(x))
                curr = curr.next
                curr.prev = tmp
        return head

    def reverse_linked_list(self, head: Node) -> Node:
        curr = head
        prev = None
        while curr:
            cache = curr.next
            curr.next = prev
            prev = curr
            curr = cache

        return prev

    def reverse_linked_list_between(self, head: Node, left: int, right: int) -> Node:
        """
        Given the head of a singly linked list and two integers left and right where left <= right
        reverse the nodes of the list from position left to position right, and return the reversed list.
        :param head: head or root node of linked list to be reversed
        :param left: start position (node index) to reverse
        :param right:  end position (node index) to reverse
        :return: reversed linked list head node
        """
        # O(n) time and O(1) space
        if left == right:
            return head
        temp = Node(0)
        temp.next = head
        prev = temp

        # traverse to first element of reversed portion
        for i in range(left-1):
            prev = prev.next
        curr = prev.next
        nxt = curr.next
        # reverse portion of list
        for i in range(right-left):
            tmp = nxt.next
            nxt.next = curr
            curr = nxt
            nxt = tmp

        # correct head and tail pointers
        prev.next.next = nxt
        prev.next = curr

        return temp.next

    def is_palindrome(self, head: Node) -> bool:
        """
        Given a linked-list root node, verify if the list is a palindrome.
        Palindrome can be read equally either normally of reversed.
        :param head: head or root of linked list
        :return: boolean indicating if list is palindrome
        """
        # Create slow and fast pointer to get to the end of the list quickly.
        slow_pointer = head
        fast_pointer = head
        # Iterate to the end of the list.
        # Slow pointer ends in the middle
        while fast_pointer.next and fast_pointer.next.next:
            slow_pointer = slow_pointer.next
            fast_pointer = fast_pointer.next.next
        # current node for next iteration is middle or next (odd and even linked lists work in this scenario)
        curr = slow_pointer.next
        prev = None
        next_cache = None
        # reverse second half of list to evaluate palindrome
        while curr:
            next_cache = curr.next
            curr.next = prev
            prev = curr
            curr = next_cache
        # Following while will break once reversed half has been traversed
        while prev:
            # Evaluate each node from two halves (regular and reversed)
            if prev.val != head.val:
                return False
            prev = prev.next
            head = head.next
        return True

    def mergeTwoLists(self, l1: Node, l2: Node) -> Node:
        curr = None
        root = None
        if not l1 and not l2:
            return None
        if not l1:
            return l2
        if not l2:
            return l1

        if l1.val <= l2.val:
            curr = l1
            l1 = l1.next
        else:
            curr = l2
            l2 = l2.next

        root = curr
        while l1 and l2:
            if l1.val <= l2.val:
                curr.next = l1
                curr = curr.next
                l1 = l1.next

            else:
                curr.next = l2
                curr = curr.next
                l2 = l2.next

        if l1:
            curr.next = l1
            curr = curr.next
        if l2:
            curr.next = l2
            curr = curr.next

        return root


class SnapshotArray:

    def __init__(self, length: int):
        self.d = {}
        self.snap_d = {}
        self.snap_times = -1

    def set(self, index: int, val: int) -> None:
        self.d[index] = val

    def snap(self) -> int:
        self.snap_times += 1
        self.snap_d[self.snap_times] = self.d.copy()
        return self.snap_times

    def get(self, index: int, snap_id: int) -> int:
        if snap_id in self.snap_d:
            d_at_snap = self.snap_d[snap_id]
            if index in d_at_snap:
                return d_at_snap[index]
        return 0

class LinkedListCyclic:
    def __init__(self):
        self.head = None
        self.last_node = None

    def add_vals(self, data):
        if not self.last_node:
            self.head = Node(data)
            self.last_node = self.last_node.next

    def get_node_val(self, index: int):
        curr = self.head
        for i in range(index):
            curr = curr.next
            if not curr:
                return None
            return curr

    def check_cycle(self, arr: list) -> bool:
        slow = arr.head
        fast = arr.head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            if slow == fast:
                return True
            return False
