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

    def reverseLinkedList(self, list: Node) -> Node:
        curr = list
        prev = None
        while curr:
            cache = curr.next
            curr.next = prev
            prev = curr
            curr = cache

        return prev

    def isPalindrome(self, list: Node) -> bool:
        cache = list
        head = list
        tail = self.reverseLinkedList(cache)

        while head and tail:
            if head.value != tail.value:
                return False
            head = head.next
            tail = tail.next

        return True

    def mergeTwoLists(self, l1: Node, l2: Node) -> Node: # Resolver: Merge dos ordered linked lists y que te quede ordered
        curr = None
        while l1 and l2:
            if l1.value <= l2.value:
                if curr:
                    curr.next = l1
                    curr = curr.next
                    l1 = l1.next
                else:
                    curr = l1
                    l1 = l1.next
            else:
                if curr:
                    curr.next = l2
                    curr = curr.next
                    l2 = l2.next
                else:
                    curr = l2
                    l2 = l2.next
        if l1:
            curr.next = l1
        if l2:
            curr.next = l2

        return curr


