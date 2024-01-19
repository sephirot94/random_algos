from typing import Any


class MyCircularDeque:

    def __init__(self, k: int):
        """
        Initialize your data structure here. Set the size of the deque to be k.
        """
        self.size = k
        self.arr = []

    def insertFront(self, value: int) -> bool:
        """
        Adds an item at the front of Deque. Return true if the operation is successful.
        """
        if len(self.arr) < self.size:
            self.arr.insert(0, value)
            return True
        return False

    def insertLast(self, value: int) -> bool:
        """
        Adds an item at the rear of Deque. Return true if the operation is successful.
        """
        if len(self.arr) < self.size:
            self.arr.append(value)
            return True
        return False

    def deleteFront(self) -> bool:
        """
        Deletes an item from the front of Deque. Return true if the operation is successful.
        """
        if self.arr:
            self.arr.pop(0)
            return True
        return False

    def deleteLast(self) -> bool:
        """
        Deletes an item from the rear of Deque. Return true if the operation is successful.
        """
        if self.arr:
            self.arr.pop()
            return True
        return False

    def getFront(self) -> int:
        """
        Get the front item from the deque.
        """
        if self.arr:
            return self.arr[0]
        return -1

    def getRear(self) -> int:
        """
        Get the last item from the deque.
        """
        if self.arr:
            return self.arr[-1]
        return -1

    def isEmpty(self) -> bool:
        """
        Checks whether the circular deque is empty or not.
        """
        return len(self.arr) == 0

    def isFull(self) -> bool:
        """
        Checks whether the circular deque is full or not.
        """
        return len(self.arr) == self.size


class MyCircularQueue:

    def __init__(self, k: int):
        self.arr = []
        self.size = k

    def enQueue(self, value: int) -> bool:
        if len(self.arr) < self.size:
            self.arr.append(value)
            return True
        return False

    def deQueue(self) -> bool:
        if self.arr:
            self.arr.pop(0)
            return True
        return False

    def Front(self) -> int:
        if self.arr:
            return self.arr[0]
        return -1

    def Rear(self) -> int:
        if self.arr:
            return self.arr[-1]
        return -1

    def isEmpty(self) -> bool:
        return len(self.arr) == 0

    def isFull(self) -> bool:
        return len(self.arr) == self.size


class FrontMiddleBackQueue:

    def __init__(self):
        self.arr = []

    def pushFront(self, val: int) -> None:
        self.arr.insert(0, val)

    def pushMiddle(self, val: int) -> None:
        self.arr.insert(len(self.arr)//2, val)

    def pushBack(self, val: int) -> None:
        self.arr.append(val)

    def popFront(self) -> int:
        if self.arr:
            return self.arr.pop(0)
        return -1

    def popMiddle(self) -> int:
        if self.arr:
            if len(self.arr)%2 == 0:
                return self.arr.pop((len(self.arr)//2)-1)
            else:
                return self.arr.pop(len(self.arr)//2)
        return -1

    def popBack(self) -> int:
        if self.arr:
            return self.arr.pop()
        return -1


class PriorityQueue(object):
    def __init__(self):
        self.queue = []

    def __str__(self):
        return ' '.join([str(i) for i in self.queue])

    # for checking if the queue is empty
    def isEmpty(self):
        return len(self.queue) == 0

    # for inserting an element in the queue
    def insert(self, data):
        self.queue.append(data)

    # for popping an element based on Priority
    def delete(self):
        try:
            max = 0
            for i in range(len(self.queue)):
                if self.queue[i] > self.queue[max]:
                    max = i
            item = self.queue[max]
            del self.queue[max]
            return item
        except IndexError:
            print()
            exit()


class CustomQueue:

    def __init__(self, q: list=None):
        self.q = q if q else []
        self.size = len(self.q)

    def enqueue(self, elem: Any):
        self.q.append(elem)
        self.size += 1

    def dequeue(self) -> Any:
        if self.size > 0:
            self.size -= 1
            return self.q.pop(0)


    def get(self, n: int) -> Any:
        if n > self.size:
            raise IndexError("Index out of range")
        if -1*self.size <= n < 0:
            return self.q[n]
        return self.q[n-1]


class TestCustomQ:

    def __init__(self):
        self.q = CustomQueue()

    def test_enqueue(self, elem: Any):
        self.q.enqueue(elem)
        assert self.q.size == 1
        assert self.q[0] == elem

    def test_dequeue(self):
        if self.q.size > 0:
            assert self.q.dequeue()
        else:
            assert not self.q.dequeue()

    def test_get(self, n: int):
        s = self.q.size
        if not s: return


