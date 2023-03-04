class MaxStack:
    def __init__(self):
        self.max = []
        self.stack = []

    # O(1)
    def push(self, val):
        """
        Push element to stack
        :param val: element to be added to stack
        """
        self.stack.append(val)
        if not self.max:
            self.max.append(val)
        else:
            self.max.append(max(val, self.max[-1]))

    # O(1)
    def pop(self) -> int:
        """
        Pop element from stack
        :return: element popped
        """
        if self.max:
            self.max.pop()
        if self.stack:
            self.stack.pop()
        else:
            return -1

    # O(1)
    def peek(self) -> int:
        """
        Peek top element without extracting
        :return: top element in stack
        """
        if self.stack:
            return self.stack[-1]
        return -1

    # O(1)
    def peekMax(self) -> int:
        """
        Peek max element
        :return: max element in stack
        """
        if self.max:
            return self.max[-1]
        return -1


class MinStack:

    def __init__(self):
        """
        initialize your data structure here.
        """
        self.stack = []
        self.min = []

    def push(self, val: int) -> None:
        self.stack.append(val)
        if not self.min:
            self.min.append(val)
        else:
            self.min.append(min(val, self.min[-1]))

    def pop(self) -> None:
        if not self.stack:
            return None
        if self.min:
            self.min.pop()
        if self.stack:
            self.stack.pop()
        else:
            return None

    def top(self) -> int:
        if self.stack:
            return self.stack[-1]
        return -1

    def getMin(self) -> int:
        if self.min:
            return self.min[-1]
        return -1

# Your MinStack object will be instantiated and called as such:
# obj = MinStack()
# obj.push(val)
# obj.pop()
# param_3 = obj.top()
# param_4 = obj.getMin()
