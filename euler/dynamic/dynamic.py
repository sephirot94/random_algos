def compute_series(n: int) -> int:
    """
    Returns the computed nth position in the series given by the following formula:
    A[n] = 2 * A[n-1] - 3 * A[n-2], Where A[0] = 0 and A[1] = 1
    """
    dp = [0 for i in range(n + 1)]
    dp[0], dp[1] = 0, 1
    for i in range(2, n + 1):
        dp[i] = 2 * dp[i - 1] - 3 * dp[i - 2]

    return dp[n]


def fibonacci(n: int) -> int:
    """
    Given a positive integer returns the Nth fibonacci number
    :param n: number representing the Nth fibonacci number
    :return Nth fibonacci number
    """
    if n < 2:
        return 1
    dp = [0 for i in range(n+1)]
    dp[0], dp[1] = 1, 1
    for i in range(2, n+1):
        dp[i] = dp[i-1] + dp[i-2]
    return dp[n]


def is_fibonacci(n: int) -> bool:
    """Given a positive integer returns whether integer belongs in the fibonacci sequence"""
    if n < 2:
        return True
    dp = [0 for i in range(n)]
    dp[0], dp[1] = 1, 1
    for i in range(2, n+1):
        dp[i] = dp[i-1] + dp[i-2]
        if n == dp[i]:
            return True
        if n < dp[i]:
            break
    return False

def rob_houses(nums: list[int]) -> int:
    """
    Returns the max amount you can get from robbing houses in a neighbourhood, represented as int in list.
    Two adjacent houses (direct neighbors) cannot be robbed the same night.
    Input example: [1,2,3,1]
    """
    if not nums:
        return 0
    rob1, rob2 = 0, 0  # initialize pointers to traverse array
    for house in nums:
        # [rob1, rob2, n, n + 1, ...] will be input array, we will move through this
        temp = max(house + rob1, rob2)
        rob1 = rob2  # we traverse the array in O(n), left to right
        rob2 = temp  # We will be storing in temp (rob2) the maximum found we can rob
    return rob2

def rob_houses_2(nums: list[int]) -> int:
    """
    Returns the max amount you can get from robbing houses in a neighbourhood, represented as int in list.
    Two adjacent houses (direct neighbors) cannot be robbed the same night.
    Input example: [1,2,3,1]
    """

    def simple_rob(nums):
        if not nums:
            return 0
        if len(nums) == 1:
            return nums[0]

        dp = [0 for i in range(len(nums))]
        dp[0] = nums[0]
        dp[1] = max(nums[0], nums[1])

        for i in range(2, len(nums)):
            dp[i] = max(dp[i - 2] + nums[i], dp[i - 1])

        return dp[-1]

    if len(nums) == 1:
        return nums[0]

    # Compare robbing from the first house to the second-to-last and second house to the last
    return max(simple_rob(nums[:-1]), simple_rob(nums[1:]))

def rob_houses_3(nums: list[int]) -> int:
    pass
