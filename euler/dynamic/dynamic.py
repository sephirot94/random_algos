def compute_series(n: int) -> int:
    """
    Returns the computed nth position in the series given by the following formula:
    A[n] = 2 * A[n-1] - 3 * A[n-2], Where A[0] = 0 and A[1] = 1
    """
    dp = [0 for i in range(n+1)]
    dp[0], dp[1] = 0, 1
    for i in range(2, n+1):
        dp[i] = 2 * dp[i-1] - 3 * dp[i-2]

    return dp[n]