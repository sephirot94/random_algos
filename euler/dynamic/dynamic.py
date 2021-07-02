def num_ways_to_paint_k_cells(n: int, p: int, k: int) -> int:
    """
    Given three integers N, P and K, the task is to find the number of ways of painting K cells of 3 x N grid
    such that no adjacent cells are painted and also no continuous P columns are left unpainted.
    Note: Diagonal cells are not considered as adjacent cells
    :param n: integer indicating number of columns in 3 x N matrix
    :param p: integer indicating number of continuous columns that are left unpainted
    :param k: integer indicating number of cells to paint
    :return: number of ways to paint cells in grid
    """
    mod = 1e9 + 7
    MAX = 301
    MAXP = 3
    MAXK = 600
    MAXPREV = 4

    dp = [[[[-1 for x in range(MAXPREV + 1)] for y in range(MAXK)]
           for z in range(MAXP + 1)] for k in range(MAX)]

    # Visited array to keep track
    # of which columns were painted
    vis = [False] * MAX

    # Recursive Function to compute the
    # number of ways to paint the K cells
    # of the 3 x N grid
    def recursive_helper(col, prevCol,
                         painted, prev,
                         N, P, K):

        # Condition to check if total
        # cells painted are K
        if (painted >= K):
            continuousCol = 0
            maxContinuousCol = 0

            # Check if any P continuous
            # columns were left unpainted
            for i in range(N):
                if (vis[i] == False):
                    continuousCol += 1
                else:
                    maxContinuousCol = max(maxContinuousCol,
                                           continuousCol)
                    continuousCol = 0

            maxContinuousCol = max(
                maxContinuousCol,
                continuousCol)

            # Condition to check if no P
            # continuous columns were
            # left unpainted
            if (maxContinuousCol < P):
                return 1

            # return 0 if there are P
            # continuous columns are
            # left unpainted
            return 0

        # Condition to check if No
        # further cells can be
        # painted, so return 0
        if (col >= N):
            return 0

        # if already calculated the value
        # return the val instead
        # of calculating again
        if (dp[col][prevCol][painted][prev] != -1):
            return dp[col][prevCol][painted][prev]

        res = 0

        # Previous column was not painted
        if (prev == 0):

            # Column is painted so,
            # make vis[col]=true
            vis[col] = True
            res += ((recursive_helper(
                col + 1, 0, painted + 1,
                1, N, P, K))
                    % mod)

            res += ((recursive_helper(
                col + 1, 0, painted + 1,
                2, N, P, K))
                    % mod)

            res += ((recursive_helper(
                col + 1, 0, painted + 1,
                3, N, P, K))
                    % mod)

            # Condition to check if the number
            # of cells to be painted is equal
            # to or more than 2, then we can
            # paint first and third row
            if (painted + 2 <= K):
                res += ((recursive_helper(
                    col + 1, 0, painted + 2,
                    4, N, P, K))
                        % mod)

            vis[col] = False

            # Condition to check if number of
            # previous continuous columns left
            # unpainted is less than P
            if (prevCol + 1 < P):
                res += ((recursive_helper(
                    col + 1, prevCol + 1,
                    painted, 0, N, P, K))
                        % mod)

        # Condition to check if first row
        # was painted in previous column
        elif (prev == 1):
            vis[col] = True
            res += ((recursive_helper(
                col + 1, 0, painted + 1,
                2, N, P, K))
                    % mod)
            res += ((recursive_helper(
                col + 1, 0, painted + 1,
                3, N, P, K))
                    % mod)
            vis[col] = False
            if (prevCol + 1 < P):
                res += ((recursive_helper(
                    col + 1, prevCol + 1,
                    painted, 0, N, P, K))
                        % mod)

        # Condition to check if second row
        # was painted in previous column
        elif (prev == 2):
            vis[col] = True
            res += ((recursive_helper(
                col + 1, 0, painted + 1,
                1, N, P, K))
                    % mod)
            res += ((recursive_helper(
                col + 1, 0, painted + 1,
                3, N, P, K))
                    % mod)

            # Condition to check if the number
            # of cells to be painted is equal to
            # or more than 2, then we can
            # paint first and third row
            if (painted + 2 <= K):
                res += ((recursive_helper(
                    col + 1, 0, painted + 2,
                    4, N, P, K))
                        % mod)

            vis[col] = False
            if (prevCol + 1 < P):
                res += ((recursive_helper(
                    col + 1, prevCol + 1,
                    painted, 0, N, P, K))
                        % mod)

        # Condition to check if third row
        # was painted in previous column
        elif (prev == 3):
            vis[col] = True
            res += ((recursive_helper(
                col + 1, 0, painted + 1,
                1, N, P, K))
                    % mod)
            res += ((recursive_helper(
                col + 1, 0, painted + 1,
                2, N, P, K))
                    % mod)
            vis[col] = False
            if (prevCol + 1 < P):
                res += ((recursive_helper(
                    col + 1, prevCol + 1,
                    painted, 0, N, P, K))
                        % mod)

        # Condition to check if first and
        # third row were painted
        # in previous column
        else:
            vis[col] = True
            res += ((recursive_helper(
                col + 1, 0, painted + 1,
                2, N, P, K))
                    % mod)
            vis[col] = False
            if (prevCol + 1 < P):
                res += ((recursive_helper(
                    col + 1, prevCol + 1,
                    painted, 0, N, P, K))
                        % mod)

        # Memoize the data and return the
        # Computed value
        dp[col][prevCol][painted][prev] = res % mod
        return dp[col][prevCol][painted][prev]

    # Set all values
    # of dp to -1;
    global dp

    # Set all values of Visited
    # array to false
    global vis

    return recursive_helper(0, 0, 0, 0, n, p, k)