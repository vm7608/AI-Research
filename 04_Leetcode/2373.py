# 2373. Largest Local Values in a Matrix
# Difficulty: easy
# Tags: implementation
# Link: https://leetcode.com/problems/largest-local-values-in-a-matrix/

# Description: You are given an n x n integer matrix grid.
# Generate an integer matrix maxLocal of size (n - 2) x (n - 2) such that:
# maxLocal[i][j] is equal to the largest value of the 3 x 3 matrix in grid centered around row i + 1 and column j + 1.
# In other words, we want to find the largest value in every contiguous 3 x 3 matrix in grid.

# Idea: use 2 loop and compare the value of each cell with its 8 neighbors

from typing import List


class Solution:
    def largestLocal(self, grid: List[List[int]]) -> List[List[int]]:
        n = len(grid)
        rs = [[0] * (n - 2) for _ in range(n - 2)]
        for i in range(1, n - 1):
            for j in range(1, n - 1):
                rs[i - 1][j - 1] = max(
                    grid[i][j],
                    grid[i - 1][j],
                    grid[i + 1][j],
                    grid[i][j - 1],
                    grid[i][j + 1],
                    grid[i - 1][j - 1],
                    grid[i - 1][j + 1],
                    grid[i + 1][j - 1],
                    grid[i + 1][j + 1],
                )
        return rs


sl = Solution()
print(sl.largestLocal(
    [[9, 9, 8, 1], [5, 6, 2, 6], [8, 2, 6, 4], [6, 2, 2, 2]]))
print(
    sl.largestLocal(
        [[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 2, 1, 1],
            [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]]
    )
)
