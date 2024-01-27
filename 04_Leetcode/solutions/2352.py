# 2352. Equal Row and Column Pairs
# Difficulty: Medium
# Tag: implementation
# Link: https://leetcode.com/problems/equal-row-and-column-pairs/

# Description: Given a 0-indexed n x n integer matrix grid, return the number of pairs (ri, cj) such that row ri and column cj are equal.
# A row and column pair is considered equal if they contain the same elements in the same order (i.e., an equal array).

# Idea: use dict have key is the string of the column, value is the number of times the column appears
# then loop through the rows, if the row is in the dict, increase the result by the value of the row in the dict
# this is the number of equal row and column pairs

from typing import List


class Solution:
    def equalPairs(self, grid: List[List[int]]) -> int:
        d = {}
        for i in range(len(grid)):
            cols = [grid[j][i] for j in range(len(grid))]
            d[str(cols)] = d.get(str(cols), 0) + 1

        rs = 0
        for i in range(len(grid)):
            rows = grid[i]
            if str(rows) in d.keys():
                rs += d[str(rows)]
        return rs


sl = Solution()
print(sl.equalPairs(
    [[3, 1, 2, 2], [1, 4, 4, 5], [2, 4, 2, 2], [2, 4, 2, 2]]))  # 3
