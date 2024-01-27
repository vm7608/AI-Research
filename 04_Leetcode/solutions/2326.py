# 2326. Spiral Matrix IV
# Dificulty: Medium
# Tags: implementation
# Link: https://leetcode.com/problems/spiral-matrix-iv/

# Description: You are given two integers m and n, which represent the dimensions of a matrix.
# You are also given the head of a linked list of integers.
# Generate an m x n matrix that contains the integers in the linked list presented in spiral order (clockwise),
# starting from the top-left of the matrix.
# If there are remaining empty spaces, fill them with -1.
# Return the generated matrix.

# Idea:
# First, we init a matrix with -1 value, size m x n
# Use 2 variables to store the direction of the spiral matrix
# 0 , 1 mean go right
# 1 , 0 mean go down
# 0 , -1 mean go left
# -1 , 0 mean go up
# at first, we go right, so row_direct = 0, col_direct = 1
# condtion to swap the direction: out of bound or the next element is not -1
# if we go right, we will go down
# if we go down, we will go left
# if we go left, we will go up
# if we go up, we will go right
# so we swap the direction by row_direct, col_direct = col_direct, -row_direct


from typing import List, Optional

# Definition for singly-linked list.


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class Solution:
    def spiralMatrix(self, m: int, n: int, head: Optional[ListNode]) -> List[List[int]]:

        # init matrix with -1 value, size m x n
        rs = [[-1 for _ in range(n)] for _ in range(m)]
        row, col = 0, 0
        row_direct, col_direct = 0, 1

        while head:
            rs[row][col] = head.val
            if row + row_direct < 0 or row + row_direct >= m or col + col_direct < 0  \
                    or col + col_direct >= n or rs[row+row_direct][col+col_direct] != -1:
                row_direct, col_direct = col_direct, -row_direct

            row += row_direct
            col += col_direct
            head = head.next

        return rs


if __name__ == "__main__":
    m = 3
    n = 5
    val = [3, 0, 2, 6, 8, 1, 7, 9, 4, 2, 5, 5, 0]
    head = ListNode(val[0])
    cur = head
    for i in range(1, len(val)):
        cur.next = ListNode(val[i])
        cur = cur.next

    # [[3, 0, 2, 6, 8], [5, 0, -1, -1, 1], [5, 2, 4, 9, 7]]
    print(Solution().spiralMatrix(m, n, head))
