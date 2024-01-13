# 2357. Make Array Zero by Subtracting Equal Amounts
# Difficulty: easy
# Tag: implementation
# Link: https://leetcode.com/problems/make-array-zero-by-subtracting-equal-amounts/

# Description: You are given a non-negative integer array nums. In one operation, you must:
# Choose a positive integer x such that x is less than or equal to the smallest non-zero element in nums.
# Subtract x from every positive element in nums.
# Return the minimum number of operations to make every element in nums equal to 0.

# idea: the number of operations is the number of unique elements in nums except 0
from typing import List

class Solution:
    def minimumOperations(self, nums: List[int]) -> int:
        # create a set of unique elements in nums except 0
        s = {num for num in nums if num > 0}
        return len(s) if len(s) > 0 else 0

sl = Solution()
print(sl.minimumOperations([1,5,0,3,5]))