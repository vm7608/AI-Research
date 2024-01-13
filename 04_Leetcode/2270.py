# 2270. Number of Ways to Split Array
# Difficulty: Medium
# Tags: math
# Link: https://leetcode.com/problems/number-of-ways-to-split-array/

# Description: You are given a 0-indexed integer array nums of length n.
# nums contains a valid split at index i if the following are true:
# The sum of the first i + 1 elements is greater than or equal to the sum of the last n - i - 1 elements.
# There is at least one element to the right of i. That is, 0 <= i < n - 1.
# Return the number of valid splits in nums.

# Idea: calculate the total sum of nums
# loop from 0 to n-1, calculate the current sum
# if the current sum is greater than or equal to the total sum - current sum
# then we find a valid split

from typing import List


class Solution:
    def waysToSplitArray(self, nums: List[int]) -> int:
        rs = 0
        n = len(nums)
        cr = 0
        total = sum(nums)
        for i in range(0, n - 1):
            cr = cr + nums[i]
            if cr >= total - cr:
                rs += 1
        return rs


sl = Solution()
print(sl.waysToSplitArray([10, 4, -8, 7]))  # 2
print(sl.waysToSplitArray([2, 3, 1, 0]))  # 2
