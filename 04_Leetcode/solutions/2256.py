# 2256. Minimum Average Difference
# Difficulty: Medium
# Tags: math
# Link: https://leetcode.com/problems/minimum-average-difference/

# Description: You are given a 0-indexed integer array nums of length n.
# The average difference of the index i is the absolute difference
# between the average of the first i + 1 elements of nums and the average of the last n - i - 1 elements.
# Both averages should be rounded down to the nearest integer.
# Return the index with the minimum average difference.
# If there are multiple such indices, return the smallest one.
# Note:
# The absolute difference of two numbers is the absolute value of their difference.
# The average of n elements is the sum of the n elements divided (integer division) by n.
# The average of 0 elements is considered to be 0.

# Idea: first calculate the total sum of the array,
# then loop through the array and calculate the absolute average difference of each index.
# Return the index with the minimum average difference.

from typing import List


class Solution:
    def minimumAverageDifference(self, nums: List[int]) -> int:
        rs = 0
        min_diff = 0
        n = len(nums)
        total = sum(nums)
        current = 0
        for i in range(n):
            current += nums[i]
            if i < n-1:
                diff = abs(current//(i+1) - (total-current)//(n-i-1))
            else:
                diff = abs(current//(i+1))
            if i == 0 or diff < min_diff:
                min_diff = diff
                rs = i
        return rs


sl = Solution()
print(sl.minimumAverageDifference([2, 5, 3, 9, 5, 3]))
print(sl.minimumAverageDifference([0]))
