# 2348. Number of Zero-Filled Subarrays
# Difficulty: medium
# Tags: math
# Link: https://leetcode.com/problems/number-of-zero-filled-subarrays/

# Description: Given an integer array nums, return the number of subarrays filled with 0.
# A subarray is a contiguous non-empty sequence of elements within an array.

# Idea: number of subarrays added by a new 0 is the length of the streak of 0s
# For exp: [0,0,0,0] -> streak = 4 -> 4 + 3 + 2 + 1 = 10 subarrays
# Just loop through the array and count the streak of 0s

from typing import List


class Solution:
    def zeroFilledSubarray(self, nums: List[int]) -> int:
        rs = 0
        streak = 0
        for i in range(len(nums)):
            if nums[i] == 0:
                streak += 1
                rs += streak

            if nums[i] != 0:
                streak = 0
                continue
        return rs


sl = Solution()
print(sl.zeroFilledSubarray([1, 3, 0, 0, 2, 0, 0, 4]))  # 6
