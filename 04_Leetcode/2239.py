# 2239. Find Closest Number to Zero
# Difficulty: easy
# Tags: math
# Link: https://leetcode.com/problems/find-closest-number-to-zero/

# Description: Given an integer array nums of size n, 
# return the number with the value closest to 0 in nums. 
# If there are multiple answers, return the number with the largest value.

# Idea: loop through the array and compare abs of each pair of numbers and return the largest one with the smallest abs value

from typing import List


class Solution:
    def findClosestNumber(self, nums: List[int]) -> int:
        rs = nums[0]
        # set the first number as the closest
        closest = abs(nums[0])
        for i in range(1, len(nums)):
            # if smaller, update the closest and rs
            if abs(nums[i]) < closest:
                rs = nums[i]
                closest = abs(nums[i])
            # if equal, update the rs (largest one)
            if abs(nums[i]) == closest:
                rs = nums[i] if nums[i] > rs else rs
        return rs


sl = Solution()
print(sl.findClosestNumber([2, -1, 1]))
