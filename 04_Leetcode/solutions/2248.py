# 2248. Intersection of Multiple Arrays
# Difficulty: Easy
# Tag: hashing
# Link: https://leetcode.com/problems/intersection-of-multiple-arrays/

# Description: Given a 2D integer array nums where nums[i] is a non-empty array of distinct positive integers,
# return the list of integers that are present in each array of nums sorted in ascending order.

# Idea: use set to save the first row, then use intersection to find the common elements in the next rows

from typing import List


class Solution:
    def intersection(self, nums: List[List[int]]) -> List[int]:
        rs = set(nums[0])
        for i in range(1, len(nums)):
            rows = set(nums[i])
            rs &= rows
        return sorted(list(rs))


sl = Solution()
print(sl.intersection([[3, 1, 2, 4, 5], [1, 2, 3, 4], [3, 4, 5, 6]]))  # [3,4]
