# 1829. Maximum XOR for Each Query
# Difficulty: Medium
# Tags: bitwise
# Link: https://leetcode.com/problems/maximum-xor-for-each-query/

# Description:
# You are given a sorted array nums of n non-negative integers and an integer maximumBit.
# You want to perform the following query n times:
# Find a non-negative integer k < 2^maximumBit
# such that nums[0] XOR nums[1] XOR ... XOR nums[nums.length-1] XOR k is maximized.
# k is the answer to the ith query.
# Remove the last element from the current array nums.
# Return an array answer, where answer[i] is the answer to the ith query.

# Idea:
# max_value is 2^maximumBit - 1
# xor ^ k = max_values -> k = xor ^ max_value
# loop through nums, xor ^= nums[i], append xor ^ max_value to answer
# return reversed answer (because we append from end to start)

from typing import List


class Solution:
    def getMaximumXor(self, nums: List[int], maximumBit: int) -> List[int]:
        ans = []
        max_value = 2 ** maximumBit - 1
        xor = 0
        for num in nums:
            xor ^= num
            ans.append(xor ^ max_value)
        return ans[:: -1]


sl = Solution()
print(sl.getMaximumXor([0, 1, 1, 3], 2))  # [0, 3, 2, 3]
