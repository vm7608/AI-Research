# 2341. Maximum Number of Pairs in Array
# Difficulty: easy
# Tags: hashing
# Link: https://leetcode.com/problems/maximum-number-of-pairs-in-an-array/

# Description: You are given a 0-indexed integer array nums. 
# In one operation, you may do the following:
# Choose two integers in nums that are equal.
# Remove both integers from nums, forming a pair.
# The operation is done on nums as many times as possible.
# Return a 0-indexed integer array answer of size 2 where answer[0] is the number of pairs
# that are formed and answer[1] is the number of leftover integers
# in nums after doing the operation as many times as possible.

# Idea: use list to save num, if num is in the list, remove it and increase the number of pairs by 1,
# otherwise, add it to the list
# return the number of pairs and the length of the list (number of leftover integers)

from typing import List


class Solution:
    def numberOfPairs(self, nums: List[int]) -> List[int]:
        numofpairs = 0
        d = {}

        for num in nums:
            if num in d:
                numofpairs += 1
                d.pop(num)
            else:
                d[num] = 0

        return [numofpairs, len(d)]


sl = Solution()
print(sl.numberOfPairs([1, 3, 2, 1, 3, 2, 2]))  # [3, 1]
