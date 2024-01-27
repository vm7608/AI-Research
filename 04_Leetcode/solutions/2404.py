# 2404. Most Frequent Even Element
# Difficulty: easy
# Tag: implementation
# Link: https://leetcode.com/problems/most-frequent-even-element/

# Description: Given an integer array nums, return the most frequent even element.
# If there is a tie, return the smallest one. If there is no such element, return -1.

from typing import List


class Solution:
    def mostFrequentEven(self, nums: List[int]) -> int:
        d = {}
        for i in nums:
            if i % 2 == 0:
                d[i] = d.get(i, 0) + 1
                
        # if there is no even number, return -1
        if len(d) == 0:
            return -1
        
        # return the key that has the most frequent,
        # if there are more than one, return the smallest one
        mostFreq = max(d.values())
        rs = [k for k, v in d.items() if v == mostFreq]
        return min(rs)


sl = Solution()
print(sl.mostFrequentEven([12, 4, 6]))
