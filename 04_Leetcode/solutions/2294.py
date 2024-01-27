# 2294. Partition Array Such That Maximum Difference Is K
# Difficulty: Medium
# Tags: sorting, implementation
# Link: https://leetcode.com/problems/partition-array-such-that-maximum-difference-is-k/

# Description:
# You are given an integer array nums and an integer k.
# You may partition nums into one or more subsequences such
# that each element in nums appears in exactly one of the subsequences.
# Return the minimum number of subsequences needed such
# that the difference between the maximum and minimum values in each subsequence is at most k.
# A subsequence is a sequence that can be derived from another sequence by deleting some
# or no elements without changing the order of the remaining elements.

# Idea: sort the array, then loop through the array
# if the current element is greater than the previous element + k
# then we need a new subsequence
# return the number of subsequences

from typing import List


class Solution:
    def partitionArray(self, nums: List[int], k: int) -> int:
        nums.sort()
        ans = 1
        temp = nums[0]
        for i in range(1, len(nums)):
            if nums[i] > (temp + k):
                ans += 1
                temp = nums[i]
        return ans


sl = Solution()
print(sl.partitionArray([3, 6, 1, 2, 5], 2))  # 2
print(sl.partitionArray([1, 2, 3], 1))  # 2
print(sl.partitionArray([2, 2, 4, 5], 0))  # 3
