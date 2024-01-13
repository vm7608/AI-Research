# 2342. Max Sum of a Pair With Equal Sum of Digits
# Difficulty: Medium
# Tag: hashing, sorting
# Link: https://leetcode.com/problems/max-sum-of-a-pair-with-equal-sum-of-digits/

# Description: You are given a 0-indexed array nums consisting of positive integers.
# You can choose two indices i and j, such that i != j,
# and the sum of digits of the number nums[i] is equal to that of nums[j].
# Return the maximum value of nums[i] + nums[j]
# that you can obtain over all possible indices i and j that satisfy the conditions.

# Idea: use a dict with key is sum of digit and value is a list of numbers
# after the dictionary is built, loop through the dict,
# if value list have more than 1 number, sort the list in descending order
# and return the sum of the first 2 numbers (the largest pair)
from typing import List


class Solution:
    def maximumSum(self, nums: List[int]) -> int:
        d = {}
        for n in nums:
            digit_sum = sum(map(int, str(n)))
            if digit_sum not in d.keys():
                d[digit_sum] = [n]
            else:
                d[digit_sum].append(n)

        ans = -1
        for val in d.values():
            if len(val) > 1:
                # sort descending to get the largest pair
                # in the first 2 numbers
                val = sorted(val, reverse=True)
                ans = max(ans, val[0] + val[1])
        return ans


sl = Solution()
print(sl.maximumSum([18, 43, 36, 13, 7]))  # 54
print(sl.maximumSum([10, 12, 19, 14]))  # -1
print(sl.maximumSum([9, 2, 2, 5]))  # 4
print(sl.maximumSum([368, 369, 307, 304, 384, 138, 90, 279, 35,
      396, 114, 328, 251, 364, 300, 191, 438, 467, 183]))  # 835
