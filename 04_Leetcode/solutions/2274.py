# 2274. Maximum Consecutive Floors Without Special Floors
# Difficulty: Medium
# Tags: sorting
# Link: https://leetcode.com/problems/maximum-consecutive-floors-without-special-floors/

# Description: Alice manages a company and has rented some floors of a building as office space.
# Alice has decided some of these floors should be special floors, used for relaxation only.
# You are given two integers bottom and top,
# which denote that Alice has rented all the floors from bottom to top (inclusive).
# You are also given the integer array special,
# where special[i] denotes a special floor that Alice has designated for relaxation.
# Return the maximum number of consecutive floors without a special floor

# Idea: sort special, then compare the first and last element of special with bottom and top
# then compare the difference between each element of special
# return the max

from typing import List


class Solution:
    def maxConsecutive(self, bottom: int, top: int, special: List[int]) -> int:
        special.sort()
        return max(special[0]-bottom, top-special[-1], max([special[i]-special[i-1] - 1 for i in range(len(special))]))


sl = Solution()
print(sl.maxConsecutive(2, 9, [4, 6]))  # 3
print(sl.maxConsecutive(6, 8, [7, 6, 8]))  # 0
print(sl.maxConsecutive(28, 50, [35, 48]))  # 12
print(sl.maxConsecutive(6, 39, [38]))  # 32
