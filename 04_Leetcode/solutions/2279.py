# 2279. Maximum Bags With Full Capacity of Rocks
# Difficulty: Medium
# Tags: sorting
# Link: https://leetcode.com/problems/maximum-bags-with-full-capacity-of-rocks/

# Description: You have n bags numbered from 0 to n - 1.
# You are given two 0-indexed integer arrays capacity and rocks.
# The ith bag can hold a maximum of capacity[i] rocks and currently contains rocks[i] rocks.
# You are also given an integer additionalRocks,
# the number of additional rocks you can place in any of the bags.
# Return the maximum number of bags that could have full capacity
# after placing the additional rocks in some bags.

# Idea: calculate the number of rocks that need to be added to each bag
# then sort and loop through the list
# if the number need to be added <= additionalRocks then add 1 to the result
# else break
# return the result

from typing import List


class Solution:
    def maximumBags(self, capacity: List[int], rocks: List[int], additionalRocks: int) -> int:
        need_to_add = [capacity[i] - rocks[i] for i in range(len(capacity))]
        need_to_add.sort()
        rs = 0
        for i in range(len(need_to_add)):
            if need_to_add[i] <= additionalRocks:
                additionalRocks -= need_to_add[i]
                rs += 1
            else:
                break
        return rs


sl = Solution()
print(sl.maximumBags([2, 3, 4, 5], [1, 2, 4, 4], 2))  # 3
print(sl.maximumBags([10, 2, 2], [2, 2, 0], 100))  # 3
