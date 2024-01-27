# 1833. Maximum Ice Cream Bars
# Difficulty: Medium
# Tags: sorting
# Link: https://leetcode.com/problems/maximum-ice-cream-bars/

# Description:
# At the store, there are n ice cream bars.
# You are given an array costs of length n,
# where costs[i] is the price of the ith ice cream bar in coins.
# The boy initially has coins coins to spend,
# and he wants to buy as many ice cream bars as possible.
# Note: The boy can buy the ice cream bars in any order.
# Return the maximum number of ice cream bars the boy can buy with coins coins.
# You must solve the problem by counting sort.

# Idea: sort the array,
# if costs[0] > coins, return 0
# else, loop through the array from 1, add costs[i - 1] to costs[i],
# if costs[i] > coins, return i (index of the first element that is greater than coins)
# if no return, return len(costs) (all elements are less than coins)

from typing import List


class Solution:
    def maxIceCream(self, costs: List[int], coins: int) -> int:
        costs.sort()
        if costs[0] > coins:
            return 0
        for i in range(1, len(costs)):
            costs[i] += costs[i-1]
            if costs[i] > coins:
                return i
        return len(costs)


sl = Solution()
print(sl.maxIceCream([1, 3, 2, 4, 1], 7))  # 4
