# 2335. Minimum Amount of Time to Fill Cups
# Difficulty: easy
# Tags: math
# Link: https://leetcode.com/problems/minimum-amount-of-time-to-fill-cups/

# Description:
# You have a water dispenser that can dispense cold, warm, and hot water.
# Every second, you can either fill up 2 cups with different types of water, or 1 cup of any type of water.
# You are given a 0-indexed integer array amount of length 3 where amount[0], amount[1], and amount[2]
# denote the number of cold, warm, and hot water cups you need to fill respectively.
# Return the minimum number of seconds needed to fill up all the cups.

# Idea:
# the minimum number of seconds needed to fill up all the cups is
# max(max(amount), ceil(sum(amount) / 2))

# answer >= max(amount)
# Because each time,
# one type can reduce at most 1 cup,

# answer >= ceil(sum(amount)
# Because each time,
# we can fill up to 2 cups,

from typing import List


class Solution:
    def fillCups(self, amount: List[int]) -> int:
        return max(max(amount), (sum(amount) + 1) // 2)


sl = Solution()
print(sl.fillCups([1, 4, 2]))  # 4
