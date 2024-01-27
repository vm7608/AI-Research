# 2293. Min Max Game
# Difficulty: easy
# Tags: implementation
# Link: https://leetcode.com/problems/min-max-game/

# Description:
# You are given a 0-indexed integer array nums whose length is a power of 2.
# Apply the following algorithm on nums:
# Let n be the length of nums. If n == 1, end the process. Otherwise, create a new 0-indexed integer array newNums of length n / 2.
# For every even index i where 0 <= i < n / 2, assign the value of newNums[i] as min(nums[2 * i], nums[2 * i + 1]).
# For every odd index i where 0 <= i < n / 2, assign the value of newNums[i] as max(nums[2 * i], nums[2 * i + 1]).
# Replace the array nums with newNums.
# Repeat the entire process starting from step 1.
# Return the last number that remains in nums after applying the algorithm.

# Idea: use a while loop to simulate the process
# then loop through the list, assign the value of nums[i] as max(nums[2 * i], nums[2 * i + 1]) if i is odd
# otherwise, assign the value of nums[i] as min(nums[2 * i], nums[2 * i + 1])
# decrease n by n // 2
# return nums[0] (the last number that remains in nums after process)

from typing import List


class Solution:
    def minMaxGame(self, nums: List[int]) -> int:
        n = len(nums)
        while n > 1:
            for i in range(n // 2):
                if i % 2 == 1:
                    nums[i] = max(nums[2 * i], nums[2 * i + 1])
                else:
                    nums[i] = min(nums[2 * i], nums[2 * i + 1])
            n -= (n // 2)
        return nums[0]


sl = Solution()
print(sl.minMaxGame([1, 3, 5, 2, 4, 8, 2, 2]))  # 1
