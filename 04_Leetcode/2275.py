# 2275. Largest Combination With Bitwise AND Greater Than Zero
# Difficulty: Medium
# Tags: bitwise
# Link: https://leetcode.com/problems/largest-combination-with-bitwise-and-greater-than-zero/

# Description: The bitwise AND of an array nums is the bitwise AND of all integers in nums.
# For example, for nums = [1, 5, 3], the bitwise AND is equal to 1 & 5 & 3 = 1.
# Also, for nums = [7], the bitwise AND is 7.
# You are given an array of positive integers candidates.
# Evaluate the bitwise AND of every combination of numbers of candidates.
# Each number in candidates may only be used once in each combination.
# Return the size of the largest combination of candidates with a bitwise AND greater than 0.

# Idea:
# Constraints: 1 <= candidates[i] <= 107 => candidates[i] <= 2^24
# ==> we can use a 24-bit array to store the number of times each bit appears in the array
# for each candidate, & it with 1, if the result is 1, that means the last bit of the candidate is 1
# then we shift the candidate to the right by 1 and repeat the process
# we do this until the candidate is 0
# finally, return the max of the array

from typing import List


class Solution:
    def largestCombination(self, candidates: List[int]) -> int:
        arr = [0] * 24
        for candidate in candidates:
            index = 0
            while candidate > 0:
                if candidate & 1 == 1:
                    arr[index] += 1
                candidate >>= 1
                index += 1
        return max(arr)


sl = Solution()
print(sl.largestCombination([16, 17, 71, 62, 12, 24, 14]))  # 4
