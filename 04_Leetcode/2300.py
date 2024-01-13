# 2300. Successful Pairs of Spells and Potions
# Difficulty: medium
# Tags: sorting, binary search
# Link: https://leetcode.com/problems/successful-pairs-of-spells-and-potions/

# Description:
# You are given two positive integer arrays spells and potions, of length n and m respectively,
# where spells[i] represents the strength of the ith spell and potions[j] represents the strength of the jth potion.
# You are also given an integer success.
# A spell and potion pair is considered successful if the product of their strengths is at least success.
# Return an integer array pairs of length n
# where pairs[i] is the number of potions that will form a successful pair with the ith spell.

# Idea: sort potions first,
# for each spell, use binary search to find the smallest potion that multiplies with the spell to be >= success
# the number of potions that will form a successful pair with the ith spell is m - left

from typing import List


class Solution:
    def successfulPairs(self, spells: List[int], potions: List[int], success: int) -> List[int]:
        n = len(spells)
        m = len(potions)
        pairs = [0] * n

        potions.sort()

        for i in range(n):
            spell = spells[i]
            left = 0
            right = m - 1
            while left <= right:
                mid = left + (right - left) // 2
                product = spell * potions[mid]
                if product >= success:
                    right = mid - 1
                else:
                    left = mid + 1
            pairs[i] = m - left
        return pairs


sl = Solution()
print(sl.successfulPairs([5, 1, 3], [1, 2, 3, 4, 5], 7))  # [4,0,3]
