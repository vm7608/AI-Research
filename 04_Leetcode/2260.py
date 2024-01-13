# 2260. Minimum Consecutive Cards to Pick Up
# Difficulty: Medium
# Tags: hashing, math
# Link: https://leetcode.com/problems/minimum-consecutive-cards-to-pick-up/

# Description: You are given an integer array cards
# where cards[i] represents the value of the ith card.
# A pair of cards are matching if the cards have the same value.
# Return the minimum number of consecutive cards you have to pick up
# to have a pair of matching cards among the picked cards.
# If it is impossible to have matching cards, return -1.

# Idea: use a dict to store the last index of each card,
# if the card not in dict, store the index of the card
# if we find a card that has already been picked up,
# compare num of cards between the two cards and update the min
# return min

from typing import List


class Solution:
    def minimumCardPickup(self, cards: List[int]) -> int:
        d = {}
        min_card = -1
        for i in range(len(cards)):
            if cards[i] not in d:
                d[cards[i]] = i
            else:
                if min_card > (i-d[cards[i]]) or min_card == -1:
                    min_card = (i - d[cards[i]]) + 1
                d[cards[i]] = i
        return min_card


sl = Solution()
print(sl.minimumCardPickup([3, 4, 2, 3, 4, 7]))  # 4
print(sl.minimumCardPickup([1, 0, 5, 3]))  # -1
print(sl.minimumCardPickup([95, 11, 8, 65, 5, 86, 30, 27, 30, 73,
      15, 91, 30, 7, 37, 26, 55, 76, 60, 43, 36, 85, 47, 96, 6]))  # 3
