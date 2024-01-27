# 2303. Calculate Amount Paid in Taxes
# Difficulty: Easy
# Tags: implementation
# Link: https://leetcode.com/problems/calculate-amount-paid-in-taxes/

# Description:
# You are given a 0-indexed 2D integer array brackets
# where brackets[i] = [upperi, percenti] means that the ith tax bracket has an upper bound of upperi
# and is taxed at a rate of percenti.
# The brackets are sorted by upper bound (i.e. upperi-1 < upperi for 0 < i < brackets.length).
# Tax is calculated as follows:
# The first upper0 dollars earned are taxed at a rate of percent0.
# The next upper1 - upper0 dollars earned are taxed at a rate of percent1.
# The next upper2 - upper1 dollars earned are taxed at a rate of percent2.
# And so on.
# You are given an integer income representing the amount of money you earned.
# Return the amount of money that you have to pay in taxes.
# Answers within 10^-5 of the actual answer will be accepted.

# Idea: loop through brackets,
# compare the upper bound of current bracket and income
# get the smaller one, then calculate the tax
# add the tax to ans, update prev to current upper bound

from typing import List


class Solution:
    def calculateTax(self, brackets: List[List[int]], income: int) -> float:
        ans = prev = 0
        for upper, percent in brackets:
            upper = min(upper, income)
            ans += (upper - prev)*percent/100
            prev = upper
        return ans


sl = Solution()

# 2.65 = 3 * 0.5 + 4 * 0.1 + 3 * 0.25
print(sl.calculateTax([[3, 50], [7, 10], [12, 25]], 10))
# 0.25 = 1 * 0 + 1 * 0.25 + 0 * 0.5
print(sl.calculateTax([[1, 0], [4, 25], [5, 50]], 2))
