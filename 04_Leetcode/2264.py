# 2264. Largest 3-Same-Digit Number in String
# Difficulty: easy
# Tags: string, implementation
# Link: https://leetcode.com/problems/largest-3-same-digit-number-in-string/

# Description: You are given a string num representing a large integer.
# An integer is good if it meets the following conditions:
# It is a substring of num with length 3.
# It consists of only one unique digit.
# Return the maximum good integer as a string
# or an empty string "" if no such integer exists.
# Note:
# A substring is a contiguous sequence of characters within a string.
# There may be leading zeroes in num or a good integer.

# Idea: loop through num, find the first 3 consecutive digits that are the same
# compare that string with the current result, return the larger one

class Solution:
    def largestGoodInteger(self, num: str) -> str:
        rs = ""
        for i in range(len(num) - 2):
            if num[i] == num[i + 1] == num[i + 2]:
                rs = max(rs, num[i:i+3])
        return rs


sl = Solution()
print(sl.largestGoodInteger("6777133339"))  # "777"
