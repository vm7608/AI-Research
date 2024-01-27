# 2259. Remove Digit From Number to Maximize Result
# Difficulty: easy
# Tags: implementation
# Link: https://leetcode.com/problems/remove-digit-from-number-to-maximize-result/

# Description: You are given a string number representing a positive integer and a character digit.
# Return the resulting string after removing exactly one occurrence of digit from number
# such that the value of the resulting string in decimal form is maximized.
# The test cases are generated such that digit occurs at least once in number.

# Idea: loop through number, find the first digit that is smaller than the next digit,
# remove that digit, return the result
# if there is no such digit, loop through number from the end,
# find the first digit that is equal to digit, remove that digit, return the result

class Solution:
    def removeDigit(self, number: str, digit: str) -> str:
        i = 0
        for i in range(len(number) - 1):
            if digit == number[i] and number[i] < number[i + 1]:
                break
        else:
            for i in range(len(number) - 1, -1, -1):
                if number[i] == digit:
                    break
        number = number[:i] + number[i+1:]
        return number


sl = Solution()
print(sl.removeDigit("1231", "1"))  # 231
print(sl.removeDigit("551", "5"))  # 51
