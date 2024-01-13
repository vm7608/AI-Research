# 2278. Percentage of Letter in String
# Difficulty: Easy
# Tags: implementation
# Link: https://leetcode.com/problems/percentage-of-letter-in-string/

# Description: Given a string s and a character letter,
# return the percentage of characters in s that equal letter
# rounded down to the nearest whole percent.

# Idea: use count() to count the number of letter in s
# then return the percentage

class Solution:
    def percentageLetter(self, s: str, letter: str) -> int:
        return int(s.count(letter) / len(s) * 100)


sl = Solution()
print(sl.percentageLetter("foobar", "o"))  # 33
print(sl.percentageLetter("jjjj", "k"))  # 0
