# 2315. Count Asterisks
# Difficulty: Easy
# Tags: implementation
# Link: https://leetcode.com/problems/count-asterisks/

# Description:
# You are given a string s, where every two consecutive vertical bars '|' are grouped into a pair.
# In other words, the 1st and 2nd '|' make a pair, the 3rd and 4th '|' make a pair, and so forth.
# Return the number of '*' in s, excluding the '*' between each pair of '|'.
# Note that each '|' will belong to exactly one pair.

# Idea: split by "|" and count * in even index 0, 2, 4, ...

class Solution:
    def countAsterisks(self, s: str) -> int:
        temp = s.split("|")
        rs = 0
        for i in range(len(temp)):
            if i % 2 == 0:
                rs += temp[i].count("*")
        return rs


sl = Solution()
print(sl.countAsterisks("l|*e*et|c**o|*de|"))  # 2
print(sl.countAsterisks("iamprogrammer"))  # 0
print(sl.countAsterisks("yo|uar|e**|b|e***au|tifu|l"))  # 5
