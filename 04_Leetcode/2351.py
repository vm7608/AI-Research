# 2351. First Letter to Appear Twice

# Difficulty: Easy
# Tag: hashing
# Link: https://leetcode.com/problems/first-letter-to-appear-twice/

# Description: Given a string s consisting of lowercase English letters, return the first letter to appear twice.
# Note:
# A letter a appears twice before another letter b if the second occurrence of a is before the second occurrence of b.
# s will contain at least one letter that appears twice.

# Idea: use a dictionary to store the letters that have appeared
# if a letter appears twice, return it

class Solution:
    def repeatedCharacter(self, s: str) -> str:
        d = {}
        for c in s:
            d[c] = d.get(c, 0) + 1
            if d[c] == 2:
                return c
        return -1
        
        
sl = Solution()
print(sl.repeatedCharacter("abccbaacz")) # a
    