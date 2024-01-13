# 2309. Greatest English Letter in Upper and Lower Case
# Difficulty: Easy
# Tags: implementation
# Link: https://leetcode.com/problems/greatest-english-letter-in-upper-and-lower-case/

# Description:
# Given a string of English letters s, return the greatest English letter
# which occurs as both a lowercase and uppercase letter in s.
# The returned letter should be in uppercase.
# If no such letter exists, return an empty string.
# An English letter b is greater than another letter a
# if b appears after a in the English alphabet.

# Idea:
# Convert the string to a set to remove duplicates
# Loop through the alphabet from Z to A,
# check if both upper and lower case of the current letter are in the set
# if yes, return the current letter in upper case (it is the greatest letter)

class Solution:
    def greatestLetter(self, s: str) -> str:
        s = set(s)
        upper, lower = ord('Z'), ord('z')
        for i in range(26):
            if chr(upper - i) in s and chr(lower - i) in s:
                return chr(upper - i)
        return ''


sl = Solution()
print(sl.greatestLetter("lEeTcOdE"))  # E
print(sl.greatestLetter("arRAzFif"))  # R
print(sl.greatestLetter("AbCdEfGhIjK"))  # ""
