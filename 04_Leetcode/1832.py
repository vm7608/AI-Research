# 1832. Check if the Sentence Is Pangram
# Difficulty: Easy
# Tags: implementation
# Link: https://leetcode.com/problems/check-if-the-sentence-is-pangram/

# Description:
# A pangram is a sentence where every letter of the English alphabet appears at least once.
# Given a string sentence containing only lowercase English letters,
# return true if sentence is a pangram, or false otherwise.

# Idea:
# If the length of the string is less than 26, return False
# Convert the string to a set to remove duplicates
# Check if the length of the set is 26
# If yes, return True else return False

class Solution:
    def checkIfPangram(self, sentence: str) -> bool:
        return len(set(sentence)) == 26


sl = Solution()
print(sl.checkIfPangram("thequickbrownfoxjumpsoverthelazydog"))  # True
