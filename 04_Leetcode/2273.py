# 2273. Find Resultant Array After Removing Anagrams
# Difficulty: easy
# Tags: string, implementation
# Link: https://leetcode.com/problems/find-resultant-array-after-removing-anagrams/

# Description:You are given a 0-indexed string array words, where words[i] consists of lowercase English letters.
# In one operation, select any index i such that 0 < i < words.length and words[i - 1] and words[i] are anagrams,
# and delete words[i] from words. Keep performing this operation as long as you can select an index that satisfies the conditions.
# Return words after performing all operations.
# It can be shown that selecting the indices for each operation in any arbitrary order will lead to the same result.
# An Anagram is a word or phrase formed by rearranging the letters of a different word or phrase using all the original letters exactly once.
# For example, "dacb" is an anagram of "abdc".

# Idea: sort each word, then compare the current word with the previous word,
# if they are anagrams, then remove the current word

from typing import List


class Solution:
    def removeAnagrams(self, words: List[str]) -> List[str]:
        for i in range(len(words)-1, 0, -1):
            if sorted(words[i]) == sorted(words[i-1]):
                words.pop(i)
        return words


sl = Solution()
print(sl.removeAnagrams(["abba", "baba", "bbaa", "cd", "cd"]))  # ["abba","cd"]
