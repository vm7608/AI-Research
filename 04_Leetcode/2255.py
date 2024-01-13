# 2255. Count Prefixes of a Given String
# Difficulty: easy
# Tags: string, implementation
# Link: https://leetcode.com/problems/count-prefixes-of-a-given-string/

# Description: You are given a string array words and a string s,
# where words[i] and s comprise only of lowercase English letters.
# Return the number of strings in words that are a prefix of s.
# A prefix of a string is a substring that occurs at the beginning of the string.
# A substring is a contiguous sequence of characters within a string.

# Idea: use startswith() to check if the string is a prefix of s
# loop through words, count the number of words that are prefixes of s

from typing import List


class Solution:
    def countPrefixes(self, words: List[str], s: str) -> int:
        return sum(s.startswith(word) for word in words)


sl = Solution()
print(sl.countPrefixes(["a", "b", "c", "ab", "bc", "abc"], "abc"))  # 3
print(sl.countPrefixes(["a", "a"], "aa"))  # 2
