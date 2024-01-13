# 2418. Sort the People
# Difficulty: easy
# Tag: sorting
# Link: https://leetcode.com/problems/sort-the-people/

# Description: You are given two lists names and heights where names[i] and heights[i]
# represent the name and height of the ith person respectively.
# Return the list of names in order of decreasing height.

# Idea: zip the names and heights then sort them by heights, return the names

from typing import List


class Solution:
    def sortPeople(self, names: List[str], heights: List[int]) -> List[str]:
        # zip the names and heights then sort them by heights, return the names
        result = [x for _, x in sorted(zip(heights, names), reverse=True)]
        return result


sl = Solution()
names = ["John", "Alex", "Jack", "Amy", "Bill"]
heights = [5, 6, 7, 8, 9]

rs = sl.sortPeople(names, heights)
print(rs)
