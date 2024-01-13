# 2379. Minimum Recolors to Get K Consecutive Black Blocks
# Difficulty: easy
# Tags: implementation
# Link: https://leetcode.com/problems/minimum-recolors-to-get-k-consecutive-black-blocks/

# Description: given a 0-indexed string blocks of length n, where blocks[i] is either 'W' or 'B', representing the color of the ith block. 
# The characters 'W' and 'B' denote the colors white and black, respectively.
# Given an integer k, which is the desired number of consecutive black blocks.
# In one operation, you can recolor a white block such that it becomes a black block.
# Return the minimum number of operations needed such that there is at least one occurrence of k consecutive black blocks.

# Idea: loop from 0 to len(blocks) - k + 1, calculate the number of W in the sub string,
# then compare the number of W with the minBlock, return the minBlock

class Solution:
    def minimumRecolors(self, blocks: str, k: int) -> int:
        minBlock = len(blocks)
        for i in range(len(blocks) - k + 1):
            count = 0
            sub = blocks[i : i + k]
            # calculate the number of W in the sub string
            # and compare with the minBlock
            count = sub.count("W")
            minBlock = count if count < minBlock else minBlock
        return minBlock


sl = Solution()
print(sl.minimumRecolors("WBWW", 2))
