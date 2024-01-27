# 2433. Find The Original Array of Prefix Xor
# Difficulty: Medium
# Tags: math
# Link: https://leetcode.com/problems/find-the-original-array-of-prefix-xor/

# Description: you are given an integer array pref of size n. Find and return the array arr of size n that satisfies:
# pref[i] = arr[0] ^ arr[1] ^ ... ^ arr[i].
# Note that ^ denotes the bitwise-xor operation.
# It can be proven that the answer is unique.

# Idea: if pref[i] = arr[0] ^ arr[1] ^ ... ^ arr[i], then arr[i] = pref[i] ^ arr[0] ^ arr[1] ^ ... ^ arr[i-1]
# so we can find arr[i] by pref[i] and arr[i-1]
# loop from n-1 to 0, we can find the array arr

from typing import List


class Solution:
    def findArray(self, pref: List[int]) -> List[int]:
        arr = pref.copy()
        n = len(arr)
        for i in range(n-1, 0, -1):
            arr[i] ^= arr[i-1]
        return arr


sl = Solution()
print(sl.findArray([5, 2, 0, 3, 1]))
