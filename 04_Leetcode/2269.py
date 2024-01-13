# 2269. Find the K-Beauty of a Number
# Difficulty: easy
# Tags: string, implementation
# Link: https://leetcode.com/problems/find-the-k-beauty-of-a-number/

# Description: The k-beauty of an integer num is defined as the number of substrings
# of num when it is read as a string that meet the following conditions:
# It has a length of k.
# It is a divisor of num.
# Given integers num and k, return the k-beauty of num.
# Note:
# Leading zeros are allowed.
# 0 is not a divisor of any value.
# A substring is a contiguous sequence of characters in a string.

# Idea: convert num to string, loop through num,
# check if the substring is a divisor of num
# if it is, increase the result by 1 and return the result
# if it is not, continue

class Solution:
    def divisorSubstrings(self, num: int, k: int) -> int:
        num = str(num)
        rs = 0
        for i in range(len(num) - k + 1):
            if int(num[i:i+k]) == 0:
                continue
            if int(num) % int(num[i:i+k]) == 0:
                rs += 1
        return rs


sl = Solution()
print(sl.divisorSubstrings(430043, 2))  # 2
