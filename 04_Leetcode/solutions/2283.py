# 2283. Check if Number Has Equal Digit Count and Digit Value
# Difficulty: Easy
# Tags: implementation
# Link: https://leetcode.com/problems/check-if-number-has-equal-digit-count-and-digit-value/

# Description: You are given a 0-indexed string num of length n consisting of digits.
# Return true if for every index i in the range 0 <= i < n, the digit i occurs num[i] times in num, otherwise return false.

# Idea:
# Constraints: 1 <= n <= 10
# ==> use array with size 10 to count the number of digits in num
# then compare every values of count with num

class Solution:
    def digitCount(self, num: str) -> bool:
        n = len(num)
        count = [0] * 10
        for i in range(n):
            count[int(num[i])] += 1
        for i in range(n):
            if count[i] != int(num[i]):
                return False
        return True


sl = Solution()
print(sl.digitCount("1210"))  # True
print(sl.digitCount("030"))  # False
