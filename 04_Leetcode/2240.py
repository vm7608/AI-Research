# 2240. Number of Ways to Buy Pens and Pencils
# Difficulty: medium
# Tags: math
# Link: https://leetcode.com/problems/number-of-ways-to-buy-pens-and-pencils/

# Desciption: There are two types of products: pens and pencils. Each product has a different price. 
# Given the prices of both pens and pencils and the total amount of money that you have,
# find the number of ways to buy one pen and one pencil such that the total money spent is less than or equal to totalMoney.

# Idea: get the small and big price,
# check if total is smaller than small, return 1
# check if total is smaller than big, return total // small + 1
# else loop through the big price case and add the number of small price can be bought to the result

class Solution:
    def waysToBuyPensPencils(self, total: int, cost1: int, cost2: int) -> int:
        small, big = min(cost1, cost2), max(cost1, cost2)

        if total < small:
            return 1

        if total < big:
            return total // small + 1
        
        rs = 0
        for i in range(total // big + 1):
            rs += (total - i * big) // small + 1
        return rs


sl = Solution()
print(sl.waysToBuyPensPencils(20, 10, 5))
