# 2288. Apply Discount to Prices
# Difficulty: Medium
# Tags: implementation
# Link: https://leetcode.com/problems/apply-discount-to-prices/

# Description:
# A sentence is a string of single-space separated words
# where each word can contain digits, lowercase letters, and the dollar sign '$'.
# A word represents a price if it is a sequence of digits preceded by a dollar sign.
# For example, "$100", "$23", and "$6" represent prices while "100", "$", and "$1e5" do not.
# You are given a string sentence representing a sentence and an integer discount.
# For each word representing a price, apply a discount of discount% on the price and update the word in the sentence.
# All updated prices should be represented with exactly two decimal places.
# Return a string representing the modified sentence.
# Note that all prices will contain at most 10 digits.

# Idea: split the sentence into words
# loop through each word
# if word starts with "$" and the rest of the word is a number
# apply discount to the number

class Solution:
    def discountPrices(self, sentence: str, discount: int) -> str:
        rs = ""
        words = sentence.split()

        for word in words:
            if word[0] == "$" and word[1:].isdigit():
                price = float(word[1:])
                price = price * (100 - discount) / 100
                rs += "$%.2f " % price
            else:
                rs += word + " "
        return rs[:-1]


sl = Solution()
# "this book is $4.50"
print(sl.discountPrices("there are $1 $2 and 5$ candies in the shop", 50))
# "1 2 $0.00 4 $0.00 $0.00 7 8$ $0.00 $10$"
print(sl.discountPrices("1 2 $3 4 $5 $6 7 8$ $9 $10$", 100))
