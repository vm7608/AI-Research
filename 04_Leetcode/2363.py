# 2363. Merge Similar Items
# Difficulty: easy
# Tag: implementation
# Link: https://leetcode.com/problems/merge-similar-items/

# Description: given two 2D integer arrays, items1 and items2, representing two sets of items. 
# Each array items has the following properties:
# items[i] = [valuei, weighti] where valuei represents the value and weighti represents the weight of the ith item.
# The value of each item in items is unique.
# Return a 2D integer array ret where ret[i] = [valuei, weighti], with weighti being the sum of weights of all items with value valuei.
# Note: ret should be returned in ascending order by value.

# Idea: merge two lists, sort the list by the first element,
# loop and calculate the sum of weight of similar items
# then remove the similar items after calculation

from typing import List


class Solution:
    def mergeSimilarItems(
        self, items1: List[List[int]], items2: List[List[int]]) -> List[List[int]]:
        # merge two lists
        items = items1 + items2
        # sort the list by the first element
        items.sort(key=lambda x: x[0])
        # merge the items
        i = 0
        while i < len(items) - 1:
            if items[i][0] == items[i + 1][0]:
                items[i][1] += items[i + 1][1]
                items.pop(i + 1)
            else:
                i += 1
        return items


sl = Solution()
print(sl.mergeSimilarItems(
    [[1, 5], [2, 5], [3, 5], [6, 6]], [[1, 2], [2, 3], [3, 4]]))
