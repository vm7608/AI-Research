# 2389. Longest Subsequence With Limited Sum
# Difficulty: easy
# Tag: sorting
# Link: https://leetcode.com/problems/longest-subsequence-with-limited-sum/

# Description: given an integer array nums of length n, and an integer array queries of length m.
# Return an array answer of length m where answer[i] is the maximum size of a subsequence
# that you can take from nums such that the sum of its elements is less than or equal to queries[i].
# A subsequence is an array that can be derived from another array by deleting some or no elements
# without changing the order of the remaining elements.

# Idea: first sort the nums, then loop through the queries,
# for each query, loop through the nums to calculate the sum

from typing import List


class Solution:
    def answerQueries(self, nums: List[int], queries: List[int]) -> List[int]:

        # check if nums have only one element
        if len(nums) == 1:
            return [0 if i < nums[0] else 1 for i in queries]

        nums.sort()
        rs = [0] * len(queries)
        
        for i in range(len(queries)):
            sub_sum = 0
            
            # if queries[i] smaller than the smallest number
            # then the answer is 0 sub
            if queries[i] < nums[0]:
                rs[i] = 0
                continue
            
            for j in range(len(nums) - 1):
                sub_sum += nums[j]
                # if queries[i] smaller than sum + next number
                # then the answer is j + 1 sub and break
                if queries[i] < sub_sum + nums[j + 1]:
                    rs[i] = j + 1
                    break

                if j == len(nums) - 2:
                    rs[i] = len(nums)
                    break
        return rs


sl = Solution()
print(sl.answerQueries([2, 3, 4, 5], [1]))
print(sl.answerQueries([1, 2, 3, 4, 5], [5, 2, 6]))
print(sl.answerQueries([624082], [972985, 564269,
      607119, 693641, 787608, 46517, 500857, 140097]))
