# 2305. Fair Distribution of Cookies
# Difficulty: Medium
# Tags: backtracking
# Link: https://leetcode.com/problems/fair-distribution-of-cookies/

# Description:
# You are given an integer array cookies, where cookies[i] denotes the number of cookies in the ith bag.
# You are also given an integer k that denotes the number of children to distribute all the bags of cookies to.
# All the cookies in the same bag must go to the same child and cannot be split up.
# The unfairness of a distribution is defined as the maximum total cookies obtained by a single child in the distribution.
# Return the minimum unfairness of all distributions.

# Idea: use backtracking to try all possible distributions
# for each cookie bag, try to give it to each child
# if all bags are distributed, (index == len(cookies)
# return the maximum total cookies obtained by a single child in the distribution
# otherwise, continue to distribute the remaining bags
# after all distributions are tried, result is the minimum unfairness of all distributions

from typing import List


class Solution:
    def distributeCookies(self, cookies: List[int], k: int) -> int:
        cookies_per_child = [0] * k

        def back_tracking(index: int) -> int:
            if index == len(cookies):
                return max(cookies_per_child)

            result = float('inf')
            for i in range(min(k, index + 1)):
                cookies_per_child[i] += cookies[index]
                result = min(result, back_tracking(index + 1))
                cookies_per_child[i] -= cookies[index]
            return result

        return back_tracking(0)


sl = Solution()
print(sl.distributeCookies([8, 15, 10, 20, 8], 2))  # 31
print(sl.distributeCookies([6, 1, 3, 2, 2, 4, 1, 2], 3))  # 7
