# 2280. Minimum Lines to Represent a Line Chart
# Difficulty: Medium
# Tags: geometry
# Link: https://leetcode.com/problems/minimum-lines-to-represent-a-line-chart/

# Description: You are given a 2D integer array stockPrices where stockPrices[i] = [dayi, pricei]
# indicates the price of the stock on day dayi is pricei.
# A line chart is created from the array by plotting the points on an XY plane
# with the X-axis representing the day and the Y-axis representing the price and connecting adjacent points.
# Return the minimum number of lines needed to represent the line chart.

# Idea: first sort the array by day, then calculate the ratio of day_diff and price_diff
# if the ratio is different, then we need a new line to represent the line chart

from typing import List


class Solution:
    def minimumLines(self, stockPrices: List[List[int]]) -> int:
        stockPrices.sort(key=lambda x: x[0])
        n = len(stockPrices)

        if n == 1:
            return 0

        ans = 1  # n >= 2 so we need at least 1 line
        for i in range(1, n-1):
            day_diff1 = stockPrices[i][0] - stockPrices[i-1][0]
            price_diff1 = stockPrices[i][1] - stockPrices[i-1][1]
            day_diff2 = stockPrices[i+1][0] - stockPrices[i][0]
            price_diff2 = stockPrices[i+1][1] - stockPrices[i][1]
            # calculate the ratio of day_diff and price_diff
            if day_diff1 * price_diff2 != day_diff2 * price_diff1:
                ans += 1
        return ans


sl = Solution()
print(sl.minimumLines([[1, 7], [2, 6], [3, 5], [
      4, 4], [5, 4], [6, 3], [7, 2], [8, 1]]))  # 3
print(sl.minimumLines([[3, 4], [1, 2], [7, 8], [2, 3]]))  # 1
