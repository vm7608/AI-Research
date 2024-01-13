# 2432. The Employee That Worked on the Longest Task
# Difficulty: Easy
# Tags: implementation
# Link: https://leetcode.com/problems/the-employee-that-worked-on-the-longest-task/

# Description: There are n employees, each with a unique id from 0 to n - 1.
# You are given a 2D integer array logs where logs[i] = [idi, leaveTimei] where:
# idi is the id of the employee that worked on the ith task, and
# leaveTimei is the time at which the employee finished the ith task.
# All the values leaveTimei are unique.
# Note that the ith task starts the moment right after the (i - 1)th task ends,
# and the 0th task starts at time 0.
# Return the id of the employee that worked the task with the longest time.
# If there is a tie between two or more employees, return the smallest id among them.

# Idea: longest contains the longest time, result contains the smallest id
# loop through logs, compare the current - previous leaveTime with longest time
# if the task time > longest, update longest and result
# if the task time == longest, update result if the id is smaller

from typing import List


class Solution:
    def hardestWorker(self, n: int, logs: List[List[int]]) -> int:
        result = -1
        longest = 0
        prev = 0
        for i in range(len(logs)):
            if logs[i][1] - prev > longest:
                longest = logs[i][1] - prev
                result = logs[i][0]
            elif logs[i][1] - prev == longest and logs[i][0] < result:
                result = logs[i][0]
            prev = logs[i][1]

        return result


sl = Solution()
print(sl.hardestWorker(10, [[0, 3], [2, 5], [0, 9], [1, 15]]))  # 1
print(sl.hardestWorker(26, [[1, 1], [3, 7], [2, 12], [7, 17]]))  # 3
print(sl.hardestWorker(2, [[0, 10], [1, 20]]))  # 0
