# 2336. Smallest Number in Infinite Set
# Difficulty: Medium
# Tags: priority queue
# Link: https://leetcode.com/problems/smallest-number-in-infinite-set/

# Description:
# You have a set which contains all positive integers [1, 2, 3, 4, 5, ...].
# Implement the SmallestInfiniteSet class:
# SmallestInfiniteSet() Initializes the SmallestInfiniteSet object to contain all positive integers.
# int popSmallest() Removes and returns the smallest integer contained in the infinite set.
# void addBack(int num) Adds a positive integer num back into the infinite set,
# if it is not already in the infinite set.

# Idea:
# Init: minimum = 1, value_set = set()
# minimum is the smallest number in the set
# value_set contain values > minimum that have been addBack
# popSmallest: if s is not empty, return min(s) and remove it from s
# else, return minimum and increase minimum by 1
# addBack: if num < minimum, add num to value_set


class SmallestInfiniteSet:

    def __init__(self):
        self.minimum = 1
        self.value_set = set()

    def popSmallest(self) -> int:
        if self.value_set:
            # because value_set contain value that have been pop then addBack
            # so the minimum is the min of value_set
            res = min(self.value_set)
            self.value_set.remove(res)
            return res
        else:
            # if value_set is empty, that mean no value has been pop then addBack
            # so min is minimum, and after pop, minimum will increase by 1
            self.minimum += 1
            return self.minimum - 1

    def addBack(self, num: int) -> None:
        # if num >= minimum, that mean num is already in the set
        if self.minimum > num:
            self.value_set.add(num)


# Test code
smallestInfiniteSet = SmallestInfiniteSet()
print(smallestInfiniteSet.addBack(2))
print(smallestInfiniteSet.popSmallest())
print(smallestInfiniteSet.popSmallest())
print(smallestInfiniteSet.popSmallest())
print(smallestInfiniteSet.addBack(1))
print(smallestInfiniteSet.popSmallest())
print(smallestInfiniteSet.popSmallest())
print(smallestInfiniteSet.popSmallest())
