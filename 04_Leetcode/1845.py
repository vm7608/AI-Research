# 1845. Seat Reservation Manager
# Difficulty: Medium
# Tags: priority queue
# Link: https:/leetcode.com/problems/seat-reservation-manager/

# Description:
# Design a system that manages the reservation state of n seats that are numbered from 1 to n.
# Implement the SeatManager class:
# SeatManager(int n) Initializes a SeatManager object that will manage n seats numbered from 1 to n.
# All seats are initially available.
# int reserve() Fetches the smallest-numbered unreserved seat, reserves it, and returns its number.
# void unreserve(int seatNumber) Unreserves the seat with the given seatNumber.

# Idea: use heapq to store available seats
# Init: heap = list(range(1, n+1))
# reserve: use heapq.heappop(heap) to get the smallest seat number
# unreserve: use heapq.heappush(heap, seatNumber) to add seatNumber to heap

import heapq


class SeatManager:

    def __init__(self, n: int):
        self.heap = list(range(1, n+1))

    def reserve(self) -> int:
        return heapq.heappop(self.heap)

    def unreserve(self, seatNumber: int) -> None:
        heapq.heappush(self.heap, seatNumber)


# Test code
seatManager = SeatManager(5)
print(seatManager.reserve())
print(seatManager.reserve())
print(seatManager.unreserve(2))
print(seatManager.reserve())
print(seatManager.reserve())
print(seatManager.reserve())
print(seatManager.reserve())
print(seatManager.unreserve(5))
