# 2284. Sender With Largest Word Count
# Difficulty: Medium
# Tags: implementation
# Link: https://leetcode.com/problems/sender-with-largest-word-count/

# Description:
# You have a chat log of n messages.
# You are given two string arrays messages and senders where messages[i] is a message sent by senders[i].
# A message is list of words that are separated by a single space with no leading or trailing spaces.
# The word count of a sender is the total number of words sent by the sender. Note that a sender may send more than one message.
# Return the sender with the largest word count.
# If there is more than one sender with the largest word count, return the one with the lexicographically largest name.
# Note:
# Uppercase letters come before lowercase letters in lexicographical order.
# "Alice" and "alice" are distinct.

# Idea: use a dictionary to store the word count of each sender
# then find the sender with the largest word count
# if there are more than one sender with the largest word count, return the one with the lexicographically largest name

from typing import List


class Solution:
    def largestWordCount(self, messages: List[str], senders: List[str]) -> str:
        d = {}
        for i in range(len(messages)):
            words = len(messages[i].split())
            d[senders[i]] = words + d.get(senders[i], 0)

        ans = ""
        largest = 0
        for name in d.keys():
            words = d[name]
            if words > largest:
                largest = words
                ans = name
            elif words == largest:
                if name > ans:
                    ans = name
        return ans


sl = Solution()
print(sl.largestWordCount(["Hello userTwooo", "Hi userThree", "Wonderful day Alice",
      "Nice day userThree"], ["Alice", "userTwo", "userThree", "Alice"]))  # Alice
print(sl.largestWordCount(["How is leetcode for everyone",
      "Leetcode is useful for practice"], ["Bob", "Charlie"]))  # Charlie
