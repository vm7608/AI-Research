# 2325. Decode the Message
# Difficulty: Easy
# Tags: string
# Link: https://leetcode.com/problems/decode-the-message/

# Description:
# You are given the strings key and message, which represent a cipher key and a secret message, respectively.
# The steps to decode message are as follows:
# Use the first appearance of all 26 lowercase English letters in key as the order of the substitution table.
# Align the substitution table with the regular English alphabet.
# Each letter in message is then substituted using the table.
# Spaces ' ' are transformed to themselves.
# For example, given key = "happy boy" (actual key would have at least one instance of each letter in the alphabet),
# we have the partial substitution table of ('h' -> 'a', 'a' -> 'b', 'p' -> 'c', 'y' -> 'd', 'b' -> 'e', 'o' -> 'f').

# Idea:
# trimm all spaces in key, then create a dictionary to map each letter in key to a letter in alphabet
# then decode the message using the dictionary

class Solution:
    def decodeMessage(self, key: str, message: str) -> str:
        # trim all spaces in key
        key = key.replace(" ", "")

        d = {}
        j = 0
        for i in range(len(key)):
            if key[i] not in d:
                d[key[i]] = chr(ord('a') + j)
                j += 1

        # decode the message
        rs = ""
        for i in range(len(message)):
            rs += " " if message[i] == " " else d[message[i]]
        return rs


sl = Solution()
print(sl.decodeMessage("the quick brown fox jumps over the lazy dog",
      "vkbs bs t suepuv"))  # this is a secret
