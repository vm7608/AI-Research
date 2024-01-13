# 2299. Strong Password Checker II
# Difficulty: easy
# Tags: implementation
# Link: https://leetcode.com/problems/strong-password-checker-ii/

# Desscription:
# A password is said to be strong if it satisfies all the following criteria:
# It has at least 8 characters.
# It contains at least one lowercase letter.
# It contains at least one uppercase letter.
# It contains at least one digit.
# It contains at least one special character.
# The special characters are the characters in the following string: "!@#$%^&*()-+".
# It does not contain 2 of the same character in adjacent positions
# (i.e., "aab" violates this condition, but "aba" does not).
# Given a string password, return true if it is a strong password.
# Otherwise, return false.

# Idea: use a loop and some condition variables to check each criteria above

class Solution:
    def strongPasswordCheckerII(self, password: str) -> bool:
        lower = upper = digit = symbol = repeat = False
        prev = None

        for c in password:
            if c.islower():
                lower = True
            if c.isupper():
                upper = True
            if c.isdigit():
                digit = True
            if c in "!@#$%^&*()-+":
                symbol = True
            if c == prev:
                repeat = True
            prev = c
        return len(password) >= 8 and lower and upper and digit and symbol and not repeat


sl = Solution()
print(sl.strongPasswordCheckerII("IloveLe3tcode!"))  # True
print(sl.strongPasswordCheckerII("Me+You--IsMyDream"))  # False
print(sl.strongPasswordCheckerII("1aB!"))  # False
