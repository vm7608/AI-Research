# 2287. Rearrange Characters to Make Target String
# Difficulty: easy
# Tags: string
# Link: https://leetcode.com/problems/rearrange-characters-to-make-target-string/

# Description:
# You are given two 0-indexed strings s and target.
# You can take some letters from s and rearrange them to form new strings.
# Return the maximum number of copies of target
# that can be formed by taking letters from s and rearranging them.

# Idea: loop through unique characters in target
# count the number of each character in target and s
# save count_s // count_target to a list
# (the number of copies of target char that can taking from s)
# min of the list is the maximum number of copies of target that can be formed from s

class Solution:
    def rearrangeCharacters(self, s: str, target: str) -> int:
        l = []
        for i in range(len(set(target))):
            count_target = target.count(target[i])
            count_str = s.count(target[i])
            if count_target <= count_str:
                l.append(count_str // count_target)
            else:
                return 0
        return min(l)


sl = Solution()
print(sl.rearrangeCharacters("ilovecodingonleetcode", "code"))  # 2
print(sl.rearrangeCharacters("abbaccaddaeea", "aaaaa"))  # 1
