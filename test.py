class Solution(object):
    def largestSumAfterKNegations(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: int
        """
        sum = 0 
        count = 0
        nums.sort(key=abs, reverse=True)
        for i in range (len(nums)):
            if k != 0 and nums[i] <0:
                k -=1
                nums[i] = -nums[i]
            sum += nums[i]
        if k%2 != 0:
            sum-=2*nums[-1]
        
        return sum
