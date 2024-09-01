# 代码随想录 Day 1

## 704. 二分查找

```python
class Solution(object):
    def search(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        left, right = 0, len(nums) - 1
        while left <= right:
            mid = left + (right - left) // 2
            if target == nums[mid]:
                return mid
            elif target < nums[mid]:
                right = mid - 1
            else:
                left = mid + 1
        return -1
```
需要注意 left == right 的情况，左闭右闭区间仍然有效。


```python
class Solution(object):
    def removeElement(self, nums, val):
        """
        :type nums: List[int]
        :type val: int
        :rtype: int
        """
        fast = 0
        slow = 0
        for fast in range(len(nums)):
            if val != nums[fast]:
                nums[slow] = nums[fast]
                slow += 1    
            fast += 1
        return slow
```
977. 有序数组的平方
```python
class Solution(object):
    def sortedSquares(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        new_list = []
        left, right = 0 , len(nums) -1
        while left <= right:
            if abs(nums[left]) <= abs(nums[right]):
                new_list.append(nums[right] ** 2)
                right -= 1
            else:
                new_list.append(nums[left] ** 2)
                left += 1
        return new_list[::-1]
        
        
```

# 代码随想录 Day 2

209.长度最小的子数组
```python
class Solution(object):
    def minSubArrayLen(self, target, nums):
        """
        :type target: int
        :type nums: List[int]
        :rtype: int
        """
        slow =0
        sum = 0
        min_len =float('inf')
        for i in range(0, len(nums)):
            sum+=nums[i]
            while sum>= target:
                min_len = min(min_len, i - slow + 1)
                sum-= nums[slow]
                slow+=1
            i+=1
            
        if min_len == float('inf'):
            return 0
        else:
            return min_len
```
need a min_len to keep track the shortest length

 59.螺旋矩阵II
```python
class Solution(object):
    def generateMatrix(self, n):
        """
        :type n: int
        :rtype: List[List[int]]
        """
        nums = [[n**2] * n for _ in range(n)]  
        startx, starty = 0, 0               
        loop = n // 2                       
        count = 1                           

        for i in range(loop):
            
            for x in range(starty, n - i - 1):
                nums[startx][x] = count
                count += 1

            
            for y in range(startx, n - i - 1):
                nums[y][n - i - 1] = count
                count += 1

            
            for x in range(n - i - 1, starty, -1):
                nums[n - i - 1][x] = count
                count += 1

            
            for y in range(n - i - 1, startx, -1):
                nums[y][starty] = count
                count += 1

            
            startx += 1
            starty += 1

        

        return nums

```
the offset is 1, as the starty and startx also move forward in each layer

区间和

TBD

# 代码随想录 Day 3

203.移除链表元素 
```python
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution(object):
    def removeElements(self, head, val):
        """
        :type head: ListNode
        :type val: int
        :rtype: ListNode
        """
        dummy_head = ListNode(next = head)
        current = dummy_head
        while current.next:
            if current.next.val == val:
                temp = current.next.next
                current.next = temp
            else:
                current = current.next
        return dummy_head.next
        
```
current= current.next in else condition for 2 reasons: 1. avoid skipping continuous val
                                                       2. avoid empty node, Nodetype Error
   
707.设计链表  
```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class MyLinkedList(object):

    def __init__(self):
        self.dum_head = ListNode()
        # self.dum_head.next = None
        self.size = 0 
        

    def get(self, index):
        """
        :type index: int
        :rtype: int
        """
        if index >=0 and index < self.size:
            current = self.dum_head #
            for i in range(index+1):
                #self.dum_head = seld.dum_head.next
                current = current.next
            return current.val
        else:
            return -1
        

    def addAtHead(self, val):
        """
        :type val: int
        :rtype: None
        """
        tmp = self.dum_head.next
        self.dum_head.next = ListNode(val, tmp)
        self.size +=1

        

    def addAtTail(self, val):
        """
        :type val: int
        :rtype: None
        """
        current = self.dum_head
        while current.next:
            current = current.next
        current.next = ListNode(val)
        self.size +=1
    
        

    def addAtIndex(self, index, val):
        """
        :type index: int
        :type val: int
        :rtype: None
        """
        if index>=0:
            if index <= self.size: 
                cur =self.dum_head
                while index >0:
                    cur = cur.next
                    index -=1
                tmp = cur.next
                cur.next =ListNode(val, tmp)
                self.size +=1


    def deleteAtIndex(self, index):
        """
        :type index: int
        :rtype: None
        """
        if index >=0 and index < self.size:
            cur = self.dum_head
            while index >0:
                cur = cur.next
                index -=1

            tmp = cur.next.next
            cur.next = tmp
            #self.dum_jead.next.val = tmp.val
            self.size -=1


# Your MyLinkedList object will be instantiated and called as such:
# obj = MyLinkedList()
# param_1 = obj.get(index)
# obj.addAtHead(val)
# obj.addAtTail(val)
# obj.addAtIndex(index,val)
# obj.deleteAtIndex(index)
```
the condtion error for add at index, should only include = and < size
use cur to transverse, so that the structure of self.head won't change 

203 翻转链表
```python
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution(object):
    def reverseList(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        pre = None
        cur = head
        while cur:
            tmp=cur.next
            cur.next = pre
            pre= cur
            cur = tmp
        return pre
```
condition is cur instead of cur.next, as it don't need to check the next existence, just first one 


# 代码随想录 Day 4
24. Swap Nodes in Pairs
```python 
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution(object):
    def swapPairs(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        dum = ListNode(next = head)
        pre = dum# cur = dum
        while pre.next and pre.next.next:
            tmp3 = pre.next.next.next
            tmp2 = pre.next.next 
            tmp1 = pre.next
            #pre.next = tmp2
            tmp2.next = tmp1#1 
            tmp1.next = tmp3#2
            pre.next = tmp2#3
            pre = pre.next.next
        return dum.next    
```
as long as 1 and 2 in sequence, 312 and 123 both work

19. Remove Nth Node From End of List
```python
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution(object):
    def removeNthFromEnd(self, head, n):
        """
        :type head: ListNode
        :type n: int
        :rtype: ListNode
        """
        dum = ListNode(0,head)
        fast= dum
        slow = dum
        #slow= dum.next leads to Nonetpye
        for i in range(n+1):
            fast = fast.next
        while fast:
            slow= slow.next
            fast= fast.next
        slow.next = slow.next.next
        return dum.next
        
```
if slow= dum.next, [1], n=1 leads to None = None.next, error