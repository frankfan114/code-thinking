# 代码随想录 Day 1

704. 二分查找

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

160. Intersection of Two Linked Lists
```python
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def getIntersectionNode(self, headA, headB):
        """
        :type head1, head1: ListNode
        :rtype: ListNode
        """
        lena =0
        lenb = 0
        cura = headA
        curb = headB
        while cura:
            cura = cura.next
            lena+=1
        while curb:
            curb = curb.next
            lenb+=1

        if lena >= lenb:
            for i in range (lena-lenb):
                headA=headA.next
            while headA:
                if headA == headB:
                    return headA
                headA = headA.next
                headB = headB.next
            return None
        else:
            for i in range (lenb-lena):
                headB=headB.next
            while headB:
                if headB == headA:
                    return headB
                headA = headA.next
                headB = headB.next
            return None
```
other method: if lenb > lena, assign a to b, b to a to reduce repeatence 
            use proprotional method, 2*3 = 3*2
142. Linked List Cycle II
```python
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def detectCycle(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        fast = head
        slow = head
        while fast and fast.next:
            fast = fast.next.next
            slow = slow.next
            if fast == slow:
                slow = head
                while slow != fast:
                    slow = slow.next
                    fast = fast.next
                return slow
            #fast = fast.next.next
            #slow = slow.next
            
        return None

        
```
the incremnet should be placed first, otherwise it will be always true as slow and fast start the same node
set method really good

# 代码随想录 Day 6
242.有效的字母异位词 
```python
class Solution(object):
    def isAnagram(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: bool
        """
        record = [0]*26
        for i in s:
            record[ord(i)-ord('a')] +=1
        for i in t:
            record[ord(i)-ord('a')] -=1
        if record ==[0]*26:
            return True
        else:
            return False

```
349. 两个数组的交集 
```python
class Solution(object):
    def intersection(self, nums1, nums2):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: List[int]
        """
        table = {}
        for num in nums1:
            table[num] = table.get(num, 0) + 1
        
        res = set()
        for num in nums2:
            if num in table:
                res.add(num)
                del table[num]
        
        return list(res)
        
```
202. 快乐数
```python
class Solution(object):
    def isHappy(self, n):
        """
        :type n: int
        :rtype: bool
        """
        sum =set()
        while True:
            #a = self.get_sum(n)
            if n ==1:
                return True
            if n in sum:
                return False
            else:
                sum.add(n)
            n = self.get_sum(n)
    
    def get_sum(self,n): 
        new_num = 0
        while n:
            n, r = divmod(n, 10)
            new_num += r ** 2
        return new_num
```
1 loop -> sum will repeat
2 inital n should be include to reduce repetance, as if it's get again, there's definitely a loop
1. 两数之和   
```python
class Solution(object):
    def twoSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        record = dict()
        for i in range(len(nums)):
            if target-nums[i] in record:
                return [i,record[target-nums[i]]]
            else:
                record[nums[i]] = i
```

# 代码随想录 Day 7
454.四数相加II 
```python
class Solution(object):
    def fourSumCount(self, nums1, nums2, nums3, nums4):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :type nums3: List[int]
        :type nums4: List[int]
        :rtype: int
        """
        record = dict()
        occ = 0
        # key: sum of a, b
        # value: times of this sum
        for i in range(len(nums1)):
            for l in range(len(nums2)):
                sum = nums1[i]+nums2[l]
                if sum in record:
                    record[sum]+=1
                else:
                    record[sum] = 1

        for i in range(len(nums3)):
            for l in range(len(nums4)):
                sum1 = nums3[i]+nums4[l]
                if -sum1 in record:
                    occ +=record[-sum1]
        return occ
```
the value for sum of a+b is the count of a+b in first two array
383. 赎金信 
```python
ransom_count = [0] * 26
        magazine_count = [0] * 26
        for c in ransomNote:
            ransom_count[ord(c) - ord('a')] += 1
        for c in magazine:
            magazine_count[ord(c) - ord('a')] += 1
        return all(ransom_count[i] <= magazine_count[i] for i in range(26))
```
array has lower space and time needed compared to map
15. 三数之和 
```python
class Solution(object):
    def threeSum(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        result = []
        nums.sort()
        

        for i in range(len(nums)):
             # 如果第一个元素已经大于0，不需要进一步检查
            if nums[i] > 0:
                return result
            if i>0 and nums[i]==nums[i-1]:
                continue
            left = i + 1
            right = len(nums) - 1
            while left< right: # check every possible case for i
                if nums[i]+nums[left]+nums[right]<0:
                    left+=1
                elif nums[i]+nums[left]+nums[right]>0:
                    right-=1
                else:
                    result.append([nums[i],nums[left],nums[right]])
                    # 跳过相同的元素以避免重复
                    while right > left and nums[right] == nums[right - 1]:
                        right -= 1
                    while right > left and nums[left] == nums[left + 1]:
                        left += 1
                    left +=1
                    right -=1
        return result
```
1 return if >0
2 jump repeat left and right

TBD dict version   

18. 四数之和 
```python
class Solution(object):
    def fourSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[List[int]]
        """
        nums.sort()
        n = len(nums)
        result = []
        for a in range(n):
            if nums[a] > target and nums[a] > 0 and target > 0:
                break
            if a>0 and nums[a-1] == nums[a]: # need a>0, prevent out of bound
                continue
            for b in range(a+1,n):
                if nums[a]+nums[b] > target and target >0: # no need for nums[b]>0
                    break
                if b>a+1 and nums[b-1] == nums[b]:
                    continue
                left = b+1
                right = n-1
                while left < right:
                    sum= nums[a]+nums[b]+nums[left]+nums[right]
                    if sum == target:
                        result.append([nums[a], nums[b], nums[left], nums[right]])
                        # reduce repetation part
                        while left < right and nums[left] == nums[left+1]:
                            left += 1
                        while left < right and nums[right] == nums[right-1]:
                            right -= 1
                        #
                        left +=1
                        right -=1
                    elif sum < target:
                        left+=1
                    else:
                        right -=1
        return result
```

# 代码随想录 Day 8
344.反转字符串
```python
class Solution(object):
    def reverseString(self, s):
        """
        :type s: List[str]
        :rtype: None Do not return anything, modify s in-place instead.
        """
        left = 0
        right = len(s)-1
        while right > left:
            tmp = s[left]
            s[left] = s[right]
            s[right] = tmp
            right -=1
            left +=1
        return s
        
```


541. 反转字符串II
```python
p = 0
        while p < len(s):
            p2 = p + k
            # Written in this could be more pythonic.
            s = s[:p] + s[p: p2][::-1] + s[p2:]
            p = p + 2 * k
        return s
```
所以当需要固定规律一段一段去处理字符串的时候，要想想在在for循环的表达式上做做文章。
对于字符串s = 'abc'，如果使用s[0:999] ===> 'abc'。字符串末尾如果超过最大长度，则会返回至字符串最后一个值，这个特性可以避免一些边界条件的处理。

卡码网：54.替换数字
TBD
# 代码随想录 Day 9

151.翻转字符串里的单词
```python
class Solution(object):
    def reverseWords(self, s):
        """
        :type s: str
        :rtype: str
        """
        ws = s.split()
        ws =ws[::-1]
        s = ' '.join(ws) 
        return s
```
1 delete space 2 reverse 3 recombine
卡码网：55.右旋转字符串
TBD
28. 实现 strStr()
```python
```
TBD
459.重复的子字符串
```python
```
TBD
# 代码随想录 Day 10
232.用栈实现队列 
```python
class MyQueue(object):

    def __init__(self):
        self.stack_in = []
        self.stack_out = []
        

    def push(self, x):
        """
        :type x: int
        :rtype: None
        """
        
        self.stack_in.append(x)
        

    def pop(self):
        """
        :rtype: int
        """
        if self.empty():
            return None
        
        if self.stack_out:
            return self.stack_out.pop()
        else:
            for i in range(len(self.stack_in)):
                self.stack_out.append(self.stack_in.pop())
            return self.stack_out.pop()

        
        

    def peek(self):
        """
        :rtype: int
        """
        ans = self.pop()
        self.stack_out.append(ans)
        return ans
        

    def empty(self):
        """
        :rtype: bool
        """
        return not (self.stack_in or self.stack_out)


# Your MyQueue object will be instantiated and called as such:
# obj = MyQueue()
# obj.push(x)
# param_2 = obj.pop()
# param_3 = obj.peek()
# param_4 = obj.empty()
```
use 2 stack, 1 for in and 1 for out
225. 用队列实现栈 
```python
class MyStack(object):

    def __init__(self):
        self.queue_in=deque()
        self.queue_out=deque()

    def push(self, x):
        """
        :type x: int
        :rtype: None
        """
        self.queue_in.append(x)
        

    def pop(self):
        """
        :rtype: int
        """
        if self.empty():
            return None

        for i in range(len(self.queue_in) - 1):
            self.queue_out.append(self.queue_in.popleft())
        
        self.queue_in, self.queue_out = self.queue_out, self.queue_in    # 交换in和out，这也是为啥in只用来存
        return self.queue_out.popleft()
        

    def top(self):
        """
        :rtype: int
        """
        if self.empty():
            return None

        for i in range(len(self.queue_in) - 1):
            self.queue_out.append(self.queue_in.popleft())
        
        self.queue_in, self.queue_out = self.queue_out, self.queue_in 
        temp = self.queue_out.popleft()   
        self.queue_in.append(temp)
        return temp
        

    def empty(self):
        """
        :rtype: bool
        """
        return len(self.queue_in) == 0
        


# Your MyStack object will be instantiated and called as such:
# obj = MyStack()
# obj.push(x)
# param_2 = obj.pop()
# param_3 = obj.top()
# param_4 = obj.empty()
```
FIF0 have feature that, the order won't reverse if you put them to a new queue
20. 有效的括号 
```python
class Solution(object):
    def isValid(self, s):
        """
        :type s: str
        :rtype: bool
        """
        cor = []
        for i in s:
            if i == '(': 
                cor.append(')')
            elif i == '[':
                cor.append(']')
            elif i == '{':
                cor.append('}')
            elif not cor or i != cor[-1]:
                return False
            else:
                cor.pop()
        if cor:
            return False
        else:
            return True
```
3 possible conditions: 1. left more, 2. right more, 3. type not meet
1047. 删除字符串中的所有相邻重复项 
```python
class Solution(object):
    def removeDuplicates(self, s):
        """
        :type s: str
        :rtype: str
        """
        stack = []
        for i in s:
            if stack and i == stack[-1]:
                stack.pop()
            else:
                stack.append(i)
        return ''.join(stack)
```

# 代码随想录 Day 11 (07/09/24)

150. 逆波兰表达式求值 
```python
def div(x, y):
    # 使用整数除法的向零取整方式
    return int(x / y) if x * y > 0 else -(abs(x) // abs(y))
    
class Solution(object):
    op_map = {'+': add, '-': sub, '*': mul, '/': div}

    def evalRPN(self, tokens):
        """
        :type tokens: List[str]
        :rtype: int
        """
        stack = []
        for token in tokens:
            if token not in {'+', '-', '*', '/'}:
                stack.append(int(token))
            else:
                op2 = stack.pop()
                op1 = stack.pop()
                stack.append(self.op_map[token](op1, op2))  # 第一个出来的在运算符后面
        return stack.pop()
```
divisioin need special treat for integer
239. 滑动窗口最大值
```python
```
TBD

347. 前 K 个高频元素
```python
```
TBD

# 代码随想录 Day 13
144.二叉树的前序遍历
```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def preorderTraversal(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        if not root:
            return []
        stack = [root]
        result = []
        while stack:
            node = stack.pop()
            # 中结点先处理
            result.append(node.val)
            # 右孩子先入栈
            if node.right:
                stack.append(node.right)
            # 左孩子后入栈
            if node.left:
                stack.append(node.left)
        return result
        

```
iterate method
145.二叉树的后序遍历
```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def postorderTraversal(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        result = []
        st = []
        if root:
            st.append(root)
        while st:
            node = st.pop()
            if node != None:
                st.append(node) #中
                st.append(None)
                
                if node.right: #右
                    st.append(node.right)
                if node.left: #左
                    st.append(node.left)
            else:
                node = st.pop()
                result.append(node.val)
        return result
```
adding NULL method
Unified method
94.二叉树的中序遍历
```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def inorderTraversal(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        output = []
        def transverse(cur, output):
            if cur == None:
                return
            transverse(cur.left,output)
            output.append(cur.val)
            transverse(cur.right, output)
            return output
        return transverse(root, output) 
```
recersive method

102. Binary Tree Level Order Traversal
```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def levelOrder(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        if not root:
            return []
        result = []
        queue = collections.deque([root])
        while queue:
            level = []
            for _ in range(len(queue)):
                cur = queue.popleft()
                level.append(cur.val)
                if cur.left:
                    queue.append(cur.left)
                if cur.right:
                    queue.append(cur.right)
            result.append(level)
        return result
```
iterative
use queue FIFO
```python
class Solution:
    def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        if not root:
            return []

        levels = []

        def traverse(node, level):
            if not node:
                return

            if len(levels) == level:
                levels.append([])

            levels[level].append(node.val)
            traverse(node.left, level + 1)
            traverse(node.right, level + 1)

        traverse(root, 0)
        return levels
```
recursive
use pointer to add node

# 代码随想录 Day 14
# 代码随想录 Day 15
# 代码随想录 Day 16
# 代码随想录 Day 17
# 代码随想录 Day 18

# 代码随想录 Day 20
# 代码随想录 Day 21
# 代码随想录 Day 22

# 代码随想录 Day 23

# 代码随想录 Day 24

# 代码随想录 Day 25

# 代码随想录 Day 27
455. Assign Cookies
```python
class Solution(object):
    def findContentChildren(self, g, s):
        """
        :type g: List[int]
        :type s: List[int]
        :rtype: int
        """
        g.sort() # not sorted in question list 
        s.sort()
        count = 0 
        smax = len(s)-1
        gmax = len(g)-1
        # while gmax >=0:
        while gmax >=0 and smax >= 0:
            if s[smax] >= g[gmax]:
                count +=1
                smax -=1
            gmax -=1
        return count
```
description sort the lists, but the question did not, so sort list first
the condition to end also because of cookie list run out.

376. Wiggle Subsequence
```python
class Solution(object):
    def wiggleMaxLength(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        
        if len(nums) < 2:
            return len(nums)

        pre_dif = 0
        count = 1 #
        for i in range(len(nums) - 1):
            cur_dif = nums[i+1] - nums[i]

            if (cur_dif > 0 and pre_dif <= 0) or (cur_dif < 0 and pre_dif >= 0):
                count += 1
                pre_dif = cur_dif 

        return count
```
only when slope change, update the pre_dif
to deal with dif == 0, that means only 1 wiggle 
53. Maximum Subarray
```python
class Solution(object):
    def maxSubArray(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        sum = 0
        max = nums[0]
        for i in range(len(nums)):
             
            if sum < 0:
                sum = nums[i]
            else:
                sum += nums[i]
                
            if max < sum:
                max = sum 

        return max
```
greedy point: what bring to the next increment need to be positive
# 代码随想录 Day 28
122. Best Time to Buy and Sell Stock II
```python
class Solution(object):
    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        sum = 0 
        if len(prices) ==0:
            return sum 
        for i in range(len(prices)-1):
            if prices[i+1]-prices[i] >0:
                sum += prices[i+1]-prices[i]
        return sum
```
55. Jump Game
```python
class Solution(object):
    def canJump(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        # coverage for the jump
        cover = 0
        if len(nums) == 1: return True
        i = 0
        while i <= cover:
            cover = max(i + nums[i], cover)
            if cover >= len(nums) - 1: return True
            i += 1
        return False
```
key part: cover = max(i+nums[i], cover)
update cover each time to enlarge the range 
45. Jump Game II
```python
class Solution(object):
    def jump(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        cur_distance = 0  # 当前覆盖的最远距离下标
        ans = 0  # 记录走的最大步数
        next_distance = 0  # 下一步覆盖的最远距离下标
        
        for i in range(len(nums) - 1):  # 注意这里是小于len(nums) - 1，这是关键所在
            next_distance = max(nums[i] + i, next_distance)  # 更新下一步覆盖的最远距离下标
            if i == cur_distance:  # 遇到当前覆盖的最远距离下标
                cur_distance = next_distance  # 更新当前覆盖的最远距离下标
                ans += 1
        
        return ans
```
the only condition for step +1 is coverga smaller than current move, before move reach end
1005. Maximize Sum Of Array After K Negations
```python
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
```
1 reverse the negative number as possible
2 flip on the smallest one for the rest of left k times

# 代码随想录 Day 29
134. Gas Station
```python
class Solution(object):
    def canCompleteCircuit(self, gas, cost):
        """
        :type gas: List[int]
        :type cost: List[int]
        :rtype: int
        """
        curSum = 0  # 当前累计的剩余油量
        totalSum = 0  # 总剩余油量
        start = 0  # 起始位置
        
        for i in range(len(gas)):
            curSum += gas[i] - cost[i]
            totalSum += gas[i] - cost[i]
            
            if curSum < 0:  # 当前累计剩余油量curSum小于0
                start = i + 1  # 起始位置更新为i+1
                curSum = 0  # curSum重新从0开始累计
        
        if totalSum < 0:
            return -1  # 总剩余油量totalSum小于0，说明无法环绕一圈
        return start
                
```
局部最优：当前累加rest[i]的和curSum一旦小于0，起始位置至少要是i+1，因为从i之前开始一定不行。全局最优：找到可以跑一圈的起始位置。

135. Candy
```python
class Solution(object):
    def candy(self, ratings):
        """
        :type ratings: List[int]
        :rtype: int
        """
        candy = [1]*len(ratings)
        sum = 0
        for i in range(1, len(ratings)):
            if ratings[i] > ratings[i-1]:
                candy[i] = candy[i-1]+1
            # else:
            #     candy[i] = candy[i+1]
            
        for i in range(len(ratings)-2, -1, -1):
            if ratings[i] > ratings[i+1]:
                #candy[i] = candy[i+1]+1 
                candy[i] = max(candy[i], candy[i+1]+1)
            # else:
            #     candy[i] = candy[i+1]
            
            sum+=candy[i]
        sum += candy[len(ratings)-1]
        return sum 
```
1 use max(a,b), instead of add 1, as the current value may alreay larger than neighbour
2 since initialize with 1 in each index, no need to candy[i]=candy[i+1]

860. Lemonade Change
```python
class Solution(object):
    def lemonadeChange(self, bills):
        """
        :type bills: List[int]
        :rtype: bool
        """
        five = 0 
        ten = 0
        for i in range(len(bills)):
            if bills[i] == 5:
                five+=1
            elif bills[i] == 10:
                if five >0:
                    five -=1
                    ten += 1
                else:
                    return False
            else:
                if ten>0 and five >0:
                    ten -= 1
                    five -= 1
                elif five >2:
                    five -=3
                else:
                    return False
        return True
```
406. Queue Reconstruction by Height
```python
class Solution(object):
    def reconstructQueue(self, people):
        """
        :type people: List[List[int]]
        :rtype: List[List[int]]
        """
        # 先按照h维度的身高顺序从高到低排序。确定第一个维度
        # lambda返回的是一个元组：当-x[0](维度h）相同时，再根据x[1]（维度k）从小到大排序
        people.sort(key=lambda x: (-x[0], x[1]))
        que = []
	
	# 根据每个元素的第二个维度k，贪心算法，进行插入
        # people已经排序过了：同一高度时k值小的排前面。
        for p in people:
            que.insert(p[1], p)
        return que
        
```
身高从大到小排序后：

局部最优：优先按身高高的people的k来插入。插入操作过后的people满足队列属性

全局最优：最后都做完插入操作，整个队列满足题目队列属性
