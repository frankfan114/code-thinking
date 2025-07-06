## 代码随想录 Day 1

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

## 代码随想录 Day 2

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

```cpp
class Solution {
public:
    vector<vector<int>> generateMatrix(int n) {
        vector<vector<int>> result(n,vector<int>(n,n*n));
        int start_x=0,  start_y=0;
        int loop= n / 2;
        int i,j;
        int offset = 1;
        int count =1;
        while(loop--){
            i= start_x;
            j=start_y;
            for (j; j<n-offset;j++){
                result[i][j]=count++;
            }

            for (i;i<n-offset;i++){
                result[i][j]=count++;
            }

            for (j;j>start_y;j--){ // current start
                result[i][j]=count++;
            }

            for (i;i>start_x;i--){ // current start
                result[i][j]=count++;
            }

            offset++;
            start_x++;
            start_y++;
        }
        return result;
    }
};
```
每次loop应该会到这层开始的地方，而不是（0，0）

区间和

TBD

## 代码随想录 Day 3

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
```cpp
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode() : val(0), next(nullptr) {}
 *     ListNode(int x) : val(x), next(nullptr) {}
 *     ListNode(int x, ListNode *next) : val(x), next(next) {}
 * };
 */
class Solution {
public:
    ListNode* removeElements(ListNode* head, int val) {
        ListNode* dummy = new ListNode(0); // initiate with 0 
        dummy->next = head;
        ListNode* current = dummy;
        while(current->next != NULL ){
            if(current->next->val == val){

                ListNode* temp = current->next->next;
                delete current->next;
                current->next=temp;
            }
            else{
                current= current->next;
            }
        } 
        ListNode* result = dummy->next;
        delete dummy; // for mem leakage
        return result;
    }
};
```
   
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

```cpp
class MyLinkedList {

private:
    struct LinkedNode{
        int val;
        LinkedNode* next;
        LinkedNode(int val):val(val),next(nullptr){}
    };
    
    LinkedNode* _dummy;
    int _size;
    

public:
    
    MyLinkedList() {
        _dummy = new LinkedNode(0);
        _size = 0; 
    }
    
    int get(int index) {
        if (index <0 || index >= _size){
            return -1;
        }
        else{
            LinkedNode* current = _dummy->next;
            for (int i = 0; i < index; ++i) {
                current = current->next;
            }
            return current->val;
        }
    }
    
    void addAtHead(int val) {
        LinkedNode* temp = _dummy->next;
        LinkedNode* first = new LinkedNode(val);
        first->next=temp;
        _dummy->next=first; 
        _size+=1;
    }
    
    void addAtTail(int val) {
        LinkedNode* current = _dummy;
        while(current->next){
            current=current->next;
        }
        LinkedNode* last = new LinkedNode(val);
        current->next = last;
        _size+=1;
    }
    
    void addAtIndex(int index, int val) {
        if(index <= _size ){
            if(index == 0){
                addAtHead(val);
            }
            else if(index == _size){
                addAtTail(val);
            }
            else{
                LinkedNode* current =_dummy;
                while(index-1>=0){
                    current= current->next;
                    index-=1;
                }
                LinkedNode* temp = current->next;
                LinkedNode* added = new LinkedNode(val);
                added->next= temp;
                current->next= added;
                _size +=1;

            }
        }
    }
    
    void deleteAtIndex(int index) {
    if (index < _size) {
        LinkedNode* current = _dummy;
        for (int i = 0; i < index; ++i) {
            current = current->next;
        }
        LinkedNode* deleted = current->next;
        current->next = deleted->next;
        delete deleted;
    	//delete命令指示释放了tmp指针原本所指的那部分内存，
        //被delete后的指针tmp的值（地址）并非就是NULL，而是随机值。也就是被delete后，
        //如果不再加上一句tmp=nullptr,tmp会成为乱指的野指针
        //如果之后的程序不小心使用了tmp，会指向难以预想的内存空间
        deleted =nullptr;
        _size--;
        
    }
}

};

/**
 * Your MyLinkedList object will be instantiated and called as such:
 * MyLinkedList* obj = new MyLinkedList();
 * int param_1 = obj->get(index);
 * obj->addAtHead(val);
 * obj->addAtTail(val);
 * obj->addAtIndex(index,val);
 * obj->deleteAtIndex(index);
 */
```


206. 翻转链表
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

```cpp
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode() : val(0), next(nullptr) {}
 *     ListNode(int x) : val(x), next(nullptr) {}
 *     ListNode(int x, ListNode *next) : val(x), next(next) {}
 * };
 */
class Solution {
public:
    // 2 pointer

    // ListNode* reverseList(ListNode* head) {
    //         
    //     ListNode* prev = nullptr;
    //     //
    //     ListNode* current = head;
    //     while(current){
    //         ListNode* tmp = current->next;
    //         current->next=prev;
    //         //
    //         prev = current;
    //         // move up
    //         current =tmp;

    //     }
    //     return prev;

    // }

    //resursive

    // ListNode* reverse(ListNode* prev, ListNode* cur){
    //     if(cur==nullptr) return prev;
    //     ListNode* tmp = cur->next;
    //     cur->next = prev;
    //     return reverse(cur,tmp);
    // }
    // ListNode* reverseList(ListNode* head) {
    //     return reverse(nullptr, head);
    // }

    //from back to front
    ListNode* reverseList(ListNode* head) {
        if(head==nullptr) return nullptr;
        if(head->next == nullptr) return head;

        ListNode* last = reverseList(head->next);
        //3->4<-5
        head->next->next=head;
        head->next=nullptr;
        return last;
    }
};
```

## 代码随想录 Day 4
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

```cpp
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode() : val(0), next(nullptr) {}
 *     ListNode(int x) : val(x), next(nullptr) {}
 *     ListNode(int x, ListNode *next) : val(x), next(next) {}
 * };
 */
class Solution {
public:
    // iterative
    
    ListNode* swap(ListNode* current){
        if (current->next == nullptr || current->next->next == nullptr) {
            return current;
        }
        ListNode* second = current->next->next;
        ListNode* first = current->next;
        
        first->next=second->next;
        second->next=first;

        current->next = second; // missing the link of current to the swapped

        return swap(first);
        //return current;
    }

    ListNode* swapPairs(ListNode* head) {
         
        ListNode* dummy = new ListNode(0);
        dummy->next = head;
        swap(dummy);
        
        // safety
        ListNode* result = dummy->next;
        delete dummy;
        return result;
        
    }
};
```

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

```cpp
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode() : val(0), next(nullptr) {}
 *     ListNode(int x) : val(x), next(nullptr) {}
 *     ListNode(int x, ListNode *next) : val(x), next(next) {}
 * };
 */
class Solution {
    
public:

    ListNode* removeNthFromEnd(ListNode* head, int n) {
        ListNode* dummy = new ListNode(0);
        dummy->next = head;
        ListNode* slow = dummy;
        ListNode* fast = dummy;
        while(n--){
            fast = fast->next;
        }     
        while(fast->next){
            slow=slow->next;
            fast=fast->next;
        }
        ListNode* tmp = slow->next;
        slow->next=tmp->next;
        delete tmp;
        ListNode* result =dummy->next;
        delete dummy;
        return result;

        // maintain 2 poitner, slow one delay by n

    }
};
```

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

```cpp
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode(int x) : val(x), next(NULL) {}
 * };
 */
class Solution {

public:
    ListNode *getIntersectionNode(ListNode *headA, ListNode *headB) {
        if (!headA || !headB) return nullptr;
        
        ListNode* pA = headA;
        ListNode* pB = headB;
        
        while (pA != pB) {
            pA = (pA == nullptr) ? headB : pA->next;
            pB = (pB == nullptr) ? headA : pB->next;
        }
        
        return pA; // could be nullptr if no intersection
    }

    // if reach, then m+n=n+m
};

    

```
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

```cpp
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode(int x) : val(x), next(NULL) {}
 * };
 */
class Solution {
public:
    ListNode *detectCycle(ListNode *head) {
    
    if (!head || !head->next) return nullptr;
        
        ListNode* slow = head;
        ListNode* fast = head;
        
        // Step 1: Detect if cycle exists
        while (fast && fast->next) {
            slow = slow->next;
            fast = fast->next->next;
            
            if (slow == fast) {
                // Step 2: Cycle detected, now find the start
                ListNode* entry = head;
                while (entry != slow) {
                    entry = entry->next;
                    slow = slow->next;
                }
                return entry; // the start of the cycle
            }
        }
        
        // No cycle
        return nullptr;


    // length before enter cycle = n*C - length after enter cycle      
    }
};
```

## 代码随想录 Day 6
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
```cpp
class Solution {
public:
    bool isAnagram(string s, string t) {
        int record[26] = {0};
        for (int i = 0; i < s.size(); i++) {
            // 并不需要记住字符a的ASCII，只要求出一个相对数值就可以了
            record[s[i] - 'a']++;
        }
        for (int i = 0; i < t.size(); i++) {
            record[t[i] - 'a']--;
        }
        for (int i = 0; i < 26; i++) {
            if (record[i] != 0) {
                // record数组如果有的元素不为零0，说明字符串s和t 一定是谁多了字符或者谁少了字符。
                return false;
            }
        }
        // record数组所有元素都为零0，说明字符串s和t是字母异位词
        return true;
    }
};
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

```cpp
class Solution {
public:
    vector<int> intersection(vector<int>& nums1, vector<int>& nums2) {
        unordered_set<int> result_set; // 存放结果，之所以用set是为了给结果集去重
        unordered_set<int> nums_set(nums1.begin(), nums1.end());
        for (int num : nums2) {
            // 发现nums2的元素 在nums_set里又出现过
            if (nums_set.find(num) != nums_set.end()) { //If it does not find it, find() returns nums_set.end(), which is a special iterator meaning "past the end", "not found".
                result_set.insert(num);
            }
        }
        return vector<int>(result_set.begin(), result_set.end());
    }
};
```
Returning vector<int> is how C++ "returns an array" dynamically and safely.


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

```cpp
class Solution {
public:
    int sumOfSquare(int n){
        int sum = 0;
        
        while(n){
            sum += (n%10)*(n%10);
            n/=10;
        }
        // sum+= n*n;
        return sum;
    }
    bool isHappy(int n) {
        unordered_set<int> set;
        while(1){
            int sum = sumOfSquare(n);
            if (sum == 1){
                return true;
            }
            
            if (set.find(sum) != set.end()) {
                return false;
            }
            set.insert(sum);
            n= sum; // let it looping

        }      
    }
};

// not use n%10 as the condition in the sum of square, as it will treat wrong for 10, add order: from top digit to bottom digit
```

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

```cpp
class Solution {
public:
    vector<int> twoSum(vector<int>& nums, int target) {
        unordered_map<int, int > map;
        for (int i=0; i< nums.size(); i++){
            int dif = target - nums[i];
            auto it = map.find(dif); 
            if( it != map.end()){
                return {(it->second), i};
            }
            map.insert(pair<int, int>(nums[i],i));
        }
        return {};
    }
};

// auto: auto give the type that has initializer in front
```
## 代码随想录 Day 7
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


```cpp
class Solution {
public:
    int fourSumCount(vector<int>& nums1, vector<int>& nums2, vector<int>& nums3, vector<int>& nums4) {
        unordered_map<int, int> map;
        int sum=0;
        int times = 0;
        // int a: nums1
        for (int i =0; i<nums1.size(); i++){
            for (int j =0; j<nums2.size(); j++){
                sum = nums1[i]+nums2[j];
                
                map[sum]++;
                
            }
        }

        for (int i =0; i<nums3.size(); i++){
            for (int j =0; j<nums4.size(); j++){
                sum = nums3[i]+nums4[j];
                if(map.find(-sum) != map.end()){
                    times+= map.find(-sum)->second;   
                }
                
            }
        }
        return times;

    }
};

// a+b = b+c, a+c = b+d
// key: sum, value: the times of the sum
// !=map.end(): find the object
```

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

```cpp
class Solution {
public:
    bool canConstruct(string ransomNote, string magazine) {
        int record[26]= {0};
        if (ransomNote.size()> magazine.size()){
            return false;
        }
        for(char l: magazine){
            record[l-'a']++;
        }
        for(char m : ransomNote){
            record[m-'a']--;
            if (record[m-'a']<0){
                return false;
            }
        }
        return true;
    }
};

// if output need has larger size than input, return false 
```

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

```cpp
class Solution {
public:
    vector<vector<int>> threeSum(vector<int>& nums) {
        vector<vector<int>> result;
        sort(nums.begin(), nums.end());
        // 找出a + b + c = 0
        // a = nums[i], b = nums[left], c = nums[right]
        for (int i = 0; i < nums.size(); i++) {
            // 排序之后如果第一个元素已经大于零，那么无论如何组合都不可能凑成三元组，直接返回结果就可以了
            if (nums[i] > 0) {
                return result;
            }
            // 错误去重a方法，将会漏掉-1,-1,2 这种情况
            /*
            if (nums[i] == nums[i + 1]) {
                continue;
            }
            */
            // 正确去重a方法
            if (i > 0 && nums[i] == nums[i - 1]) {
                continue;
            }
            int left = i + 1;
            int right = nums.size() - 1;
            while (right > left) {
                // 去重复逻辑如果放在这里，0，0，0 的情况，可能直接导致 right<=left 了，从而漏掉了 0,0,0 这种三元组
                /*
                while (right > left && nums[right] == nums[right - 1]) right--;
                while (right > left && nums[left] == nums[left + 1]) left++;
                */
                if (nums[i] + nums[left] + nums[right] > 0) right--;
                else if (nums[i] + nums[left] + nums[right] < 0) left++;
                else {
                    result.push_back(vector<int>{nums[i], nums[left], nums[right]});
                    // 去重逻辑应该放在找到一个三元组之后，对b 和 c去重
                    while (right > left && nums[right] == nums[right - 1]) right--;
                    while (right > left && nums[left] == nums[left + 1]) left++;

                    // 找到答案时，双指针同时收缩
                    right--;
                    left++;
                }
            }

        }
        return result;
    }
};

// vector<int>: no hash specialization
// if n <3, return 
// fix 2, move 1 will miss item

```
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

```cpp
class Solution {
public:
    vector<vector<int>> fourSum(vector<int>& nums, int target) {
        vector<vector<int>> result;
        sort(nums.begin(), nums.end());
        int n = nums.size();
        
        for (int i = 0; i < n - 3; i++) {
            // skip duplicate i’s
            if (i > 0 && nums[i] == nums[i-1]) continue;
            // if smallest possible 4-sum > target, we're done
            long long min_i = (long long)nums[i]
                            + nums[i+1]
                            + nums[i+2]
                            + nums[i+3];
            if (min_i > target) break;
            // if largest possible 4-sum < target, skip ahead
            long long max_i = (long long)nums[i]
                            + nums[n-1]
                            + nums[n-2]
                            + nums[n-3];
            if (max_i < target) continue;
            
            for (int j = i + 1; j < n - 2; j++) {
                // skip duplicate j’s
                if (j > i + 1 && nums[j] == nums[j-1]) continue;
                // prune by 3-sum bounds
                long long min_j = (long long)nums[i]
                                + nums[j]
                                + nums[j+1]
                                + nums[j+2];
                if (min_j > target) break;
                long long max_j = (long long)nums[i]
                                + nums[j]
                                + nums[n-1]
                                + nums[n-2];
                if (max_j < target) continue;
                
                int left = j + 1, right = n - 1;
                while (left < right) {
                    long long sum = (long long)nums[i]
                                  + nums[j]
                                  + nums[left]
                                  + nums[right];
                    if (sum < target) {
                        left++;
                    }
                    else if (sum > target) {
                        right--;
                    }
                    else {
                        result.push_back({nums[i],
                                          nums[j],
                                          nums[left],
                                          nums[right]});
                        // skip duplicates on left/right
                        while (left < right && nums[left] == nums[left+1])  left++;
                        while (left < right && nums[right] == nums[right-1]) right--;
                        left++;
                        right--;
                    }
                }
            }
        }
        
        return result;
    }
};

```

## 代码随想录 Day 8
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

```cpp

class Solution {
public:
    void reverseString(vector<char>& s) {
        int front = 0;
        int back = s.size()-1;
        while (front < back){
            char tmp = s[front];
            s[front++]= s[back];
            s[back--]=tmp;

        }
    }
};

// return type: void
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

```cpp
class Solution {
public:
    void reverse(string& s, int start, int end) {
        for (int i = start, j = end; i < j; i++, j--) {
            swap(s[i], s[j]);
        }
    }
    string reverseStr(string s, int k) {
        for (int i = 0; i < s.size(); i += (2 * k)) {
            // 1. 每隔 2k 个字符的前 k 个字符进行反转
            // 2. 剩余字符小于 2k 但大于或等于 k 个，则反转前 k 个字符
            if (i + k <= s.size()) {
                reverse(s, i, i + k - 1);
                continue;
            }
            // 3. 剩余字符少于 k 个，则将剩余字符全部反转。
            reverse(s, i, s.size() - 1);
        }
        return s;
    }
};


//当需要固定规律一段一段去处理字符串的时候，要想想在for循环的表达式上做做文章。

//reverse is called on half‐open ranges ([first, last)
```
卡码网：54.替换数字
TBD
## 代码随想录 Day 9

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
```cpp
class Solution {
public:
    void removeExtraSpaces(string& s) {
        int slow = 0;
        for(int i=0; i<s.size(); i++){
            if(s[i] != ' '){
                // If slow != 0, we have already written at least one word;
                // so before writing this new word, insert exactly one space.
                // (This also prevents a leading space.)
                if (slow != 0) 
                    s[slow++] = ' ';
                
                while (i < s.size() && s[i] != ' ') {
                   s[slow++] = s[i++];
                }
            }
        }
        return s.resize(slow); 
    }

    void reverse(string& s, int start, int end) {
        for (int i = start, j = end; i < j; i++, j--) {
            swap(s[i], s[j]);
        }   
    }

    string reverseWords(string s) {
        removeExtraSpaces(s);
        reverse(s,0,s.size()-1);

        int i=0;
        for (int j=0; j<=s.size(); j++){
            if(s[j] == ' '|| j == s.size()){
                reverse(s, i, j-1);
                i=j+1;
            }
        }
        
        return s;
    }
};

// when j==n, s[n] out of bound, leading to undefined behavior

// 整个字符串都反转过来，那么单词的顺序指定是倒序了，只不过单词本身也倒序了，那么再把单词反转一下，单词不就正过来了。
```

1 delete space 2 reverse 3 recombine
卡码网：55.右旋转字符串
TBD
28. 实现 strStr()
```cpp
class Solution {
public:
    // aabaaf
    // 01212
    void getNext(int* next, string& needle){
        int j = 0;
        // i: suffix end
        // j: prefix end
        // 1. initialization, differnt, same, update
        next[0]= j; 
        for(int i=1; i<needle.size();i++){
            
            while(needle[j] != needle[i] && j>0){
                j=next[j-1];
            }

            if(needle[i]==needle[j]){
                j++;
            }
            next[i]=j;
        }
    }

    int strStr(string haystack, string needle) {
        if (needle.size() == 0) {
            return 0;
        }
		vector<int> next(needle.size());
		getNext(&next[0], needle);

        int j=0;
        for (int i=0; i<haystack.size(); i++){
            while(needle[j] != haystack[i] && j> 0){
                j=next[j-1];
            }
            if(needle[j]== haystack[i]){
                j++;
            }
            if(j== needle.size() ){
                return i-j+1;
            }
        }
        return -1;
        
        
    }
};

```

459.重复的子字符串
```cpp
class Solution {
public:
    //当 最长相等前后缀不包含的子串的长度 可以被 字符串s的长度整除，那么不包含的子串 就是s的最小重复子串。
    //next[len - 1] != 0，则说明字符串有最长相同的前后缀
    //最长相等前后缀的长度为：next[len - 1]
    //len - next[len - 1] 是最长相等前后缀不包含的子串的长度。

    void getNext (int* next, const string& s){
        next[0] = 0;
        int j = 0;
        for(int i = 1;i < s.size(); i++){
            while(j > 0 && s[i] != s[j]) {
                j = next[j - 1];
            }
            if(s[i] == s[j]) {
                j++;
            }
            next[i] = j;
        }
    }

    bool repeatedSubstringPattern(string s) {
        if (s.size() == 0) {
            return false;
        }
        int next[s.size()];
        getNext(next, s);
        int len = s.size();
        if (next[len - 1] != 0 && len % (len - (next[len - 1] )) == 0) {
            return true;
        }
        return false;
       
    }
};
```
## 代码随想录 Day 10
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
cpp```
class MyQueue {
public:
    stack<int> stIn;
    stack<int> stOut;
    MyQueue() {
        
    }
    
    void push(int x) {
        stIn.push(x);
    }
    
    int pop() {
        if(stOut.empty()){
  
            while(!stIn.empty()){
                stOut.push(stIn.top());
                stIn.pop();
            }
            
        }
        int res = stOut.top();
        stOut.pop();
        return  res;
        
    }
    
    int peek() {
        if(stOut.empty()){
            while(!stIn.empty()){

                stOut.push(stIn.top());
                stIn.pop();
            }
        }
        return stOut.top();
    }
    
    bool empty() {
        return stOut.empty() && stIn.empty();
    }
};

/**
 * Your MyQueue object will be instantiated and called as such:
 * MyQueue* obj = new MyQueue();
 * obj->push(x);
 * int param_2 = obj->pop();
 * int param_3 = obj->peek();
 * bool param_4 = obj->empty();
 */

 // pop not return element 
 // int res = this->pop(); 
 // 直接使用已有的pop函数
     
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

```cpp
class MyStack {

public:
    queue<int> que;

    MyStack() {
        
    }
    
    void push(int x) {
        que.push(x);    
    }
    
    int pop() {
        int size = que.size();
        while(size>1){
            int tmp = que.front();
            que.pop();
            que.push(tmp);
            size--; 
        }
        int res = que.front();
        que.pop();
        return res;
    }
    
    int top() {
        int res= this->pop();
        this->push(res);
        return res;
    }
    
    bool empty() {
        return que.empty();
    }
};

/**
 * Your MyStack object will be instantiated and called as such:
 * MyStack* obj = new MyStack();
 * obj->push(x);
 * int param_2 = obj->pop();
 * int param_3 = obj->top();
 * bool param_4 = obj->empty();
 */

 // update size in pop
```

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

```cpp
class Solution {
public:
    bool isValid(string s) {
        if (s.size() % 2 != 0) return false; // 如果s的长度为奇数，一定不符合要求
        stack<char> st;

        for (int i = 0; i < s.size(); i++) {
            if (s[i] == '(') st.push(')');
            else if (s[i] == '{') st.push('}');
            else if (s[i] == '[') st.push(']');
            // 第三种情况：遍历字符串匹配的过程中，栈已经为空了，没有匹配的字符了，说明右括号没有找到对应的左括号 return false
            // 第二种情况：遍历字符串匹配的过程中，发现栈里没有我们要匹配的字符。所以return false
            else if (st.empty() || st.top() != s[i]) return false;
            else st.pop(); // st.top() 与 s[i]相等，栈弹出元素
        }
        // 第一种情况：此时我们已经遍历完了字符串，但是栈不为空，说明有相应的左括号没有右括号来匹配，所以return false，否则就return true
        return st.empty();
    }
};

// 3 conditions 
```

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

```cpp
class Solution {
public:
    string removeDuplicates(string s) {
        // stack<char> st;
        // for(char i: s){
        //     if(st.empty()|| st.top() != i){
        //         st.push(i);
        //     }
        //     else if(st.top() == i){
        //         st.pop();
        //     }
        // } 
        // //
        // string result = "";
        // while (!st.empty()) { // 将栈中元素放到result字符串汇总
        //     result += st.top();
        //     st.pop();
        // }
        // reverse (result.begin(), result.end()); // 此时字符串需要反转一下
        // return result;
        string result;
        for(char i : s) {
            if(result.empty() || result.back() != i) {
                result.push_back(i);
            }
            else {
                result.pop_back();
            }
        }
        return result;
    }
};

//字符串直接作为栈
```

## 代码随想录 Day 11 (07/09/24)

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

```cpp
class Solution {
public:
    int evalRPN(vector<string>& tokens) {
        // 力扣修改了后台测试数据，需要用longlong
        stack<long long> st; 
        for (int i = 0; i < tokens.size(); i++) {
            if (tokens[i] == "+" || tokens[i] == "-" || tokens[i] == "*" || tokens[i] == "/") {
                long long num1 = st.top();
                st.pop();
                long long num2 = st.top();
                st.pop();
                if (tokens[i] == "+") st.push(num2 + num1);
                if (tokens[i] == "-") st.push(num2 - num1);
                if (tokens[i] == "*") st.push(num2 * num1);
                if (tokens[i] == "/") st.push(num2 / num1);
            } else {
                st.push(stoll(tokens[i]));
            }
        }

        long long result = st.top();
        st.pop(); // 把栈里最后一个元素弹出（其实不弹出也没事）
        return result;
    }
};
```
239. 滑动窗口最大值
```cpp

class Solution {
private: 
    class MyQueue { //单调队列（从大到小）
    public:
        deque<int> que; // 使用deque来实现单调队列
        // 每次弹出的时候，比较当前要弹出的数值是否等于队列出口元素的数值，如果相等则弹出。
        // 同时pop之前判断队列当前是否为空。
        void pop(int value) {
            if (!que.empty() && value == que.front()) {
                que.pop_front();
            }
        }
        // 如果push的数值大于入口元素的数值，那么就将队列后端的数值弹出，直到push的数值小于等于队列入口元素的数值为止。
        // 这样就保持了队列里的数值是单调从大到小的了。
        void push(int value) {
            while (!que.empty() && value > que.back()) {
                que.pop_back();
            }
            que.push_back(value);

        }
        // 查询当前队列里的最大值 直接返回队列前端也就是front就可以了。
        int front() {
            return que.front();
        }
    };


public:
    vector<int> maxSlidingWindow(vector<int>& nums, int k) {
        MyQueue que;
        vector<int> result;
        for (int i = 0; i < k; i++) { // 先将前k的元素放进队列
            que.push(nums[i]);
        }
        result.push_back(que.front()); // result 记录前k的元素的最大值
        for (int i = k; i < nums.size(); i++) {
            que.pop(nums[i - k]); // 滑动窗口移除最前面元素
            que.push(nums[i]); // 滑动窗口前加入最后面的元素
            result.push_back(que.front()); // 记录对应的最大值
        }
        return result;
    }
};

//单调队列
```


347. 前 K 个高频元素
```cpp
class Solution {
public:
    // 小顶堆
    class mycomparison {
    public:
        bool operator()(const pair<int, int>& lhs, const pair<int, int>& rhs) {
            return lhs.second > rhs.second;
        }
    };
    vector<int> topKFrequent(vector<int>& nums, int k) {
        // 要统计元素出现频率
        unordered_map<int, int> map; // map<nums[i],对应出现的次数>
        for (int i = 0; i < nums.size(); i++) {
            map[nums[i]]++;
        }

        // 对频率排序
        // 定义一个小顶堆，大小为k
        priority_queue<
            pair<int, int>, 
            vector<pair<int, int>>, 
            mycomparison
        > pri_que;

        // 用固定大小为k的小顶堆，扫面所有频率的数值
        for (unordered_map<int, int>::iterator it = map.begin(); it != map.end(); it++) {
            pri_que.push(*it);
            if (pri_que.size() > k) { // 如果堆的大小大于了K，则队列弹出，保证堆的大小一直为k
                pri_que.pop();
            }
        }

        // 找出前K个高频元素，因为小顶堆先弹出的是最小的，所以倒序来输出到数组
        vector<int> result(k);
        for (int i = k - 1; i >= 0; i--) {
            result[i] = pri_que.top().first;
            pri_que.pop();
        }
        return result;
    }
};
//优先级队列呢？
//其实就是一个披着队列外衣的堆
//it is an iterator over map, so *it is a pair<int,int> = (someValue, itsCount).
```

## 代码随想录 Day 13
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

```cpp
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
 *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
 *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
 * };
 */
class Solution {
public:
    vector<int> inorderTraversal(TreeNode* root) {
        // vector<int> result;
        // stack<TreeNode*> st;
        // if (root != NULL) st.push(root);
        // while(!st.empty()){
        //     TreeNode* top = st.top();
        //     if(top!=NULL){
        //         st.pop();
        //         if(top->right)st.push(top->right);
        //         st.push(top);                          
        //         st.push(NULL); 
        //         if(top->left)st.push(top->left);
        //     }
        //     else{
        //         st.pop();
        //         top= st.top();
        //         st.pop();
        //         result.push_back(top->val);

        //     }
        
        // }
        // return result;
        vector<int> result;
        stack<pair<TreeNode*, bool>> st;
        if (root != nullptr)
            st.push(make_pair(root, false));
        while(!st.empty()){
            auto top =st.top().first;
            auto visited= st.top().second;
            st.pop();
            if (visited) { 
                result.push_back(top->val);
            }
            else{

                if (top->right){
                    st.push(make_pair(top->right, false));
                }
                
            
                st.push(make_pair(top, true));

                if (top->left){
                    st.push(make_pair(top->left, false)); // 左儿子最后入栈，最先出栈
                }
            }

        }
        return result;
    }
};

// the check time and add time is different
// adding status value for checked node
```

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

```cpp
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
 *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
 *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
 * };
 */
class Solution {
public:
    vector<vector<int>> levelOrder(TreeNode* root) {
        queue<TreeNode*> que;
        if(root!= NULL) que.push(root);
        vector<vector<int>> result;
        while(!que.empty()){
            int size = que.size();
            vector<int> vec;
            for(int i=0; i<size;i++){
                TreeNode* node = que.front();
                que.pop();
                vec.push_back(node->val);
                if (node->left) que.push(node->left);
                if (node->right) que.push(node->right);
            }
            result.push_back(vec);
        }
        return result;

    }
};
//  void order(TreeNode* cur, vector<vector<int>>& result, int depth)
//     {
//         if (cur == nullptr) return;
//         if (result.size() == depth) result.push_back(vector<int>());
// when size ==depth, means the current depth has not been touch
//         result[depth].push_back(cur->val);
//         order(cur->left, result, depth + 1);
//         order(cur->right, result, depth + 1);
//     }
```
107. Binary Tree Level Order Traversal II
```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def levelOrderBottom(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        if not root:
            return []
        queue = collections.deque([root])
        result = []
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
        return result[::-1]
```
199. Binary Tree Right Side View
```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def rightSideView(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
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
                if cur.right:
                    queue.append(cur.right)
                if cur.left:
                    queue.append(cur.left)
            result.append(level[0])
        return result
```
637. Average of Levels in Binary Tree
```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def averageOfLevels(self, root):
        """
        :type root: TreeNode
        :rtype: List[float]
        """
        if not root:
            return []
        result = []
        queue = collections.deque([root])
        while queue:
            level=[]
            for _ in range(len(queue)):
                cur = queue.popleft()
                level.append(cur.val)
                if cur.left:
                    queue.append(cur.left)
                if cur.right:
                    queue.append(cur.right)
                
            average = float(sum(level)) / len(level) #
            result.append(round(average, 5))         #a third intermediate value influnence the result
        return result
```

TBD
429
```python
```
515
116
117
104
111
## 代码随想录 Day 14
226. Invert Binary Tree
```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def invertTree(self, root):
        """
        :type root: TreeNode
        :rtype: TreeNode
        """
        queue= collections.deque([root])
        while queue:
            for _ in range(len(queue)):
                cur = queue.popleft()
                if cur: # avoid none type 
                    cur.left, cur.right = cur.right, cur.left
                    if cur.left:
                        queue.append(cur.left)
                    if cur.right:
                        queue.append(cur.right)
        return root
```
(breath first )
check node not NULL first 
101. Symmetric Tree
```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def isSymmetric(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        if not root:
            return False
        queue=collections.deque()
        queue.append(root.left)
        queue.append(root.right)
        while queue:
            leftNode = queue.popleft()
            rightNode = queue.popleft()
            if rightNode == None and leftNode == None:
                continue
            if (rightNode !=None and leftNode == None) or (rightNode ==None and leftNode != None) or leftNode.val != rightNode.val:
                return False
            queue.append(leftNode.left)
            queue.append(rightNode.right)
            queue.append(leftNode.right)
            queue.append(rightNode.left)
        return True
```
depth first

```cpp
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
 *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
 *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
 * };
 */
class Solution {
public:
    // recursive 
    // bool compare(TreeNode* left, TreeNode* right){
    //     if(left ==NULL && right ==NULL) return true;
    //     else if(left != NULL && right == NULL) return false;
    //     else if(left == NULL && right != NULL) return false;
    //     else if(left->val != right->val) return false;

    //     bool leftS = compare(left->left, right->right);
    //     bool rightS = compare(left->right, right->left);
    //     bool mid= leftS && rightS;
    //     return mid;
    // }

    bool isSymmetric(TreeNode* root) {
        // if(root ==NULL) return true;
        // bool res = compare(root->left, root->right);
        // return res;

        // iterative
        if (root == NULL) return true;
        stack<TreeNode*> st; // 这里改成了栈
        st.push(root->left);
        st.push(root->right);
        while (!st.empty()) {
            TreeNode* rightNode = st.top(); st.pop();
            TreeNode* leftNode = st.top(); st.pop();
            if (!leftNode && !rightNode) {
                continue;
            }
            if ((!leftNode || !rightNode || (leftNode->val != rightNode->val))) {
                return false;
            }
            st.push(leftNode->left);
            st.push(rightNode->right);
            st.push(leftNode->right);
            st.push(rightNode->left);
        }
        return true;

    }
};
```
104. Maximum Depth of Binary Tree
```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def maxDepth(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        return self.getdepth(root)
        
    def getdepth(self, node):
        if not node:
            return 0
        leftheight = self.getdepth(node.left) #左
        rightheight = self.getdepth(node.right) #右
        height = 1 + max(leftheight, rightheight) #中
        return height
```
```cpp
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
 *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
 *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
 * };
 */
class Solution {
public:

    int maxDepth(TreeNode* root) {
        // recursive
        // if (!root) return 0;
        // return 1 + max(maxDepth(root->left), maxDepth(root->right)); 

        //iterative
        if (root == NULL) return 0;
        int depth = 0;
        queue<TreeNode*> que;
        que.push(root);
        while(!que.empty()) {
            int size = que.size();
            depth++; // 记录深度
            for (int i = 0; i < size; i++) {
                TreeNode* node = que.front();
                que.pop();
                if (node->left) que.push(node->left);
                if (node->right) que.push(node->right);
            }
        }
        return depth;
    }
};
```

????
```cpp
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
 *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
 *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
 * };
 */
class Solution {
public:

    TreeNode* invertTree(TreeNode* root) {
        // 1. resursive
        // if (!root) return nullptr;

        // invertTree(root->left);   // left
        // invertTree(root->right);  // right
        // // node
        // TreeNode* tmp = root->left;
        // root->left = root->right;
        // root->right = tmp;
        // return root;

        // 2. iterative
        // stack<TreeNode*> st;
        // if(root!= NULL) st.push(root);
        // while(!st.empty()){
        //     TreeNode* top = st.top();
        //     if (top!= NULL){
        //         st.pop();
        //         if(top->right) st.push(top->right);
        //         if(top->left) st.push(top->left);
        //         st.push(top);
        //         st.push(NULL);
        //     }
        //     else{
        //         st.pop();
        //         top = st.top();
        //         swap(top->left, top->right);
        //         st.pop();
        //     }
        // }
        // return root;

        // 3. breath first
        queue<TreeNode*> que;
        if (root != NULL) que.push(root);
        while (!que.empty()) {
            int size = que.size();
            for (int i = 0; i < size; i++) {
                TreeNode* node = que.front();
                que.pop();
                swap(node->left, node->right); // 节点处理
                if (node->left) que.push(node->left);
                if (node->right) que.push(node->right);
            }
        }
        return root;

    }
};
```

559 
TBD
111. Minimum Depth of Binary Tree
```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def minDepth(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        if not root:
            return 0
        depth = 0
        queue = collections.deque([root])
        
        while queue:
            depth += 1 
            for _ in range(len(queue)):
                node = queue.popleft()
                
                if not node.left and not node.right:
                    return depth
            
                if node.left:
                    queue.append(node.left)
                    
                if node.right:
                    queue.append(node.right)

        return depth
```
when meet a leaf, that means the shortest depth from root

```cpp
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
 *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
 *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
 * };
 */
class Solution {
public:
    int minDepth(TreeNode* root) {
        // recursive
        // if (root == NULL) return 0;
        
        // int l = minDepth(root->left);
        // int r = minDepth(root->right);
        
        // if (root->left != NULL && root->right ==NULL ){
        //     return l+1;    
        // }
        // if (root->left == NULL && root->right !=NULL ){
        //     return r+1;    
        // }
        // return 1+ min(l,r);
        if (root == NULL) return 0;
        int depth = 0;
        queue<TreeNode*> que;
        que.push(root);
        while(!que.empty()) {
            int size = que.size();
            depth++; // 记录最小深度
            for (int i = 0; i < size; i++) {
                TreeNode* node = que.front();
                que.pop();
                if (node->left) que.push(node->left);
                if (node->right) que.push(node->right);
                if (!node->left && !node->right) { // 当左右孩子都为空的时候，说明是最低点的一层了，退出
                    return depth;
                }
            }
        }
        return depth;
    }
};
```
## 代码随想录 Day 15
110. Balanced Binary Tree
```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def isBalanced(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        result = self.getHeight(root)
        if result == -1:
            return False
        else:
            return True

    def getHeight(self, node):
        if node == None:
            return 0 
        
        leftHeight = self.getHeight(node.left)
        if leftHeight == -1:
            return -1
        rightHeight = self.getHeight(node.right)
        if rightHeight ==-1:
            return -1

        if leftHeight- rightHeight >1 or leftHeight- rightHeight <-1:
            return -1
        else:
            return 1+max(leftHeight, rightHeight)
```
因为求深度可以从上到下去查 所以需要前序遍历（中左右），而高度只能从下到上去查，所以只能后序遍历（左右中）

```cpp
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
 *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
 *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
 * };
 */
class Solution {
public:
    int getHeight(TreeNode* cur){
        if(cur==NULL) return 0;
        int l = getHeight(cur->left);
        if (l == -1) return -1;
        int r= getHeight(cur->right);
        if (r==-1) return -1;
        if((r-l)<(-1) || (r-l)>1) return -1;
        else return 1+max(l,r);
    }

    bool isBalanced(TreeNode* root) {
        return getHeight(root) == -1 ? false : true;
    }
};
```

TBD iterative method
257. Binary Tree Paths
```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def binaryTreePaths(self, root):
        """
        :type root: TreeNode
        :rtype: List[str]
        """
        result = []
        self.pathfind(root, result, [])
        return result
    
    def pathfind(self, node, result, path):
        path.append(str(node.val))
        if node.left == None and node.right == None:
            result.append('->'.join(path))
            return 
        if node.left:
            #self.pathfind(node.left,result,path)
            self.pathfind(node.left,result,path[:])
        if node.right:
            self.pathfind(node.right,result,path[:]) 
```
difference between path and path[:]
path: original refernce 
path[:]: shallow copy

```cpp
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
 *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
 *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
 * };
 */
class Solution {
public:
    void Path(vector<string>& res, TreeNode* root, vector<int>& path){
        path.push_back(root->val);
        if(root->left ==NULL && root->right == NULL){
            string sPath;
            for (int i = 0; i < path.size() - 1; i++) {
                sPath += to_string(path[i]);
                sPath += "->";
            }
            sPath += to_string(path[path.size() - 1]);
            res.push_back(sPath);
            return;
        }
        if(root->left){
            Path(res, root->left, path);
            path.pop_back();
        }
        if(root->right){
            Path(res, root->right, path);
            path.pop_back();
        }
    } 

    vector<string> binaryTreePaths(TreeNode* root) {
        vector<string> res;
        vector<int> path;
        if (root == NULL) return res;
        Path(res, root, path);
        return res;
    }
};
```

404. Sum of Left Leaves
```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def sumOfLeftLeaves(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        return self.sumLeft(root, 0)
    
    def sumLeft(self, node, sum):
        # Base case: if the node is None, return the current sum
        if node is None:
            return sum

        # Check if the left child is a leaf
        if node.left and not node.left.left and not node.left.right:
            sum += node.left.val
        
        # Recur for left and right subtrees, passing the current sum
        sum = self.sumLeft(node.left, sum)
        sum = self.sumLeft(node.right, sum)
        
        return sum
```

only left leave take into account

```cpp
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
 *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
 *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
 * };
 */
class Solution {
public:
    int sumOfLeftLeaves(TreeNode* root) {
        // recursive
        // if (root == NULL) return 0;
        // if (root->left == NULL && root->right== NULL) return 0;

        // int leftValue = sumOfLeftLeaves(root->left);    // 左
        // if (root->left && !root->left->left && !root->left->right) { // 左子树就是一个左叶子的情况
        //     leftValue = root->left->val;
        // }
        // int rightValue = sumOfLeftLeaves(root->right);  // 右

        // int sum = leftValue + rightValue;               // 中
        // return sum;
        
        //iterative
        stack<TreeNode*> st;
        if (root == NULL) return 0;
        st.push(root);
        int result = 0;
        while (!st.empty()) {
            TreeNode* cur;
            int size = st.size();
            for(int i=0; i<size;i++){
                cur = st.top();
                st.pop();
                if (cur->left){
                    if(!cur->left->left && !cur->left->right){
                        result+= cur->left->val;
                    }
                    else{
                        st.push(cur->left);
                    }
                    
                }
                if(cur->right){
                    st.push(cur->right);
                }
                
            }
        }
        return result;
        
    }
};
```
use parent node 

513. Find Bottom Left Tree Value
```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def findBottomLeftValue(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        value, depth = self.leftValue(root, 0)
        return value

    def leftValue(self, node, depth):
        if node.left == None and node.right == None:
            return node.val, depth
        
        if node.left:
            l_value, l_depth = self.leftValue(node.left, depth+1)
        if node.right:
            r_value, r_depth = self.leftValue(node.right, depth+1)

        if node.left and node.right:
            if r_depth > l_depth:
                return  r_value, r_depth
            else:
                return l_value, l_depth
        elif node.left and not node.right:
            return l_value, l_depth
        elif not node.left and  node.right:
            return  r_value, r_depth
            
```

```cpp
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
 *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
 *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
 * };
 */
class Solution {
public:
int maxDepth = INT_MIN;
    int result;
    void traversal(TreeNode* root, int depth) {
        if (root->left == NULL && root->right == NULL) {
            if (depth > maxDepth) {
                maxDepth = depth;
                result = root->val;
            }
            return;
        }
        if (root->left) {
            depth++;
            traversal(root->left, depth);
            depth--; // 回溯
        }
        if (root->right) {
            depth++;
            traversal(root->right, depth);
            depth--; // 回溯
        }
        return;
    }

    int findBottomLeftValue(TreeNode* root) {
        // traversal(root, 0);
        // return result;    

         queue<TreeNode*> que;
        if (root != NULL) que.push(root);
        int result = 0;
        while (!que.empty()) {
            int size = que.size();
            for (int i = 0; i < size; i++) {
                TreeNode* node = que.front();
                que.pop();
                if (i == 0) result = node->val; // 记录最后一行第一个元素
                if (node->left) que.push(node->left);
                if (node->right) que.push(node->right);
            }
        }
        return result;
        // each time update with the first one in the layer
    }
};
```

## 代码随想录 Day 16

## 代码随想录 Day 17
## 代码随想录 Day 18

## 代码随想录 Day 20
## 代码随想录 Day 21

## 代码随想录 Day 22
77. Combinations
```python
class Solution(object):
    def combine(self, n, k):
        """
        :type n: int
        :type k: int
        :rtype: List[List[int]]
        """
        result = []  # 存放结果集
        self.backtracking(n, k, 1, [], result)
        return result
    def backtracking(self, n, k, startIndex, path, result):
        if len(path) == k:
            result.append(path[:])
            return
        for i in range(startIndex, n - (k - len(path)) + 2):  # 优化的地方
            path.append(i)  # 处理节点
            self.backtracking(n, k, i + 1, path, result)
            path.pop()  # 回溯，撤销处理的节点
```
回溯三部曲
void backtracking(参数) {
    if (终止条件) {
        存放结果;
        return;
    }

    for (选择：本层集合中元素（树中节点孩子的数量就是集合的大小）) {
        处理节点;
        backtracking(路径，选择列表); // 递归
        回溯，撤销处理结果
    }
}

```cpp
class Solution {
public:
    vector<vector<int>> res; // 存放符合条件结果的集合
    vector<int> path; // 用来存放符合条件单一结果
    void backtracking(int n, int k, int startIndex){
        
        if (path.size() == k){
            res.push_back(path);
            return;
        }

        for (int i = startIndex; i <= 1+n-(k-path.size()); i++) { // 控制树的横向遍历
            path.push_back(i); // 处理节点
            backtracking(n, k, i + 1); // 递归：控制树的纵向遍历，注意下一层搜索要从i+1开始
            path.pop_back(); // 回溯，撤销处理的节点
        }
    }
    
    
    vector<vector<int>> combine(int n, int k) {
        backtracking(n,k,1);
        return res;
    }
};
```
216. Combination Sum III
```python
class Solution(object):
    def combinationSum3(self, k, n):
        """
        :type k: int
        :type n: int
        :rtype: List[List[int]]
        """
    #     result = []
    #     self.backtracking(k,n,[],result,1)
    #     return result
    # def backtracking(self, k,n, path,result, startIndex):
    #     if len(path) == k:
    #         result.append(path[:])
    #         return
    #     for i in range (startIndex,n-k+1):
    #         if n-i>0:
    #             path.append(i)
    #             self.backtracking(k-1,n, path,result, i)
    #             path.pop(i)
        result = []  # 存放结果集
        self.backtracking(n, k, 0, 1, [], result)
        return result

    def backtracking(self, targetSum, k, currentSum, startIndex, path, result):
        if currentSum > targetSum:  # 剪枝操作
            return  # 如果path的长度等于k但currentSum不等于targetSum，则直接返回
        if len(path) == k:
            if currentSum == targetSum:
                result.append(path[:])
            return
        for i in range(startIndex, 9 - (k - len(path)) + 2):  # 剪枝
            currentSum += i  # 处理
            path.append(i)  # 处理
            self.backtracking(targetSum, k, currentSum, i + 1, path, result)  # 注意i+1调整startIndex
            currentSum -= i  # 回溯
            path.pop()  # 回溯

```
1. currentSum> targetSum剪枝
2. 9-(k-len(path))+2) : k-len(path) need number of integer
                       9-: max start place 
			+2 start and right exclusive 

time complexity: O(n*2^n), 2^n max combination from C(n,k), each need O(n) time to push

k-size<=9-i+1

```cpp
class Solution {
public:
    vector<vector<int>> res;
    vector<int> path;
    void backtracking(int k, int tar, int sum, int start){
        if(path.size() ==k){
            if(tar==sum){
                res.push_back(path);
            }
            return; 
        }
        
        for(int i=start; i<= 9 - (k - path.size()) + 1; i++ ){
            path.push_back(i);
            if (sum+i>tar){
                path.pop_back();
                return;    
            }
            backtracking(k,tar, sum+i, i+1);
            path.pop_back();
        }
    }
    vector<vector<int>> combinationSum3(int k, int n) {
        backtracking(k, n,0,1 );
        return res;
    }
};
```
17. Letter Combinations of a Phone Number
```python
class Solution(object):
    def __init__(self):
        self.letterMap = [
            "",     # 0
            "",     # 1
            "abc",  # 2
            "def",  # 3
            "ghi",  # 4
            "jkl",  # 5
            "mno",  # 6
            "pqrs", # 7
            "tuv",  # 8
            "wxyz"  # 9
        ]
        self.result = []
        self.s = ""
    def letterCombinations(self, digits):
        """
        :type digits: str
        :rtype: List[str]
        """
        result = []
        if len(digits) == 0:
            return result
        self.getCombinations(digits, 0, [], result)
        return result
 
    
    def getCombinations(self, digits, index, path, result):
        if index == len(digits):
            result.append(''.join(path))
            return
        digit = int(digits[index])
        letters = self.letterMap[digit]
        for letter in letters:
            path.append(letter)
            self.getCombinations(digits, index + 1, path, result)
            path.pop()
```
use mapping for letter
no startIndex, this is combination of different set, not combination inside one set

```cpp
class Solution {
private:
    const string letterMap[10] = {
        "", // 0
        "", // 1
        "abc", // 2
        "def", // 3
        "ghi", // 4
        "jkl", // 5
        "mno", // 6
        "pqrs", // 7
        "tuv", // 8
        "wxyz", // 9
    };

public:
    vector<string> result;
    string s;
    void backtracking(const string& digits, int index) {
        if (index == digits.size()) {
            result.push_back(s);
            return;
        }
        int digit = digits[index] - '0';        // 将index指向的数字转为int
        string letters = letterMap[digit];      // 取数字对应的字符集
        for (int i = 0; i < letters.size(); i++) {
            s.push_back(letters[i]);            // 处理
            backtracking(digits, index + 1);    // 递归，注意index+1，一下层要处理下一个数字了
            s.pop_back();                       // 回溯
        }
    }
    vector<string> letterCombinations(string digits) {
        if (digits.size() == 0) {
            return result;
        }
        backtracking(digits, 0);
        return result;
    }
};
```
ASCII code map number string with int, e.g. '0'=48, '1'=49
## 代码随想录 Day 23
39. Combination Sum
```python
class Solution(object):
    def combinationSum(self, candidates, target):
        """
        :type candidates: List[int]
        :type target: int
        :rtype: List[List[int]]
        """
        result = []
        candidates.sort() # ordering
        self.backtracking(candidates, target, result, [], 0, 0)
        return result

    def backtracking(self, candidates, target, result, path, startIndex,sum):
        # if curSum>target:
        #     return
        if sum == target:
            result.append(path[:])
            return
        for i in range(startIndex, len(candidates)): # pruning
            if candidates[i] + sum>target:
                break
            sum +=candidates[i]
            path.append(candidates[i])
            self.backtracking(candidates, target, result, path, i, sum)
            path.pop()
            sum-=candidates[i]
```
use sort for security

```cpp
class Solution {
public:
    vector<vector<int>> res;
    vector<int> path;

    void backtracking(vector<int>& can, int target, int curSum, int start){
        if (curSum == target){
            res.push_back(path);
            
            return;
        }

        for(int i=start; i< can.size() && curSum + can[i] <= target; i++){
            curSum+= can[i];
            path.push_back(can[i]);
            backtracking(can, target, curSum, i);
            curSum-=can[i];
            path.pop_back();

        }

        return;
    }
    vector<vector<int>> combinationSum(vector<int>& candidates, int target) {

        if (target ==0) return res;
        int curSum=0;
        sort(candidates.begin(), candidates.end());    
        backtracking(candidates, target, curSum,0);
        return res;
    }
};
// 一个集合求组合，需要startIndex
// can repeat, so startIndex not +1 when backtracking
```

40. Combination Sum II
```python
class Solution(object):
    def combinationSum2(self, candidates, target):
        """
        :type candidates: List[int]
        :type target: int
        :rtype: List[List[int]]
        """
        result = []
        candidates.sort() # ordering
        self.backtracking(candidates, target, result, [], 0, 0)
        return result

    def backtracking(self, candidates, target, result, path, startIndex,sum):
        # startIndex for pruning
        if sum == target:
            result.append(path[:])
            return
        for i in range(startIndex, len(candidates)): # pruning
            # 要对同一树层使用过的元素进行跳过
            if (i > startIndex and candidates[i] == candidates[i - 1]):
                continue
            
            if candidates[i] + sum>target:
                break
            sum +=candidates[i]
            path.append(candidates[i])
            self.backtracking(candidates, target, result, path, i+1, sum)
            path.pop()
            sum-=candidates[i]
```
same level pruning

```cpp
class Solution {
public:
    vector<vector<int>> res;
    vector<int> path;
    void backtracking(vector<int>& candidates, int target, int sum, int startIndex, vector<bool>& used){
        if(sum == target){
            res.push_back(path);
            return;
        }
        for (int i = startIndex; i < candidates.size() && sum + candidates[i] <= target; i++) {
            // used[i - 1] == true，说明同一树枝candidates[i - 1]使用过
            // used[i - 1] == false，说明同一树层candidates[i - 1]使用过
            // 要对同一树层使用过的元素进行跳过
            if (i > 0 && candidates[i] == candidates[i - 1] && used[i - 1] == false) {
                continue;
            }
            sum += candidates[i];
            path.push_back(candidates[i]);
            used[i] = true;
            backtracking(candidates, target, sum, i + 1, used); // 和39.组合总和的区别1：这里是i+1，每个数字在每个组合中只能使用一次
            used[i] = false;
            sum -= candidates[i];
            path.pop_back();
        }
    }
    vector<vector<int>> combinationSum2(vector<int>& candidates, int target) {
        vector<bool> used(candidates.size(), false);
        sort(candidates.begin(), candidates.end());
        backtracking(candidates, target, 0, 0, used);
        return res;
        
    }
};
```

131. Palindrome Partitioning
```python
class Solution(object):
    def partition(self, s):
        """
        :type s: str
        :rtype: List[List[str]]
        """
        result = []
        self.backtracking(s, 0, [], result)
        return result
    def backtracking(self, s, startIndex, path, result):
        # 如果起始位置已经大于s的大小，说明已经找到了一组分割方案了
        if startIndex == len(s):
            result.append(path[:])
            return 
        
        for i in range(startIndex, len(s)):
            # 判断被截取的这一段子串([start_index, i])是否为回文串
            # 若反序和正序相同，意味着这是回文串
            if s[startIndex: i + 1] == s[startIndex: i + 1][::-1]:
                path.append(s[startIndex:i+1])
                self.backtracking(s, i+1, path, result)   # 递归纵向遍历：从下一处进行切割，判断其余是否仍为回文串
                path.pop()             # 回溯
        
```
切割问题可以抽象为组合问题

```cpp
class Solution {
public:
    vector<vector<string>> result;
    vector<string> path;
    bool isPalindrome(const string& s, int start, int end) {
        for (int i = start, j = end; i < j; i++, j--) {
            if (s[i] != s[j]) {
                return false;
            }
        }
        return true;
    }
    void backtracking(string & s, int startIndex){
        if(startIndex == s.size()){
            result.push_back(path);
            return;
        }
        for(int i=startIndex; i< s.size(); i++){
            if (isPalindrome(s, startIndex, i)) { // 是回文子串
                // 获取[startIndex,i]在s中的子串
                string str = s.substr(startIndex, i - startIndex + 1);
                path.push_back(str);
            } else {                // 如果不是则直接跳过
                continue;
            }
            backtracking(s, i + 1); // 寻找i+1为起始位置的子串
            path.pop_back();  
        }
    }
    vector<vector<string>> partition(string s) {
        if(s.size()==0) return result;
        int start =0;
        backtracking(s,start);
        return result;
    }
};
```
## 代码随想录 Day 24
93. Restore IP Addresses
```python
class Solution(object):
    def restoreIpAddresses(self, s):
        """
        :type s: str
        :rtype: List[str]
        """
        result = []
        self.backtracking(s, result, [], 0)
        return result
    def backtracking(self, s, result, path, startIndex):
        if len(path) == 4 and startIndex == len(s):
            result.append(".".join(path))
            return 
        if len(path) > 4:  # 剪枝
            return
        for i in range(startIndex, min(startIndex + 3, len(s))): # pruning
            if int(s[startIndex:i+1])<=255:
                if (i!= startIndex and s[startIndex]!="0") or i == startIndex:
                    path.append(s[startIndex:i+1])
                    self.backtracking(s, result, path, i+1)
                    path.pop()
```
2 pruning method

```cpp
class Solution {
public:
    vector<string> res;
    bool isValid(const string& s, int start, int end) {
        if (start > end) {
            return false;
        }
        if (s[start] == '0' && start != end) { // 0开头的数字不合法
                return false;
        }
        int num = 0;
        for (int i = start; i <= end; i++) {
            if (s[i] > '9' || s[i] < '0') { // 遇到非数字字符不合法
                return false;
            }
            num = num * 10 + (s[i] - '0');
            if (num > 255) { // 如果大于255了不合法
                return false;
            }
        }
        return true;
    }

    void backtracking(string& s, int startIndex, int point){
        if(point ==3){
            if (isValid(s, startIndex, s.size() - 1)) {
                res.push_back(s);
            }
            return;
        }
        for(int i=startIndex; i<startIndex+3 && s.size();i++){
            if (isValid(s, startIndex, i)) {
                s.insert(s.begin() + i + 1 , '.');
                point++;
                backtracking(s, i+2, point);
                point--;
                s.erase(s.begin() + i + 1 );
            }
            else break;
        }
    }
    vector<string> restoreIpAddresses(string s) {
        if(s=="") return res;      
        int start=0, point=0;
        backtracking(s, start, point);
        return res;
    }
};

// check start> end for size(), size()-1 case
// time complexity O(1), as n<20
```
78. Subsets
```python
class Solution(object):
    def subsets(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        result = []
        self.backtracking(nums, 0, [], result)
        return result
    def backtracking(self, nums, startIndex, path, result):
        #剩余集合为空停止
        #不写终止条件，因为本来我们就要遍历整棵树
        result.append(path[:])
        if startIndex == len(nums):
            return
        for i in range(startIndex, len(nums)):
            path.append(nums[i])
            self.backtracking(nums, i + 1, path, result)
            path.pop()
```
append each time

```cpp
class Solution {
public:
    vector<int> path;
    vector<vector<int>> res;
    void backtracking(vector<int>& nums, int start){
        if(start==nums.size()){
            return;
        }
        for(int i= start; i<nums.size(); i++){
            path.push_back(nums[i]);
            res.push_back(path);
            backtracking(nums, i+1);
            path.pop_back();
        }
    }

    vector<vector<int>> subsets(vector<int>& nums) {
        int start = 0;
        res.push_back(path);
        backtracking(nums,start);
        return res;
    }
    //以不需要加终止条件，因为startIndex >= nums.size()，本层for循环本来也结束了。
};
```
90. Subsets II
```python
class Solution(object):
    def subsetsWithDup(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        result = []
        nums.sort()  # Sort the array to ensure duplicates are adjacent
        self.backtracking(nums, 0, [], result)
        return result

    def backtracking(self, nums, startIndex, path, result):
        result.append(path[:]) 
        
        for i in range(startIndex, len(nums)):
            # Skip duplicates: If current element is the same as the previous one, skip it
            if i > startIndex and nums[i] == nums[i - 1]: 
                continue
            
            path.append(nums[i]) 
            self.backtracking(nums, i + 1, path, result) 
            path.pop() 
        
```
1. sort first
2. use i>startIndex to let i == startIndex pass

```cpp
class Solution {
public:
    vector<int> path;
    vector<vector<int>> res;
    void backtracking(vector<int>& nums, int start){
        res.push_back(path);
        for(int i = start; i< nums.size() ; i++){
            if (i > start && nums[i] == nums[i-1]) //!!!
                continue;
            path.push_back(nums[i]);
            backtracking(nums, i+1);
            path.pop_back();
        }
    }
    vector<vector<int>> subsetsWithDup(vector<int>& nums) {
        sort(nums.begin(), nums.end());
        backtracking(nums, 0);
        return res;
        
    }
};
//树层去重需要先排序
// i> start, instead of i>1 to skip in the same layer
```

## 代码随想录 Day 25
491. Non-decreasing Subsequences
TBD
```python

```
```cpp
class Solution {
public:
    vector<int> path;
    vector<vector<int>> res;
    void backtracking(vector<int>& nums, int start){
        if(path.size()>1){
            res.push_back(path);
        }
        int used[201] = {0}; 
        for(int i = start; i< nums.size(); i++ ){
            if(!path.empty() && nums[i]<path.back()) continue;
            if(used[nums[i]+100] == 1) continue;
            path.push_back(nums[i]);
            used[nums[i]+100]=1;
            backtracking(nums, i+1);
            path.pop_back();

        }
    }
    vector<vector<int>> findSubsequences(vector<int>& nums) {
        int start = 0;
        backtracking(nums, 0);
        return res;
    }
};
// use array as hash set since the range of input is [-100,100]
```

46. Permutations
```python
class Solution(object):
    def permute(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        result = []
        self.backtracking(nums, [], [False] * len(nums), result)
        return result

    def backtracking(self, nums, path, used, result):
        if len(path) == len(nums):
            result.append(path[:])
            return
        for i in range(len(nums)):
            if used[i]:
                continue
            used[i] = True
            path.append(nums[i])
            self.backtracking(nums, path, used, result)
            path.pop()
            used[i] = False
```
used list for used number

```cpp
class Solution {
public:
    vector<int> path;
    vector<vector<int>> res;
    void backtracking(vector<int>& nums, vector<bool> used){
        if(path.size()== nums.size()){
            res.push_back(path);
            return;
        }
        for(int i =0; i<nums.size(); i++){
            if(used[i]) continue;
            path.push_back(nums[i]);
            used[i]= true;
            backtracking(nums, used);
            path.pop_back();
            used[i]=false;
        }
    }
    vector<vector<int>> permute(vector<int>& nums) {
        vector<bool> used(nums.size(), false);
        backtracking(nums,used);
        return res;
    }
};
```
47. Permutations II
```python
class Solution(object):
    def permuteUnique(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        nums.sort()
        result = []
        self.backtracking(nums, [], [False] * len(nums), result)
        return result

    def backtracking(self, nums, path, used, result):
        if len(path) == len(nums):
            result.append(path[:])
            return
        for i in range(len(nums)):
            if used[i] or ((i > 0 and nums[i] == nums[i - 1] and used[i - 1])): # duplicate case
                #used[i]= True
                continue
            used[i]=True
            path.append(nums[i])
            self.backtracking(nums, path, used, result)
            path.pop()
            used[i] = False
```
 ((i > 0 and nums[i] == nums[i - 1] and used[i - 1])
 not used[i-1]: level pruning
 used[i-1]: branch pruning
 1a 1b 2 
 1b 1a 2
 same effect 树层上去重效率更高

```cpp
class Solution {
public:
    vector<int> path;
    vector<vector<int>> res;

    void backtrack(const vector<int>& nums, vector<bool>& used) {
        if (path.size() == nums.size()) {
            res.push_back(path);
            return;
        }
        for (int i = 0; i < nums.size(); i++) {
            if (used[i]) 
                continue;                                 // already used this index
            if (i > 0 && nums[i] == nums[i-1] && !used[i-1])
                continue;                                 // skip duplicate
            used[i] = true;
            path.push_back(nums[i]);
            backtrack(nums, used);
            path.pop_back();
            used[i] = false;
        }
    }

    vector<vector<int>> permuteUnique(vector<int>& nums) {
        vector<bool> used(nums.size(),false);
        sort(nums.begin(), nums.end());
        backtrack(nums, used);
        return res;
    }
};
// allow repeat in the nums, so can't create the bool array by the size of the possible input size
```
## 代码随想录 Day 27
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

```cpp
class Solution {
public:
    int findContentChildren(vector<int>& g, vector<int>& s) {
        sort(g.begin(), g.end());
        sort(s.begin(), s.end());
        int index = s.size() - 1; // 饼干数组的下标
        int result = 0;
        for (int i = g.size() - 1; i >= 0; i--) { // 遍历胃口
            if (index >= 0 && s[index] >= g[i]) { // 遍历饼干
                result++;
                index--;
            }
        }
        return result;
    }
};
```
time complexity: O(nlogn), as the sort takes O(nlogn). 

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

```cpp
class Solution {
public:
    int wiggleMaxLength(vector<int>& nums) {
        if (nums.size() <= 1) return nums.size();
        int curDiff = 0; // 当前一对差值
        int preDiff = 0; // 前一对差值
        int result = 1;  // 记录峰值个数，序列默认序列最右边有一个峰值
        for (int i = 0; i < nums.size() - 1; i++) {
            curDiff = nums[i + 1] - nums[i];
            // 出现峰值
            if ((preDiff <= 0 && curDiff > 0) || (preDiff >= 0 && curDiff < 0)) {
                result++;
                preDiff = curDiff; // 注意这里，只在摆动变化的时候更新prediff
            }
        }
        return result;
    }
};
```

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
## 代码随想录 Day 28
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

## 代码随想录 Day 29
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

## 代码随想录 Day 30
452. Minimum Number of Arrows to Burst Balloons
```python
class Solution(object):
    def findMinArrowShots(self, points):
        """
        :type points: List[List[int]]
        :rtype: int
        """
        points.sort(key = lambda x: (x[0], x[1]))
        # avoid negative , not initialize as 0, 0
        count = 1
        end = points[0][1] 
        
        for i in range(1,len(points)):
            if points[i][0]> end:
                end = points[i][1]
                count +=1
            #update end 
            elif points[i][0]<= end:
                end = min(end, points[i][1])
        return count
```
435. Non-overlapping Intervals
```python
class Solution(object):
    def eraseOverlapIntervals(self, intervals):
        """
        :type intervals: List[List[int]]
        :rtype: int
        """
        if not intervals:
            return 0        
        intervals.sort(key = lambda x: x[0])
        end=intervals[0][1]
        count = 0 
        for i in range(1, len(intervals)):
            if intervals[i][0] < end:
                count +=1
                end = min(end, intervals[i][1]) # check to update end
            else:
                end = intervals[i][1] 
        return count 
```
update to smaller end so that more possible non-overlapping
763. Partition Labels
```python
class Solution(object):
    def partitionLabels(self, s):
        """
        :type s: str
        :rtype: List[int]
        """
        last_occurrence = {}  # 存储每个字符最后出现的位置
        for i, ch in enumerate(s):
            last_occurrence[ch] = i

        result = []
        start = 0
        end = 0
        for i, ch in enumerate(s):
            end = max(end, last_occurrence[ch])  # 找到当前字符出现的最远位置
            if i == end:  # 如果当前位置是最远位置，表示可以分割出一个区间
                result.append(end - start + 1)
                start = i + 1

        return result
        
```
method 1 如果找到字符最远出现位置下标和当前下标相等了，则找到了分割点
```python
```
method 2 TBD
## 代码随想录 Day 31
56. Merge Intervals
```python
class Solution(object):
    def merge(self, intervals):
        """
        :type intervals: List[List[int]]
        :rtype: List[List[int]]
        """
		if len(intervals) == 0:
            return result  # 区间集合为空直接返回
        intervals.sort(key=lambda x:(x[0],x[1]))
        output = []
        start = intervals[0][0]
        end = intervals[0][1]
        for i in range(1,len(intervals)):
            if intervals[i][0]<=end:
                end = max(intervals[i][1],end)
            else:
                output.append([start, end])# append the finished part here
                start = intervals[i][0]
                end = intervals[i][1]
        output.append([start,end])
        return output
```

738. Monotone Increasing Digits
```python
class Solution(object):
    def monotoneIncreasingDigits(self, n):
        """
        :type n: int
        :rtype: int
        """

        # for i in range(len(n)-1,0,-1):
        #     if n[i]<n[i+1]:
        #         n[i+1]= n[i]
        #         n[i+1] = 9
        # return n  
         # 将整数转换为字符串
        strNum = str(n)
        # flag用来标记赋值9从哪里开始
        # 设置为字符串长度，为了防止第二个for循环在flag没有被赋值的情况下执行
        flag = len(strNum)
        
        # 从右往左遍历字符串
        for i in range(len(strNum) - 1, 0, -1):
            # 如果当前字符比前一个字符小，说明需要修改前一个字符
            if strNum[i - 1] > strNum[i]:
                flag = i  # 更新flag的值，记录需要修改的位置
                # 将前一个字符减1，以保证递增性质
                strNum = strNum[:i - 1] + str(int(strNum[i - 1]) - 1) + strNum[i:]
        
        # 将flag位置及之后的字符都修改为9，以保证最大的递增数字
        for i in range(flag, len(strNum)):
            strNum = strNum[:i] + '9' + strNum[i + 1:]
        
        # 将最终的字符串转换回整数并返回
        return int(strNum)
```
从个例推断出最大是每次比较结果为(n-1)9的形式
前一个字符减一, 从而避免20->9的情况
968. Binary Tree Cameras
```python
```
TBD

## 代码随想录 Day 32
509. Fibonacci Number
```python
class Solution(object):
    def fib(self, n):
        """
        :type n: int
        :rtype: int
        """
        #i: order
        # dp[i]:value
        dp=[0]*3
        dp[0]=0
        dp[1]=1
        if n==0:
            return 0 
        if n==1:
            return 1
        for i in range(1,n):
            dp[2]=dp[0]+dp[1]
            dp[0]=dp[1]
            dp[1]=dp[2]
        return dp[2]
```
1. i and dp[i] meaning
2. iterate formula 
3. initialize
4. order of iteration
5. use example

70. Climbing Stairs
```python
class Solution(object):
    def climbStairs(self, n):
        """
        :type n: int
        :rtype: int
        """
        # i: number of steps, dp[i]: ways of get 
        dp=[0]*2
        dp[0]=1
        dp[1]=1
        
        if n ==0:
            return 1
        if n ==1:
            return 1
        for i in range(1,n):
            sum = dp[1]+dp[0]
            dp[0]=dp[1]
            dp[1]=sum
        return sum
```
746. Min Cost Climbing Stairs
```python
class Solution(object):
    def minCostClimbingStairs(self, cost):
        """
        :type cost: List[int]
        :rtype: int
        """
        # i : floor, dp[i]:cost to reach i floor
        dp = [0]*(len(cost)+1)
        for i in range(2,len(cost)+1):
            dp[i] = min(dp[i-1]+cost[i-1],dp[i-2]+cost[i-2])
        return dp[len(cost)]
```

## 代码随想录 Day 33
62. Unique Paths
```python
class Solution(object):
    def uniquePaths(self, m, n):
        """
        :type m: int
        :type n: int
        :rtype: int
        """
        # i: mn, dp[i]: number of paths
        # dp[m][n]= dp[m-1][n] + dp[m][n-1]
        dp = [[0 for _ in range(n)] for _ in range(m)]
        dp[0] = [1]*n
        for i in range(m):
            dp[i][0] = 1
        for i in range(1, m):
            for l in range(1, n):
                dp[i][l] = dp[i-1][l]+dp[i][l-1]
        return dp[m-1][n-1] 
               
```
63. Unique Paths II
```python
class Solution(object):
    def uniquePathsWithObstacles(self, obstacleGrid):
        """
        :type obstacleGrid: List[List[int]]
        :rtype: int
        """
        m = len(obstacleGrid)
        n = len(obstacleGrid[0])
        dp = [[0 for _ in range(n)] for _ in range(m)]
        for i in range(m):
            if obstacleGrid[i][0] == 0:  # 遇到障碍物时，直接退出循环，后面默认都是0
                dp[i][0] = 1
            else:
                break
        for j in range(n):
            if obstacleGrid[0][j] == 0:
                dp[0][j] = 1
            else:
                break
        for i in range(1, m):
            for l in range(1,len(obstacleGrid[i])):
                if obstacleGrid[i][l]!= 1:
                    dp[i][l] = dp[i-1][l]+dp[i][l-1]
        return dp[m-1][n-1]
```
1. initialize should not leave 0 when meet obstacle

96.
TBD
343.
TBD
## 代码随想录 Day 34
TBD 
carl1
TBD
carl2

416. Partition Equal Subset Sum
```python
class Solution(object):
    def canPartition(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        # item i, value nums[i], volumn sum
        
        double_target = sum(nums)
        if double_target%2 ==1:
            return False
        else:
            target = double_target//2
        dp=[0]*(target+1)
        for i in range(len(nums)):
            for j in range(target,nums[i]-1,-1):
                dp[j] = max(dp[j-nums[i]]+nums[i],dp[j])

        if dp[target] == target:
            return True
        else:
            return False
```
1. lower bound for inner loop in nums[i]-1, not 0



222. count complete tree nodes
```cpp
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
 *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
 *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
 * };
 */
class Solution {
public:
    int countNodes(TreeNode* root) {
        if (root == NULL) return 0;
        TreeNode* left= root->left;
        TreeNode* right = root->right;
        int l=0,r=0;
        while(left){
            left=left->left;
            l++;
        }
        while(right){
            right = right->right;
            r++;
        }
        
        if (l==r) return (2<<l)-1;
        else  return 1+countNodes(root->left)+countNodes(root->right);
    }
};

112. path sum
```cpp
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
 *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
 *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
 * };
 */
class Solution {
public:
    // recursive 
    bool traversal(TreeNode* cur, int sum){
        if(!cur->left&&! cur->right && ((sum-cur->val)==0)) return true;
        if(!cur->left&&! cur->right && ((sum-cur->val)!=0)) return false;
        bool l= false, r= false;
        if(cur->left)  l=traversal(cur->left, (sum-cur->val));
        if(cur->right)  r=traversal(cur->right, (sum-cur->val));
        return l || r;
    }
    bool hasPathSum(TreeNode* root, int targetSum) {
        if (!root) return false;
        bool res= traversal(root, targetSum);
        return res;
    }
};
```

// time complexity: log(n) *log(n), worst case every level has to recursive 
```

113 path sum 2
```cpp
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
 *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
 *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
 * };
 */
class Solution {
public:
    vector<vector<int>> pathSum(TreeNode* root, int targetSum) {
        
        stack<tuple<TreeNode*, vector<int>, int>> st;
        vector<vector<int>> res;
        if (!root) return res;

        st.push(tuple<TreeNode*, vector<int>, int>(root,{root->val},root->val ));
        while (!st.empty()) {
            auto [node, path, sum] = st.top();
            st.pop();

            if (!node->left && !node->right && sum == targetSum) {
                res.push_back(path);
            }

            if (node->right) {
                path.push_back(node->right->val);
                st.push({node->right, path, sum + node->right->val});
                path.pop_back();
            }

            if (node->left) {
                
                path.push_back(node->left->val);
                st.push({node->left, path, sum + node->left->val});
                path.pop_back();
            }
        }

        return res;

    }
};
```

106. construct binary tree from inorder and postorder
```cpp
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
 *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
 *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
 * };
 */
class Solution {
public:

    TreeNode* buildTree(vector<int>& inorder, vector<int>& postorder) {
        if (postorder.size() == 0) return NULL;

        // 后序遍历数组最后一个元素，就是当前的中间节点
        int rootValue = postorder[postorder.size() - 1];
        TreeNode* root = new TreeNode(rootValue);

        // 叶子节点
        if (postorder.size() == 1) return root;

        // 找到中序遍历的切割点
        int delimiterIndex;
        for (delimiterIndex = 0; delimiterIndex < inorder.size(); delimiterIndex++) {
            if (inorder[delimiterIndex] == rootValue) break;
        }

        // 切割中序数组
        // 左闭右开区间：[0, delimiterIndex)
        vector<int> leftInorder(inorder.begin(), inorder.begin() + delimiterIndex);
        // [delimiterIndex + 1, end)
        vector<int> rightInorder(inorder.begin() + delimiterIndex + 1, inorder.end() );

        // postorder 舍弃末尾元素
        postorder.resize(postorder.size() - 1);

        // 切割后序数组
        // 依然左闭右开，注意这里使用了左中序数组大小作为切割点
        // [0, leftInorder.size)
        vector<int> leftPostorder(postorder.begin(), postorder.begin() + leftInorder.size());
        // [leftInorder.size(), end)
        vector<int> rightPostorder(postorder.begin() + leftInorder.size(), postorder.end());

        root->left = buildTree(leftInorder, leftPostorder);
        root->right = buildTree(rightInorder, rightPostorder);

        return root;
    }
};
```
105. construct binary tree from preorder and inorder 
```cpp
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
 *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
 *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
 * };
 */
class Solution {
public:
    TreeNode* buildTree(vector<int>& preorder, vector<int>& inorder) {
        if (preorder.empty()) return nullptr;

        int rootValue = preorder[0];
        TreeNode* root = new TreeNode(rootValue);

        if (preorder.size() == 1) return root;

        // Find delimiter in inorder
        int delimiterIndex = 0;
        while (inorder[delimiterIndex] != rootValue) {
            delimiterIndex++;
        }

        // Slice inorder
        vector<int> leftInorder(inorder.begin(), inorder.begin() + delimiterIndex);
        vector<int> rightInorder(inorder.begin() + delimiterIndex + 1, inorder.end());

        // Slice preorder
        // Skip first element (the root), and take left subtree size
        vector<int> leftPreorder(preorder.begin() + 1, preorder.begin() + 1 + leftInorder.size());
        vector<int> rightPreorder(preorder.begin() + 1 + leftInorder.size(), preorder.end());

        root->left = buildTree(leftPreorder, leftInorder);
        root->right = buildTree(rightPreorder, rightInorder);

        return root;
    }
};
```

654. maximum binary tree
```cpp
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
 *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
 *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
 * };
 */
class Solution {
public:
    TreeNode* traversal(vector<int>& nums, int left, int right) {
        if (left == right) return nullptr;

        // 分割点下标：maxValueIndex
        int maxValueIndex = left;
        for (int i = left + 1; i < right; ++i) {
            if (nums[i] > nums[maxValueIndex]) maxValueIndex = i;
        }

        TreeNode* root = new TreeNode(nums[maxValueIndex]);

        // 左闭右开：[left, maxValueIndex)
        root->left = traversal(nums, left, maxValueIndex);

        // 左闭右开：[maxValueIndex + 1, right)
        root->right = traversal(nums, maxValueIndex + 1, right);

        return root;
    }
    TreeNode* constructMaximumBinaryTree(vector<int>& nums) {
        return traversal(nums, 0, nums.size());


    }
};
```
332. reconstruct itineray
```cpp
class Solution {
    unordered_map<string, multiset<string>> adj;
    void dfs(const string& u, vector<string>& route) {
        auto &ms = adj[u];
        while (!ms.empty()) {
            // pick the smallest lexicographical flight
            auto it = ms.begin();
            string v = *it;
            ms.erase(it);
            dfs(v, route);
        }
        route.push_back(u);
    }
public:
    vector<string> findItinerary(vector<vector<string>>& tickets) {
        // 1. Build graph
        for (auto &t : tickets) {
            adj[t[0]].insert(t[1]);
        }
        // 2. Eulerian DFS
        vector<string> route;
        dfs("JFK", route);
        // 3. route is in reverse
        reverse(route.begin(), route.end());
        return route;
    }
};

```

use graphs ???
