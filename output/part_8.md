# [剑指 Offer II 021. 删除链表的倒数第 n 个结点](https://leetcode.cn/problems/SLwz0R/)

- 标签：链表、双指针
- 难度：中等

## 题目链接

- [剑指 Offer II 021. 删除链表的倒数第 n 个结点 - 力扣](https://leetcode.cn/problems/SLwz0R/)

## 题目大意

给你一个链表的头节点 `head` 和一个整数 `n`。

要求：删除链表的倒数第 `n` 个节点，并且返回链表的头节点。并且要求使用一次遍历实现。

## 解题思路

常规思路是遍历一遍链表，求出链表长度，再遍历一遍到对应位置，删除该位置上的节点。

如果用一次遍历实现的话，可以使用快慢指针。让快指针先走 n 步，然后快慢指针、慢指针再同时走，每次一步，这样等快指针遍历到链表尾部的时候，慢指针就刚好遍历到了倒数第 n 个节点位置。将该位置上的节点删除即可。

需要注意的是要删除的节点可能包含了头节点。我们可以考虑在遍历之前，新建一个头节点，让其指向原来的头节点。这样，最终如果删除的是头节点，则删除原头节点即可。返回结果的时候，可以直接返回新建头节点的下一位节点。

## 代码

```python
class Solution:
    def removeNthFromEnd(self, head: ListNode, n: int) -> ListNode:
        newHead = ListNode(0, head)
        fast = head
        slow = newHead
        while n:
            fast = fast.next
            n -= 1
        while fast:
            fast = fast.next
            slow = slow.next
        slow.next = slow.next.next
        return newHead.next
```

# [剑指 Offer II 022. 链表中环的入口节点](https://leetcode.cn/problems/c32eOV/)

- 标签：哈希表、链表、双指针
- 难度：中等

## 题目链接

- [剑指 Offer II 022. 链表中环的入口节点 - 力扣](https://leetcode.cn/problems/c32eOV/)

## 题目大意

给定一个链表的头节点 `head`。

要求：判断链表中是否有环，如果有环则返回入环的第一个节点，无环则返回 `None`。

## 解题思路

利用两个指针，一个慢指针每次前进一步，快指针每次前进两步（两步或多步效果是等价的）。如果两个指针在链表头节点以外的某一节点相遇（即相等）了，那么说明链表有环，否则，如果（快指针）到达了某个没有后继指针的节点时，那么说明没环。

如果有环，则再定义一个指针，和慢指针一起每次移动一步，两个指针相遇的位置即为入口节点。

这是因为：假设入环位置为 A，快慢指针在在 B 点相遇，则相遇时慢指针走了 a + b 步，快指针走了 $a + n(b+c) + b$ 步。

$2(a + b) = a + n(b + c) + b$。可以推出：$a = c + (n-1)(b + c)$。

我们可以发现：从相遇点到入环点的距离 $c$ 加上 $n-1$ 圈的环长 $b + c$ 刚好等于从链表头部到入环点的距离。

## 代码

```python
class Solution:
    def detectCycle(self, head: ListNode) -> ListNode:
        fast, slow = head, head
        while True:
            if not fast or not fast.next:
                return None
            fast = fast.next.next
            slow = slow.next
            if fast == slow:
                break

        ans = head
        while ans != slow:
            ans, slow = ans.next, slow.next
        return ans    
```

# [剑指 Offer II 023. 两个链表的第一个重合节点](https://leetcode.cn/problems/3u1WK4/)

- 标签：哈希表、链表、双指针
- 难度：简单

## 题目链接

- [剑指 Offer II 023. 两个链表的第一个重合节点 - 力扣](https://leetcode.cn/problems/3u1WK4/)

## 题目大意

给定 `A`、`B` 两个链表。

要求：判断两个链表是否相交，返回相交的起始点。如果不相交，则返回 `None`。

比如：链表 A 为 [4, 1, 8, 4, 5]，链表 B 为 [5, 0, 1, 8, 4, 5]。则如下图所示，两个链表相交的起始节点为 8，则输出结果为 8。

![](https://assets.leetcode.com/uploads/2018/12/13/160_example_1.png)



## 解题思路

如果两个链表相交，那么从相交位置开始，到结束，必有一段等长且相同的节点。假设链表 A 的长度为 m、链表 B 的长度为 n，他们的相交序列有 k 个，则相交情况可以如下如所示：

![](https://qcdn.itcharge.cn/images/20210401113538.png)

现在问题是如何找到 m-k 或者 n-k 的位置。

考虑将链表 A 的末尾拼接上链表 B，链表 B 的末尾拼接上链表 A。

然后使用两个指针 pA 、PB，分别从链表 A、链表 B 的头节点开始遍历，如果走到共同的节点，则返回该节点。

否则走到两个链表末尾，返回 None。

![](https://qcdn.itcharge.cn/images/20210401114100.png)

## 代码

```python
class Solution:
    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> ListNode:
        if headA == None or headB == None:
            return None
        pA = headA
        pB = headB
        while pA != pB:
            pA = pA.next if pA != None else headB
            pB = pB.next if pB != None else headA
        return pA
```

# [剑指 Offer II 024. 反转链表](https://leetcode.cn/problems/UHnkqh/)

- 标签：递归、链表
- 难度：简单

## 题目链接

- [剑指 Offer II 024. 反转链表 - 力扣](https://leetcode.cn/problems/UHnkqh/)

## 题目大意

**描述**：给定一个单链表的头节点 `head`。

**要求**：将其进行反转，并返回反转后的链表的头节点。

## 解题思路

### 思路 1. 迭代

1. 使用两个指针 `cur` 和 `pre` 进行迭代。`pre` 指向 `cur` 前一个节点位置。初始时，`pre` 指向 `None`，`cur` 指向 `head`。

2. 将 `pre` 和 `cur` 的前后指针进行交换，指针更替顺序为：
   1. 使用 `next` 指针保存当前节点 `cur` 的后一个节点，即 `next = cur.next`；
   2. 断开当前节点 `cur` 的后一节点链接，将 `cur` 的 `next` 指针指向前一节点 `pre`，即 `cur.next = pre`；
   3. `pre` 向前移动一步，移动到 `cur` 位置，即 `pre = cur`；
   4. `cur` 向前移动一步，移动到之前 `next` 指针保存的位置，即 `cur = next`。
3. 继续执行第 2 步中的 1、2、3、4。
4. 最后等到 `cur` 遍历到链表末尾，即 `cur == None`，时，`pre` 所在位置就是反转后链表的头节点，返回新的头节点 `pre`。

使用迭代法反转链表的示意图如下所示：

![](https://qcdn.itcharge.cn/images/20220111133639.png)

### 思路 2. 递归

具体做法如下：

- 首先定义递归函数含义为：将链表反转，并返回反转后的头节点。
- 然后从 `head.next` 的位置开始调用递归函数，即将 `head.next` 为头节点的链表进行反转，并返回该链表的头节点。
- 递归到链表的最后一个节点，将其作为最终的头节点，即为 `new_head`。
- 在每次递归函数返回的过程中，改变 `head` 和 `head.next` 的指向关系。也就是将 `head.next` 的`next` 指针先指向当前节点 `head`，即 `head.next.next = head `。
- 然后让当前节点 `head` 的 `next` 指针指向 `None`，从而实现从链表尾部开始的局部反转。
- 当递归从末尾开始顺着递归栈的退出，从而将整个链表进行反转。
- 最后返回反转后的链表头节点 `new_head`。

使用递归法反转链表的示意图如下所示：

![](https://qcdn.itcharge.cn/images/20220111134246.png)

## 代码

1. 迭代

```python
class Solution:
    def reverseList(self, head: ListNode) -> ListNode:
        pre = None
        cur = head
        while cur != None:
            next = cur.next
            cur.next = pre
            pre = cur
            cur = next
        return pre
```

2. 递归

```python
class Solution:
    def reverseList(self, head: ListNode) -> ListNode:
        if head == None or head.next == None:
            return head
        new_head = self.reverseList(head.next)
        head.next.next = head
        head.next = None
        return new_head
```

## 参考资料

- 【题解】[反转链表 - 反转链表 - 力扣](https://leetcode.cn/problems/reverse-linked-list/solution/fan-zhuan-lian-biao-by-leetcode-solution-d1k2/)
- 【题解】[【反转链表】：双指针，递归，妖魔化的双指针 - 反转链表 - 力扣（LeetCode）](https://leetcode.cn/problems/reverse-linked-list/solution/fan-zhuan-lian-biao-shuang-zhi-zhen-di-gui-yao-mo-/)# [剑指 Offer II 025. 链表中的两数相加](https://leetcode.cn/problems/lMSNwu/)

- 标签：栈、链表、数学
- 难度：中等

## 题目链接

- [剑指 Offer II 025. 链表中的两数相加 - 力扣](https://leetcode.cn/problems/lMSNwu/)

## 题目大意

给定两个非空链表的头节点 `l1` 和 `l2` 来代表两个非负整数。数字最高位位于链表开始位置。每个节点只储存一位数字。除了数字 `0` 之外，这两个链表代表的数字都不会以 `0` 开头。

要求：将这两个数相加会返回一个新的链表。

## 解题思路

链表中最高位位于链表开始位置，最低位位于链表结束位置。这与我们做加法的数位顺序是相反的。为了将链表逆序，从而从低位开始处理数位，我们可以借用两个栈：将链表中所有数字分别压入两个栈中，再依次取出相加。

同时，在相加的时候，还要考虑进位问题。具体步骤如下：

- 将链表 `l1` 中所有节点值压入 `stack1` 栈中，再将链表 `l2` 中所有节点值压入 `stack2` 栈中。
- 使用 `res` 存储新的结果链表，一开始指向 `None`，`carry` 记录进位。
- 如果 `stack1` 或 `stack2` 不为空，或着进位 `carry` 不为 `0`，则：
  - 从 `stack1` 中取出栈顶元素 `num1`，如果 `stack1` 为空，则 `num1 = 0`。
  - 从 `stack2` 中取出栈顶元素 `num2`，如果 `stack2` 为空，则 `num2 = 0`。
  - 计算相加结果，并计算进位。
  - 建立新节点，存储进位后余下的值，并令其指向 `res`。
  - `res` 指向新节点，继续判断。
- 如果 `stack1`、`stack2` 都为空，并且进位 `carry` 为 `0`，则输出 `res`。

## 代码

```python
class Solution:
    def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
        stack1, stack2 = [], []
        while l1:
            stack1.append(l1.val)
            l1 = l1.next
        while l2:
            stack2.append(l2.val)
            l2 = l2.next

        res = None
        carry = 0
        while stack1 or stack2 or carry != 0:
            num1 = stack1.pop() if stack1 else 0
            num2 = stack2.pop() if stack2 else 0
            cur_sum = num1 + num2 + carry
            carry = cur_sum // 10
            cur_sum %= 10
            cur_node = ListNode(cur_sum)
            cur_node.next = res
            res = cur_node
        return res
```

# [剑指 Offer II 026. 重排链表](https://leetcode.cn/problems/LGjMqU/)

- 标签：栈、递归、链表、双指针
- 难度：中等

## 题目链接

- [剑指 Offer II 026. 重排链表 - 力扣](https://leetcode.cn/problems/LGjMqU/)

## 题目大意

给定一个单链表 `L` 的头节点 `head`，单链表 `L` 表示为：$L_0$ -> $L_1$ -> $L_2$ -> ... -> $L_{n-1}$ -> $L_n$。

要求：将单链表 `L` 重新排列为：$L_0$ -> $L_n$ -> $L_1$ -> $L_{n-1}$ -> $L_2$ -> $L_{n-2}$ -> $L_3$ -> $L_{n-3}$ -> ...。

注意：需要将实际节点进行交换。

## 解题思路

链表不能像数组那样直接进行随机访问。所以我们可以先将链表转为线性表。然后直接按照提要要求的排列顺序访问对应数据元素，重新建立链表。

## 代码

```python
class Solution:
    def reorderList(self, head: ListNode) -> None:
        """
        Do not return anything, modify head in-place instead.
        """
        if not head:
            return

        vec = []
        node = head
        while node:
            vec.append(node)
            node = node.next

        left, right = 0, len(vec) - 1
        while left < right:
            vec[left].next = vec[right]
            left += 1
            if left == right:
                break
            vec[right].next = vec[left]
            right -= 1
        vec[left].next = None
```

# [剑指 Offer II 027. 回文链表](https://leetcode.cn/problems/aMhZSa/)

- 标签：栈、递归、链表、双指针
- 难度：简单

## 题目链接

- [剑指 Offer II 027. 回文链表 - 力扣](https://leetcode.cn/problems/aMhZSa/)

## 题目大意

给定一个链表的头节点 `head`。

要求：判断该链表是否为回文链表。

## 解题思路

利用数组，将链表元素依次存入。然后再使用两个指针，一个指向数组开始位置，一个指向数组结束位置，依次判断首尾对应元素是否相等，若都相等，则为回文链表。若不相等，则不是回文链表。

## 代码

```python
class Solution:
    def isPalindrome(self, head: ListNode) -> bool:
        nodes = []
        p1 = head
        while p1 != None:
            nodes.append(p1.val)
            p1 = p1.next
        return nodes == nodes[::-1]
```

# [剑指 Offer II 028. 展平多级双向链表](https://leetcode.cn/problems/Qv1Da2/)

- 标签：深度优先搜索、链表、双向链表
- 难度：中等

## 题目链接

- [剑指 Offer II 028. 展平多级双向链表 - 力扣](https://leetcode.cn/problems/Qv1Da2/)

## 题目大意

给定一个带子链表指针 `child` 的双向链表。

要求：将 `child` 的子链表进行扁平化处理，使所有节点出现在单级双向链表中。

扁平化处理如下：

```
原链表：
1---2---3---4---5---6--NULL
        |
        7---8---9---10--NULL
            |
            11--12--NULL
扁平化之后：
1---2---3---7---8---11---12---9---10---4---5---6--NULL
```

## 解题思路

递归处理多层链表的扁平化。遍历链表，找到 `child` 非空的节点， 将其子链表链接到当前节点的 `next` 位置（自身扁平化处理）。然后继续向后遍历，不断找到 `child` 节点，并进行链接。直到处理到尾部位置。

## 代码

```python
class Solution:
    def dfs(self, node: 'Node'):
        # 找到链表的尾节点或 child 链表不为空的节点
        while node.next and not node.child:
            node = node.next
        tail = None
        if node.child:
            # 如果 child 链表不为空，将 child 链表扁平化
            tail = self.dfs(node.child)

            # 将扁平化的 child 链表链接在该节点之后
            temp = node.next
            node.next = node.child
            node.next.prev = node
            node.child = None
            tail.next = temp
            if temp:
                temp.prev = tail
            # 链接之后，从 child 链表的尾节点继续向后处理链表
            return self.dfs(tail)
        # child 链表为空，则该节点是尾节点，直接返回
        return node

    def flatten(self, head: 'Node') -> 'Node':
        if not head:
            return head
        self.dfs(head)
        return head
```

# [剑指 Offer II 029. 排序的循环链表](https://leetcode.cn/problems/4ueAj6/)

- 标签：链表
- 难度：中等

## 题目链接

- [剑指 Offer II 029. 排序的循环链表 - 力扣](https://leetcode.cn/problems/4ueAj6/)

## 题目大意

给定循环升序链表中的一个节点 `head` 和一个整数 `insertVal`。

要求：将整数 `insertVal` 插入循环升序链表中，并且满足链表仍为循环升序链表。最终返回原先给定的节点。

## 解题思路

- 先判断所给节点 `head` 是否为空，为空直接创建一个值为 `insertVal` 的新节点，并指向自己，返回即可。

- 如果 `head` 不为空，把 `head` 赋值给 `node` ，方便最后返回原节点 `head`。
- 然后遍历 `node`，判断插入值 `insertVal` 与 `node.val` 和 `node.next.val` 的关系，找到插入位置，具体判断如下：
  - 如果新节点值在两个节点值中间， 即 `node.val <= insertVal <= node.next.val`。则说明新节点值在最大值最小值中间，应将新节点插入到当前位置，则应将 `insertVal` 插入到这个位置。
  - 如果新节点值比当前节点值和当前节点下一节点值都大，并且当前节点值比当前节点值的下一节点值大，即 `node.next.val < node.val <= insertVal`，则说明 `insertVal` 比链表最大值都大，应插入最大值后边。
  - 如果新节点值比当前节点值和当前节点下一节点值都小，并且当前节点值比当前节点值的下一节点值大，即 `insertVal < node.next.val < node.val`，则说明 `insertVal` 比链表中最小值都小，应插入最小值前边。
- 找到插入位置后，跳出循环，在插入位置插入值为 `insertVal` 的新节点。

## 代码

```python
class Solution:
    def insert(self, head: 'Node', insertVal: int) -> 'Node':
        if not head:
            node = Node(insertVal)
            node.next = node
            return node

        node = head
        while node.next != head:
            if node.val <= insertVal <= node.next.val:
                break
            elif node.next.val < node.val <= insertVal:
                break
            elif insertVal < node.next.val < node.val:
                break
            else:
                node = node.next

        insert_node = Node(insertVal)
        insert_node.next = node.next
        node.next = insert_node
        return head
```

# [剑指 Offer II 030. 插入、删除和随机访问都是 O(1) 的容器](https://leetcode.cn/problems/FortPu/)

- 标签：设计、数组、哈希表、数学、随机化
- 难度：中等

## 题目链接

- [剑指 Offer II 030. 插入、删除和随机访问都是 O(1) 的容器 - 力扣](https://leetcode.cn/problems/FortPu/)

## 题目大意

设计一个数据结构 ，支持时间复杂度为 $O(1)$ 的以下操作：

- `insert(val)`：当元素 val 不存在时，向集合中插入该项。
- `remove(val)`：元素 val 存在时，从集合中移除该项。
- `getRandom`：随机返回现有集合中的一项。每个元素应该有相同的概率被返回。

## 解题思路

普通动态数组进行访问操作，需要线性时间查找解决。我们可以利用哈希表记录下每个元素的下标，这样在访问时可以做到常数时间内访问元素了。对应的插入、删除、后去随机元素需要做相应的变化。

- 插入操作：将元素直接插入到数组尾部，并用哈希表记录插入元素的下标位置。
- 删除操作：使用哈希表找到待删除元素所在位置，将其与数组末尾位置元素相互交换，更新哈希表中交换后元素的下标值，并将末尾元素删除。
- 获取随机元素：使用` random.choice` 获取。

## 代码

```python
import random

class RandomizedSet:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.dict = dict()
        self.list = list()


    def insert(self, val: int) -> bool:
        """
        Inserts a value to the set. Returns true if the set did not already contain the specified element.
        """
        if val in self.dict:
            return False
        self.dict[val] = len(self.list)
        self.list.append(val)
        return True


    def remove(self, val: int) -> bool:
        """
        Removes a value from the set. Returns true if the set contained the specified element.
        """
        if val in self.dict:
            idx = self.dict[val]
            last = self.list[-1]
            self.list[idx] = last
            self.dict[last] = idx
            self.list.pop()
            self.dict.pop(val)
            return True
        return False


    def getRandom(self) -> int:
        """
        Get a random element from the set.
        """
        return random.choice(self.list)
```

# [剑指 Offer II 031. 最近最少使用缓存](https://leetcode.cn/problems/OrIXps/)

- 标签：设计、哈希表、链表、双向链表
- 难度：中等

## 题目链接

- [剑指 Offer II 031. 最近最少使用缓存 - 力扣](https://leetcode.cn/problems/OrIXps/)

## 题目大意

要求：实现一个 `LRU（最近最少使用）缓存机制`，并且在 `O(1)` 时间复杂度内完成 `get`、`put` 操作。

实现 `LRUCache` 类：

- `LRUCache(int capacity)` 以正整数作为容量 `capacity` 初始化 LRU 缓存。
- `int get(int key)` 如果关键字 `key` 存在于缓存中，则返回关键字的值，否则返回 `-1`。
- `void put(int key, int value)` 如果关键字已经存在，则变更其数据值；如果关键字不存在，则插入该组「关键字-值」。当缓存容量达到上限时，它应该在写入新数据之前删除最久未使用的数据值，从而为新的数据值留出空间。

## 解题思路

LRU（最近最少使用缓存）是一种常用的页面置换算法，选择最近最久未使用的页面予以淘汰。LRU 更新和插入新页面都发生在链表首，删除页面都发生在链表尾。

## 代码

```python
class Node:
    def __init__(self, key=None, val=None, prev=None, next=None):
        self.key = key
        self.val = val
        self.prev = prev
        self.next = next

class LRUCache:

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.hashmap = dict()
        self.head = Node()
        self.tail = Node()
        self.head.next = self.tail
        self.tail.prev = self.head


    def get(self, key: int) -> int:
        if key not in self.hashmap:
            return -1
        node = self.hashmap[key]
        self.move_node(node)
        return node.val


    def put(self, key: int, value: int) -> None:
        if key in self.hashmap:
            node = self.hashmap[key]
            node.val = value
            self.move_node(node)
            return
        if len(self.hashmap) == self.capacity:
            self.hashmap.pop(self.head.next.key)
            self.remove_node(self.head.next)

        node = Node(key=key, val=value)
        self.hashmap[key] = node
        self.add_node(node)

    def remove_node(self, node):
        node.prev.next = node.next
        node.next.prev = node.prev


    def add_node(self, node):
        self.tail.prev.next = node
        node.prev = self.tail.prev
        node.next = self.tail
        self.tail.prev = node


    def move_node(self, node):
        self.remove_node(node)
        self.add_node(node)
```

# [剑指 Offer II 032. 有效的变位词](https://leetcode.cn/problems/dKk3P7/)

- 标签：哈希表、字符串、排序
- 难度：简单

## 题目链接

- [剑指 Offer II 032. 有效的变位词 - 力扣](https://leetcode.cn/problems/dKk3P7/)

## 题目大意

给定两个字符串 `s` 和 `t`。

要求：判断 `t` 和 `s` 是否使用了相同的字符构成（字符出现的种类和数目都相同，字符顺序不完全相同）。

## 解题思路

1. 先判断字符串 `s` 和 `t` 的长度，不一样直接返回 `False`；
2. 如果 `s` 和 `t` 相等，则直接返回 `False`，因为变位词的字符顺序不完全相同；
3. 分别遍历字符串 `s` 和 `t`。先遍历字符串 `s`，用哈希表存储字符串 `s` 中字符出现的频次；
4. 再遍历字符串 `t`，哈希表中减去对应字符的频次，出现频次小于 `0` 则输出 `False`；
5. 如果没出现频次小于 `0`，则输出 `True`。

## 代码

```python
class Solution:
    def isAnagram(self, s: str, t: str) -> bool:
        if len(s) != len(t) or s == t:
            return False
        strDict = dict()
        for ch in s:
            if ch in strDict:
                strDict[ch] += 1
            else:
                strDict[ch] = 1
        for ch in t:
            if ch in strDict:
                strDict[ch] -= 1
                if strDict[ch] < 0:
                    return False
            else:
                return False
        return True
```

# [剑指 Offer II 033. 变位词组](https://leetcode.cn/problems/sfvd7V/)

- 标签：数组、哈希表、字符串、排序
- 难度：中等

## 题目链接

- [剑指 Offer II 033. 变位词组 - 力扣](https://leetcode.cn/problems/sfvd7V/)

## 题目大意

给定一个字符串数组 `strs`。

要求：将包含字母相同的字符串组合在一起，不需要考虑输出顺序。

## 解题思路

使用哈希表记录字母相同的字符串。对每一个字符串进行排序，按照 排序字符串：字母相同的字符串数组 的键值顺序进行存储。最终将哈希表的值转换为对应数组返回结果。

## 代码

```python
class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        str_dict = dict()
        res = []
        for s in strs:
            sort_s = str(sorted(s))
            if sort_s in str_dict:
                str_dict[sort_s] += [s]
            else:
                str_dict[sort_s] = [s]

        for sort_s in str_dict:
            res += [str_dict[sort_s]]
        return res
```

# [剑指 Offer II 034. 外星语言是否排序](https://leetcode.cn/problems/lwyVBB/)

- 标签：数组、哈希表、字符串
- 难度：简单

## 题目链接

- [剑指 Offer II 034. 外星语言是否排序 - 力扣](https://leetcode.cn/problems/lwyVBB/)

## 题目大意

给定一组用外星语书写的单词字符串数组 `words`，以及表示外星字母表的顺序的字符串 `order` 。

要求：判断 `words` 中的单词是否都是按照 `order` 来排序的。如果是，则返回 `True`，否则返回 `False`。

## 解题思路

如果所有单词是按照 `order` 的规则升序排列，则所有单词都符合规则。而判断所有单词是升序排列，只需要两两比较相邻的单词即可。所以我们可以先用哈希表存储所有字母的顺序，然后对所有相邻单词进行两两比较，如果最终是升序排列，则符合要求。具体步骤如下：

- 使用哈希表 `order_map` 存储字母的顺序。
- 遍历单词数组 `words`，比较相邻单词 `word1` 和 `word2` 中所有字母在 `order_map` 中的下标，看是否满足 `word1 <= word2`。
- 如果全部满足，则返回 `True`。如果有不满足的情况，则直接返回 `False`。 

## 代码

```python
class Solution:
    def isAlienSorted(self, words: List[str], order: str) -> bool:
        order_map = dict()
        for i in range(len(order)):
            order_map[order[i]] = i
        for i in range(len(words) - 1):
            word1 = words[i]
            word2 = words[i + 1]

            flag = True

            for j in range(min(len(word1), len(word2))):
                if word1[j] != word2[j]:
                    if order_map[word1[j]] > order_map[word2[j]]:
                        return False
                    else:
                        flag = False
                        break

            if flag and len(word1) > len(word2):
                return False
        return True
```

# [剑指 Offer II 035. 最小时间差](https://leetcode.cn/problems/569nqc/)

- 标签：数组、数学、字符串、排序
- 难度：中等

## 题目链接

- [剑指 Offer II 035. 最小时间差 - 力扣](https://leetcode.cn/problems/569nqc/)

## 题目大意

给定一个 24 小时制形式（小时:分钟 "HH:MM"）的时间列表 `timePoints`。

要求：找出列表中任意两个时间的最小时间差并以分钟数表示。

## 解题思路

- 遍历时间列表 `timePoints`，将每个时间转换为以分钟计算的整数形式，比如时间 `14:20`，将其转换为 `14 * 60 + 20 = 860`，存放到新的时间列表 `times` 中。
- 为了处理最早时间、最晚时间之间的时间间隔，我们将 `times` 中最小时间添加到列表末尾一起进行排序。
- 然后将新的时间列表 `times` 按照升序排列。
- 遍历排好序的事件列表 `times` ，找出相邻两个时间的最小间隔值即可。

## 代码

```python
class Solution:
    def changeTime(self, timePoint: str):
        hours, minutes = timePoint.split(':')
        return int(hours) * 60 + int(minutes)

    def findMinDifference(self, timePoints: List[str]) -> int:
        if not timePoints or len(timePoints) > 24 * 60:
            return 0

        times = sorted(self.changeTime(time) for time in timePoints)
        times.append(times[0] + 24 * 60)
        res = times[-1]
        for i in range(1, len(times)):
            res = min(res, times[i] - times[i - 1])
        return res
```

# [剑指 Offer II 036. 后缀表达式](https://leetcode.cn/problems/8Zf90G/)

- 标签：栈、数组、数学
- 难度：中等

## 题目链接

- [剑指 Offer II 036. 后缀表达式 - 力扣](https://leetcode.cn/problems/8Zf90G/)

## 题目大意

给定一个字符串数组 `tokens`，表示「逆波兰表达式」，求解表达式的值。

## 解题思路

栈的典型应用。遍历字符串数组。遇到操作字符的时候，取出栈顶两个元素，进行运算之后，再将结果入栈。遇到数字，则直接入栈。

## 代码

```python
class Solution:
    def evalRPN(self, tokens: List[str]) -> int:
        stack = []
        for token in tokens:
            if token == '+':
                stack.append(stack.pop() + stack.pop())
            elif token == '-':
                stack.append(-stack.pop() + stack.pop())
            elif token == '*':
                stack.append(stack.pop() * stack.pop())
            elif token == '/':
                stack.append(int(1 / stack.pop() * stack.pop()))
            else:
                stack.append(int(token))
        return stack.pop()
```

# [剑指 Offer II 037. 小行星碰撞](https://leetcode.cn/problems/XagZNi/)

- 标签：栈、数组
- 难度：中等

## 题目链接

- [剑指 Offer II 037. 小行星碰撞 - 力扣](https://leetcode.cn/problems/XagZNi/)

## 题目大意

给定一个整数数组 `asteroids`，表示在同一行的小行星。

数组中的每一个元素，其绝对值表示小行星的大小，正负表示小行星的移动方向（正表示向右移动，负表示向左移动）。每一颗小行星以相同的速度移动。小行星按照下面的规则发生碰撞。

-  碰撞规则：两个行星相互碰撞，较小的行星会爆炸。如果两颗行星大小相同，则两颗行星都会爆炸。两颗移动方向相同的行星，永远不会发生碰撞。

要求：找出碰撞后剩下的所有小行星，将答案存入数组并返回。

## 解题思路

用栈模拟小行星碰撞，具体步骤如下：

- 遍历数组 `asteroids`。
- 如果栈为空或者当前元素 `asteroid` 为正数，将其压入栈。
- 如果当前栈不为空并且当前元素 `asteroid` 为负数：
  - 与栈中元素发生碰撞，判断当前元素和栈顶元素的大小和方向，如果栈顶元素为正数，并且当前元素的绝对值大于栈顶元素，则将栈顶元素弹出，并继续与栈中元素发生碰撞。
  - 碰撞完之后，如果栈为空并且栈顶元素为负数，则将当前元素 `asteroid` 压入栈，表示碰撞完剩下了 `asteroid`。
  - 如果栈顶元素恰好与当前元素值大小相等、方向相反，则弹出栈顶元素，表示碰撞完两者都爆炸了。
- 最后返回栈作为答案。

## 代码

```python
class Solution:
    def asteroidCollision(self, asteroids: List[int]) -> List[int]:
        stack = []
        for asteroid in asteroids:
            if not stack or asteroid > 0:
                stack.append(asteroid)
            else:
                while stack and 0 < stack[-1] < -asteroid:
                    stack.pop()
                if not stack or stack[-1] < 0:
                    stack.append(asteroid)
                elif stack[-1] == -asteroid:
                    stack.pop()

        return stack
```

# [剑指 Offer II 038. 每日温度](https://leetcode.cn/problems/iIQa4I/)

- 标签：栈、数组、单调栈
- 难度：中等

## 题目链接

- [剑指 Offer II 038. 每日温度 - 力扣](https://leetcode.cn/problems/iIQa4I/)

## 题目大意

给定一个列表 `temperatures`，每一个位置对应每天的气温。要求输出一个列表，列表上每个位置代表如果要观测到更高的气温，至少需要等待的天数。如果之后的气温不再升高，则用 `0` 来代替。

## 解题思路

题目的意思实际上就是给定一个数组，每个位置上有整数值。对于每个位置，在该位置后侧找到第一个比当前值更高的值。求该点与该位置的距离，将所有距离保存为数组返回结果。

很简单的思路是对于每个温度值，向后依次进行搜索，找到比当前温度更高的值。

更好的方式使用「递减栈」。栈中保存元素的下标。

首先，将答案数组全部赋值为 0。然后遍历数组每个位置元素。

- 如果栈为空，则将当前元素的下标入栈。
- 如果栈不为空，且当前数字大于栈顶元素对应数字，则栈顶元素出栈，并计算下标差。
    - 此时当前元素就是栈顶元素的下一个更高值，将其下标差存入答案数组中保存起来，判断栈顶元素。
- 直到当前数字小于或等于栈顶元素，则停止出栈，将当前元素下标入栈。

## 代码

```python
class Solution:
    def dailyTemperatures(self, temperatures: List[int]) -> List[int]:
        n = len(temperatures)
        stack = []
        ans = [0 for _ in range(n)]
        for i in range(n):
            while stack and temperatures[i] > temperatures[stack[-1]]:
                index = stack.pop()
                ans[index] = (i - index)
            stack.append(i)
        return ans
```

# [剑指 Offer II 039. 直方图最大矩形面积](https://leetcode.cn/problems/0ynMMM/)

- 标签：栈、数组、单调栈
- 难度：困难

## 题目链接

- [剑指 Offer II 039. 直方图最大矩形面积 - 力扣](https://leetcode.cn/problems/0ynMMM/)

## 题目大意

给定一个非负整数数组 `heights` ，`heights[i]` 用来表示柱状图中各个柱子的高度。每个柱子彼此相邻，且宽度为 1 。

要求：计算出在该柱状图中，能够勾勒出来的矩形的最大面积。

## 解题思路

思路一：枚举「宽度」。一重循环枚举所有柱子，第二重循环遍历柱子右侧的柱子，所得的宽度就是两根柱子形成区间的宽度，高度就是这段区间中的最小高度。然后计算出对应面积，记录并更新最大面积。这样下来，时间复杂度为 $O(n^2)$。

思路二：枚举「高度」。一重循环枚举所有柱子，以柱子高度为当前矩形高度，然后向两侧延伸，遇到小于当前矩形高度的情况就停止。然后计算当前矩形面积，记录并更新最大面积。这样下来，时间复杂度也是 $O(n^2)$。

思路三：利用「单调栈」减少两侧延伸的复杂度。

- 枚举所有柱子。
- 如果当前柱子高度较大，大于等于栈顶柱体的高度，则直接将当前柱体入栈。
- 如果当前柱体高度较小，小于栈顶柱体的高度，则一直出栈，直到当前柱体大于等于栈顶柱体高度。
  - 出栈后，说明当前柱体是出栈柱体向右找到的第一个小于当前柱体高度的柱体，那么就可以向右将宽度扩展到当前柱体。
  - 出栈后，说明新的栈顶柱体是出栈柱体向左找到的第一个小于新的栈顶柱体高度的柱体，那么就可以向左将宽度扩展到新的栈顶柱体。
  - 以新的栈顶柱体为左边界，当前柱体为右边界，以出栈柱体为高度。计算矩形面积，然后记录并更新最大面积。

## 代码

```python
class Solution:
    def largestRectangleArea(self, heights: List[int]) -> int:
        heights.append(0)
        ans = 0
        stack = []
        for i in range(len(heights)):
            while stack and heights[stack[-1]] >= heights[i]:
                cur = stack.pop(-1)
                left = stack[-1] + 1 if stack else 0
                right = i - 1
                ans = max(ans, (right - left + 1) * heights[cur])
            stack.append(i)

        return ans
```

# [剑指 Offer II 041. 滑动窗口的平均值](https://leetcode.cn/problems/qIsx9U/)

- 标签：设计、队列、数组、数据流
- 难度：简单

## 题目链接

- [剑指 Offer II 041. 滑动窗口的平均值 - 力扣](https://leetcode.cn/problems/qIsx9U/)

## 题目大意

**描述**：给定一个整数数据流和一个窗口大小 `size`。

**要求**：根据滑动窗口的大小，计算滑动窗口里所有数字的平均值。要求实现 `MovingAverage` 类：

- `MovingAverage(int size)`：用窗口大小 `size` 初始化对象。
- `double next(int val)`：成员函数 `next` 每次调用的时候都会往滑动窗口增加一个整数，请计算并返回数据流中最后 `size` 个值的移动平均值，即滑动窗口里所有数字的平均值。

**说明**：

- $1 \le size \le 1000$。
- $-10^5 \le val \le 10^5$。
- 最多调用 `next` 方法 $10^4$ 次。

**示例**：

- 示例 1：

```python
输入：
inputs = ["MovingAverage", "next", "next", "next", "next"]
inputs = [[3], [1], [10], [3], [5]]
输出：
[null, 1.0, 5.5, 4.66667, 6.0]

解释：
MovingAverage movingAverage = new MovingAverage(3);
movingAverage.next(1); // 返回 1.0 = 1 / 1
movingAverage.next(10); // 返回 5.5 = (1 + 10) / 2
movingAverage.next(3); // 返回 4.66667 = (1 + 10 + 3) / 3
movingAverage.next(5); // 返回 6.0 = (10 + 3 + 5) / 3
```

## 解题思路

### 思路 1：队列

1. 使用队列保存滑动窗口的元素，并记录对应窗口大小和元素和。
2. 当队列长度小于窗口大小的时候，直接向队列中添加元素，并记录当前窗口中的元素和。
3. 当队列长度等于窗口大小的时候，先将队列头部元素弹出，再添加元素，并记录当前窗口中的元素和。
4. 然后根据元素和和队列中元素个数计算出平均值。

### 思路 1：代码

```python
class MovingAverage:

    def __init__(self, size: int):
        """
        Initialize your data structure here.
        """
        self.queue = []
        self.size = size
        self.sum = 0


    def next(self, val: int) -> float:
        if len(self.queue) < self.size:
            self.queue.append(val)
        else:
            if self.queue:
                self.sum -= self.queue[0]
                self.queue.pop(0)
            self.queue.append(val)
        self.sum += val
        return self.sum / len(self.queue)
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(1)$。初始化方法和每次调用 `next` 方法的时间复杂度都是 $O(1)$。
- **空间复杂度**：$O(size)$。其中 $size$ 就是给定的滑动窗口的大小。# [剑指 Offer II 042. 最近请求次数](https://leetcode.cn/problems/H8086Q/)

- 标签：设计、队列、数据流
- 难度：简单

## 题目链接

- [剑指 Offer II 042. 最近请求次数 - 力扣](https://leetcode.cn/problems/H8086Q/)

## 题目大意

要求：实现一个用来计算特定时间范围内的最近请求的 `RecentCounter` 类：

- `RecentCounter()` 初始化计数器，请求数为 0 。
- `int ping(int t)` 在时间 `t` 时添加一个新请求，其中 `t` 表示以毫秒为单位的某个时间，并返回在 `[t-3000, t]` 内发生的请求数。

## 解题思路

使用一个队列，用于存储 `[t - 3000, t]` 范围内的请求。

获取请求数时，将队首所有小于 `t - 3000` 时间的请求将其从队列中移除，然后返回队列的长度即可。

## 代码

```python
class RecentCounter:

    def __init__(self):
        self.queue = []


    def ping(self, t: int) -> int:
        self.queue.append(t)
        while self.queue[0] < t - 3000:
            self.queue.pop(0)
        return len(self.queue)
```

# [剑指 Offer II 043. 往完全二叉树添加节点](https://leetcode.cn/problems/NaqhDT/)

- 标签：树、广度优先搜索、设计、二叉树
- 难度：中等

## 题目链接

- [剑指 Offer II 043. 往完全二叉树添加节点 - 力扣](https://leetcode.cn/problems/NaqhDT/)

## 题目大意

要求：设计一个用完全二叉树初始化的数据结构 `CBTInserter`，并支持以下几种操作：

- `CBTInserter(TreeNode root)` 使用根节点为 `root` 的给定树初始化该数据结构；
- `CBTInserter.insert(int v)`  向树中插入一个新节点，节点类型为 `TreeNode`，值为 `v`。使树保持完全二叉树的状态，并返回插入的新节点的父节点的值；
- `CBTInserter.get_root()` 返回树的根节点。

## 解题思路

使用数组标记完全二叉树中节点的序号，初始化数组为 `[None]`。完全二叉树中节点的序号从 `1` 开始，对于序号为 `k` 的节点，其左子节点序号为 `2k`，右子节点的序号为 `2k + 1`，其父节点的序号为 `k // 2`。

然后在初始化和插入节点的同时，按顺序向数组中插入节点。

## 代码

```python
class CBTInserter:

    def __init__(self, root: TreeNode):
        self.queue = [root]
        self.nodelist = [None]

        while self.queue:
            node = self.queue.pop(0)
            self.nodelist.append(node)
            if node.left:
                self.queue.append(node.left)
            if node.right:
                self.queue.append(node.right)


    def insert(self, v: int) -> int:
        self.nodelist.append(TreeNode(v))
        index = len(self.nodelist) - 1
        father = self.nodelist[index // 2]
        if index % 2 == 0:
            father.left = self.nodelist[-1]
        else:
            father.right = self.nodelist[-1]
        return father.val


    def get_root(self) -> TreeNode:
        return self.nodelist[1]
```

# [剑指 Offer II 044. 二叉树每层的最大值](https://leetcode.cn/problems/hPov7L/)

- 标签：树、深度优先搜索、广度优先搜索、二叉树
- 难度：中等

## 题目链接

- [剑指 Offer II 044. 二叉树每层的最大值 - 力扣](https://leetcode.cn/problems/hPov7L/)

## 题目大意

给定一棵二叉树的根节点 `root`。

要求：找出二叉树中每一层的最大值。

## 解题思路

利用队列进行层序遍历，并记录下每一层的最大值，将其存入答案数组中。

## 代码

```python
class Solution:
    def largestValues(self, root: TreeNode) -> List[int]:
        queue = []
        res = []
        if root:
            queue.append(root)
        while queue:
            max_level = float('-inf')
            size_level = len(queue)
            for i in range(size_level):
                node = queue.pop(0)
                max_level = max(max_level, node.val)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            res.append(max_level)
        return res
```

# [剑指 Offer II 045. 二叉树最底层最左边的值](https://leetcode.cn/problems/LwUNpT/)

- 标签：树、深度优先搜索、广度优先搜索、二叉树
- 难度：中等

## 题目链接

- [剑指 Offer II 045. 二叉树最底层最左边的值 - 力扣](https://leetcode.cn/problems/LwUNpT/)

## 题目大意

给定一个二叉树的根节点 `root`。

要求：找出该二叉树 「最底层」的「最左边」节点的值。

## 解题思路

这个问题拆开来看，一是如何找到「最底层」，而是在「最底层」如何找到最左边的节点。

通过层序遍历，我们可以直接确定最底层节点。而「最底层」的「最左边」节点可以改变层序遍历的左右节点访问顺序。

每层元素先访问右节点，在访问左节点，则最后一个遍历的元素就是「最底层」的「最左边」节点，即左下角的节点，返回该点对应的值即可。

## 代码

```python
import collections

class Solution:
    def findBottomLeftValue(self, root: TreeNode) -> int:
        if not root:
            return -1
        queue = collections.deque()
        queue.append(root)
        while queue:
            cur = queue.popleft()
            if cur.right:
                queue.append(cur.right)
            if cur.left:
                queue.append(cur.left)
        return cur.val
```

# [剑指 Offer II 046. 二叉树的右侧视图](https://leetcode.cn/problems/WNC0Lk/)

- 标签：树、深度优先搜索、广度优先搜索、二叉树
- 难度：中等

## 题目链接

- [剑指 Offer II 046. 二叉树的右侧视图 - 力扣](https://leetcode.cn/problems/WNC0Lk/)

## 题目大意

给定一棵二叉树的根节点 `root`。

要求：按照从顶部到底部的顺序，返回从右侧能看到的节点值。

## 解题思路

二叉树的层次遍历，不过遍历每层节点的时候，只需要将最后一个节点加入结果数组即可。

## 代码

```python
class Solution:
    def rightSideView(self, root: TreeNode) -> List[int]:
        if not root:
            return []
        queue = [root]
        order = []
        while queue:
            level = []
            size = len(queue)
            for i in range(size):
                curr = queue.pop(0)
                level.append(curr.val)
                if curr.left:
                    queue.append(curr.left)
                if curr.right:
                    queue.append(curr.right)
            if i == size - 1:
                order.append(curr.val)
        return order
```

# [剑指 Offer II 047. 二叉树剪枝](https://leetcode.cn/problems/pOCWxh/)

- 标签：树、深度优先搜索、二叉树
- 难度：中等

## 题目链接

- [剑指 Offer II 047. 二叉树剪枝 - 力扣](https://leetcode.cn/problems/pOCWxh/)

## 题目大意

给定一棵二叉树的根节点 `root`，树的每个节点值要么是 `0`，要么是 `1`。

要求：剪除该二叉树中所有节点值为 `0` 的子树。

- 节点 `node` 的子树为： `node` 本身，以及所有 `node` 的后代。

## 解题思路

定义辅助方法 `containsOnlyZero(root)` 递归判断以 `root` 为根的子树中是否只包含 `0`。如果子树中只包含 `0`，则返回 `True`。如果子树中含有 `1`，则返回 `False`。当 `root` 为空时，也返回 `True`。

然后递归遍历二叉树，判断当前节点 `root` 是否只包含 `0`。如果只包含 `0`，则将其置空，返回 `None`。否则递归遍历左右子树，并设置对应的左右指针。

最后返回根节点 `root`。

## 代码

```python
class Solution:
    def containsOnlyZero(self, root: TreeNode):
        if not root:
            return True
        if root.val == 1:
            return False
        return self.containsOnlyZero(root.left) and self.containsOnlyZero(root.right)

    def pruneTree(self, root: TreeNode) -> TreeNode:
        if not root:
            return root
        if self.containsOnlyZero(root):
            return None

        root.left = self.pruneTree(root.left)
        root.right = self.pruneTree(root.right)
        return root
```

# [剑指 Offer II 048. 序列化与反序列化二叉树](https://leetcode.cn/problems/h54YBf/)

- 标签：树、深度优先搜索、广度优先搜索、设计、字符串、二叉树
- 难度：困难

## 题目链接

- [剑指 Offer II 048. 序列化与反序列化二叉树 - 力扣](https://leetcode.cn/problems/h54YBf/)

## 题目大意

要求：设计一个算法，来实现二叉树的序列化与反序列化。

## 解题思路

### 1. 序列化：将二叉树转为字符串数据表示

按照前序递归遍历二叉树，并将根节点跟左右子树的值链接起来（中间用 `,` 隔开）。

注意：如果遇到空节点，则标记为 'None'，这样在反序列化时才能唯一确定一棵二叉树。

### 2. 反序列化：将字符串数据转为二叉树结构

先将字符串按 `,` 分割成数组。然后递归处理每一个元素。

- 从数组左侧取出一个元素。
    - 如果当前元素为 'None'，则返回 None。
    - 如果当前元素不为空，则新建一个二叉树节点作为根节点，保存值为当前元素值。并递归遍历左右子树，不断重复从数组中取出元素，进行判断。
    - 最后返回当前根节点。

## 代码

```python
class Codec:

    def serialize(self, root):
        """Encodes a tree to a single string.
        
        :type root: TreeNode
        :rtype: str
        """
        if not root:
            return 'None'
        return str(root.val) + ',' + str(self.serialize(root.left)) + ',' + str(self.serialize(root.right))

        

    def deserialize(self, data):
        """Decodes your encoded data to tree.
        
        :type data: str
        :rtype: TreeNode
        """

        def dfs(datalist):
            val = datalist.pop(0)
            if val == 'None':
                return None
            root = TreeNode(int(val))
            root.left = dfs(datalist)
            root.right = dfs(datalist)
            return root

        datalist = data.split(',')
        return dfs(datalist)
```

# [剑指 Offer II 049. 从根节点到叶节点的路径数字之和](https://leetcode.cn/problems/3Etpl5/)

- 标签：树、深度优先搜索、二叉树
- 难度：中等

## 题目链接

- [剑指 Offer II 049. 从根节点到叶节点的路径数字之和 - 力扣](https://leetcode.cn/problems/3Etpl5/)

## 题目大意

给定一个二叉树的根节点 `root`，树中每个节点都存放有一个 `0` 到 `9` 之间的数字。每条从根节点到叶节点的路径都代表一个数字。例如，从根节点到叶节点的路径是 `1` -> `2` -> `3`，表示数字 `123`。

要求：计算从根节点到叶节点生成的所有数字的和。

## 解题思路

使用深度优先搜索，记录下路径上所有节点构成的数字，使用 `pretotal` 保存下当前路径上构成的数字。如果遇到叶节点直接返回当前数字，否则递归遍历左右子树，并累加对应结果。

## 代码

```python
class Solution:
    def dfs(self, root, pretotal):
        if not root:
            return 0
        total = pretotal * 10 + root.val
        if not root.left and not root.right:
            return total
        return self.dfs(root.left, total) + self.dfs(root.right, total)

    def sumNumbers(self, root: TreeNode) -> int:
        return self.dfs(root, 0)
```

# [剑指 Offer II 050. 向下的路径节点之和](https://leetcode.cn/problems/6eUYwP/)

- 标签：树、深度优先搜索、二叉树
- 难度：中等

## 题目链接

- [剑指 Offer II 050. 向下的路径节点之和 - 力扣](https://leetcode.cn/problems/6eUYwP/)

## 题目大意



## 解题思路



## 代码

```python
class Solution:
    prefixsum_count = dict()
    def dfs(self, root, prefixsum_count, target_sum, cur_sum):
        if not root:
            return 0
        res = 0
        cur_sum += root.val
        res += prefixsum_count.get(cur_sum - target_sum, 0)
        prefixsum_count[cur_sum] = prefixsum_count.get(cur_sum, 0) + 1

        res += self.dfs(root.left, prefixsum_count, target_sum, cur_sum)
        res += self.dfs(root.right, prefixsum_count, target_sum, cur_sum)

        prefixsum_count[cur_sum] -= 1
        return res

    def pathSum(self, root: TreeNode, targetSum: int) -> int:
        if not root:
            return 0
        prefixsum_count = dict()
        prefixsum_count[0] = 1
        return self.dfs(root, prefixsum_count, targetSum, 0)
```

# [剑指 Offer II 051. 节点之和最大的路径](https://leetcode.cn/problems/jC7MId/)

- 标签：树、深度优先搜索、动态规划、二叉树
- 难度：困难

## 题目链接

- [剑指 Offer II 051. 节点之和最大的路径 - 力扣](https://leetcode.cn/problems/jC7MId/)

## 题目大意

给定一个二叉树的根节点 `root`。

要求：返回其最大路径和。

- 路径：从树中的任意节点出发，沿父节点——子节点连接，到达任意节点的序列。同一个节点在一条路径序列中至多出现一次。该路径至少包含一个节点，且不一定经过根节点。
- 路径和：路径中各节点值的总和。

## 解题思路

深度优先搜索遍历二叉树。递归的同时，维护一个最大路径和变量。定义函数 `dfs(self, root)` 计算二叉树中以该节点为根节点，并且经过该节点的最大贡献值。

计算的结果可能的情况有 2 种：

- 经过空节点的最大贡献值等于 `0`。

- 经过非空节点的最大贡献值等于 当前节点值 + 左右子节点的最大贡献值中较大的一个。

在递归时，我们先计算左右子节点的最大贡献值，再更新维护当前最大路径和变量。

最终 `max_sum` 即为答案。

## 代码

```python
class Solution:
    def __init__(self):
        self.max_sum = float('-inf')

    def dfs(self, root):
        if not root:
            return 0
        left_max = max(self.dfs(root.left), 0)
        right_max = max(self.dfs(root.right), 0)
        self.max_sum = max(self.max_sum, root.val + left_max + right_max)
        return root.val + max(left_max, right_max)

    def maxPathSum(self, root: TreeNode) -> int:
        self.dfs(root)
        return self.max_sum
```

# [剑指 Offer II 052. 展平二叉搜索树](https://leetcode.cn/problems/NYBBNL/)

- 标签：栈、树、深度优先搜索、二叉搜索树、二叉树
- 难度：简单

## 题目链接

- [剑指 Offer II 052. 展平二叉搜索树 - 力扣](https://leetcode.cn/problems/NYBBNL/)

## 题目大意

给定一棵二叉搜索树的根节点 `root`。

要求：按中序遍历顺序将其重新排列为一棵递增顺序搜索树，使树中最左边的节点成为树的根节点，并且每个节点没有左子节点，只有一个右子节点。

## 解题思路

可以分为两步：

1. 中序遍历二叉搜索树，将节点先存储到列表中。
2. 将列表中的节点构造成一棵递增顺序搜索树。

中序遍历直接按照 `左 -> 根 -> 右` 的顺序递归遍历，然后将遍历的节点存储到 `res` 中。

构造递增顺序搜索树，则用 `head` 保存头节点位置。遍历列表中的每个节点，将其左右指针先置空，再将其连接在上一个节点的右子节点上。

最后返回 `head.right` 即可。

## 代码

```python
class Solution:
    def inOrder(self, root, res):
        if not root:
            return
        self.inOrder(root.left, res)
        res.append(root)
        self.inOrder(root.right, res)

    def increasingBST(self, root: TreeNode) -> TreeNode:
        res = []
        self.inOrder(root, res)

        if not res:
            return
        head = TreeNode(-1)
        cur = head
        for node in res:
            node.left = node.right = None
            cur.right = node
            cur = cur.right
        return head.right
```

# [剑指 Offer II 053. 二叉搜索树中的中序后继](https://leetcode.cn/problems/P5rCT8/)

- 标签：树、深度优先搜索、二叉搜索树、二叉树
- 难度：中等

## 题目链接

- [剑指 Offer II 053. 二叉搜索树中的中序后继 - 力扣](https://leetcode.cn/problems/P5rCT8/)

## 题目大意

给定一棵二叉搜索树的根节点 `root` 和其中一个节点 `p`。

要求：找到该节点在树中的中序后继，即按照中序遍历的顺序节点 `p` 的下一个节点。

## 解题思路

递归遍历，具体步骤如下：

- 如果 `root.val` 小于等于 `p.val`，则直接从 `root` 的右子树递归查找比 `p.val` 大的节点，从而找到中序后继。
- 如果 `root.val` 大于 `p.val`，则 `root` 有可能是中序后继，也有可能是 `root` 的左子树。则从 `root` 的左子树递归查找更接近（更小的）。如果查找的值为 `None`，则当前 `root` 就是中序后继，否则继续递归查找，从而找到中序后继。

## 代码 

```python
class Solution:
    def inorderSuccessor(self, root: 'TreeNode', p: 'TreeNode') -> 'TreeNode':
        if not p or not root:
            return None

        if root.val <= p.val:
            node = self.inorderSuccessor(root.right, p)
        else:
            node = self.inorderSuccessor(root.left, p)
            if not node:
                node = root
        return node
```

# [剑指 Offer II 054. 所有大于等于节点的值之和](https://leetcode.cn/problems/w6cpku/)

- 标签：树、深度优先搜索、二叉搜索树、二叉树
- 难度：中等

## 题目链接

- [剑指 Offer II 054. 所有大于等于节点的值之和 - 力扣](https://leetcode.cn/problems/w6cpku/)

## 题目大意

给定一棵二叉搜索树（BST）的根节点 `root`，且二叉搜索树的节点值各不相同。要求将其转化为「累加树」，使其每个节点 `node` 的新值等于原树中大于或等于 `node.val` 的值之和。

二叉搜索树的定义：

- 若左子树不为空，则左子树上所有节点值均小于它的根节点值；
- 若右子树不为空，则右子树上所有节点值均大于它的根节点值；
- 任意节点的左、右子树也分别为二叉搜索树。

## 解题思路

题目要求将每个节点的值修改为原来的节点值加上大于它的节点值之和。已知二叉搜索树的中序遍历可以得到一个升序数组。

题目就可以变为：修改升序数组中每个节点值为末尾元素累加和。由于末尾元素累加和的求和过程和遍历顺序相反，所以我们可以考虑换种思路。

二叉搜索树的中序遍历顺序为：左 -> 根 -> 右，从而可以得到一个升序数组，那么我们将左右反着遍历，即顺序为：右 -> 根 -> 左，就可以得到一个降序数组，这样就可以在遍历的同时求前缀和。

当然我们在计算前缀和的时候，需要用到前一个节点的值，所以需要用变量 `pre` 存储前一节点的值。

## 代码

```python
class Solution:
    pre = 0

    def createBinaryTree(self, root: TreeNode):
        if not root:
            return
        self.createBinaryTree(root.right)
        root.val += self.pre
        self.pre = root.val
        self.createBinaryTree(root.left)

    def convertBST(self, root: TreeNode) -> TreeNode:
        self.pre = 0
        self.createBinaryTree(root)
        return root
```

# [剑指 Offer II 055. 二叉搜索树迭代器](https://leetcode.cn/problems/kTOapQ/)

- 标签：栈、树、设计、二叉搜索树、二叉树、迭代器
- 难度：中等

## 题目链接

- [剑指 Offer II 055. 二叉搜索树迭代器 - 力扣](https://leetcode.cn/problems/kTOapQ/)

## 题目大意

要求：实现一个二叉搜索树的迭代器 `BSTIterator`。表示一个按中序遍历二叉搜索树（BST）的迭代器：

- `def __init__(self, root: TreeNode):`：初始化 `BSTIterator` 类的一个对象，会给出二叉搜索树的根节点。
- `def hasNext(self) -> bool:`：如果向右指针遍历存在数字，则返回 `True`，否则返回 `False`。
- `def next(self) -> int:`：将指针向右移动，返回指针处的数字。

## 解题思路

中序遍历的顺序是：左、根、右。我们使用一个栈来保存节点，以便于迭代的时候取出对应节点。

- 初始的遍历当前节点的左子树，将其路径上的节点存储到栈中。
- 调用 next 方法的时候，从栈顶取出节点，因为之前已经将路径上的左子树全部存入了栈中，所以此时该节点的左子树为空，这时候取出节点右子树，再将右子树的左子树进行递归遍历，并将其路径上的节点存储到栈中。
- 调用 hasNext 的方法的时候，直接判断栈中是否有值即可。

## 代码

```python
class BSTIterator:

    def __init__(self, root: TreeNode):
        self.stack = []
        self.in_order(root)


    def in_order(self, node):
        while node:
            self.stack.append(node)
            node = node.left


    def next(self) -> int:
        node = self.stack.pop()
        if node.right:
            self.in_order(node.right)
        return node.val


    def hasNext(self) -> bool:
        return len(self.stack) != 0
```

# [剑指 Offer II 056. 二叉搜索树中两个节点之和](https://leetcode.cn/problems/opLdQZ/)

- 标签：树、深度优先搜索、广度优先搜索、二叉搜索树、哈希表、双指针、二叉树
- 难度：简单

## 题目链接

- [剑指 Offer II 056. 二叉搜索树中两个节点之和 - 力扣](https://leetcode.cn/problems/opLdQZ/)

## 题目大意

给定一个二叉搜索树的根节点 `root` 和一个整数 `k`。

要求：判断该二叉搜索树是否存在两个节点值的和等于 `k`。如果存在，则返回 `True`，不存在则返回 `False`。

## 解题思路

二叉搜索树中序遍历的结果是从小到大排序，所以我们可以先对二叉搜索树进行中序遍历，将中序遍历结果存储到列表中。再使用左右指针查找节点值和为 `k` 的两个节点。

## 代码

```python
class Solution:
    def inOrder(self, root, nums):
        if not root:
            return
        self.inOrder(root.left, nums)
        nums.append(root.val)
        self.inOrder(root.right, nums)

    def findTarget(self, root: TreeNode, k: int) -> bool:
        nums = []
        self.inOrder(root, nums)
        left, right = 0, len(nums) - 1
        while left < right:
            sum = nums[left] + nums[right]
            if sum == k:
                return True
            elif sum < k:
                left += 1
            else:
                right -= 1
        return False
```

# [剑指 Offer II 057. 值和下标之差都在给定的范围内](https://leetcode.cn/problems/7WqeDu/)

- 标签：数组、桶排序、有序集合、排序、滑动窗口
- 难度：中等

## 题目链接

- [剑指 Offer II 057. 值和下标之差都在给定的范围内 - 力扣](https://leetcode.cn/problems/7WqeDu/)

## 题目大意

给定一个整数数组 `nums`，以及两个整数 `k`、`t`。判断数组中是否存在两个不同下标的 `i` 和 `j`，其对应元素满足 `abs(nums[i] - nums[j]) <= t`，同时满足 `abs(i - j) <= k`。如果满足条件则返回 `True`，不满足条件返回 `False`。

## 解题思路

对于第 `i` 个元素 `nums[i]`，需要查找的区间为 `[i - t, i + t]`。可以利用桶排序的思想。

桶的大小设置为 `t + 1`。我们将元素按照大小依次放入不同的桶中。

遍历数组 `nums` 中的元素，对于元素 `nums[i]` ：

- 如果 `nums[i]` 放入桶之前桶里已经有元素了，那么这两个元素必然满足 `abs(nums[i] - nums[j]) <= t`，
- 如果之前桶里没有元素，那么就将 `nums[i]` 放入对应桶中。
- 然后再判断左右桶的左右两侧桶中是否有元素满足 `abs(nums[i] - nums[j]) <= t`。
- 然后将 `nums[i - k]` 之前的桶清空，因为这些桶中的元素与 `nums[i]` 已经不满足 `abs(i - j) <= k` 了。

最后上述满足条件的情况就返回 `True`，最终遍历完仍不满足条件就返回 `False`。

## 代码

```python
class Solution:
    def containsNearbyAlmostDuplicate(self, nums: List[int], k: int, t: int) -> bool:
        bucket_dict = dict()
        for i in range(len(nums)):
            # 将 nums[i] 划分到大小为 t + 1 的不同桶中
            num = nums[i] // (t + 1)

            # 桶中已经有元素了
            if num in bucket_dict:
                return True

            # 把 nums[i] 放入桶中
            bucket_dict[num] = nums[i]

            # 判断左侧桶是否满足条件
            if (num - 1) in bucket_dict and abs(bucket_dict[num - 1] - nums[i]) <= t:
                return True
            # 判断右侧桶是否满足条件
            if (num + 1) in bucket_dict and abs(bucket_dict[num + 1] - nums[i]) <= t:
                return True
            # 将 i-k 之前的旧桶清除，因为之前的桶已经不满足条件了
            if i >= k:
                bucket_dict.pop(nums[i - k] // (t + 1))

        return False
```

# [剑指 Offer II 059. 数据流的第 K 大数值](https://leetcode.cn/problems/jBjn9C/)

- 标签：树、设计、二叉搜索树、二叉树、数据流、堆（优先队列）
- 难度：简单

## 题目链接

- [剑指 Offer II 059. 数据流的第 K 大数值 - 力扣](https://leetcode.cn/problems/jBjn9C/)

## 题目大意

设计一个 ` KthLargest` 类，用于找到数据流中第 `k` 大元素。

- `KthLargest(int k, int[] nums)`：使用整数 `k` 和整数流 `nums` 初始化对象。
- `int add(int val)`：将 `val` 插入数据流 `nums` 后，返回当前数据流中第 `k` 大的元素。

## 解题思路

- 建立大小为 `k` 的大顶堆，堆中元素保证不超过 k 个。
- 每次 `add` 操作时，将新元素压入堆中，如果堆中元素超出了 `k` 个，则将堆中最小元素（堆顶）移除。
- 此时堆中最小元素（堆顶）就是整个数据流中的第 `k` 大元素。

## 代码

```python
import heapq

class KthLargest:

    def __init__(self, k: int, nums: List[int]):
        self.min_heap = []
        self.k = k
        for num in nums:
            heapq.heappush(self.min_heap, num)
            if len(self.min_heap) > k:
                heapq.heappop(self.min_heap)


    def add(self, val: int) -> int:
        heapq.heappush(self.min_heap, val)
        if len(self.min_heap) > self.k:
            heapq.heappop(self.min_heap)
        return self.min_heap[0]
```

# [剑指 Offer II 060. 出现频率最高的 k 个数字](https://leetcode.cn/problems/g5c51o/)

- 标签：数组、哈希表、分治、桶排序、计数、快速选择、排序、堆（优先队列）
- 难度：中等

## 题目链接

- [剑指 Offer II 060. 出现频率最高的 k 个数字 - 力扣](https://leetcode.cn/problems/g5c51o/)

## 题目大意

给定一个整数数组 `nums` 和一个整数 `k`。

要求：返回出现频率前 `k` 高的元素。可以按任意顺序返回答案。

## 解题思路

- 使用哈希表记录下数组中各个元素的频数。时间复杂度 $O(n)$，空间复杂度 $O(n)$。
- 然后将哈希表中的元素去重，转换为新数组。时间复杂度 $O(n)$，空间复杂度 $O(n)$。
- 利用建立大顶堆，此时堆顶元素即为频数最高的元素。时间复杂度 $O(n)$，空间复杂度 $O(n)$。
- 将堆顶元素加入到答案数组中，并交换堆顶元素与末尾元素，此时末尾元素已移出堆。继续调整大顶堆。时间复杂度 $O(log{n})$。
- 调整玩大顶堆之后，此时堆顶元素为频数第二高的元素，和上一步一样，将其加入到答案数组中，继续交换堆顶元素与末尾元素，继续调整大顶堆。
- 不断重复上步，直到 k 次结束。调整 k 次的时间复杂度 $O(nlog{n})$。

总体时间复杂度 $O(nlog{n})$。

因为用的是大顶堆，堆的规模是 N 个元素，调整 k 次，所以时间复杂度是 $O(nlog{n})$。
如果用小顶堆，只需维护 k 个元素的小顶堆，不断向堆中替换元素即可，时间复杂度为 $O(nlog{k})$。

## 代码

```python
class Solution:
    # 调整为大顶堆
    def heapify(self, nums, nums_dict, index, end):
        left = index * 2 + 1
        right = left + 1
        while left <= end:
            # 当前节点为非叶子节点
            max_index = index
            if nums_dict[nums[left]] > nums_dict[nums[max_index]]:
                max_index = left
            if right <= end and nums_dict[nums[right]] > nums_dict[nums[max_index]]:
                max_index = right
            if index == max_index:
                # 如果不用交换，则说明已经交换结束
                break
            nums[index], nums[max_index] = nums[max_index], nums[index]
            # 继续调整子树
            index = max_index
            left = index * 2 + 1
            right = left + 1

    # 初始化大顶堆
    def buildMaxHeap(self, nums, nums_dict):
        size = len(nums)
        # (size-2) // 2 是最后一个非叶节点，叶节点不用调整
        for i in range((size - 2) // 2, -1, -1):
            self.heapify(nums, nums_dict, i, size - 1)
        return nums

    # 堆排序方法（本题未用到）
    def maxHeapSort(self, nums, nums_dict):
        self.buildMaxHeap(nums)
        size = len(nums)
        for i in range(size):
            nums[0], nums[size - i - 1] = nums[size - i - 1], nums[0]
            self.heapify(nums, nums_dict, 0, size - i - 2)
        return nums

    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        # 统计元素频数
        nums_dict = dict()
        for num in nums:
            if num in nums_dict:
                nums_dict[num] += 1
            else:
                nums_dict[num] = 1

        # 使用 set 方法去重，得到新数组
        new_nums = list(set(nums))
        size = len(new_nums)
        # 初始化大顶堆
        self.buildMaxHeap(new_nums, nums_dict)
        res = list()
        for i in range(k):
            # 堆顶元素为当前堆中频数最高的元素，将其加入答案中
            res.append(new_nums[0])
            # 交换堆顶和末尾元素，继续调整大顶堆
            new_nums[0], new_nums[size - i - 1] = new_nums[size - i - 1], new_nums[0]
            self.heapify(new_nums, nums_dict, 0, size - i - 2)
        return res
```

# [剑指 Offer II 062. 实现前缀树](https://leetcode.cn/problems/QC3q1f/)

- 标签：设计、字典树、哈希表、字符串
- 难度：中等

## 题目链接

- [剑指 Offer II 062. 实现前缀树 - 力扣](https://leetcode.cn/problems/QC3q1f/)

## 题目大意

要求：实现前缀树数据结构的相关类 `Trie` 类。

`Trie` 类：

- `Trie()` 初始化前缀树对象。
- `void insert(String word)` 向前缀树中插入字符串 `word`。
- `boolean search(String word)` 如果字符串 `word` 在前缀树中，返回 `True`（即，在检索之前已经插入）；否则，返回 `False`。
- `boolean startsWith(String prefix)` 如果之前已经插入的字符串 `word` 的前缀之一为 `prefix`，返回 `True`；否则，返回 `False`。

## 解题思路

前缀树（字典树）是一棵多叉数，其中每个节点包含指向子节点的指针数组 `children`，以及布尔变量 `isEnd`。`children` 用于存储当前字符节点，一般长度为所含字符种类个数，也可以使用哈希表代替指针数组。`isEnd` 用于判断该节点是否为字符串的结尾。

下面依次讲解插入、查找前缀的具体步骤：

插入字符串：

- 从根节点开始插入字符串。对于待插入的字符，有两种情况：
  - 如果该字符对应的节点存在，则沿着指针移动到子节点，继续处理下一个字符。
  - 如果该字符对应的节点不存在，则创建一个新的节点，保存在 `children` 中对应位置上，然后沿着指针移动到子节点，继续处理下一个字符。
- 重复上述步骤，直到最后一个字符，然后将该节点标记为字符串的结尾。

查找前缀：

- 从跟姐点开始查找前缀，对于待查找的字符，有两种情况：
  - 如果该字符对应的节点存在，则沿着指针移动到子节点，继续查找下一个字符。
  - 如果该字符对应的节点不存在，则说明字典树中不包含该前缀，直接返回空指针。
- 重复上述步骤，直到最后一个字符搜索完毕，则说明字典树中存在该前缀。

## 代码

```python
class Trie:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.children = dict()
        self.isEnd = False


    def insert(self, word: str) -> None:
        """
        Inserts a word into the trie.
        """
        cur = self
        for ch in word:
            if ch not in cur.children:
                cur.children[ch] = Trie()
            cur = cur.children[ch]
        cur.isEnd = True


    def search(self, word: str) -> bool:
        """
        Returns if the word is in the trie.
        """
        cur = self
        for ch in word:
            if ch not in cur.children:
                return False
            cur = cur.children[ch]

        return cur is not None and cur.isEnd


    def startsWith(self, prefix: str) -> bool:
        """
        Returns if there is any word in the trie that starts with the given prefix.
        """
        cur = self
        for ch in prefix:
            if ch not in cur.children:
                return False
            cur = cur.children[ch]
        return cur is not None
```

# [剑指 Offer II 063. 替换单词](https://leetcode.cn/problems/UhWRSj/)

- 标签：字典树、数组、哈希表、字符串
- 难度：中等

## 题目链接

- [剑指 Offer II 063. 替换单词 - 力扣](https://leetcode.cn/problems/UhWRSj/)

## 题目大意

给定一个由许多词根组成的字典列表 `dictionary`，以及一个句子字符串 `sentence`。

要求：将句子中有词根的单词用词根替换掉。如果单词有很多词根，则用最短的词根替换掉他。最后输出替换之后的句子。

## 解题思路

将所有的词根存入到前缀树（字典树）中。然后在树上查找每个单词的最短词根。

## 代码

```python
class Trie:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.children = dict()
        self.isEnd = False


    def insert(self, word: str) -> None:
        """
        Inserts a word into the trie.
        """
        cur = self
        for ch in word:
            if ch not in cur.children:
                cur.children[ch] = Trie()
            cur = cur.children[ch]
        cur.isEnd = True


    def search(self, word: str) -> str:
        """
        Returns if the word is in the trie.
        """
        cur = self
        index = 0
        for ch in word:
            if ch not in cur.children:
                return word
            cur = cur.children[ch]
            index += 1
            if cur.isEnd:
                break
        return word[:index]


class Solution:
    def replaceWords(self, dictionary: List[str], sentence: str) -> str:
        trie_tree = Trie()
        for word in dictionary:
            trie_tree.insert(word)

        words = sentence.split(" ")
        size = len(words)
        for i in range(size):
            word = words[i]
            words[i] = trie_tree.search(word)
        return ' '.join(words)
```

# [剑指 Offer II 064. 神奇的字典](https://leetcode.cn/problems/US1pGT/)

- 标签：设计、字典树、哈希表、字符串
- 难度：中等

## 题目链接

- [剑指 Offer II 064. 神奇的字典 - 力扣](https://leetcode.cn/problems/US1pGT/)

## 题目大意

要求：设计一个使用单词表进行初始化的数据结构。单词表中的单词互不相同。如果给出一个单词，要求判定能否将该单词中的一个字母替换成另一个字母，是的所形成的新单词已经在够构建的单词表中。

实现 MagicDictionary 类：

- `MagicDictionary()` 初始化对象。
- `void buildDict(String[] dictionary)` 使用字符串数组 `dictionary` 设定该数据结构，`dictionary` 中的字符串互不相同。
- `bool search(String searchWord)` 给定一个字符串 `searchWord`，判定能否只将字符串中一个字母换成另一个字母，使得所形成的新字符串能够与字典中的任一字符串匹配。如果可以，返回 `True`；否则，返回 `False`。

## 解题思路

- 初始化使用字典树结构。

- `buildDict` 方法中将所有单词存入字典树中。

- `search` 方法中替换 `searchWord` 每一个位置上的字符，然后在字典树中查询。

## 代码

```python
class Trie:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.children = dict()
        self.isEnd = False


    def insert(self, word: str) -> None:
        """
        Inserts a word into the trie.
        """
        cur = self
        for ch in word:
            if ch not in cur.children:
                cur.children[ch] = Trie()
            cur = cur.children[ch]
        cur.isEnd = True


    def search(self, word: str) -> bool:
        """
        Returns if the word is in the trie.
        """
        cur = self
        for ch in word:
            if ch not in cur.children:
                return False
            cur = cur.children[ch]

        return cur is not None and cur.isEnd

class MagicDictionary:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.trie_tree = Trie()


    def buildDict(self, dictionary: List[str]) -> None:
        for word in dictionary:
            self.trie_tree.insert(word)


    def search(self, searchWord: str) -> bool:
        size = len(searchWord)
        for i in range(size):
            for j in range(26):
                new_ch = chr(ord('a') + j)
                if searchWord[i] != new_ch:
                    new_word = searchWord[:i] + new_ch + searchWord[i + 1:]
                    if self.trie_tree.search(new_word):
                        return True
        return False
```

# [剑指 Offer II 065. 最短的单词编码](https://leetcode.cn/problems/iSwD2y/)

- 标签：字典树、数组、哈希表、字符串
- 难度：中等

## 题目链接

- [剑指 Offer II 065. 最短的单词编码 - 力扣](https://leetcode.cn/problems/iSwD2y/)

## 题目大意

给定一个单词数组 `words`。要求对 `words` 进行编码成一个助记字符串，用来帮助记忆。`words` 中拥有相同字符后缀的单词可以合并成一个单词，比如`time` 和 `me` 可以合并成 `time`。同时每个不能再合并的单词末尾以 `#` 为结束符，将所有合并后的单词排列起来就是一个助记字符串。

要求：返回对 `words` 进行编码的最小助记字符串 `s` 的长度。

## 解题思路

构建一个字典树。然后对字符串长度进行从小到大排序。

再依次将去重后的所有单词插入到字典树中。如果出现比当前单词更长的单词，则将短单词的结尾置为 `False`，意为替换掉短单词。

然后再依次在字典树中查询所有单词，「单词长度 + 1」就是当前不能在合并的单词，累加起来就是答案。

## 代码

```python
class Trie:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.children = dict()
        self.isEnd = False


    def insert(self, word: str) -> None:
        """
        Inserts a word into the trie.
        """
        cur = self
        for ch in word:
            if ch not in cur.children:
                cur.children[ch] = Trie()
            cur = cur.children[ch]
            cur.isEnd = False
        cur.isEnd = True


    def search(self, word: str) -> bool:
        """
        Returns if the word is in the trie.
        """
        cur = self
        for ch in word:
            if ch not in cur.children:
                return False
            cur = cur.children[ch]

        return cur is not None and cur.isEnd

class Solution:
    def minimumLengthEncoding(self, words: List[str]) -> int:
        trie_tree = Trie()
        words = list(set(words))
        words.sort(key=lambda i: len(i))

        ans = 0
        for word in words:
            trie_tree.insert(word[::-1])

        for word in words:
            if trie_tree.search(word[::-1]):
                ans += len(word) + 1

        return ans
```

# [剑指 Offer II 066. 单词之和](https://leetcode.cn/problems/z1R5dt/)

- 标签：设计、字典树、哈希表、字符串
- 难度：中等

## 题目链接

- [剑指 Offer II 066. 单词之和 - 力扣](https://leetcode.cn/problems/z1R5dt/)

## 题目大意

要求：实现一个 MapSum 类，支持两个方法，`insert` 和 `sum`：

- `MapSum()` 初始化 MapSum 对象。
- `void insert(String key, int val)` 插入 `key-val` 键值对，字符串表示键 `key`，整数表示值 `val`。如果键 `key` 已经存在，那么原来的键值对将被替代成新的键值对。
- `int sum(string prefix)` 返回所有以该前缀 `prefix` 开头的键 `key` 的值的总和。

## 解题思路

可以构造前缀树（字典树）解题。

- 初始化时，构建一棵前缀树（字典树），并增加 `val` 变量。

- 调用插入方法时，用字典树存储 `key`，并在对应字母节点存储对应的 `val`。
- 在调用查询总和方法时，先查找该前缀 `prefix` 对应的前缀树节点，从该节点开始，递归遍历该节点的子节点，并累积子节点的 `val`，进行求和，并返回求和累加结果。

## 代码

```python
class Trie:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.children = dict()
        self.isEnd = False
        self.value = 0


    def insert(self, word: str, value: int) -> None:
        """
        Inserts a word into the trie.
        """
        cur = self
        for ch in word:
            if ch not in cur.children:
                cur.children[ch] = Trie()
            cur = cur.children[ch]
        cur.isEnd = True
        cur.value = value


    def search(self, word: str) -> int:
        """
        Returns if the word is in the trie.
        """
        cur = self
        for ch in word:
            if ch not in cur.children:
                return 0
            cur = cur.children[ch]
        return self.dfs(cur)

    def dfs(self, root) -> int:
        if not root:
            return 0
        res = root.value
        for node in root.children.values():
            res += self.dfs(node)
        return res



class MapSum:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.trie_tree = Trie()


    def insert(self, key: str, val: int) -> None:
        self.trie_tree.insert(key, val)


    def sum(self, prefix: str) -> int:
        return self.trie_tree.search(prefix)
```

# [剑指 Offer II 067. 最大的异或](https://leetcode.cn/problems/ms70jA/)

- 标签：位运算、字典树、数组、哈希表
- 难度：中等

## 题目链接

- [剑指 Offer II 067. 最大的异或 - 力扣](https://leetcode.cn/problems/ms70jA/)

## 题目大意

给定一个整数数组 `nums`。

要求：返回 `num[i] XOR nums[j]` 的最大运算结果。其中 `0 ≤ i ≤ j < n`。

## 解题思路

最直接的想法暴力求解。两层循环计算两两之间的异或结果，记录并更新最大异或结果。

更好的做法可以减少一重循环。首先，要取得异或结果的最大值，那么从二进制的高位到低位，尽可能的让每一位异或结果都为 `1`。

将数组中所有数字的二进制形式从高位到低位依次存入字典树中。然后是利用异或运算交换律：如果 `a ^ b = max` 成立，那么 `a ^ max = b` 与 `b ^ max = a` 均成立。这样当我们知道 `a` 和 `max` 时，可以通过交换律求出 `b`。`a` 是我们遍历的每一个数，`max` 是我们想要尝试的最大值，从 `111111...` 开始，从高位到低位依次填 `1`。

对于 `a` 和 `max`，如果我们所求的 `b` 也在字典树中，则表示 `max` 是可以通过 `a` 和 `b` 得到的，那么 `max` 就是所求最大的异或。如果 `b` 不在字典树中，则减小 `max` 值继续判断，或者继续查询下一个 `a`。

## 代码

```python
class Trie:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.children = dict()
        self.isEnd = False


    def insert(self, num: int, max_bit: int) -> None:
        """
        Inserts a word into the trie.
        """
        cur = self
        for i in range(max_bit, -1, -1):
            bit = num >> i & 1
            if bit not in cur.children:
                cur.children[bit] = Trie()
            cur = cur.children[bit]
        cur.isEnd = True

    def search(self, num: int, max_bit: int) -> int:
        """
        Returns if the word is in the trie.
        """
        cur = self
        res = 0
        for i in range(max_bit, -1, -1):
            bit = num >> i & 1
            if 1 - bit not in cur.children:
                res = res * 2
                cur = cur.children[bit]
            else:
                res = res * 2 + 1
                cur = cur.children[1 - bit]
        return res

class Solution:
    def findMaximumXOR(self, nums: List[int]) -> int:
        trie_tree = Trie()
        max_bit = len(format(max(nums), 'b')) - 1
        ans = 0
        for num in nums:
            trie_tree.insert(num, max_bit)
            ans = max(ans, trie_tree.search(num, max_bit))
            
        return ans
```

# [剑指 Offer II 068. 查找插入位置](https://leetcode.cn/problems/N6YdxV/)

- 标签：数组、二分查找
- 难度：简单

## 题目链接

- [剑指 Offer II 068. 查找插入位置 - 力扣](https://leetcode.cn/problems/N6YdxV/)

## 题目大意

给定一个排好序的数组 `nums`，以及一个目标值 `target`。

要求：在数组中找到目标值，并返回下标。如果找不到，则返回目标值按顺序插入数组的位置。

## 解题思路

二分查找法。利用两个指针 `left` 和 `right`，分别指向数组首尾位置。每次用 `left` 和 `right` 中间位置上的元素值与目标值做比较，如果等于目标值，则返回当前位置。如果小于目标值，则更新 `left` 位置为 `mid + 1`，继续查找。如果大于目标值，则更新 `right` 位置为 `mid - 1`，继续查找。直到查找到目标值，或者 `left > right` 值时停止查找。然后返回 `left` 所在位置，即是代插入数组的位置。

## 代码

```python
class Solution:
    def searchInsert(self, nums: List[int], target: int) -> int:
        n = len(nums)
        left = 0
        right = n - 1
        while left <= right:
            mid = left + (right - left) // 2
            if nums[mid] == target:
                return mid
            elif nums[mid] < target:
                left = mid + 1
            else:
                right = mid - 1

        return left
```

# [剑指 Offer II 072. 求平方根](https://leetcode.cn/problems/jJ0w9p/)

- 标签：数学、二分查找
- 难度：简单

## 题目链接

- [剑指 Offer II 072. 求平方根 - 力扣](https://leetcode.cn/problems/jJ0w9p/)

## 题目大意

要求：实现 `int sqrt(int x)` 函数。计算并返回 `x` 的平方根（只保留整数部分），其中 `x` 是非负整数。

## 解题思路

因为求解的是 x 开方的整数部分。所以我们可以从 0~x 的范围进行遍历，找到 k^2 <= x 的最大结果。

为了减少时间复杂度，使用二分查找的方式来搜索答案。

## 代码

```python
class Solution:
    def mySqrt(self, x: int) -> int:
        left = 0
        right = x
        ans = -1
        while left <= right:
            mid = (left + right) // 2
            if mid * mid <= x:
                ans = mid
                left = mid + 1
            else:
                right = mid - 1
        return ans
```

# [剑指 Offer II 073. 狒狒吃香蕉](https://leetcode.cn/problems/nZZqjQ/)

- 标签：数组、二分查找
- 难度：中等

## 题目链接

- [剑指 Offer II 073. 狒狒吃香蕉 - 力扣](https://leetcode.cn/problems/nZZqjQ/)

## 题目大意

给定一个数组 `piles` 代表 `n` 堆香蕉。其中 `piles[i]` 表示第 `i` 堆香蕉的个数。再给定一个整数 `h` ，表示最多可以在 `h` 小时内吃完所有香蕉。狒狒决定以速度每小时 `k`（未知）根的速度吃香蕉。每一个小时，她将选择其中一堆香蕉，从中吃掉 `k` 根。如果这堆香蕉少于 `k` 根，狒狒将在这一小时吃掉这堆的所有香蕉，并且这一小时不会再吃其他堆的香蕉。  

要求：返回狒狒可以在 `h` 小时内吃掉所有香蕉的最小速度 `k`（`k` 为整数）。

## 解题思路

先来看 `k` 的取值范围，因为 `k` 是整数，且速度肯定不能为 `0` 吧，为 `0` 的话就永远吃不完了。所以`k` 的最小值可以取 `1`。`k` 的最大值根香蕉中最大堆的香蕉个数有关，因为 `1` 个小时内只能选择一堆吃，不能再吃其他堆的香蕉，则 `k` 的最大值取香蕉堆的最大值即可。即 `k` 的最大值为 `max(piles)`。

我们的目标是求出 `h` 小时内吃掉所有香蕉的最小速度 `k`。现在有了区间「`[1, max(piles)]`」，有了目标「最小速度 `k`」。接下来使用二分查找算法来查找「最小速度 `k`」。至于计算 `h` 小时内能否以 `k` 的速度吃完香蕉，我们可以再写一个方法 `canEat` 用于判断。如果能吃完就返回 `True`，不能吃完则返回 `False`。下面说一下算法的具体步骤。

- 使用两个指针 `left`、`right`。令 `left` 指向 `1`，`right` 指向 `max(piles)`。代表待查找区间为 `[left, right]`

- 取两个节点中心位置 `mid`，判断是否能在 `h` 小时内以 `k` 的速度吃完香蕉。
  - 如果不能吃完，则将区间 `[left, mid]` 排除掉，继续在区间 `[mid + 1, right]` 中查找。
  - 如果能吃完，说明 `k` 还可以继续减小，则继续在区间 `[left, mid]` 中查找。
- 当 `left == right` 时跳出循环，返回 `left`。

## 代码

```python
class Solution:
    def canEat(self, piles, hour, speed):
        time = 0
        for pile in piles:
            time += (pile + speed - 1) // speed
        return time <= hour

    def minEatingSpeed(self, piles: List[int], h: int) -> int:
        left, right = 1, max(piles)

        while left < right:
            mid = left + (right - left) // 2
            if not self.canEat(piles, h, mid):
                left = mid + 1
            else:
                right = mid

        return left
```

# [剑指 Offer II 074. 合并区间](https://leetcode.cn/problems/SsGoHC/)

- 标签：数组、排序
- 难度：中等

## 题目链接

- [剑指 Offer II 074. 合并区间 - 力扣](https://leetcode.cn/problems/SsGoHC/)

## 题目大意

给定一个数组 `intervals` 表示若干个区间的集合，`intervals[i] = [starti, endi]` 表示单个区间。

要求：合并所有重叠的区间，并返回一个不重叠的区间数组，该数组需要恰好覆盖原数组中的所有区间。

## 解题思路

设定一个数组 `ans` 用于表示最终不重叠的区间数组，然后对原始区间先按照区间左端点大小从小到大进行排序。

遍历所有区间。先将第一个区间加入 `ans` 数组中。然后依次考虑后边的区间，如果第 `i` 个区间左端点在前一个区间右端点右侧，则这两个区间不会重合，直接将该区间加入 `ans` 数组中。否则的话，这两个区间重合，判断一下两个区间的右区间值，更新前一个区间的右区间值为较大值，然后继续考虑下一个区间，以此类推。

## 代码

```python
class Solution:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        intervals.sort(key=lambda x: x[0])

        ans = []
        for interval in intervals:
            if not ans or ans[-1][1] < interval[0]:
                ans.append(interval)
            else:
                ans[-1][1] = max(ans[-1][1], interval[1])
        return ans
```

# [剑指 Offer II 075. 数组相对排序](https://leetcode.cn/problems/0H97ZC/)

- 标签：数组、哈希表、计数排序、排序
- 难度：简单

## 题目链接

- [剑指 Offer II 075. 数组相对排序 - 力扣](https://leetcode.cn/problems/0H97ZC/)

## 题目大意

给定两个数组，`arr1` 和 `arr2`，其中 `arr2` 中的元素各不相同，`arr2` 中的每个元素都出现在 `arr1` 中。

要求：对 `arr1` 中的元素进行排序，使 `arr1` 中项的相对顺序和 `arr2` 中的相对顺序相同。未在 `arr2` 中出现过的元素需要按照升序放在 `arr1` 的末尾。

注意：

- `1 <= arr1.length, arr2.length <= 1000`。
- `0 <= arr1[i], arr2[i] <= 1000`。

## 解题思路

因为元素值范围在 `[0, 1000]`，所以可以使用计数排序的思路来解题。

使用数组 `count` 统计 `arr1` 各个元素个数。

遍历 `arr2` 数组，将对应元素`num2` 按照个数 `count[num2]` 添加到答案数组 `ans` 中，同时在 `count` 数组中减去对应个数。

然后在处理 `count` 中剩余元素，将 `count` 中大于 `0` 的元素下标依次添加到答案数组 `ans` 中。

## 代码

```python
class Solution:
    def relativeSortArray(self, arr1: List[int], arr2: List[int]) -> List[int]:
        count = [0 for _ in range(1010)]
        for num1 in arr1:
            count[num1] += 1
        res = []
        for num2 in arr2:
            while count[num2] > 0:
                res.append(num2)
                count[num2] -= 1

        for num in range(len(count)):
            while count[num] > 0:
                res.append(num)
                count[num] -= 1

        return res
```

# [剑指 Offer II 076. 数组中的第 k 大的数字](https://leetcode.cn/problems/xx4gT2/)

- 标签：数组、分治、快速选择、排序、堆（优先队列）
- 难度：中等

## 题目链接

- [剑指 Offer II 076. 数组中的第 k 大的数字 - 力扣](https://leetcode.cn/problems/xx4gT2/)

## 题目大意

给定一个未排序的数组 `nums`，从中找到第 `k` 个最大的数字。

## 解题思路

很不错的一道题，面试常考。

直接可以想到的思路是：排序后输出数组上对应第 k 位大的数。所以问题关键在于排序方法的复杂度。

冒泡排序、选择排序、插入排序时间复杂度 $O(n^2)$ 太高了，解答会超时。

可考虑堆排序、归并排序、快速排序。

这道题的要求是找到第 k 大的元素，使用归并排序只有到最后排序完毕才能返回第 k 大的数。而堆排序每次排序之后，就会确定一个元素的准确排名，同理快速排序也是如此。

### 1. 堆排序

升序堆排序的思路如下：

1. 先建立大顶堆

2. 让堆顶最大元素与最后一个交换，然后调整第一个元素到倒数第二个元素，这一步获取最大值

3. 再交换堆顶元素与倒数第二个元素，然后调整第一个元素到倒数第三个元素，这一步获取第二大值

4. 以此类推，直到最后一个元素交换之后完毕。

这道题我们只需进行 1 次建立大顶堆， k-1 次调整即可得到第 k 大的数。

时间复杂度：$O(n^2)$

### 2. 快速排序

快速排序每次调整，都会确定一个元素的最终位置，且以该元素为界限，将数组分成了两个数组，前一个数组元素都比该元素小，后一个元素都比该元素大。

这样，只要某次划分的元素恰好是第 k 个下标就找到了答案。并且我们只需关注 k 元素所在区间的排序情况，与 k 元素无关的区间排序都可以忽略。这样进一步减少了执行步骤。

### 3. 借用标准库（不建议）

提交代码中的最快代码是调用了 Python 的 heapq 库，或者 sort 方法。
这样的确可以通过，但是不建议这样做。借用标准库实现，只能说对这个库的 API 和相关数据结构的用途相对熟悉，而不代表着掌握了这个数据结构。可以问问自己，如果换一种语言，自己还能不能实现对应的数据结构？刷题的本质目的是为了把算法学会学透，而不仅仅是调 API。

## 代码

1. 堆排序

```python
class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        # 调整为大顶堆
        def heapify(nums, index, end):
            left = index * 2 + 1
            right = left + 1
            while left <= end:
                # 当前节点为非叶子节点
                max_index = index
                if nums[left] > nums[max_index]:
                    max_index = left
                if right <= end and nums[right] > nums[max_index]:
                    max_index = right
                if index == max_index:
                    # 如果不用交换，则说明已经交换结束
                    break
                nums[index], nums[max_index] = nums[max_index], nums[index]
                # 继续调整子树
                index = max_index
                left = index * 2 + 1
                right = left + 1
                
        # 初始化大顶堆
        def buildMaxHeap(nums):
            size = len(nums)
            # (size-2) // 2 是最后一个非叶节点，叶节点不用调整
            for i in range((size - 2) // 2, -1, -1):
                heapify(nums, i, size - 1)
            return nums

        buildMaxHeap(nums)
        size = len(nums)
        for i in range(k-1):
            nums[0], nums[size-i-1] = nums[size-i-1], nums[0]
            heapify(nums, 0, size-i-2)
        return nums[0]
```

2. 快速排序

```python
import random
class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        def randomPartition(nums, low, high):
            i = random.randint(low, high)
            nums[i], nums[high] = nums[high], nums[i]
            return partition(nums, low, high)

        def partition(nums, low, high):
            x = nums[high]
            i = low-1
            for j in range(low, high):
                if nums[j] <= nums[high]:
                    i += 1
                    nums[i], nums[j] = nums[j], nums[i]
            nums[i+1], nums[high] = nums[high], nums[i+1]
            return i+1

        def quickSort(nums, low, high, k):
            n = len(nums)
            if low < high:
                pi = randomPartition(nums, low, high)
                if pi == n-k:
                    return nums[len(nums)-k]
                if pi > n-k:
                    quickSort(nums, low, pi-1, k)
                if pi < n-k:
                    quickSort(nums, pi+1, high, k)

            return nums[len(nums)-k]

        return quickSort(nums, 0, len(nums)-1, k)
```

3. 借用标准库

```python
class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        nums.sort()
        return nums[len(nums)-k]
```

```python
import heapq
class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        res = []
        for n in nums:
            if len(res) < k:
                heapq.heappush(res, n)
            elif n > res[0]:
                heapq.heappop(res)
                heapq.heappush(res, n)
        return heapq.heappop(res)
```



# [剑指 Offer II 077. 链表排序](https://leetcode.cn/problems/7WHec2/)

- 标签：链表、双指针、分治、排序、归并排序
- 难度：中等

## 题目链接

- [剑指 Offer II 077. 链表排序 - 力扣](https://leetcode.cn/problems/7WHec2/)

## 题目大意

给定链表的头节点 `head`。

要求：按照升序排列并返回排序后的链表。

## 解题思路

归并排序。

1. 利用快慢指针找到链表的中点，以中点为界限将链表拆分成两个子链表。
2. 然后对两个子链表分别递归排序。
3. 将排序后的子链表进行归并排序，得到完整的排序后的链表。

## 代码

```python
class Solution:
    def merge_sort(self, head: ListNode, tail: ListNode) -> ListNode:
        if not head:
            return head
        if head.next == tail:
            head.next = None
            return head
        slow = fast = head
        while fast != tail:
            slow = slow.next
            fast = fast.next
            if fast != tail:
                fast = fast.next
        mid = slow
        return self.merge(self.merge_sort(head, mid), self.merge_sort(mid, tail))

    def merge(self, a: ListNode, b: ListNode) -> ListNode:
        root = ListNode(-1)
        cur = root
        while a and b:
            if a.val < b.val:
                cur.next = a
                a = a.next
            else:
                cur.next = b
                b = b.next
            cur = cur.next
        if a:
            cur.next = a
        if b:
            cur.next = b
        return root.next

    def sortList(self, head: ListNode) -> ListNode:
        return self.merge_sort(head, None)
```

# [剑指 Offer II 078. 合并排序链表](https://leetcode.cn/problems/vvXgSW/)

- 标签：链表、分治、堆（优先队列）、归并排序
- 难度：困难

## 题目链接

- [剑指 Offer II 078. 合并排序链表 - 力扣](https://leetcode.cn/problems/vvXgSW/)

## 题目大意

给定一个链表数组 `lists`，每个链表都已经按照升序排列。

要求：将所有链表合并到一个升序链表中，返回合并后的链表。

## 解题思路

分而治之的思想。将链表数组不断二分，转为规模为二分之一的子问题，然后再进行归并排序。

## 代码

```python
class Solution:
    def merge_sort(self, lists: List[ListNode], left: int, right: int) -> ListNode:
        if left == right:
            return lists[left]
        mid = left + (right - left) // 2
        node_left = self.merge_sort(lists, left, mid)
        node_right = self.merge_sort(lists, mid + 1, right)
        return self.merge(node_left, node_right)

    def merge(self, a: ListNode, b: ListNode) -> ListNode:
        root = ListNode(-1)
        cur = root
        while a and b:
            if a.val < b.val:
                cur.next = a
                a = a.next
            else:
                cur.next = b
                b = b.next
            cur = cur.next
        if a:
            cur.next = a
        if b:
            cur.next = b
        return root.next

    def mergeKLists(self, lists: List[ListNode]) -> ListNode:
        if not lists:
            return None
        size = len(lists)
        return self.merge_sort(lists, 0, size - 1)
```

# [剑指 Offer II 079. 所有子集](https://leetcode.cn/problems/TVdhkn/)

- 标签：位运算、数组、回溯
- 难度：中等

## 题目链接

- [剑指 Offer II 079. 所有子集 - 力扣](https://leetcode.cn/problems/TVdhkn/)

## 题目大意

给定一个整数数组 `nums`，数组中的元素互不相同。

要求：返回该数组所有可能的不重复子集。

## 解题思路

回溯算法，遍历数组 `nums`。为了使得子集不重复，每次遍历从当前位置的下一个位置进行下一层遍历。

## 代码

```python
class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:
        def backtrack(size, subset, index):
            res.append(subset)
            for i in range(index, size):
                backtrack(size, subset + [nums[i]], i + 1)

        size = len(nums)
        res = list()
        backtrack(size, [], 0)
        return res
```

# [剑指 Offer II 080. 含有 k 个元素的组合](https://leetcode.cn/problems/uUsW3B/)

- 标签：数组、回溯
- 难度：中等

## 题目链接

- [剑指 Offer II 080. 含有 k 个元素的组合 - 力扣](https://leetcode.cn/problems/uUsW3B/)

## 题目大意

给定两个整数 `n` 和 `k`。

要求：返回范围 `[1, n]` 中所有可能的 `k` 个数的组合。可以按任何顺序返回答案。

## 解题思路

组合问题通常可以用回溯算法来解决。定义两个数组 `res`、`path`。`res` 用来存放最终答案，`path` 用来存放当前符合条件的一个结果。再使用一个变量 `start_index` 来表示从哪一个数开始遍历。

定义回溯方法，`start_index = 1` 开始进行回溯。

- 如果 `path` 数组的长度等于 `k`，则将 `path` 中的元素加入到 `res` 数组中。
- 然后对 `[start_index, n]` 范围内的数进行遍历取值。
    - 将当前元素 `i` 加入 `path` 数组。
    - 递归遍历 `[start_index, n]` 上的数。
    - 将遍历的 `i` 元素进行回退。
- 最终返回 `res` 数组。

## 代码

```python
class Solution:
    res = []
    path = []

    def backtrack(self, n: int, k: int, start_index: int):
        if len(self.path) == k:
            self.res.append(self.path[:])
            return
        for i in range(start_index, n - (k - len(self.path)) + 2):
            self.path.append(i)
            self.backtrack(n, k, i + 1)
            self.path.pop()

    def combine(self, n: int, k: int) -> List[List[int]]:
        self.res.clear()
        self.path.clear()
        self.backtrack(n, k, 1)
        return self.res
```

# [剑指 Offer II 081. 允许重复选择元素的组合](https://leetcode.cn/problems/Ygoe9J/)

- 标签：数组、回溯
- 难度：中等

## 题目链接

- [剑指 Offer II 081. 允许重复选择元素的组合 - 力扣](https://leetcode.cn/problems/Ygoe9J/)

## 题目大意

给定一个无重复元素的正整数数组 `candidates` 和一个正整数 `target`。

要求：找出 `candidates` 中所有可以使数字和为目标数 `target` 的唯一组合。

注意：数组 `candidates` 中的数字可以无限重复选取，且 `1 ≤ candidates[i] ≤ 200`。

## 解题思路

回溯算法，因为 `1 ≤ candidates[i] ≤ 200`，所以即便是 `candidates[i]` 值为 `1`，重复选取也会等于或大于 target，从而终止回溯。

建立两个数组 `res`、`path`。`res` 用于存放所有满足题意的组合，`path` 用于存放当前满足题意的一个组合。

定义回溯方法，`start_index = 1` 开始进行回溯。

- 如果 `sum > target`，则直接返回。
- 如果 `sum == target`，则将 `path` 中的元素加入到 `res` 数组中。
- 然后对 `[start_index, n]` 范围内的数进行遍历取值。
    - 如果 `sum + candidates[i] > target`，可以直接跳出循环。
    - 将和累积，即 `sum += candidates[i]`，然后将当前元素 `i` 加入 `path` 数组。
    - 递归遍历 `[start_index, n]` 上的数。
    - 加之前的和回退，即 `sum -= candidates[i]`，然后将遍历的 `i` 元素进行回退。
- 最终返回 `res` 数组。

## 代码

```python
class Solution:
    res = []
    path = []

    def backtrack(self, candidates: List[int], target: int, sum: int, start_index: int):
        if sum > target:
            return

        if sum == target:
            self.res.append(self.path[:])
            return

        for i in range(start_index, len(candidates)):
            if sum + candidates[i] > target:
                break
            sum += candidates[i]
            self.path.append(candidates[i])
            self.backtrack(candidates, target, sum, i)
            sum -= candidates[i]
            self.path.pop()

    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        self.res.clear()
        self.path.clear()
        candidates.sort()
        self.backtrack(candidates, target, 0, 0)
        return self.res
```

# [剑指 Offer II 082. 含有重复元素集合的组合](https://leetcode.cn/problems/4sjJUc/)

- 标签：数组、回溯
- 难度：中等

## 题目链接

- [剑指 Offer II 082. 含有重复元素集合的组合 - 力扣](https://leetcode.cn/problems/4sjJUc/)

## 题目大意

给定一个数组 `candidates` 和一个目标数 `target`。

要求：找出 `candidates` 中所有可以使数字和为目标数 `target` 的组合。

数组 `candidates` 中的数字在每个组合中只能使用一次，且 `1 ≤ candidates[i] ≤ 50`。

## 解题思路

本题不能有重复组合，关键步骤在于去重。

在回溯遍历的时候，下一层递归的 `start_index` 要从当前节点的后一位开始遍历，即 `i + 1` 位开始。而且统一递归层不能使用相同的元素，即需要增加一句判断 `if i > start_index and candidates[i] == candidates[i - 1]: continue`。

## 代码

```python
class Solution:
    res = []
    path = []

    def backtrack(self, candidates: List[int], target: int, sum: int, start_index: int):
        if sum > target:
            return
        if sum == target:
            self.res.append(self.path[:])
            return

        for i in range(start_index, len(candidates)):
            if sum + candidates[i] > target:
                break
            if i > start_index and candidates[i] == candidates[i - 1]:
                continue
            sum += candidates[i]
            self.path.append(candidates[i])
            self.backtrack(candidates, target, sum, i + 1)
            sum -= candidates[i]
            self.path.pop()

    def combinationSum2(self, candidates: List[int], target: int) -> List[List[int]]:
        self.res.clear()
        self.path.clear()
        candidates.sort()
        self.backtrack(candidates, target, 0, 0)
        return self.res
```

# [剑指 Offer II 083. 没有重复元素集合的全排列](https://leetcode.cn/problems/VvJkup/)

- 标签：数组、回溯
- 难度：中等

## 题目链接

- [剑指 Offer II 083. 没有重复元素集合的全排列 - 力扣](https://leetcode.cn/problems/VvJkup/)

## 题目大意

给定一个不含重复数字的数组 `nums` 。

要求：返回其有可能的全排列，可以按任意顺序返回。

## 解题思路

回溯算法递归遍历 `nums` 元素。同时使用 `visited` 数组来标记该元素在当前排列中是否被访问过。若未被访问过则将其加入排列中，并在访问后将该元素变为未访问状态。

## 代码

```python
class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        def backtrack(size, arrange, index):
            if index == size:
                res.append(arrange)
                return
            for i in range(size):
                if visited[i] == True:
                    continue
                visited[i] = True
                backtrack(size, arrange + [nums[i]], index + 1)
                visited[i] = False

        size = len(nums)
        res = list()
        visited = [False for _ in range(size)]
        backtrack(size, [], 0)
        return res
```

# [剑指 Offer II 084. 含有重复元素集合的全排列](https://leetcode.cn/problems/7p8L0Z/)

- 标签：数组、回溯
- 难度：中等

## 题目链接

- [剑指 Offer II 084. 含有重复元素集合的全排列 - 力扣](https://leetcode.cn/problems/7p8L0Z/)

## 题目大意

给定一个可包含重复数字的序列 `nums` 。

要求：按任意顺序返回所有不重复的全排列。

## 解题思路

这道题跟「[剑指 Offer II 083. 没有重复元素集合的全排列](https://leetcode.cn/problems/VvJkup/)」不一样的地方在于增加了序列中的元素可重复这一条件。这就涉及到了去重。先对 `nums` 进行排序，然后使用 visited 数组标记该元素在当前排列中是否被访问过。若未被访问过则将其加入排列中，并在访问后将该元素变为未访问状态。

然后再递归遍历下一层元素之前，增加一句语句进行判重：`if i > 0 and nums[i] == nums[i - 1] and not visited[i - 1]: continue`。

然后进行回溯遍历。

## 代码

```python
class Solution:
    res = []
    path = []

    def backtrack(self, nums: List[int], visited: List[bool]):
        if len(self.path) == len(nums):
            self.res.append(self.path[:])
            return
        for i in range(len(nums)):
            if i > 0 and nums[i] == nums[i - 1] and not visited[i - 1]:
                continue

            if not visited[i]:
                visited[i] = True
                self.path.append(nums[i])
                self.backtrack(nums, visited)
                self.path.pop()
                visited[i] = False

    def permuteUnique(self, nums: List[int]) -> List[List[int]]:
        self.res.clear()
        self.path.clear()
        nums.sort()
        visited = [False for _ in range(len(nums))]
        self.backtrack(nums, visited)
        return self.res
```

# [剑指 Offer II 085. 生成匹配的括号](https://leetcode.cn/problems/IDBivT/)

- 标签：字符串、动态规划、回溯
- 难度：中等

## 题目链接

- [剑指 Offer II 085. 生成匹配的括号 - 力扣](https://leetcode.cn/problems/IDBivT/)

## 题目大意

给定一个整数 `n`。

要求：生成所有有可能且有效的括号组合。

## 解题思路

通过回溯算法生成所有答案。为了生成的括号组合是有效的，回溯的时候，使用一个标记变量 `symbol` 来表示是否当前组合是否成对匹配。

如果在当前组合中增加一个 `(`，则 `symbol += 1`，如果增加一个 `)`，则 `symbol -= 1`。显然只有在 `symbol < n` 的时候，才能增加 `(`，在 `symbol > 0` 的时候，才能增加 `)`。

如果最终生成 `2 * n` 的括号组合，并且 `symbol == 0`，则说明当前组合是有效的，将其加入到最终答案数组中。

最终输出最终答案数组。

## 代码

```python
class Solution:
    def generateParenthesis(self, n: int) -> List[str]:
        def backtrack(parenthesis, symbol, index):
            if n * 2 == index:
                if symbol == 0:
                    parentheses.append(parenthesis)
            else:
                if symbol < n:
                    backtrack(parenthesis + '(', symbol + 1, index + 1)
                if symbol > 0:
                    backtrack(parenthesis + ')', symbol - 1, index + 1)

        parentheses = list()
        backtrack("", 0, 0)
        return parentheses
```

# [剑指 Offer II 086. 分割回文子字符串](https://leetcode.cn/problems/M99OJA/)

- 标签：深度优先搜索、广度优先搜索、图、哈希表
- 难度：中等

## 题目链接

- [剑指 Offer II 086. 分割回文子字符串 - 力扣](https://leetcode.cn/problems/M99OJA/)

## 题目大意

给定一个字符串 `s`将 `s` 分割成一些子串，保证每个子串都是「回文串」。

要求：返回 `s` 所有可能的分割方案。

## 解题思路

回溯算法，建立两个数组 `res`、`path`。`res` 用于存放所有满足题意的组合，`path` 用于存放当前满足题意的一个组合。

在回溯的时候判断当前子串是否为回文串，如果不是则跳过，如果是则继续向下一层遍历。

定义判断是否为回文串的方法和回溯方法，从 `start_index = 0` 的位置开始回溯。

- 如果 `start_index >= len(s)`，则将 `path` 中的元素加入到 `res` 数组中。
- 然后对 `[start_index, len(s) - 1]` 范围内的子串进行遍历取值。
    - 如果字符串 `s` 在范围 `[start_index, i]` 所代表的子串是回文串，则将其加入 `path` 数组。
    - 递归遍历 `[i + 1, len(s) - 1]` 范围上的子串。
    - 然后将遍历的范围 `[start_index, i]` 所代表的子串进行回退。
- 最终返回 `res` 数组。

## 代码

```python
class Solution:
    res = []
    path = []

    def backtrack(self, s: str, start_index: int):
        if start_index >= len(s):
            self.res.append(self.path[:])
            return
        for i in range(start_index, len(s)):
            if self.ispalindrome(s, start_index, i):
                self.path.append(s[start_index: i + 1])
                self.backtrack(s, i + 1)
                self.path.pop()

    def ispalindrome(self, s: str, start: int, end: int):
        i, j = start, end
        while i < j:
            if s[i] != s[j]:
                return False
            i += 1
            j -= 1
        return True

    def partition(self, s: str) -> List[List[str]]:
        self.res.clear()
        self.path.clear()
        self.backtrack(s, 0)
        return self.res
```

# [剑指 Offer II 087. 复原 IP](https://leetcode.cn/problems/0on3uN/)

- 标签：字符串、回溯
- 难度：中等

## 题目链接

- [剑指 Offer II 087. 复原 IP - 力扣](https://leetcode.cn/problems/0on3uN/)

## 题目大意

给定一个只包含数字的字符串，用来表示一个 IP 地址。

要求：返回所有由 `s` 构成的有效 IP 地址，可以按任何顺序返回答案。

- 有效 IP 地址：正好由四个整数（每个整数由 0~255 的数构成，且不能含有前导 0），整数之间用 `.` 分割。

例如：`0.1.2.201` 和 `192.168.1.1` 是有效 IP 地址，但是 `0.011.255.245`、`192.168.1.312` 和 `192.168@1.1` 是 无效 IP 地址。

## 解题思路

回溯算法。使用 `res` 存储所有有效 IP 地址。用 `point_num` 表示当前 IP 地址的 `.` 符号个数。

定义回溯方法，从 `start_index` 位置开始遍历字符串。

- 如果字符串中添加的 `.` 符号数量为 `3`，则判断当前字符串是否为有效 IP 地址，若为有效 IP 地址则加入到 `res` 数组中。直接返回。
- 然后在 `[start_index, len(s) - 1]` 范围循环遍历，判断 `[start_index, i]` 范围所代表的子串是否合法。如果合法：
    - 则 `point_num += 1`。
    - 然后在 i 位置后边增加 `.` 符号，继续回溯遍历。
    - 最后 `point_num -= 1` 进行回退。
- 不符合则直接跳出循环。
- 最后返回 `res`。

## 代码

```python
class Solution:
    res = []

    def backstrack(self, s: str, start_index: int, point_num: int):
        if point_num == 3:
            if self.isValid(s, start_index, len(s) - 1):
                self.res.append(s)
            return
        for i in range(start_index, len(s)):
            if self.isValid(s, start_index, i):
                point_num += 1
                self.backstrack(s[:i + 1] + '.' + s[i + 1:], i + 2, point_num)
                point_num -= 1
            else:
                break

    def isValid(self, s: str, start: int, end: int):
        if start > end:
            return False
        if s[start] == '0' and start != end:
            return False
        num = 0
        for i in range(start, end + 1):
            if s[i] > '9' or s[i] < '0':
                return False
            num = num * 10 + ord(s[i]) - ord('0')
            if num > 255:
                return False
        return True

    def restoreIpAddresses(self, s: str) -> List[str]:
        self.res.clear()
        if len(s) > 12:
            return self.res
        self.backstrack(s, 0, 0)
        return self.res
```

# [剑指 Offer II 088. 爬楼梯的最少成本](https://leetcode.cn/problems/GzCJIP/)

- 标签：数组、动态规划
- 难度：简单

## 题目链接

- [剑指 Offer II 088. 爬楼梯的最少成本 - 力扣](https://leetcode.cn/problems/GzCJIP/)

## 题目大意

给定一个数组 `cost` 代表一段楼梯，`cost[i]` 代表爬上第 `i` 阶楼梯醒酒药花费的体力值（下标从 `0` 开始）。

每爬上一个阶梯都要花费对应的体力值，一旦支付了相应的体力值，你就可以选择向上爬一个阶梯或者爬两个阶梯。

要求：找出达到楼层顶部的最低花费。在开始时，你可以选择从下标为 `0` 或 `1` 的元素作为初始阶梯。

## 解题思路

使用动态规划方法。

状态 `dp[i]` 表示为：到达第 `i` 个台阶所花费的最少体⼒。

则状态转移方程为： `dp[i] = min(dp[i - 1], dp[i - 2]) + cost[i]`。

表示为：到达第 `i` 个台阶所花费的最少体⼒ = 到达第 `i - 1` 个台阶所花费的最小体力 与 到达第 `i - 2` 个台阶所花费的最小体力中的最小值 + 到达第 `i` 个台阶所需要花费的体力值。

## 代码

```python
class Solution:
    def minCostClimbingStairs(self, cost: List[int]) -> int:
        size = len(cost)
        dp = [0 for _ in range(size + 1)]
        for i in range(2, size + 1):
            dp[i] = min(dp[i - 1] + cost[i - 1], dp[i - 2] + cost[i - 2])
        return dp[size]
```

# [剑指 Offer II 089. 房屋偷盗](https://leetcode.cn/problems/Gu0c2T/)

- 标签：数组、动态规划
- 难度：中等

## 题目链接

- [剑指 Offer II 089. 房屋偷盗 - 力扣](https://leetcode.cn/problems/Gu0c2T/)

## 题目大意

给定一个数组 `nums`，`num[i]` 代表第 `i` 间房屋存放的金额。相邻的房屋装有防盗系统，假如相邻的两间房屋同时被偷，系统就会报警。假如你是一名专业的小偷。

要求：计算在不触动警报装置的情况下，一夜之内能够偷窃到的最高金额。

## 解题思路

可以用动态规划来解决问题，关键点在于找到状态转移方程。

先考虑最简单的情况。假如只有一间房，则直接偷这间屋子就能偷到最高金额，即 `dp[0] = nums[i]`。假如只有两间房屋，那么就选择金额最大的那间屋进行偷窃，就可以偷到最高金额，即 `dp[1] = max(nums[0], nums[1])`。

如果房屋大于两间，则偷窃第 `i` 间房屋的时候，就有两种状态：

- 偷窃第 `i` 间房屋，那么第 `i - 1` 间房屋就不能偷窃了，偷窃的最高金额为：前 `i - 2` 间房屋的最高总金额 + 第 `i` 间房屋的金额，即 `dp[i] = dp[i-2] + nums[i]`；
- 不偷窃第 `i` 间房屋，那么第 `i - 1` 间房屋可以偷窃，偷窃的最高金额为：前 `i - 1` 间房屋的最高总金额，即 `dp[i] = dp[i-1]`。

然后这两种状态取最大值即可，即 `dp[i] = max(dp[i-2] + nums[i], dp[i-1])`。

总结下就是：

$dp[i] = \begin{cases} \begin{array} {**lr**}  nums[0] & i = 0 \cr max( nums[0], nums[1]) & i = 1 \cr max( dp[i-2] + nums[i], dp[i-1]) & i \ge 2 \end{array} \end{cases}$

## 代码

```python
class Solution:
    def rob(self, nums: List[int]) -> int:
        size = len(nums)
        dp = [0 for _ in range(size)]
        for i in range(size):
            if i == 0:
                dp[i] = nums[i]
            elif i == 1:
                dp[i] = max(nums[i - 1], nums[i])
            else:
                dp[i] = max(dp[i - 2] + nums[i], dp[i - 1])

        return dp[size - 1]
```

# [剑指 Offer II 090. 环形房屋偷盗](https://leetcode.cn/problems/PzWKhm/)

- 标签：数组、动态规划
- 难度：中等

## 题目链接

- [剑指 Offer II 090. 环形房屋偷盗 - 力扣](https://leetcode.cn/problems/PzWKhm/)

## 题目大意

给定一个数组 `nums`，`num[i]` 代表第 `i` 间房屋存放的金额，假设房屋可以围成一圈，首尾相连。相邻的房屋装有防盗系统，假如相邻的两间房屋同时被偷，系统就会报警。假如你是一名专业的小偷。

要求：计算在不触动警报装置的情况下，一夜之内能够偷窃到的最高金额。

## 解题思路

「[剑指 Offer II 089. 房屋偷盗](https://leetcode.cn/problems/Gu0c2T/)」的升级版。可以用动态规划来解决问题，关键点在于找到状态转移方程。

先来考虑最简单的情况。

假如只有一间房屋，则直接偷这间房屋就能偷到最高金额，即 $dp[0] = nums[i]$。假如有两间房屋，那么就选择金额最大的那间房屋进行偷窃，就可以偷到最高金额，即 $dp[1] = max(nums[0], nums[1])$。

两间屋子以下，最多只能偷窃一间房屋，则不用考虑首尾相连的情况。如果三个屋子以上，偷窃了第一间房屋，则不能偷窃最后一间房屋。同样偷窃了最后一间房屋则不能偷窃第一间房屋。

假设总共房屋数量为 N，这种情况可以转换为分别求解 $[0, N - 2]$ 和 $[1, N - 1]$ 范围下首尾不相连的房屋所能偷窃的最高金额，这就变成了「[剑指 Offer II 089. 房屋偷盗](https://leetcode.cn/problems/Gu0c2T/)」的求解问题。

「[剑指 Offer II 089. 房屋偷盗](https://leetcode.cn/problems/Gu0c2T/)」求解思路如下：

如果房屋大于两间，则偷窃第 `i` 间房屋的时候，就有两种状态：

- 偷窃第 `i` 间房屋，那么第 `i - 1` 间房屋就不能偷窃了，偷窃的最高金额为：前 `i - 2` 间房屋的最高总金额 + 第 `i` 间房屋的金额，即 $dp[i] = dp[i-2] + nums[i]$；
- 不偷窃第 `i` 间房屋，那么第 `i - 1` 间房屋可以偷窃，偷窃的最高金额为：前 `i - 1` 间房屋的最高总金额，即 $dp[i] = dp[i-1]$。

然后这两种状态取最大值即可，即 $dp[i] = max( dp[i-2] + nums[i], dp[i-1])$。

总结下就是：

$dp[i] = \begin{cases} nums[0], &  i = 0 \cr max( nums[0], nums[1]) & i = 1 \cr max( dp[i-2] + nums[i], dp[i-1]) & i \ge 2 \end{cases}$

## 代码

```python
class Solution:
    def rob(self, nums: List[int]) -> int:
        def helper(nums):
            size = len(nums)
            if size == 1:
                return nums[0]
            dp = [0 for _ in range(size)]
            for i in range(size):
                if i == 0:
                    dp[i] = nums[0]
                elif i == 1:
                    dp[i] = max(nums[i - 1], nums[i])
                else:
                    dp[i] = max(dp[i - 2] + nums[i], dp[i - 1])
            return dp[-1]

        if len(nums) == 1:
            return nums[0]
        else:
            return max(helper(nums[1:]), helper(nums[:-1]))
```

# [剑指 Offer II 093. 最长斐波那契数列](https://leetcode.cn/problems/Q91FMA/)

- 标签：数组、哈希表、动态规划
- 难度：中等

## 题目链接

- [剑指 Offer II 093. 最长斐波那契数列 - 力扣](https://leetcode.cn/problems/Q91FMA/)

## 题目大意

给定一个严格递增的正整数数组 `arr`。

要求：从 `arr` 中找出最长的斐波那契式的子序列的长度。如果不存斐波那契式的子序列，则返回 `0`。

- 斐波那契式序列：如果序列 $X_1, X_2, ..., X_n$ 满足：

    - $n \ge 3$；
    - 对于所有 $i + 2 \le n$，都有 $X_i + X_{i+1} = X_{i+2}$。

    则称该序列为斐波那契式序列。

- 斐波那契式子序列：从序列 `arr` 中挑选若干元素组成子序列，并且子序列满足斐波那契式序列，则称该序列为斐波那契式子序列。例如：`arr = [3, 4, 5, 6, 7, 8]`。则 `[3, 5, 8]` 是 `arr` 的一个斐波那契式子序列。

## 解题思路

我们先从最简单的暴力做法思考。

**1. 暴力做法：**

我们先来考虑暴力做法怎么做。

假设 `arr[i]`、`arr[j]`、`arr[k]` 是序列 `arr` 中的 3 个元素，且满足关系：`arr[i] + arr[j] == arr[k]`，则 `arr[i]`、`arr[j]`、`arr[k]` 就构成了 A 的一个斐波那契式子序列。

通过  `arr[i]`、`arr[j]`，我们可以确定下一个斐波那契式子序列元素的值为 `arr[i] + arr[j]`。

因为给定的数组是严格递增的，所以对于一个斐波那契式子序列，如果确定了 `arr[i]`、`arr[j]`，则可以顺着 `arr` 序列，从第 `j + 1` 的元素开始，查找值为 `arr[i] + arr[j]` 的元素 。找到 `arr[i] + arr[j]` 之后，然后在顺着查找子序列的下一个元素。

简单来说，就是确定了 `arr[i]`、`arr[j]`，就能尽可能的得到一个长的斐波那契式子序列，此时我们记录下子序列长度。然后对于不同的  `arr[i]`、`arr[j]`，统计不同的斐波那契式子序列的长度。将这些长度进行比较，其中最长的长度就是答案。

下面是暴力做法的代码：

```python
class Solution:
    def lenLongestFibSubseq(self, arr: List[int]) -> int:
        size = len(arr)
        ans = 0
        for i in range(size):
            for j in range(i + 1, size):
                temp_ans = 0
                temp_i = i
                temp_j = j
                k = j + 1
                while k < size:
                    if arr[temp_i] + arr[temp_j] == arr[k]:
                        temp_ans += 1
                        temp_i = temp_j
                        temp_j = k
                    k += 1
                if temp_ans > ans:
                    ans = temp_ans

        if ans > 0:
            return ans + 2
        else:
            return ans
```

毫无意外的，超出时间限制了。

那么我们怎么来优化呢？

**2. 使用哈希表优化做法：**

我们注意到：对于 `arr[i]`、`arr[j]`，要查找的元素 `arr[i] + arr[j]` 是否在 `arr` 中，我们可以预先建立一个反向的哈希表。键值对关系为 `value : idx`，这样就能在 `O(1)` 的时间复杂度通过 `arr[i] + arr[j]` 的值查找到对应的 `k` 值，而不用像原先一样线性查找 `arr[k]` 了。

使用哈希表优化之后的代码如下：

```python
class Solution:
    def lenLongestFibSubseq(self, arr: List[int]) -> int:
        size = len(arr)
        ans = 0
        idx_map = dict()
        for idx, value in enumerate(arr):
            idx_map[value] = idx
        
        for i in range(size):
            for j in range(i + 1, size):
                temp_ans = 0
                temp_i = i
                temp_j = j
                while arr[temp_i] + arr[temp_j] in idx_map:
                    temp_ans += 1
                    k = idx_map[arr[temp_i] + arr[temp_j]]
                    temp_i = temp_j
                    temp_j = k

                if temp_ans > ans:
                    ans = temp_ans

        if ans > 0:
            return ans + 2
        else:
            return ans
```

再次提交，通过了。

但是，这道题我们还可以用动态规划来做。

**3. 动态规划做法：**

这道题用动态规划来做，难点在于如何「定义状态」和「定义状态转移方程」。

- 定义状态：`dp[i][j]` 表示以 `arr[i]`、`arr[j]` 为结尾的斐波那契式子序列的最大长度。
- 定义状态转移方程：$dp[j][k] = max_{(arr[i] + arr[j] = arr[k], i < j < k)}(dp[i][j] + 1)$
    - 意思为：以 `arr[j]`、`arr[k]` 结尾的斐波那契式子序列的最大长度 = 满足 `arr[i] + arr[j] = arr[k]` 条件下，以 `arr[i]`、`arr[j]` 结尾的斐波那契式子序列的最大长度 + 1。

但是直接这样做其实跟 **1. 暴力解法** 一样仍会超时，所以我们依旧采用哈希表优化的方式来提高效率，降低算法的时间复杂度。

具体代码如下：

## 代码

```python
class Solution:
    def lenLongestFibSubseq(self, arr: List[int]) -> int:
        size = len(arr)
        # 初始化 dp
        dp = [[0 for _ in range(size)] for _ in range(size)]
        ans = 0
        idx_map = {}
        # 将 value : idx 映射为哈希表，这样可以快速通过 value 获取到 idx
        for idx, value in enumerate(arr):
            idx_map[value] = idx

        for i in range(size):
            for j in range(i + 1, size):
                if arr[i] + arr[j] in idx_map:
                    # 获取 arr[i] + arr[j] 的 idx，即斐波那契式子序列下一项元素
                    k = idx_map[arr[i] + arr[j]]

                    dp[j][k] = max(dp[j][k], dp[i][j] + 1)
                    ans = max(ans, dp[j][k])

        if ans > 0:
            return ans + 2
        else:
            return ans
```

# [剑指 Offer II 095. 最长公共子序列](https://leetcode.cn/problems/qJnOS7/)

- 标签：字符串、动态规划
- 难度：中等

## 题目链接

- [剑指 Offer II 095. 最长公共子序列 - 力扣](https://leetcode.cn/problems/qJnOS7/)

## 题目大意

给定两个字符串 `text1` 和 `text2`。

要求：返回两个字符串的最长公共子序列的长度。如果不存在公共子序列，则返回 `0`。

- 子序列：原字符串在不改变字符的相对顺序的情况下删除某些字符（也可以不删除任何字符）后组成的新字符串。
- 公共子序列：两个字符串所共同拥有的子序列。

## 解题思路

用动态规划来做。

动态规划的状态 `dp[i][j]` 表示为：前 `i` 个字符组成的字符串 `str1` 与前 `j` 个字符组成的字符串 `str2` 的最长公共子序列长度为 `dp[i][j]`。

遍历字符串 `text1` 和 `text2`，则状态转移方程为：

- 如果 `text1[i - 1] == text2[j - 1]`，则找到了一个公共元素，则 `dp[i][j] = dp[i - 1][j - 1] + 1`。
- 如果 `text1[i - 1] != text2[j - 1]`，则 `dp[i][j]` 需要考虑两种情况，取其中最大的那种：
    - `text1` 前 `i - 1` 个字符组成的字符串 `str1` 与 `text2` 前 `j` 个字符组成的 `str2` 的最长公共子序列长度，即 `dp[i - 1][j]`。
    - `text1` 前 `i` 个字符组成的字符串 `str1` 与 `text2` 前 `j - 1` 个字符组成的 `str2` 的最长公共子序列长度，即 `dp[i][j - 1]`。

最后输出 `dp[sise1][size2]` 即可，`size1`、`size2` 分别为 `text1`、`text2` 的字符串长度。

## 代码

```python
class Solution:
    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        size1 = len(text1)
        size2 = len(text2)
        dp = [[0 for _ in range(size2 + 1)] for _ in range(size1 + 1)]
        for i in range(1, size1 + 1):
            for j in range(1, size2 + 1):
                if text1[i - 1] == text2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

        return dp[size1][size2]
```

# [剑指 Offer II 097. 子序列的数目](https://leetcode.cn/problems/21dk04/)

- 标签：字符串、动态规划
- 难度：困难

## 题目链接

- [剑指 Offer II 097. 子序列的数目 - 力扣](https://leetcode.cn/problems/21dk04/)

## 题目大意

给定两个字符串 `s` 和 `t`。

要求：计算在 `s` 的子序列中 `t` 出现的个数。

## 解题思路

动态规划求解。

定义状态 `dp[i][j]`表示为：以 `i - 1` 为结尾的 `s` 子序列中出现以 `j - 1` 为结尾的 `t` 的个数。

则状态转移方程为：

- 如果 `s[i - 1] == t[j - 1]`，则：`dp[i][j] = dp[i - 1][j - 1] + dp[i - 1][j]`。即 `dp[i][j]` 来源于两部分：
    - 使用 `s[i - 1]` 匹配 `t[j - 1]`，则 `dp[i][j]` 取源于以 `i - 2` 为结尾的 `s` 子序列中出现以 `j - 2` 为结尾的 `t` 的个数，即 `dp[i - 1][j - 1]`。
    - 不使用 `s[i - 1]` 匹配 `t[j - 1]`，则 `dp[i][j]` 取源于以 `i - 2` 为结尾的 `s` 子序列中出现以 `j - 1` 为结尾的 `t` 的个数，即 `dp[i - 1][j]`。
- 如果 `s[i - 1] != t[j - 1]`，那么肯定不能用 `s[i - 1]` 匹配 `t[j - 1]`，则 `dp[i][j]` 取源于 `dp[i - 1][j]`。

下面来看看初始化：

- `dp[i][0]` 表示以 `i - 1` 为结尾的 `s` 子序列中出现空字符串的个数。把 `s` 中的元素全删除，出现空字符串的个数就是 `1`，则 `dp[i][0] = 1`。
- `dp[0][j]` 表示空字符串中出现以 `j - 1` 结尾的 `t` 的个数，空字符串无论怎么变都不会变成 `t`，则 `dp[0][j] = 0`
- `dp[0][0]` 表示空字符串中出现空字符串的个数，这个应该是 `1`，即 `dp[0][0] = 1`。

然后递推求解，最后输出 `dp[size_s][size_t]`。

## 代码

```python
class Solution:
    def numDistinct(self, s: str, t: str) -> int:
        size_s = len(s)
        size_t = len(t)
        dp = [[0 for _ in range(size_t + 1)] for _ in range(size_s + 1)]
        for i in range(size_s):
            dp[i][0] = 1
        for i in range(1, size_s + 1):
            for j in range(1, size_t + 1):
                if s[i - 1] == t[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + dp[i - 1][j]
                else:
                    dp[i][j] = dp[i - 1][j]
        return dp[size_s][size_t]
```

# [剑指 Offer II 098. 路径的数目](https://leetcode.cn/problems/2AoeFn/)

- 标签：数学、动态规划、组合数学
- 难度：中等

## 题目链接

- [剑指 Offer II 098. 路径的数目 - 力扣](https://leetcode.cn/problems/2AoeFn/)

## 题目大意

给定一个 `m * n` 的棋盘， 机器人在左上角的位置，机器人每次只能向右、或者向下移动一步。

要求：求出到达棋盘右下角共有多少条不同的路径。

## 解题思路

可以用动态规划求解，设 `dp[i][j]` 是从 `(0, 0)`到 `(i, j)` 的不同路径数。显然 `dp[i][j] = dp[i-1][j] + dp[i][j-1]`。对于第一行、第一列，因为只能超一个方向走，所以 `dp[i][0] = 1`，`dp[0][j] = 1`。

## 代码

```python
class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        dp = [1 for _ in range(n)]
        for i in range(1, m):
            for j in range(1, n):
                dp[j] += dp[j - 1]
        return dp[-1]
```

# [剑指 Offer II 101. 分割等和子集](https://leetcode.cn/problems/NUPfPr/)

- 标签：数学、字符串、模拟
- 难度：简单

## 题目链接

- [剑指 Offer II 101. 分割等和子集 - 力扣](https://leetcode.cn/problems/NUPfPr/)

## 题目大意

给定一个只包含正整数的非空数组 `nums`。

要求：判断是否可以将这个数组分成两个子集，使得两个子集的元素和相等。

## 解题思路

动态规划求解。

如果两个子集和相等，则两个子集元素和刚好等于整个数组元素和的一半。这就相当于 `0-1` 背包问题。

定义 `dp[i][j]` 表示从 `[0, i]` 个数中任意选取一些数，放进容量为 j 的背包中，价值总和最大为多少。则 `dp[i][j] = max(dp[i - 1][j], dp[i - 1][j - nums[i]] + nums[i])`。

转换为一维 dp 就是：`dp[j] = max(dp[j], dp[j - nums[i]] + nums[i])`。

然后进行递归求解。最后判断 `dp[target]` 和 `target` 是否相等即可。

## 代码

```python
class Solution:
    def canPartition(self, nums: List[int]) -> bool:
        size = 100010
        dp = [0 for _ in range(size)]
        sum_nums = sum(nums)
        if sum_nums & 1:
            return False
        target = sum_nums // 2
        for i in range(len(nums)):
            for j in range(target, nums[i] - 1, -1):
                dp[j] = max(dp[j], dp[j - nums[i]] + nums[i])

        if dp[target] == target:
            return True
        return False
```

# [剑指 Offer II 102. 加减的目标值](https://leetcode.cn/problems/YaVDxD/)

- 标签：数组、动态规划、回溯
- 难度：中等

## 题目链接

- [剑指 Offer II 102. 加减的目标值 - 力扣](https://leetcode.cn/problems/YaVDxD/)

## 题目大意

给定一个整数数组 `nums` 和一个整数 `target`。数组长度不超过 `20`。向数组中每个整数前加 `+` 或 `-`。然后串联起来构造成一个表达式。

要求：返回通过上述方法构造的、运算结果等于 `target` 的不同表达式数目。

## 解题思路

暴力方法就是使用深度优先搜索对每位数字遍历 `+`、`-`，并统计符合要求的表达式数目。但是实际发现超时了。所以采用动态规划的方法来做。

假设数组中所有元素和为 `sum`，数组中所有符号为 `+` 的元素为 `sum_x`，符号为 `-` 的元素和为 `sum_y`。则 `target = sum_x - sum_y`。

而 `sum_x + sum_y = sum`。根据两个式子可以求出 `2 * sum_x = target + sum `，即 `sum_x = (target + sum) / 2`。

那么这道题就变成了，如何在数组中找到一个集合，使集合中元素和为 `(target + sum) / 2`。这就变为了求容量为 `(target + sum) / 2` 的 `01` 背包问题。

动态规划的状态 `dp[i]` 表示为：填满容量为 `i` 的背包，有 `dp[i]` 种方法。

动态规划的状态转移方程为：`dp[i] = dp[i] + dp[i-num]`，意思为填满容量为 `i` 的背包的方法数 = 不使用当前 `num`，只使用之前元素填满容量为 `i` 的背包的方法数 + 填满容量 `i - num` 的包的方法数，再填入 `num` 的方法数。

## 代码

```python
class Solution:
    def findTargetSumWays(self, nums: List[int], target: int) -> int:
        sum_nums = sum(nums)
        if target > sum_nums or (target + sum_nums) % 2 == 1:
            return 0
        size = (target + sum_nums) // 2
        dp = [0 for _ in range(size + 1)]
        dp[0] = 1
        for num in nums:
            for i in range(size, num - 1, -1):
                dp[i] = dp[i] + dp[i - num]
        return dp[size]
```

# [剑指 Offer II 103. 最少的硬币数目](https://leetcode.cn/problems/gaM7Ch/)

- 标签：广度优先搜索、数组、动态规划
- 难度：中等

## 题目链接

- [剑指 Offer II 103. 最少的硬币数目 - 力扣](https://leetcode.cn/problems/gaM7Ch/)

## 题目大意

给定不同面额的硬币 `coins` 和一个总金额 `amount`。

乔秋：计算出凑成总金额所需的最少的硬币个数。如果无法凑出，则返回 `-1`。

## 解题思路

完全背包问题。

可以转换为有 `n` 枚不同的硬币，每种硬币可以无限次使用。凑成总金额为 `amount` 的背包，最少需要多少硬币。

动态规划的状态 `dp[i]` 可以表示为：凑成总金额为 `i` 的组合中，至少有 `dp[i]` 枚硬币。

动态规划的状态转移方程为：`dp[i] = min(dp[i], + dp[i-coin] + 1`，意思为凑成总金额为 `i` 最少硬币数量 = 「不使用当前 `coin`，只使用之前硬币凑成金额 `i` 的最少硬币数量」和「凑成金额 `i - num` 的最少硬币数量，再加上当前硬币」两者的较小值。

## 代码

```python
class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        dp = [float('inf') for _ in range(amount + 1)]
        dp[0] = 0

        for coin in coins:
            for i in range(coin, amount + 1):
                dp[i] = min(dp[i], dp[i - coin] + 1)

        if dp[amount] != float('inf'):
            return dp[amount]
        else:
            return -1
```

# [剑指 Offer II 104. 排列的数目](https://leetcode.cn/problems/D0F0SV/)

- 标签：数组、动态规划
- 难度：中等

## 题目链接

- [剑指 Offer II 104. 排列的数目 - 力扣](https://leetcode.cn/problems/D0F0SV/)

## 题目大意

给定一个由不同整数组成的数组 `nums` 和一个目标整数 `target`。

要求：从 `nums` 中找出并返回总和为 `target` 的元素组合个数。

## 解题思路

完全背包问题。题目求解的是组合数。

动态规划的状态 `dp[i]` 可以表示为：凑成总和 `i` 的组合数。

动态规划的状态转移方程为：`dp[i] = dp[i] + dp[i - nums[j]]`，意思为凑成总和为 `i` 的组合数 = 「不使用当前 `nums[j]`，只使用之前整数凑成和为 `i` 的组合数」+「使用当前 `nums[j]` 凑成金额 `i - nums[j]` 的方案数」。

最终输出 `dp[target]`。

## 代码

```python
class Solution:
    def combinationSum4(self, nums: List[int], target: int) -> int:
        dp = [0 for _ in range(target + 1)]
        dp[0] = 1
        size = len(nums)
        for i in range(target + 1):
            for j in range(size):
                if i - nums[j] >= 0:
                    dp[i] += dp[i - nums[j]]
        return dp[target]
```

# [剑指 Offer II 105. 岛屿的最大面积](https://leetcode.cn/problems/ZL6zAn/)

- 标签：深度优先搜索、广度优先搜索、并查集、数组、矩阵
- 难度：中等

## 题目链接

- [剑指 Offer II 105. 岛屿的最大面积 - 力扣](https://leetcode.cn/problems/ZL6zAn/)

## 题目大意

给定一个只包含 `0`、`1` 元素的二维数组，`1` 代表岛屿，`0` 代表水。一座岛的面积就是上下左右相邻相邻的 `1` 所组成的连通块的数目。找到最大的岛屿面积。

## 解题思路

使用深度优先搜索方法。遍历二维数组的每一个元素，对于每个值为 `1` 的元素，记下其面积。然后将该值置为 `0`（防止二次重复计算），再递归其上下左右四个位置，并将深度优先搜索搜到的值为 `1` 的元素个数，进行累积统计。

## 代码

```python
class Solution:
    def dfs(self, grid, i, j):
        size_n = len(grid)
        size_m = len(grid[0])
        if i < 0 or i >= size_n or j < 0 or j >= size_m or grid[i][j] == 0:
            return 0
        ans = 1
        grid[i][j] = 0
        ans += self.dfs(grid, i + 1, j)
        ans += self.dfs(grid, i, j + 1)
        ans += self.dfs(grid, i - 1, j)
        ans += self.dfs(grid, i, j - 1)
        return ans

    def maxAreaOfIsland(self, grid: List[List[int]]) -> int:
        ans = 0
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] == 1:
                    ans = max(ans, self.dfs(grid, i, j))
        return ans
```

# [剑指 Offer II 106. 二分图](https://leetcode.cn/problems/vEAB3K/)

- 标签：深度优先搜索、广度优先搜索、并查集、图
- 难度：中等

## 题目链接

- [剑指 Offer II 106. 二分图 - 力扣](https://leetcode.cn/problems/vEAB3K/)

## 题目大意

给定一个代表 n 个节点的无向图的二维数组 `graph`，其中 `graph[u]` 是一个节点数组，由节点 `u` 的邻接节点组成。对于 `graph[u]` 中的每个 `v`，都存在一条位于节点 `u` 和节点 `v` 之间的无向边。

该无向图具有以下属性：

- 不存在自环（`graph[u]` 不包含 `u`）。
- 不存在平行边（`graph[u]` 不包含重复值）。
- 如果 `v` 在 `graph[u]` 内，那么 `u` 也应该在 `graph[v]` 内（该图是无向图）。
- 这个图可能不是连通图，也就是说两个节点 `u` 和 `v` 之间可能不存在一条连通彼此的路径。

要求：判断该图是否是二分图，如果是二分图，则返回 `True`；否则返回 `False`。

- 二分图：如果能将一个图的节点集合分割成两个独立的子集 `A` 和 `B`，并使图中的每一条边的两个节点一个来自 `A` 集合，一个来自 `B` 集合，就将这个图称为 二分图 。

## 解题思路

对于图中的任意节点 `u` 和 `v`，如果 `u` 和 `v` 之间有一条无向边，那么 `u` 和 `v` 必然属于不同的集合。

我们可以通过在深度优先搜索中对邻接点染色标记的方式，来识别该图是否是二分图。具体做法如下：

- 找到一个没有染色的节点 `u`，将其染成红色。
- 然后遍历该节点直接相连的节点 `v`，如果该节点没有被染色，则将该节点直接相连的节点染成蓝色，表示两个节点不是同一集合。如果该节点已经被染色并且颜色跟 `u` 一样，则说明该图不是二分图，直接返回 `False`。
- 从上面染成蓝色的节点 `v` 出发，遍历该节点直接相连的节点。。。依次类推的递归下去。
- 如果所有节点都顺利染上色，则说明该图为二分图，返回 `True`。否则，如果在途中不能顺利染色，则返回 `False`。

## 代码

```python
class Solution:
    def dfs(self, graph, colors, i, color):
        colors[i] = color
        for j in graph[i]:
            if colors[j] == colors[i]:
                return False
            if colors[j] == 0 and not self.dfs(graph, colors, j, -color):
                return False
        return True

    def isBipartite(self, graph: List[List[int]]) -> bool:
        size = len(graph)
        colors = [0 for _ in range(size)]
        for i in range(size):
            if colors[i] == 0 and not self.dfs(graph, colors, i, 1):
                return False
        return True
```

# [剑指 Offer II 107. 矩阵中的距离](https://leetcode.cn/problems/2bCMpM/)

- 标签：广度优先搜索、数组、动态规划、矩阵
- 难度：中等

## 题目链接

- [剑指 Offer II 107. 矩阵中的距离 - 力扣](https://leetcode.cn/problems/2bCMpM/)

## 题目大意

给定一个由 `0` 和 `1` 组成的矩阵，两个相邻元素间的距离为 `1` 。

要求：找出每个元素到最近的 `0` 的距离，并输出为矩阵。

## 解题思路

题目要求的是每个 `1` 到 `0`的最短曼哈顿距离。换句话也可以求每个 `0` 到 `1` 的最短曼哈顿距离。这样做的好处是，可以从所有值为 `0` 的元素开始进行搜索，可以不断累积距离，直到遇到值为 `1` 的元素时，可以直接将累积距离直接赋值。

具体操作如下：将所有值为 `0` 的元素坐标加入访问集合中，对所有值为`0` 的元素上下左右进行搜索。每进行一次上下左右搜索，更新新位置的距离值，并把新的位置坐标加入队列和访问集合中，直到遇见值为 `1` 的元素停止搜索。

## 代码

```python
class Solution:
    def updateMatrix(self, mat: List[List[int]]) -> List[List[int]]:
        row_count = len(mat)
        col_count = len(mat[0])
        dist_map = [[0 for _ in range(col_count)] for _ in range(row_count)]
        zeroes_pos = []
        for i in range(row_count):
            for j in range(col_count):
                if mat[i][j] == 0:
                    zeroes_pos.append((i, j))

        directions = {(1, 0), (-1, 0), (0, 1), (0, -1)}
        queue = collections.deque(zeroes_pos)
        visited = set(zeroes_pos)

        while queue:
            i, j = queue.popleft()
            for direction in directions:
                new_i = i + direction[0]
                new_j = j + direction[1]
                if 0 <= new_i < row_count and 0 <= new_j < col_count and (new_i, new_j) not in visited:
                    dist_map[new_i][new_j] = dist_map[i][j] + 1
                    queue.append((new_i, new_j))
                    visited.add((new_i, new_j))
        return dist_map
```

# [剑指 Offer II 108. 单词演变](https://leetcode.cn/problems/om3reC/)

- 标签：广度优先搜索、哈希表、字符串
- 难度：困难

## 题目链接

- [剑指 Offer II 108. 单词演变 - 力扣](https://leetcode.cn/problems/om3reC/)

## 题目大意

给定两个单词 `beginWord` 和 `endWord`，以及一个字典 `wordList`。找到从 `beginWord` 到 `endWord` 的最短转换序列中的单词数目。如果不存在这样的转换序列，则返回 0。

转换需要遵守的规则如下：

- 每次转换只能改变一个字母。
- 转换过程中的中间单词必须为字典中的单词。

## 解题思路

广度优先搜索。使用队列存储将要遍历的单词和单词数目。

从 `beginWord` 开始变换，把单词的每个字母都用 `a ~ z` 变换一次，变换后的单词是否是 `endWord`，如果是则直接返回。

否则查找变换后的词是否在 `wordList` 中。如果在 `wordList` 中找到就加入队列，找不到就输出 `0`。然后按照广度优先搜索的算法急需要遍历队列中的节点，直到所有单词都出队时结束。

## 代码

```python
class Solution:
    def ladderLength(self, beginWord: str, endWord: str, wordList: List[str]) -> int:
        if not wordList or endWord not in wordList:
            return 0
        word_set = set(wordList)
        if beginWord in word_set:
            word_set.remove(beginWord)

        queue = collections.deque()
        queue.append((beginWord, 1))
        while queue:
            word, level = queue.popleft()
            if word == endWord:
                return level

            for i in range(len(word)):
                for j in range(26):
                    new_word = word[:i] + chr(ord('a') + j) + word[i + 1:]
                    if new_word in word_set:
                        word_set.remove(new_word)
                        queue.append((new_word, level + 1))

        return 0
```

# [剑指 Offer II 109. 开密码锁](https://leetcode.cn/problems/zlDJc7/)

- 标签：广度优先搜索、数组、哈希表、字符串
- 难度：中等

## 题目链接

- [剑指 Offer II 109. 开密码锁 - 力扣](https://leetcode.cn/problems/zlDJc7/)

## 题目大意

有一把带有四个数字的密码锁，每个位置上有 0~9 共 10 个数字。每次只能将其中一个位置上的数字转动一下。可以向上转，也可以向下转。比如：1 -> 2、2 -> 1。

密码锁的初始数字为：`0000`。现在给定一组表示死亡数字的字符串数组 `deadends`，和一个带有四位数字的目标字符串 `target`。

如果密码锁转动到 `deadends` 中任一字符串状态，则锁就会永久锁定，无法再次旋转。

要求：求出最小的选择次数，使得锁的状态由 `0000` 转动到 `target`。

## 解题思路

使用宽度优先搜索遍历，将`0000` 状态入队。

- 将队列中的元素出队，判断是否为死亡字符串
- 如果为死亡字符串，则跳过该状态，否则继续执行。

- 如果为目标字符串，则返回当前路径长度，否则继续执行。
- 枚举当前状态所有位置所能到达的所有状态，并判断是否访问过该状态。

- 如果之前出现过该状态，则继续执行，否则将其存入队列，并标记访问。

## 代码

```python
class Solution:
    def openLock(self, deadends: List[str], target: str) -> int:
        queue = collections.deque(['0000'])
        visited = set(['0000'])
        deadset = set(deadends)
        level = 0
        while queue:
            size = len(queue)
            for _ in range(size):
                cur = queue.popleft()
                if cur in deadset:
                    continue
                if cur == target:
                    return level
                for i in range(len(cur)):
                    up = self.upward_adjust(cur, i)
                    if up not in visited:
                        queue.append(up)
                        visited.add(up)
                    down = self.downward_adjust(cur, i)
                    if down not in visited:
                        queue.append(down)
                        visited.add(down)
            level += 1
        return -1

    def upward_adjust(self, s, i):
        s_list = list(s)
        if s_list[i] == '9':
            s_list[i] = '0'
        else:
            s_list[i] = chr(ord(s_list[i]) + 1)
        return "".join(s_list)

    def downward_adjust(self, s, i):
        s_list = list(s)
        if s_list[i] == '0':
            s_list[i] = '9'
        else:
            s_list[i] = chr(ord(s_list[i]) - 1)
        return "".join(s_list)
```

# [剑指 Offer II 111. 计算除法](https://leetcode.cn/problems/vlzXQL/)

- 标签：深度优先搜索、广度优先搜索、并查集、图、数组、最短路
- 难度：中等

## 题目链接

- [剑指 Offer II 111. 计算除法 - 力扣](https://leetcode.cn/problems/vlzXQL/)

## 题目大意

给定一个变量对数组 `equations` 和一个实数数组 `values` 作为已知条件，其中 `equations[i] = [Ai, Bi]`  和 `values[i]` 共同表示 `Ai / Bi = values[i]`。每个 `Ai` 或 `Bi` 是一个表示单个变量的字符串。

再给定一个表示多个问题的数组 `queries`，其中 `queries[j] = [Cj, Dj]` 表示第 `j` 个问题，要求：根据已知条件找出 `Cj / Dj = ?` 的结果作为答案。返回所有问题的答案。如果某个答案无法确定，则用 `-1.0` 代替，如果问题中出现了给定的已知条件中没有出现的表示变量的字符串，则也用 `-1.0` 代替这个答案。

## 解题思路

在「[等式方程的可满足性](https://leetcode.cn/problems/satisfiability-of-equality-equations)」的基础上增加了倍数关系。在「[等式方程的可满足性](https://leetcode.cn/problems/satisfiability-of-equality-equations)」中我们处理传递关系使用了并查集，这道题也是一样，不过在使用并查集的同时还要维护倍数关系。

举例说明：

- `a / b = 2.0`：说明 `a = 2b`，`a` 和 `b` 在同一个集合。
- `b / c = 3.0`：说明 `b = 3c`，`b`  和 `c`  在同一个集合。

根据上述两式可得：`a`、`b`、`c` 都在一个集合中，且 `a = 2b = 6c`。

我们可以将同一集合中的变量倍数关系都转换为与根节点变量的倍数关系，比如上述例子中都转变为与 `a` 的倍数关系。

具体操作如下：

- 定义并查集结构，并在并查集中定义一个表示倍数关系的 `multiples` 数组。
- 遍历 `equations` 数组、`values` 数组，将每个变量按顺序编号，并使用 `union` 将其并入相同集合。
- 遍历 `queries` 数组，判断两个变量是否在并查集中，并且是否在同一集合。如果找到对应关系，则将计算后的倍数关系存入答案数组，否则则将 `-1` 存入答案数组。
- 最终输出答案数组。

并查集中维护倍数相关方法说明：

- `find` 方法： 
    - 递推寻找根节点，并将倍数累乘，然后进行路径压缩，并且更新当前节点的倍数关系。
- `union` 方法：
    - 如果两个节点属于同一集合，则直接返回。
    - 如果两个节点不属于同一个集合，合并之前当前节点的倍数关系更新，然后再进行更新。
- `is_connect` 方法：
    - 如果两个节点不属于同一集合，返回 `-1`。
    - 如果两个节点属于同一集合，则返回倍数关系。

## 代码

```python
class UnionFind:

    def __init__(self, n):
        self.parent = [i for i in range(n)]
        self.multiples = [1 for _ in range(n)]

    def find(self, x):
        multiple = 1.0
        origin = x
        while x != self.parent[x]:
            multiple *= self.multiples[x]
            x = self.parent[x]
        self.parent[origin] = x
        self.multiples[origin] = multiple
        return x

    def union(self, x, y, multiple):
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x == root_y:
            return
        self.parent[root_x] = root_y
        self.multiples[root_x] = multiple * self.multiples[y] / self.multiples[x]
        return

    def is_connected(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x != root_y:
            return -1.0

        return self.multiples[x] / self.multiples[y]

class Solution:
    def calcEquation(self, equations: List[List[str]], values: List[float], queries: List[List[str]]) -> List[float]:
        equations_size = len(equations)
        hash_map = dict()
        union_find = UnionFind(2 * equations_size)

        id = 0
        for i in range(equations_size):
            equation = equations[i]
            var1, var2 = equation[0], equation[1]
            if var1 not in hash_map:
                hash_map[var1] = id
                id += 1
            if var2 not in hash_map:
                hash_map[var2] = id
                id += 1
            union_find.union(hash_map[var1], hash_map[var2], values[i])

        queries_size = len(queries)
        res = []
        for i in range(queries_size):
            query = queries[i]
            var1, var2 = query[0], query[1]
            if var1 not in hash_map or var2 not in hash_map:
                res.append(-1.0)
            else:
                id1 = hash_map[var1]
                id2 = hash_map[var2]
                res.append(union_find.is_connected(id1, id2))

        return res
```

# [剑指 Offer II 112. 最长递增路径](https://leetcode.cn/problems/fpTFWP/)

- 标签：深度优先搜索、广度优先搜索、图、拓扑排序、记忆化搜索、数组、动态规划、矩阵
- 难度：困难

## 题目链接

- [剑指 Offer II 112. 最长递增路径 - 力扣](https://leetcode.cn/problems/fpTFWP/)

## 题目大意

给定一个 `m * n` 大小的整数矩阵 `matrix`。要求：找出其中最长递增路径的长度。

对于每个单元格，可以往上、下、左、右四个方向移动，不能向对角线方向移动或移动到边界外。

## 解题思路

深度优先搜索。使用二维数组 `record` 存储遍历过的单元格最大路径长度，已经遍历过的单元格就不需要再次遍历了。

## 代码

```python
class Solution:
    max_len = 0
    directions = {(1, 0), (-1, 0), (0, 1), (0, -1)}

    def longestIncreasingPath(self, matrix: List[List[int]]) -> int:
        if not matrix:
            return 0
        rows, cols = len(matrix), len(matrix[0])
        record = [[0 for _ in range(cols)] for _ in range(rows)]

        def dfs(i, j):
            record[i][j] = 1
            for direction in self.directions:
                new_i, new_j = i + direction[0], j + direction[1]
                if 0 <= new_i < rows and 0 <= new_j < cols and matrix[new_i][new_j] > matrix[i][j]:
                    if record[new_i][new_j] == 0:
                        dfs(new_i, new_j)
                    record[i][j] = max(record[i][j], record[new_i][new_j] + 1)
            self.max_len = max(self.max_len, record[i][j])

        for i in range(rows):
            for j in range(cols):
                if record[i][j] == 0:
                    dfs(i, j)
        return self.max_len
```

# [剑指 Offer II 113. 课程顺序](https://leetcode.cn/problems/QA2IGt/)

- 标签：深度优先搜索、广度优先搜索、图、拓扑排序
- 难度：中等

## 题目链接

- [剑指 Offer II 113. 课程顺序 - 力扣](https://leetcode.cn/problems/QA2IGt/)

## 题目大意

给定一个整数 `numCourses`，代表这学期必须选修的课程数量，课程编号为 `0` 到 `numCourses - 1`。再给定一个数组 `prerequisites` 表示先修课程关系，其中 `prerequisites[i] = [ai, bi]` 表示如果要学习课程 `ai` 则必须要学习课程 `bi`。

要求：返回学完所有课程所安排的学习顺序。如果有多个正确的顺序，只要返回其中一种即可。如果无法完成所有课程，则返回空数组。

## 解题思路

拓扑排序。这道题是「[0207. 课程表](https://leetcode.cn/problems/course-schedule/)」的升级版，只需要在上一题的基础上增加一个答案数组即可。

1. 使用列表 `edges` 存放课程关系图，并统计每门课程节点的入度，存入入度列表 `indegrees`。

2. 借助队列 `queue`，将所有入度为 `0` 的节点入队。

3. 从队列中选择一个节点，并将其加入到答案数组 `res` 中，再让课程数 -1。
4. 将该顶点以及该顶点为出发点的所有边的另一个节点入度 -1。如果入度 -1 后的节点入度不为 `0`，则将其加入队列 `queue`。
5. 重复 3~4 的步骤，直到队列中没有节点。
6. 最后判断剩余课程数是否为 `0`，如果为 `0`，则返回答案数组 `res`，否则，返回空数组。

## 代码

```python
class Solution:
    def findOrder(self, numCourses: int, prerequisites: List[List[int]]) -> List[int]:
        indegrees = [0 for _ in range(numCourses)]
        edges = collections.defaultdict(list)
        res = []
        for x, y in prerequisites:
            edges[y].append(x)
            indegrees[x] += 1
        queue = collections.deque([])
        for i in range(numCourses):
            if not indegrees[i]:
                queue.append(i)
        while queue:
            y = queue.popleft()
            res.append(y)
            numCourses -= 1
            for x in edges[y]:
                indegrees[x] -= 1
                if not indegrees[x]:
                    queue.append(x)
        if not numCourses:
            return res
        else:
            return []
```

# [剑指 Offer II 116. 省份数量](https://leetcode.cn/problems/bLyHh0/)

- 标签：深度优先搜索、广度优先搜索、并查集、图
- 难度：中等

## 题目链接

- [剑指 Offer II 116. 省份数量 - 力扣](https://leetcode.cn/problems/bLyHh0/)

## 题目大意

一个班上有 `n` 个同学，其中一些彼此是朋友，另一些不是。如果 `a` 与 `b` 是直接朋友，且 `b` 与 `c` 也是直接朋友，那么 `a` 与 `c` 是间接朋友。

现在定义「朋友圈」是由一组直接或间接朋友组成的集合。

现在给定一个 `n * n` 的矩阵 `isConnected` 表示班上的朋友关系。其中 `isConnected[i][j] = 1` 表示第 `i` 个同学和第 `j` 个同学是直接朋友，`isConnected[i][j] = 0` 表示第 `i` 个同学和第 `j` 个同学不是直接朋友。

要求：根据给定的同学关系，返回「朋友圈」的数量。

## 解题思路

可以利用并查集来做。具体做法如下：

遍历矩阵 `isConnected`。如果 `isConnected[i][j] = 1`，将 `i` 节点和 `j` 节点相连。然后判断每个同学节点的根节点，然后统计不重复的根节点有多少个，即为「朋友圈」的数量。

## 代码

```python
class UnionFind:

    def __init__(self, n):
        self.parent = [i for i in range(n)]
        self.count = n

    def find(self, x):
        while x != self.parent[x]:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x == root_y:
            return

        self.parent[root_x] = root_y
        self.count -= 1

    def is_connected(self, x, y):
        return self.find(x) == self.find(y)

class Solution:
    def findCircleNum(self, isConnected: List[List[int]]) -> int:
        size = len(isConnected)
        union_find = UnionFind(size)
        for i in range(size):
            for j in range(i + 1, size):
                if isConnected[i][j] == 1:
                    union_find.union(i, j)

        return union_find.count
```

# [剑指 Offer II 118. 多余的边](https://leetcode.cn/problems/7LpjUW/)

- 标签：深度优先搜索、广度优先搜索、并查集、图
- 难度：中等

## 题目链接

- [剑指 Offer II 118. 多余的边 - 力扣](https://leetcode.cn/problems/7LpjUW/)

## 题目大意

一个 `n` 个节点的树（节点值为 `1~n`）添加一条边后就形成了图，添加的这条边不属于树中已经存在的边。图的信息记录存储与长度为 `n` 的二维数组 `edges`，`edges[i] = [ai, bi]` 表示图中在 `ai` 和 `bi` 之间存在一条边。

现在给定代表边信息的二维数组 `edges`。

要求：找到一条可以山区的边，使得删除后的剩余部分是一个有着 `n` 个节点的树。如果有多个答案，则返回数组 `edges` 中最后出现的边。

## 解题思路

树可以看做是无环的图，这道题就是要找出那条添加边之后成环的边。可以考虑用并查集来做。

从前向后遍历每一条边，如果边的两个节点不在同一个集合，就加入到一个集合（链接到同一个根节点）。如果边的节点已经出现在同一个集合里，说明边的两个节点已经连在一起了，再加入这条边一定会出现环，则这条边就是所求答案。

## 代码

```python
class UnionFind:

    def __init__(self, n):
        self.parent = [i for i in range(n)]

    def find(self, x):
        while x != self.parent[x]:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)
        self.parent[root_x] = root_y

    def is_connected(self, x, y):
        return self.find(x) == self.find(y)

class Solution:
    def findRedundantConnection(self, edges: List[List[int]]) -> List[int]:
        size = len(edges)
        union_find = UnionFind(size + 1)

        for edge in edges:
            if union_find.is_connected(edge[0], edge[1]):
                return edge
            union_find.union(edge[0], edge[1])

        return None
```

# [剑指 Offer II 119. 最长连续序列](https://leetcode.cn/problems/WhsWhI/)

- 标签：并查集、数组、哈希表
- 难度：中等

## 题目链接

- [剑指 Offer II 119. 最长连续序列 - 力扣](https://leetcode.cn/problems/WhsWhI/)

## 题目大意

给定一个未排序的整数数组 `nums`。

要求：找出数字连续的最长序列（不要求序列元素在原数组中连续）的长度。并且要用时间复杂度为 $O(n)$ 的算法解决此问题。

## 解题思路

暴力做法有两种思路。第 1 种思路是先排序再依次判断，这种做法时间复杂度最少是 $O(n \log n)$。第 2 种思路是枚举数组中的每个数 `num`，考虑以其为起点，不断尝试匹配 `num + 1`、`num + 2`、`...` 是否存在，最长匹配次数为 `len(nums)`。这样下来时间复杂度为 $O(n^2)$。但是可以使用集合或哈希表优化这个步骤。

- 先将数组存储到集合中进行去重，然后使用 `curr_streak` 维护当前连续序列长度，使用 `ans` 维护最长连续序列长度。
- 遍历集合中的元素，对每个元素进行判断，如果该元素不是序列的开始（即 `num - 1` 在集合中），则跳过。
- 如果 `num - 1` 不在集合中，说明 `num` 是序列的开始，判断 `num + 1` 、`nums + 2`、`...` 是否在哈希表中，并不断更新当前连续序列长度 `curr_streak`。并在遍历结束之后更新最长序列的长度。
- 最后输出最长序列长度。

将数组存储到集合中进行去重的操作的时间复杂度是 $O(n)$。查询每个数是否在集合中的时间复杂度是 $O(1)$ ，并且跳过了所有不是起点的元素。更新当前连续序列长度 `curr_streak` 的时间复杂度是 $O(n)$，所以最终的时间复杂度是 $O(n)$。符合题意要求。

## 代码

```python
class Solution:
    def longestConsecutive(self, nums: List[int]) -> int:
        ans = 0
        nums_set = set(nums)
        for num in nums_set:
            if num - 1 not in nums_set:
                curr_num = num
                curr_streak = 1

                while curr_num + 1 in nums_set:
                    curr_num += 1
                    curr_streak += 1
                ans = max(ans, curr_streak)

        return ans
```

# [面试题 01.07. 旋转矩阵](https://leetcode.cn/problems/rotate-matrix-lcci/)

- 标签：数组、数学、矩阵
- 难度：中等

## 题目链接

- [面试题 01.07. 旋转矩阵 - 力扣](https://leetcode.cn/problems/rotate-matrix-lcci/)

## 题目大意

给定一个 `n * n` 大小的二维矩阵用来表示图像，其中每个像素的大小为 4 字节。

要求：设计一种算法，将图像旋转 90 度。并且要不占用额外内存空间。

## 解题思路

题目要求不占用额外内存空间，就是要在原二维矩阵上直接进行旋转操作。我们可以用翻转操作代替旋转操作。具体可以分为两步：

1. 上下翻转。

2. 主对角线翻转。

举个例子：

```
 1  2  3  4
 5  6  7  8
 9 10 11 12              
13 14 15 16              
```

上下翻转后变为：

```
13 14 15 16
 9 10 11 12
 5  6  7  8
 1  2  3  4 
```

在经过主对角线翻转后变为：

```
13  9  5  1
14 10  6  2
15 11  7  3
16 12  8  4
```

## 代码

```python
class Solution:
    def rotate(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """
        size = len(matrix)
        for i in range(size // 2):
            for j in range(size):
                matrix[i][j], matrix[size - i - 1][j] = matrix[size - i - 1][j], matrix[i][j]
        for i in range(size):
            for j in range(i):
                matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
```

# [面试题 01.08. 零矩阵](https://leetcode.cn/problems/zero-matrix-lcci/)

- 标签：数组、哈希表、矩阵
- 难度：中等

## 题目链接

- [面试题 01.08. 零矩阵 - 力扣](https://leetcode.cn/problems/zero-matrix-lcci/)

## 题目大意

给定一个 `m * n` 大小的二维矩阵 `matrix`。

要求：编写一种算法，如果矩阵中某个元素为 `0`，增将其所在行与列清零。

## 解题思路

直观上可以使用两个数组或者集合来标记行和列出现 `0` 的情况，但更好的做法是不用开辟新的数组或集合，直接原本二维矩阵 `matrix` 的空间。使用数组原本的元素进行记录出现 0 的情况。

设定两个变量 `flag_row0`、`flag_col0` 来标记第一行、第一列是否出现了 `0`。

接下来我们使用数组第一行、第一列来标记 `0` 的情况。

对数组除第一行、第一列之外的每个元素进行遍历，如果某个元素出现 `0` 了，则使用数组的第一行、第一列对应位置来存储 `0` 的标记。

再对数组除第一行、第一列之外的每个元素进行遍历，通过对第一行、第一列的标记 0 情况，进行置为 `0` 的操作。

最后再根据 `flag_row0`、`flag_col0` 的标记情况，对第一行、第一列进行置为 `0` 的操作。

## 代码

```python
class Solution:
    def setZeroes(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """
        rows = len(matrix)
        cols = len(matrix[0])
        flag_col0 = False
        flag_row0 = False
        for i in range(rows):
            if matrix[i][0] == 0:
                flag_col0 = True
                break

        for j in range(cols):
            if matrix[0][j] == 0:
                flag_row0 = True
                break

        for i in range(1, rows):
            for j in range(1, cols):
                if matrix[i][j] == 0:
                    matrix[i][0] = matrix[0][j] = 0

        for i in range(1, rows):
            for j in range(1, cols):
                if matrix[i][0] == 0 or matrix[0][j] == 0:
                    matrix[i][j] = 0

        if flag_col0:
            for i in range(rows):
                matrix[i][0] = 0

        if flag_row0:
            for j in range(cols):
                matrix[0][j] = 0
```

# [面试题 02.02. 返回倒数第 k 个节点](https://leetcode.cn/problems/kth-node-from-end-of-list-lcci/)

- 标签：链表、双指针
- 难度：简单

## 题目链接

- [面试题 02.02. 返回倒数第 k 个节点 - 力扣](https://leetcode.cn/problems/kth-node-from-end-of-list-lcci/)

## 题目大意

给定一个链表的头节点 `head`，以及一个整数 `k`。

要求：返回链表的倒数第 `k` 个节点的值。

## 解题思路

常规思路是遍历一遍链表，求出链表长度，再遍历一遍到对应位置，返回该位置上的节点。

如果用一次遍历实现的话，可以使用快慢指针。让快指针先走 `k` 步，然后快慢指针、慢指针再同时走，每次一步，这样等快指针遍历到链表尾部的时候，慢指针就刚好遍历到了倒数第 `k` 个节点位置。返回该该位置上的节点即可。

## 代码

```python
class Solution:
    def kthToLast(self, head: ListNode, k: int) -> int:
        slow = head
        fast = head
        for _ in range(k):
            if fast == None:
                return fast
            fast = fast.next
        while fast:
            slow = slow.next
            fast = fast.next
        return slow.val
```

# [面试题 02.05. 链表求和](https://leetcode.cn/problems/sum-lists-lcci/)

- 标签：递归、链表、数学
- 难度：中等

## 题目链接

- [面试题 02.05. 链表求和 - 力扣](https://leetcode.cn/problems/sum-lists-lcci/)

## 题目大意

给定两个非空的链表 `l1` 和 `l2`，表示两个非负整数，每位数字都是按照逆序的方式存储的，每个节点存储一位数字。

要求：计算两个整数的和，并逆序返回表示和的链表。

## 解题思路

模拟大数加法，按位相加，将结果添加到新链表上。需要注意进位和对 `10` 取余。

## 代码

```python
class Solution:
    def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
        head = curr = ListNode(0)
        carry = 0
        while l1 or l2 or carry:
            if l1:
                num1 = l1.val
                l1 = l1.next
            else:
                num1 = 0
            if l2:
                num2 = l2.val
                l2 = l2.next
            else:
                num2 = 0

            sum = num1 + num2 + carry
            carry = sum // 10

            curr.next = ListNode(sum % 10)
            curr = curr.next

        return head.next
```

# [面试题 02.06. 回文链表](https://leetcode.cn/problems/palindrome-linked-list-lcci/)

- 标签：栈、递归、链表、双指针
- 难度：简单

## 题目链接

- [面试题 02.06. 回文链表 - 力扣](https://leetcode.cn/problems/palindrome-linked-list-lcci/)

## 题目大意

给定一个链表的头节点 `head`。

要求：判断该链表是否为回文链表。

## 解题思路

利用数组，将链表元素依次存入。然后再使用两个指针，一个指向数组开始位置，一个指向数组结束位置，依次判断首尾对应元素是否相等，若都相等，则为回文链表。若不相等，则不是回文链表。

## 代码

```python
class Solution:
    def isPalindrome(self, head: ListNode) -> bool:
        nodes = []
        p1 = head
        while p1 != None:
            nodes.append(p1.val)
            p1 = p1.next
        return nodes == nodes[::-1]
```

# [面试题 02.07. 链表相交](https://leetcode.cn/problems/intersection-of-two-linked-lists-lcci/)

- 标签：哈希表、链表、双指针
- 难度：简单

## 题目链接

- [面试题 02.07. 链表相交 - 力扣](https://leetcode.cn/problems/intersection-of-two-linked-lists-lcci/)

## 题目大意

给定两个链表的头节点 `headA`、`headB`。

要求：找出并返回两个单链表相交的起始节点。如果两个链表没有交点，返回 `None` 。

比如：链表 A 为 `[4, 1, 8, 4, 5]`，链表 B 为 `[5, 0, 1, 8, 4, 5]`。则如下图所示，两个链表相交的起始节点为 `8`，则输出结果为 `8`。

![](https://assets.leetcode.com/uploads/2018/12/13/160_example_1.png)





## 解题思路

如果两个链表相交，那么从相交位置开始，到结束，必有一段等长且相同的节点。假设链表 `A` 的长度为 `m`、链表 `B` 的长度为 `n`，他们的相交序列有 `k` 个，则相交情况可以如下如所示：

![](https://qcdn.itcharge.cn/images/20210401113538.png)

现在问题是如何找到 `m - k` 或者 `n - k` 的位置。

考虑将链表 `A` 的末尾拼接上链表 `B`，链表 `B` 的末尾拼接上链表 `A`。

然后使用两个指针 `pA` 、`pB`，分别从链表 `A`、链表 `B` 的头节点开始遍历，如果走到共同的节点，则返回该节点。

否则走到两个链表末尾，返回 `None`。

![](https://qcdn.itcharge.cn/images/20210401114100.png)

## 代码

```python
class Solution:
    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> ListNode:
        if headA == None or headB == None:
            return None
        pA = headA
        pB = headB
        while pA != pB :
            pA = pA.next if pA != None else headB
            pB = pB.next if pB != None else headA
        return pA
```

# [面试题 02.08. 环路检测](https://leetcode.cn/problems/linked-list-cycle-lcci/)

- 标签：哈希表、链表、双指针
- 难度：中等

## 题目链接

- [面试题 02.08. 环路检测 - 力扣](https://leetcode.cn/problems/linked-list-cycle-lcci/)

## 题目大意

给定一个链表的头节点 `head`。

要求：判断链表中是否有环，如果有环则返回入环的第一个节点，无环则返回 None。

## 解题思路

利用两个指针，一个慢指针每次前进一步，快指针每次前进两步（两步或多步效果是等价的）。如果两个指针在链表头节点以外的某一节点相遇（即相等）了，那么说明链表有环，否则，如果（快指针）到达了某个没有后继指针的节点时，那么说明没环。

如果有环，则再定义一个指针，和慢指针一起每次移动一步，两个指针相遇的位置即为入口节点。

这是因为：假设入环位置为 A，快慢指针在在 B 点相遇，则相遇时慢指针走了 $a + b$ 步，快指针走了 $a + n(b+c) + b$ 步。

$2(a + b) = a + n(b + c) + b$。可以推出：$a = c + (n-1)(b + c)$。

我们可以发现：从相遇点到入环点的距离 $c$ 加上 $n-1$ 圈的环长 $b + c$ 刚好等于从链表头部到入环点的距离。

## 代码

```python
class Solution:
    def detectCycle(self, head: ListNode) -> ListNode:
        fast, slow = head, head
        while True:
            if not fast or not fast.next:
                return None
            fast = fast.next.next
            slow = slow.next
            if fast == slow:
                break

        ans = head
        while ans != slow:
            ans, slow = ans.next, slow.next
        return ans
```

# [面试题 03.02. 栈的最小值](https://leetcode.cn/problems/min-stack-lcci/)

- 标签：栈、设计
- 难度：简单

## 题目链接

- [面试题 03.02. 栈的最小值 - 力扣](https://leetcode.cn/problems/min-stack-lcci/)

## 题目大意

设计一个「栈」，要求实现  `push` ，`pop` ，`top` ，`getMin` 操作，其中 `getMin` 要求能在常数时间内实现。

## 解题思路

使用一个栈，栈元素中除了保存当前值之外，再保存一个当前最小值。

-  `push` 操作：如果栈不为空，则判断当前值与栈顶元素所保存的最小值，并更新当前最小值，将新元素保存到栈中。
-  `pop`操作：正常出栈
-  `top` 操作：返回栈顶元素保存的值。
-  `getMin` 操作：返回栈顶元素保存的最小值。

## 代码

```python
class MinStack:

    def __init__(self):
        """
        initialize your data structure here.
        """
        self.stack = []

    class Node:
        def __init__(self, x):
            self.val = x
            self.min = x

    def push(self, x: int) -> None:
        node = self.Node(x)
        if len(self.stack) == 0:
            self.stack.append(node)
        else:
            topNode = self.stack[-1]
            if node.min > topNode.min:
                node.min = topNode.min

            self.stack.append(node)

    def pop(self) -> None:
        self.stack.pop()

    def top(self) -> int:
        return self.stack[-1].val

    def getMin(self) -> int:
        return self.stack[-1].min
```

# [面试题 03.04. 化栈为队](https://leetcode.cn/problems/implement-queue-using-stacks-lcci/)

- 标签：栈、设计、队列
- 难度：简单

## 题目链接

- [面试题 03.04. 化栈为队 - 力扣](https://leetcode.cn/problems/implement-queue-using-stacks-lcci/)

## 题目大意

要求：实现一个 MyQueue 类，要求仅使用两个栈实现先入先出队列。

## 解题思路

使用两个栈，`inStack` 用于输入，`outStack` 用于输出。

- `push` 操作：将元素压入 `inStack` 中。
- `pop` 操作：如果 `outStack` 输出栈为空，将 `inStack` 输入栈元素依次取出，按顺序压入 `outStack` 栈。这样 `outStack` 栈的元素顺序和之前 `inStack` 元素顺序相反，`outStack` 顶层元素就是要取出的队头元素，将其移出，并返回该元素。如果 `outStack` 输出栈不为空，则直接取出顶层元素。
- `peek` 操作：和 `pop` 操作类似，只不过最后一步不需要取出顶层元素，直接将其返回即可。
- `empty` 操作：如果 `inStack` 和 `outStack` 都为空，则队列为空，否则队列不为空。

## 代码

```python
class MyQueue:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.inStack = []
        self.outStack = []


    def push(self, x: int) -> None:
        """
        Push element x to the back of queue.
        """
        self.inStack.append(x)


    def pop(self) -> int:
        """
        Removes the element from in front of queue and returns that element.
        """
        if (len(self.outStack) == 0):
            while (len(self.inStack) != 0):
                self.outStack.append(self.inStack[-1])
                self.inStack.pop()
        top = self.outStack[-1]
        self.outStack.pop()
        return top


    def peek(self) -> int:
        """
        Get the front element.
        """
        if (len(self.outStack) == 0):
            while (len(self.inStack) != 0):
                self.outStack.append(self.inStack[-1])
                self.inStack.pop()
        top = self.outStack[-1]
        return top


    def empty(self) -> bool:
        """
        Returns whether the queue is empty.
        """
        return len(self.outStack) == 0 and len(self.inStack) == 0
```

# [面试题 04.02. 最小高度树](https://leetcode.cn/problems/minimum-height-tree-lcci/)

- 标签：树、二叉搜索树、数组、分治、二叉树
- 难度：简单

## 题目链接

- [面试题 04.02. 最小高度树 - 力扣](https://leetcode.cn/problems/minimum-height-tree-lcci/)

## 题目大意

给定一个升序的有序数组 `nums`。

要求：创建一棵高度最小的二叉搜索树（高度平衡的二叉搜索树）。

## 解题思路

直观上，如果把数组的中间元素当做根，那么数组左侧元素都小于根节点，右侧元素都大于根节点，且左右两侧元素个数相同，或最多相差 `1` 个。那么构建的树高度差也不会超过 `1`。所以猜想出：如果左右子树约平均，树就越平衡。这样我们就可以每次取中间元素作为当前的根节点，两侧的元素作为左右子树递归建树，左侧区间 `[L, mid - 1]` 作为左子树，右侧区间 `[mid + 1, R]` 作为右子树。

## 代码

```python
class Solution:
    def sortedArrayToBST(self, nums: List[int]) -> TreeNode:
        size = len(nums)
        if size == 0:
            return None
        mid = size // 2
        root = TreeNode(nums[mid])
        root.left = Solution.sortedArrayToBST(self, nums[:mid])
        root.right = Solution.sortedArrayToBST(self, nums[mid + 1:])
        return root
```

# [面试题 04.05. 合法二叉搜索树](https://leetcode.cn/problems/legal-binary-search-tree-lcci/)

- 标签：树、深度优先搜索、二叉搜索树、二叉树
- 难度：中等

## 题目链接

- [面试题 04.05. 合法二叉搜索树 - 力扣](https://leetcode.cn/problems/legal-binary-search-tree-lcci/)

## 题目大意

给定一个二叉树的根节点 `root`。

要求：检查该二叉树是否为二叉搜索树。

二叉搜索树特征：

- 节点的左子树只包含小于当前节点的数。
- 节点的右子树只包含大于当前节点的数。
- 所有左子树和右子树自身必须也是二叉搜索树。

## 解题思路

根据题意进行递归遍历即可。前序、中序、后序遍历都可以。

以前序遍历为例，递归函数为：`preorderTraversal(root, min_v, max_v)`

前序遍历时，先判断根节点的值是否在 `(min_v, max_v)` 之间。如果不在则直接返回 `False`。在区间内，则继续递归检测左右子树是否满足，都满足才是一棵二叉搜索树。

递归遍历左子树的时候，要将上界 `max_v` 改为左子树的根节点值，因为左子树上所有节点的值均小于根节点的值。同理，遍历右子树的时候，要将下界 `min_v` 改为右子树的根节点值，因为右子树上所有节点的值均大于根节点。

## 代码

```python
class Solution:
    def isValidBST(self, root: TreeNode) -> bool:
        def preorderTraversal(root, min_v, max_v):
            if root == None:
                return True
            if root.val >= max_v or root.val <= min_v:
                return False
            return preorderTraversal(root.left, min_v, root.val) and preorderTraversal(root.right, root.val, max_v)

        return preorderTraversal(root, float('-inf'), float('inf'))
```

# [面试题 04.06. 后继者](https://leetcode.cn/problems/successor-lcci/)

- 标签：树、深度优先搜索、二叉搜索树、二叉树
- 难度：中等

## 题目链接

- [面试题 04.06. 后继者 - 力扣](https://leetcode.cn/problems/successor-lcci/)

## 题目大意

给定一棵二叉搜索树的根节点 `root` 和其中一个节点 `p`。

要求：找出该节点在树中的中序后继，即按照中序遍历的顺序节点 `p` 的下一个节点。如果节点 `p` 没有对应的下一个节点，则返回 `None`。

## 解题思路

递归遍历，具体步骤如下：

- 如果 `root.val` 小于等于 `p.val`，则直接从 `root` 的右子树递归查找比 `p.val` 大的节点，从而找到中序后继。
- 如果 `root.val` 大于 `p.val`，则 `root` 有可能是中序后继，也有可能是 `root` 的左子树。则从 `root` 的左子树递归查找更接近（更小的）。如果查找的值为 `None`，则当前 `root` 就是中序后继，否则继续递归查找，从而找到中序后继。

## 代码

```python
class Solution:
    def inorderSuccessor(self, root: TreeNode, p: TreeNode) -> TreeNode:
        if not p or not root:
            return None

        if root.val <= p.val:
            node = self.inorderSuccessor(root.right, p)
        else:
            node = self.inorderSuccessor(root.left, p)
            if not node:
                node = root
        return node
```

# [面试题 04.08. 首个共同祖先](https://leetcode.cn/problems/first-common-ancestor-lcci/)

- 标签：树、深度优先搜索、二叉树
- 难度：中等

## 题目链接

- [面试题 04.08. 首个共同祖先 - 力扣](https://leetcode.cn/problems/first-common-ancestor-lcci/)

## 题目大意

给定一个二叉树，要求找到该树中指定节点 `p`、`q` 的最近公共祖先：

- 祖先：若节点 `p` 在节点 `node` 的左子树或右子树中，或者 `p = node`，则称 `node` 是 `p` 的祖先。

- 最近公共祖先：对于树的两个节点 `p`、`q`，最近公共祖先表示为一个节点 `lca_node`，满足 `lca_node` 是 `p`、`q` 的祖先且 `lca_node` 的深度尽可能大（一个节点也可以是自己的祖先）。

## 解题思路

设 `lca_node` 为节点 `p`、`q` 的最近公共祖先。则 `lca_node` 只能是下面几种情况：

- `p`、`q` 在 `lca_node` 的子树中，且分别在 `lca_node` 的两侧子树中。
- `p == lca_node`，且 `q` 在 `lca_node` 的左子树或右子树中。
- `q == lca_node`，且 `p` 在 `lca_node` 的左子树或右子树中。

下面递归求解 `lca_node`。递归需要满足以下条件：

- 如果 `p`、`q` 都不为空，则返回 `p`、`q` 的公共祖先。
- 如果 `p`、`q` 只有一个存在，则返回存在的一个。
- 如果 `p`、`q` 都不存在，则返回存在的一个。

具体思路为：

- 如果当前节点 `node` 为 `None`，则说明 `p`、`q` 不在 `node` 的子树中，不可能为公共祖先，直接返回 `None`。
- 如果当前节点 `node` 等于 `p` 或者 `q`，那么 `node` 就是 `p`、`q` 的最近公共祖先，直接返回 `node`。
- 递归遍历左子树、右子树，并判断左右子树结果。
  - 如果左子树为空，则返回右子树。
  - 如果右子树为空，则返回左子树。
  - 如果左右子树都不为空，则说明 `p`、`q` 在当前根节点的两侧，当前根节点就是他们的最近公共祖先。
  - 如果左右子树都为空，则返回空。

## 代码

```python
class Solution:
    def lowestCommonAncestor(self, root: TreeNode, p: TreeNode, q: TreeNode) -> TreeNode:
        if root == p or root == q:
            return root

        if root:
            node_left = self.lowestCommonAncestor(root.left, p, q)
            node_right = self.lowestCommonAncestor(root.right, p, q)
            if node_left and node_right:
                return root
            elif not node_left:
                return node_right
            else:
                return node_left
        return None
```

# [面试题 04.12. 求和路径](https://leetcode.cn/problems/paths-with-sum-lcci/)

- 标签：树、深度优先搜索、二叉树
- 难度：中等

## 题目链接

- [面试题 04.12. 求和路径 - 力扣](https://leetcode.cn/problems/paths-with-sum-lcci/)

## 题目大意

给定一个二叉树的根节点 `root`，和一个整数 `targetSum`。

要求：求出该二叉树里节点值之和等于 `targetSum` 的路径的数目。

- 路径：不需要从根节点开始，也不需要在叶子节点结束，但是路径方向必须是向下的（只能从父节点到子节点）。

## 解题思路

直观想法是：

以每一个节点 `node` 为起始节点，向下检测延伸的路径。递归遍历每一个节点所有可能的路径，然后将这些路径数目加起来即为答案。

但是这样会存在许多重复计算。我们可以定义节点的前缀和来减少重复计算。

- 节点的前缀和：从根节点到当前节点路径上所有节点的和。

有了节点的前缀和，我们就可以通过前缀和来计算两节点之间的路劲和。即：`则两节点之间的路径和 = 两节点之间的前缀和之差`。

为了计算符合要求的路径数量，我们用哈希表存储「前缀和的节点数量」。哈希表以「当前节点的前缀和」为键，以「该前缀和的节点数量」为值。这样就能通过哈希表直接计算出符合要求的路径数量，从而累加到答案上。

整个算法的具体步骤如下：

- 通过先序遍历方式递归遍历二叉树，计算每一个节点的前缀和 `cur_sum`。
- 从哈希表中取出 `cur_sum - target_sum` 的路径数量（也就是表示存在从前缀和为 `cur_sum - target_sum` 所对应的节点到前缀和为 `cur_sum` 所对应的节点的路径个数）累加到答案 `res` 中。
- 然后以「当前节点的前缀和」为键，以「该前缀和的节点数量」为值，存入哈希表中。
- 递归遍历二叉树，并累加答案值。
- 恢复哈希表「当前前缀和的节点数量」，返回答案。

## 代码

```python

```

# [面试题 08.04. 幂集](https://leetcode.cn/problems/power-set-lcci/)

- 标签：位运算、数组、回溯
- 难度：中等

## 题目链接

- [面试题 08.04. 幂集 - 力扣](https://leetcode.cn/problems/power-set-lcci/)

## 题目大意

给定一个集合 `nums`，集合中不包含重复元素。

压枪欧秋：返回该集合的所有子集。

## 解题思路

回溯算法，遍历集合 `nums`。为了使得子集不重复，每次遍历从当前位置的下一个位置进行下一层遍历。

## 代码

```python
class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:
        def backtrack(size, subset, index):
            res.append(subset)
            for i in range(index, size):
                backtrack(size, subset + [nums[i]], i + 1)

        size = len(nums)
        res = list()
        backtrack(size, [], 0)
        return res
```

# [面试题 08.07. 无重复字符串的排列组合](https://leetcode.cn/problems/permutation-i-lcci/)

- 标签：字符串、回溯
- 难度：中等

## 题目链接

- [面试题 08.07. 无重复字符串的排列组合 - 力扣](https://leetcode.cn/problems/permutation-i-lcci/)

## 题目大意

给定一个字符串 `S`。

要求：打印出该字符串中字符的所有排列。可以以任意顺序返回这个字符串数组，但里边不能有重复元素。

## 解题思路

使用 `visited` 数组标记该元素在当前排列中是否被访问过。若未被访问过则将其加入排列中，并在访问后将该元素变为未访问状态。然后进行回溯遍历。

## 代码

```python
class Solution:
    res = []
    path = []

    def backtrack(self, S, visited):
        if len(self.path) == len(S):
            self.res.append(''.join(self.path))
            return
        for i in range(len(S)):
            if not visited[i]:
                visited[i] = True
                self.path.append(S[i])
                self.backtrack(S, visited)
                self.path.pop()
                visited[i] = False

    def permutation(self, S: str) -> List[str]:
        self.res.clear()
        self.path.clear()
        visited = [False for _ in range(len(S))]
        self.backtrack(S, visited)
        return self.res
```

# [面试题 08.08. 有重复字符串的排列组合](https://leetcode.cn/problems/permutation-ii-lcci/)

- 标签：字符串、回溯
- 难度：中等

## 题目链接

- [面试题 08.08. 有重复字符串的排列组合 - 力扣](https://leetcode.cn/problems/permutation-ii-lcci/)

## 题目大意

给定一个字符串 `s`，字符串中包含有重复字符。

要求：打印出该字符串中字符的所有排列。可以以任意顺序返回这个字符串数组。

## 解题思路

因为原字符串可能含有重复元素，所以在回溯的时候需要进行去重。先将字符串 `s` 转为 `list` 列表，再对列表进行排序，然后使用 `visited` 数组标记该元素在当前排列中是否被访问过。若未被访问过则将其加入排列中，并在访问后将该元素变为未访问状态。

然后再递归遍历下一层元素之前，增加一句语句进行判重：`if i > 0 and nums[i] == nums[i - 1] and not visited[i - 1]: continue`。

然后进行回溯遍历。

## 代码

```python
class Solution:
    res = []
    path = []

    def backtrack(self, ls, visited):
        if len(self.path) == len(ls):
            self.res.append(''.join(self.path))
            return
        for i in range(len(ls)):
            if i > 0 and ls[i] == ls[i - 1] and not visited[i - 1]:
                continue

            if not visited[i]:
                visited[i] = True
                self.path.append(ls[i])
                self.backtrack(ls, visited)
                self.path.pop()
                visited[i] = False

    def permutation(self, S: str) -> List[str]:
        self.res.clear()
        self.path.clear()
        ls = list(S)
        ls.sort()
        visited = [False for _ in range(len(S))]
        self.backtrack(ls, visited)
        return self.res
```

# [面试题 08.09. 括号](https://leetcode.cn/problems/bracket-lcci/)

- 标签：字符串、动态规划、回溯
- 难度：中等

## 题目链接

- [面试题 08.09. 括号 - 力扣](https://leetcode.cn/problems/bracket-lcci/)

## 题目大意

给定一个整数 `n`。

要求：生成所有有可能且有效的括号组合。

## 解题思路

通过回溯算法生成所有答案。为了生成的括号组合是有效的，回溯的时候，使用一个标记变量 `symbol` 来表示是否当前组合是否成对匹配。

如果在当前组合中增加一个 `(`，则 `symbol += 1`，如果增加一个 `)`，则 `symbol -= 1`。显然只有在 `symbol < n` 的时候，才能增加 `(`，在 `symbol > 0` 的时候，才能增加 `)`。

如果最终生成 `2 * n` 的括号组合，并且 `symbol == 0`，则说明当前组合是有效的，将其加入到最终答案数组中。

最终输出最终答案数组。

## 代码

```python
class Solution:
    def generateParenthesis(self, n: int) -> List[str]:
        def backtrack(parenthesis, symbol, index):
            if n * 2 == index:
                if symbol == 0:
                    parentheses.append(parenthesis)
            else:
                if symbol < n:
                    backtrack(parenthesis + '(', symbol + 1, index + 1)
                if symbol > 0:
                    backtrack(parenthesis + ')', symbol - 1, index + 1)

        parentheses = list()
        backtrack("", 0, 0)
        return parentheses
```

# [面试题 08.10. 颜色填充](https://leetcode.cn/problems/color-fill-lcci/)

- 标签：深度优先搜索、广度优先搜索、数组、矩阵
- 难度：简单

## 题目链接

- [面试题 08.10. 颜色填充 - 力扣](https://leetcode.cn/problems/color-fill-lcci/)

## 题目大意

给定一个二维整数矩阵 `image`，其中 `image[i][j]` 表示矩阵第 `i` 行、第 `j` 列上网格块的颜色值。再给定一个起始位置 `(sr, sc)`，以及一个目标颜色 `newColor`。

要求：对起始位置 `(sr, sc)` 所在位置周围区域填充颜色为 `newColor`。并返回填充后的图像 `image`。

- 周围区域：颜色相同且在上、下、左、右四个方向上存在相连情况的若干元素。

## 解题思路

深度优先搜索。使用二维数组 `visited` 标记访问过的节点。遍历上、下、左、右四个方向上的点。如果下一个点位置越界，或者当前位置与下一个点位置颜色不一样，则对该节点进行染色。

在遍历的过程中注意使用 `visited` 标记访问过的节点，以免重复遍历。

## 代码

```python
class Solution:
    directs = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    def dfs(self, image, i, j, origin_color, color, visited):
        rows, cols = len(image), len(image[0])

        for direct in self.directs:
            new_i = i + direct[0]
            new_j = j + direct[1]

            # 下一个位置越界，则当前点在边界，对其进行着色
            if new_i < 0 or new_i >= rows or new_j < 0 or new_j >= cols:
                image[i][j] = color
                continue

            # 如果访问过，则跳过
            if visited[new_i][new_j]:
                continue

            # 如果下一个位置颜色与当前颜色相同，则继续搜索
            if image[new_i][new_j] == origin_color:
                visited[new_i][new_j] = True
                self.dfs(image, new_i, new_j, origin_color, color, visited)
            # 下一个位置颜色与当前颜色不同，则当前位置为连通区域边界，对其进行着色
            else:
                image[i][j] = color

    def floodFill(self, image: List[List[int]], sr: int, sc: int, newColor: int) -> List[List[int]]:
        if not image:
            return image

        rows, cols = len(image), len(image[0])
        visited = [[False for _ in range(cols)] for _ in range(rows)]
        visited[sr][sc] = True

        self.dfs(image, sr, sc, image[sr][sc], newColor, visited)

        return image
```

# [面试题 08.12. 八皇后](https://leetcode.cn/problems/eight-queens-lcci/)

- 标签：数组、回溯
- 难度：困难

## 题目链接

- [面试题 08.12. 八皇后 - 力扣](https://leetcode.cn/problems/eight-queens-lcci/)

## 题目大意

- n 皇后问题：将 n 个皇后放置在 `n * n` 的棋盘上，并且使得皇后彼此之间不能攻击。
- 皇后彼此不能相互攻击：指的是任何两个皇后都不能处于同一条横线、纵线或者斜线上。

现在给定一个整数 `n`，返回所有不同的「n 皇后问题」的解决方案。每一种解法包含一个不同的「n 皇后问题」的棋子放置方案，该方案中的 `Q` 和 `.` 分别代表了皇后和空位。

## 解题思路

经典的回溯问题。使用 `chessboard` 来表示棋盘，`Q` 代表皇后，`.` 代表空位，初始都为 `.`。然后使用 `res` 存放最终答案。

先定义棋盘合理情况判断方法，判断同一条横线、纵线或者斜线上是否存在两个以上的皇后。

再定义回溯方法，从第一行开始进行遍历。

- 如果当前行 `row` 等于 `n`，则当前棋盘为一个可行方案，将其拼接加入到 `res` 数组中。
-  遍历 `[0, n]` 列元素，先验证棋盘是否可行，如果可行：
  - 将当前行当前列尝试换为 `Q`。
  - 然后继续递归下一行。
  - 再将当前行回退为 `.`。
- 最终返回 `res` 数组。

## 代码

```python
class Solution:
    res = []
    def backtrack(self, n: int, row: int, chessboard: List[List[str]]):
        if row == n:
            temp_res = []
            for temp in chessboard:
                temp_str = ''.join(temp)
                temp_res.append(temp_str)
            self.res.append(temp_res)
            return
        for col in range(n):
            if self.isValid(n, row, col, chessboard):
                chessboard[row][col] = 'Q'
                self.backtrack(n, row + 1, chessboard)
                chessboard[row][col] = '.'

    def isValid(self, n: int, row: int, col: int, chessboard: List[List[str]]):
        for i in range(row):
            if chessboard[i][col] == 'Q':
                return False

        i, j = row - 1, col - 1
        while i >= 0 and j >= 0:
            if chessboard[i][j] == 'Q':
                return False
            i -= 1
            j -= 1
        i, j = row - 1, col + 1
        while i >= 0 and j < n:
            if chessboard[i][j] == 'Q':
                return False
            i -= 1
            j += 1

        return True

    def solveNQueens(self, n: int) -> List[List[str]]:
        self.res.clear()
        chessboard = [['.' for _ in range(n)] for _ in range(n)]
        self.backtrack(n, 0, chessboard)
        return self.res
```

# [面试题 10.01. 合并排序的数组](https://leetcode.cn/problems/sorted-merge-lcci/)

- 标签：数组、双指针、排序
- 难度：简单

## 题目链接

- [面试题 10.01. 合并排序的数组 - 力扣](https://leetcode.cn/problems/sorted-merge-lcci/)

## 题目大意

**描述**：给定两个排序后的数组 `A` 和 `B`，以及 `A` 的元素数量 `m` 和 `B` 的元素数量 `n`。 `A` 的末端有足够的缓冲空间容纳 `B`。

**要求**：编写一个方法，将 `B` 合并入 `A` 并排序。

**说明**：

- $A.length == n + m$。

**示例**：

- 示例 1：

```python
输入:
A = [1,2,3,0,0,0], m = 3
B = [2,5,6],       n = 3

输出: [1,2,2,3,5,6]
```

## 解题思路

### 思路 1：归并排序

可以利用归并排序算法的归并步骤思路。

1. 使用两个指针分别表示`A`、`B` 正在处理的元素下标。
2. 对 `A`、`B` 进行归并操作，将结果存入新数组中。归并之后，再将所有元素赋值到数组 `A` 中。

### 思路 1：代码

```python
class Solution:
    def merge(self, A: List[int], m: int, B: List[int], n: int) -> None:
        """
        Do not return anything, modify A in-place instead.
        """
        arr = []
        index_A, index_B = 0, 0
        while index_A < m and index_B < n:
            if A[index_A] <= B[index_B]:
                arr.append(A[index_A])
                index_A += 1
            else:
                arr.append(B[index_B])
                index_B += 1
        while index_A < m:
            arr.append(A[index_A])
            index_A += 1
        while index_B < n:
            arr.append(B[index_B])
            index_B += 1
        for i in range(m + n):
            A[i] = arr[i]
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(m + n)$。
- **空间复杂度**：$O(m + n)$。

# [面试题 10.02. 变位词组](https://leetcode.cn/problems/group-anagrams-lcci/)

- 标签：数组、哈希表、字符串、排序
- 难度：中等

## 题目链接

- [面试题 10.02. 变位词组 - 力扣](https://leetcode.cn/problems/group-anagrams-lcci/)

## 题目大意

给定一个字符串数组 `strs`。

要求：将所有变位词组合在一起。不需要考虑输出顺序。

- 变位词：字母相同，但排列不同的字符串。

## 解题思路

使用哈希表记录变位词。对每一个字符串进行排序，按照 `排序字符串：变位词数组` 的键值顺序进行存储。

最终将哈希表的值转换为对应数组返回结果。

## 代码

```python
class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        str_dict = dict()
        res = []
        for s in strs:
            sort_s = str(sorted(s))
            if sort_s in str_dict:
                str_dict[sort_s] += [s]
            else:
                str_dict[sort_s] = [s]

        for sort_s in str_dict:
            res += [str_dict[sort_s]]
        return res
```

# [面试题 10.09. 排序矩阵查找](https://leetcode.cn/problems/sorted-matrix-search-lcci/)

- 标签：数组、二分查找、分治、矩阵
- 难度：中等

## 题目链接

- [面试题 10.09. 排序矩阵查找 - 力扣](https://leetcode.cn/problems/sorted-matrix-search-lcci/)

## 题目大意

给定一个 `m * n` 大小的有序整数矩阵。每一行、每一列都按升序排列。再给定一个目标值 `target`。

要求：判断矩阵中是否可以找到 `target`，若找到 `target`，返回 `True`，否则返回 `False`。

## 解题思路

矩阵是有序的，可以考虑使用二分搜索来进行查找。

迭代对角线元素，假设对角线元素的坐标为 `(row, col)`。把数组元素按对角线分为右上角部分和左下角部分。

则对于当前对角线元素右侧第 `row` 行、对角线元素下侧第 `col` 列进行二分查找。

- 如果找到目标，直接返回 `True`。
- 如果找不到目标，则缩小范围，继续查找。
- 直到所有对角线元素都遍历完，依旧没找到，则返回 `False`。

## 代码

```python
class Solution:
    def diagonalBinarySearch(self, matrix, diagonal, target):
        left = 0
        right = diagonal
        while left < right:
            mid = left + (right - left) // 2
            if matrix[mid][mid] < target:
                left = mid + 1
            else:
                right = mid
        return left

    def rowBinarySearch(self, matrix, begin, cols, target):
        left = begin
        right = cols
        while left < right:
            mid = left + (right - left) // 2
            if matrix[begin][mid] < target:
                left = mid + 1
            elif matrix[begin][mid] > target:
                right = mid - 1
            else:
                left = mid
                break
        return begin <= left <= cols and matrix[begin][left] == target

    def colBinarySearch(self, matrix, begin, rows, target):
        left = begin + 1
        right = rows
        while left < right:
            mid = left + (right - left) // 2
            if matrix[mid][begin] < target:
                left = mid + 1
            elif matrix[mid][begin] > target:
                right = mid - 1
            else:
                left = mid
                break
        return begin <= left <= rows and matrix[left][begin] == target

    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        rows = len(matrix)
        if rows == 0:
            return False
        cols = len(matrix[0])
        if cols == 0:
            return False

        min_val = min(rows, cols)
        index = self.diagonalBinarySearch(matrix, min_val - 1, target)
        if matrix[index][index] == target:
            return True
        for i in range(index + 1):
            row_search = self.rowBinarySearch(matrix, i, cols - 1, target)
            col_search = self.colBinarySearch(matrix, i, rows - 1, target)
            if row_search or col_search:
                return True
        return False
```

# [面试题 16.02. 单词频率](https://leetcode.cn/problems/words-frequency-lcci/)

- 标签：设计、字典树、数组、哈希表、字符串
- 难度：中等

## 题目链接

- [面试题 16.02. 单词频率 - 力扣](https://leetcode.cn/problems/words-frequency-lcci/)

## 题目大意

要求：设计一个方法，找出任意指定单词在一本书中的出现频率。

支持如下操作：

- `WordsFrequency(book)` 构造函数，参数为字符串数组构成的一本书。
- `get(word)` 查询指定单词在书中出现的频率。

## 解题思路

使用字典树统计单词频率。

构造函数时，构建一个字典树，并将所有单词存入字典树中，同时在字典树中记录并维护单词频率。

查询时，调用字典树查询方法，查询单词频率。

## 代码

```python
class Trie:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.children = dict()
        self.isEnd = False
        self.count = 0


    def insert(self, word: str) -> None:
        """
        Inserts a word into the trie.
        """
        cur = self
        for ch in word:
            if ch not in cur.children:
                cur.children[ch] = Trie()
            cur = cur.children[ch]
        cur.isEnd = True
        cur.count += 1


    def search(self, word: str) -> bool:
        """
        Returns if the word is in the trie.
        """
        cur = self
        for ch in word:
            if ch not in cur.children:
                return 0
            cur = cur.children[ch]
        if cur and cur.isEnd:
            return cur.count
        return 0

class WordsFrequency:

    def __init__(self, book: List[str]):
        self.tire_tree = Trie()
        for word in book:
            self.tire_tree.insert(word)


    def get(self, word: str) -> int:
        return self.tire_tree.search(word)
```

# [面试题 16.05. 阶乘尾数](https://leetcode.cn/problems/factorial-zeros-lcci/)

- 标签：数学
- 难度：简单

## 题目链接

- [面试题 16.05. 阶乘尾数 - 力扣](https://leetcode.cn/problems/factorial-zeros-lcci/)

## 题目大意

给定一个整数 `n`。

要求：计算 `n` 的阶乘中尾随零的数量。

注意：$0 <= n <= 10^4$。

## 解题思路

阶乘中，末尾 `0` 的来源只有 `2 * 5`。所以尾随 `0` 的个数为 `2` 的倍数个数和 `5` 的倍数个数的最小值。又因为 `2 < 5`，`2` 的倍数个数肯定小于等于 `5` 的倍数，所以直接统计 `5` 的倍数个数即可。

## 代码

```python
class Solution:
    def trailingZeroes(self, n: int) -> int:
        count = 0
        while n > 0:
            count += n // 5
            n = n // 5
        return count
```

# [面试题 16.26. 计算器](https://leetcode.cn/problems/calculator-lcci/)

- 标签：栈、数学、字符串
- 难度：中等

## 题目链接

- [面试题 16.26. 计算器 - 力扣](https://leetcode.cn/problems/calculator-lcci/)

## 题目大意

给定一个包含正整数、加（`+`）、减（`-`）、乘（`*`）、除（`/`）的算出表达式（括号除外）。表达式仅包含非负整数，`+`、`-`、`*`、`/` 四种运算符和空格 ` `。整数除法仅保留整数部分。

要求：计算其结果。

## 解题思路

计算表达式中，乘除运算优先于加减运算。我们可以先进行乘除运算，再将进行乘除运算后的整数值放入原表达式中相应位置，再依次计算加减。

可以考虑使用一个栈来保存进行乘除运算后的整数值。正整数直接压入栈中，负整数，则将对应整数取负号，再压入栈中。这样最终计算结果就是栈中所有元素的和。

具体做法：

- 遍历字符串 s，使用变量 op 来标记数字之前的运算符，默认为 `+`。
- 如果遇到数字，继续向后遍历，将数字进行累积，得到完整的整数 num。判断当前 op 的符号。
  - 如果 op 为 `+`，则将 num 压入栈中。
  - 如果 op 为 `-`，则将 -num 压入栈中。
  - 如果 op 为 `*`，则将栈顶元素 top 取出，计算 top * num，并将计算结果压入栈中。
  - 如果 op 为 `/`，则将栈顶元素 top 取出，计算 int(top / num)，并将计算结果压入栈中。
- 如果遇到 `+`、`-`、`*`、`/` 操作符，则更新 op。
- 最后将栈中整数进行累加，并返回结果。

## 代码

```python
class Solution:
    def calculate(self, s: str) -> int:
        size = len(s)
        stack = []
        op = '+'
        index = 0
        while index < size:
            if s[index] == ' ':
                index += 1
                continue
            if s[index].isdigit():
                num = ord(s[index]) - ord('0')
                while index + 1 < size and s[index + 1].isdigit():
                    index += 1
                    num = 10 * num + ord(s[index]) - ord('0')
                if op == '+':
                    stack.append(num)
                elif op == '-':
                    stack.append(-num)
                elif op == '*':
                    top = stack.pop()
                    stack.append(top * num)
                elif op == '/':
                    top = stack.pop()
                    stack.append(int(top / num))
            elif s[index] in "+-*/":
                op = s[index]
            index += 1
        return sum(stack)
```

# [面试题 17.06. 2出现的次数](https://leetcode.cn/problems/number-of-2s-in-range-lcci/)

- 标签：递归、数学、动态规划
- 难度：困难

## 题目链接

- [面试题 17.06. 2出现的次数 - 力扣](https://leetcode.cn/problems/number-of-2s-in-range-lcci/)

## 题目大意

**描述**：给定一个整数 $n$。

**要求**：计算从 $0$ 到 $n$ (包含 $n$) 中数字 $2$ 出现的次数。

**说明**：

- $n \le 10^9$。

**示例**：

- 示例 1：

```python
输入: 25
输出: 9
解释: (2, 12, 20, 21, 22, 23, 24, 25)(注意 22 应该算作两次)
```

## 解题思路

### 思路 1：动态规划 + 数位 DP

将 $n$ 转换为字符串 $s$，定义递归函数 `def dfs(pos, cnt, isLimit):` 表示构造第 $pos$ 位及之后所有数位中数字 $2$ 出现的个数。接下来按照如下步骤进行递归。

1. 从 `dfs(0, 0, True)` 开始递归。 `dfs(0, 0, True)` 表示：
	1. 从位置 $0$ 开始构造。
	2. 初始数字 $2$ 出现的个数为 $0$。
	3. 开始时受到数字 $n$ 对应最高位数位的约束。
2. 如果遇到  $pos == len(s)$，表示到达数位末尾，此时：返回数字 $2$ 出现的个数 $cnt$。
3. 如果 $pos \ne len(s)$，则定义方案数 $ans$，令其等于 $0$，即：`ans = 0`。
4. 如果遇到 $isNum == False$，说明之前位数没有填写数字，当前位可以跳过，这种情况下方案数等于 $pos + 1$ 位置上没有受到 $pos$ 位的约束，并且之前没有填写数字时的方案数，即：`ans = dfs(i + 1, state, False, False)`。
5. 如果 $isNum == True$，则当前位必须填写一个数字。此时：
	1. 因为不需要考虑前导 $0$ 所以当前位数位所能选择的最小数字（$minX$）为 $0$。
	2. 根据 $isLimit$ 来决定填当前位数位所能选择的最大数字（$maxX$）。
	3. 然后根据 $[minX, maxX]$ 来枚举能够填入的数字 $d$。
	4. 方案数累加上当前位选择 $d$ 之后的方案数，即：`ans += dfs(pos + 1, cnt + (d == 2), isLimit and d == maxX)`。
		1. `cnt + (d == 2)` 表示之前数字 $2$ 出现的个数加上当前位为数字 $2$ 的个数。
		2. `isLimit and d == maxX` 表示 $pos + 1$ 位受到之前位 $pos$ 位限制。
6. 最后的方案数为 `dfs(0, 0, True)`，将其返回即可。

### 思路 1：代码

```python
class Solution:
    def numberOf2sInRange(self, n: int) -> int:
        # 将 n 转换为字符串 s
        s = str(n)
        
        @cache
        # pos: 第 pos 个数位
        # cnt: 之前数字 2 出现的个数。
        # isLimit: 表示是否受到选择限制。如果为真，则第 pos 位填入数字最多为 s[pos]；如果为假，则最大可为 9。
        def dfs(pos, cnt, isLimit):
            if pos == len(s):
                return cnt
            
            ans = 0            
            # 不需要考虑前导 0，则最小可选择数字为 0
            minX = 0
            # 如果受到选择限制，则最大可选择数字为 s[pos]，否则最大可选择数字为 9。
            maxX = int(s[pos]) if isLimit else 9
            
            # 枚举可选择的数字
            for d in range(minX, maxX + 1): 
                ans += dfs(pos + 1, cnt + (d == 2), isLimit and d == maxX)
            return ans
    
        return dfs(0, 0, True)
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(\log n)$。
- **空间复杂度**：$O(\log n)$。
# [面试题 17.14. 最小K个数](https://leetcode.cn/problems/smallest-k-lcci/)

- 标签：数组、分治、快速选择、排序、堆（优先队列）
- 难度：中等

## 题目链接

- [面试题 17.14. 最小K个数 - 力扣](https://leetcode.cn/problems/smallest-k-lcci/)

## 题目大意

给定整数数组 `arr`，再给定一个整数 `k`。

要求：返回数组 `arr` 中最小的 `k` 个数。

## 解题思路

直接可以想到的思路是：排序后输出数组上对应的最小的 k 个数。所以问题关键在于排序方法的复杂度。

冒泡排序、选择排序、插入排序时间复杂度 $O(n^2)$ 太高了，解答会超时。

可考虑堆排序、归并排序、快速排序。本题使用堆排序。具体做法如下：

1. 利用数组前 `k` 个元素，建立大小为 `k` 的大顶堆。
2. 遍历数组 `[k, size - 1]` 的元素，判断其与堆顶元素关系，如果比堆顶元素小，则将其赋值给堆顶元素，再对大顶堆进行调整。
3. 最后输出前调整过后的大顶堆的前 `k` 个元素。

## 代码

```python
class Solution:
    def heapify(self, nums: [int], index: int, end: int):
        left = index * 2 + 1
        right = left + 1
        while left <= end:
            # 当前节点为非叶子节点
            max_index = index
            if nums[left] > nums[max_index]:
                max_index = left
            if right <= end and nums[right] > nums[max_index]:
                max_index = right
            if index == max_index:
                # 如果不用交换，则说明已经交换结束
                break
            nums[index], nums[max_index] = nums[max_index], nums[index]
            # 继续调整子树
            index = max_index
            left = index * 2 + 1
            right = left + 1

    # 初始化大顶堆
    def buildMaxHeap(self, nums: [int], k: int):
        # (k-2) // 2 是最后一个非叶节点，叶节点不用调整
        for i in range((k - 2) // 2, -1, -1):
            self.heapify(nums, i, k - 1)
        return nums

    def smallestK(self, arr: List[int], k: int) -> List[int]:
        size = len(arr)
        if k <= 0 or not arr:
            return []
        if size <= k:
            return arr

        self.buildMaxHeap(arr, k)
        for i in range(k, size):
            if arr[i] < arr[0]:
                arr[i], arr[0] = arr[0], arr[i]
                self.heapify(arr, 0, k - 1)

        return arr[:k]
```

# [面试题 17.15. 最长单词](https://leetcode.cn/problems/longest-word-lcci/)

- 标签：字典树、数组、哈希表、字符串
- 难度：中等

## 题目链接

- [面试题 17.15. 最长单词 - 力扣](https://leetcode.cn/problems/longest-word-lcci/)

## 题目大意

给定一组单词 `words`。

要求：找出其中的最长单词，且该单词由这组单词中的其他单词组合而成。若有多个长度相同的结果，返回其中字典序最小的一项，若没有符合要求的单词则返回空字符串。

## 解题思路

先将所有单词按照长度从长到短排序，相同长度的字典序小的排在前面。然后将所有单词存入字典树中。

然后一重循环遍历所有单词 `word`，二重循环遍历单词中所有字符 `word[i]`。

如果当前遍历的字符为单词末尾，递归判断从 `i + 1` 位置开始，剩余部分是否可以切分为其他单词组合，如果可以切分，则返回当前单词 `word`。如果不可以切分，则返回空字符串 `""`。

## 代码

```python
class Trie:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.children = dict()
        self.isEnd = False


    def insert(self, word: str) -> None:
        """
        Inserts a word into the trie.
        """
        cur = self
        for ch in word:
            if ch not in cur.children:
                cur.children[ch] = Trie()
            cur = cur.children[ch]
        cur.isEnd = True


    def search(self, word: str) -> bool:
        """
        Returns if the word is in the trie.
        """
        cur = self
        for ch in word:
            if ch not in cur.children:
                return False
            cur = cur.children[ch]

        return cur is not None and cur.isEnd

    def splitToWord(self, remain):
        if not remain or remain == "":
            return True
        cur = self
        for i in range(len(remain)):
            ch = remain[i]
            if ch not in cur.children:
                return False
            if cur.children[ch].isEnd and self.splitToWord(remain[i + 1:]):
                return True
            cur = cur.children[ch]
        return False

    def dfs(self, words):
        for word in words:
            cur = self
            size = len(word)
            for i in range(size):
                ch = word[i]
                if i < size - 1 and cur.children[ch].isEnd and self.splitToWord(word[i+1:]):
                    return word
                cur = cur.children[ch]
        return ""

class Solution:
    def longestWord(self, words: List[str]) -> str:
        words.sort(key=lambda x: (-len(x), x))
        trie_tree = Trie()
        for word in words:
            trie_tree.insert(word)

        ans = trie_tree.dfs(words)
        return ans
```

# [面试题 17.17. 多次搜索](https://leetcode.cn/problems/multi-search-lcci/)

- 标签：字典树、数组、哈希表、字符串、字符串匹配、滑动窗口
- 难度：中等

## 题目链接

- [面试题 17.17. 多次搜索 - 力扣](https://leetcode.cn/problems/multi-search-lcci/)

## 题目大意

给定一个较长字符串 `big` 和一个包含较短字符串的数组 `smalls`。

要求：设计一个方法，根据 `smalls` 中的每一个较短字符串，对 `big` 进行搜索。输出 `smalls` 中的字符串在 `big` 里出现的所有位置 `positions`，其中 `positions[i]` 为 `smalls[i]` 出现的所有位置。

## 解题思路

构建字典树，将 `smalls` 中所有字符串存入字典树中，并在字典树中记录下插入字符串的顺序下标。

然后一重循环遍历 `big`，表示从第 `i` 位置开始的字符串 `big[i:]`。然后在字符串前缀中搜索对应的单词，将所有符合要求的单词插入顺序位置存入列表中，返回列表。

对于列表中每个单词插入下标顺序 `index` 和 `big[i:]` 来说， `i` 就是 `smalls` 中第 `index` 个字符串所对应在 `big` 中的开始位置，将其存入答案数组并返回即可。

## 代码

```python
class Trie:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.children = dict()
        self.isEnd = False
        self.index = -1


    def insert(self, word: str, index: int) -> None:
        """
        Inserts a word into the trie.
        """
        cur = self
        for ch in word:
            if ch not in cur.children:
                cur.children[ch] = Trie()
            cur = cur.children[ch]
        cur.isEnd = True
        cur.index = index


    def search(self, text: str) -> list:
        """
        Returns if the word is in the trie.
        """
        cur = self
        res = []
        for i in range(len(text)):
            ch = text[i]
            if ch not in cur.children:
                return res
            cur = cur.children[ch]
            if cur.isEnd:
                res.append(cur.index)
        return res

class Solution:
    def multiSearch(self, big: str, smalls: List[str]) -> List[List[int]]:
        trie_tree = Trie()
        for i in range(len(smalls)):
            word = smalls[i]
            trie_tree.insert(word, i)

        res = [[] for _ in range(len(smalls))]

        for i in range(len(big)):
            for index in trie_tree.search(big[i:]):
                res[index].append(i)
        return res
```

