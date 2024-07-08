# [0518. 零钱兑换 II](https://leetcode.cn/problems/coin-change-ii/)

- 标签：数组、动态规划
- 难度：中等

## 题目链接

- [0518. 零钱兑换 II - 力扣](https://leetcode.cn/problems/coin-change-ii/)

## 题目大意

**描述**：给定一个整数数组 $coins$ 表示不同面额的硬币，另给一个整数 $amount$ 表示总金额。

**要求**：计算并返回可以凑成总金额的硬币方案数。如果无法凑出总金额，则返回 $0$。

**说明**：

- 每一种面额的硬币枚数为无限个。
- $1 \le coins.length \le 300$。
- $1 \le coins[i] \le 5000$。
- $coins$ 中的所有值互不相同。
- $0 \le amount \le 5000$。

**示例**：

- 示例 1：

```python
输入：amount = 5, coins = [1, 2, 5]
输出：4
解释：有四种方式可以凑成总金额：
5=5
5=2+2+1
5=2+1+1+1
5=1+1+1+1+1
```

- 示例 2：

```python
输入：amount = 3, coins = [2]
输出：0
解释：只用面额 2 的硬币不能凑成总金额 3。
```

## 解题思路

### 思路 1：动态规划

这道题可以转换为：有 $n$ 种不同的硬币，$coins[i]$ 表示第 $i$ 种硬币的面额，每种硬币可以无限次使用。请问凑成总金额为 $amount$ 的背包，一共有多少种方案？

这就变成了完全背包问题。「[322. 零钱兑换](https://leetcode.cn/problems/coin-change/)」中计算的是凑成总金额的最少硬币个数，而这道题计算的是凑成总金额的方案数。

###### 1. 划分阶段

按照当前背包的载重上限进行阶段划分。

###### 2. 定义状态

定义状态 $dp[i]$ 表示为：凑成总金额为 $i$ 的方案总数。

###### 3. 状态转移方程

凑成总金额为 $i$ 的方案数 = 「不使用当前 $coin$，只使用之前硬币凑成金额 $i$ 的方案数」+「使用当前 $coin$ 凑成金额 $i - coin$ 的方案数」。即状态转移方程为：$dp[i] = dp[i] + dp[i - coin]$。

###### 4. 初始条件

- 凑成总金额为 $0$ 的方案数为 $1$，即 $dp[0] = 1$。

###### 5. 最终结果

根据我们之前定义的状态，$dp[i]$ 表示为：凑成总金额为 $i$ 的方案总数。 所以最终结果为 $dp[amount]$。

### 思路 1：代码

```python
class Solution:
    def change(self, amount: int, coins: List[int]) -> int:

        dp = [0 for _ in range(amount + 1)]
        dp[0] = 1
        for coin in coins:
            for i in range(coin, amount + 1):
                dp[i] += dp[i - coin]

        return dp[amount]
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(n \times amount)$，其中 $n$ 为数组 $coins$ 的元素个数，$amount$ 为总金额。
- **空间复杂度**：$O(amount)$。

# [0525. 连续数组](https://leetcode.cn/problems/contiguous-array/)

- 标签：数组、哈希表、前缀和
- 难度：中等

## 题目链接

- [0525. 连续数组 - 力扣](https://leetcode.cn/problems/contiguous-array/)

## 题目大意

给定一个二进制数组 `nums`。

要求：找到含有相同数量 `0` 和 `1` 的最长连续子数组，并返回该子数组的长度。

## 解题思路

「`0` 和 `1` 数量相同」等价于「`1` 的数量减去 `0` 的数量等于 `0`」。

我们可以使用一个变量 `pre_diff` 来记录下前 `i` 个数中，`1` 的数量比 `0` 的数量多多少个。我们把这个 `pre_diff`叫做「`1` 和 `0` 数量差」，也可以理解为变种的前缀和。

然后我们再用一个哈希表 `pre_dic` 来记录「`1` 和 `0` 数量差」第一次出现的下标。

那么，如果我们在遍历的时候，发现 `pre_diff` 相同的数量差已经在之前出现过了，则说明：这两段之间相减的 `1` 和 `0` 数量差为 `0`。

什么意思呢？

比如说：`j < i`，前 `j` 个数中第一次出现 `pre_diff == 2` ，然后前 `i` 个数中个第二次又出现了 `pre_diff == 2`。那么这两段形成的子数组 `nums[j + 1: i]` 中 `1` 比 `0` 多 `0` 个，则 `0` 和 `1` 数量相同的子数组长度为 `i - j`。

而第二次之所以又出现 `pre_diff == 2` ，是因为前半段子数组 `nums[0: j]`  贡献了相同的差值。

接下来还有一个小问题，如何计算「`1` 和 `0` 数量差」？

我们可以把数组中的 `1` 记为贡献 `+1`，`0` 记为贡献 `-1`。然后使用一个变量 `count`，只要出现 `1` 就让 `count` 加上 `1`，意思是又多出了 `1` 个 `1`。只要出现 `0`，将让 `count` 减去 `1`，意思是 `0` 和之前累积的 `1` 个 `1` 相互抵消掉了。这样遍历完数组，也就计算出了对应的「`1` 和 `0` 数量差」。

整个思路的具体做法如下：

- 创建一个哈希表，键值对关系为「`1` 和 `0` 的数量差：最早出现的下标 `i`」。
- 使用变量 `pre_diff` 来计算「`1` 和 `0` 数量差」，使用变量 `count` 来记录 `0` 和 `1` 数量相同的连续子数组的最长长度，然后遍历整个数组。
- 如果 `nums[i] == 1`，则让 `pre_diff += 1`；如果 `nums[i] == 0`，则让 `pre_diff -= 1`。
- 如果在哈希表中发现了相同的 `pre_diff`，则计算相应的子数组长度，与 `count` 进行比较并更新 `count` 值。
- 如果在哈希表中没有发现相同的 `pre_diff`，则在哈希表中记录下第一次出现 `pre_diff` 的下标 `i`。
- 最后遍历完输出 `count`。

> 注意：初始化哈希表为：`pre_dic = {0: -1}`，意思为空数组时，默认「`1` 和 `0` 数量差」为 `0`，且第一次出现的下标为 `-1`。
>
> 之所以这样做，是因为在遍历过程中可能会直接出现 `pre_diff == 0` 的情况，这种情况下说明 `nums[0: i]` 中 `0` 和 `1` 数量相同，如果像上边这样初始化后，就可以直接计算出此时子数组长度为 `i - (-1) = i + 1`。

## 代码

```python
class Solution:
    def findMaxLength(self, nums: List[int]) -> int:
        pre_dic = {0: -1}
        count = 0
        pre_sum = 0
        for i in range(len(nums)):
            if nums[i]:
                pre_sum += 1
            else:
                pre_sum -= 1
            if pre_sum in pre_dic:
                count = max(count, i - pre_dic[pre_sum])
            else:
                pre_dic[pre_sum] = i
        return count
```

# [0526. 优美的排列](https://leetcode.cn/problems/beautiful-arrangement/)

- 标签：位运算、数组、动态规划、回溯、状态压缩
- 难度：中等

## 题目链接

- [0526. 优美的排列 - 力扣](https://leetcode.cn/problems/beautiful-arrangement/)

## 题目大意

**描述**：给定一个整数 $n$。

**要求**：返回可以构造的「优美的排列」的数量。

**说明**：

- **优美的排列**：假设有 $1 \sim n$ 的 $n$ 个整数。如果用这些整数构造一个数组 $perm$（下标从 $1$ 开始），使得数组第 $i$ 位元素 $perm[i]$ 满足下面两个条件之一，则该数组就是一个「优美的排列」：
  - $perm[i]$ 能够被 $i$ 整除；
  - $i$ 能够被 $perm[i]$ 整除。

- $1 \le n \le 15$。

**示例**：

- 示例 1：

```python
输入：n = 2
输出：2
解释：
第 1 个优美的排列是 [1,2]：
    - perm[1] = 1 能被 i = 1 整除
    - perm[2] = 2 能被 i = 2 整除
第 2 个优美的排列是 [2,1]:
    - perm[1] = 2 能被 i = 1 整除
    - i = 2 能被 perm[2] = 1 整除
```

- 示例 2：

```python
输入：n = 1
输出：1
```

## 解题思路

### 思路 1：回溯算法

这道题可以看做是「[0046. 全排列](https://leetcode.cn/problems/permutations/)」的升级版。

1. 通过回溯算法我们可以将数组的所有排列情况列举出来。
2. 因为只有满足第 $i$ 位元素能被 $i$ 整除，或者满足 $i$ 能整除第 $i$ 位元素的条件下才符合要求，所以我们可以进行剪枝操作，不再考虑不满足要求的情况。
3. 最后回溯完输出方案数。

### 思路 1：代码

```python
class Solution:
    def countArrangement(self, n: int) -> int:
        ans = 0
        visited = set()

        def backtracking(index):
            nonlocal ans
            if index == n + 1:
                ans += 1
                return

            for i in range(1, n + 1):
                if i in visited:
                    continue
                if i % index == 0 or index % i == 0:
                    visited.add(i)
                    backtracking(index + 1)
                    visited.remove(i)

        backtracking(1)
        return ans
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(n!)$，其中 $n$ 为给定整数。
- **空间复杂度**：$O(n)$，递归栈空间大小为 $O(n)$。

### 思路 2：状态压缩 DP

因为 $n$ 最大只有 $15$，所以我们可以考虑使用「状态压缩」。

「状态压缩」指的是使用一个 $n$ 位的二进制数来表示排列中数的选取情况。

举个例子：

1. $n = 4, state = (1001)_2$，表示选择了数字 $1, 4$，剩余数字 $2$ 和 $3$ 未被选择。
2. $n = 6, state = (011010)_2$，表示选择了数字 $2, 4, 5$，剩余数字 $1, 3, 6$ 未被选择。

这样我们就可以使用 $n$ 位的二进制数 $state$ 来表示当前排列中数的选取情况。

如果我们需要检查值为 $k$ 的数字是否被选择时，可以通过判断 $(state \text{ >} \text{> } (k - 1)) \text{ \& } 1$ 是否为 $1$ 来确定。

如果为 $1$，则表示值为 $k$ 的数字被选择了，如果为 $0$，则表示值为 $k$ 的数字没有被选择。

###### 1. 划分阶段

按照排列的数字个数、数字集合的选择情况进行阶段划分。

###### 2. 定义状态

定义状态 $dp[i][state]$ 表示为：考虑前 $i$ 个数，且当数字集合的选择情况为 $state$ 时的方案数。

###### 3. 状态转移方程

假设 $dp[i][state]$ 中第 $i$ 个位置所选数字为 $k$，则：$state$ 中第 $k$ 位为 $1$，且 $k \mod i == 0$ 或者 $i \mod k == 0$。

那么 $dp[i][state]$ 肯定是由考虑前 $i - 1$ 个位置，且 $state$ 第 $k$ 位为 $0$ 的状态而来，即：$dp[i - 1][state \& (\neg(1 \text{ <}\text{< } (k - 1)))]$。

所以状态转移方程为：$dp[i][state] = \sum_{k = 1}^n dp[i - 1][state \text{ \& } (\neg(1 \text{ <} \text{< } (k - 1)))]$。

###### 4. 初始条件

- 不考虑任何数（$i = 0, state = 0$）的情况下，方案数为 $1$。

###### 5. 最终结果

根据我们之前定义的状态，$dp[i][state]$ 表示为：考虑前 $i$ 个数，且当数字集合的选择情况为 $state$ 时的方案数。所以最终结果为 $dp[i][states -  1]$，其中 $states = 1 \text{ <} \text{< } n$。

### 思路 2：代码

```python
class Solution:
    def countArrangement(self, n: int) -> int:
        states = 1 << n
        dp = [[0 for _ in range(states)] for _ in range(n + 1)]
        dp[0][0] = 1

        for i in range(1, n + 1):                   # 枚举第 i 个位置
            for state in range(states):             # 枚举所有状态
                one_num = bin(state).count("1")     # 计算当前状态中选择了多少个数字（即统计 1 的个数）
                if one_num != i:                    # 只有 i 与选择数字个数相同时才能计算
                    continue
                for k in range(1, n + 1):           # 枚举第 i 个位置（最后 1 位）上所选的数字
                    if state >> (k - 1) & 1 == 0:   # 只有 state 第 k 个位置上为 1 才表示选了该数字
                        continue
                    if k % i == 0 or i % k == 0:    # 只有满足整除关系才符合要求
                        # dp[i][state] 由前 i - 1 个位置，且 state 第 k 位为 0 的状态而来
                        dp[i][state] += dp[i - 1][state & (~(1 << (k - 1)))]

        return dp[i][states - 1]
```

### 思路 2：复杂度分析

- **时间复杂度**：$O(n^2 \times 2^n)$，其中 $n$ 为给定整数。
- **空间复杂度**：$O(n \times 2^n)$。

### 思路 3：状态压缩 DP + 优化

通过二维的「状态压缩 DP」可以看出，当我们在考虑第 $i$ 个位置时，其选择数字个数也应该为 $i$。

而我们可以根据 $state$ 中 $1$ 的个数来判断当前选择的数字个数，这样我们就可以减少用于枚举第 $i$ 个位置的循环，改用统计 $state$ 中 $1$ 的个数来判断前选择的数字个数或者说当前正在考虑的元素位置。

而这样，我们还可以进一步优化状态的定义，将二维的状态优化为一维的状态。具体做法如下：

###### 1. 划分阶段

按照数字集合的选择情况进行阶段划分。

###### 2. 定义状态

定义状态 $dp[state]$ 表示为：当数字集合的选择情况为 $state$ 时的方案数。

###### 3. 状态转移方程

对于状态 $state$，先统计出 $state$ 中选择的数字个数（即统计二进制中 $1$ 的个数）$one\underline{\hspace{0.5em}}num$。

则 $dp[state]$ 表示选择了前 $one\underline{\hspace{0.5em}}num$ 个数字，且选择情况为 $state$ 时的方案数。

$dp[state]$ 的状态肯定是由前 $one\underline{\hspace{0.5em}}num - 1$ 个数字，且 $state$ 第 $k$ 位为 $0$ 的状态而来对应状态转移而来，即：$dp[state \oplus (1 << (k - 1))]$。

所以状态转移方程为：$dp[state] = \sum_{k = 1}^n dp[state \oplus (1 << (k - 1))]$

###### 4. 初始条件

- 不考虑任何数的情况下，方案数为 $1$，即：$dp[0] = 1$。

###### 5. 最终结果

根据我们之前定义的状态，$dp[state]$ 表示为：当数字集合选择状态为 $state$ 时的方案数。所以最终结果为 $dp[states -  1]$，其中 $states = 1 << n$。

### 思路 3：代码

```python
class Solution:
    def countArrangement(self, n: int) -> int:
        states = 1 << n
        dp = [0 for _ in range(states)]
        dp[0] = 1

        for state in range(states):                         # 枚举所有状态
            one_num = bin(state).count("1")                 # 计算当前状态中选择了多少个数字（即统计 1 的个数）
            for k in range(1, n + 1):                       # 枚举最后 1 位上所选的数字
                if state >> (k - 1) & 1 == 0:               # 只有 state 第 k 个位置上为 1 才表示选了该数字
                    continue
                if one_num % k == 0 or k % one_num == 0:    # 只有满足整除关系才符合要求
                    # dp[state] 由前 one_num - 1 个位置，且 state 第 k 位为 0 的状态而来
                    dp[state] += dp[state ^ (1 << (k - 1))]

        return dp[states - 1]
```

### 思路 3：复杂度分析

- **时间复杂度**：$O(n \times 2^n)$，其中 $n$ 为给定整数。
- **空间复杂度**：$O(2^n)$。

## 参考资料

- 【题解】[【宫水三叶】详解两种状态压缩 DP 思路 - 优美的排列](https://leetcode.cn/problems/beautiful-arrangement/solution/gong-shui-san-xie-xiang-jie-liang-chong-vgsia/)
# [0530. 二叉搜索树的最小绝对差](https://leetcode.cn/problems/minimum-absolute-difference-in-bst/)

- 标签：树、深度优先搜索、广度优先搜索、二叉搜索树、二叉树
- 难度：

## 题目链接

- [0530. 二叉搜索树的最小绝对差 - 力扣](https://leetcode.cn/problems/minimum-absolute-difference-in-bst/)

## 题目大意

**描述**：给定一个二叉搜索树的根节点 $root$。

**要求**：返回树中任意两不同节点值之间的最小差值。

**说明**：

- **差值**：是一个正数，其数值等于两值之差的绝对值。
- 树中节点的数目范围是 $[2, 10^4]$。
- $0 \le Node.val \le 10^5$。

**示例**：

- 示例 1：

![](https://assets.leetcode.com/uploads/2021/02/05/bst1.jpg)

```python
输入：root = [4,2,6,1,3]
输出：1
```

- 示例 2：

![](https://assets.leetcode.com/uploads/2021/02/05/bst2.jpg)

```python
输入：root = [1,0,48,null,null,12,49]
输出：1
```

## 解题思路

### 思路 1：中序遍历

先来看二叉搜索树的定义：

- 若左子树不为空，则左子树上所有节点值均小于它的根节点值；
- 若右子树不为空，则右子树上所有节点值均大于它的根节点值；
- 任意节点的左、右子树也分别为二叉搜索树。

题目要求二叉搜索树上任意两节点的差的绝对值的最小值。

二叉树的中序遍历顺序是：左 -> 根 -> 右，二叉搜索树的中序遍历最终得到就是一个升序数组。而升序数组中绝对值差的最小值就是比较相邻两节点差值的绝对值，找出其中最小值。

那么我们就可以先对二叉搜索树进行中序遍历，并保存中序遍历的结果。然后再比较相邻节点差值的最小值，从而找出最小值。

### 思路 1：代码

```Python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def inorderTraversal(self, root: TreeNode) -> List[int]:
        res = []
        def inorder(root):
            if not root:
                return
            inorder(root.left)
            res.append(root.val)
            inorder(root.right)

        inorder(root)
        return res

    def getMinimumDifference(self, root: Optional[TreeNode]) -> int:
        inorder = self.inorderTraversal(root)
        ans = float('inf')
        for i in range(1, len(inorder)):
            ans = min(ans, abs(inorder[i - 1] - inorder[i]))

        return ans
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(n)$，其中 $n$ 为二叉搜索树中的节点数量。
- **空间复杂度**：$O(n)$。

# [0538. 把二叉搜索树转换为累加树](https://leetcode.cn/problems/convert-bst-to-greater-tree/)

- 标签：树、深度优先搜索、二叉搜索树、二叉树
- 难度：中等

## 题目链接

- [0538. 把二叉搜索树转换为累加树 - 力扣](https://leetcode.cn/problems/convert-bst-to-greater-tree/)

## 题目大意

给定一棵二叉搜索树（BST）的根节点，且二叉搜索树的节点值各不相同。要求将其转化为「累加树」，使其每个节点 `node` 的新值等于原树中大于或等于 `node.val` 的值之和。

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

# [0539. 最小时间差](https://leetcode.cn/problems/minimum-time-difference/)

- 标签：数组、数学、字符串、排序
- 难度：中等

## 题目链接

- [0539. 最小时间差 - 力扣](https://leetcode.cn/problems/minimum-time-difference/)

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

# [0542. 01 矩阵](https://leetcode.cn/problems/01-matrix/)

- 标签：广度优先搜索、数组、动态规划、矩阵
- 难度：中等

## 题目链接

- [0542. 01 矩阵 - 力扣](https://leetcode.cn/problems/01-matrix/)

## 题目大意

**描述**：给定一个 $m * n$ 大小的、由 `0` 和 `1` 组成的矩阵 $mat$。

**要求**：输出一个大小相同的矩阵 $res$，其中 $res[i][j]$ 表示对应位置元素（即 $mat[i][j]$）到最近的 $0$ 的距离。

**说明**：

- 两个相邻元素间的距离为 $1$。
- $m == mat.length$。
- $n == mat[i].length$。
- $1 \le m, n \le 10^4$。
- $1 \le m * n \le 10^4$。
- $mat[i][j] === 0$ 或者 $mat[i][j] == 1$。
- $mat$ 中至少有一个 $0$。

**示例**：

- 示例 1：

![](https://pic.leetcode-cn.com/1626667201-NCWmuP-image.png)

```python
输入：mat = [[0,0,0],[0,1,0],[0,0,0]]
输出：[[0,0,0],[0,1,0],[0,0,0]]
```

- 示例 2：

![](https://pic.leetcode-cn.com/1626667205-xFxIeK-image.png)

```python
输入：mat = [[0,0,0],[0,1,0],[1,1,1]]
输出：[[0,0,0],[0,1,0],[1,2,1]]
```

## 解题思路

### 思路 1：广度优先搜索

题目要求的是每个 `1` 到 `0`的最短曼哈顿距离。

比较暴力的做法是，从每个 `1` 开始进行广度优先搜索，每一步累积距离，当搜索到第一个 `0`，就是离这个 `1`  最近的 `0`，我们更新对应 `1` 位置上的答案距离。然后从下一个 `1` 开始进行广度优先搜索。

这样做每次进行广度优先搜索的时间复杂度为 $O(m \times n)$。对于 $m \times n$ 个节点来说，每个节点可能都要进行一次广度优先搜索，总的时间复杂度为 $O(m^2 \times n^2)$。时间复杂度太高了。

我们可以换个角度：求每个 `0` 到 `1` 的最短曼哈顿距离（和求每个 `1` 到 `0` 是等价的）。

我们将所有值为 `0` 的元素位置保存到队列中，然后对所有值为 `0` 的元素开始进行广度优先搜索，每搜一步距离加 `1`，当每次搜索到 `1` 时，就可以得到 `0` 到这个 `1` 的最短距离，也就是当前离这个 `1` 最近的 `0` 的距离。

这样对于所有节点来说，总共需要进行一次广度优先搜索就可以了，时间复杂度为 $O(m \times n)$。

具体步骤如下：

1. 使用一个集合变量 `visited` 存储所有值为 `0` 的元素坐标。使用队列变量 `queue` 存储所有值为 `0` 的元素坐标。使用二维数组 `res` 存储对应位置元素（即 $mat[i][j]$）到最近的 $0$ 的距离。
2. 我们从所有为如果队列 `queue` 不为空，则从队列中依次取出值为 `0` 的元素坐标，遍历其上、下、左、右位置。
3. 如果相邻区域未被访问过（说明遇到了值为 `1` 的元素），则更新相邻位置的距离值，并把相邻位置坐标加入队列 `queue` 和访问集合 `visited` 中。
4. 继续执行 2  ~ 3 步，直到队列为空时，返回 `res`。

### 思路 1：代码

```python
import collections

class Solution:
    def updateMatrix(self, mat: List[List[int]]) -> List[List[int]]:
        rows, cols = len(mat), len(mat[0])
        res = [[0 for _ in range(cols)] for _ in range(rows)]
        visited = set()

        for i in range(rows):
            for j in range(cols):
                if mat[i][j] == 0:
                    visited.add((i, j))
        
        directions = {(1, 0), (-1, 0), (0, 1), (0, -1)}
        queue = collections.deque(visited)

        while queue:
            i, j = queue.popleft()
            for direction in directions:
                new_i = i + direction[0]
                new_j = j + direction[1]
                if 0 <= new_i < rows and 0 <= new_j < cols and (new_i, new_j) not in visited:
                    res[new_i][new_j] = res[i][j] + 1
                    queue.append((new_i, new_j))
                    visited.add((new_i, new_j))
        return res
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(m \times n)$。
- **空间复杂度**：$O(m \times n)$。

# [0543. 二叉树的直径](https://leetcode.cn/problems/diameter-of-binary-tree/)

- 标签：树、深度优先搜索、二叉树
- 难度：简单

## 题目链接

- [0543. 二叉树的直径 - 力扣](https://leetcode.cn/problems/diameter-of-binary-tree/)

## 题目大意

**描述**：给一个二叉树的根节点 $root$。

**要求**：计算该二叉树的直径长度。

**说明**：

- **二叉树的直径长度**：二叉树中任意两个节点路径长度中的最大值。
- 两节点之间的路径长度是以它们之间边的数目表示。
- 这条路径可能穿过也可能不穿过根节点。

**示例**：

- 示例 1：

```python
给定二叉树：
          1
         / \
        2   3
       / \     
      4   5    
输出：3
解释：该二叉树的长度是路径 [4,2,1,3] 或者 [5,2,1,3]。
```

## 解题思路

### 思路 1：树形 DP + 深度优先搜索

这道题重点是理解直径长度的定义。「二叉树的直径长度」的定义为：二叉树中任意两个节点路径长度中的最大值。并且这条路径可能穿过也可能不穿过根节点。

对于根为 $root$ 的二叉树来说，其直径长度并不简单等于「左子树高度」加上「右子树高度」。

根据路径是否穿过根节点，我们可以将二叉树分为两种：

1. 直径长度所对应的路径穿过根节点。
2. 直径长度所对应的路径不穿过根节点。

我们来看下图中的两个例子。

![](https://qcdn.itcharge.cn/images/20230427111005.png)

如图所示，左侧这棵二叉树就是一棵常见的平衡二叉树，其直径长度所对应的路径是穿过根节点的（$D\rightarrow B \rightarrow A \rightarrow C$）。这种情况下：$\text{二叉树的直径} = \text{左子树高度} + \text{右子树高度}$。

而右侧这棵特殊的二叉树，其直径长度所对应的路径是没有穿过根节点的（$F \rightarrow D \rightarrow B \rightarrow E \rightarrow G$）。这种情况下：$\text{二叉树的直径} = \text{所有子树中最大直径长度}$。

也就是说根为 $root$ 的二叉树的直径长度可能来自于  $\text{左子树高度} + \text{右子树高度}$，也可能来自于 $\text{子树中的最大直径}$，即 $\text{二叉树的直径} = max(\text{左子树高度} + \text{右子树高度}, \quad \text{所有子树中最大直径长度})$。

那么现在问题就变成为如何求「子树的高度」和「子树中的最大直径」。

1. 子树的高度：我们可以利用深度优先搜索方法，递归遍历左右子树，并分别返回左右子树的高度。
2. 子树中的最大直径：我们可以在递归求解子树高度的时候维护一个 $ans$ 变量，用于记录所有 $\text{左子树高度} + \text{右子树高度}$ 中的最大值。

最终 $ans$ 就是我们所求的该二叉树的最大直径，将其返回即可。

### 思路 1：代码

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def __init__(self):
        self.ans = 0

    def dfs(self, node):
        if not node:
            return 0
        left_height = self.dfs(node.left)                     # 左子树高度
        right_height = self.dfs(node.right)                   # 右子树高度
        self.ans = max(self.ans, left_height + right_height)  # 维护所有路径中的最大直径
        return max(left_height, right_height) + 1             # 返回该节点的高度 = 左右子树最大高度 + 1

    def diameterOfBinaryTree(self, root: Optional[TreeNode]) -> int:
        self.dfs(root)
        return self.ans
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(n)$，其中 $n$ 是二叉树的节点数目。
- **空间复杂度**：$O(n)$。递归函数需要用到栈空间，栈空间取决于递归深度，最坏情况下递归深度为 $n$，所以空间复杂度为 $O(n)$。

# [0546. 移除盒子](https://leetcode.cn/problems/remove-boxes/)

- 标签：记忆化搜索、数组、动态规划
- 难度：困难

## 题目链接

- [0546. 移除盒子 - 力扣](https://leetcode.cn/problems/remove-boxes/)

## 题目大意

**描述**：给定一个代表不同颜色盒子的正数数组 $boxes$，盒子的颜色由不同正数组成，其中 $boxes[i]$ 表示第 $i$ 个盒子的颜色。

我们将经过若干轮操作去去掉盒子，直到所有盒子都去掉为止。每一轮我们可以移除具有相同颜色的连续 $k$ 个盒子（$k \ge 1$），这样一轮之后，我们将获得 $k \times k$ 个积分。

**要求**：返回我们能获得的最大积分和。

**说明**：

- $1 \le boxes.length \le 100$。
- $1 \le boxes[i] \le 100$。

**示例**：

- 示例 1：

```python
输入：boxes = [1,3,2,2,2,3,4,3,1]
输出：23
解释：
[1, 3, 2, 2, 2, 3, 4, 3, 1] 
----> [1, 3, 3, 4, 3, 1] (3*3=9 分) 
----> [1, 3, 3, 3, 1] (1*1=1 分) 
----> [1, 1] (3*3=9 分) 
----> [] (2*2=4 分)
```

- 示例 2：

```python
输入：boxes = [1,1,1]
输出：9
```

## 解题思路

### 思路 1：动态规划

对于每个盒子，

如果使用二维状态 $dp[i][j]$ 表示为：移除区间 $[i, j]$ 之间的盒子，所能够得到的最大积分和。但实际上，移除区间 $[i, j]$ 之间盒子，所能得到的最大积分和，并不只依赖于子区间，也依赖于之前移除其他区间对当前区间的影响。比如当前区间的某个值和其他区间的相同值连起来可以获得更高的额分数。

因此，我们需要再二维状态的基础上，增加更多维数的状态。

对于当前区间 $[i, j]$，我们需要凑一些尽可能长的同色盒子一起消除，从而获得更高的分数。我们不妨每次都选择消除区间 $[i, j]$ 中最后一个盒子 $boxes[j]$，并且记录 $boxes[j]$ 之后与 $boxes[j]$ 颜色相同的盒子数量。

###### 1. 划分阶段

按照区间长度进行阶段划分。

###### 2. 定义状态

定义状态 $dp[i][j][k]$ 表示为：移除区间 $[i, j]$ 之间的盒子，并且区间右侧有 $k$ 个与 $boxes[j]$ 颜色相同的盒子，所能够得到的最大积分和。

###### 3. 状态转移方程

- 当区间长度为 $1$ 时，当前区间只有一个盒子，区间末尾有 $k$ 个与 $boxes[j]$ 颜色相同的盒子，所能够得到的最大积分为 $(k + 1) \times (k + 1)$。
- 当区间长度大于 $1$ 时，对于区间末尾的 $k$ 个与 $boxes[j]$ 颜色相同的盒子，有两种处理方式：
  - 将末尾的盒子移除，所能够得到的最大积分为：移除末尾盒子之前能够获得的最大积分和，再加上本轮移除末尾盒子能够获得的积分和，即：$dp[i][j - 1][0] + (k + 1) \times (k + 1)$。
  - 在区间中找到一个位置 $t$，使得第 $t$ 个盒子与第 $j$ 个盒子颜色相同，先将区间 $[t + 1, j - 1]$ 的盒子消除，然后继续凑同色盒子，即：$dp[t + 1][j - 1][0] + dp[i][t][k + 1]$。

###### 4. 初始条件

- 区间长度为 $1$ 时，当前区间只有一个盒子，区间末尾有 $k$ 个与 $boxes[j]$ 颜色相同的盒子，所能够得到的最大积分为 $(k + 1) \times (k + 1)$。

###### 5. 最终结果

根据我们之前定义的状态，$dp[i][j][k]$ 表示为：移除区间 $[i, j]$ 之间的盒子，并且区间右侧有 $k$ 个与 $boxes[j]$ 颜色相同的盒子，所能够得到的最大积分和。所以最终结果为 $dp[0][size - 1][0]$。

### 思路 1：代码

```python
class Solution:
    def removeBoxes(self, boxes: List[int]) -> int:
        size = len(boxes)

        dp = [[[0 for _ in range(size)] for _ in range(size)] for _ in range(size)]
        for l in range(1, size + 1):
            for i in range(size):
                j = i + l - 1
                if j >= size:
                    break

                for k in range(size - j):
                    if l == 1:
                        dp[i][j][k] = max(dp[i][j][k], (k + 1) * (k + 1))
                    else:
                        dp[i][j][k] = max(dp[i][j][k], dp[i][j - 1][0] + (k + 1) * (k + 1))
                    for t in range(i, j):
                        if boxes[t] == boxes[j]:
                            dp[i][j][k] = max(dp[i][j][k], dp[t + 1][j - 1][0] + dp[i][t][k + 1])

        return dp[0][size - 1][0]
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(n^4)$，其中 $n$ 为数组 $boxes$ 的元素个数。
- **空间复杂度**：$O(n^3)$。

# [0547. 省份数量](https://leetcode.cn/problems/number-of-provinces/)

- 标签：深度优先搜索、广度优先搜索、并查集、图
- 难度：中等

## 题目链接

- [0547. 省份数量 - 力扣](https://leetcode.cn/problems/number-of-provinces/)

## 题目大意

**描述**：有 `n` 个城市，其中一些彼此相连，另一些没有相连。如果城市 `a` 与城市 `b` 直接相连，且城市 `b` 与城市 `c` 直接相连，那么城市 `a` 与城市 `c` 间接相连。

「省份」是由一组直接或间接链接的城市组成，组内不含有其他没有相连的城市。

现在给定一个 `n * n` 的矩阵 `isConnected` 表示城市的链接关系。其中 `isConnected[i][j] = 1` 表示第 `i` 个城市和第 `j` 个城市直接相连，`isConnected[i][j] = 0` 表示第 `i` 个城市和第 `j` 个城市没有相连。

**要求**：根据给定的城市关系，返回「省份」的数量。

**说明**：

- $1 \le n \le 200$。
- $n == isConnected.length$。
- $n == isConnected[i].length$。
- $isConnected[i][j]$ 为 $1$ 或 $0$。
- $isConnected[i][i] == 1$。
- $isConnected[i][j] == isConnected[j][i]$。

**示例**：

- 示例 1：

![](https://assets.leetcode.com/uploads/2020/12/24/graph1.jpg)

```python
输入：isConnected = [[1,1,0],[1,1,0],[0,0,1]]
输出：2
```

- 示例 2：

![](https://assets.leetcode.com/uploads/2020/12/24/graph2.jpg)

```python
输入：isConnected = [[1,0,0],[0,1,0],[0,0,1]]
输出：3
```

## 解题思路

### 思路 1：并查集

1. 遍历矩阵 `isConnected`。如果 `isConnected[i][j] == 1`，将 `i` 节点和 `j` 节点相连。
2. 然后判断每个城市节点的根节点，然后统计不重复的根节点有多少个，即为「省份」的数量。

### 思路 1：代码

```python
class UnionFind:
    def __init__(self, n):                          # 初始化
        self.fa = [i for i in range(n)]             # 每个元素的集合编号初始化为数组 fa 的下标索引
    
    def find(self, x):                              # 查找元素根节点的集合编号内部实现方法
        while self.fa[x] != x:                      # 递归查找元素的父节点，直到根节点
            self.fa[x] = self.fa[self.fa[x]]        # 隔代压缩优化
            x = self.fa[x]
        return x                                    # 返回元素根节点的集合编号

    def union(self, x, y):                          # 合并操作：令其中一个集合的树根节点指向另一个集合的树根节点
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x == root_y:                        # x 和 y 的根节点集合编号相同，说明 x 和 y 已经同属于一个集合
            return False
        self.fa[root_x] = root_y                    # x 的根节点连接到 y 的根节点上，成为 y 的根节点的子节点
        return True

    def is_connected(self, x, y):                   # 查询操作：判断 x 和 y 是否同属于一个集合
        return self.find(x) == self.find(y)

class Solution:
    def findCircleNum(self, isConnected: List[List[int]]) -> int:
        size = len(isConnected)
        union_find = UnionFind(size)
        for i in range(size):
            for j in range(i + 1, size):
                if isConnected[i][j] == 1:
                    union_find.union(i, j)

        res = set()
        for i in range(size):
            res.add(union_find.find(i))
        return len(res)
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(n^2 \times \alpha(n))$。其中 $n$ 是城市的数量，$\alpha$ 是反 `Ackerman` 函数。
- **空间复杂度**：$O(n)$。# [0557. 反转字符串中的单词 III](https://leetcode.cn/problems/reverse-words-in-a-string-iii/)

- 标签：双指针、字符串
- 难度：简单

## 题目链接

- [0557. 反转字符串中的单词 III - 力扣](https://leetcode.cn/problems/reverse-words-in-a-string-iii/)

## 题目大意

**描述**：给定一个字符串 `s`。

**要求**：将字符串中每个单词的字符顺序进行反装，同时仍保留空格和单词的初始顺序。

**说明**：

- $1 \le s.length \le 5 * 10^4$。
- `s` 包含可打印的 ASCII 字符。
- `s` 不包含任何开头或结尾空格。
- `s` 里至少有一个词。
- `s` 中的所有单词都用一个空格隔开。

**示例**：

- 示例 1：

```python
输入：s = "Let's take LeetCode contest"
输出："s'teL ekat edoCteeL tsetnoc"
```

- 示例 2：

```python
输入： s = "God Ding"
输出："doG gniD"
```

## 解题思路

### 思路 1：使用额外空间

因为 Python 的字符串是不可变的，所以在原字符串空间上进行切换顺序操作肯定是不可行的了。但我们可以利用切片方法。

1. 将字符串按空格进行分割，分割成一个个的单词。
2. 再将每个单词进行反转。
3. 最后将每个单词连接起来。

### 思路 1：代码

```python
class Solution:
    def reverseWords(self, s: str) -> str:
        return " ".join(word[::-1] for word in s.split(" "))
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(n)$。
- **空间复杂度**：$O(n)$。# [0560. 和为 K 的子数组](https://leetcode.cn/problems/subarray-sum-equals-k/)

- 标签：数组、哈希表、前缀和
- 难度：中等

## 题目链接

- [0560. 和为 K 的子数组 - 力扣](https://leetcode.cn/problems/subarray-sum-equals-k/)

## 题目大意

**描述**：给定一个整数数组 $nums$ 和一个整数 $k$。

**要求**：找到该数组中和为 $k$ 的连续子数组的个数。

**说明**：

- $1 \le nums.length \le 2 \times 10^4$。
- $-1000 \le nums[i] \le 1000$。
  $-10^7 \le k \le 10^7$。

**示例**：

- 示例 1：

```python
输入：nums = [1,1,1], k = 2
输出：2
```

- 示例 2：

```python
输入：nums = [1,2,3], k = 3
输出：2
```

## 解题思路

### 思路 1：枚举算法（超时）

先考虑暴力做法，外层两重循环，遍历所有连续子数组，然后最内层再计算一下子数组的和。部分代码如下：

```python
for i in range(len(nums)):
    for j in range(i + 1):
        sum = countSum(i, j)
```

这样下来时间复杂度就是 $O(n^3)$ 了。下一步是想办法降低时间复杂度。

对于以 $i$ 开头，以 $j$ 结尾（$i \le j$）的子数组 $nums[i]…nums[j]$ 来说，我们可以通过顺序遍历 $j$，逆序遍历 $i$ 的方式（或者前缀和的方式），从而在 $O(n^2)$ 的时间复杂度内计算出子数组的和，同时使用变量 $cnt$ 统计出和为 $k$ 的子数组个数。

但这样提交上去超时了。

### 思路 1：代码

```python
class Solution:
    def subarraySum(self, nums: List[int], k: int) -> int:
        cnt = 0
        for j in range(len(nums)):
            sum = 0
            for i in range(j, -1, -1):
                sum += nums[i]
                if sum == k:
                    cnt += 1
        
        return cnt
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(n^2)$。
- **空间复杂度**：$O(1)$。

### 思路 2：前缀和 + 哈希表

先用一重循环遍历数组，计算出数组 $nums$ 中前 $j$ 个元素的和（前缀和），保存到一维数组 $pre\underline{\hspace{0.5em}}sum$ 中，那么对于任意 $nums[i]…nums[j]$ 的子数组的和为 $pre\underline{\hspace{0.5em}}sum[j] - pre\underline{\hspace{0.5em}}sum[i - 1]$。这样计算子数组和的时间复杂度降为了 $O(1)$。总体时间复杂度为 $O(n^2)$。

但是还是超时了。。

由于我们只关心和为 $k$ 出现的次数，不关心具体的解，可以使用哈希表来加速运算。

$pre\underline{\hspace{0.5em}}sum[i]$ 的定义是前 $i$ 个元素和，则 $pre\underline{\hspace{0.5em}}sum[i]$ 可以由 $pre\underline{\hspace{0.5em}}sum[i - 1]$ 递推而来，即：$pre\underline{\hspace{0.5em}}sum[i] = pre\underline{\hspace{0.5em}}sum[i - 1] + num[i]$。 $[i..j]$ 子数组和为 $k$ 可以转换为：$pre\underline{\hspace{0.5em}}sum[j] - pre\underline{\hspace{0.5em}}sum[i - 1] == k$。

综合一下，可得：$pre\underline{\hspace{0.5em}}sum[i - 1] == pre\underline{\hspace{0.5em}}sum[j] - k $。

所以，当我们考虑以 $j$ 结尾和为 $k$ 的连续子数组个数时，只需要统计有多少个前缀和为 $pre\underline{\hspace{0.5em}}sum[j] - k$ （即 $pre\underline{\hspace{0.5em}}sum[i - 1]$）的个数即可。具体做法如下：

- 使用 $pre\underline{\hspace{0.5em}}sum$ 变量记录前缀和（代表 $pre\underline{\hspace{0.5em}}sum[j]$）。
- 使用哈希表 $pre\underline{\hspace{0.5em}}dic$ 记录 $pre\underline{\hspace{0.5em}}sum[j]$ 出现的次数。键值对为 $pre\underline{\hspace{0.5em}}sum[j] : pre\underline{\hspace{0.5em}}sum\underline{\hspace{0.5em}}count$。
- 从左到右遍历数组，计算当前前缀和 $pre\underline{\hspace{0.5em}}sum$。
- 如果 $pre\underline{\hspace{0.5em}}sum - k$ 在哈希表中，则答案个数累加上 $pre\underline{\hspace{0.5em}}dic[pre\underline{\hspace{0.5em}}sum - k]$。
- 如果 $pre\underline{\hspace{0.5em}}sum$ 在哈希表中，则前缀和个数累加 $1$，即 $pre\underline{\hspace{0.5em}}dic[pre\underline{\hspace{0.5em}}sum] += 1$。
- 最后输出答案个数。

### 思路 2：代码

```python
class Solution:
    def subarraySum(self, nums: List[int], k: int) -> int:
        pre_dic = {0: 1}
        pre_sum = 0
        count = 0
        for num in nums:
            pre_sum += num
            if pre_sum - k in pre_dic:
                count += pre_dic[pre_sum - k]
            if pre_sum in pre_dic:
                pre_dic[pre_sum] += 1
            else:
                pre_dic[pre_sum] = 1
        return count
```

### 思路 2：复杂度分析

- **时间复杂度**：$O(n)$。
- **空间复杂度**：$O(n)$。

# [0561. 数组拆分](https://leetcode.cn/problems/array-partition/)

- 标签：贪心、数组、计数排序、排序
- 难度：简单

## 题目链接

- [0561. 数组拆分 - 力扣](https://leetcode.cn/problems/array-partition/)

## 题目大意

**描述**：给定一个长度为 $2 \times n$ 的整数数组 $nums$。

**要求**：将数组中的数拆分成 $n$ 对，每对数求最小值，求 $n$ 对数最小值的最大总和是多少。

**说明**：

- $1 \le n \le 10^4$。
- $nums.length == 2 * n$。
- $-10^4 \le nums[i] \le 10^4$。

**示例**：

- 示例 1：

```python
输入：nums = [1,4,3,2]
输出：4
解释：所有可能的分法（忽略元素顺序）为：
1. (1, 4), (2, 3) -> min(1, 4) + min(2, 3) = 1 + 2 = 3
2. (1, 3), (2, 4) -> min(1, 3) + min(2, 4) = 1 + 2 = 3
3. (1, 2), (3, 4) -> min(1, 2) + min(3, 4) = 1 + 3 = 4
所以最大总和为 4
```
- 示例 2：

```python
输入：nums = [6,2,6,5,1,2]
输出：9
解释：最优的分法为 (2, 1), (2, 5), (6, 6). min(2, 1) + min(2, 5) + min(6, 6) = 1 + 2 + 6 = 9
```

## 解题思路

### 思路 1：计数排序

因为 $nums[i]$ 的范围为 $[-10^4, 10^4]$，范围不是很大，所以我们可以使用计数排序算法先将数组 $nums$ 进行排序。

要想每对数最小值的总和最大，就得使每对数的最小值尽可能大。只有让较大的数与较大的数一起组合，较小的数与较小的数一起结合，才能才能使总和最大。所以，排序完之后将相邻两个元素的最小值进行相加，即得到结果。

###  思路 1：代码

```python
class Solution:
    def countingSort(self, nums: [int]) -> [int]:
        # 计算待排序数组中最大值元素 nums_max 和最小值元素 nums_min
        nums_min, nums_max = min(nums), max(nums)
        # 定义计数数组 counts，大小为 最大值元素 - 最小值元素 + 1
        size = nums_max - nums_min + 1
        counts = [0 for _ in range(size)]
        
        # 统计值为 num 的元素出现的次数
        for num in nums:
            counts[num - nums_min] += 1
        
        # 生成累积计数数组
        for i in range(1, size):
            counts[i] += counts[i - 1]

        # 反向填充目标数组
        res = [0 for _ in range(len(nums))]
        for i in range(len(nums) - 1, -1, -1):
            num = nums[i]
            # 根据累积计数数组，将 num 放在数组对应位置
            res[counts[num - nums_min] - 1] = num
            # 将 num 的对应放置位置减 1，从而得到下个元素 num 的放置位置
            counts[nums[i] - nums_min] -= 1

        return res

    def arrayPairSum(self, nums: List[int]) -> int:
        nums = self.countingSort(nums)
        return sum(nums[::2])
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(n + k)$，其中 $k$ 代表数组 $nums$ 的值域。
- **空间复杂度**：$O(k)$。

### 思路 2：排序

要想每对数最小值的总和最大，就得使每对数的最小值尽可能大。只有让较大的数与较大的数一起组合，较小的数与较小的数一起结合，才能才能使总和最大。

1. 对 $nums$ 进行排序。
2. 将相邻两个元素的最小值进行相加，即得到结果。

### 思路 1：代码

```python
class Solution:
    def arrayPairSum(self, nums: List[int]) -> int:
        nums.sort()
        return sum(nums[::2])
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(n \times \log n)$。
- **空间复杂度**：$O(1)$。

# [0567. 字符串的排列](https://leetcode.cn/problems/permutation-in-string/)

- 标签：哈希表、双指针、字符串、滑动窗口
- 难度：中等

## 题目链接

- [0567. 字符串的排列 - 力扣](https://leetcode.cn/problems/permutation-in-string/)

## 题目大意

**描述**：给定两个字符串 $s1$ 和 $s2$ 。

**要求**：判断 $s2$ 是否包含 $s1$ 的排列。如果包含，返回 $True$；否则，返回 $False$。

**说明**：

- $1 \le s1.length, s2.length \le 10^4$。
- $s1$ 和 $s2$ 仅包含小写字母。

**示例**：

- 示例 1：

```python
输入：s1 = "ab" s2 = "eidbaooo"
输出：true
解释：s2 包含 s1 的排列之一 ("ba").
```

- 示例 2：

```python
输入：s1= "ab" s2 = "eidboaoo"
输出：False
```

## 解题思路

### 思路 1：滑动窗口

题目要求判断 $s2$ 是否包含 $s1$ 的排列，则 $s2$ 的子串长度等于 $s1$ 的长度。我们可以维护一个长度为字符串 $s1$ 长度的固定长度的滑动窗口。

先统计出字符串  $s1$ 中各个字符的数量，我们用 $s1\underline{\hspace{0.5em}}count$ 来表示。这个过程可以用字典、数组来实现，也可以直接用 `collections.Counter()` 实现。再统计 $s2$ 对应窗口内的字符数量 $window\underline{\hspace{0.5em}}count$，然后不断向右滑动，然后进行比较。如果对应字符数量相同，则返回 $True$，否则继续滑动。直到末尾时，返回 $False$。整个解题步骤具体如下：

1. $s1\underline{\hspace{0.5em}}count$ 用来统计 $s1$ 中各个字符数量。$window\underline{\hspace{0.5em}}count$ 用来维护窗口中 $s2$ 对应子串的各个字符数量。$window\underline{\hspace{0.5em}}size$ 表示固定窗口的长度，值为 $len(s1)$。
2. 先统计出 $s1$ 中各个字符数量。
3. $left$ 、$right$ 都指向序列的第一个元素，即：`left = 0`，`right = 0`。
4. 向右移动 $right$，先将 $len(s1)$ 个元素填入窗口中。
5. 当窗口元素个数为 $window\underline{\hspace{0.5em}}size$ 时，即：$right - left + 1 \ge window\underline{\hspace{0.5em}}size$ 时，判断窗口内各个字符数量 $window\underline{\hspace{0.5em}}count$ 是否等于 $s1 $ 中各个字符数量 $s1\underline{\hspace{0.5em}}count$。
   1. 如果等于，直接返回 $True$。
   2. 如果不等于，则向右移动 $left$，从而缩小窗口长度，即 `left += 1`，使得窗口大小始终保持为 $window\underline{\hspace{0.5em}}size$。
6. 重复 $4 \sim 5$ 步，直到 $right$ 到达数组末尾。返回 $False$。

### 思路 1：代码

```python
import collections

class Solution:
    def checkInclusion(self, s1: str, s2: str) -> bool:
        left, right = 0, 0
        s1_count = collections.Counter(s1)
        window_count = collections.Counter()
        window_size = len(s1)

        while right < len(s2):
            window_count[s2[right]] += 1

            if right - left + 1 >= window_size:
                if window_count == s1_count:
                    return True
                window_count[s2[left]] -= 1
                if window_count[s2[left]] == 0:
                    del window_count[s2[left]]
                left += 1
            right += 1
        return False
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(n + m + |\sum|)$，其中 $n$、$m$ 分别是字符串 $s1$、$s2$ 的长度，$\sum$ 是字符集，本题中 $|\sum| = 26$。
- **空间复杂度**：$O(|\sum|)$。

# [0575. 分糖果](https://leetcode.cn/problems/distribute-candies/)

- 标签：数组、哈希表
- 难度：简单

## 题目链接

- [0575. 分糖果 - 力扣](https://leetcode.cn/problems/distribute-candies/)

## 题目大意

给定一个偶数长度为 `n` 的数组，其中不同的数字代表不同种类的糖果，每一个数字代表一个糖果。

要求：将这些糖果按种类平均分为一个弟弟和一个妹妹。返回妹妹可以获得的最大糖果的种类数。

## 解题思路

`n` 个糖果分为两个人，每个人最多只能得到 `n // 2` 个糖果。假设糖果种数为 `m`。则如果糖果种类数大于糖果总数的一半，即 `m > n // 2`，则返回糖果数量的一半就好，也就说糖果总数一半的糖果都可以是不同种类的糖果。妹妹能获得最多 `n // 2` 种糖果。而如果让给种类数小于等于糖果总数的一半，即 `m <= n // 2`，则返回种类数，也就是说妹妹可以最多获得 `m` 种糖果。

综合这两种情况，其最终结果就是 `ans = min(m, n // 2)`。

计算糖果种类可以用 set 集合来做。

## 代码

```python
class Solution:
    def distributeCandies(self, candyType: List[int]) -> int:
        candy_set = set(candyType)
        return min(len(candyType) // 2, len(candy_set))
```

# [0576. 出界的路径数](https://leetcode.cn/problems/out-of-boundary-paths/)

- 标签：动态规划
- 难度：中等

## 题目链接

- [0576. 出界的路径数 - 力扣](https://leetcode.cn/problems/out-of-boundary-paths/)

## 题目大意

**描述**：有一个大小为 $m \times n$ 的网络和一个球。球的起始位置为 $(startRow, startColumn)$。你可以将球移到在四个方向上相邻的单元格内（可以穿过网格边界到达网格之外）。最多可以移动 $maxMove$ 次球。

现在给定五个整数 $m$、$n$、$maxMove$、$startRow$ 以及 $startColumn$。

**要求**：找出并返回可以将球移出边界的路径数量。因为答案可能非常大，返回对 $10^9 + 7$ 取余后的结果。

**说明**：

- $1 \le m, n \le 50$。
- $0 \le maxMove \le 50$。
- $0 \le startRow < m$。
- $0 \le startColumn < n$。

**示例**：

- 示例 1：

```python
输入：m = 2, n = 2, maxMove = 2, startRow = 0, startColumn = 0
输出：6
```

![](https://assets.leetcode.com/uploads/2021/04/28/out_of_boundary_paths_1.png)

## 解题思路

### 思路 1：记忆化搜索

1. 问题的状态定义为：从位置 $(i, j)$ 出发，最多使用 $moveCount$ 步，可以将球移出边界的路径数量。
2. 定义一个 $m \times n \times (maxMove + 1)$ 的三维数组 $memo$ 用于记录已经计算过的路径数量。
3. 定义递归函数 $dfs(i, j, moveCount)$ 用于计算路径数量。
   1. 如果 $(i, j)$ 已经出界，则说明找到了一条路径，返回方案数为 $1$。
   2. 如果没有移动次数了，则返回方案数为 $0$。
   3. 定义方案数 $ans$，遍历四个方向，递归计算四个方向的方案数，累积到 $ans$ 中，并进行取余。
   4. 返回方案数 $ans$。
4. 调用递归函数 $dfs(startRow, startColumn, maxMove)$，并将其返回值作为答案进行返回。

### 思路 1：代码

```python
class Solution:
    def findPaths(self, m: int, n: int, maxMove: int, startRow: int, startColumn: int) -> int:
        directions = {(1, 0), (-1, 0), (0, 1), (0, -1)}
        mod = 10 ** 9 + 7

        memo = [[[-1 for _ in range(maxMove + 1)] for _ in range(n)] for _ in range(m)]

        def dfs(i, j, moveCount):
            if i < 0 or i >= m or j < 0 or j >= n:
                return 1
            
            if moveCount == 0:
                return 0

            if memo[i][j][moveCount] != -1:
                return memo[i][j][moveCount]

            ans = 0
            for direction in directions:
                new_i = i + direction[0]
                new_j = j + direction[1]
                ans += dfs(new_i, new_j, moveCount - 1)
                ans %= mod

            memo[i][j][moveCount] = ans
            return ans

        return dfs(startRow, startColumn, maxMove)
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(m \times n \times maxMove)$。
- **空间复杂度**：$O(m \times n \times maxMove)$。

### 思路 2：动态规划

我们需要统计从 $(startRow, startColumn)$ 位置出发，最多移动 $maxMove$ 次能够穿过边界的所有路径数量。则我们可以根据位置和移动步数来划分阶段和定义状态。

###### 1. 划分阶段

按照位置进行阶段划分。

###### 2. 定义状态

定义状态 $dp[i][j][k]$ 表示为：从位置 $(i, j)$ 最多移动 $k$ 次最终穿过边界的所有路径数量。

###### 3. 状态转移方程

因为球可以在上下左右四个方向上进行移动，所以对于位置 $(i, j)$，最多移动 $k$ 次最终穿过边界的所有路径数量取决于周围四个方向上最多经过 $k - 1$ 次穿过对应位置上的所有路径数量和。

即：$dp[i][j][k] = dp[i - 1][j][k - 1] + dp[i + 1][j][k - 1] + dp[i][j - 1][k - 1] + dp[i][j + 1][k - 1]$。

###### 4. 初始条件

如果位置 $[i, j]$ 已经处于边缘，只差一步就穿过边界。则此时位置 $(i, j)$ 最多移动 $k$ 次最终穿过边界的所有路径数量取决于有相邻多少个方向是边界。也可以通过对上面 $(i - 1, j)$、$(i + 1, j)$、$(i, j - 1)$、$(i, j + 1)$ 是否已经穿过边界进行判断（每一个方向穿过一次，就累积一次），来计算路径数目。然后将其作为初始条件。

###### 5. 最终结果

根据我们之前定义的状态，$dp[i][j][k]$ 表示为：从位置 $(i, j)$ 最多移动 $k$ 次最终穿过边界的所有路径数量。则最终答案为 $dp[startRow][startColumn][maxMove]$。

### 思路 2：动态规划代码

```python
class Solution:
    def findPaths(self, m: int, n: int, maxMove: int, startRow: int, startColumn: int) -> int:
        directions = {(1, 0), (-1, 0), (0, 1), (0, -1)}
        mod = 10 ** 9 + 7
        
        dp = [[[0 for _ in range(maxMove + 1)] for _ in range(n)] for _ in range(m)]
        for i in r
        for k in range(1, maxMove + 1):
            for i in range(m):
                for j in range(n):
                    for direction in directions:
                        new_i = i + direction[0]
                        new_j = j + direction[1]
                        if 0 <= new_i < m and 0 <= new_j < n:
                            dp[i][j][k] = (dp[i][j][k] + dp[new_i][new_j][k - 1]) % mod
                        else:
                            dp[i][j][k] = (dp[i][j][k] + 1) % mod
        
        return dp[startRow][startColumn][maxMove]
```

### 思路 2：复杂度分析

- **时间复杂度**：$O(m \times n \times maxMove)$。三重循环遍历的时间复杂度为 $O(m \times n \times maxMove)$。
- **空间复杂度**：$O(m \times n \times maxMove)$。使用了三维数组保存状态，所以总体空间复杂度为 $O(m \times n \times maxMove)$。
# [0583. 两个字符串的删除操作](https://leetcode.cn/problems/delete-operation-for-two-strings/)

- 标签：字符串、动态规划
- 难度：中等

## 题目链接

- [0583. 两个字符串的删除操作 - 力扣](https://leetcode.cn/problems/delete-operation-for-two-strings/)

## 题目大意

给定两个单词 `word1` 和 `word2`，找到使得 `word1` 和 `word2` 相同所需的最小步数，每步可以删除任意一个字符串中的一个字符。

## 解题思路

动态规划求解。

先定义状态 `dp[i][j]` 为以 `i - 1` 为结尾的字符串 `word1` 和以 `j - 1` 字结尾的字符串 `word2` 想要达到相等，所需要删除元素的最少次数。

然后确定状态转移方程。

- 如果 `word1[i - 1] == word2[j - 1]`，`dp[i][j]` 取源于以 `i - 2` 结尾结尾的字符串 `word1` 和以 `j - 1` 结尾的字符串 `word2`，即 `dp[i][j] = dp[i - 1][j - 1]`。
- 如果 `word1[i - 1] != word2[j - 1]`，`dp[i][j]` 取源于以下三种情况中的最小情况：
  - 删除 `word1[i - 1]`，最少操作次数为：`dp[i - 1][j] + 1`。
  - 删除 `word2[j - 1]`，最少操作次数为：`dp[i][j - 1] + 1`。
  - 同时删除 `word1[i - 1]`、`word2[j - 1]`，最少操作次数为 `dp[i - 1][j - 1] + 2`。

然后确定一下边界条件。

- 当 `word1` 为空字符串，以 `j - 1` 结尾的字符串 `word2` 要删除 `j` 个字符才能和 `word1` 相同，即 `dp[0][j] = j`。
- 当 `word2` 为空字符串，以 `i - 1` 结尾的字符串 `word1` 要删除 `i` 个字符才能和 `word2` 相同，即 `dp[i][0] = i`。

最后递推求解，最终输出 `dp[size1][size2]` 为答案。

## 代码

```python
class Solution:
    def minDistance(self, word1: str, word2: str) -> int:
        size1 = len(word1)
        size2 = len(word2)
        dp = [[0 for _ in range(size2 + 1)] for _ in range(size1 + 1)]

        for i in range(size1 + 1):
            dp[i][0] = i
        for j in range(size2 + 1):
            dp[0][j] = j

        for i in range(1, size1 + 1):
            for j in range(1, size2 + 1):
                if word1[i - 1] == word2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = min(dp[i - 1][j - 1] + 2, dp[i - 1][j] + 1, dp[i][j - 1] + 1)

        return dp[size1][size2]
```

# [0589. N 叉树的前序遍历](https://leetcode.cn/problems/n-ary-tree-preorder-traversal/)

- 标签：栈、树、深度优先搜索
- 难度：简单

## 题目链接

- [0589. N 叉树的前序遍历 - 力扣](https://leetcode.cn/problems/n-ary-tree-preorder-traversal/)

## 题目大意

给定一棵 N 叉树的根节点 `root`。

要求：返回其节点值的前序遍历。

进阶：使用迭代法完成。

## 解题思路

递归法很好写。迭代法需要借助于栈。

- 用栈保存根节点 `root`。然后遍历栈。
- 循环判断栈是否为空。
- 如果栈不为空，取出栈顶节点，将节点值加入答案数组。
- 逆序遍历栈顶节点的子节点，将其依次放入栈中（逆序保证取出顺序为正）。
- 然后继续第 2 ~ 4 步，直到栈为空。

最后输出答案数组。

## 代码

```python
class Solution:
    def preorder(self, root: 'Node') -> List[int]:
        res = []
        stack = []
        if not root:
            return res
        stack.append(root)
        while stack:
            node = stack.pop()
            res.append(node.val)
            for i in range(len(node.children) - 1, -1, -1):
                if node.children[i]:
                    stack.append(node.children[i])
        return res
```

# [0590. N 叉树的后序遍历](https://leetcode.cn/problems/n-ary-tree-postorder-traversal/)

- 标签：栈、树、深度优先搜索
- 难度：简单

## 题目链接

- [0590. N 叉树的后序遍历 - 力扣](https://leetcode.cn/problems/n-ary-tree-postorder-traversal/)

## 题目大意

给定一个 N 叉树的根节点 `root`。

要求：返回其节点值的后序遍历。

## 解题思路

N 叉树的后序遍历顺序为：子节点顺序递归遍历 -> 根节点。

一个取巧的方法是先按照：根节点 -> 子节点逆序递归遍历 的顺序将遍历顺序存储到答案数组。

然后再将其进行翻转就变为了后序遍历顺序。具体操作如下：

- 用栈保存根节点 `root`。然后遍历栈。
- 循环判断栈是否为空。
- 如果栈不为空，取出栈顶节点，将节点值加入答案数组。
- 顺序遍历栈顶节点的子节点，将其依次放入栈中（顺序遍历保证取出顺序为逆序）。
- 然后继续第 2 ~ 4 步，直到栈为空。

最后将答案数组逆序返回。

## 代码

```python
class Solution:
    def postorder(self, root: 'Node') -> List[int]:
        res = []
        stack = []
        if not root:
            return res

        stack.append(root)
        while stack:
            node = stack.pop()
            res.append(node.val)
            for child in node.children:
                stack.append(child)

        return res[::-1]
```

# [0599. 两个列表的最小索引总和](https://leetcode.cn/problems/minimum-index-sum-of-two-lists/)

- 标签：数组、哈希表、字符串
- 难度：简单

## 题目链接

- [0599. 两个列表的最小索引总和 - 力扣](https://leetcode.cn/problems/minimum-index-sum-of-two-lists/)

## 题目大意

Andy 和 Doris 都有一个表示最喜欢餐厅的列表 list1、list2，每个餐厅的名字用字符串表示。

找出他们共同喜爱的餐厅，要求两个餐厅在列表中的索引和最小，如果答案不唯一，则输出所有答案。

## 解题思路

遍历 list1，建立一个哈希表 list1_dict，以 list1[i] : i 键值对的方式，将 list1 的下标存储起来。

然后遍历 list2，判断 list2[i] 是否在哈希表中，如果在，则根据 i + list1_dict[i] 和 min_sum 的比较，判断是否需要更新最小索引和。如果 i + list1_dict[i] < min_sum，则更新最小索引和，并清空答案数据，添加新的答案。如果 i + list1_dict[i] == min_sum，则更新最小索引和，并添加答案。

## 代码

```python
class Solution:
    def findRestaurant(self, list1: List[str], list2: List[str]) -> List[str]:
        list1_dict = dict()
        len1 = len(list1)
        len2 = len(list2)
        for i in range(len1):
            list1_dict[list1[i]] = i

        min_sum = len1 + len2
        res = []
        for i in range(len2):
            if list2[i] in list1_dict:
                sum = i + list1_dict[list2[i]]
                if sum < min_sum:
                    res = [list2[i]]
                    min_sum = sum
                elif sum == min_sum:
                    res.append(list2[i])
        return res
```

# [0600. 不含连续1的非负整数](https://leetcode.cn/problems/non-negative-integers-without-consecutive-ones/)

- 标签：动态规划
- 难度：困难

## 题目链接

- [0600. 不含连续1的非负整数 - 力扣](https://leetcode.cn/problems/non-negative-integers-without-consecutive-ones/)

## 题目大意

**描述**：给定一个正整数 $n$。

**要求**：统计在 $[0, n]$ 范围的非负整数中，有多少个整数的二进制表示中不存在连续的 $1$。

**说明**：

- $1 \le n \le 10^9$。

**示例**：

- 示例 1：

```python
输入: n = 5
输出: 5
解释: 
下面列出范围在 [0, 5] 的非负整数与其对应的二进制表示：
0 : 0
1 : 1
2 : 10
3 : 11
4 : 100
5 : 101
其中，只有整数 3 违反规则（有两个连续的 1 ），其他 5 个满足规则。
```

- 示例 2：

```python
输入: n = 1
输出: 2
```

## 解题思路

### 思路 1：动态规划 + 数位 DP

将 $n$ 转换为字符串 $s$，定义递归函数 `def dfs(pos, pre, isLimit):` 表示构造第 $pos$ 位及之后所有数位的合法方案数。其中：

1. $pos$ 表示当前枚举的数位位置。
2. $pre$ 表示前一位是否为 $1$，用于过滤连续 $1$ 的不合法方案。
3. $isLimit$ 表示前一位数位是否等于上界，用于限制本次搜索的数位范围。

接下来按照如下步骤进行递归。

1. 从 `dfs(0, False, True)` 开始递归。 `dfs(0, False, True)` 表示：
   1. 从位置 $0$ 开始构造。
   2. 开始时前一位不为 $1$。
   3. 开始时受到数字 $n$ 对应最高位数位的约束。
2. 如果遇到  $pos == len(s)$，表示到达数位末尾，当前为合法方案，此时：直接返回方案数 $1$。
3. 如果 $pos \ne len(s)$，则定义方案数 $ans$，令其等于 $0$，即：`ans = 0`。
4. 因为不需要考虑前导 $0$，所以当前所能选择的最小数字 $minX$ 为 $0$。
5. 根据 $isLimit$ 来决定填当前位数位所能选择的最大数字（$maxX$）。
6. 然后根据 $[minX, maxX]$ 来枚举能够填入的数字 $d$。
7. 如果前一位为 $1$ 并且当前为 $d$ 也为 $1$，则说明当前方案出现了连续的 $1$，则跳过。
8. 方案数累加上当前位选择 $d$ 之后的方案数，即：`ans += dfs(pos + 1, d == 1, isLimit and d == maxX)`。
   1. `d == 1` 表示下一位 $pos - 1$ 的前一位 $pos$ 是否为 $1$。
   2. `isLimit and d == maxX` 表示 $pos + 1$ 位受到之前位限制和 $pos$ 位限制。
9. 最后的方案数为 `dfs(0, False, True)`，将其返回即可。

### 思路 1：代码

```python
class Solution:
    def findIntegers(self, n: int) -> int:
        # 将 n 的二进制转换为字符串 s
        s = str(bin(n))[2:]
        
        @cache
        # pos: 第 pos 个数位
        # pre: 第 pos - 1 位是否为 1
        # isLimit: 表示是否受到选择限制。如果为真，则第 pos 位填入数字最多为 s[pos]；如果为假，则最大可为 9。
        def dfs(pos, pre, isLimit):
            if pos == len(s):
                return 1
            
            ans = 0
            # 不需要考虑前导 0，则最小可选择数字为 0
            minX = 0
            # 如果受到选择限制，则最大可选择数字为 s[pos]，否则最大可选择数字为 1。
            maxX = int(s[pos]) if isLimit else 1
            
            # 枚举可选择的数字
            for d in range(minX, maxX + 1): 
                if pre and d == 1:
                    continue
                ans += dfs(pos + 1, d == 1, isLimit and d == maxX)

            return ans
    
        return dfs(0, False, True)
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(\log n)$。
- **空间复杂度**：$O(\log n)$。
# [0611. 有效三角形的个数](https://leetcode.cn/problems/valid-triangle-number/)

- 标签：贪心、数组、双指针、二分查找、排序
- 难度：中等

## 题目链接

- [0611. 有效三角形的个数 - 力扣](https://leetcode.cn/problems/valid-triangle-number/)

## 题目大意

**描述**：给定一个包含非负整数的数组 $nums$，其中 $nums[i]$ 表示第 $i$ 条边的边长。

**要求**：统计数组中可以组成三角形三条边的三元组个数。

**说明**：

- $1 \le nums.length \le 1000$。
- $0 \le nums[i] \le 1000$。

**示例**：

- 示例 1：

```python
输入: nums = [2,2,3,4]
输出: 3
解释:有效的组合是: 
2,3,4 (使用第一个 2)
2,3,4 (使用第二个 2)
2,2,3
```

- 示例 2：

```python
输入: nums = [4,2,3,4]
输出: 4
```

## 解题思路

### 思路 1：对撞指针

构成三角形的条件为：任意两边和大于第三边，或者任意两边差小于第三边。只要满足这两个条件之一就可以构成三角形。以任意两边和大于第三边为例，如果用 $a$、$b$、$c$ 来表示的话，应该同时满足 $a + b > c$、$a + c > b$、$b + c > a$。如果我们将三条边升序排序，假设 $a \le b \le c$，则如果满足 $a + b > c$，则 $a + c > b$ 和 $b + c > a$ 一定成立。

所以我们可以先对 $nums$ 进行排序。然后固定最大边 $i$，利用对撞指针 $left$、$right$ 查找较小的两条边。然后判断是否构成三角形并统计三元组个数。

为了避免重复计算和漏解，要严格保证三条边的序号关系为：$left < right < i$。具体做法如下：

- 对数组从小到大排序，使用 $ans$ 记录三元组个数。
- 从 $i = 2$ 开始遍历数组的每一条边，$i$ 作为最大边。
- 使用双指针 $left$、$right$。$left$ 指向 $0$，$right$ 指向 $i - 1$。
  - 如果 $nums[left] + nums[right] \le nums[i]$，说明第一条边太短了，可以增加第一条边长度，所以将 $left$ 右移，即 `left += 1`。
  - 如果 $nums[left] + nums[right] > nums[i]$，说明可以构成三角形，并且第二条边固定为 $right$ 边的话，第一条边可以在 $[left, right - 1]$ 中任意选择。所以三元组个数要加上 $right - left$。即 `ans += (right - left)`。
- 直到 $left == right$ 跳出循环，输出三元组个数 $ans$。

### 思路 1：代码

```python
class Solution:
    def triangleNumber(self, nums: List[int]) -> int:
        nums.sort()
        size = len(nums)
        ans = 0

        for i in range(2, size):
            left = 0
            right = i - 1
            while left < right:
                if nums[left] + nums[right] <= nums[i]:
                    left += 1
                else:
                    ans += (right - left)
                    right -= 1
        return ans
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(n^2)$，其中 $n$ 为数组中的元素个数。
- **空间复杂度**：$O(\log n)$，排序需要 $\log n$ 的栈空间。

# [0616. 给字符串添加加粗标签](https://leetcode.cn/problems/add-bold-tag-in-string/)

- 标签：字典树、数组、哈希表、字符串、字符串匹配
- 难度：中等

## 题目链接

- [0616. 给字符串添加加粗标签 - 力扣](https://leetcode.cn/problems/add-bold-tag-in-string/)

## 题目大意

给定一个字符串 `s` 和一个字符串列表 `words`。

要求：如果 `s` 的子串在字符串列表 `words` 中出现过，则在该子串前后添加加粗闭合标签 `<b>` 和 `</b>`。如果两个子串有重叠部分，则将它们一起用一对闭合标签包围起来。同理，如果两个子字符串连续被加粗，那么你也需要把它们合起来用一对加粗标签包围。最后返回添加加粗标签后的字符串 `s`。

## 解题思路

构建字典树，将字符串列表 `words` 中所有字符串添加到字典树中。

然后遍历字符串 `s`，从每一个位置开始查询字典树。在第一个符合要求的单词前面添加 `<b>`。在连续符合要求的单词中的最后一个单词后面添加 `</b>`。

最后返回添加加粗标签后的字符串 `s`。

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


class Solution:
    def addBoldTag(self, s: str, words: List[str]) -> str:
        trie_tree = Trie()
        for word in words:
            trie_tree.insert(word)

        size = len(s)
        bold_left, bold_right = -1, -1
        ans = ""
        for i in range(size):
            cur = trie_tree
            if s[i] in cur.children:
                bold_left = i
                while bold_left < size and s[bold_left] in cur.children:
                    cur = cur.children[s[bold_left]]
                    bold_left += 1
                    if cur.isEnd:
                        if bold_right == -1:
                            ans += "<b>"
                        bold_right = max(bold_left, bold_right)
            if i == bold_right:
                ans += "</b>"
                bold_right = -1
            ans += s[i]
        if bold_right >= 0:
            ans += "</b>"
        return ans
```

# [0617. 合并二叉树](https://leetcode.cn/problems/merge-two-binary-trees/)

- 标签：树、深度优先搜索、广度优先搜索、二叉树
- 难度：简单

## 题目链接

- [0617. 合并二叉树 - 力扣](https://leetcode.cn/problems/merge-two-binary-trees/)

## 题目大意

给定两个二叉树，将两个二叉树合并成一个新的二叉树。合并规则如下：

- 如果两个二叉树对应节点重叠，则将两个节点的值相加并作为新的二叉树节点。
- 如果两个二叉树对应节点其中一个为空，另一个不为空，则将不为空的节点左心新的二叉树节点。

最终返回新的二叉树的根节点。

## 解题思路

利用前序遍历二叉树，并按照规则递归建立二叉树。将其对应节点值相加或者取其中不为空的节点做为新节点。

## 代码

```python
class Solution:
    def mergeTrees(self, root1: TreeNode, root2: TreeNode) -> TreeNode:
        if not root1:
            return root2
        if not root2:
            return root1

        merged = TreeNode(root1.val + root2.val)
        merged.left = self.mergeTrees(root1.left, root2.left)
        merged.right = self.mergeTrees(root1.right, root2.right)
        return merged

```

# [0621. 任务调度器](https://leetcode.cn/problems/task-scheduler/)

- 标签：贪心、数组、哈希表、计数、排序、堆（优先队列）
- 难度：中等

## 题目链接

- [0621. 任务调度器 - 力扣](https://leetcode.cn/problems/task-scheduler/)

## 题目大意

给定一个字符数组 tasks 表示 CPU 需要执行的任务列表。tasks 中每个字母表示一种不同种类的任务。任务可以按任意顺序执行，并且每个任务执行时间为 1 个单位时间。在任何一个单位时间，CPU 可以完成一个任务，或者也可以处于待命状态。

但是两个相同种类的任务之间需要 n 个单位时间的冷却时间，所以不能在连续的 n 个单位时间内执行相同的任务。

要求计算出完成 tasks 中所有任务所需要的「最短时间」。

## 解题思路

因为相同种类的任务之间最少需要 n 个单位时间间隔，所以为了最短时间，应该优先考虑任务出现此次最多的任务。

先找出出现次数最多的任务，然后中间间隔的单位来安排别的任务，或者处于待命状态。

然后将第二出现次数最多的任务，按照 n 个时间间隔安排起来。如果第二出现次数最多的任务跟第一出现次数最多的任务出现次数相同，则最短时间就会加一。

最后我们会发现：最短时间跟出现次数最多的任务正相关。

假设出现次数最多的任务为 "A"。与 "A" 出现次数相同的任务数为 count。则：

- `最短时间 = （A 出现次数 - 1）* （n + 1）+ count`。

最后还应该比较一下总的任务个数跟计算出的最短时间答案。如果最短时间比总的任务个数还少，说明间隔中放不下所有的任务，会有任务「溢出」。则应该将多余任务插入间隔中，则答案应为总的任务个数。

## 代码

```python
class Solution:
    def leastInterval(self, tasks: List[str], n: int) -> int:
        # 记录每个任务出现的次数
        tasks_counts = [0 for _ in range(26)]
        for i in range(len(tasks)):
            num = ord(tasks[i]) - ord('A')
            tasks_counts[num] += 1
        max_task_count = max(tasks_counts)
        # 统计多少个出现最多次的任务
        count = 0
        for task_count in tasks_counts:
            if task_count == max_task_count:
               count += 1

        # 如果结果比任务数量少，则返回总任务数
        return max((max_task_count - 1) * (n + 1) + count, len(tasks))
```

# [0622. 设计循环队列](https://leetcode.cn/problems/design-circular-queue/)

- 标签：设计、队列、数组、链表
- 难度：中等

## 题目链接

- [0622. 设计循环队列 - 力扣](https://leetcode.cn/problems/design-circular-queue/)

## 题目大意

**要求**：设计实现一个循环队列，支持以下操作：

- `MyCircularQueue(k)`: 构造器，设置队列长度为 `k`。
- `Front`: 从队首获取元素。如果队列为空，返回 `-1`。
- `Rear`: 获取队尾元素。如果队列为空，返回 `-1`。
- `enQueue(value)`: 向循环队列插入一个元素。如果成功插入则返回真。
- `deQueue()`: 从循环队列中删除一个元素。如果成功删除则返回真。
- `isEmpty()`: 检查循环队列是否为空。
- `isFull()`: 检查循环队列是否已满。

**说明**：

- 所有的值都在 `0` 至 `1000` 的范围内。
- 操作数将在 `1` 至 `1000` 的范围内。
- 请不要使用内置的队列库。

**示例**：

- 示例 1：

```python
MyCircularQueue circularQueue = new MyCircularQueue(3); // 设置长度为 3
circularQueue.enQueue(1);  // 返回 true
circularQueue.enQueue(2);  // 返回 true
circularQueue.enQueue(3);  // 返回 true
circularQueue.enQueue(4);  // 返回 false，队列已满
circularQueue.Rear();  // 返回 3
circularQueue.isFull();  // 返回 true
circularQueue.deQueue();  // 返回 true
circularQueue.enQueue(4);  // 返回 true
circularQueue.Rear();  // 返回 4
```

## 解题思路

这道题可以使用数组，也可以使用链表来实现循环队列。

### 思路 1：使用数组模拟

建立一个容量为 `k + 1` 的数组 `queue`。并保存队头指针 `front`、队尾指针 `rear`，队列容量 `capacity` 为 `k + 1`（这里之所以用了 `k + 1` 的容量，是为了判断空和满，需要空出一个）。

然后实现循环队列的各个接口：

1. `MyCircularQueue(k)`: 
   1. 将数组 `queue` 初始化大小为 `k + 1` 的数组。
   2. `front`、`rear` 初始化为 `0`。
2. `Front`: 
   1. 先检测队列是否为空。如果队列为空，返回 `-1`。
   2. 如果不为空，则返回队头元素。
3. `Rear`: 
   1. 先检测队列是否为空。如果队列为空，返回 `-1`。
   2. 如果不为空，则返回队尾元素。
4. `enQueue(value)`: 
   1. 如果队列已满，则无法插入，返回 `False`。
   2. 如果队列未满，则将队尾指针 `rear` 向右循环移动一位，并进行插入操作。然后返回 `True`。
5. `deQueue()`: 
   1. 如果队列为空，则无法删除，返回 `False`。
   2. 如果队列不空，则将队头指针 `front` 指向元素赋值为 `None`，并将 `front` 向右循环移动一位。然后返回 `True`。
6. `isEmpty()`: 如果 `rear` 等于 `front`，则说明队列为空，返回 `True`。否则，队列不为空，返回 `False`。
7. `isFull()`: 如果 `(rear + 1) % capacity` 等于 `front`，则说明队列已满，返回 `True`。否则，队列未满，返回 `False`。

### 思路 1：代码

```python
class MyCircularQueue:

    def __init__(self, k: int):
        self.capacity = k + 1
        self.queue = [0 for _ in range(k + 1)]
        self.front = 0
        self.rear = 0

    def enQueue(self, value: int) -> bool:
        if self.isFull():
            return False
        self.rear = (self.rear + 1) % self.capacity
        self.queue[self.rear] = value
        return True

    def deQueue(self) -> bool:
        if self.isEmpty():
            return False
        self.front = (self.front + 1) % self.capacity
        return True

    def Front(self) -> int:
        if self.isEmpty():
            return -1
        return self.queue[(self.front + 1)  % self.capacity]

    def Rear(self) -> int:
        if self.isEmpty():
            return -1
        return self.queue[self.rear]

    def isEmpty(self) -> bool:
        return self.front == self.rear

    def isFull(self) -> bool:
        return (self.rear + 1) % self.capacity == self.front
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(1)$。初始化和每项操作的时间复杂度均为 $O(1)$。
- **空间复杂度**：$O(k)$。其中 $k$ 为给定队列的元素数目。

# [0633. 平方数之和](https://leetcode.cn/problems/sum-of-square-numbers/)

- 标签：数学、双指针、二分查找
- 难度：中等

## 题目链接

- [0633. 平方数之和 - 力扣](https://leetcode.cn/problems/sum-of-square-numbers/)

## 题目大意

给定一个非负整数 c，判断是否存在两个整数 a 和 b，使得 $a^2 + b^2 = c$，如果存在则返回 True，不存在返回 False。

## 解题思路

最直接的办法就是枚举 a、b 所有可能。这样遍历下来的时间复杂度为 $O(c^2)$。但是没必要进行二重遍历。可以只遍历 a，然后去判断 $\sqrt{c - b^2}$ 是否为整数，并且 a 只需遍历到 $\sqrt{c}$ 即可，时间复杂度为 $O(\sqrt{c})$。

另一种方法是双指针。定义两个指针 left，right 分别指向 0 和 $\sqrt{c}$。判断 $left^2 + right^2$ 与 c 之间的关系。

- 如果 $a^2 + b^2 == c$，则返回 True。
- 如果 $a^2 + b^2 < c$，则将 a 值加一，继续查找。
- 如果 $a^2 + b^2 > c$，则将 b 值减一，继续查找。
- 当 $a == b$ 时，结束查找。如果此时仍没有找到满足 $a^2 + b^2 == c$ 的 a、b 值，则返回 False。

## 代码

```python
class Solution:
    def judgeSquareSum(self, c: int) -> bool:
        a, b = 0, int(c ** 0.5)
        while a <= b:
            sum = a*a + b*b
            if sum == c:
                return True
            elif sum < c:
                a += 1
            else:
                b -= 1
        return False
```

# [0639. 解码方法 II](https://leetcode.cn/problems/decode-ways-ii/)

- 标签：字符串、动态规划
- 难度：困难

## 题目链接

- [0639. 解码方法 II - 力扣](https://leetcode.cn/problems/decode-ways-ii/)

## 题目大意

**描述**：给定一个包含数字和字符 `'*'` 的字符串 $s$。该字符串已经按照下面的映射关系进行了编码：

- `A` 映射为 $1$。
- `B` 映射为 $2$。
- ...
- `Z` 映射为 $26$。

除了上述映射方法，字符串 $s$ 中可能包含字符 `'*'`，可以表示 $1$ ~ $9$ 的任一数字（不包括 $0$）。例如字符串 `"1*"` 可以表示为 `"11"`、`"12"`、…、`"18"`、`"19"` 中的任何一个编码。

基于上述映射的方法，现在对字符串 `s` 进行「解码」。即从数字到字母进行反向映射。比如 `"11106"` 可以映射为：

- `"AAJF"`，将消息分组为 $(1 1 10 6)$。
- `"KJF"`，将消息分组为 $(11 10 6)$。

**要求**：计算出共有多少种可能的解码方案。

**说明**：

- $1 \le s.length \le 100$。
- $s$ 只包含数字，并且可能包含前导零。
- 题目数据保证答案肯定是一个 $32$ 位的整数。

```python
输入：s = "*"
输出：9
解释：这一条编码消息可以表示 "1"、"2"、"3"、"4"、"5"、"6"、"7"、"8" 或 "9" 中的任意一条。可以分别解码成字符串 "A"、"B"、"C"、"D"、"E"、"F"、"G"、"H" 和 "I" 。因此，"*" 总共有 9 种解码方法。
```

## 解题思路

### 思路 1：动态规划

这道题是「[91. 解码方法 - 力扣](https://leetcode.cn/problems/decode-ways/)」的升级版，其思路是相似的，只不过本题的状态转移方程的条件和公式不太容易想全。

###### 1. 划分阶段

按照字符串的结尾位置进行阶段划分。

###### 2. 定义状态

定义状态 $dp[i]$ 表示为：字符串 $s$ 前 $i$ 个字符构成的字符串可能构成的翻译方案数。

###### 3. 状态转移方程

$dp[i]$ 的来源有两种情况：

1. 使用了一个字符，对 $s[i]$ 进行翻译：
   1. 如果 `s[i] == '*'`，则 `s[i]` 可以视作区间 `[1, 9]` 上的任意一个数字，可以被翻译为 `A` ~ `I`。此时当前位置上的方案数为 `9`，即 `dp[i] = dp[i - 1] * 9`。
   2. 如果 `s[i] == '0'`，则无法被翻译，此时当前位置上的方案数为 `0`，即 `dp[i] = dp[i - 1] * 0`。
   3. 如果是其他情况（即 `s[i]` 是区间 `[1, 9]` 上某一个数字），可以被翻译为 `A` ~ `I` 对应位置上的某个字母。此时当前位置上的方案数为 `1`，即 `dp[i] = dp[i - 1] * 1`。

2. 使用了两个字符，对 `s[i - 1]` 和 `s[i]` 进行翻译：
   1. 如果 `s[i - 1] == '*'` 并且 `s[i] == '*'`，则 `s[i]` 可以视作区间 `[11, 19]` 或者 `[21, 26]` 上的任意一个数字。此时当前位置上的方案数为 `15`，即 `dp[i] = dp[i - 2] * 15`。
   2. 如果 `s[i - 1] == '*'` 并且 `s[i] != '*'`，则：
      1. 如果 `s[i]` 在区间 `[1, 6]` 内，`s[i - 1]` 可以选择 `1` 或 `2`。此时当前位置上的方案数为 `2`，即 `dp[i] = dp[i - 2] * 2`。
      2. 如果 `s[i]` 不在区间 `[1, 6]` 内，`s[i - 1]` 只能选择 `1`。此时当前位置上的方案数为 `1`，即 `dp[i] = dp[i - 2] * 1`。

   3. 如果 `s[i - 1] == '1'` 并且 `s[i] == '*'`，`s[i]` 可以视作区间 `[1, 9]` 上任意一个数字。此时当前位置上的方案数为 `9`，即 `dp[i] = dp[i - 2] * 9`。
   4. 如果 `s[i - 1] == '1'` 并且 `s[i] != '*'`，`s[i]` 可以视作区间 `[1, 9]` 上的某一个数字。此时当前位置上的方案数为 `1`，即 `dp[i] = dp[i - 2] * 1`。
   5. 如果 `s[i - 1] == '2'` 并且 `s[i] == '*'`，`s[i]` 可以视作区间 `[1, 6]` 上任意一个数字。此时当前位置上的方案数为 `6`，即 `dp[i] = dp[i - 2] * 6`。
   6. 如果 `s[i - 1] == '2'` 并且 `s[i] != '*'`，则：
      1. 如果 `s[i]` 在区间 `[1, 6]` 内，此时当前位置上的方案数为 `1`，即 `dp[i] = dp[i - 2] * 1`。
      2. 如果 `s[i]` 不在区间 `[1, 6]` 内，此时当前位置上的方案数为 `0`，即 `dp[i] = dp[i - 2] * 0`。

   7. 其他情况下（即 `s[i - 1]` 在区间 `[3, 9]` 内），则无法被翻译，此时当前位置上的方案数为 `0`，即 `dp[i] = dp[i - 2] * 0`。


在进行转移的时候，需要将使用一个字符的翻译方案数与使用两个字符的翻译方案数进行相加。同时还要注意对 $10^9 + 7$ 的取余。

这里我们可以单独写两个方法 `，分别来表示「单个字符 `s[i]` 的翻译方案数」和「两个字符 `s[i - 1]` 和 `s[i]` 的翻译方案数」，这样代码逻辑会更加清晰。

###### 4. 初始条件

- 字符串为空时，只有一个翻译方案，翻译为空字符串，即 `dp[0] = 1`。
- 字符串只有一个字符时，单个字符 `s[i]` 的翻译方案数为转移条件的第一种求法，即`dp[1] = self.parse1(s[0])`。

###### 5. 最终结果

根据我们之前定义的状态，`dp[i]` 表示为：字符串 `s` 前 `i` 个字符构成的字符串可能构成的翻译方案数。则最终结果为 `dp[size]`，`size` 为字符串长度。

### 思路 1：动态规划代码

```python
class Solution:
    def parse1(self, ch):
        if ch == '*':
            return 9
        if ch == '0':
            return 0
        return 1

    def parse2(self, ch1, ch2):
        if ch1 == '*' and ch2 == '*':
            return 15
        if ch1 == '*' and ch2 != '*':
            return 2 if ch2 <= '6' else 1

        if ch1 == '1' and ch2 == '*':
            return 9
        if ch1 == '1' and ch2 != '*':
            return 1

        if ch1 == '2' and ch2 == '*':
            return 6
        if ch1 == '2' and ch2 != '*':
            return 1 if ch2 <= '6' else 0

        return 0

    def numDecodings(self, s: str) -> int:
        mod = 10 ** 9 + 7
        size = len(s)

        dp = [0 for _ in range(size + 1)]
        dp[0] = 1
        dp[1] = self.parse1(s[0])

        for i in range(2, size + 1):
            dp[i] += dp[i - 1] * self.parse1(s[i - 1])
            dp[i] += dp[i - 2] * self.parse2(s[i - 2], s[i - 1])
            dp[i] %= mod
        
        return dp[size]
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(n)$。一重循环遍历的时间复杂度是 $O(n)$。
- **空间复杂度**：$O(n)$。用到了一维数组保存状态，所以总体空间复杂度为 $O(n)$。# [0642. 设计搜索自动补全系统](https://leetcode.cn/problems/design-search-autocomplete-system/)

- 标签：设计、字典树、字符串、数据流
- 难度：困难

## 题目链接

- [0642. 设计搜索自动补全系统 - 力扣](https://leetcode.cn/problems/design-search-autocomplete-system/)

## 题目大意

要求：设计一个搜索自动补全系统。用户会输入一条语句（最少包含一个字母，以特殊字符 `#` 结尾）。除 `#` 以外用户输入的每个字符，返回历史中热度前三并以当前输入部分为前缀的句子。下面是详细规则：

- 一条句子的热度定义为历史上用户输入这个句子的总次数。
- 返回前三的句子需要按照热度从高到低排序（第一个是最热门的）。如果有多条热度相同的句子，请按照 ASCII 码的顺序输出（ASCII 码越小排名越前）。
- 如果满足条件的句子个数少于 3，将它们全部输出。
- 如果输入了特殊字符，意味着句子结束了，请返回一个空集合。

你的工作是实现以下功能：

- 构造函数： `AutocompleteSystem(String[] sentences, int[] times):` 
  - 输入历史数据。 `sentences` 是之前输入过的所有句子，`times` 是每条句子输入的次数，你的系统需要记录这些历史信息。

- 输入函数（用户输入一条新的句子，下面的函数会提供用户输入的下一个字符）：`List<String> input(char c):` 
  - 其中 `c` 是用户输入的下一个字符。字符只会是小写英文字母（`a` 到 `z` ），空格（` `）和特殊字符（`#`）。输出历史热度前三的具有相同前缀的句子。

## 解题思路

使用字典树来保存输入过的所有句子 `sentences`，并且在字典树中维护每条句子的输入次数 `times`。

构造函数中：

- 将所有句子及对应输入次数插入到字典树中。

输入函数中：

- 使用 `path` 变量保存当前输入句子的前缀。
- 如果遇到 `#`，则将当前句子插入到字典树中。
- 如果遇到其他字符，用 `path` 保存当前字符 `c`。并在字典树中搜索以 `path` 为前缀的节点的所有分支，将每个分支对应的单词 `path` 和它们出现的次数 `times` 存入数组中。然后借助 `heapq` 进行堆排序，根据出现次数和 ASCII 码大小排序，找出 `times` 最多的前三个单词。

## 代码

```python
import heapq

class Trie:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.children = dict()
        self.isEnd = False
        self.times = 0


    def insert(self, word: str, times=1) -> None:
        """
        Inserts a word into the trie.
        """
        cur = self
        for ch in word:
            if ch not in cur.children:
                cur.children[ch] = Trie()
            cur = cur.children[ch]
        cur.isEnd = True
        cur.times += times


    def search(self, word: str):
        """
        Returns if the word is in the trie.
        """
        cur = self

        for ch in word:
            if ch not in cur.children:
                return []
            cur = cur.children[ch]

        res = []
        path = [word]
        cur.dfs(res, path)
        return res


    def dfs(self, res, path):
        cur = self
        if cur.isEnd:
            res.append((-cur.times, ''.join(path)))
        for ch in cur.children:
            node = cur.children[ch]
            path.append(ch)
            node.dfs(res, path)
            path.pop()


class AutocompleteSystem:

    def __init__(self, sentences: List[str], times: List[int]):
        self.path = ''
        self.exists = True
        self.trie_tree = Trie()
        for i in range(len(sentences)):
            self.trie_tree.insert(sentences[i], times[i])


    def input(self, c: str) -> List[str]:
        if c == '#':
            self.trie_tree.insert(self.path, 1)
            self.path = ''
            self.exists = True
            return []
        else:
            self.path += c
            if not self.exists:
                return []
            words = self.trie_tree.search(self.path)
            if words:
                heapq.heapify(words)
                res = []
                while words and len(res) < 3:
                    res.append(heapq.heappop(words)[1])
                return res
            else:
                self.exists = False
                return []
```

# [0643. 子数组最大平均数 I](https://leetcode.cn/problems/maximum-average-subarray-i/)

- 标签：数组、滑动窗口
- 难度：简单

## 题目链接

- [0643. 子数组最大平均数 I - 力扣](https://leetcode.cn/problems/maximum-average-subarray-i/)

## 题目大意

**描述**：给定一个由 $n$ 个元素组成的整数数组 $nums$ 和一个整数 $k$。

**要求**：找出平均数最大且长度为 $k$ 的连续子数组，并输出该最大平均数。

**说明**：

- 任何误差小于 $10^{-5}$ 的答案都将被视为正确答案。
- $n == nums.length$。
- $1 \le k \le n \le 10^5$。
- $-10^4 \le nums[i] \le 10^4$。

**示例**：

- 示例 1：

```python
输入：nums = [1,12,-5,-6,50,3], k = 4
输出：12.75
解释：最大平均数 (12-5-6+50)/4 = 51/4 = 12.75
```

- 示例 2：

```python
输入：nums = [5], k = 1
输出：5.00000
```

## 解题思路

### 思路 1：滑动窗口（固定长度）

这道题目是典型的固定窗口大小的滑动窗口题目。窗口大小为 $k$。具体做法如下：

1. $ans$ 用来维护子数组最大平均数，初始值为负无穷，即 `float('-inf')`。$window\underline{\hspace{0.5em}}total$ 用来维护窗口中元素的和。
2. $left$ 、$right$ 都指向序列的第一个元素，即：`left = 0`，`right = 0`。
3. 向右移动 $right$，先将 $k$ 个元素填入窗口中。
4. 当窗口元素个数为 $k$ 时，即：$right - left + 1 >= k$ 时，计算窗口内的元素和平均值，并维护子数组最大平均数。
5. 然后向右移动 $left$，从而缩小窗口长度，即 `left += 1`，使得窗口大小始终保持为 $k$。
6. 重复 $4 \sim 5$ 步，直到 $right$ 到达数组末尾。
7. 最后输出答案 $ans$。

### 思路 1：代码

```python
class Solution:
    def findMaxAverage(self, nums: List[int], k: int) -> float:
        left = 0
        right = 0
        window_total = 0
        ans = float('-inf')
        while right < len(nums):
            window_total += nums[right]

            if right - left + 1 >= k:
                ans = max(window_total / k, ans)
                window_total -= nums[left]
                left += 1

            # 向右侧增大窗口
            right += 1

        return ans
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(n)$。其中 $n$ 为数组 $nums$ 的元素个数。
- **空间复杂度**：$O(1)$。# [0647. 回文子串](https://leetcode.cn/problems/palindromic-substrings/)

- 标签：字符串、动态规划
- 难度：中等

## 题目链接

- [0647. 回文子串 - 力扣](https://leetcode.cn/problems/palindromic-substrings/)

## 题目大意

给定一个字符串 `s`，计算 `s` 中有多少个回文子串。

## 解题思路

动态规划求解。

先定义状态 `dp[i][j]` 表示为区间 `[i, j]` 的子串是否为回文子串，如果是，则 `dp[i][j] = True`，如果不是，则 `dp[i][j] = False`。

接下来确定状态转移共识：

如果 `s[i] == s[j]`，分为以下几种情况：

- `i == j`，单字符肯定是回文子串，`dp[i][j] == True`。
- `j - i == 1`，比如 `aa` 肯定也是回文子串，`dp[i][j] = True`。
- 如果 `j - i > 1`，则需要看 `[i + 1, j - 1]` 区间是不是回文子串，`dp[i][j] = dp[i + 1][j - 1]`。

如果 `s[i] != s[j]`，那肯定不是回文子串，`dp[i][j] = False`。

下一步确定遍历方向。

由于 `dp[i][j]` 依赖于 `dp[i + 1][j - 1]`，所以我们可以从左下角向右上角遍历。

同时，在递推过程中记录下 `dp[i][j] == True` 的个数，即为最后结果。

## 代码

```python
class Solution:
    def countSubstrings(self, s: str) -> int:
        size = len(s)
        dp = [[False for _ in range(size)] for _ in range(size)]
        res = 0
        for i in range(size - 1, -1, -1):
            for j in range(i, size):
                if s[i] == s[j]:
                    if j - i <= 1:
                        dp[i][j] = True
                    else:
                        dp[i][j] = dp[i + 1][j - 1]
                else:
                    dp[i][j] = False
                if dp[i][j]:
                    res += 1
        return res
```

# [0648. 单词替换](https://leetcode.cn/problems/replace-words/)

- 标签：字典树、数组、哈希表、字符串
- 难度：中等

## 题目链接

- [0648. 单词替换 - 力扣](https://leetcode.cn/problems/replace-words/)

## 题目大意

**描述**：给定一个由许多词根组成的字典列表 `dictionary`，以及一个句子字符串 `sentence`。

**要求**：将句子中有词根的单词用词根替换掉。如果单词有很多词根，则用最短的词根替换掉他。最后输出替换之后的句子。

**说明**：

- $1 \le dictionary.length \le 1000$。
- $1 \le dictionary[i].length \le 100$。
- `dictionary[i]` 仅由小写字母组成。
- $1 \le sentence.length \le 10^6$。
- `sentence` 仅由小写字母和空格组成。
- `sentence` 中单词的总量在范围 $[1, 1000]$ 内。
- `sentence` 中每个单词的长度在范围 $[1, 1000]$ 内。
- `sentence` 中单词之间由一个空格隔开。
- `sentence` 没有前导或尾随空格。

**示例**：

- 示例 1：

```python
输入：dictionary = ["cat","bat","rat"], sentence = "the cattle was rattled by the battery"
输出："the cat was rat by the bat"
```

- 示例 2：

```python
输入：dictionary = ["a","b","c"], sentence = "aadsfasf absbs bbab cadsfafs"
输出："a a b c"
```

## 解题思路

### 思路 1：字典树

1. 构造一棵字典树。
2. 将所有的词根存入到前缀树（字典树）中。
3. 然后在树上查找每个单词的最短词根。

### 思路 1：代码

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

### 思路 1：复杂度分析

- **时间复杂度**：$O(|dictionary| + |sentence|)$。其中 $|dictionary|$ 是字符串数组 `dictionary` 中的字符总数，$|sentence|$ 是字符串 `sentence` 的字符总数。
- **空间复杂度**：$O(|dictionary| + |sentence|)$。# [0650. 只有两个键的键盘](https://leetcode.cn/problems/2-keys-keyboard/)

- 标签：数学、动态规划
- 难度：中等

## 题目链接

- [0650. 只有两个键的键盘 - 力扣](https://leetcode.cn/problems/2-keys-keyboard/)

## 题目大意

**描述**：最初记事本上只有一个字符 `'A'`。你每次可以对这个记事本进行两种操作：

- **Copy All（复制全部）**：复制这个记事本中的所有字符（不允许仅复制部分字符）。
- **Paste（粘贴）**：粘贴上一次复制的字符。

现在，给定一个数字 $n$，需要使用最少的操作次数，在记事本上输出恰好 $n$ 个 `'A'` 。

**要求**：返回能够打印出 $n$ 个 `'A'` 的最少操作次数。

**说明**：

- $1 \le n \le 1000$。

**示例**：

- 示例 1：

```python
输入：3
输出：3
解释
最初, 只有一个字符 'A'。
第 1 步, 使用 Copy All 操作。
第 2 步, 使用 Paste 操作来获得 'AA'。
第 3 步, 使用 Paste 操作来获得 'AAA'。
```

- 示例 2：

```python
输入：n = 1
输出：0
```

## 解题思路

### 思路 1：动态规划

###### 1. 划分阶段

按照字符 `'A'`  的个数进行阶段划分。

###### 2. 定义状态

定义状态 $dp[i]$ 表示为：通过「复制」和「粘贴」操作，得到 $i$ 个字符 `'A'`，最少需要的操作数。

###### 3. 状态转移方程

1. 对于 $i$ 个字符 `'A'`，如果 $i$ 可以被一个小于 $i$ 的整数 $j$ 除尽（$j$ 是 $i$ 的因子），则说明 $j$ 个字符 `'A'` 可以通过「复制」+「粘贴」总共 $\frac{i}{j}$ 次得到 $i$ 个字符 `'A'`。
2. 而得到 $j$ 个字符 `'A'`，最少需要的操作数可以通过 $dp[j]$ 获取。

则我们可以枚举 $i$ 的因子，从中找到在满足 $j$ 能够整除 $i$ 的条件下，最小的 $dp[j] + \frac{i}{j}$，即为 $dp[i]$，即 $dp[i] = min_{j | i}(dp[i], dp[j] + \frac{i}{j})$。

由于 $j$ 能够整除 $i$，则 $j$ 与 $\frac{i}{j}$ 都是 $i$ 的因子，两者中必有一个因子是小于等于 $\sqrt{i}$ 的，所以在枚举 $i$ 的因子时，我们只需要枚举区间 $[1, \sqrt{i}]$ 即可。

综上所述，状态转移方程为：$dp[i] = min_{j | i}(dp[i], dp[j] + \frac{i}{j}, dp[\frac{i}{j}] + j)$。

###### 4. 初始条件

- 当 $i$ 为 $1$ 时，最少需要的操作数为 $0$。所以 $dp[1] = 0$。

###### 5. 最终结果

根据我们之前定义的状态，$dp[i]$ 表示为：通过「复制」和「粘贴」操作，得到 $i$ 个字符 `'A'`，最少需要的操作数。 所以最终结果为 $dp[n]$。

### 思路 1：动态规划代码

```python
import math

class Solution:
    def minSteps(self, n: int) -> int:
        dp = [0 for _ in range(n + 1)]
        for i in range(2, n + 1):
            dp[i] = float('inf')
            for j in range(1, int(math.sqrt(n)) + 1):
                if i % j == 0:
                    dp[i] = min(dp[i], dp[j] + i // j, dp[i // j] + j)

        return dp[n]
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(n \sqrt{n})$。外层循环遍历的时间复杂度是 $O(n)$，内层循环遍历的时间复杂度是 $O(\sqrt{n})$，所以总体时间复杂度为 $O(n \sqrt{n})$。
- **空间复杂度**：$O(n)$。用到了一维数组保存状态，所以总体空间复杂度为 $O(n)$。
# [0652. 寻找重复的子树](https://leetcode.cn/problems/find-duplicate-subtrees/)

- 标签：树、深度优先搜索、哈希表、二叉树
- 难度：中等

## 题目链接

- [0652. 寻找重复的子树 - 力扣](https://leetcode.cn/problems/find-duplicate-subtrees/)

## 题目大意

给定一个二叉树，返回所有重复的子树。对于重复的子树，只需返回其中任意一棵的根节点。

## 解题思路

对二叉树进行先序遍历，对遍历的所有的子树进行序列化处理，将序列化处理后的字符串作为哈希表的键，记录每棵子树出现的次数。

当出现第二次时，则说明该子树是重复的子树，将其加入答案数组。最后返回答案数组即可。

## 代码

```python
class Solution:
    def findDuplicateSubtrees(self, root: TreeNode) -> List[TreeNode]:
        tree_dict = dict()
        res = []
        def preorder(node):
            if not node:
                return '#'
            sub_tree = str(node.val) + ',' + preorder(node.left) + ',' + preorder(node.right)
            if sub_tree in tree_dict:
                tree_dict[sub_tree] += 1
            else:
                tree_dict[sub_tree] = 1
            if tree_dict[sub_tree] == 2:
                res.append(node)
            return sub_tree
        preorder(root)
        return res
```

# [0653. 两数之和 IV - 输入二叉搜索树](https://leetcode.cn/problems/two-sum-iv-input-is-a-bst/)

- 标签：树、深度优先搜索、广度优先搜索、二叉搜索树、哈希表、双指针、二叉树
- 难度：简单

## 题目链接

- [0653. 两数之和 IV - 输入二叉搜索树 - 力扣](https://leetcode.cn/problems/two-sum-iv-input-is-a-bst/)

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

# [0654. 最大二叉树](https://leetcode.cn/problems/maximum-binary-tree/)

- 标签：栈、树、数组、分治、二叉树、单调栈
- 难度：中等

## 题目链接

- [0654. 最大二叉树 - 力扣](https://leetcode.cn/problems/maximum-binary-tree/)

## 题目大意

给定一个不含重复元素的整数数组 `nums`。一个以此数组构建的最大二叉树定义如下：

- 二叉树的根是数组中的最大元素。
- 左子树是通过数组中最大值左边部分构造出的最大二叉树。
- 右子树是通过数组中最大值右边部分构造出的最大二叉树。

要求通过给定的数组构建最大二叉树，并且输出这个树的根节点。

## 解题思路

根据题意可知，数组中最大元素位置为根节点，最大元素位置左右部分可分别作为左右子树。则我们可以通过递归的方式构建最大二叉树。

- 定义 left、right 分别表示当前数组的左右边界位置，定义 `max_value_index` 为当前数组中最大值位置。
- 遍历当前数组，找到最大值位置 `max_value_index`，并建立根节点 `root`，将数组 `nums` 分为 `[left, max_value_index]` 和 `[max_value_index, right]` 两部分，并分别递归建树。
- 将其赋值给 `root` 的左右子节点，最后返回 root 节点。

## 代码

```python
class Solution:
    def createBinaryTree(self, nums: List[int], left: int, right: int) -> TreeNode:
        if left >= right:
            return None
        max_value_index = left
        for i in range(left + 1, right):
            if nums[i] > nums[max_value_index]:
                max_value_index = i

        root = TreeNode(nums[max_value_index])
        root.left = self.createBinaryTree(nums, left, max_value_index)
        root.right = self.createBinaryTree(nums, max_value_index + 1, right)

        return root

    def constructMaximumBinaryTree(self, nums: List[int]) -> TreeNode:
        return self.createBinaryTree(nums, 0, len(nums))
```

# [0658. 找到 K 个最接近的元素](https://leetcode.cn/problems/find-k-closest-elements/)

- 标签：数组、双指针、二分查找、排序、滑动窗口、堆（优先队列）
- 难度：中等

## 题目链接

- [0658. 找到 K 个最接近的元素 - 力扣](https://leetcode.cn/problems/find-k-closest-elements/)

## 题目大意

**描述**：给定一个有序数组 $arr$，以及两个整数 $k$、$x$。

**要求**：从数组中找到最靠近 $x$（两数之差最小）的 $k$ 个数。返回包含这 $k$ 个数的有序数组。

**说明**：

- 整数 $a$ 比整数 $b$ 更接近 $x$ 需要满足：
  - $|a - x| < |b - x|$ 或者
  - $|a - x| == |b - x|$ 且 $a < b$。

- $1 \le k \le arr.length$。
- $1 \le arr.length \le 10^4$。
- $arr$ 按升序排列。
- $-10^4 \le arr[i], x \le 10^4$。

**示例**：

- 示例 1：

```python
输入：arr = [1,2,3,4,5], k = 4, x = 3
输出：[1,2,3,4]
```

- 示例 2：

```python
输入：arr = [1,2,3,4,5], k = 4, x = -1
输出：[1,2,3,4]
```

## 解题思路

### 思路 1：二分查找算法

数组的区间为 $[0, n-1]$，查找的子区间长度为 $k$。我们可以通过查找子区间左端点位置，从而确定子区间。

查找子区间左端点可以通过二分查找来降低复杂度。

因为子区间为 $k$，所以左端点最多取到 $n - k$ 的位置。

设定两个指针 $left$，$right$。$left$ 指向 $0$，$right$ 指向 $n - k$。

每次取 $left$ 和 $right$ 中间位置，判断 $x$ 与左右边界的差值。$x$ 与左边的差值为 $x - arr[mid]$，$x$ 与右边界的差值为 $arr[mid + k] - x$。

- 如果 $x$ 与左边界的差值大于 $x$ 与右边界的差值，即 $x - arr[mid] > arr[mid + k] - x$，将 $left$ 右移，$left = mid + 1$，从右侧继续查找。
- 如果 $x$ 与左边界的差值小于等于 $x$ 与右边界的差值， 即 $x - arr[mid] \le arr[mid + k] - x$，则将 $right$ 向左侧靠拢，$right = mid$，从左侧继续查找。

最后返回 $arr[left, left + k]$ 即可。

### 思路 1：代码

```python
class Solution:
    def findClosestElements(self, arr: List[int], k: int, x: int) -> List[int]:
        n = len(arr)
        left = 0
        right = n - k
        while left < right:
            mid = left + (right - left) // 2
            if x - arr[mid] > arr[mid + k] - x:
                left = mid + 1
            else:
                right = mid
        return arr[left: left + k]
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(\log (n - k) + k)$，其中 $n$ 为数组中的元素个数。
- **空间复杂度**：$O(1)$。

# [0662. 二叉树最大宽度](https://leetcode.cn/problems/maximum-width-of-binary-tree/)

- 标签：树、深度优先搜索、广度优先搜索、二叉树
- 难度：中等

## 题目链接

- [0662. 二叉树最大宽度 - 力扣](https://leetcode.cn/problems/maximum-width-of-binary-tree/)

## 题目大意

**描述**：给你一棵二叉树的根节点 `root`。

**要求**：返回树的最大宽度。

**说明**：

- **每一层的宽度**：为该层最左和最右的非空节点（即两个端点）之间的长度。将这个二叉树视作与满二叉树结构相同，两端点间会出现一些延伸到这一层的 `null` 节点，这些 `null` 节点也计入长度。
- **树的最大宽度**：是所有层中最大的宽度。
- 题目数据保证答案将会在 32 位带符号整数范围内。
- 树中节点的数目范围是 $[1, 3000]$。
- $-100 \le Node.val \le 100$。

**示例**：

- 示例 1：

![](https://assets.leetcode.com/uploads/2021/05/03/width1-tree.jpg)

```python
输入：root = [1,3,2,5,3,null,9]
输出：4
解释：最大宽度出现在树的第 3 层，宽度为 4 (5,3,null,9)。
```

- 示例 2：

![](https://assets.leetcode.com/uploads/2022/03/14/maximum-width-of-binary-tree-v3.jpg)

```python
输入：root = [1,3,2,5,null,null,9,6,null,7]
输出：7
解释：最大宽度出现在树的第 4 层，宽度为 7 (6,null,null,null,null,null,7) 。
```

## 解题思路

### 思路 1：广度优先搜索

最直观的做法是，求出每一层的宽度，然后求出所有层高度的最大值。

在计算每一层宽度时，根据题意，两端点之间的 `null` 节点也计入长度，所以我们可以对包括 `null` 节点在内的该二叉树的所有节点进行编号。

也就是满二叉树的编号规则：如果当前节点的编号为 $i$，则左子节点编号记为 $i \times 2 + 1$，则右子节点编号为 $i \times 2 + 2$。

接下来我们使用广度优先搜索方法遍历每一层的节点，在向队列中添加节点时，将该节点与该节点对应的编号一同存入队列中。

这样在计算每一层节点的宽度时，我们可以通过队列中队尾节点的编号与队头节点的编号，快速计算出当前层的宽度。并计算出所有层宽度的最大值。

### 思路 1：代码

```python
class Solution:
    def widthOfBinaryTree(self, root: Optional[TreeNode]) -> int:
        if not root:
            return False

        queue = collections.deque([[root, 0]])
        ans = 0
        while queue:
            ans = max(ans, queue[-1][1] - queue[0][1] + 1)
            size = len(queue)
            for _ in range(size):
                cur, index = queue.popleft()
                if cur.left:
                    queue.append([cur.left, index * 2 + 1])
                if cur.right:
                    queue.append([cur.right, index * 2 + 2])
        return ans
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(n)$，其中 $n$ 为二叉树的节点数。
- **空间复杂度**：$O(n)$。
# [0664. 奇怪的打印机](https://leetcode.cn/problems/strange-printer/)

- 标签：字符串、动态规划
- 难度：困难

## 题目链接

- [0664. 奇怪的打印机 - 力扣](https://leetcode.cn/problems/strange-printer/)

## 题目大意

**描述**：有一台奇怪的打印机，有以下两个功能：

1. 打印机每次只能打印由同一个字符组成的序列，比如：`"aaaa"`、`"bbb"`。
2. 每次可以从起始位置到结束的任意为止打印新字符，并且会覆盖掉原有字符。

现在给定一个字符串 $s$。

**要求**：计算这个打印机打印出字符串 $s$ 需要的最少打印次数。

**说明**：

- $1 \le s.length \le 100$。
- $s$ 由小写英文字母组成。

**示例**：

- 示例 1：

```python
输入：s = "aaabbb"
输出：2
解释：首先打印 "aaa" 然后打印 "bbb"。
```

- 示例 2：

```python
输入：s = "aba"
输出：2
解释：首先打印 "aaa" 然后在第二个位置打印 "b" 覆盖掉原来的字符 'a'。
```

## 解题思路

对于字符串 $s$，我们可以先考虑区间 $[i, j]$ 上的子字符串需要的最少打印次数。

1. 如果区间 $[i, j]$ 内只有 $1$ 种字符，则最少打印次数为 $1$，即：$dp[i][i] = 1$。
2. 如果区间 $[i, j]$ 内首尾字符相同，即 $s[i] == s[j]$，则我们在打印 $s[i]$ 的同时我们可以顺便打印 $s[j]$，这样我们可以忽略 $s[j]$，只考虑剩下区间 $[i, j - 1]$ 的打印情况，即：$dp[i][j] = dp[i][j - 1]$。
3. 如果区间 $[i, j]$ 上首尾字符不同，即 $s[i] \ne s[j]$，则枚举分割点 $k$，将区间 $[i, j]$ 分为区间 $[i, k]$ 与区间 $[k + 1, j]$，使得 $dp[i][k] + dp[k + 1][j]$ 的值最小即为 $dp[i][j]$。

### 思路 1：动态规划

###### 1. 划分阶段

按照区间长度进行阶段划分。

###### 2. 定义状态

定义状态 $dp[i][j]$ 表示为：打印第 $i$ 个字符到第 $j$ 个字符需要的最少打印次数。

###### 3. 状态转移方程

1. 如果 $s[i] == s[j]$，则我们在打印 $s[i]$ 的同时我们可以顺便打印 $s[j]$，这样我们可以忽略 $s[j]$，只考虑剩下区间 $[i, j - 1]$ 的打印情况，即：$dp[i][j] = dp[i][j - 1]$。
2. 如果 $s[i] \ne s[j]$，则枚举分割点 $k$，将区间 $[i, j]$ 分为区间 $[i, k]$ 与区间 $[k + 1, j]$，使得 $dp[i][k] + dp[k + 1][j]$ 的值最小即为 $dp[i][j]$，即：$dp[i][j] = min(dp[i][j], dp[i][k] + dp[k + 1][j])$。

###### 4. 初始条件

- 初始时，打印单个字符的最少打印次数为 $1$，即 $dp[i][i] = 1$。

###### 5. 最终结果

根据我们之前定义的状态，$dp[i][j]$ 表示为：打印第 $i$ 个字符到第 $j$ 个字符需要的最少打印次数。 所以最终结果为 $dp[0][size - 1]$。

### 思路 1：代码

```python
class Solution:
    def strangePrinter(self, s: str) -> int:
        size = len(s)
        dp = [[float('inf') for _ in range(size)] for _ in range(size)]
        for i in range(size):
            dp[i][i] = 1
            
        for l in range(2, size + 1):
            for i in range(size):
                j = i + l - 1
                if j >= size:
                    break
                if s[i] == s[j]:
                    dp[i][j] = dp[i][j - 1]
                else:
                    for k in range(i, j):
                        dp[i][j] = min(dp[i][j], dp[i][k] + dp[k + 1][j])

        return dp[0][size - 1]
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(n^3)$，其中 $n$ 为字符串 $s$ 的长度。
- **空间复杂度**：$O(n^2)$。

# [0665. 非递减数列](https://leetcode.cn/problems/non-decreasing-array/)

- 标签：数组
- 难度：中等

## 题目链接

- [0665. 非递减数列 - 力扣](https://leetcode.cn/problems/non-decreasing-array/)

## 题目大意

给定一个整数数组 nums，问能否在最多改变 1 个元素的条件下，使数组变为非递减序列。若能，返回 True，不能则返回 False。

## 解题思路

循环遍历数组，寻找 nums[i] > nums[i+1] 的情况，一旦这种情况出现超过 2 次，则不可能最多改变 1 个元素，直接返回 False。

遇到 nums[i] > nums[i+1] 的情况，应该手动调节某位置上元素使数组有序。此时，有两种选择：

- 将 nums[i] 调低，与 nums[i-1] 持平
- 将 nums[i+1] 调高，与 nums[i] 持平

若选择第一种调节方式，如果调节前 nums[i-1] > nums[i+1]，那么调节完 nums[i] 之后，nums[i-1] 还是比 nums[i+1] 大，不可取。

所以应选择第二种调节方式，如果调节前 nums[i-1] > nums[i+1]，那么调节完 nums[i+1] 之后 nums[i-1] < nums[i] <= nums[i+1]，满足非递减要求。

最终如果最多调整过一次，且 nums[i] > nums[i+1] 的情况也最多出现过一次，则返回 True。

## 代码

```python
class Solution:
    def checkPossibility(self, nums: List[int]) -> bool:
        count = 0
        for i in range(len(nums)-1):
            if nums[i] > nums[i+1]:
                count += 1
                if count > 1:
                    return False
                if i > 0 and nums[i-1] > nums[i+1]:
                    nums[i+1] = nums[i]

        return True
```

# [0669. 修剪二叉搜索树](https://leetcode.cn/problems/trim-a-binary-search-tree/)

- 标签：树、深度优先搜索、二叉搜索树、二叉树
- 难度：中等

## 题目链接

- [0669. 修剪二叉搜索树 - 力扣](https://leetcode.cn/problems/trim-a-binary-search-tree/)

## 题目大意

给定一棵二叉搜索树的根节点 `root`，同时给定最小边界 `low` 和最大边界 `high`。通过修建二叉搜索树，使得所有节点值都在 `[low, high]` 中。修剪树不应该改变保留在树中的元素的相对结构（即如果没有移除节点，则该节点的父节点关系、子节点关系都应当保留）。

现在要求返回修建过后的二叉树的根节点。

## 解题思路

递归修剪，函数返回值为修剪之后的树。

- 如果当前根节点为空，则直接返回 None。
- 如果当前根节点的值小于 `low`，则该节点左子树全部都小于最小边界，则删除左子树，然后递归遍历右子树，在右子树中寻找符合条件的节点。
- 如果当前根节点的值大于 `hight`，则该节点右子树全部都大于最大边界，则删除右子树，然后递归遍历左子树，在左子树中寻找符合条件的节点。
- 如果在最小边界和最大边界的区间内，则分别从左右子树寻找符合条件的节点作为根的左右子树。

## 代码

```python
class Solution:
    def trimBST(self, root: TreeNode, low: int, high: int) -> TreeNode:
        if not root:
            return None
        if root.val < low:
            right = self.trimBST(root.right, low, high)
            return right
        if root.val > high:
            left = self.trimBST(root.left, low, high)
            return left

        root.left = self.trimBST(root.left, low, high)
        root.right = self.trimBST(root.right, low, high)
        return root
```

# [0673. 最长递增子序列的个数](https://leetcode.cn/problems/number-of-longest-increasing-subsequence/)

- 标签：树状数组、线段树、数组、动态规划
- 难度：中等

## 题目链接

- [0673. 最长递增子序列的个数 - 力扣](https://leetcode.cn/problems/number-of-longest-increasing-subsequence/)

## 题目大意

**描述**：给定一个未排序的整数数组 `nums`。

**要求**：返回最长递增子序列的个数。

**说明**：

- 子数列必须是严格递增的。
- $1 \le nums.length \le 2000$。
- $-10^6 \le nums[i] \le 10^6$。

**示例**：

- 示例 1：

```python
输入：[1,3,5,4,7]
输出：2
解释：有两个最长递增子序列，分别是 [1, 3, 4, 7] 和[1, 3, 5, 7]。
```

## 解题思路

### 思路 1：动态规划

可以先做题目 [0300. 最长递增子序列](https://leetcode.cn/problems/longest-increasing-subsequence/)。

动态规划的状态 `dp[i]` 表示为：以第 `i` 个数字结尾的前 `i` 个元素中最长严格递增子序列的长度。

两重循环遍历前 `i` 个数字，对于 $0 \le j \le i$：

- 当 `nums[j] < nums[i]` 时，`nums[i]` 可以接在 `nums[j]` 后面，此时以第 `i` 个数字结尾的最长严格递增子序列长度 + 1，即 `dp[i] = dp[j] + 1`。
- 当 `nums[j] ≥ nums[i]` 时，可以直接跳过。

则状态转移方程为：`dp[i] = max(dp[i], dp[j] + 1)`，`0 ≤ j ≤ i`，`nums[j] < nums[i]`。

最后再遍历一遍 dp 数组，求出最大值即为最长递增子序列的长度。

现在求最长递增子序列的个数。则需要在求解的过程中维护一个 `count` 数组，用来保存以 `nums[i]` 结尾的最长递增子序列的个数。

对于 $0 \le j \le i$：

- 当 `nums[j] < nums[i]`，而且 `dp[j] + 1 > dp[i]` 时，说明第一次找到 `dp[j] + 1`长度且以`nums[i]`结尾的最长递增子序列，则以 `nums[i]` 结尾的最长递增子序列的组合数就等于以 `nums[j]` 结尾的组合数，即 `count[i] = count[j]`。
- 当 `nums[j] < nums[i]`，而且 `dp[j] + 1 == dp[i]` 时，说明以 `nums[i]` 结尾且长度为 `dp[j] + 1` 的递增序列已找到过一次了，则以 `nums[i]` 结尾的最长递增子序列的组合数要加上以 `nums[j]` 结尾的组合数，即 `count[i] += count[j]`。

- 然后根据遍历 dp 数组得到的最长递增子序列的长度 max_length，然后再一次遍历 dp 数组，将所有 `dp[i] == max_length` 情况下的组合数 `coun[i]` 累加起来，即为最长递增序列的个数。

### 思路 1：动态规划代码

```python
class Solution:
    def findNumberOfLIS(self, nums: List[int]) -> int:
        size = len(nums)
        dp = [1 for _ in range(size)]
        count = [1 for _ in range(size)]
        for i in range(size):
            for j in range(i):
                if nums[j] < nums[i]:
                    if dp[j] + 1 > dp[i]:
                        dp[i] = dp[j] + 1
                        count[i] = count[j]
                    elif dp[j] + 1 == dp[i]:
                        count[i] += count[j]

        max_length = max(dp)
        res = 0
        for i in range(size):
            if dp[i] == max_length:
                res += count[i]
        return res
```

### 思路 2：线段树

题目中 `nums` 的长度 为 $[1, 2000]$，值域为 $[-10^6, 10^6]$。

值域范围不是特别大，我们可以直接用线段树保存整个值域区间。但因为数组的长度只有 `2000`，所以算法效率更高的做法是先对数组进行离散化处理。把数组中的元素按照大小依次映射到 `[0, len(nums) - 1]` 这个区间。

1. 构建一棵长度为 `len(nums)` 的线段树，其中每个线段树的节点保存一个二元组。这个二元组 `val = [length, count]` 用来表示：以当前节点为结尾的子序列所能达到的最长递增子序列长度 `length` 和最长递增子序列对应的数量 `count`。
2. 顺序遍历数组 `nums`。对于当前元素 `nums[i]`：
3. 查找 `[0, nums[i - 1]]` 离散化后对应区间节点的二元组，也就是查找以区间 `[0, nums[i - 1]]` 上的点为结尾的子序列所能达到的最长递增子序列长度和其对应的数量，即 `val = [length, count]`。
   - 如果所能达到的最长递增子序列长度为 `0`，则加入 `nums[i]` 之后最长递增子序列长度变为 `1`，且数量也变为 `1`。
   - 如果所能达到的最长递增子序列长度不为 `0`，则加入 `nums[i]` 之后最长递增子序列长度 +1，但数量不变。
4. 根据上述计算的 `val` 值更新 `nums[i]` 对应节点的 `val` 值。 
5. 然后继续向后遍历，重复进行第 `3` ~ `4` 步操作。
6. 最后查询以区间 `[0, nums[len(nums) - 1]]` 上的点为结尾的子序列所能达到的最长递增子序列长度和其对应的数量。返回对应的数量即为答案。

### 思路 2：线段树代码

```python
# 线段树的节点类
class SegTreeNode:
    def __init__(self, val=[0, 1]):
        self.left = -1                              # 区间左边界
        self.right = -1                             # 区间右边界
        self.val = val                              # 节点值（区间值）
        
        
        
# 线段树类
class SegmentTree:
    # 初始化线段树接口
    def __init__(self, size):
        self.size = size
        self.tree = [SegTreeNode() for _ in range(4 * self.size)]  # 维护 SegTreeNode 数组
        if self.size > 0:
            self.__build(0, 0, self.size - 1)
    
    # 单点更新接口：将 nums[i] 更改为 val
    def update_point(self, i, val):
        self.__update_point(i, val, 0)
        
    # 区间查询接口：查询区间为 [q_left, q_right] 的区间值
    def query_interval(self, q_left, q_right):
        return self.__query_interval(q_left, q_right, 0)

        
    # 以下为内部实现方法
    
    # 构建线段树实现方法：节点的存储下标为 index，节点的区间为 [left, right]
    def __build(self, index, left, right):
        self.tree[index].left = left
        self.tree[index].right = right
        if left == right:                           # 叶子节点，节点值为对应位置的元素值
            self.tree[index].val = [0, 0]
            return
    
        mid = left + (right - left) // 2            # 左右节点划分点
        left_index = index * 2 + 1                  # 左子节点的存储下标
        right_index = index * 2 + 2                 # 右子节点的存储下标
        self.__build(left_index, left, mid)         # 递归创建左子树
        self.__build(right_index, mid + 1, right)   # 递归创建右子树

        self.tree[index].val = self.merge(self.tree[left_index].val, self.tree[right_index].val)   # 向上更新节点的区间值
    
    # 单点更新实现方法：将 nums[i] 更改为 val，节点的存储下标为 index
    def __update_point(self, i, val, index):
        left = self.tree[index].left
        right = self.tree[index].right
        
        if left == i and right == i:
            self.tree[index].val = self.merge(self.tree[index].val, val)
            return
        
        mid = left + (right - left) // 2            # 左右节点划分点
        left_index = index * 2 + 1                  # 左子节点的存储下标
        right_index = index * 2 + 2                 # 右子节点的存储下标
        if i <= mid:                                # 在左子树中更新节点值
            self.__update_point(i, val, left_index)
        else:                                       # 在右子树中更新节点值
            self.__update_point(i, val, right_index)
        
        self.tree[index].val = self.merge(self.tree[left_index].val, self.tree[right_index].val)   # 向上更新节点的区间值
    
    
    # 区间查询实现方法：在线段树中搜索区间为 [q_left, q_right] 的区间值
    def __query_interval(self, q_left, q_right, index):
        left = self.tree[index].left
        right = self.tree[index].right
        
        if left >= q_left and right <= q_right:     # 节点所在区间被 [q_left, q_right] 所覆盖
            return self.tree[index].val             # 直接返回节点值
        if right < q_left or left > q_right:        # 节点所在区间与 [q_left, q_right] 无关
            return [0, 0]

        mid = left + (right - left) // 2            # 左右节点划分点
        left_index = index * 2 + 1                  # 左子节点的存储下标
        right_index = index * 2 + 2                 # 右子节点的存储下标
        res_left = [0, 0]
        res_right = [0, 0]
        if q_left <= mid:                           # 在左子树中查询
            res_left = self.__query_interval(q_left, q_right, left_index)
        if q_right > mid:                           # 在右子树中查询
            res_right = self.__query_interval(q_left, q_right, right_index)
        
        # 返回合并结果
        return self.merge(res_left, res_right)

    # 向上合并实现方法
    def merge(self, val1, val2):
        val = [0, 0]
        if val1[0] == val2[0]:                      # 递增子序列长度一致，则合并后最长递增子序列个数为之前两者之和
            val = [val1[0], val1[1] + val2[1]]
        elif val1[0] < val2[0]:                     # 如果递增子序列长度不一致，则合并后最长递增子序列个数取较长一方的个数
            val = [val2[0], val2[1]]
        else:
            val = [val1[0], val1[1]]
        return val

class Solution:
    def findNumberOfLIS(self, nums: List[int]) -> int:

        # 离散化处理
        num_dict = dict()
        nums_sort = sorted(nums)
        for i in range(len(nums_sort)):
            num_dict[nums_sort[i]] = i
        
        # 构造线段树
        self.STree = SegmentTree(len(nums_sort))

        for num in nums:
            index = num_dict[num]
            # 查询 [0, nums[index - 1]] 区间上以 nums[index - 1] 结尾的子序列所能达到的最长递增子序列长度和对应数量
            val = self.STree.query_interval(0, index - 1)
            # 如果当前最长递增子序列长度为 0，则加入 num 之后最长递增子序列长度为 1，且数量为 1
            # 如果当前最长递增子序列长度不为 0，则加入 num 之后最长递增子序列长度 +1，但数量不变
            if val[0] == 0:
                val = [1, 1]
            else:
                val = [val[0] + 1, val[1]]
            self.STree.update_point(index, val)
        return self.STree.query_interval(0, len(nums_sort) - 1)[1]
```

# [0674. 最长连续递增序列](https://leetcode.cn/problems/longest-continuous-increasing-subsequence/)

- 标签：数组
- 难度：简单

## 题目链接

- [0674. 最长连续递增序列 - 力扣](https://leetcode.cn/problems/longest-continuous-increasing-subsequence/)

## 题目大意

**描述**：给定一个未经排序的数组 $nums$。

**要求**：找到最长且连续递增的子序列，并返回该序列的长度。

**说明**：

- **连续递增的子序列**：可以由两个下标 $l$ 和 $r$（$l < r$）确定，如果对于每个 $l \le i < r$，都有 $nums[i] < nums[i + 1] $，那么子序列 $[nums[l], nums[l + 1], ..., nums[r - 1], nums[r]]$ 就是连续递增子序列。
- $1 \le nums.length \le 10^4$。
- $-10^9 \le nums[i] \le 10^9$。

**示例**：

- 示例 1：

```python
输入：nums = [1,3,5,4,7]
输出：3
解释：最长连续递增序列是 [1,3,5], 长度为 3。尽管 [1,3,5,7] 也是升序的子序列, 但它不是连续的，因为 5 和 7 在原数组里被 4 隔开。 
```

- 示例 2：

```python
输入：nums = [2,2,2,2,2]
输出：1
解释：最长连续递增序列是 [2], 长度为 1。
```

## 解题思路

### 思路 1：动态规划

###### 1. 定义状态

定义状态 $dp[i]$ 表示为：以 $nums[i]$ 结尾的最长且连续递增的子序列长度。

###### 2. 状态转移方程

因为求解的是连续子序列，所以只需要考察相邻元素的状态转移方程。

如果一个较小的数右侧相邻元素为一个较大的数，则会形成一个更长的递增子序列。

对于相邻的数组元素 $nums[i - 1]$ 和 $nums[i]$ 来说：

- 如果 $nums[i - 1] < nums[i]$，则 $nums[i]$ 可以接在 $nums[i - 1]$ 后面，此时以 $nums[i]$ 结尾的最长递增子序列长度会在「以 $nums[i - 1]$ 结尾的最长递增子序列长度」的基础上加 $1$，即 $dp[i] = dp[i - 1] + 1$。

- 如果 $nums[i - 1] >= nums[i]$，则 $nums[i]$ 不可以接在 $nums[i - 1]$ 后面，可以直接跳过。

综上，我们的状态转移方程为：$dp[i] = dp[i - 1] + 1$，$nums[i - 1] < nums[i]$。

###### 3. 初始条件

默认状态下，把数组中的每个元素都作为长度为 $1$ 的最长且连续递增的子序列长度。即 $dp[i] = 1$。

###### 4. 最终结果

根据我们之前定义的状态，$dp[i]$ 表示为：以 $nums[i]$ 结尾的最长且连续递增的子序列长度。则为了计算出最大值，则需要再遍历一遍 $dp$ 数组，求出最大值即为最终结果。

### 思路 1：动态规划代码

```python
class Solution:
    def findLengthOfLCIS(self, nums: List[int]) -> int:
        size = len(nums)
        dp = [1 for _ in range(size)]

        for i in range(1, size):
            if nums[i - 1] < nums[i]:
                dp[i] = dp[i - 1] + 1
        
        return max(dp)
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(n)$。一重循环遍历的时间复杂度为 $O(n)$，最后求最大值的时间复杂度是 $O(n)$，所以总体时间复杂度为 $O(n)$。
- **空间复杂度**：$O(n)$。用到了一维数组保存状态，所以总体空间复杂度为 $O(n)$。

### 思路 2：滑动窗口（不定长度）

1. 设定两个指针：$left$、$right$，分别指向滑动窗口的左右边界，保证窗口内为连续递增序列。使用 $window\underline{\hspace{0.5em}}len$ 存储当前窗口大小，使用 $max\underline{\hspace{0.5em}}len$ 维护最大窗口长度。
2. 一开始，$left$、$right$ 都指向 $0$。
3. 将最右侧元素 $nums[right]$ 加入当前连续递增序列中，即当前窗口长度加 $1$（`window_len += 1`）。
4. 判断当前元素 $nums[right]$ 是否满足连续递增序列。
5. 如果 $right > 0$ 并且 $nums[right - 1] \ge nums[right]$ ，说明不满足连续递增序列，则将 $left$ 移动到窗口最右侧，重置当前窗口长度为 $1$（`window_len = 1`）。
6. 记录当前连续递增序列的长度，并更新最长连续递增序列的长度。
7. 继续右移 $right$，直到 $right \ge len(nums)$ 结束。
8. 输出最长连续递增序列的长度 $max\underline{\hspace{0.5em}}len$。

### 思路 2：代码

```python
class Solution:
    def findLengthOfLCIS(self, nums: List[int]) -> int:
        size = len(nums)
        left, right = 0, 0
        window_len = 0
        max_len = 0
        
        while right < size:
            window_len += 1
            
            if right > 0 and nums[right - 1] >= nums[right]:
                left = right
                window_len = 1

            max_len = max(max_len, window_len)
            right += 1
            
        return max_len
```

### 思路 2：复杂度分析

- **时间复杂度**：$O(n)$。
- **空间复杂度**：$O(1)$。# [0676. 实现一个魔法字典](https://leetcode.cn/problems/implement-magic-dictionary/)

- 标签：设计、字典树、哈希表、字符串
- 难度：中等

## 题目链接

- [0676. 实现一个魔法字典 - 力扣](https://leetcode.cn/problems/implement-magic-dictionary/)

## 题目大意

**要求**：设计一个使用单词表进行初始化的数据结构。单词表中的单词互不相同。如果给出一个单词，要求判定能否将该单词中的一个字母替换成另一个字母，是的所形成的新单词已经在够构建的单词表中。

实现 MagicDictionary 类：

- `MagicDictionary()` 初始化对象。
- `void buildDict(String[] dictionary)` 使用字符串数组 `dictionary` 设定该数据结构，`dictionary` 中的字符串互不相同。
- `bool search(String searchWord)` 给定一个字符串 `searchWord`，判定能否只将字符串中一个字母换成另一个字母，使得所形成的新字符串能够与字典中的任一字符串匹配。如果可以，返回 `True`；否则，返回 `False`。

**说明**：

- $1 \le dictionary.length \le 100$。
- $1 \le dictionary[i].length \le 100$。
- `dictionary[i]` 仅由小写英文字母组成。
- `dictionary` 中的所有字符串互不相同。
- $1 \le searchWord.length \le 100$。
- `searchWord` 仅由小写英文字母组成。
- `buildDict` 仅在 `search` 之前调用一次。
- 最多调用 $100$ 次 `search`。

**示例**：

- 示例 1：

```python
输入
["MagicDictionary", "buildDict", "search", "search", "search", "search"]
[[], [["hello", "leetcode"]], ["hello"], ["hhllo"], ["hell"], ["leetcoded"]]
输出
[null, null, false, true, false, false]

解释
MagicDictionary magicDictionary = new MagicDictionary();
magicDictionary.buildDict(["hello", "leetcode"]);
magicDictionary.search("hello"); // 返回 False
magicDictionary.search("hhllo"); // 将第二个 'h' 替换为 'e' 可以匹配 "hello" ，所以返回 True
magicDictionary.search("hell"); // 返回 False
magicDictionary.search("leetcoded"); // 返回 False
```

## 解题思路

### 思路 1：字典树

1. 构造一棵字典树。
2. `buildDict` 方法中将所有单词存入字典树中。
3. `search` 方法中替换 `searchWord` 每一个位置上的字符，然后在字典树中查询。

### 思路 1：代码

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

### 思路 1：复杂度分析

- **时间复杂度**：初始化操作是 $O(1)$。构建操作是 $O(|dictionary|)$，搜索操作是 $O(|searchWord| \times |\sum|)$。其中 $|dictionary|$ 是字符串数组 `dictionary` 中的字符个数，$|searchWord|$ 是查询操作中字符串的长度，$|\sum|$ 是字符集的大小。
- **空间复杂度**：$O(|dicitonary|)$。# [0677. 键值映射](https://leetcode.cn/problems/map-sum-pairs/)

- 标签：设计、字典树、哈希表、字符串
- 难度：中等

## 题目链接

- [0677. 键值映射 - 力扣](https://leetcode.cn/problems/map-sum-pairs/)

## 题目大意

**要求**：实现一个 MapSum 类，支持两个方法，`insert` 和 `sum`：

- `MapSum()` 初始化 MapSum 对象。
- `void insert(String key, int val)` 插入 `key-val` 键值对，字符串表示键 `key`，整数表示值 `val`。如果键 `key` 已经存在，那么原来的键值对将被替代成新的键值对。
- `int sum(string prefix)` 返回所有以该前缀 `prefix` 开头的键 `key` 的值的总和。

**说明**：

- $1 \le key.length, prefix.length \le 50$。
- `key` 和 `prefix` 仅由小写英文字母组成。
- $1 \le val \le 1000$。
- 最多调用 $50$ 次 `insert` 和 `sum`。

**示例**：

- 示例 1：

```python
输入：
["MapSum", "insert", "sum", "insert", "sum"]
[[], ["apple", 3], ["ap"], ["app", 2], ["ap"]]
输出：
[null, null, 3, null, 5]

解释：
MapSum mapSum = new MapSum();
mapSum.insert("apple", 3);  
mapSum.sum("ap");           // 返回 3 (apple = 3)
mapSum.insert("app", 2);    
mapSum.sum("ap");           // 返回 5 (apple + app = 3 + 2 = 5)
```

## 解题思路

### 思路 1：字典树

可以构造前缀树（字典树）解题。

- 初始化时，构建一棵前缀树（字典树），并增加 `val` 变量。

- 调用插入方法时，用字典树存储 `key`，并在对应字母节点存储对应的 `val`。
- 在调用查询总和方法时，先查找该前缀 `prefix` 对应的前缀树节点，从该节点开始，递归遍历该节点的子节点，并累积子节点的 `val`，进行求和，并返回求和累加结果。

### 思路 1：代码

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

### 思路 1：复杂度分析

- **时间复杂度**：`insert` 操作的时间复杂度为 $O(|key|)$。其中 $|key|$ 是每次插入字符串 `key` 的长度。`sum` 操作的时间复杂度是 $O(|prefix|)$，其中 $O(| prefix |)$ 是查询字符串 `prefix` 的长度。
- **空间复杂度**：$O(|T| \times m)$。其中 $|T|$ 表示字符串 `key` 的最大长度，$m$ 表示 `key - val` 的键值数目。# [0678. 有效的括号字符串](https://leetcode.cn/problems/valid-parenthesis-string/)

- 标签：栈、贪心、字符串、动态规划
- 难度：中等

## 题目链接

- [0678. 有效的括号字符串 - 力扣](https://leetcode.cn/problems/valid-parenthesis-string/)

## 题目大意

**描述**：给定一个只包含三种字符的字符串：`（` ，`)` 和 `*`。有效的括号字符串具有如下规则：

1. 任何左括号 `(` 必须有相应的右括号 `)`。
2. 任何右括号 `)` 必须有相应的左括号 `(`。
3. 左括号 `(` 必须在对应的右括号之前 `)`。
4. `*` 可以被视为单个右括号 `)`，或单个左括号 `(`，或一个空字符串。
5. 一个空字符串也被视为有效字符串。

**要求**：验证这个字符串是否为有效字符串。如果是，则返回 `True`；否则，则返回 `False`。

**说明**：

- 字符串大小将在 `[1, 100]` 范围内。

**示例**：

- 示例 1：

```python
输入："(*)"
输出：True
```

## 解题思路

### 思路 1：动态规划（时间复杂度为 $O(n^3)$）

###### 1. 划分阶段

按照子串的起始位置进行阶段划分。

###### 2. 定义状态

定义状态 `dp[i][j]` 表示为：从下标 `i` 到下标 `j` 的子串是否为有效的括号字符串，其中 （$0 \le i < j < size$，$size$ 为字符串长度）。如果是则 `dp[i][j] = True`，否则，`dp[i][j] = False`。

###### 3. 状态转移方程

长度大于 `2` 时，我们需要根据 `s[i]` 和 `s[j]` 的情况，以及子串中间的有效字符串情况来判断 `dp[i][j]`。

- 如果 `s[i]`、`s[j]` 分别表示左括号和右括号，或者为 `'*'`（此时 `s[i]`、`s[j]` 可以分别看做是左括号、右括号）。则如果 `dp[i + 1][j - 1] == True` 时，`dp[i][j] = True`。
- 如果可以将从下标 `i` 到下标 `j` 的子串从中间分开为两个有效字符串，则 `dp[i][j] = True`。即如果存在 $i \le k < j$，使得 `dp[i][k] == True` 并且 `dp[k + 1][j] == True`，则 `dp[i][j] = True`。

###### 4. 初始条件

- 当子串的长度为 `1`，并且该字符串为 `'*'` 时，子串可看做是空字符串，此时子串是有效的括号字符串。
- 当子串的长度为 `2` 时，如果两个字符可以分别看做是左括号和右括号，子串可以看做是 `"()"`，此时子串是有效的括号字符串。

###### 5. 最终结果

根据我们之前定义的状态，`dp[i][j]` 表示为：从下标 `i` 到下标 `j` 的子串是否为有效的括号字符串。则最终结果为 `dp[0][size - 1]`。

### 思路 1：动态规划（时间复杂度为 $O(n^3)$）代码

```python
class Solution:
    def checkValidString(self, s: str) -> bool:
        size = len(s)
        dp = [[False for _ in range(size)] for _ in range(size)]

        for i in range(size):
            if s[i] == '*':
                dp[i][i] = True

        for i in range(1, size):
            if (s[i - 1] == '(' or s[i - 1] == '*') and (s[i] == ')' or s[i] == '*'):
                dp[i - 1][i] = True

        for i in range(size - 3, -1, -1):
            for j in range(i + 2, size):
                if (s[i] == '(' or s[i] == '*') and (s[j] == ')' or s[j] == '*'):
                    dp[i][j] = dp[i + 1][j - 1]
                for k in range(i, j):
                    if dp[i][j]:
                        break
                    dp[i][j] = dp[i][k] and dp[k + 1][j]

        return dp[0][size - 1]
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(n^3)$。三重循环遍历的时间复杂度是 $O(n^3)$。
- **空间复杂度**：$O(n^2)$。用到了二维数组保存状态，所以总体空间复杂度为 $O(n^2)$。

### 思路 2：动态规划（时间复杂度为 $O(n^2)$）

###### 1. 划分阶段

按照字符串的结束位置进行阶段划分。

###### 2. 定义状态

定义状态 `dp[i][j]` 表示为：前 `i` 个字符能否通过补齐 `j` 个右括号成为有效的括号字符串。

###### 3. 状态转移方程

1. 如果 `s[i] == '('`，则如果前 `i - 1` 个字符通过补齐 `j - 1` 个右括号成为有效的括号字符串，则前 `i` 个字符就能通过补齐 `j` 个右括号成为有效的括号字符串（比前 `i - 1` 个字符需要多补一个右括号）。也就是说，如果 `s[i] == '('` 并且 `dp[i - 1][j - 1] == True`，则 `dp[i][j] = True`。
2. 如果 `s[i] == ')'`，则如果前 `i - 1` 个字符通过补齐 `j + 1` 个右括号成为有效的括号字符串，则前 `i` 个字符就能通过补齐 `j` 个右括号成为有效的括号字符串（比前 `i - 1` 个字符需要少补一个右括号）。也就是说，如果 `s[i] == ')'` 并且 `dp[i - 1][j + 1] == True`，则 `dp[i][j] = True`。
3. 如果 `s[i] == '*'`，而 `'*'` 可以表示空字符串、左括号或者右括号，则 `dp[i][j]` 取决于这三种情况，只要有一种情况为 `True`，则 `dp[i][j] = True`。也就是说，如果 `s[i] == '*'`，则 `dp[i][j] = dp[i - 1][j] or dp[i - 1][j - 1]`。

###### 4. 初始条件

- `0` 个字符可以通过补齐 `0` 个右括号成为有效的括号字符串（空字符串），即 `dp[0][0] = 0`。

###### 5. 最终结果

根据我们之前定义的状态，`dp[i][j]` 表示为：前 `i` 个字符能否通过补齐 `j` 个右括号成为有效的括号字符串。。则最终结果为 `dp[size][0]`。

### 思路 2：动态规划（时间复杂度为 $O(n^2)$）代码

```python
class Solution:
    def checkValidString(self, s: str) -> bool:
        size = len(s)
        dp = [[False for _ in range(size + 1)] for _ in range(size + 1)]
        dp[0][0] = True
        for i in range(1, size + 1):
            for j in range(i + 1):
                if s[i - 1] == '(':
                    if j > 0:
                        dp[i][j] = dp[i - 1][j - 1]
                elif s[i - 1] == ')':
                    if j < i:
                        dp[i][j] = dp[i - 1][j + 1]
                else:
                    dp[i][j] = dp[i - 1][j]
                    if j > 0:
                        dp[i][j] = dp[i][j] or dp[i - 1][j - 1]
                    if j < i:
                        dp[i][j] = dp[i][j] or dp[i - 1][j + 1]

        return dp[size][0]
```

### 思路 2：复杂度分析

- **时间复杂度**：$O(n^2)$。两重循环遍历的时间复杂度是 $O(n^2)$。
- **空间复杂度**：$O(n^2)$。用到了二维数组保存状态，所以总体空间复杂度为 $O(n^2)$。# [0680. 验证回文串 II](https://leetcode.cn/problems/valid-palindrome-ii/)

- 标签：贪心、双指针、字符串
- 难度：简单

## 题目链接

- [0680. 验证回文串 II - 力扣](https://leetcode.cn/problems/valid-palindrome-ii/)

## 题目大意

给定一个非空字符串 `s`。

要求：判断如果最多从字符串中删除一个字符能否得到一个回文字符串。

## 解题思路

题目要求在最多删除一个字符的情况下是否能得到一个回文字符串。最直接的思路是遍历各个字符，判断将该字符删除之后，剩余字符串是否是回文串。但是这种思路的时间复杂度是 $O(n^2)$，解答的话会超时。

我们可以通过双指针 + 贪心算法来减少时间复杂度。具体做法如下：

- 使用两个指针变量 `left`、`right` 分别指向字符串的开始和结束位置。

- 判断 `s[left]` 是否等于 `s[right]`。
  - 如果等于，则 `left` 右移、`right`左移。
  - 如果不等于，则判断 `s[left: right - 1]` 或 `s[left + 1, right]` 是为回文串。
    - 如果是则返回 `True`。
    - 如果不是则返回 `False`，然后继续判断。
- 如果 `right >= left`，则说明字符串 `s` 本身就是回文串，返回 `True`。

## 代码

```python
class Solution:
    def checkPalindrome(self, s: str, left: int, right: int):
        i, j = left, right
        while i < j:
            if s[i] != s[j]:
                return False
            i += 1
            j -= 1
        return True

    def validPalindrome(self, s: str) -> bool:
        left, right = 0, len(s) - 1
        while left < right:
            if s[left] == s[right]:
                left += 1
                right -= 1
            else:
                return self.checkPalindrome(s, left + 1, right) or self.checkPalindrome(s, left, right - 1)
        return True
```

# [0683. K 个关闭的灯泡](https://leetcode.cn/problems/k-empty-slots/)

- 标签：树状数组、数组、有序集合、滑动窗口
- 难度：困难

## 题目链接

- [0683. K 个关闭的灯泡 - 力扣](https://leetcode.cn/problems/k-empty-slots/)

## 题目大意

**描述**：$n$ 个灯泡排成一行，编号从 $1$ 到 $n$。最初，所有灯泡都关闭。每天只打开一个灯泡，直到 $n$ 天后所有灯泡都打开。

给定一个长度为 $n$ 的灯泡数组 $blubs$，其中 `bulls[i] = x` 意味着在第 $i + 1$ 天，我们会把在位置 $x$ 的灯泡打开，其中 $i$ 从 $0$ 开始，$x$ 从 $1$ 开始。

再给定一个整数 $k$。

**要求**：输出在第几天恰好有两个打开的灯泡，使得它们中间正好有 $k$ 个灯泡且这些灯泡全部是关闭的 。如果不存在这种情况，则返回 $-1$。如果有多天都出现这种情况，请返回最小的天数 。

**说明**：

- $n == bulbs.length$。
- $1 \le n \le 2 \times 10^4$。
- $1 \le bulbs[i] \le n$。
- $bulbs$ 是一个由从 $1$ 到 $n$ 的数字构成的排列。
- $0 \le k \le 2 \times 10^4$。

**示例**：

- 示例 1：

```python
输入：
bulbs = [1,3,2]，k = 1
输出：2
解释：
第一天 bulbs[0] = 1，打开第一个灯泡 [1,0,0]
第二天 bulbs[1] = 3，打开第三个灯泡 [1,0,1]
第三天 bulbs[2] = 2，打开第二个灯泡 [1,1,1]
返回2，因为在第二天，两个打开的灯泡之间恰好有一个关闭的灯泡。
```

- 示例 2：

```python
输入：bulbs = [1,2,3]，k = 1
输出：-1
```

## 解题思路

### 思路 1：滑动窗口

$blubs[i]$ 记录的是第 $i + 1$ 天开灯的位置。我们将其转换一下，使用另一个数组 $days$ 来存储每个灯泡的开灯时间，其中 $days[i]$ 表示第 $i$ 个位置上的灯泡的开灯时间。

- 使用 $ans$ 记录最小满足条件的天数。维护一个窗口 $left$、$right$。其中 `right = left + k + 1`。使得区间 $(left, right)$ 中所有灯泡（总共为 $k$ 个）开灯时间都晚于 $days[left]$ 和 $days[right]$。
- 对于区间 $[left, right]$，$left < i < right$：
  - 如果出现 $days[i] < days[left]$ 或者 $days[i] < days[right]$，说明不符合要求。将 $left$、$right$ 移动到 $[i, i + k + 1]$，继续进行判断。
  - 如果对于 $left < i < right$ 中所有的 $i$，都满足 $days[i] \ge days[left]$ 并且 $days[i] \ge days[right]$，说明此时满足要求。将当前答案与 $days[left]$ 和 $days[right]$ 中的较大值作比较。如果比当前答案更小，则更新答案。同时将窗口向右移动 $k $位。继续检测新的不相交间隔 $[right, right + k + 1]$。
    - 注意：之所以检测新的不相交间隔，是因为如果检测的是相交间隔，原来的 $right$ 位置元素仍在区间中，肯定会出现 $days[right] < days[right_new]$，不满足要求。所以此时相交的区间可以直接跳过，直接检测不相交的间隔。
- 直到 $right \ge len(days)$ 时跳出循环，判断是否有符合要求的答案，并返回答案 $ans$。

### 思路 1：代码

```python
class Solution:
    def kEmptySlots(self, bulbs: List[int], k: int) -> int:
        size = len(bulbs)
        days = [0 for _ in range(size)]
        for i in range(size):
            days[bulbs[i] - 1] = i + 1

        left, right = 0, k + 1
        ans = float('inf')
        while right < size:
            check_flag = True
            for i in range(left + 1, right):
                if days[i] < days[left] or days[i] < days[right]:
                    left, right = i, i + k + 1
                    check_flag = False
                    break
            if check_flag:
                ans = min(ans, max(days[left], days[right]))
                left, right = right, right + k + 1

        if ans != float('inf'):
            return ans
        else:
            return -1
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(n)$，其中 $n$ 为数组 $bulbs$ 的长度。
- **空间复杂度**：$O(n)$。

# [0684. 冗余连接](https://leetcode.cn/problems/redundant-connection/)

- 标签：深度优先搜索、广度优先搜索、并查集、图
- 难度：中等

## 题目链接

- [0684. 冗余连接 - 力扣](https://leetcode.cn/problems/redundant-connection/)

## 题目大意

**描述**：一个 `n` 个节点的树（节点值为 `1~n`）添加一条边后就形成了图，添加的这条边不属于树中已经存在的边。图的信息记录存储与长度为 `n` 的二维数组 `edges`，`edges[i] = [ai, bi]` 表示图中在 `ai` 和 `bi` 之间存在一条边。

现在给定代表边信息的二维数组 `edges`。

**要求**：找到一条可以山区的边，使得删除后的剩余部分是一个有着 `n` 个节点的树。如果有多个答案，则返回数组 `edges` 中最后出现的边。

**说明**：

- $n == edges.length$。
- $3 \le n \le 1000$。
- $edges[i].length == 2$。
- $1 \le ai < bi \le edges.length$。
- $ai ≠ bi$。
- $edges$ 中无重复元素。
- 给定的图是连通的。

**示例**：

- 示例 1：

![img](https://pic.leetcode-cn.com/1626676174-hOEVUL-image.png)

```python
输入: edges = [[1,2], [1,3], [2,3]]
输出: [2,3]
```

- 示例 2：

![img](https://pic.leetcode-cn.com/1626676179-kGxcmu-image.png)

```python
输入: edges = [[1,2], [2,3], [3,4], [1,4], [1,5]]
输出: [1,4]
```

## 解题思路

### 思路 1：并查集

树可以看做是无环的图，这道题就是要找出那条添加边之后成环的边。可以考虑用并查集来做。

1. 从前向后遍历每一条边。
2. 如果边的两个节点不在同一个集合，就加入到一个集合（链接到同一个根节点）。
3. 如果边的节点已经出现在同一个集合里，说明边的两个节点已经连在一起了，再加入这条边一定会出现环，则这条边就是所求答案。

### 思路 1：代码

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

### 思路 1：复杂度分析

- **时间复杂度**：$O(n \times \alpha(n))$。其中 $n$ 是图中的节点个数，$\alpha$ 是反 `Ackerman` 函数。
- **空间复杂度**：$O(n)$。# [0686. 重复叠加字符串匹配](https://leetcode.cn/problems/repeated-string-match/)

- 标签：字符串、字符串匹配
- 难度：中等

## 题目链接

- [0686. 重复叠加字符串匹配 - 力扣](https://leetcode.cn/problems/repeated-string-match/)

## 题目大意

**描述**：给定两个字符串 `a` 和 `b`。

**要求**：寻找重复叠加字符串 `a` 的最小次数，使得字符串 `b` 成为叠加后的字符串 `a` 的子串，如果不存在则返回 `-1`。

**说明**：

- 字符串 `"abc"` 重复叠加 `0` 次是 `""`，重复叠加 `1` 次是 `"abc"`，重复叠加 `2` 次是 `"abcabc"`。
- $1 \le a.length \le 10^4$。
- $1 \le b.length \le 10^4$。
- `a` 和 `b` 由小写英文字母组成。

**示例**：

- 示例 1：

```python
输入：a = "abcd", b = "cdabcdab"
输出：3
解释：a 重复叠加三遍后为 "abcdabcdabcd", 此时 b 是其子串。
```

- 示例 2：

```python
输入：a = "a", b = "aa"
输出：2
```

## 解题思路

### 思路 1：KMP 算法

假设字符串 `a` 的长度为 `n`，`b` 的长度为 `m`。

把 `b` 看做是模式串，把字符串 `a` 叠加后的字符串看做是文本串，这道题就变成了单模式串匹配问题。

我们可以模拟叠加字符串 `a` 后进行单模式串匹配问题。模拟叠加字符串可以通过在遍历字符串匹配时对字符串 `a` 的长度 `n` 取余来实现。

那么问题关键点就变为了如何高效的进行单模式串匹配，以及字符串循环匹配的退出条件是什么。

**单模式串匹配问题**：可以用 KMP 算法来做。

**循环匹配退出条件问题**：假设我们用 `i` 遍历 `a` 叠加后字符串，用 `j` 遍历字符串 `b`。如果字符串 `b` 是 `a` 叠加后字符串的子串，那么 `b` 有两种可能：

1. `b` 直接是原字符串 `a` 的子串：这种情况下，最多遍历到 `len(a)`。
2. `b` 是 `a` 叠加后的字符串的子串：
   1. 最多遍历到 `len(a) + len(b)`，可以写为 `while i < len(a) + len(b):`，当 `i == len(a) + len(b)` 时跳出循环。
   2. 也可以写为 `while i - j < len(a):`，这种写法中 `i - j ` 表示的是字符匹配开始的位置，如果匹配到 `len(a)` 时（即 `i - j == len(a)` 时）最开始位置的字符仍没有匹配，那么 `b` 也不可能是 `a` 叠加后的字符串的子串了，此时跳出循环。

最后我们需要计算一下重复叠加字符串 `a` 的最小次数。假设 `index` 使我们求出的匹配位置。

1. 如果 `index == -1`，则说明 `b` 不可能是 `a` 叠加后的字符串的子串，返回 `False`。
2. 如果 `len(a) - index >= len(b)`，则说明匹配位置未超过字符串 `a` 的长度，叠加 `1` 次（字符串 `a` 本身）就可以匹配。
3. 如果 `len(a) - index < len(b)`，则说明需要叠加才能匹配。此时最小叠加次数为 $\lfloor \frac{index + len(b) - 1}{len(a)} \rfloor + 1$。其中 `index`  代笔匹配开始前的字符串长度，加上 `len(b)` 后就是匹配到字符串 `b` 结束时最少需要的字符数，再 `-1` 是为了向下取整。 除以 `len(a)` 表示至少需要几个 `a`， 因为是向下取整，所以最后要加上 `1`。写成代码就是：`(index + len(b) - 1) // len(a) + 1`。

### 思路 1：代码

```python
class Solution:
    # KMP 匹配算法，T 为文本串，p 为模式串
    def kmp(self, T: str, p: str) -> int:
        n, m = len(T), len(p)

        next = self.generateNext(p)

        i, j = 0, 0
        while i - j < n:
            while j > 0 and T[i % n] != p[j]:
                j = next[j - 1]
            if T[i % n] == p[j]:
                j += 1
            if j == m:
                return i - m + 1
            i += 1
        return -1

    def generateNext(self, p: str):
        m = len(p)
        next = [0 for _ in range(m)]

        left = 0
        for right in range(1, m):
            while left > 0 and p[left] != p[right]:
                left = next[left - 1]
            if p[left] == p[right]:
                left += 1
            next[right] = left

        return next

    def repeatedStringMatch(self, a: str, b: str) -> int:
        len_a = len(a)
        len_b = len(b)
        index = self.kmp(a, b)
        if index == -1:
            return -1
        if len_a - index >= len_b:
            return 1
        return (index + len(b) - 1) // len(a) + 1
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(n + m)$，其中文本串 $a$ 的长度为 $n$，模式串 $b$ 的长度为 $m$。
- **空间复杂度**：$O(m)$。

# [0687. 最长同值路径](https://leetcode.cn/problems/longest-univalue-path/)

- 标签：树、深度优先搜索、二叉树
- 难度：中等

## 题目链接

- [0687. 最长同值路径 - 力扣](https://leetcode.cn/problems/longest-univalue-path/)

## 题目大意

**描述**：给定一个二叉树的根节点 $root$。

**要求**：返回二叉树中最长的路径的长度，该路径中每个节点具有相同值。 这条路径可以经过也可以不经过根节点。

**说明**：

- 树的节点数的范围是 $[0, 10^4]$。
- $-1000 \le Node.val \le 1000$。
- 树的深度将不超过 $1000$。
- 两个节点之间的路径长度：由它们之间的边数表示。

**示例**：

- 示例 1：

![](https://assets.leetcode.com/uploads/2020/10/13/ex1.jpg)

```python
输入：root = [5,4,5,1,1,5]
输出：2
```

- 示例 2：

![](https://assets.leetcode.com/uploads/2020/10/13/ex2.jpg)

```python
输入：root = [1,4,5,4,4,5]
输出：2
```

## 解题思路

### 思路 1：树形 DP + 深度优先搜索

这道题如果先不考虑「路径中每个节点具有相同值」这个条件，那么这道题就是在求「二叉树的直径长度（最长路径的长度）」。

「二叉树的直径长度」的定义为：二叉树中任意两个节点路径长度中的最大值。并且这条路径可能穿过也可能不穿过根节点。

对于根为 $root$ 的二叉树来说，其直径长度并不简单等于「左子树高度」加上「右子树高度」。

根据路径是否穿过根节点，我们可以将二叉树分为两种：

1. 直径长度所对应的路径穿过根节点，这种情况下：$\text{二叉树的直径} = \text{左子树高度} + \text{右子树高度}$。
2. 直径长度所对应的路径不穿过根节点，这种情况下：$\text{二叉树的直径} = \text{所有子树中最大直径长度}$。

也就是说根为 $root$ 的二叉树的直径长度可能来自于  $\text{左子树高度} + \text{右子树高度}$，也可能来自于 $\text{子树中的最大直径}$，即 $\text{二叉树的直径} = max(\text{左子树高度} + \text{右子树高度}, \quad \text{所有子树中最大直径长度})$。

那么现在问题就变成为如何求「子树的高度」和「子树中的最大直径」。

1. 子树的高度：我们可以利用深度优先搜索方法，递归遍历左右子树，并分别返回左右子树的高度。
2. 子树中的最大直径：我们可以在递归求解子树高度的时候维护一个 $ans$ 变量，用于记录所有 $\text{左子树高度} + \text{右子树高度}$ 中的最大值。

最终 $ans$ 就是我们所求的该二叉树的最大直径。

接下来我们再来加上「路径中每个节点具有相同值」这个限制条件。

1. 「左子树高度」应变为「左子树最长同值路径长度」。
2. 「右子树高度」应变为「右子树最长同值路径长度」。
3. 题目变为求「二叉树的最长同值路径长度」，式子为：$\text{二叉树的最长同值路径长度} = max(\text{左子树最长同值路径长度} + \text{右子树最长同值路径长度}, \quad \text{所有子树中最长同值路径长度})$。

在递归遍历的时候，我们还需要当前节点与左右子节点的值的相同情况，来维护更新「包含当前节点的最长同值路径长度」。

1. 在递归遍历左子树时，如果当前节点与左子树的值相同，则：$\text{包含当前节点向左的最长同值路径长度} = \text{左子树最长同值路径长度} + 1$，否则为 $0$。
2. 在递归遍历左子树时，如果当前节点与左子树的值相同，则：$\text{包含当前节点向右的最长同值路径长度} = \text{右子树最长同值路径长度} + 1$，否则为 $0$。

则：$\text{包含当前节点向左的最长同值路径长度} = max(\text{包含当前节点向左的最长同值路径长度}, \quad \text{包含当前节点向右的最长同值路径长度})$。

### 思路 1：代码

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def __init__(self):
        self.ans = 0

    def dfs(self, node):
        if not node:
            return 0

        left_len = self.dfs(node.left)          # 左子树高度
        right_len = self.dfs(node.right)        # 右子树高度
        if node.left and node.left.val == node.val:
            left_len += 1
        else:
            left_len = 0
        if node.right and node.right.val == node.val:
            right_len += 1
        else:
            right_len = 0
        self.ans = max(self.ans, left_len + right_len)
        return max(left_len, right_len)

    def longestUnivaluePath(self, root: Optional[TreeNode]) -> int:
        self.dfs(root)

        return self.ans
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(n)$，其中 $n$ 为二叉树的节点个数。
- **空间复杂度**：$O(n)$。
# [0688. 骑士在棋盘上的概率](https://leetcode.cn/problems/knight-probability-in-chessboard/)

- 标签：动态规划
- 难度：中等

## 题目链接

- [0688. 骑士在棋盘上的概率 - 力扣](https://leetcode.cn/problems/knight-probability-in-chessboard/)

## 题目大意

**描述**：在一个 `n * n` 的国际象棋棋盘上，一个骑士从单元格 `(row, column)` 开始，尝试进行 `k` 次 移动。行和列是从 `0` 开始的，左上角的单元格是 `(0, 0)`，右下角的单元格是 `(n - 1, n - 1)`。

象棋骑士有 `8` 种可能的走法，如下图所示。每次移动在基本方向上是两个单元格，然后在正交方向上是一个单元格。

![](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2018/10/12/knight.png)

每次骑士要移动时，它都会随机从 `8` 种可能的移动中选择一种（即使棋子会离开棋盘），然后移动到那里。骑士继续移动，直到它走了 `k` 步或离开了棋盘。

现在给定代表棋盘大小的整数 `n`、代表骑士移动次数的整数 `k`，以及代表骑士初始位置的坐标 `row` 和 `column`。

**要求**：返回骑士在棋盘停止移动后仍留在棋盘上的概率。

**说明**：

- $1 \le n \le 25$。
- $0 \le k \le 100$。
- $0 \le row, column \le n$。

**示例**：

- 示例 1：

```python
输入：n = 3, k = 2, row = 0, column = 0
输出：0.0625
解释：有两步(到(1,2)，(2,1))可以让骑士留在棋盘上。在每一个位置上，也有两种移动可以让骑士留在棋盘上。骑士留在棋盘上的总概率是 0.0625。
```

## 解题思路

### 思路 1：动态规划

###### 1. 划分阶段

按照骑士所在位置和所走步数进行阶段划分。

###### 2. 定义状态

定义状态 `dp[i][j][p]` 表示为：从位置 `(i, j)` 出发，移动不超过 `p` 步的情况下，最后仍留在棋盘内的概率。

###### 3. 状态转移方程

根据象棋骑士的 `8` 种可能的走法，`dp[i][j][p]` 的来源有八个方向（超出棋盘的无需再考虑）：

- 假设下一步的落点为 `(new_i, new_j)`。从当前步选择 `8` 个方向其中之一作为下一步方向的概率为 $\frac{1}{8}$。
- 而每个方向上落点仍在棋盘内的概率为 `dp[new_i][new_j][p - 1]`。所以从 `(i, j)` 走到 `(new_i, new_j)` 的可能性为 $dp[new_i][new_j] \times \frac{1}{8}$。

最终 $dp[i][j][p]$ 来源为 `8` 个方向上落点的概率之和，即：$dp[i][j][p] = \sum{ dp[new_i][new_j] \times \frac{1}{8} }$。

###### 4. 初始条件

- 从位置 `(i, j)` 出发，移动不超过 `0` 步的情况下，最后仍留在棋盘内的概率为 `1`。

###### 5. 最终结果

根据我们之前定义的状态，`dp[i][j][p]` 表示为：从位置 `(i, j)` 出发，移动不超过 `p` 步的情况下，最后仍留在棋盘内的概率。则最终结果为 `dp[row][column][k]`。

### 思路 1：动态规划代码

```python
class Solution:
    def knightProbability(self, n: int, k: int, row: int, column: int) -> float:
        dp = [[[0 for _ in range(k + 1)] for _ in range(n)] for _ in range(n)]
        for i in range(n):
            for j in range(n):
                dp[i][j][0] = 1

        directions = {(-1, -2), (-1, 2), (1, -2), (1, 2), (-2, -1), (-2, 1), (2, -1), (2, 1)}

        for p in range(1, k + 1):
            for i in range(n):
                for j in range(n):
                    for direction in directions:
                        new_i = i + direction[0]
                        new_j = j + direction[1]
                        if 0 <= new_i < n and 0 <= new_j < n:
                            dp[i][j][p] += dp[new_i][new_j][p - 1] / 8
        
        return dp[row][column][k]
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(n^2 * k)$。外三层循环的时间复杂度为 $O(n^2 * k)$，内层关于 `directions` 的循环每次执行 `8` 次，可以看做是常数级时间复杂度。
- **空间复杂度**：$O(n^2 * k)$。用到了三维数组保存状态。
# [0690. 员工的重要性](https://leetcode.cn/problems/employee-importance/)

- 标签：深度优先搜索、广度优先搜索、哈希表
- 难度：中等

## 题目链接

- [0690. 员工的重要性 - 力扣](https://leetcode.cn/problems/employee-importance/)

## 题目大意

给定一个公司的所有员工信息。其中每个员工信息包含：该员工 id，该员工重要度，以及该员工的所有下属 id。

再给定一个员工 id，要求返回该员工和他所有下属的重要度之和。

## 解题思路

利用哈希表，以「员工 id: 员工数据结构」的形式将员工信息存入哈希表中。然后深度优先搜索该员工以及下属员工。在搜索的同时，计算重要度之和，最终返回结果即可。

## 代码

```python
class Solution:
    def getImportance(self, employees: List['Employee'], id: int) -> int:
        employee_dict = dict()
        for employee in employees:
            employee_dict[employee.id] = employee

        def dfs(index: int) -> int:
            total = employee_dict[index].importance
            for sub_index in employee_dict[index].subordinates:
                total += dfs(sub_index)
            return total

        return dfs(id)
```

# [0691. 贴纸拼词](https://leetcode.cn/problems/stickers-to-spell-word/)

- 标签：位运算、数组、字符串、动态规划、回溯、状态压缩
- 难度：困难

## 题目链接

- [0691. 贴纸拼词 - 力扣](https://leetcode.cn/problems/stickers-to-spell-word/)

## 题目大意

**描述**：给定一个字符串数组 $stickers$ 表示不同的贴纸，其中 $stickers[i]$ 表示第 $i$ 张贴纸上的小写英文单词。再给定一个字符串 $target$。为了拼出给定字符串 $target$，我们需要从贴纸中切割单个字母并重新排列它们。贴纸的数量是无限的，可以重复多次使用。

**要求**：返回需要拼出 $target$ 的最小贴纸数量。如果任务不可能，则返回 $-1$。

**说明**：

- 在所有的测试用例中，所有的单词都是从 $1000$ 个最常见的美国英语单词中随机选择的，并且 $target$ 被选择为两个随机单词的连接。
- $n == stickers.length$。
- $1 \le n \le 50$。
- $1 \le stickers[i].length \le 10$。
- $1 \le target.length \le 15$。
- $stickers[i]$ 和 $target$ 由小写英文单词组成。

**示例**：

- 示例 1：

```python
输入：stickers = ["with","example","science"], target = "thehat"
输出：3
解释：
我们可以使用 2 个 "with" 贴纸，和 1 个 "example" 贴纸。
把贴纸上的字母剪下来并重新排列后，就可以形成目标 “thehat“ 了。
此外，这是形成目标字符串所需的最小贴纸数量。
```

- 示例 2：

```python
输入：stickers = ["notice","possible"], target = "basicbasic"
输出：-1
解释：我们不能通过剪切给定贴纸的字母来形成目标“basicbasic”。
```

## 解题思路

### 思路 1：状态压缩 DP + 广度优先搜索

根据题意，$target$ 的长度最大为 $15$，所以我们可以使用一个长度最多为 $15$ 位的二进制数 $state$ 来表示 $target$ 的某个子序列，如果 $state$ 第 $i$ 位二进制值为 $1$，则说明 $target$ 的第 $i$ 个字母被选中。

然后我们从初始状态 $state = 0$（没有选中 $target$ 中的任何字母）开始进行广度优先搜索遍历。

在广度优先搜索过程中，对于当前状态 $cur\underline{\hspace{0.5em}}state$，我们遍历所有贴纸的所有字母，如果当前字母可以拼到 $target$ 中的某个位置上，则更新状态 $next\underline{\hspace{0.5em}}state$ 为「选中 $target$ 中对应位置上的字母」。

为了得到最小最小贴纸数量，我们可以使用动态规划的方法，定义 $dp[state]$ 表示为到达 $state$ 状态需要的最小贴纸数量。

那么在广度优先搜索中，在更新状态时，同时进行状态转移，即 $dp[next\underline{\hspace{0.5em}}state] = dp[cur\underline{\hspace{0.5em}}state] + 1$。

> 注意：在进行状态转移时，要跳过 $dp[next\underline{\hspace{0.5em}}state]$ 已经有值的情况。

这样在到达状态 $1 \text{ <}\text{< } len(target) - 1$ 时，所得到的 $dp[1 \text{ <}\text{< } len(target) - 1]$ 即为答案。

如果最终到达不了 $dp[1 \text{ <}\text{< } len(target) - 1]$，则说明无法完成任务，返回 $-1$。

### 思路 1：代码

```python
class Solution:
    def minStickers(self, stickers: List[str], target: str) -> int:
        size = len(target)
        states = 1 << size
        dp = [0 for _ in range(states)]

        queue = collections.deque([0])

        while queue:
            cur_state = queue.popleft()
            for sticker in stickers:
                next_state = cur_state
                cnts = [0 for _ in range(26)]
                for ch in sticker:
                    cnts[ord(ch) - ord('a')] += 1
                for i in range(size):
                    if cnts[ord(target[i]) - ord('a')] and next_state & (1 << i) == 0:
                        next_state |= (1 << i)
                        cnts[ord(target[i]) - ord('a')] -= 1
                
                if dp[next_state] or next_state == 0:
                    continue
                
                queue.append(next_state)
                dp[next_state] = dp[cur_state] + 1
                if next_state == states - 1:
                    return dp[next_state]
        return -1
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(2^n \times \sum_{i = 0}^{m - 1} len(stickers[i]) \times n$，其中 $n$ 为 $target$ 的长度，$m$ 为 $stickers$ 的元素个数。
- **空间复杂度**：$O(2^n)$。

# [0695. 岛屿的最大面积](https://leetcode.cn/problems/max-area-of-island/)

- 标签：深度优先搜索、广度优先搜索、并查集、数组、矩阵
- 难度：中等

## 题目链接

- [0695. 岛屿的最大面积 - 力扣](https://leetcode.cn/problems/max-area-of-island/)

## 题目大意

**描述**：给定一个只包含 $0$、$1$ 元素的二维数组，$1$ 代表岛屿，$0$ 代表水。一座岛的面积就是上下左右相邻的 $1$ 所组成的连通块的数目。

**要求**：计算出最大的岛屿面积。

**说明**：

- $m == grid.length$。
- $n == grid[i].length$。
- $1 \le m, n \le 50$。
- $grid[i][j]$ 为 $0$ 或 $1$。

**示例**：

- 示例 1：

![](https://assets.leetcode.com/uploads/2021/05/01/maxarea1-grid.jpg)

```python
输入：grid = [[0,0,1,0,0,0,0,1,0,0,0,0,0],[0,0,0,0,0,0,0,1,1,1,0,0,0],[0,1,1,0,1,0,0,0,0,0,0,0,0],[0,1,0,0,1,1,0,0,1,0,1,0,0],[0,1,0,0,1,1,0,0,1,1,1,0,0],[0,0,0,0,0,0,0,0,0,0,1,0,0],[0,0,0,0,0,0,0,1,1,1,0,0,0],[0,0,0,0,0,0,0,1,1,0,0,0,0]]
输出：6
解释：答案不应该是 11 ，因为岛屿只能包含水平或垂直这四个方向上的 1 。
```

- 示例 2：

```python
输入：grid = [[0,0,0,0,0,0,0,0]]
输出：0
```

## 解题思路

### 思路 1：深度优先搜索

1. 遍历二维数组的每一个元素，对于每个值为 $1$ 的元素：
   1. 将该位置上的值置为 $0$（防止二次重复计算）。
   2. 递归搜索该位置上下左右四个位置，并统计搜到值为 $1$ 的元素个数。
   3. 返回值为 $1$ 的元素个数（即为该岛的面积）。
2. 维护并更新最大的岛面积。
3. 返回最大的到面积。

### 思路 1：代码

```python
class Solution:
    def dfs(self, grid, i, j):
        n = len(grid)
        m = len(grid[0])
        if i < 0 or i >= n or j < 0 or j >= m or grid[i][j] == 0:
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

### 思路 1：复杂度分析

- **时间复杂度**：$O(n \times m)$，其中 $m$ 和 $n$ 分别为行数和列数。
- **空间复杂度**：$O(n \times m)$。

### 思路 2：广度优先搜索

1. 使用 $ans$ 记录最大岛屿面积。
2. 遍历二维数组的每一个元素，对于每个值为 $1$ 的元素：
   1. 将该元素置为 $0$。并使用队列  $queue$ 存储该节点位置。使用 $temp\underline{\hspace{0.5em}}ans$ 记录当前岛屿面积。
   2. 然后从队列 $queue$ 中取出第一个节点位置 $(i, j)$。遍历该节点位置上、下、左、右四个方向上的相邻节点。并将其置为 $0$（避免重复搜索）。并将其加入到队列中。并累加当前岛屿面积，即 `temp_ans += 1`。
   3. 不断重复上一步骤，直到队列 $queue$ 为空。
   4. 更新当前最大岛屿面积，即 `ans = max(ans, temp_ans)`。
3. 将 $ans$ 作为答案返回。

### 思路 2：代码

```python
import collections

class Solution:
    def maxAreaOfIsland(self, grid: List[List[int]]) -> int:
        directs = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        rows, cols = len(grid), len(grid[0])
        ans = 0
        for i in range(rows):
            for j in range(cols):
                if grid[i][j] == 1:
                    grid[i][j] = 0
                    temp_ans = 1
                    q = collections.deque([(i, j)])
                    while q:
                        i, j = q.popleft()
                        for direct in directs:
                            new_i = i + direct[0]
                            new_j = j + direct[1]
                            if new_i < 0 or new_i >= rows or new_j < 0 or new_j >= cols or grid[new_i][new_j] == 0:
                                continue
                            grid[new_i][new_j] = 0
                            q.append((new_i, new_j))
                            temp_ans += 1

                    ans = max(ans, temp_ans)
        return ans
```

### 思路 2：复杂度分析

- **时间复杂度**：$O(n \times m)$，其中 $m$ 和 $n$ 分别为行数和列数。
- **空间复杂度**：$O(n \times m)$。
# [0698. 划分为k个相等的子集](https://leetcode.cn/problems/partition-to-k-equal-sum-subsets/)

- 标签：位运算、记忆化搜索、数组、动态规划、回溯、状态压缩
- 难度：中等

## 题目链接

- [0698. 划分为k个相等的子集 - 力扣](https://leetcode.cn/problems/partition-to-k-equal-sum-subsets/)

## 题目大意

**描述**：给定一个整数数组 $nums$ 和一个正整数 $k$。

**要求**：找出是否有可能把这个数组分成 $k$ 个非空子集，其总和都相等。

**说明**：

- $1 \le k \le len(nums) \le 16$。
- $0 < nums[i] < 10000$。
- 每个元素的频率在 $[1, 4]$ 范围内。

**示例**：

- 示例 1：

```python
输入： nums = [4, 3, 2, 3, 5, 2, 1], k = 4
输出： True
说明： 有可能将其分成 4 个子集（5），（1,4），（2,3），（2,3）等于总和。
```

- 示例 2：

```python
输入: nums = [1,2,3,4], k = 3
输出: False
```

## 解题思路

### 思路 1：状态压缩 DP

根据题目要求，我们可以将几种明显不符合要求的情况过滤掉，比如：元素个数小于 $k$、元素总和不是 $k$ 的倍数、数组 $nums$ 中最大元素超过 $k$ 等分的目标和这几种情况。

然后再来考虑一般情况下，如何判断是否符合要求。

因为题目给定数组 $nums$ 的长度最多为 $16$，所以我们可以使用一个长度为 $16$ 位的二进制数来表示数组子集的选择状态。我们可以定义 $dp[state]$ 表示为当前选择状态下，是否可行。如果 $dp[state] == True$，表示可行；如果 $dp[state] == False$，则表示不可行。

接下来使用动态规划方法，进行求解。具体步骤如下：

###### 1. 划分阶段

按照数组元素选择情况进行阶段划分。

###### 2. 定义状态

定义状态 $dp[state]$ 表示为：当数组元素选择情况为 $state$ 时，是否存在一种方案，使得方案中的数字必定能分割成 $p(0 \le p \le k)$ 组恰好数字和等于目标和 $target$ 的集合和至多 $1$ 组数字和小于目标和 $target$ 的集合。

###### 3. 状态转移方程

对于当前状态 $state$，如果：

1. 当数组元素选择情况为 $state$ 时可行，即 $dp[state] == True$；
2. 第 $i$ 位数字没有被使用；
3. 加上第 $i$ 位元素后的状态为 $next\underline{\hspace{0.5em}}state$；
4. 加上第 $i$ 位元素后没有超出目标和。

则：$dp[next\underline{\hspace{0.5em}}state] = True$。

###### 4. 初始条件

- 当不选择任何元素时，可按照题目要求

###### 5. 最终结果

根据我们之前定义的状态，$dp[state]$ 表示为：当数组元素选择情况为 $state$ 时，是否存在一种方案，使得方案中的数字必定能分割成 $p(0 \le p \le k)$ 组恰好数字和等于目标和 $target$ 的集合和至多 $1$ 组数字和小于目标和 $target$ 的集合。

所以当 $state == 1 << n - 1$ 时，状态就变为了：当数组元素都选上的情况下，是否存在一种方案，使得方案中的数字必定能分割成 $k$ 组恰好数字和等于目标和 $target$ 的集合。

这里之所以是 $k$ 组恰好数字和等于目标和 $target$ 的集合，是因为一开我们就限定了 $total \mod k == 0$ 这个条件，所以只能是 $k$ 组恰好数字和等于目标和 $target$ 的集合。

所以最终结果为 $dp[states - 1]$，其中 $states = 1 << n$。

### 思路 1：代码

```python
class Solution:
    def canPartitionKSubsets(self, nums: List[int], k: int) -> bool:
        size = len(nums)
        if size < k:            # 元素个数小于 k
            return False

        total = sum(nums)
        if total % k != 0:      # 元素总和不是 k 的倍数
            return False

        target = total // k
        if nums[-1] > target:   # 最大元素超过 k 等分的目标和
            return False

        nums.sort()
        states = 1 << size      # 子集选择状态总数
        cur_sum = [0 for _ in range(states)]
        dp = [False for _ in range(states)]
        dp[0] = True

        for state in range(states):
            if not dp[state]:                   # 基于 dp[state] == True 前提下进行转移        
                continue
            for i in range(size):
                if state & (1 << i) != 0:       # 当前数字已被使用
                    continue
                
                if cur_sum[state] % target + nums[i] > target:
                    break                       # 如果加入当前数字超出目标和，则后续不用继续遍历

                next_state = state | (1 << i)   # 加入当前数字
                if dp[next_state]:              # 如果新状态能划分，则跳过继续
                    continue
                
                cur_sum[next_state] = cur_sum[state] + nums[i]  # 更新新状态下子集和
                dp[next_state] = True           # 更新新状态
                if dp[states - 1]:              # 找到一个符合要求的划分方案，提前返回
                    return True
                
        return dp[states - 1]
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(n \times 2^n)$，其中 $n$ 为数组 $nums$ 的长度。
- **空间复杂度**：$O(2^n)$。

## 参考资料

- 【题解】[状态压缩的定义理解 - 划分为k个相等的子集](https://leetcode.cn/problems/partition-to-k-equal-sum-subsets/solution/zhuang-tai-ya-suo-de-ding-yi-li-jie-by-c-fo1b/)
# [0700. 二叉搜索树中的搜索](https://leetcode.cn/problems/search-in-a-binary-search-tree/)

- 标签：树、二叉搜索树、二叉树
- 难度：简单

## 题目链接

- [0700. 二叉搜索树中的搜索 - 力扣](https://leetcode.cn/problems/search-in-a-binary-search-tree/)

## 题目大意

**描述**：给定一个二叉搜索树和一个值 `val`。

**要求**：在二叉搜索树中查找节点值等于 `val` 的节点，并返回该节点。

**说明**：

- 数中节点数在 $[1, 5000]$ 范围内。
- $1 \le Node.val \le 10^7$。
- `root` 是二叉搜索树。
- $1 \le val \le 10^7$。

**示例**：

- 示例 1：

![img](https://assets.leetcode.com/uploads/2021/01/12/tree1.jpg)

```python
输入：root = [4,2,7,1,3], val = 2
输出：[2,1,3]
```

- 示例 2：

![](https://assets.leetcode.com/uploads/2021/01/12/tree2.jpg)

```python
输入：root = [4,2,7,1,3], val = 5
输出：[]
```

## 解题思路

### 思路 1：递归

1. 从根节点 `root` 开始向下递归遍历。
   1. 如果 `val` 等于当前节点的值，即 `val == root.val`，则返回 `root`；
   2. 如果 `val` 小于当前节点的值 ，即 `val < root.val`，则递归遍历左子树，继续查找；
   3. 如果 `val` 大于当前节点的值 ，即 `val > root.val`，则递归遍历右子树，继续查找。
2. 如果遍历到最后也没有找到，则返回空节点。

### 思路 1：代码

```python
class Solution:
    def searchBST(self, root: TreeNode, val: int) -> TreeNode:
        if not root or val == root.val:
            return root
        if val < root.val:
            return self.searchBST(root.left, val)
        else:
            return self.searchBST(root.right, val)
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(n)$。其中 $n$ 是二叉搜索树的节点数。
- **空间复杂度**：$O(n)$。# [0701. 二叉搜索树中的插入操作](https://leetcode.cn/problems/insert-into-a-binary-search-tree/)

- 标签：树、二叉搜索树、二叉树
- 难度：中等

## 题目链接

- [0701. 二叉搜索树中的插入操作 - 力扣](https://leetcode.cn/problems/insert-into-a-binary-search-tree/)

## 题目大意

**描述**：给定一个二叉搜索树的根节点和要插入树中的值 `val`。

**要求**：将 `val` 插入到二叉搜索树中，返回新的二叉搜索树的根节点。

**说明**：

- 树中的节点数将在 $[0, 10^4]$ 的范围内。
- $-10^8 \le Node.val \le 10^8$
- 所有值 `Node.val` 是独一无二的。
- $-10^8 \le val \le 10^8$。
- **保证** $val$ 在原始 BST 中不存在。

**示例**：

- 示例 1：

```python
输入：root = [4,2,7,1,3], val = 5
输出：[4,2,7,1,3,5]
解释：另一个满足题目要求可以通过的树是：
```

- 示例 2：

```python
输入：root = [40,20,60,10,30,50,70], val = 25
输出：[40,20,60,10,30,50,70,null,null,25]
```

## 解题思路

### 思路 1：递归

已知搜索二叉树的性质：

- 左子树上任意节点值均小于根节点，即 `root.left.val < root.val`。
- 右子树上任意节点值均大于根节点，即 `root.left.val > root.val`。

那么根据 `val` 和当前节点的大小关系，则可以确定将 `val` 插入到当前节点的哪个子树上。具体步骤如下：

1. 从根节点 `root` 开始向下递归遍历。根据 `val` 值和当前子树节点 `cur` 的大小关系：
   1. 如果 `val < cur.val`，则应在当前节点的左子树继续遍历判断。
      1. 如果左子树为空，则新建节点，赋值为 `val`。链接到该子树的父节点上。并停止遍历。
      2. 如果左子树不为空，则继续向左子树移动。
   2. 如果 `val >= cur.val`，则应在当前节点的右子树继续遍历判断。
      1. 如果右子树为空，则新建节点，赋值为 `val`。链接到该子树的父节点上。并停止遍历。
      2. 如果右子树不为空，则继续向左子树移动。
2. 遍历完返回根节点 `root`。

### 思路 1：代码

```python
class Solution:
    def insertIntoBST(self, root: TreeNode, val: int) -> TreeNode:
        if not root:
            return TreeNode(val)

        cur = root
        while cur:
            if val < cur.val:
                if not cur.left:
                    cur.left = TreeNode(val)
                    break
                else:
                    cur = cur.left
            else:
                if not cur.right:
                    cur.right = TreeNode(val)
                    break
                else:
                    cur = cur.right
        return root
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(n)$。其中 $n$ 是二叉搜索树的节点数。
- **空间复杂度**：$O(n)$。# [0702. 搜索长度未知的有序数组](https://leetcode.cn/problems/search-in-a-sorted-array-of-unknown-size/)

- 标签：数组、二分查找、交互
- 难度：中等

## 题目链接

- [0702. 搜索长度未知的有序数组 - 力扣](https://leetcode.cn/problems/search-in-a-sorted-array-of-unknown-size/)

## 题目大意

**描述**：给定一个升序数组 $secret$，但是数组的大小是未知的。我们无法直接访问数组，智能通过 `ArrayReader` 接口去访问他。我们可以通过接口 `reader.get(k)`：

1. 如果数组访问未越界，则返回数组 $secret$ 中第 $k$ 个下标位置的元素值。
2. 如果数组访问越界，则接口返回 $2^{31} - 1$。

现在再给定一个数字 $target$。

**要求**：从 $secret$ 中找出 $secret[k] == target$ 的下标位置 $k$，如果 $secret$ 中不存在 $target$，则返回 $-1$。

**说明**：

- $1 \le secret.length \le 10^4$。
- $-10^4 \le secret[i], target \le 10^4$。
- $secret$ 严格递增。

**示例**：

- 示例 1：

```python
输入: secret = [-1,0,3,5,9,12], target = 9
输出: 4
解释: 9 存在在 nums 中，下标为 4
```

- 示例 2：

```python
输入: secret = [-1,0,3,5,9,12], target = 2
输出: -1
解释: 2 不在数组中所以返回 -1
```

## 解题思路

### 思路 1：二分查找算法

这道题的关键点在于找到数组的大小，以便确定查找的右边界位置。右边界可以通过倍增的方式快速查找。在查找右边界的同时，也能将左边界的范围进一步缩小。等确定了左右边界，就可以使用二分查找算法快速查找 $target$。

### 思路 1：代码

```python
class Solution:
    def binarySearch(self, reader, left, right, target):
        while left < right:
            mid = left + (right - left) // 2
            if target > reader.get(mid):
                left = mid + 1
            else:
                right = mid
        if reader.get(left) == target:
            return left
        else:
            return -1

    def search(self, reader, target):
        left = 0
        right = 1
        while reader.get(right) < target:
            left = right
            right <<= 1

        return self.binarySearch(reader, left, right, target)
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(\log n)$，其中 $n$ 为数组长度。
- **空间复杂度**：$O(1)$。

# [0703. 数据流中的第 K 大元素](https://leetcode.cn/problems/kth-largest-element-in-a-stream/)

- 标签：树、设计、二叉搜索树、二叉树、数据流、堆（优先队列）
- 难度：简单

## 题目链接

- [0703. 数据流中的第 K 大元素 - 力扣](https://leetcode.cn/problems/kth-largest-element-in-a-stream/)

## 题目大意

**要求**：设计一个 KthLargest 类，用于找到数据流中第 $k$ 大元素。

实现 KthLargest 类：

- `KthLargest(int k, int[] nums)`：使用整数 $k$ 和整数流 $nums$ 初始化对象。
- `int add(int val)`：将 $val$ 插入数据流 $nums$ 后，返回当前数据流中第 $k$ 大的元素。

**说明**：

- $1 \le k \le 10^4$。
- $0 \le nums.length \le 10^4$。
- $-10^4 \le nums[i] \le 10^4$。
- $-10^4 \le val \le 10^4$。
- 最多调用 `add` 方法 $10^4$ 次。
- 题目数据保证，在查找第 $k$ 大元素时，数组中至少有 $k$ 个元素。

**示例**：

- 示例 1：

```python
输入：
["KthLargest", "add", "add", "add", "add", "add"]
[[3, [4, 5, 8, 2]], [3], [5], [10], [9], [4]]
输出：
[null, 4, 5, 5, 8, 8]

解释：
KthLargest kthLargest = new KthLargest(3, [4, 5, 8, 2]);
kthLargest.add(3);   // return 4
kthLargest.add(5);   // return 5
kthLargest.add(10);  // return 5
kthLargest.add(9);   // return 8
kthLargest.add(4);   // return 8
```

## 解题思路

### 思路 1：堆

1. 建立大小为 $k$ 的大顶堆，堆中元素保证不超过 $k$ 个。
2. 每次 `add` 操作时，将新元素压入堆中，如果堆中元素超出了 $k$ 个，则将堆中最小元素（堆顶）移除。

- 此时堆中最小元素（堆顶）就是整个数据流中的第 $k$ 大元素。

### 思路 1：代码

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

### 思路 1：复杂度分析

- **时间复杂度**：
  - 初始化时间复杂度：$O(n \times \log k)$，其中 $n$ 为 $nums$ 初始化时的元素个数。
  - 单次插入时间复杂度：$O(\log k)$。
- **空间复杂度**：$O(k)$。

# [0704. 二分查找](https://leetcode.cn/problems/binary-search/)

- 标签：数组、二分查找
- 难度：简单

## 题目链接

- [0704. 二分查找 - 力扣](https://leetcode.cn/problems/binary-search/)

## 题目大意

**描述**：给定一个升序的数组 $nums$，和一个目标值 $target$。

**要求**：返回 $target$ 在数组中的位置，如果找不到，则返回 -1。

**说明**：

- 你可以假设 $nums$ 中的所有元素是不重复的。
- $n$ 将在 $[1, 10000]$之间。
- $nums$ 的每个元素都将在 $[-9999, 9999]$之间。

**示例**：

- 示例 1：

```python
输入: nums = [-1,0,3,5,9,12], target = 9
输出: 4
解释: 9 出现在 nums 中并且下标为 4
```

- 示例 2：

```python
输入: nums = [-1,0,3,5,9,12], target = 2
输出: -1
解释: 2 不存在 nums 中因此返回 -1
```

## 解题思路

### 思路 1：二分查找

设定左右节点为数组两端，即 `left = 0`，`right = len(nums) - 1`，代表待查找区间为 $[left, right]$（左闭右闭）。

取两个节点中心位置 $mid$，先比较中心位置值 $nums[mid]$ 与目标值 $target$ 的大小。

- 如果 $target == nums[mid]$，则返回中心位置。
- 如果 $target > nums[mid]$，则将左节点设置为 $mid + 1$，然后继续在右区间 $[mid + 1, right]$ 搜索。
- 如果中心位置值 $target < nums[mid]$，则将右节点设置为 $mid - 1$，然后继续在左区间 $[left, mid - 1]$ 搜索。

### 思路 1：代码

```python
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        left, right = 0, len(nums) - 1
        
        # 在区间 [left, right] 内查找 target
        while left <= right:
            # 取区间中间节点
            mid = (left + right) // 2
            # 如果找到目标值，则直接返回中心位置
            if nums[mid] == target:
                return mid
            # 如果 nums[mid] 小于目标值，则在 [mid + 1, right] 中继续搜索
            elif nums[mid] < target:
                left = mid + 1
            # 如果 nums[mid] 大于目标值，则在 [left, mid - 1] 中继续搜索
            else:
                right = mid - 1
        # 未搜索到元素，返回 -1
        return -1
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(\log n)$。
- **空间复杂度**：$O(1)$。

# [0705. 设计哈希集合](https://leetcode.cn/problems/design-hashset/)

- 标签：设计、数组、哈希表、链表、哈希函数
- 难度：简单

## 题目链接

- [0705. 设计哈希集合 - 力扣](https://leetcode.cn/problems/design-hashset/)

## 题目大意

**要求**：不使用内建的哈希表库，自行实现一个哈希集合（HashSet）。

需要满足以下操作：

- `void add(key)` 向哈希集合中插入值 $key$。
- `bool contains(key)` 返回哈希集合中是否存在这个值 $key$。
- `void remove(key)` 将给定值 $key$ 从哈希集合中删除。如果哈希集合中没有这个值，什么也不做。

**说明**：

- $0 \le key \le 10^6$。
- 最多调用 $10^4$ 次 `add`、`remove` 和 `contains`。

**示例**：

- 示例 1：

```python
输入：
["MyHashSet", "add", "add", "contains", "contains", "add", "contains", "remove", "contains"]
[[], [1], [2], [1], [3], [2], [2], [2], [2]]
输出：
[null, null, null, true, false, null, true, null, false]

解释：
MyHashSet myHashSet = new MyHashSet();
myHashSet.add(1);      // set = [1]
myHashSet.add(2);      // set = [1, 2]
myHashSet.contains(1); // 返回 True
myHashSet.contains(3); // 返回 False ，（未找到）
myHashSet.add(2);      // set = [1, 2]
myHashSet.contains(2); // 返回 True
myHashSet.remove(2);   // set = [1]
myHashSet.contains(2); // 返回 False ，（已移除）
```

## 解题思路

### 思路 1：数组 + 链表

定义一个一维长度为 $buckets$ 的二维数组 $table$。

第一维度用于计算哈希函数，为 $key$ 进行分桶。第二个维度用于寻找 $key$ 存放的具体位置。第二维度的数组会根据 $key$ 值动态增长，模拟真正的链表。

### 思路 1：代码

```python
class MyHashSet:

    def __init__(self):
        self.buckets = 1003
        self.table = [[] for _ in range(self.buckets)]

        
    def hash(self, key):
        return key % self.buckets

    
    def add(self, key: int) -> None:
        hash_key = self.hash(key)
        if key in self.table[hash_key]:
            return
        self.table[hash_key].append(key)


    def remove(self, key: int) -> None:
        hash_key = self.hash(key)
        if key not in self.table[hash_key]:
            return
        self.table[hash_key].remove(key)


    def contains(self, key: int) -> bool:
        hash_key = self.hash(key)
        return key in self.table[hash_key]
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(\frac{n}{m})$，其中 $n$ 为哈希表中的元素数量，$b$ 为 $table$ 的元素个数，也就是链表的数量。
- **空间复杂度**：$O(n + m)$。

# [0706. 设计哈希映射](https://leetcode.cn/problems/design-hashmap/)

- 标签：设计、数组、哈希表、链表、哈希函数
- 难度：简单

## 题目链接

- [0706. 设计哈希映射 - 力扣](https://leetcode.cn/problems/design-hashmap/)

## 题目大意

**要求**：不使用任何内建的哈希表库设计一个哈希映射（`HashMap`）。

需要满足以下操作：

- `MyHashMap()` 用空映射初始化对象。
- `void put(int key, int value) 向 HashMap` 插入一个键值对 `(key, value)` 。如果 `key` 已经存在于映射中，则更新其对应的值 `value`。
- `int get(int key)` 返回特定的 `key` 所映射的 `value`；如果映射中不包含 `key` 的映射，返回 `-1`。
- `void remove(key)` 如果映射中存在 key 的映射，则移除 `key` 和它所对应的 `value` 。

**说明**：

- $0 \le key, value \le 10^6$。
- 最多调用 $10^4$ 次 `put`、`get` 和 `remove` 方法。

**示例**：

- 示例 1：

```python
输入：
["MyHashMap", "put", "put", "get", "get", "put", "get", "remove", "get"]
[[], [1, 1], [2, 2], [1], [3], [2, 1], [2], [2], [2]]
输出：
[null, null, null, 1, -1, null, 1, null, -1]

解释：
MyHashMap myHashMap = new MyHashMap();
myHashMap.put(1, 1); // myHashMap 现在为 [[1,1]]
myHashMap.put(2, 2); // myHashMap 现在为 [[1,1], [2,2]]
myHashMap.get(1);    // 返回 1 ，myHashMap 现在为 [[1,1], [2,2]]
myHashMap.get(3);    // 返回 -1（未找到），myHashMap 现在为 [[1,1], [2,2]]
myHashMap.put(2, 1); // myHashMap 现在为 [[1,1], [2,1]]（更新已有的值）
myHashMap.get(2);    // 返回 1 ，myHashMap 现在为 [[1,1], [2,1]]
myHashMap.remove(2); // 删除键为 2 的数据，myHashMap 现在为 [[1,1]]
myHashMap.get(2);    // 返回 -1（未找到），myHashMap 现在为 [[1,1]]
```

## 解题思路

### 思路 1：链地址法

和 [0705. 设计哈希集合](https://leetcode.cn/problems/design-hashset/) 类似。这里我们使用「链地址法」来解决哈希冲突。即利用「数组 + 链表」的方式实现哈希集合。

1. 定义哈希表长度 `buckets` 为 `1003`。
2. 定义一个一维长度为 `buckets` 的二维数组 `table`。其中第一维度用于计算哈希函数，为关键字 `key` 分桶。第二个维度用于存放 `key` 和对应的 `value`。第二维度的数组会根据 `key` 值动态增长，用数组模拟真正的链表。
3. 定义一个 `hash(key)` 的方法，将 `key` 转换为对应的地址 `hash_key`。
4. 进行 `put` 操作时，根据 `hash(key)` 方法，获取对应的地址 `hash_key`。然后遍历 `hash_key` 对应的数组元素，查找与 `key` 值一样的元素。
   1. 如果找到与 `key` 值相同的元素，则更改该元素对应的 `value` 值。
   2. 如果没找到与 `key` 值相同的元素，则在第二维数组 `table[hask_key]` 中增加元素，元素为 `(key, value)` 组成的元组。

5. 进行 `get` 操作跟 `put` 操作差不多。根据 `hash(key)` 方法，获取对应的地址 `hash_key`。然后遍历 `hash_key` 对应的数组元素，查找与 `key` 值一样的元素。
   1. 如果找到与 `key` 值相同的元素，则返回该元素对应的 `value`。
   2. 如果没找到与 `key` 值相同的元素，则返回 `-1`。

### 思路 1：代码

```python
class MyHashMap:

    def __init__(self):
        self.buckets = 1003
        self.table = [[] for _ in range(self.buckets)]


    def hash(self, key):
        return key % self.buckets


    def put(self, key: int, value: int) -> None:
        hash_key = self.hash(key)
        for item in self.table[hash_key]:
            if key == item[0]:
                item[1] = value
                return
        self.table[hash_key].append([key, value])


    def get(self, key: int) -> int:
        hash_key = self.hash(key)
        for item in self.table[hash_key]:
            if key == item[0]:
                return item[1]
        return -1


    def remove(self, key: int) -> None:
        hash_key = self.hash(key)
        for i, item in enumerate(self.table[hash_key]):
            if key == item[0]:
                self.table[hash_key].pop(i)
                return
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(\frac{n}{b})$。其中 $n$ 为哈希表中元素数量，$b$ 为链表的数量。
- **空间复杂度**：$O(n + b)$。# [0707. 设计链表](https://leetcode.cn/problems/design-linked-list/)

- 标签：设计、链表
- 难度：中等

## 题目链接

- [0707. 设计链表 - 力扣](https://leetcode.cn/problems/design-linked-list/)

## 题目大意

**要求**：设计实现一个链表，需要支持以下操作：

- `get(index)`：获取链表中第 `index` 个节点的值。如果索引无效，则返回 `-1`。
- `addAtHead(val)`：在链表的第一个元素之前添加一个值为 `val` 的节点。插入后，新节点将成为链表的第一个节点。
- `addAtTail(val)`：将值为 `val` 的节点追加到链表的最后一个元素。
- `addAtIndex(index, val)`：在链表中的第 `index` 个节点之前添加值为 `val`  的节点。如果 `index` 等于链表的长度，则该节点将附加到链表的末尾。如果 `index` 大于链表长度，则不会插入节点。如果 `index` 小于 `0`，则在头部插入节点。
- `deleteAtIndex(index)`：如果索引 `index` 有效，则删除链表中的第 `index` 个节点。

**说明**：

- 所有`val`值都在 $[1, 1000]$ 之内。
- 操作次数将在 $[1, 1000]$ 之内。
- 请不要使用内置的 `LinkedList` 库。

**示例**：

- 示例 1：

```python
MyLinkedList linkedList = new MyLinkedList();
linkedList.addAtHead(1);
linkedList.addAtTail(3);
linkedList.addAtIndex(1,2);   // 链表变为 1 -> 2 -> 3
linkedList.get(1);            // 返回 2
linkedList.deleteAtIndex(1);  // 现在链表是 1-> 3
linkedList.get(1);            // 返回 3
```

## 解题思路

### 思路 1：单链表

新建一个带有 `val` 值 和 `next` 指针的链表节点类， 然后按照要求对节点进行操作。

### 思路 1：代码

```python
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None


class MyLinkedList:
    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.size = 0
        self.head = ListNode(0)


    def get(self, index: int) -> int:
        """
        Get the value of the index-th node in the linked list. If the index is invalid, return -1.
        """
        if index < 0 or index >= self.size:
            return -1

        curr = self.head
        for _ in range(index + 1):
            curr = curr.next
        return curr.val


    def addAtHead(self, val: int) -> None:
        """
        Add a node of value val before the first element of the linked list. After the insertion, the new node will be the first node of the linked list.
        """
        self.addAtIndex(0, val)


    def addAtTail(self, val: int) -> None:
        """
        Append a node of value val to the last element of the linked list.
        """
        self.addAtIndex(self.size, val)


    def addAtIndex(self, index: int, val: int) -> None:
        """
        Add a node of value val before the index-th node in the linked list. If index equals to the length of linked list, the node will be appended to the end of linked list. If index is greater than the length, the node will not be inserted.
        """
        if index > self.size:
            return

        if index < 0:
            index = 0

        self.size += 1
        pre = self.head
        for _ in range(index):
            pre = pre.next

        add_node = ListNode(val)
        add_node.next = pre.next
        pre.next = add_node


    def deleteAtIndex(self, index: int) -> None:
        """
        Delete the index-th node in the linked list, if the index is valid.
        """
        if index < 0 or index >= self.size:
            return

        self.size -= 1
        pre = self.head
        for _ in range(index):
            pre = pre.next

        pre.next = pre.next.next
```

### 思路 1：复杂度分析

- **时间复杂度**：
  - `addAtHead(val)`：$O(1)$。
  - `get(index)`、`addAtTail(val)`、`del eteAtIndex(index)`：$O(k)$。$k$ 指的是元素的索引。
  - `addAtIndex(index, val)`：$O(n)$。$n$ 指的是链表的元素个数。

- **空间复杂度**：$O(1)$。

### 思路 2：双链表

新建一个带有 `val` 值和 `next` 指针、`prev` 指针的链表节点类，然后按照要求对节点进行操作。

### 思路 2：代码

```python
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None
        self.prev = None

class MyLinkedList:
    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.size = 0
        self.head = ListNode(0)
        self.tail = ListNode(0)
        self.head.next = self.tail
        self.tail.prev = self.head


    def get(self, index: int) -> int:
        """
        Get the value of the index-th node in the linked list. If the index is invalid, return -1.
        """
        if index < 0 or index >= self.size:
            return -1

        if index + 1 < self.size - index:
            curr = self.head
            for _ in range(index + 1):
                curr = curr.next
        else:
            curr = self.tail
            for _ in range(self.size - index):
                curr = curr.prev
        return curr.val


    def addAtHead(self, val: int) -> None:
        """
        Add a node of value val before the first element of the linked list. After the insertion, the new node will be the first node of the linked list.
        """
        self.addAtIndex(0, val)


    def addAtTail(self, val: int) -> None:
        """
        Append a node of value val to the last element of the linked list.
        """
        self.addAtIndex(self.size, val)


    def addAtIndex(self, index: int, val: int) -> None:
        """
        Add a node of value val before the index-th node in the linked list. If index equals to the length of linked list, the node will be appended to the end of linked list. If index is greater than the length, the node will not be inserted.
        """
        if index > self.size:
            return

        if index < 0:
            index = 0

        if index < self.size - index:
            prev = self.head
            for _ in range(index):
                prev = prev.next
            next = prev.next
        else:
            next = self.tail
            for _ in range(self.size - index):
                next = next.prev
            prev = next.prev

        self.size += 1
        add_node = ListNode(val)
        add_node.prev = prev
        add_node.next = next
        prev.next = add_node
        next.prev = add_node

    def deleteAtIndex(self, index: int) -> None:
        """
        Delete the index-th node in the linked list, if the index is valid.
        """
        if index < 0 or index >= self.size:
            return

        if index < self.size - index:
            prev = self.head
            for _ in range(index):
                prev = prev.next
            next = prev.next.next
        else:
            next = self.tail
            for _ in range(self.size - index - 1):
                next = next.prev
            prev = next.prev.prev

        self.size -= 1
        prev.next = next
        next.prev = prev
```

### 思路 2：复杂度分析

- **时间复杂度**：
  - `addAtHead(val)`、`addAtTail(val)`：$O(1)$。
  - `get(index)`、`addAtIndex(index, val)`、`del eteAtIndex(index)`：$O(min(k, n - k))$。$n$ 指的是链表的元素个数，$k$ 指的是元素的索引。
- **空间复杂度**：$O(1)$。

# [0708. 循环有序列表的插入](https://leetcode.cn/problems/insert-into-a-sorted-circular-linked-list/)

- 标签：链表
- 难度：中等

## 题目链接

- [0708. 循环有序列表的插入 - 力扣](https://leetcode.cn/problems/insert-into-a-sorted-circular-linked-list/)

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

# [0709. 转换成小写字母](https://leetcode.cn/problems/to-lower-case/)

- 标签：字符串
- 难度：简单

## 题目链接

- [0709. 转换成小写字母 - 力扣](https://leetcode.cn/problems/to-lower-case/)

## 题目大意

**描述**：给定一个字符串 $s$。

**要求**：将该字符串中的大写字母转换成相同的小写字母，返回新的字符串。

**说明**：

- $1 \le s.length \le 100$。
- $s$ 由 ASCII 字符集中的可打印字符组成。

**示例**：

- 示例 1：

```python
输入：s = "Hello"
输出："hello"
```

- 示例 2：

```python
输入：s = "LOVELY"
输出："lovely"
```

## 解题思路

### 思路 1：直接模拟

- 大写字母 $A \sim Z$ 的 ASCII 码范围为 $[65, 90]$。
- 小写字母 $a \sim z$ 的 ASCII 码范围为 $[97, 122]$。

将大写字母的 ASCII 码加 $32$，就得到了对应的小写字母，则解决步骤如下：

1. 使用一个字符串变量 $ans$ 存储最终答案字符串。
2. 遍历字符串 $s$，对于当前字符 $ch$：
   1. 如果 $ch$ 的 ASCII 码范围在 $[65, 90]$，则说明 $ch$ 为大写字母。将 $ch$ 的 ASCII 码增加 $32$，再转换为对应的字符，存入字符串 $ans$ 的末尾。
   2. 如果 $ch$ 的 ASCII 码范围不在 $[65, 90]$，则说明 $ch$ 为小写字母。直接将 $ch$ 存入字符串 $ans$ 的末尾。
3. 遍历完字符串 $s$，返回答案字符串 $ans$。

### 思路 1：代码

```python
class Solution:
    def toLowerCase(self, s: str) -> str:
        ans = ""
        for ch in s:
            if ord('A') <= ord(ch) <= ord('Z'):
                ans += chr(ord(ch) + 32)
            else:
                ans += ch
        return ans
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(n)$。一重循环遍历的时间复杂度为 $O(n)$。
- **空间复杂度**：$O(n)$。如果算上答案数组的空间占用，则空间复杂度为 $O(n)$。不算上则空间复杂度为 $O(1)$。

### 思路 2：使用 API

Python 语言中自带大写字母转小写字母的 API：`lower()`，用 API 转换完成之后，直接返回新的字符串。

### 思路 2：代码

```python
class Solution:
    def toLowerCase(self, s: str) -> str:
        return s.lower()
```

### 思路 2：复杂度分析

- **时间复杂度**：$O(n)$。一重循环遍历的时间复杂度为 $O(n)$。
- **空间复杂度**：$O(n)$。如果算上答案数组的空间占用，则空间复杂度为 $O(n)$。不算上则空间复杂度为 $O(1)$。# [0713. 乘积小于 K 的子数组](https://leetcode.cn/problems/subarray-product-less-than-k/)

- 标签：数组、滑动窗口
- 难度：中等

## 题目链接

- [0713. 乘积小于 K 的子数组 - 力扣](https://leetcode.cn/problems/subarray-product-less-than-k/)

## 题目大意

**描述**：给定一个正整数数组 $nums$ 和整数 $k$。

**要求**：找出该数组内乘积小于 $k$ 的连续的子数组的个数。

**说明**：

- $1 \le nums.length \le 3 * 10^4$。
- $1 \le nums[i] \le 1000$。
- $0 \le k \le 10^6$。

**示例**：

- 示例 1：

```python
输入：nums = [10,5,2,6], k = 100
输出：8
解释：8 个乘积小于 100 的子数组分别为：[10]、[5]、[2],、[6]、[10,5]、[5,2]、[2,6]、[5,2,6]。需要注意的是 [10,5,2] 并不是乘积小于 100 的子数组。
```

- 示例 2：

```python
输入：nums = [1,2,3], k = 0
输出：0
```

## 解题思路

### 思路 1：滑动窗口（不定长度）

1. 设定两个指针：$left$、$right$，分别指向滑动窗口的左右边界，保证窗口内所有数的乘积 $window\underline{\hspace{0.5em}}product$ 都小于 $k$。使用 $window\underline{\hspace{0.5em}}product$ 记录窗口中的乘积值，使用 $count$ 记录符合要求的子数组个数。
2. 一开始，$left$、$right$ 都指向 $0$。
3. 向右移动 $right$，将最右侧元素加入当前子数组乘积 $window\underline{\hspace{0.5em}}product$ 中。
4. 如果 $window\underline{\hspace{0.5em}}product \ge k$，则不断右移 $left$，缩小滑动窗口长度，并更新当前乘积值 $window\underline{\hspace{0.5em}}product$  直到 $window\underline{\hspace{0.5em}}product < k$。
5. 记录累积答案个数加 $1$，继续右移 $right$，直到 $right \ge len(nums)$ 结束。
6. 输出累积答案个数。

### 思路 1：代码

```python
class Solution:
    def numSubarrayProductLessThanK(self, nums: List[int], k: int) -> int:
        if k <= 1:
            return 0

        size = len(nums)
        left = 0
        right = 0
        window_product = 1
        
        count = 0
        
        while right < size:
            window_product *= nums[right]

            while window_product >= k:
                window_product /= nums[left]
                left += 1

            count += (right - left + 1)
            right += 1
            
        return count
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(n)$。
- **空间复杂度**：$O(1)$。

# [0714. 买卖股票的最佳时机含手续费](https://leetcode.cn/problems/best-time-to-buy-and-sell-stock-with-transaction-fee/)

- 标签：贪心、数组、动态规划
- 难度：中等

## 题目链接

- [0714. 买卖股票的最佳时机含手续费 - 力扣](https://leetcode.cn/problems/best-time-to-buy-and-sell-stock-with-transaction-fee/)

## 题目大意

给定一个整数数组 `prices`，其中第 `i` 个元素代表了第 `i` 天的股票价格 ；整数 `fee` 代表了交易股票的手续费用。

你可以无限次地完成交易，但是你每笔交易都需要付手续费。如果你已经购买了一个股票，在卖出它之前你就不能再继续购买股票了。

最后要求返回获得利润的最大值。

## 解题思路

这道题的解题思路和「[0122. 买卖股票的最佳时机 II](https://leetcode.cn/problems/best-time-to-buy-and-sell-stock-ii/)」类似，同样可以买卖多次。122 题是在跌入谷底的时候买入，在涨到波峰的时候卖出，这道题多了手续费，则在判断波峰波谷的时候还要考虑手续费。贪心策略如下：

- 当股票价格小于当前最低股价时，更新最低股价，不卖出。
- 当股票价格大于最小价格 + 手续费时，累积股票利润（实质上暂未卖出，等到波峰卖出），同时最低股价减去手续费，以免重复计算。

## 代码

```python
class Solution:
    def maxProfit(self, prices: List[int], fee: int) -> int:
        res = 0
        min_price = prices[0]

        for i in range(1, len(prices)):
            if prices[i] < min_price:
                min_price = prices[i]
            elif prices[i] > min_price + fee:
                res += prices[i] - min_price - fee
                min_price = prices[i] - fee
        return res
```

# [0715. Range 模块](https://leetcode.cn/problems/range-module/)

- 标签：设计、线段树、有序集合
- 难度：困难

## 题目链接

- [0715. Range 模块 - 力扣](https://leetcode.cn/problems/range-module/)

## 题目大意

**描述**：`Range` 模块是跟踪数字范围的模块。

**要求**：

- 设计一个数据结构来跟踪查询半开区间 `[left, right)` 内的数字是否被跟踪。
- 实现 `RangeModule` 类：
  - `RangeModule()` 初始化数据结构的对象。
  - `void addRange(int left, int right)` 添加半开区间 `[left, right)`，跟踪该区间中的每个实数。添加与当前跟踪的数字部分重叠的区间时，应当添加在区间 `[left, right)` 中尚未跟踪的任何数字到该区间中。
  - `boolean queryRange(int left, int right)` 只有在当前正在跟踪区间 `[left, right)` 中的每一个实数时，才返回 `True` ，否则返回 `False`。
  - `void removeRange(int left, int right)` 停止跟踪半开区间 `[left, right)` 中当前正在跟踪的每个实数。

**说明**：

- $1 \le left < right \le 10^9$。

**示例**：

- 示例 1：

```
rangeModule = RangeModule() -> null
rangeModule.addRange(10, 20) -> null
rangeModule.removeRange(14, 16) -> null
rangeModule.queryRange(10, 14) -> True
rangeModule.queryRange(13, 15) -> False
rangeModule.queryRange(16, 17) -> True
```

## 解题思路

### 思路 1：线段树

这道题可以使用线段树来做，但是效率比较差。

区间的范围是 $[0, 10^9]$，普通数组构成的线段树不满足要求。需要用到动态开点线段树。题目要求的是半开区间 `[left, right)` ，而线段树中常用的是闭合区间。但是我们可以将半开区间 `[left, right)` 转为 `[left, right - 1]` 的闭合空间。

这样构建线段树的时间复杂度为 $O(\log n)$，单次区间更新的时间复杂度为 $O(\log n)$，单次区间查询的时间复杂度为 $O(\log n)$。总体时间复杂度为 $O(\log n)$。

## 代码

### 思路 1 代码：

```python
# 线段树的节点类
class TreeNode:
    def __init__(self, left, right, val=False, lazy_tag=None, letNode=None, rightNode=None):
        self.left = left  # 区间左边界
        self.right = right  # 区间右边界
        self.mid = (left + right) >> 1
        self.leftNode = letNode  # 区间左节点
        self.rightNode = rightNode  # 区间右节点
        self.val = val  # 节点值（区间值）
        self.lazy_tag = lazy_tag  # 区间问题的延迟更新标记


class RangeModule:

    def __init__(self):
        self.tree = TreeNode(0, int(1e9))

    # 向上更新 node 节点区间值，节点的区间值等于该节点左右子节点元素值的聚合计算结果
    def __pushup(self, node):
        if node.leftNode and node.rightNode:
            node.val = node.leftNode.val and node.rightNode.val
        else:
            node.val = False

    # 向下更新 node 节点所在区间的左右子节点的值和懒惰标记
    def __pushdown(self, node):
        if not node.leftNode:
            node.leftNode = TreeNode(node.left, node.mid)
        if not node.rightNode:
            node.rightNode = TreeNode(node.mid + 1, node.right)
        if node.lazy_tag is not None:
            node.leftNode.lazy_tag = node.lazy_tag  # 更新左子节点懒惰标记
            node.leftNode.val = node.lazy_tag  # 左子节点每个元素值增加 lazy_tag

            node.rightNode.lazy_tag = node.lazy_tag  # 更新右子节点懒惰标记
            node.rightNode.val = node.lazy_tag  # 右子节点每个元素值增加 lazy_tag

            node.lazy_tag = None  # 更新当前节点的懒惰标记

    # 区间更新
    def __update_interval(self, q_left, q_right, val, node):
        if q_left <= node.left and node.right <= q_right:  # 节点所在区间被 [q_left, q_right] 所覆盖
            node.lazy_tag = val  # 将当前节点的延迟标记增加 val
            node.val = val  # 当前节点所在区间每个元素值增加 val
            return

        self.__pushdown(node)

        if q_left <= node.mid:
            self.__update_interval(q_left, q_right, val, node.leftNode)
        if q_right > node.mid:
            self.__update_interval(q_left, q_right, val, node.rightNode)

        self.__pushup(node)

    # 区间查询，在线段树的 [left, right] 区间范围中搜索区间为 [q_left, q_right] 的区间值
    def __query_interval(self, q_left, q_right, node):
        if q_left <= node.left and node.right <= q_right:  # 节点所在区间被 [q_left, q_right] 所覆盖
            return node.val  # 直接返回节点值

        # 需要向下更新节点所在区间的左右子节点的值和懒惰标记
        self.__pushdown(node)

        if q_right <= node.mid:
            return self.__query_interval(q_left, q_right, node.leftNode)
        if q_left > node.mid:
            return self.__query_interval(q_left, q_right, node.rightNode)

        return self.__query_interval(q_left, q_right, node.leftNode) and self.__query_interval(q_left, q_right, node.rightNode)  # 返回左右子树元素值的聚合计算结果


    def addRange(self, left: int, right: int) -> None:
        self.__update_interval(left, right - 1, True, self.tree)


    def queryRange(self, left: int, right: int) -> bool:
        return self.__query_interval(left, right - 1, self.tree)


    def removeRange(self, left: int, right: int) -> None:
        self.__update_interval(left, right - 1, False, self.tree)
```

# [0718. 最长重复子数组](https://leetcode.cn/problems/maximum-length-of-repeated-subarray/)

- 标签：数组、二分查找、动态规划、滑动窗口、哈希函数、滚动哈希
- 难度：中等

## 题目链接

- [0718. 最长重复子数组 - 力扣](https://leetcode.cn/problems/maximum-length-of-repeated-subarray/)

## 题目大意

**描述**：给定两个整数数组 $nums1$、$nums2$。

**要求**：计算两个数组中公共的、长度最长的子数组长度。

**说明**：

- $1 \le nums1.length, nums2.length \le 1000$。
- $0 \le nums1[i], nums2[i] \le 100$。

**示例**：

- 示例 1：

```python
输入：nums1 = [1,2,3,2,1], nums2 = [3,2,1,4,7]
输出：3
解释：长度最长的公共子数组是 [3,2,1] 。
```

- 示例 2：

```python
输入：nums1 = [0,0,0,0,0], nums2 = [0,0,0,0,0]
输出：5
```

## 解题思路

### 思路 1：暴力（超时）

1. 枚举数组 $nums1$ 和 $nums2$ 的子数组开始位置 $i$、$j$。
2. 如果遇到相同项，即 $nums1[i] == nums2[j]$，则以 $nums1[i]$、$nums2[j]$ 为前缀，同时向后遍历，计算当前的公共子数组长度 $subLen$ 最长为多少。
3. 直到遇到超出数组范围或者 $nums1[i + subLen] == nums2[j + subLen]$ 情况时，停止遍历，并更新答案。
4. 继续执行 $1 \sim 3$ 步，直到遍历完，输出答案。

### 思路 1：代码

```python
class Solution:
    def findLength(self, nums1: List[int], nums2: List[int]) -> int:
        size1, size2 = len(nums1), len(nums2)
        ans = 0
        for i in range(size1):
            for j in range(size2):
                if nums1[i] == nums2[j]:
                    subLen = 1
                    while i + subLen < size1 and j + subLen < size2 and nums1[i + subLen] == nums2[j + subLen]:
                        subLen += 1
                    ans = max(ans, subLen)
        
        return ans
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(n \times m \times min(n, m))$。其中 $n$ 是数组 $nums1$ 的长度，$m$ 是数组 $nums2$ 的长度。
- **空间复杂度**：$O(1)$。

### 思路 2：滑动窗口

暴力方法中，因为子数组在两个数组中的位置不同，所以会导致子数组之间会进行多次比较。

我们可以将两个数组分别看做是两把直尺。然后将数组 $nums1$ 固定， 让 $nums2$ 的尾部与 $nums1$ 的头部对齐，如下所示。

```python
nums1 =             [1, 2, 3, 2, 1]
nums2 = [3, 2, 1, 4, 7]
```

然后逐渐向右移动直尺 $nums2$，比较 $nums1$ 与 $nums2$ 重叠部分中的公共子数组的长度，直到直尺 $nums2$ 的头部移动到 $nums1$ 的尾部。

```python
nums1 =             [1, 2, 3, 2, 1]
nums2 =    [3, 2, 1, 4, 7]

nums1 =             [1, 2, 3, 2, 1]
nums2 =       [3, 2, 1, 4, 7]

nums1 =             [1, 2, 3, 2, 1]
nums2 =          [3, 2, 1, 4, 7]

nums1 =             [1, 2, 3, 2, 1]
nums2 =             [3, 2, 1, 4, 7]

nums1 =             [1, 2, 3, 2, 1]
nums2 =                [3, 2, 1, 4, 7]

nums1 =             [1, 2, 3, 2, 1]
nums2 =                   [3, 2, 1, 4, 7]

nums1 =             [1, 2, 3, 2, 1]
nums2 =                      [3, 2, 1, 4, 7]

nums1 =             [1, 2, 3, 2, 1]
nums2 =                         [3, 2, 1, 4, 7]
```

在这个过程中求得的 $nums1$ 与 $nums2$ 重叠部分中的最大的公共子数组的长度就是 $nums1$ 与 $nums2$ 数组中公共的、长度最长的子数组长度。

### 思路 2：代码

```python
class Solution:
    def findMaxLength(self, nums1, nums2, i, j):
        size1, size2 = len(nums1), len(nums2)
        max_len = 0
        cur_len = 0
        while i < size1 and j < size2:
            if nums1[i] == nums2[j]:
                cur_len += 1
                max_len = max(max_len, cur_len)
            else:
                cur_len = 0
            i += 1
            j += 1
        return max_len

    def findLength(self, nums1: List[int], nums2: List[int]) -> int:
        size1, size2 = len(nums1), len(nums2)
        res = 0
        for i in range(size1):
            res = max(res, self.findMaxLength(nums1, nums2, i, 0))

        for i in range(size2):
            res = max(res, self.findMaxLength(nums1, nums2, 0, i))
        
        return res
```

### 思路 2：复杂度分析

- **时间复杂度**：$O(n + m) \times min(n, m)$。其中 $n$ 是数组 $nums1$ 的长度，$m$ 是数组 $nums2$ 的长度。
- **空间复杂度**：$O(1)$。

### 思路 3：动态规划

###### 1. 划分阶段

按照子数组结尾位置进行阶段划分。

###### 2. 定义状态

定义状态 $dp[i][j]$ 为：「以 $nums1$ 中前 $i$ 个元素为子数组（$nums1[0]...nums2[i - 1]$）」和「以 $nums2$ 中前 $j$ 个元素为子数组（$nums2[0]...nums2[j - 1]$）」的最长公共子数组长度。

###### 3. 状态转移方程

1. 如果 $nums1[i - 1] = nums2[j - 1]$，则当前元素可以构成公共子数组，此时 $dp[i][j] = dp[i - 1][j - 1] + 1$。
2. 如果 $nums1[i - 1] \ne nums2[j - 1]$，则当前元素不能构成公共子数组，此时 $dp[i][j] = 0$。

###### 4. 初始条件

- 当 $i = 0$ 时，$nums1[0]...nums1[i - 1]$ 表示的是空数组，空数组与 $nums2[0]...nums2[j - 1]$ 的最长公共子序列长度为 $0$，即 $dp[0][j] = 0$。
- 当 $j = 0$ 时，$nums2[0]...nums2[j - 1]$ 表示的是空数组，空数组与 $nums1[0]...nums1[i - 1]$ 的最长公共子序列长度为 $0$，即 $dp[i][0] = 0$。

###### 5. 最终结果

- 根据状态定义， $dp[i][j]$ 为：「以 $nums1$ 中前 $i$ 个元素为子数组（$nums1[0]...nums2[i - 1]$）」和「以 $nums2$ 中前 $j$ 个元素为子数组（$nums2[0]...nums2[j - 1]$）」的最长公共子数组长度。在遍历过程中，我们可以使用 $res$ 记录下所有 $dp[i][j]$ 中最大值即为答案。

### 思路 3：代码

```python
class Solution:
    def findLength(self, nums1: List[int], nums2: List[int]) -> int:
        size1 = len(nums1)
        size2 = len(nums2)
        dp = [[0 for _ in range(size2 + 1)] for _ in range(size1 + 1)]
        res = 0
        for i in range(1, size1 + 1):
            for j in range(1, size2 + 1):
                if nums1[i - 1] == nums2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                if dp[i][j] > res:
                    res = dp[i][j]

        return res
```

### 思路 3：复杂度分析

- **时间复杂度**：$O(n \times m)$。其中 $n$ 是数组 $nums1$ 的长度，$m$ 是数组 $nums2$ 的长度。
- **空间复杂度**：$O(n \times m)$。# [0719. 找出第 K 小的距离对](https://leetcode.cn/problems/find-k-th-smallest-pair-distance/)

- 标签：数组、双指针、二分查找、排序
- 难度：困难

## 题目链接

- [0719. 找出第 K 小的距离对 - 力扣](https://leetcode.cn/problems/find-k-th-smallest-pair-distance/)

## 题目大意

**描述**：给定一个整数数组 $nums$，对于数组中不同的数 $nums[i]$、$nums[j]$ 之间的距离定义为 $nums[i]$ 和 $nums[j]$ 的绝对差值，即 $dist(nums[i], nums[j]) = abs(nums[i] - nums[j])$。

**要求**：求所有数对之间第 $k$ 个最小距离。

**说明**：

- $n == nums.length$
- $2 \le n \le 10^4$。
- $0 \le nums[i] \le 10^6$。
- $1 \le k \le n \times (n - 1) / 2$。

**示例**：

- 示例 1：

```python
输入：nums = [1,3,1], k = 1
输出：0
解释：数对和对应的距离如下：
(1,3) -> 2
(1,1) -> 0
(3,1) -> 2
距离第 1 小的数对是 (1,1) ，距离为 0。
```

- 示例 2：

```python
输入：nums = [1,1,1], k = 2
输出：0
```

## 解题思路

### 思路 1：二分查找算法

一般来说 topK 问题都可以用堆排序来解决。但是这道题使用堆排序超时了。所以需要换其他方法。

先来考虑第 $k$ 个最小距离的范围。这个范围一定在 $[0, max(nums) - min(nums)]$ 之间。

我们可以对 $nums$ 先进行排序，然后得到最小距离为 $0$，最大距离为 $nums[-1] - nums[0]$。我们可以在这个区间上进行二分，对于二分的位置 $mid$，统计距离小于等于 $mid$ 的距离对数，并根据它和 $k$ 的关系调整区间上下界。

统计对数可以使用双指针来计算出所有小于等于 $mid$ 的距离对数目。

1. 维护两个指针 $left$、$right$。$left$、$right$ 都指向数组开头位置。
2. 然后不断移动 $right$，计算 $nums[right]$ 和 $nums[left]$ 之间的距离。
3. 如果大于 $mid$，则 $left$ 向右移动，直到距离小于等于 $mid$ 时，统计当前距离对数为 $right - left$。
4. 最终将这些符合要求的距离对数累加，就得到了所有小于等于 $mid$ 的距离对数目。

### 思路 1：代码

```python
class Solution:
    def smallestDistancePair(self, nums: List[int], k: int) -> int:
        def get_count(dist):
            left, count = 0, 0
            for right in range(1, len(nums)):
                while nums[right] - nums[left] > dist:
                    left += 1
                count += (right - left)
            return count

        nums.sort()
        left, right = 0, nums[-1] - nums[0]
        while left < right:
            mid = left + (right - left) // 2
            if get_count(mid) >= k:
                right = mid
            else:
                left = mid + 1
        return left
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(n \times \log n)$，其中 $n$ 为数组 $nums$ 中的元素个数。
- **空间复杂度**：$O(\log n)$，排序算法所用到的空间复杂度为 $O(\log n)$。
# [0720. 词典中最长的单词](https://leetcode.cn/problems/longest-word-in-dictionary/)

- 标签：字典树、数组、哈希表、字符串、排序
- 难度：中等

## 题目链接

- [0720. 词典中最长的单词 - 力扣](https://leetcode.cn/problems/longest-word-in-dictionary/)

## 题目大意

给出一个字符串数组 `words` 组成的一本英语词典。

要求：从中找出最长的一个单词，该单词是由 `words` 词典中其他单词逐步添加一个字母组成。若其中有多个可行的答案，则返回答案中字典序最小的单词。若无答案，则返回空字符串。

## 解题思路

使用字典树存储每一个单词。再在字典树中查找每一个单词，查找的时候判断是否有以当前单词为前缀的单词。如果有，则该单词可以由前缀构成的单词逐步添加字母获得。此时，如果该单词比答案单词更长，则维护更新答案单词。

最后输出答案单词。

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
            if ch not in cur.children or not cur.children[ch].isEnd:
                return False
            cur = cur.children[ch]

        return cur is not None and cur.isEnd

class Solution:
    def longestWord(self, words: List[str]) -> str:

        trie_tree = Trie()
        for word in words:
            trie_tree.insert(word)

        ans = ""
        for word in words:
            if trie_tree.search(word):
                if len(word) > len(ans):
                    ans = word
                elif len(word) == len(ans) and word < ans:
                    ans = word
        return ans
```

# [0724. 寻找数组的中心下标](https://leetcode.cn/problems/find-pivot-index/)

- 标签：数组、前缀和
- 难度：简单

## 题目链接

- [0724. 寻找数组的中心下标 - 力扣](https://leetcode.cn/problems/find-pivot-index/)

## 题目大意

**描述**：给定一个数组 $nums$。

**要求**：找到「左侧元素和」与「右侧元素和相等」的位置，若找不到，则返回 $-1$。

**说明**：

- $1 \le nums.length \le 10^4$。
- $-1000 \le nums[i] \le 1000$。

**示例**：

- 示例 1：

```python
输入：nums = [1, 7, 3, 6, 5, 6]
输出：3
解释：
中心下标是 3 。
左侧数之和 sum = nums[0] + nums[1] + nums[2] = 1 + 7 + 3 = 11，
右侧数之和 sum = nums[4] + nums[5] = 5 + 6 = 11，二者相等。
```

- 示例 2：

```python
输入：nums = [1, 2, 3]
输出：-1
解释：
数组中不存在满足此条件的中心下标。
```

## 解题思路

### 思路 1：两次遍历

两次遍历，第一次遍历先求出数组全部元素和。第二次遍历找到左侧元素和恰好为全部元素和一半的位置。

### 思路 1：代码

```python
class Solution:
    def pivotIndex(self, nums: List[int]) -> int:
        sum = 0
        for i in range(len(nums)):
            sum += nums[i]
        curr_sum = 0
        for i in range(len(nums)):
            if curr_sum * 2 + nums[i] == sum:
                return i
            curr_sum += nums[i]
        return -1
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(n)$。两次遍历的时间复杂度为 $O(2 \times n)$ ，$O(2 \times n) == O(n)$。
- **空间复杂度**：$O(1)$。

# [0727. 最小窗口子序列](https://leetcode.cn/problems/minimum-window-subsequence/)

- 标签：字符串、动态规划、滑动窗口
- 难度：困难

## 题目链接

- [0727. 最小窗口子序列 - 力扣](https://leetcode.cn/problems/minimum-window-subsequence/)

## 题目大意

给定字符串 `s1` 和 `s2`。

要求：找出 `s1` 中最短的（连续）子串 `w`，使得 `s2` 是 `w` 的子序列 。如果 `s1` 中没有窗口可以包含 `s2` 中的所有字符，返回空字符串 `""`。如果有不止一个最短长度的窗口，返回开始位置最靠左的那个。

## 解题思路

这道题跟「[76. 最小覆盖子串](https://leetcode.cn/problems/minimum-window-substring/)」有点类似。但这道题中字符的相对顺序需要保持一致。求解的思路如下：

- 向右扩大窗口，匹配字符，直到匹配完 `s2` 的最后一个字符。
- 当满足条件时，缩小窗口，并更新最小窗口的起始位置和最短长度。
- 缩小窗口到不满足条件为止。

这道题的难点在于第二步中如何缩小窗口。当匹配到一个子序列时，可以采用逆向匹配的方式，从 `s2` 的最后一位字符匹配到 `s2` 的第一位字符。找到符合要求的最大下标，即是窗口的左边界。

整个算法的解题步骤如下：

- 使用两个指针 `left`、`right` 代表窗口的边界，一开始都指向 `0` 。`min_len` 用来记录最小子序列的长度。`i`、`j` 作为索引，用于遍历字符串 `s1` 和 `s2`，一开始都为 `0`。
- 遍历字符串 `s1` 的每一个字符，如果 `s1[i] == s2[j]`，则说明 `s2` 中第 `j` 个字符匹配了，向右移动 `j`，即 `j += 1`，然后继续匹配。
- 如果 `j == len(s2)`，则说明 `s2` 中所有字符都匹配了。
  - 此时确定了窗口的右边界 `right = i`，并令 `j` 指向 `s2` 最后一个字符位置。
  - 从右至左逆向匹配字符串，找到窗口的左边界。
  - 判断当前窗口长度和窗口的最短长度，并更新最小窗口的起始位置和最短长度。
  - 令 `j = 0`，重新继续匹配 `s2`。
- 向右移动 `i`，继续匹配。
- 遍历完输出窗口的最短长度（需要判断是否有解）。

## 代码

```python
class Solution:
    def minWindow(self, s1: str, s2: str) -> str:
        i, j = 0, 0
        min_len = float('inf')
        left, right = 0, 0
        while i < len(s1):
            if s1[i] == s2[j]:
                j += 1
            # 完成了匹配
            if j == len(s2):
                right = i
                j -= 1
                while j >= 0:
                    if s1[i] == s2[j]:
                        j -= 1
                    i -= 1
                i += 1
                if right - i + 1 < min_len:
                    left = i
                    min_len = right - left + 1
                j = 0
            i += 1
        if min_len != float('inf'):
            return s1[left: left + min_len]
        return ""
```

## 参考资料

- 【题解】[c++ 简单好理解的 滑动窗口解法 和 动态规划解法 - 最小窗口子序列 - 力扣](https://leetcode.cn/problems/minimum-window-subsequence/solution/c-jian-dan-hao-li-jie-de-hua-dong-chuang-wguk/)
- 【题解】[727. 最小窗口子序列 C++ 滑动窗口 - 最小窗口子序列 - 力扣](https://leetcode.cn/problems/minimum-window-subsequence/solution/727-zui-xiao-chuang-kou-zi-xu-lie-c-hua-dong-chuan/)

# [0729. 我的日程安排表 I](https://leetcode.cn/problems/my-calendar-i/)

- 标签：设计、线段树、二分查找、有序集合
- 难度：中等

## 题目链接

- [0729. 我的日程安排表 I - 力扣](https://leetcode.cn/problems/my-calendar-i/)

## 题目大意

**要求**：实现一个 `MyCalendar` 类来存放你的日程安排。如果要添加的日程安排不会造成重复预订 ，则可以存储这个新的日程安排。

日程可以用一对整数 $start$ 和 $end$ 表示，这里的时间是半开区间，即 $[start, end)$，实数 $x$ 的范围为 $start \le x < end$。

`MyCalendar` 类：

- `MyCalendar()` 初始化日历对象。
- `boolean book(int start, int end)` 如果可以将日程安排成功添加到日历中而不会导致重复预订，返回 `True` 。否则，返回 `False` 并且不要将该日程安排添加到日历中。

**说明**：

- 重复预订：当两个日程安排有一些时间上的交叉时（例如两个日程安排都在同一时间内），就会产生重复预订 。
- $0 \le start < end \le 10^9$
- 每个测试用例，调用 `book` 方法的次数最多不超过 `1000` 次。

**示例**：

- 示例 1：

```python
输入：
["MyCalendar", "book", "book", "book"]
[[], [10, 20], [15, 25], [20, 30]]

输出：
[null, true, false, true]

解释：
MyCalendar myCalendar = new MyCalendar();
myCalendar.book(10, 20); // return True
myCalendar.book(15, 25); // return False ，这个日程安排不能添加到日历中，因为时间 15 已经被另一个日程安排预订了。
myCalendar.book(20, 30); // return True ，这个日程安排可以添加到日历中，因为第一个日程安排预订的每个时间都小于 20 ，且不包含时间 20 。
```

## 解题思路

### 思路 1：线段树

这道题可以使用线段树来做。

因为区间的范围是 $[0, 10^9]$，普通数组构成的线段树不满足要求。需要用到动态开点线段树。

- 构建一棵线段树。每个线段树的节点类存储当前区间中保存的日程区间个数。

- 在 `book` 方法中，从线段树中查询 `[start, end - 1]` 区间上保存的日程区间个数。
  - 如果日程区间个数大于等于 `1`，则说明该日程添加到日历中会导致重复预订，则直接返回 `False`。
  - 如果日程区间个数小于 `1`，则说明该日程添加到日历中不会导致重复预定，则在线段树中将区间 `[start, end - 1]` 的日程区间个数 + 1，然后返回 `True`。

### 思路 1：线段树代码

```python
# 线段树的节点类
class SegTreeNode:
    def __init__(self, left=-1, right=-1, val=0, lazy_tag=None, leftNode=None, rightNode=None):
        self.left = left                            # 区间左边界
        self.right = right                          # 区间右边界
        self.mid = left + (right - left) // 2
        self.leftNode = leftNode                    # 区间左节点
        self.rightNode = rightNode                  # 区间右节点
        self.val = val                              # 节点值（区间值）
        self.lazy_tag = lazy_tag                    # 区间问题的延迟更新标记
        
        
# 线段树类
class SegmentTree:
    # 初始化线段树接口
    def __init__(self, function):
        self.tree = SegTreeNode(0, int(1e9))
        self.function = function                    # function 是一个函数，左右区间的聚合方法
            
    # 单点更新，将 nums[i] 更改为 val
    def update_point(self, i, val):
        self.__update_point(i, val, self.tree)
    
    # 区间更新，将区间为 [q_left, q_right] 上的元素值修改为 val
    def update_interval(self, q_left, q_right, val):
        self.__update_interval(q_left, q_right, val, self.tree)
        
    # 区间查询，查询区间为 [q_left, q_right] 的区间值
    def query_interval(self, q_left, q_right):
        return self.__query_interval(q_left, q_right, self.tree)
    
    # 获取 nums 数组接口：返回 nums 数组
    def get_nums(self, length):
        nums = [0 for _ in range(length)]
        for i in range(length):
            nums[i] = self.query_interval(i, i)
        return nums
    
    
    # 以下为内部实现方法
    
    # 单点更新，将 nums[i] 更改为 val。node 节点的区间为 [node.left, node.right]
    def __update_point(self, i, val, node):
        if node.left == node.right:
            node.val = val                          # 叶子节点，节点值修改为 val
            return
        
        if i <= node.mid:                           # 在左子树中更新节点值
            self.__update_point(i, val, node.leftNode)
        else:                                       # 在右子树中更新节点值
            self.__update_point(i, val, node.rightNode)
        self.__pushup(node)                         # 向上更新节点的区间值
    
    # 区间更新
    def __update_interval(self, q_left, q_right, val, node):
        if node.left >= q_left and node.right <= q_right:  # 节点所在区间被 [q_left, q_right] 所覆盖
            if node.lazy_tag is not None:
                node.lazy_tag += val                # 将当前节点的延迟标记增加 val
            else:
                node.lazy_tag = val                 # 将当前节点的延迟标记增加 val
            node.val += val          # 当前节点所在区间每个元素值增加 val
            return
        if node.right < q_left or node.left > q_right:  # 节点所在区间与 [q_left, q_right] 无关
            return 0
    
        self.__pushdown(node)                       # 向下更新节点所在区间的左右子节点的值和懒惰标记
    
        if q_left <= node.mid:                      # 在左子树中更新区间值
            self.__update_interval(q_left, q_right, val, node.leftNode)
        if q_right > node.mid:                      # 在右子树中更新区间值
            self.__update_interval(q_left, q_right, val, node.rightNode)
            
        self.__pushup(node)
    
    # 区间查询，在线段树的 [left, right] 区间范围中搜索区间为 [q_left, q_right] 的区间值
    def __query_interval(self, q_left, q_right, node):
        if node.left >= q_left and node.right <= q_right:   # 节点所在区间被 [q_left, q_right] 所覆盖
            return node.val                         # 直接返回节点值
        if node.right < q_left or node.left > q_right:  # 节点所在区间与 [q_left, q_right] 无关
            return 0
                                  
        self.__pushdown(node)                       # 向下更新节点所在区间的左右子节点的值和懒惰标记
        
        res_left = 0                                # 左子树查询结果
        res_right = 0                               # 右子树查询结果
        if q_left <= node.mid:                      # 在左子树中查询
            res_left = self.__query_interval(q_left, q_right, node.leftNode)
        if q_right > node.mid:                      # 在右子树中查询
            res_right = self.__query_interval(q_left, q_right, node.rightNode)
        return self.function(res_left, res_right)   # 返回左右子树元素值的聚合计算结果

    # 向上更新 node 节点区间值，节点的区间值等于该节点左右子节点元素值的聚合计算结果
    def __pushup(self, node):
        if node.leftNode and node.rightNode:
            node.val = self.function(node.leftNode.val, node.rightNode.val)
    
    # 向下更新 node 节点所在区间的左右子节点的值和懒惰标记
    def __pushdown(self, node):
        if node.leftNode is None:
            node.leftNode = SegTreeNode(node.left, node.mid)
        if node.rightNode is None:
            node.rightNode = SegTreeNode(node.mid + 1, node.right)
            
        lazy_tag = node.lazy_tag
        if node.lazy_tag is None:
            return
            
        if node.leftNode.lazy_tag is not None:
            node.leftNode.lazy_tag += lazy_tag      # 更新左子节点懒惰标记
        else:
            node.leftNode.lazy_tag = lazy_tag       # 更新左子节点懒惰标记
        node.leftNode.val += lazy_tag               # 左子节点每个元素值增加 lazy_tag
        
        if node.rightNode.lazy_tag is not None:
            node.rightNode.lazy_tag += lazy_tag     # 更新右子节点懒惰标记
        else:
            node.rightNode.lazy_tag = lazy_tag      # 更新右子节点懒惰标记
        node.rightNode.val += lazy_tag              # 右子节点每个元素值增加 lazy_tag
        
        node.lazy_tag = None                        # 更新当前节点的懒惰标记

class MyCalendar:

    def __init__(self):
        self.STree = SegmentTree(lambda x, y: max(x, y))


    def book(self, start: int, end: int) -> bool:
        if self.STree.query_interval(start, end - 1) >= 1:
            return False
        self.STree.update_interval(start, end - 1, 1)
        return True
```
# [731. 我的日程安排表 II](https://leetcode.cn/problems/my-calendar-ii/)

- 标签：设计、线段树、二分查找、有序集合
- 难度：中等

## 题目链接

- [731. 我的日程安排表 II - 力扣](https://leetcode.cn/problems/my-calendar-ii/)

## 题目大意

**要求**：实现一个 `MyCalendar` 类来存放你的日程安排。如果要添加的时间内不会导致三重预订时，则可以存储这个新的日程安排。

日程可以用一对整数 $start$ 和 $end$ 表示，这里的时间是半开区间，即 $[start, end)$，实数 $x$ 的范围为 $start \le x < end$。

`MyCalendar` 类：

- `MyCalendar()` 初始化日历对象。
- `boolean book(int start, int end)` 如果可以将日程安排成功添加到日历中而不会导致三重预订，返回 `True` 。否则，返回 `False` 并且不要将该日程安排添加到日历中。

**说明**：

- 三重预定：当三个日程安排有一些时间上的交叉时（例如三个日程安排都在同一时间内），就会产生三重预订 。
- $0 \le start < end \le 10^9$。
- 每个测试用例，调用 `book` 方法的次数最多不超过 `1000` 次。

**示例**：

- 示例 1：

```python
输入：
["MyCalendar", "book", "book", "book"]
[[], [10, 20], [15, 25], [20, 30]]
输出：
[null, true, false, true]

解释：
MyCalendar myCalendar = new MyCalendar();
myCalendar.book(10, 20); // return True
myCalendar.book(15, 25); // return False ，这个日程安排不能添加到日历中，因为时间 15 已经被另一个日程安排预订了。
myCalendar.book(20, 30); // return True ，这个日程安排可以添加到日历中，因为第一个日程安排预订的每个时间都小于 20 ，且不包含时间 20 。
```

## 解题思路

### 思路 1：线段树

这道题可以使用线段树来做。

因为区间的范围是 $[0, 10^9]$，普通数组构成的线段树不满足要求。需要用到动态开点线段树。

- 构建一棵线段树。每个线段树的节点类存储当前区间中保存的日程区间个数。

- 在 `book` 方法中，从线段树中查询 `[start, end - 1]` 区间上保存的日程区间个数。
  - 如果日程区间个数大于等于 `2`，则说明该日程添加到日历中会导致三重预订，则直接返回 `False`。
  - 如果日程区间个数小于 `2`，则说明该日程添加到日历中不会导致三重预订，则在线段树中将区间 `[start, end - 1]` 的日程区间个数 + 1，然后返回 `True`。

### 思路 1：线段树代码

```python
# 线段树的节点类
class SegTreeNode:
    def __init__(self, left=-1, right=-1, val=0, lazy_tag=None, leftNode=None, rightNode=None):
        self.left = left                            # 区间左边界
        self.right = right                          # 区间右边界
        self.mid = left + (right - left) // 2
        self.leftNode = leftNode                    # 区间左节点
        self.rightNode = rightNode                  # 区间右节点
        self.val = val                              # 节点值（区间值）
        self.lazy_tag = lazy_tag                    # 区间问题的延迟更新标记
        
        
# 线段树类
class SegmentTree:
    # 初始化线段树接口
    def __init__(self, function):
        self.tree = SegTreeNode(0, int(1e9))
        self.function = function                    # function 是一个函数，左右区间的聚合方法
            
    # 单点更新，将 nums[i] 更改为 val
    def update_point(self, i, val):
        self.__update_point(i, val, self.tree)
    
    # 区间更新，将区间为 [q_left, q_right] 上的元素值修改为 val
    def update_interval(self, q_left, q_right, val):
        self.__update_interval(q_left, q_right, val, self.tree)
        
    # 区间查询，查询区间为 [q_left, q_right] 的区间值
    def query_interval(self, q_left, q_right):
        return self.__query_interval(q_left, q_right, self.tree)
    
    # 获取 nums 数组接口：返回 nums 数组
    def get_nums(self, length):
        nums = [0 for _ in range(length)]
        for i in range(length):
            nums[i] = self.query_interval(i, i)
        return nums
    
    
    # 以下为内部实现方法
    
    # 单点更新，将 nums[i] 更改为 val。node 节点的区间为 [node.left, node.right]
    def __update_point(self, i, val, node):
        if node.left == node.right:
            node.val = val                          # 叶子节点，节点值修改为 val
            return
        
        if i <= node.mid:                           # 在左子树中更新节点值
            self.__update_point(i, val, node.leftNode)
        else:                                       # 在右子树中更新节点值
            self.__update_point(i, val, node.rightNode)
        self.__pushup(node)                         # 向上更新节点的区间值
    
    # 区间更新
    def __update_interval(self, q_left, q_right, val, node):
        if node.left >= q_left and node.right <= q_right:  # 节点所在区间被 [q_left, q_right] 所覆盖
            if node.lazy_tag is not None:
                node.lazy_tag += val                # 将当前节点的延迟标记增加 val
            else:
                node.lazy_tag = val                 # 将当前节点的延迟标记增加 val
            node.val += val                         # 当前节点所在区间增加 val
            return
        if node.right < q_left or node.left > q_right:  # 节点所在区间与 [q_left, q_right] 无关
            return 0
    
        self.__pushdown(node)                       # 向下更新节点所在区间的左右子节点的值和懒惰标记
    
        if q_left <= node.mid:                      # 在左子树中更新区间值
            self.__update_interval(q_left, q_right, val, node.leftNode)
        if q_right > node.mid:                      # 在右子树中更新区间值
            self.__update_interval(q_left, q_right, val, node.rightNode)
            
        self.__pushup(node)
    
    # 区间查询，在线段树的 [left, right] 区间范围中搜索区间为 [q_left, q_right] 的区间值
    def __query_interval(self, q_left, q_right, node):
        if node.left >= q_left and node.right <= q_right:   # 节点所在区间被 [q_left, q_right] 所覆盖
            return node.val                         # 直接返回节点值
        if node.right < q_left or node.left > q_right:  # 节点所在区间与 [q_left, q_right] 无关
            return 0
                                  
        self.__pushdown(node)                       # 向下更新节点所在区间的左右子节点的值和懒惰标记
        
        res_left = 0                                # 左子树查询结果
        res_right = 0                               # 右子树查询结果
        if q_left <= node.mid:                      # 在左子树中查询
            res_left = self.__query_interval(q_left, q_right, node.leftNode)
        if q_right > node.mid:                      # 在右子树中查询
            res_right = self.__query_interval(q_left, q_right, node.rightNode)
        return self.function(res_left, res_right)   # 返回左右子树元素值的聚合计算结果

    # 向上更新 node 节点区间值，节点的区间值等于该节点左右子节点元素值的聚合计算结果
    def __pushup(self, node):
        if node.leftNode and node.rightNode:
            node.val = self.function(node.leftNode.val, node.rightNode.val)
    
    # 向下更新 node 节点所在区间的左右子节点的值和懒惰标记
    def __pushdown(self, node):
        if node.leftNode is None:
            node.leftNode = SegTreeNode(node.left, node.mid)
        if node.rightNode is None:
            node.rightNode = SegTreeNode(node.mid + 1, node.right)
            
        lazy_tag = node.lazy_tag
        if node.lazy_tag is None:
            return
            
        if node.leftNode.lazy_tag is not None:
            node.leftNode.lazy_tag += lazy_tag      # 更新左子节点懒惰标记
        else:
            node.leftNode.lazy_tag = lazy_tag       # 更新左子节点懒惰标记
        node.leftNode.val += lazy_tag               # 左子节点区间增加 lazy_tag
        
        if node.rightNode.lazy_tag is not None:
            node.rightNode.lazy_tag += lazy_tag     # 更新右子节点懒惰标记
        else:
            node.rightNode.lazy_tag = lazy_tag      # 更新右子节点懒惰标记
        node.rightNode.val += lazy_tag              # 右子节点区间增加 lazy_tag
        
        node.lazy_tag = None                        # 更新当前节点的懒惰标记

class MyCalendarTwo:

    def __init__(self):
        self.STree = SegmentTree(lambda x, y: max(x, y))


    def book(self, start: int, end: int) -> bool:
        if self.STree.query_interval(start, end - 1) >= 2:
            return False
        self.STree.update_interval(start, end - 1, 1)
        return True
```# [0732. 我的日程安排表 III](https://leetcode.cn/problems/my-calendar-iii/)

- 标签：设计、线段树、二分查找、有序集合
- 难度：困难

## 题目链接

- [0732. 我的日程安排表 III - 力扣](https://leetcode.cn/problems/my-calendar-iii/)

## 题目大意

**要求**：实现一个 `MyCalendarThree` 类来存放你的日程安排，你可以一直添加新的日程安排。

日程可以用一对整数 $start$ 和 $end$ 表示，这里的时间是半开区间，即 $[start, end)$，实数 $x$ 的范围为 $start \le x < end$。

`MyCalendarThree` 类：

- `MyCalendarThree()` 初始化对象。
- `int book(int start, int end)` 返回一个整数 `k`，表示日历中存在的 `k` 次预订的最大值。

**说明**：

- `k` 次预定：当 `k` 个日程安排有一些时间上的交叉时（例如 `k` 个日程安排都在同一时间内），就会产生 `k` 次预订。
- $0 \le start < end \le 10^9$
- 每个测试用例，调用 `book` 函数最多不超过 `400` 次。

**示例**：

- 示例 1：

```python
输入
["MyCalendarThree", "book", "book", "book", "book", "book", "book"]
[[], [10, 20], [50, 60], [10, 40], [5, 15], [5, 10], [25, 55]]
输出
[null, 1, 1, 2, 3, 3, 3]

解释
MyCalendarThree myCalendarThree = new MyCalendarThree();
myCalendarThree.book(10, 20); // 返回 1 ，第一个日程安排可以预订并且不存在相交，所以最大 k 次预订是 1 次预订。
myCalendarThree.book(50, 60); // 返回 1 ，第二个日程安排可以预订并且不存在相交，所以最大 k 次预订是 1 次预订。
myCalendarThree.book(10, 40); // 返回 2 ，第三个日程安排 [10, 40) 与第一个日程安排相交，所以最大 k 次预订是 2 次预订。
myCalendarThree.book(5, 15); // 返回 3 ，剩下的日程安排的最大 k 次预订是 3 次预订。
myCalendarThree.book(5, 10); // 返回 3
myCalendarThree.book(25, 55); // 返回 3
```

## 解题思路

### 思路 1：线段树

这道题可以使用线段树来做。

因为区间的范围是 $[0, 10^9]$，普通数组构成的线段树不满足要求。需要用到动态开点线段树。

- 构建一棵线段树。每个线段树的节点类存储当前区间中保存的日程区间个数。

- 在 `book` 方法中，在线段树中更新 `[start, end - 1]` 的交叉日程区间个数，即令其区间值整体加 `1`。

- 然后从线段树中查询区间 $[0, 10^9]$ 上保存的交叉日程区间个数，并返回。


### 思路 1：代码

```python
# 线段树的节点类
class SegTreeNode:
    def __init__(self, left=-1, right=-1, val=0, lazy_tag=None, leftNode=None, rightNode=None):
        self.left = left                            # 区间左边界
        self.right = right                          # 区间右边界
        self.mid = left + (right - left) // 2
        self.leftNode = leftNode                    # 区间左节点
        self.rightNode = rightNode                  # 区间右节点
        self.val = val                              # 节点值（区间值）
        self.lazy_tag = lazy_tag                    # 区间问题的延迟更新标记
        
        
# 线段树类
class SegmentTree:
    # 初始化线段树接口
    def __init__(self, function):
        self.tree = SegTreeNode(0, int(1e9))
        self.function = function                    # function 是一个函数，左右区间的聚合方法
            
    # 单点更新，将 nums[i] 更改为 val
    def update_point(self, i, val):
        self.__update_point(i, val, self.tree)
    
    # 区间更新，将区间为 [q_left, q_right] 上的元素值修改为 val
    def update_interval(self, q_left, q_right, val):
        self.__update_interval(q_left, q_right, val, self.tree)
        
    # 区间查询，查询区间为 [q_left, q_right] 的区间值
    def query_interval(self, q_left, q_right):
        return self.__query_interval(q_left, q_right, self.tree)
    
    # 获取 nums 数组接口：返回 nums 数组
    def get_nums(self, length):
        nums = [0 for _ in range(length)]
        for i in range(length):
            nums[i] = self.query_interval(i, i)
        return nums
    
    
    # 以下为内部实现方法
    
    # 单点更新，将 nums[i] 更改为 val。node 节点的区间为 [node.left, node.right]
    def __update_point(self, i, val, node):
        if node.left == node.right:
            node.val = val                          # 叶子节点，节点值修改为 val
            return
        
        if i <= node.mid:                           # 在左子树中更新节点值
            self.__update_point(i, val, node.leftNode)
        else:                                       # 在右子树中更新节点值
            self.__update_point(i, val, node.rightNode)
        self.__pushup(node)                         # 向上更新节点的区间值
    
    # 区间更新
    def __update_interval(self, q_left, q_right, val, node):
        if node.left >= q_left and node.right <= q_right:  # 节点所在区间被 [q_left, q_right] 所覆盖
            if node.lazy_tag is not None:
                node.lazy_tag += val                # 将当前节点的延迟标记增加 val
            else:
                node.lazy_tag = val                 # 将当前节点的延迟标记增加 val
            node.val += val                         # 当前节点所在区间增加 val
            return
        if node.right < q_left or node.left > q_right:  # 节点所在区间与 [q_left, q_right] 无关
            return 0
    
        self.__pushdown(node)                       # 向下更新节点所在区间的左右子节点的值和懒惰标记
    
        if q_left <= node.mid:                      # 在左子树中更新区间值
            self.__update_interval(q_left, q_right, val, node.leftNode)
        if q_right > node.mid:                      # 在右子树中更新区间值
            self.__update_interval(q_left, q_right, val, node.rightNode)
            
        self.__pushup(node)
    
    # 区间查询，在线段树的 [left, right] 区间范围中搜索区间为 [q_left, q_right] 的区间值
    def __query_interval(self, q_left, q_right, node):
        if node.left >= q_left and node.right <= q_right:   # 节点所在区间被 [q_left, q_right] 所覆盖
            return node.val                         # 直接返回节点值
        if node.right < q_left or node.left > q_right:  # 节点所在区间与 [q_left, q_right] 无关
            return 0
                                  
        self.__pushdown(node)                       # 向下更新节点所在区间的左右子节点的值和懒惰标记
        
        res_left = 0                                # 左子树查询结果
        res_right = 0                               # 右子树查询结果
        if q_left <= node.mid:                      # 在左子树中查询
            res_left = self.__query_interval(q_left, q_right, node.leftNode)
        if q_right > node.mid:                      # 在右子树中查询
            res_right = self.__query_interval(q_left, q_right, node.rightNode)
        return self.function(res_left, res_right)   # 返回左右子树元素值的聚合计算结果

    # 向上更新 node 节点区间值，节点的区间值等于该节点左右子节点元素值的聚合计算结果
    def __pushup(self, node):
        if node.leftNode and node.rightNode:
            node.val = self.function(node.leftNode.val, node.rightNode.val)
    
    # 向下更新 node 节点所在区间的左右子节点的值和懒惰标记
    def __pushdown(self, node):
        if node.leftNode is None:
            node.leftNode = SegTreeNode(node.left, node.mid)
        if node.rightNode is None:
            node.rightNode = SegTreeNode(node.mid + 1, node.right)
            
        lazy_tag = node.lazy_tag
        if node.lazy_tag is None:
            return
            
        if node.leftNode.lazy_tag is not None:
            node.leftNode.lazy_tag += lazy_tag      # 更新左子节点懒惰标记
        else:
            node.leftNode.lazy_tag = lazy_tag       # 更新左子节点懒惰标记
        node.leftNode.val += lazy_tag               # 左子节点区间增加 lazy_tag
        
        if node.rightNode.lazy_tag is not None:
            node.rightNode.lazy_tag += lazy_tag     # 更新右子节点懒惰标记
        else:
            node.rightNode.lazy_tag = lazy_tag      # 更新右子节点懒惰标记
        node.rightNode.val += lazy_tag              # 右子节点区间增加 lazy_tag
        
        node.lazy_tag = None                        # 更新当前节点的懒惰标记


class MyCalendarThree:

    def __init__(self):
        self.STree = SegmentTree(lambda x, y: max(x, y))


    def book(self, start: int, end: int) -> int:
        self.STree.update_interval(start, end - 1, 1)
        return self.STree.query_interval(0, int(1e9))



# Your MyCalendarThree object will be instantiated and called as such:
# obj = MyCalendarThree()
# param_1 = obj.book(start,end)
```
# [0733. 图像渲染](https://leetcode.cn/problems/flood-fill/)

- 标签：深度优先搜索、广度优先搜索、数组、矩阵
- 难度：简单

## 题目链接

- [0733. 图像渲染 - 力扣](https://leetcode.cn/problems/flood-fill/)

## 题目大意

给定一个二维数组 image 表示图画，数组的每个元素值表示该位置的像素值大小。再给定一个坐标 (sr, sc) 表示图像渲染开始的位置。然后再给定一个新的颜色值 newColor。现在要求：将坐标 (sr, sc) 以及 (sr, sc) 相连的上下左右区域上与 (sr, sc) 原始颜色相同的区域染色为 newColor。返回染色后的二维数组。



## 解题思路

从起点开始，对上下左右四个方向进行广度优先搜索。每次搜索到一个位置时，如果该位置上的像素值与初始位置像素值相同，则更新该位置像素值，并将该位置加入队列中。最后将二维数组返回。

- 注意：如果起点位置初始颜色和新颜色值 newColor 相同，则不需要染色，直接返回原数组即可。

## 代码

```python
import collections

class Solution:
    def floodFill(self, image: List[List[int]], sr: int, sc: int, newColor: int) -> List[List[int]]:
        if newColor == image[sr][sc]:
            return image
        directions = {(1, 0), (-1, 0), (0, 1), (0, -1)}
        queue = collections.deque([(sr, sc)])
        oriColor = image[sr][sc]
        while queue:
            point = queue.popleft()
            image[point[0]][point[1]] = newColor
            for direction in directions:
                new_i = point[0] + direction[0]
                new_j = point[1] + direction[1]
                if 0 <= new_i < len(image) and 0 <= new_j < len(image[0]) and image[new_i][new_j] == oriColor:
                    queue.append((new_i, new_j))
        return image
```

# [0735. 行星碰撞](https://leetcode.cn/problems/asteroid-collision/)

- 标签：栈、数组
- 难度：中等

## 题目链接

- [0735. 行星碰撞 - 力扣](https://leetcode.cn/problems/asteroid-collision/)

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

# [0738. 单调递增的数字](https://leetcode.cn/problems/monotone-increasing-digits/)

- 标签：贪心、数学
- 难度：中等

## 题目链接

- [0738. 单调递增的数字 - 力扣](https://leetcode.cn/problems/monotone-increasing-digits/)

## 题目大意

给定一个非负整数 n，找出小于等于 n 的最大整数，同时该整数需要满足其各个位数上的数字是单调递增的。

## 解题思路

为了方便操作，我们先将整数 n 转为 list 数组，即 n_list。

题目要求这个整数尽可能的大，那么这个数从高位开始，就应该尽可能的保持不变。那么我们需要从高位到低位，找到第一个满足 `n_list[i - 1] > n_list[i]` 的位置，然后把 `n_list[i] - 1`，再把剩下的低位都变为 9。 

##  代码

```python
class Solution:
    def monotoneIncreasingDigits(self, n: int) -> int:
        n_list = list(str(n))
        size = len(n_list)
        start_i = size
        for i in range(size - 1, 0, -1):
            if n_list[i - 1] > n_list[i]:
                start_i = i
                n_list[i - 1] = chr(ord(n_list[i - 1]) - 1)

        for i in range(start_i, size, 1):
            n_list[i] = '9'
        res = int(''.join(n_list))
        return res
```

# [0739. 每日温度](https://leetcode.cn/problems/daily-temperatures/)

- 标签：栈、数组、单调栈
- 难度：中等

## 题目链接

- [0739. 每日温度 - 力扣](https://leetcode.cn/problems/daily-temperatures/)

## 题目大意

**描述**：给定一个列表 `temperatures`，`temperatures[i]` 表示第 `i` 天的气温。

**要求**：输出一个列表，列表上每个位置代表「如果要观测到更高的气温，至少需要等待的天数」。如果之后的气温不再升高，则用 `0` 来代替。

**说明**：

- $1 \le temperatures.length \le 10^5$。
- $30 \le temperatures[i] \le 100$。

**示例**：

- 示例 1：

```python
输入: temperatures = [73,74,75,71,69,72,76,73]
输出: [1,1,4,2,1,1,0,0]
```

- 示例 2：

```python
输入: temperatures = [30,40,50,60]
输出: [1,1,1,0]
```

## 解题思路

题目的意思实际上就是给定一个数组，每个位置上有整数值。对于每个位置，在该位置右侧找到第一个比当前元素更大的元素。求「该元素」与「右侧第一个比当前元素更大的元素」之间的距离，将所有距离保存为数组返回结果。

最简单的思路是对于每个温度值，向后依次进行搜索，找到比当前温度更高的值。

更好的方式使用「单调递增栈」，栈中保存元素的下标。

### 思路 1：单调栈

1. 首先，将答案数组 `ans` 全部赋值为 0。然后遍历数组每个位置元素。
2. 如果栈为空，则将当前元素的下标入栈。
3. 如果栈不为空，且当前数字大于栈顶元素对应数字，则栈顶元素出栈，并计算下标差。
4. 此时当前元素就是栈顶元素的下一个更高值，将其下标差存入答案数组 `ans` 中保存起来，判断栈顶元素。
5. 直到当前数字小于或等于栈顶元素，则停止出栈，将当前元素下标入栈。
6. 最后输出答案数组 `ans`。

### 思路 1：代码

```python
class Solution:
    def dailyTemperatures(self, T: List[int]) -> List[int]:
        n = len(T)
        stack = []
        ans = [0 for _ in range(n)]
        for i in range(n):
            while stack and T[i] > T[stack[-1]]:
                index = stack.pop()
                ans[index] = (i-index)
            stack.append(i)
        return ans
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(n)$。
- **空间复杂度**：$O(n)$。

# [0744. 寻找比目标字母大的最小字母](https://leetcode.cn/problems/find-smallest-letter-greater-than-target/)

- 标签：数组、二分查找
- 难度：简单

## 题目链接

- [0744. 寻找比目标字母大的最小字母 - 力扣](https://leetcode.cn/problems/find-smallest-letter-greater-than-target/)

## 题目大意

**描述**：给你一个字符数组 $letters$，该数组按非递减顺序排序，以及一个字符 $target$。$letters$ 里至少有两个不同的字符。

**要求**：找出 $letters$ 中大于 $target$ 的最小的字符。如果不存在这样的字符，则返回 $letters$ 的第一个字符。

**说明**：

- $2 \le letters.length \le 10^4$。
- $letters[i]$$ 是一个小写字母。
- $letters$ 按非递减顺序排序。
- $letters$ 最少包含两个不同的字母。
- $target$ 是一个小写字母。

**示例**：

- 示例 1：

```python
输入: letters = ["c", "f", "j"]，target = "a"
输出: "c"
解释：letters 中字典上比 'a' 大的最小字符是 'c'。
```

- 示例 2：

```python
输入: letters = ["c","f","j"], target = "c"
输出: "f"
解释：letters 中字典顺序上大于 'c' 的最小字符是 'f'。
```

## 解题思路

### 思路 1：二分查找

利用二分查找，找到比 $target$ 大的字母。注意 $target$ 可能大于 $letters$ 的所有字符，此时应返回 $letters$ 的第一个字母。

我们可以假定 $target$ 的取值范围为 $[0, len(letters)]$。当 $target$ 取到 $len(letters)$ 时，说明 $target$ 大于 $letters$ 的所有字符，对 $len(letters)$ 取余即可得到 $letters[0]$。

### 思路 1：代码

```python
class Solution:
    def nextGreatestLetter(self, letters: List[str], target: str) -> str:
        n = len(letters)
        left = 0
        right = n
        while left < right:
            mid = left + (right - left) // 2
            if letters[mid] <= target:
                left = mid + 1
            else:
                right = mid
        return letters[left % n]
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(n)$。其中 $n$ 为字符数组 $letters$ 的长度。
- **空间复杂度**：$O(1)$。

# [0746. 使用最小花费爬楼梯](https://leetcode.cn/problems/min-cost-climbing-stairs/)

- 标签：数组、动态规划
- 难度：简单

## 题目链接

- [0746. 使用最小花费爬楼梯 - 力扣](https://leetcode.cn/problems/min-cost-climbing-stairs/)

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
        for i in range(2, size+1):
            dp[i] = min(dp[i - 1] + cost[i - 1], dp[i - 2] + cost[i - 2])
        return dp[size]
```

# [0752. 打开转盘锁](https://leetcode.cn/problems/open-the-lock/)

- 标签：广度优先搜索、数组、哈希表、字符串
- 难度：中等

## 题目链接

- [0752. 打开转盘锁 - 力扣](https://leetcode.cn/problems/open-the-lock/)

## 题目大意

**描述**：有一把带有四个数字的密码锁，每个位置上有 `0` ~ `9` 共 `10` 个数字。每次只能将其中一个位置上的数字转动一下。可以向上转，也可以向下转。比如：`1 -> 2`、`2 -> 1`。

密码锁的初始数字为：`0000`。现在给定一组表示死亡数字的字符串数组 `deadends`，和一个带有四位数字的目标字符串 `target`。

如果密码锁转动到 `deadends` 中任一字符串状态，则锁就会永久锁定，无法再次旋转。

**要求**：给出使得锁的状态由 `0000` 转动到 `target` 的最小的选择次数。如果无论如何不能解锁，返回 `-1` 。

**说明**：

- $1 \le deadends.length \le 500$
  $deadends[i].length == 4$
  $target.length == 4$
  $target$ 不在 $deadends$ 之中
  $target$ 和 $deadends[i]$ 仅由若干位数字组成。

**示例**：

- 示例 1：

```python
输入：deadends = ["0201","0101","0102","1212","2002"], target = "0202"
输出：6
解释：
可能的移动序列为 "0000" -> "1000" -> "1100" -> "1200" -> "1201" -> "1202" -> "0202"。
注意 "0000" -> "0001" -> "0002" -> "0102" -> "0202" 这样的序列是不能解锁的，
因为当拨动到 "0102" 时这个锁就会被锁定。
```

- 示例 2：

```python
输入: deadends = ["8887","8889","8878","8898","8788","8988","7888","9888"], target = "8888"
输出：-1
解释：无法旋转到目标数字且不被锁定。
```

## 解题思路

### 思路 1：广度优先搜索

1. 定义 `visited` 为标记访问节点的 set 集合变量，`queue` 为存放节点的队列。
2. 将`0000` 状态标记为访问，并将其加入队列 `queue`。
3. 将当前队列中的所有状态依次出队，判断这些状态是否为死亡字符串。
   1. 如果为死亡字符串，则跳过该状态，否则继续执行。
   2. 如果为目标字符串，则返回当前路径长度，否则继续执行。

4. 枚举当前状态所有位置所能到达的所有状态（通过向上或者向下旋转），并判断是否访问过该状态。
5. 如果之前出现过该状态，则继续执行，否则将其存入队列，并标记访问。
6. 遍历完步骤 3 中当前队列中的所有状态，令路径长度加 `1`，继续执行 3 ~ 5 步，直到队列为空。
7. 如果队列为空，也未能到达目标状态，则返回 `-1`。

### 思路 1：代码

```python
import collections

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

### 思路 1：复杂度分析

- **时间复杂度**：$O(10^d \times d^2 + m \times d)$。其中 $d$ 是数字的位数，$m$ 是数组 $deadends$ 的长度。
- **空间复杂度**：$O(10^D \times d + m)$。
# [0758. 字符串中的加粗单词](https://leetcode.cn/problems/bold-words-in-string/)

- 标签：字典树、数组、哈希表、字符串、字符串匹配
- 难度：中等

## 题目链接

- [0758. 字符串中的加粗单词 - 力扣](https://leetcode.cn/problems/bold-words-in-string/)

## 题目大意

给定一个关键词集合 `words` 和一个字符串 `s`。

要求：在所有 `s` 中出现的关键词前后位置上添加加粗闭合标签 `<b>` 和 `</b>`。如果两个子串有重叠部分，则将它们一起用一对闭合标签包围起来。同理，如果两个子字符串连续被加粗，那么你也需要把它们合起来用一对加粗标签包围。最后返回添加加粗标签后的字符串 `s`。

## 解题思路

构建字典树，将字符串列表 `words` 中所有字符串添加到字典树中。

然后遍历字符串 `s`，从每一个位置开始查询字典树。在第一个符合要求的单词前面添加 `<b>`。在连续符合要求的单词中的最后一个单词后面添加 `</b>`。

最后返回添加加粗标签后的字符串 `s`。

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

class Solution:
    def boldWords(self, words: List[str], s: str) -> str:
        trie_tree = Trie()
        for word in words:
            trie_tree.insert(word)

        size = len(s)
        bold_left, bold_right = -1, -1
        ans = ""
        for i in range(size):
            cur = trie_tree
            if s[i] in cur.children:
                bold_left = i
                while bold_left < size and s[bold_left] in cur.children:
                    cur = cur.children[s[bold_left]]
                    bold_left += 1
                    if cur.isEnd:
                        if bold_right == -1:
                            ans += "<b>"
                        bold_right = max(bold_left, bold_right)
            if i == bold_right:
                ans += "</b>"
                bold_right = -1
            ans += s[i]
        if bold_right >= 0:
            ans += "</b>"
        return ans
```

# [0763. 划分字母区间](https://leetcode.cn/problems/partition-labels/)

- 标签：贪心、哈希表、双指针、字符串
- 难度：中等

## 题目链接

- [0763. 划分字母区间 - 力扣](https://leetcode.cn/problems/partition-labels/)

## 题目大意

给定一个由小写字母组成的字符串 `s`。要把这个字符串划分为尽可能多的片段，同一字母最多出现在一个片段中。

要求：返回一个表示每个字符串片段的长度的列表。

## 解题思路

因为同一字母最多出现在一个片段中，则同一字母第一次出现的下标位置和最后一次出现的下标位置肯定在同一个片段中。

我们先遍历一遍字符串，用哈希表 letter_map 存储下每一个字母最后一次出现的下标位置。

为了得到尽可能的片段，我们使用贪心的思想：

- 从头开始遍历字符串，遍历同时维护当前片段的开始位置 start 和结束位置 end。
- 对于字符串中的每个字符 `s[i]`，得到当前字母的最后一次出现的下标位置 `letter_map[s[i]]`，则当前片段的结束位置一定不会早于 `letter_map[s[i]]`，所以更新 end 值为 `end = max(end, letter_map[s[i]])`。
- 当访问到 `i == end` 时，当前片段访问结束，当前片段的下标范围为 `[start, end]`，长度为 `end - start + 1`，将其长度加入答案数组，并更新 start 值为 `i + 1`，继续遍历。
- 最终返回答案数组。

## 代码

```python
class Solution:
    def partitionLabels(self, s: str) -> List[int]:
        letter_map = dict()
        for i in range(len(s)):
            letter_map[s[i]] = i
        res = []
        start, end = 0, 0
        for i in range(len(s)):
            end = max(end, letter_map[s[i]])
            if i == end:
                res.append(end - start + 1)
                start = i + 1
        return res
```

# [0765. 情侣牵手](https://leetcode.cn/problems/couples-holding-hands/)

- 标签：贪心、深度优先搜索、广度优先搜索、并查集、图
- 难度：困难

## 题目链接

- [0765. 情侣牵手 - 力扣](https://leetcode.cn/problems/couples-holding-hands/)

## 题目大意

**描述**：$n$ 对情侣坐在连续排列的 $2 \times n$ 个座位上，想要牵对方的手。人和座位用 $0 \sim 2 \times n - 1$ 的整数表示。情侣按顺序编号，第一对是 $(0, 1)$，第二对是 $(2, 3)$，以此类推，最后一对是 $(2 \times n - 2, 2 \times n - 1)$。

给定代表情侣初始座位的数组 `row`，`row[i]` 表示第 `i` 个座位上的人的编号。

**要求**：计算最少交换座位的次数，以便每对情侣可以并肩坐在一起。每一次交换可以选择任意两人，让他们互换座位。

**说明**：

- $2 \times n == row.length$。
- $2 \le n \le 30$。
- $n$ 是偶数。
- $0 \le row[i] < 2 \times n$。
- $row$ 中所有元素均无重复。

**示例**：

- 示例 1：

```python
输入: row = [0,2,1,3]
输出: 1
解释: 只需要交换row[1]和row[2]的位置即可。
```

- 示例 2：

```python
输入: row = [3,2,0,1]
输出: 0
解释: 无需交换座位，所有的情侣都已经可以手牵手了。
```

## 解题思路

### 思路 1：并查集

先观察一下可以直接牵手的情侣特点：

- 编号一定相邻。
- 编号为一个奇数一个偶数。
- 偶数 + 1 = 奇数。

将每对情侣的编号 `(0, 1) (2, 3) (4, 5) ...` 除以 `2` 可以得到 `(0, 0) (1, 1) (2, 2) ...`，这样相同编号就代表是一对情侣。

1. 按照 `2` 个一组的顺序，遍历一下所有编号。
   1. 如果相邻的两人编号除以 `2` 相同，则两人是情侣，将其合并到一个集合中。
   2. 如果相邻的两人编号不同，则将其合并到同一个集合中，而这两个人分别都有各自的对象，所以在后续遍历中两个人各自的对象和他们同组上的另一个人一定都会并到统一集合中，最终形成一个闭环。比如 `(0, 1) (1, 3) (2, 0) (3, 2)`。假设闭环对数为 `k`，最少需要交换 `k  - 1` 次才能让情侣牵手。
2. 假设 `n` 对情侣中有 `m` 个闭环，则 `至少交换次数 = (n1 - 1) + (n2 - 1) + ... + (nn - 1) = n - m`。

### 思路 1：代码

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
        if root_x == root_y:
            return False
        self.parent[root_x] = root_y
        return True

    def is_connected(self, x, y):
        return self.find(x) == self.find(y)

class Solution:
    def minSwapsCouples(self, row: List[int]) -> int:
        size = len(row)
        n = size // 2
        count = n
        union_find = UnionFind(n)
        for i in range(0, size, 2):
            if union_find.union(row[i] // 2, row[i + 1] // 2):
                count -= 1
        return n - count
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(n \times \alpha(n))$。其中 $n$ 是数组  $row$ 长度，$\alpha$ 是反 `Ackerman` 函数。
- **空间复杂度**：$O(n)$。# [0766. 托普利茨矩阵](https://leetcode.cn/problems/toeplitz-matrix/)

- 标签：数组、矩阵
- 难度：简单

## 题目链接

- [0766. 托普利茨矩阵 - 力扣](https://leetcode.cn/problems/toeplitz-matrix/)

## 题目大意

**描述**：给定一个 $m \times n$ 大小的矩阵 $matrix$。

**要求**：如果 $matrix$ 是托普利茨矩阵，则返回 `True`；否则返回 `False`。

**说明**：

- **托普利茨矩阵**：矩阵上每一条由左上到右下的对角线上的元素都相同。
- $m == matrix.length$。
- $n == matrix[i].length$。
- $1 \le m, n \le 20$。
- $0 \le matrix[i][j] \le 99$。

**示例**：

- 示例 1：

![](https://assets.leetcode.com/uploads/2020/11/04/ex1.jpg)

```python
输入：matrix = [[1,2,3,4],[5,1,2,3],[9,5,1,2]]
输出：true
解释：
在上述矩阵中, 其对角线为: 
"[9]", "[5, 5]", "[1, 1, 1]", "[2, 2, 2]", "[3, 3]", "[4]"。 
各条对角线上的所有元素均相同, 因此答案是 True。
```

- 示例 2：

![](https://assets.leetcode.com/uploads/2020/11/04/ex2.jpg)

```python
输入：matrix = [[1,2],[2,2]]
输出：false
解释：
对角线 "[1, 2]" 上的元素不同。
```

## 解题思路

### 思路 1：简单模拟

1. 两层循环遍历矩阵，依次判断矩阵当前位置 $(i, j)$ 上的值 $matrix[i][j]$ 与其左上角位置 $(i - 1, j - 1)$ 位置上的值 $matrix[i - 1][j - 1]$ 是否相等。
2. 如果不相等，则返回 `False`。
3. 遍历完，则返回 `True`。

### 思路 1：代码

```python
class Solution:
    def isToeplitzMatrix(self, matrix: List[List[int]]) -> bool:
        for i in range(1, len(matrix)):
            for j in range(1, len(matrix[0])):
                if matrix[i][j] != matrix[i - 1][j - 1]:
                    return False
        return True
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(m \times n)$，其中 $m$、$n$ 分别是矩阵 $matrix$ 的行数、列数。
- **空间复杂度**：$O(m \times n)$。
# [0771. 宝石与石头](https://leetcode.cn/problems/jewels-and-stones/)

- 标签：哈希表、字符串
- 难度：简单

## 题目链接

- [0771. 宝石与石头 - 力扣](https://leetcode.cn/problems/jewels-and-stones/)

## 题目大意

**描述**：给定一个字符串 $jewels$ 代表石头中宝石的类型，再给定一个字符串 $stones$ 代表你拥有的石头。$stones$ 中每个字符代表了一种你拥有的石头的类型。

**要求**：计算出拥有的石头中有多少是宝石。

**说明**：

- 字母区分大小写，因此 $a$ 和 $A$ 是不同类型的石头。
- $1 \le jewels.length, stones.length \le 50$。
- $jewels$ 和 $stones$ 仅由英文字母组成。
- $jewels$ 中的所有字符都是唯一的。

**示例**：

- 示例 1：

```python
输入：jewels = "aA", stones = "aAAbbbb"
输出：3
```

- 示例 2：

```python
输入：jewels = "z", stones = "ZZ"
输出：0
```

## 解题思路

### 思路 1：哈希表

1. 用 $count$ 来维护石头中的宝石个数。
2. 先使用哈希表或者集合存储宝石。
3. 再遍历数组 $stones$，并统计每块石头是否在哈希表中或集合中。
   1. 如果当前石头在哈希表或集合中，则令 $count$ 加 $1$。
   2. 如果当前石头不在哈希表或集合中，则不统计。
4. 最后返回 $count$。

### 思路 1：代码

```python
class Solution:
    def numJewelsInStones(self, jewels: str, stones: str) -> int:
        jewel_dict = dict()
        for jewel in jewels:
            jewel_dict[jewel] = 1
        count = 0
        for stone in stones:
            if stone in jewel_dict:
                count += 1
        return count
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(m + n)$，其中 $m$ 是字符串 $jewels$ 的长度，$n$ 是 $stones$ 的长度。
- **空间复杂度**：$O(m)$，其中 $m$ 是字符串 $jewels$ 的长度。

# [0778. 水位上升的泳池中游泳](https://leetcode.cn/problems/swim-in-rising-water/)

- 标签：深度优先搜索、广度优先搜索、并查集、数组、二分查找、矩阵、堆（优先队列）
- 难度：困难

## 题目链接

- [0778. 水位上升的泳池中游泳 - 力扣](https://leetcode.cn/problems/swim-in-rising-water/)

## 题目大意

**描述**：给定一个 $n \times n$ 大小的二维数组 $grid$，每一个方格的值 $grid[i][j]$ 表示为位置 $(i, j)$ 的高度。

现在要从左上角 $(0, 0)$ 位置出发，经过方格的一些点，到达右下角 $(n - 1, n - 1)$  位置上。其中所经过路径的花费为这条路径上所有位置的最大高度。

**要求**：计算从 $(0, 0)$ 位置到 $(n - 1, n - 1)$  的最优路径的花费。

**说明**：

- **最优路径**：路径上最大高度最小的那条路径。
- $n == grid.length$。
- $n == grid[i].length$。
- $1 \le n \le 50$。
- $0 \le grid[i][j] < n2$。
- $grid[i][j]$ 中每个值均无重复。

**示例**：

- 示例 1：

![](https://assets.leetcode.com/uploads/2021/06/29/swim1-grid.jpg)

```python
输入: grid = [[0,2],[1,3]]
输出: 3
解释:
时间为 0 时，你位于坐标方格的位置为 (0, 0)。
此时你不能游向任意方向，因为四个相邻方向平台的高度都大于当前时间为 0 时的水位。
等时间到达 3 时，你才可以游向平台 (1, 1). 因为此时的水位是 3，坐标方格中的平台没有比水位 3 更高的，所以你可以游向坐标方格中的任意位置。
```

- 示例 2：

![](https://assets.leetcode.com/uploads/2021/06/29/swim2-grid-1.jpg)

```python
输入: grid = [[0,1,2,3,4],[24,23,22,21,5],[12,13,14,15,16],[11,17,18,19,20],[10,9,8,7,6]]
输出: 16
解释: 最终的路线用加粗进行了标记。
我们必须等到时间为 16，此时才能保证平台 (0, 0) 和 (4, 4) 是连通的。
```

## 解题思路

### 思路 1：并查集

将整个网络抽象为一个无向图，每个点与相邻的点（上下左右）之间都存在一条无向边，边的权重为两个点之间的最大高度。

我们要找到左上角到右下角的最优路径，可以遍历所有的点，将所有的边存储到数组中，每条边的存储格式为 $[x, y, h]$，意思是编号 $x$ 的点和编号为 $y$ 的点之间的权重为 $h$。

然后按照权重从小到大的顺序，对所有边进行排序。

再按照权重大小遍历所有边，将其依次加入并查集中。并且每次都需要判断 $(0, 0)$ 点和 $(n - 1, n - 1)$ 点是否连通。

如果连通，则该边的权重即为答案。

### 思路 1：代码

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
    def swimInWater(self, grid: List[List[int]]) -> int:
        row_size = len(grid)
        col_size = len(grid[0])
        size = row_size * col_size
        edges = []
        for row in range(row_size):
            for col in range(col_size):
                if row < row_size - 1:
                    x = row * col_size + col
                    y = (row + 1) * col_size + col
                    h = max(grid[row][col], grid[row + 1][col])
                    edges.append([x, y, h])
                if col < col_size - 1:
                    x = row * col_size + col
                    y = row * col_size + col + 1
                    h = max(grid[row][col], grid[row][col + 1])
                    edges.append([x, y, h])

        edges.sort(key=lambda x: x[2])

        union_find = UnionFind(size)

        for edge in edges:
            x, y, h = edge[0], edge[1], edge[2]
            union_find.union(x, y)
            if union_find.is_connected(0, size - 1):
                return h
        return 0
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(m \times n \times \alpha(m \times n))$，其中 $\alpha$ 是反 Ackerman 函数。
- **空间复杂度**：$O(m \times n)$。

# [0779. 第K个语法符号](https://leetcode.cn/problems/k-th-symbol-in-grammar/)

- 标签：位运算、递归、数学
- 难度：中等

## 题目链接

- [0779. 第K个语法符号 - 力扣](https://leetcode.cn/problems/k-th-symbol-in-grammar/)

## 题目大意

**描述**：给定两个整数 $n$ 和 $k$​。我们可以按照下面的规则来生成字符串：

- 第一行写上一个 $0$。
- 从第二行开始，每一行将上一行的 $0$ 替换成 $01$，$1$ 替换为 $10$。

**要求**：输出第 $n$ 行字符串中的第 $k$ 个字符。

**说明**：

- $1 \le n \le 30$。
- $1 \le k \le 2^{n - 1}$。

**示例**：

- 示例 1：

```python
输入: n = 2, k = 1
输出: 0
解释: 
第一行: 0 
第二行: 01
```

- 示例 2：

```python
输入: n = 4, k = 4
输出: 0
解释: 
第一行：0
第二行：01
第三行：0110
第四行：01101001
```

## 解题思路

### 思路 1：递归算法 + 找规律

每一行都是由上一行生成的。我们可以将多行写到一起找下规律。

可以发现：第 $k$ 个数字是由上一位对应位置上的数字生成的。 

- $k$ 在奇数位时，由上一行 $(k + 1) / 2$ 位置的值生成。且与上一行 $(k + 1) / 2$ 位置的值相同；
- $k$ 在偶数位时，由上一行 $k / 2$ 位置的值生成。且与上一行 $k / 2$ 位置的值相反。

接下来就是递归求解即可。

### 思路 1：代码

```python
class Solution:
    def kthGrammar(self, n: int, k: int) -> int:
        if n == 0:
            return 0
        if k % 2 == 1:
            return self.kthGrammar(n - 1, (k + 1) // 2)
        else:
            return abs(self.kthGrammar(n - 1, k // 2) - 1)
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(n)$。
- **空间复杂度**：$O(n)$。

# [0783. 二叉搜索树节点最小距离](https://leetcode.cn/problems/minimum-distance-between-bst-nodes/)

- 标签：树、深度优先搜索、广度优先搜索、二叉搜索树、二叉树
- 难度：简单

## 题目链接

- [0783. 二叉搜索树节点最小距离 - 力扣](https://leetcode.cn/problems/minimum-distance-between-bst-nodes/)

## 题目大意

**描述**：给定一个二叉搜索树的根节点 $root$。

**要求**：返回树中任意两不同节点值之间的最小差值。

**说明**：

- **差值**：是一个正数，其数值等于两值之差的绝对值。
- 树中节点的数目范围是 $[2, 100]$。
- $0 \le Node.val \le 10^5$。

**示例**：

- 示例 1：

![](https://assets.leetcode.com/uploads/2021/02/05/bst1.jpg)

```python
输入：root = [4,2,6,1,3]
输出：1
```

- 示例 2：

![](https://assets.leetcode.com/uploads/2021/02/05/bst2.jpg)

```python
输入：root = [1,0,48,null,null,12,49]
输出：1
```

## 解题思路

### 思路 1：中序遍历

先来看二叉搜索树的定义：

- 若左子树不为空，则左子树上所有节点值均小于它的根节点值；
- 若右子树不为空，则右子树上所有节点值均大于它的根节点值；
- 任意节点的左、右子树也分别为二叉搜索树。

题目要求二叉搜索树上任意两节点的差的绝对值的最小值。

二叉树的中序遍历顺序是：左 -> 根 -> 右，二叉搜索树的中序遍历最终得到就是一个升序数组。而升序数组中绝对值差的最小值就是比较相邻两节点差值的绝对值，找出其中最小值。

那么我们就可以先对二叉搜索树进行中序遍历，并保存中序遍历的结果。然后再比较相邻节点差值的最小值，从而找出最小值。

### 思路 1：代码

```Python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def inorderTraversal(self, root: TreeNode) -> List[int]:
        res = []
        def inorder(root):
            if not root:
                return
            inorder(root.left)
            res.append(root.val)
            inorder(root.right)

        inorder(root)
        return res

    def minDiffInBST(self, root: Optional[TreeNode]) -> int:
        inorder = self.inorderTraversal(root)
        ans = float('inf')
        for i in range(1, len(inorder)):
            ans = min(ans, abs(inorder[i - 1] - inorder[i]))

        return ans
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(n)$，其中 $n$ 为二叉搜索树中的节点数量。
- **空间复杂度**：$O(n)$。

# [0784. 字母大小写全排列](https://leetcode.cn/problems/letter-case-permutation/)

- 标签：位运算、字符串、回溯
- 难度：中等

## 题目链接

- [0784. 字母大小写全排列 - 力扣](https://leetcode.cn/problems/letter-case-permutation/)

## 题目大意

**描述**：给定一个字符串 $s$，通过将字符串 $s$ 中的每个字母转变大小写，我们可以获得一个新的字符串。

**要求**：返回所有可能得到的字符串集合。

**说明**：

- 答案可以以任意顺序返回输出。
- $1 \le s.length \le 12$。
- $s$ 由小写英文字母、大写英文字母和数字组成。

**示例**：

- 示例 1：

```python
输入：s = "a1b2"
输出：["a1b2", "a1B2", "A1b2", "A1B2"]
```

- 示例 2：

```python
输入: s = "3z4"
输出: ["3z4","3Z4"]
```

## 解题思路

### 思路 1：回溯算法

- $i$ 代表当前要处理的字符在字符串 $s$ 中的下标，$path$ 表示当前路径，$ans$ 表示答案数组。
- 如果处理到 $i == len(s)$ 时，将当前路径存入答案数组中返回，否则进行递归处理。
  - 不修改当前字符，直接递归处理第 $i + 1$ 个字符。
  - 如果当前字符是小写字符，则变为大写字符之后，递归处理第 $i + 1$ 个字符。
  - 如果当前字符是大写字符，则变为小写字符之后，递归处理第 $i + 1$ 个字符。

### 思路 1：代码

```python
class Solution:
    def dfs(self, s, path, i, ans):
        if i == len(s):
            ans.append(path)
            return

        self.dfs(s, path + s[i], i + 1, ans)
        if ord('a') <= ord(s[i]) <= ord('z'):
            self.dfs(s, path + s[i].upper(), i + 1, ans)
        elif ord('A') <= ord(s[i]) <= ord('Z'):
            self.dfs(s, path + s[i].lower(), i + 1, ans)

    def letterCasePermutation(self, s: str) -> List[str]:
        ans, path = [], ""
        self.dfs(s, path, 0, ans)
        return ans
```

### 思路 1：复杂度分析

- **时间复杂度**：$n \times 2^n$，其中 $n$ 为字符串的长度。
- **空间复杂度**：$O(1)$，除返回值外不需要额外的空间。

# [0785. 判断二分图](https://leetcode.cn/problems/is-graph-bipartite/)

- 标签：深度优先搜索、广度优先搜索、并查集、图
- 难度：中等

## 题目链接

- [0785. 判断二分图 - 力扣](https://leetcode.cn/problems/is-graph-bipartite/)

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

# [0788. 旋转数字](https://leetcode.cn/problems/rotated-digits/)

- 标签：数学、动态规划
- 难度：中等

## 题目链接

- [0788. 旋转数字 - 力扣](https://leetcode.cn/problems/rotated-digits/)

## 题目大意

**描述**：给定搞一个正整数 $n$。

**要求**：计算从 $1$ 到 $n$ 中有多少个数 $x$ 是好数。

**说明**：

- **好数**：如果一个数 $x$ 的每位数字逐个被旋转 180 度之后，我们仍可以得到一个有效的，且和 $x$ 不同的数，则成该数为好数。
- 如果一个数的每位数字被旋转以后仍然还是一个数字， 则这个数是有效的。$0$、$1$ 和 $8$ 被旋转后仍然是它们自己；$2$ 和 $5$ 可以互相旋转成对方（在这种情况下，它们以不同的方向旋转，换句话说，$2$ 和 $5$ 互为镜像）；$6$ 和 $9$ 同理，除了这些以外其他的数字旋转以后都不再是有效的数字。
- $n$ 的取值范围是 $[1, 10000]$。

**示例**：

- 示例 1：

```python
输入: 10
输出: 4
解释: 
在 [1, 10] 中有四个好数： 2, 5, 6, 9。
注意 1 和 10 不是好数, 因为他们在旋转之后不变。
```

## 解题思路

### 思路 1：枚举算法

根据题目描述，一个数满足：数中没有出现 $3$、$4$、$7$，并且至少出现一次 $2$、$5$、$6$ 或 $9$，就是好数。

因此，我们可以枚举 $[1, n]$ 中的每一个正整数 $x$，并判断该正整数 $x$ 的数位中是否满足没有出现 $3$、$4$、$7$，并且至少一次出现了 $2$、$5$、$6$ 或 $9$，如果满足，则该正整数 $x$ 位好数，否则不是好数。

最后统计好数的方案个数并将其返回即可。

### 思路 1：代码

```python
class Solution:
    def rotatedDigits(self, n: int) -> int:
        check = [0, 0, 1, -1, -1, 1, 1, -1, 0, 1]
        ans = 0
        for i in range(1, n + 1):
            flag = False
            num = i
            while num:
                digit = num % 10
                num //= 10
                if check[digit] == 1:
                    flag = True
                elif check[digit] == -1:
                    flag = False
                    break
            if flag:
                ans += 1
            	
        return ans
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(n \times \log n)$。
- **空间复杂度**：$O(\log n)$。

### 思路 2：动态规划 + 数位 DP

将 $n$ 转换为字符串 $s$，定义递归函数 `def dfs(pos, hasDiff, isLimit):` 表示构造第 $pos$ 位及之后所有数位的合法方案数。其中：

1. $pos$ 表示当前枚举的数位位置。
2. $hasDiff$ 表示当前是否用到 $2$、$5$、$6$ 或 $9$ 中任何一个数字。
3. $isLimit$ 表示前一位数位是否等于上界，用于限制本次搜索的数位范围。

接下来按照如下步骤进行递归。

1. 从 `dfs(0, False, True)` 开始递归。 `dfs(0, False, True)` 表示：
   1. 从位置 $0$ 开始构造。
   2. 初始没有用到 $2$、$5$、$6$ 或 $9$ 中任何一个数字。
   3. 开始时受到数字 $n$ 对应最高位数位的约束。
2. 如果遇到  $pos == len(s)$，表示到达数位末尾，此时：
   1. 如果 $hasDiff == True$，说明当前方案符合要求，则返回方案数 $1$。
   2. 如果 $hasDiff == False$，说明当前方案不符合要求，则返回方案数 $0$。
3. 如果 $pos \ne len(s)$，则定义方案数 $ans$，令其等于 $0$，即：`ans = 0`。
4. 因为不需要考虑前导 $0$，所以当前所能选择的最小数字 $minX$ 为 $0$。
5. 根据 $isLimit$ 来决定填当前位数位所能选择的最大数字（$maxX$）。
6. 然后根据 $[minX, maxX]$ 来枚举能够填入的数字 $d$。
7. 如果当前数位与之前数位没有出现 $3$、$4$、$7$，则方案数累加上当前位选择 $d$ 之后的方案数，即：`ans += dfs(pos + 1, hasDiff or check[d], isLimit and d == maxX)`。
   1. `hasDiff or check[d]` 表示当前是否用到 $2$、$5$、$6$ 或 $9$ 中任何一个数字或者没有用到 $3$、$4$、$7$。
   2. `isLimit and d == maxX` 表示 $pos + 1$ 位受到之前位限制和 $pos$ 位限制。
8. 最后的方案数为 `dfs(0, False, True)`，将其返回即可。

### 思路 2：代码

```python
class Solution:
    def rotatedDigits(self, n: int) -> int:
        check = [0, 0, 1, -1, -1, 1, 1, -1, 0, 1]

        # 将 n 转换为字符串 s
        s = str(n)
        
        @cache
        # pos: 第 pos 个数位
        # hasDiff: 之前选过的数字是否包含 2,5,6,9 中至少一个。
        # isLimit: 表示是否受到选择限制。如果为真，则第 pos 位填入数字最多为 s[pos]；如果为假，则最大可为 9。
        def dfs(pos, hasDiff, isLimit):
            if pos == len(s):
                # isNum 为 True，则表示当前方案符合要求
                return int(hasDiff)
            
            ans = 0
            # 不需要考虑前导 0，则最小可选择数字为 0
            minX = 0
            # 如果受到选择限制，则最大可选择数字为 s[pos]，否则最大可选择数字为 9。
            maxX = int(s[pos]) if isLimit else 9
            
            # 枚举可选择的数字
            for d in range(minX, maxX + 1): 
                # d 不在选择的数字集合中，即之前没有选择过 d
                if check[d] != -1:
                    ans += dfs(pos + 1, hasDiff or check[d], isLimit and d == maxX)
            return ans
    
        return dfs(0, False, True)
```

### 思路 2：复杂度分析

- **时间复杂度**：$O(\log n)$。
- **空间复杂度**：$O(\log n)$。

# [0795. 区间子数组个数](https://leetcode.cn/problems/number-of-subarrays-with-bounded-maximum/)

- 标签：数组、双指针
- 难度：中等

## 题目链接

- [0795. 区间子数组个数 - 力扣](https://leetcode.cn/problems/number-of-subarrays-with-bounded-maximum/)

## 题目大意

给定一个元素都是正整数的数组`A` ，正整数 `L` 以及 `R` (`L <= R`)。

求连续、非空且其中最大元素满足大于等于`L` 小于等于`R`的子数组个数。

## 解题思路

最大元素满足大于等于`L` 小于等于`R`的子数组个数 = 最大元素小于等于 `R` 的子数组个数 - 最大元素小于 `L` 的子数组个数。

其中「最大元素小于 `L` 的子数组个数」也可以转变为「最大元素小于等于 `L - 1` 的子数组个数」。那么现在的问题就变为了如何计算最大元素小于等于 `k` 的子数组个数。

我们使用 `count` 记录 小于等于 `k` 的连续元素数量，遍历一遍数组，如果遇到 `nums[i] <= k` 时，`count` 累加，表示在此位置上结束的有效子数组数量为 `count + 1`。如果遇到 `nums[i] > k` 时，`count` 重新开始计算。每次遍历完将有效子数组数量累加到答案中。

## 代码

```python
class Solution:
    def numSubarrayMaxK(self, nums, k):
        ans = 0
        count = 0
        for i in range(len(nums)):
            if nums[i] <= k:
                count += 1
            else:
                count = 0
            ans += count
        return ans

    def numSubarrayBoundedMax(self, nums: List[int], left: int, right: int) -> int:
        return self.numSubarrayMaxK(nums, right) - self.numSubarrayMaxK(nums, left - 1)
```

# [0796. 旋转字符串](https://leetcode.cn/problems/rotate-string/)

- 标签：字符串、字符串匹配
- 难度：简单

## 题目链接

- [0796. 旋转字符串 - 力扣](https://leetcode.cn/problems/rotate-string/)

## 题目大意

**描述**：给定两个字符串 `s` 和 `goal`。

**要求**：如果 `s` 在若干次旋转之后，能变为 `goal`，则返回 `True`，否则返回 `False`。

**说明**：

- `s` 的旋转操作：将 `s` 最左侧的字符移动到最右边。
  - 比如：`s = "abcde"`，在旋转一次之后结果就是 `s = "bcdea"`。
- $1 \le s.length, goal.length \le 100$。
- `s` 和 `goal` 由小写英文字母组成。

**示例**：

- 示例 1：

```python
输入: s = "abcde", goal = "cdeab"
输出: true
```

- 示例 2：

```python
输入: s = "abcde", goal = "abced"
输出: false
```

## 解题思路

### 思路 1：KMP 算法

其实将两个字符串 `s` 拼接在一起，就包含了所有从 `s` 进行旋转后的字符串。那么我们只需要判断一下 `goal` 是否为 `s + s` 的子串即可。可以用 KMP 算法来做。

1. 先排除掉几种不可能的情况，比如 `s` 为空串的情况，`goal` 为空串的情况，`len(s) != len(goal)` 的情况。
2. 然后使用 KMP 算法计算出 `goal` 在 `s + s` 中的下标位置 `index`（`s + s` 可用取余运算模拟）。
3. 如果 `index == -1`，则说明 `s` 在若干次旋转之后，不能能变为 `goal`，则返回 `False`。
4. 如果 `index != -1`，则说明 `s` 在若干次旋转之后，能变为 `goal`，则返回 `True`。

### 思路 1：代码

```python
class Solution:
    def kmp(self, T: str, p: str) -> int:
        n, m = len(T), len(p)

        next = self.generateNext(p)

        i, j = 0, 0
        while i - j < n:
            while j > 0 and T[i % n] != p[j]:
                j = next[j - 1]
            if T[i % n] == p[j]:
                j += 1
            if j == m:
                return i - m + 1
            i += 1
        return -1

    def generateNext(self, p: str):
        m = len(p)
        next = [0 for _ in range(m)]

        left = 0
        for right in range(1, m):
            while left > 0 and p[left] != p[right]:
                left = next[left - 1]
            if p[left] == p[right]:
                left += 1
            next[right] = left

        return next

    def rotateString(self, s: str, goal: str) -> bool:
        if not s or not goal or len(s) != len(goal):
            return False
        index = self.kmp(s, goal)
        if index == -1:
            return False
        return True
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(n + m)$，其中文本串 $s$ 的长度为 $n$，模式串 $goal$ 的长度为 $m$。
- **空间复杂度**：$O(m)$。
# [0797. 所有可能的路径](https://leetcode.cn/problems/all-paths-from-source-to-target/)

- 标签：深度优先搜索、广度优先搜索、图、回溯
- 难度：中等

## 题目链接

- [0797. 所有可能的路径 - 力扣](https://leetcode.cn/problems/all-paths-from-source-to-target/)

## 题目大意

给定一个有 `n` 个节点的有向无环图（DAG），用二维数组 `graph` 表示。

要求：找出所有从节点 `0` 到节点 `n - 1` 的路径并输出（不要求按特定顺序）。

二维数组 `graph` 的第 `i` 个数组 `graph[i]` 中的单元都表示有向图中 `i` 号节点所能到达的下一个节点，如果为空就是没有下一个结点了。

## 解题思路

从第 `0` 个节点开始进行深度优先搜索遍历。在遍历的同时，通过回溯来寻找所有路径。具体做法如下：

- 使用 `ans` 数组存放所有答案路径，使用 `path` 数组记录当前路径。
- 从第 `0` 个节点开始进行深度优先搜索遍历。
  - 如果当前开始节点 `start` 等于目标节点 `target`。则将当前路径 `path` 添加到答案数组 `ans` 中，并返回。
  - 然后遍历当前节点 `start` 所能达到的下一个节点。
    - 将下一个节点加入到当前路径中。
    - 从该节点出发进行深度优先搜索遍历。
    - 然后将下一个节点从当前路径中移出，进行回退操作。
- 最后返回答案数组 `ans`。

## 代码

```python
class Solution:
    def dfs(self, graph, start, target, path, ans):
        if start == target:
            ans.append(path[:])
            return
        for end in graph[start]:
            path.append(end)
            self.dfs(graph, end, target, path, ans)
            path.remove(end)

    def allPathsSourceTarget(self, graph: List[List[int]]) -> List[List[int]]:
        path = [0]
        ans = []
        self.dfs(graph, 0, len(graph) - 1, path, ans)
        return ans
```

# [0800. 相似 RGB 颜色](https://leetcode.cn/problems/similar-rgb-color/)

- 标签：数学、字符串、枚举
- 难度：简单

## 题目链接

- [0800. 相似 RGB 颜色 - 力扣](https://leetcode.cn/problems/similar-rgb-color/)

## 题目大意

**描述**：RGB 颜色 `"#AABBCC"` 可以简写成 `"#ABC"` 。例如，`"#1155cc"` 可以简写为 `"#15c"`。现在给定一个按 `"#ABCDEF"` 形式定义的字符串 `color` 表示 RGB 颜色。

**要求**：返回一个与 `color` 相似度最大并且可以简写的颜色。

**说明**：

- 两个颜色 `"#ABCDEF"` 和 `"#UVWXYZ"` 的相似度计算公式为：$-(AB - UV)^2 - (CD - WX)^2 - (EF - YZ)^2$。

**示例**：

- 示例 1：

```python
输入 color = "#09f166"
输出 "#11ee66"
解释： 因为相似度计算得出 -(0x09 - 0x11)^2 -(0xf1 - 0xee)^2 - (0x66 - 0x66)^2 = -64 -9 -0 = -73，这是所有可以简写的颜色中与 color 最相似的颜色
```

## 解题思路

### 思路 1：枚举算法

所有可以简写的颜色范围是 `"#000"` ~ `"#fff"`，共 $16^3 = 4096$ 种颜色。因此，我们可以枚举这些可以简写的颜色，并计算出其与 $color$的相似度，从而找出与 $color$ 最相似的颜色。具体做法如下：

- 将  $color$ 转换为十六进制数，即 `hex_color = int(color[1:], 16)`。
- 三重循环遍历 $R$、$G$、$B$ 三个通道颜色，每一重循环范围为 $0 \sim 15$。
- 计算出每一种可以简写的颜色对应的十六进制，即 $17 \times R \times (1 << 16) + 17 \times G \times (1 << 8) + 17 \times B$，$17$ 是 $0x11 = 16 + 1 = 17$，$(1 << 16)$ 为 $R$ 左移的位数，$17 \times R \times (1 << 16)$ 就表示 $R$ 通道上对应的十六进制数。$(1 << 8)$ 为 $G$ 左移的位数，$17 \times G \times (1 << 8)$ 就表示 $G$ 通道上对应的十六进制数。$17 \times B$ 就表示 $B$ 通道上对应的十六进制数。
- 然后我们根据 $color$ 的十六进制数，与每一个可以简写的颜色对应的十六进制数，计算出相似度，并找出大相似对应的颜色。将其转换为字符串，并输出。

### 思路 1：枚举算法代码

```python
class Solution:
    def similar(self, hex1, hex2):
        r1, g1, b1 = hex1 >> 16, (hex1 >> 8) % 256, hex1 % 256
        r2, g2, b2 = hex2 >> 16, (hex2 >> 8) % 256, hex2 % 256
        return - (r1 - r2) ** 2 - (g1 - g2) ** 2 - (b1 - b2) ** 2

    def similarRGB(self, color: str) -> str:
        ans = 0
        hex_color = int(color[1:], 16)
        for r in range(16):
            for g in range(16):
                for b in range(16):
                    hex_cur = 17 * r * (1 << 16) + 17 * g * (1 << 8) + 17 * b
                    if self.similar(hex_color, hex_cur) > self.similar(hex_color, ans):
                        ans = hex_cur
        
        return "#{:06x}".format(ans)
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(16^3)$。
- **空间复杂度**：$O(1)$。
# [0801. 使序列递增的最小交换次数](https://leetcode.cn/problems/minimum-swaps-to-make-sequences-increasing/)

- 标签：数组、动态规划
- 难度：困难

## 题目链接

- [0801. 使序列递增的最小交换次数 - 力扣](https://leetcode.cn/problems/minimum-swaps-to-make-sequences-increasing/)

## 题目大意

给定两个长度相等的整形数组 A 和 B。可以交换两个数组相同位置上的元素，比如 A[i] 与 B[i] 交换，可以交换多个位置，但要保证交换之后保证数组 A和数组 B 是严格递增的。

要求：返回使得数组 A和数组 B 保持严格递增状态的最小交换次数。假设给定的输入一定有效。

## 解题思路

可以用动态规划来做。

对于两个数组每一个位置上的元素 A[i] 和 B[i] 来说，只有两种情况：换或者不换。

动态规划的状态 `dp[i][j]` 表示为：第 i 个位置元素，不交换（j = 0）、交换（j = 1）状态时的最小交换次数。

如果数组元素个数只有一个，则：

- `dp[0][0] = 0` ，第 0 个元素不做交换，交换次数为 0。
- `dp[0][1] = 1`，第 0 个元素做交换，交换次数为 1。

如果有 2 个元素，为了保证两个数组中的相邻元素都为递增元素，则第 2 个元素交换与否与第 1 个元素有关。同理如果有多个元素，那么第 i 个元素交换与否，只与第 i - 1 个元素有关。现在来考虑第 i 个元素与第 i - 1 的元素的情况。

先按原本数组当前是否满足递增关系来划分，可以划分为：

- 原本数组都满足递增关系，即 `A[i - 1] < A[i]` 并且 `B[i - 1] < B[i]`。
- 不满足上述递增关系的情况，即 `A[i - 1] >= A[i]` 或者 `B[i - 1] >= B[i]`。

可以看出，不满足递增关系的情况下是肯定要交换的。只需要考虑交换第 i 位元素，还是第 i - 1 位元素。

- `dp[i][0] = dp[i - 1][1]`，第 i 位若不交换，则第 i - 1 位必须交换。
- `dp[i][1] = dp[i - 1][0] + 1`，第 i 位交换，则第 i - 1 位不能交换。

下面再来考虑原本数组都满足递增关系的情况。考虑两个数组间相邻元素的关系。

-  `A[i - 1] < B[i]` 并且 `B[i - 1] < A[i]`。
-  `A[i - 1] >= B[i]` 或者 `B[i - 1] >= A[i]`。

如果是 `A[i - 1] < B[i]` 并且 `B[i - 1] < A[i]` 情况下，第 i 位交换，与第 i - 1 位交换与否无关，则 `dp[i][j]` 只需取 `dp[i-1][j]` 上较小结果进行计算即可，即：

- `dp[i][0] = min(dp[i-1][0], dp[i-1][1])`
- `dp[i][1] = min(dp[i-1][0], dp[i-1][1]) + 1`

如果是 `A[i - 1] >= B[i]` 或者 `B[i - 1] >= A[i]` 情况下，则如果第 i 位交换，则第 i - 1 位必须跟着交换。如果第 i 位不交换，则第 i - 1 为也不能交换，即：

- `dp[i][0] = dp[i - 1][0]`，如果第 i 位不交换，则第 i - 1 位也不交换。
- `dp[i][1] = dp[i - 1][1] + 1`，如果第 i 位交换，则第 i - 1 位也必须交换。

这样就考虑了所有的情况，最终返回最后一个元素，（交换、不交换）状态下的最小值即可。

## 代码

```python
class Solution:
    def minSwap(self, nums1: List[int], nums2: List[int]) -> int:
        size = len(nums1)
        dp = [[0 for _ in range(size)] for _ in range(size)]
        dp[0][1] = 1
        for i in range(1, size):
            if nums1[i - 1] < nums1[i] and nums2[i - 1] < nums2[i]:
                if nums1[i - 1] < nums2[i] and nums2[i - 1] < nums1[i]:
                    # 第 i 位交换，与第 i - 1 位交换与否无关
                    dp[i][0] = min(dp[i-1][0], dp[i-1][1])
                    dp[i][1] = min(dp[i-1][0], dp[i-1][1]) + 1
                else:
                    # 如果第 i 位不交换，则第 i - 1 位也不交换
                    # 如果第 i 位交换，则第 i - 1 位也必须交换
                    dp[i][0] = dp[i - 1][0]
                    dp[i][1] = dp[i - 1][1] + 1
            else:
                dp[i][0] = dp[i - 1][1]         # 如果第 i 位若不交换，则第 i - 1 位必须交换
                dp[i][1] = dp[i - 1][0] + 1     # 如果第 i 位交换，则第 i - 1 位不能交换
        return min(dp[size - 1][0], dp[size - 1][1])
```

# [0802. 找到最终的安全状态](https://leetcode.cn/problems/find-eventual-safe-states/)

- 标签：深度优先搜索、广度优先搜索、图、拓扑排序
- 难度：中等

## 题目链接

- [0802. 找到最终的安全状态 - 力扣](https://leetcode.cn/problems/find-eventual-safe-states/)

## 题目大意

**描述**：给定一个有向图 $graph$，其中 $graph[i]$ 是与节点 $i$ 相邻的节点列表，意味着从节点 $i$ 到节点 $graph[i]$ 中的每个节点都有一条有向边。

**要求**：找出图中所有的安全节点，将其存入数组作为答案返回，答案数组中的元素应当按升序排列。

**说明**：

- **终端节点**：如果一个节点没有连出的有向边，则它是终端节点。或者说，如果没有出边，则节点为终端节点。
- **安全节点**：如果从该节点开始的所有可能路径都通向终端节点，则该节点为安全节点。
- $n == graph.length$。
- $1 \le n \le 10^4$。
- $0 \le graph[i].length \le n$。
- $0 \le graph[i][j] \le n - 1$。
- $graph[i]$ 按严格递增顺序排列。
- 图中可能包含自环。
- 图中边的数目在范围 $[1, 4 \times 10^4]$ 内。

**示例**：

- 示例 1：

![](https://s3-lc-upload.s3.amazonaws.com/uploads/2018/03/17/picture1.png)

```python
输入：graph = [[1,2],[2,3],[5],[0],[5],[],[]]
输出：[2,4,5,6]
解释：示意图如上。
节点 5 和节点 6 是终端节点，因为它们都没有出边。
从节点 2、4、5 和 6 开始的所有路径都指向节点 5 或 6。
```

- 示例 2：

```python
输入：graph = [[1,2,3,4],[1,2],[3,4],[0,4],[]]
输出：[4]
解释:
只有节点 4 是终端节点，从节点 4 开始的所有路径都通向节点 4。
```

## 解题思路

### 思路 1：拓扑排序

1. 根据题意可知，安全节点所对应的终点，一定是出度为 $0$ 的节点。而安全节点一定能在有限步内到达终点，则说明安全节点一定不在「环」内。
2. 我们可以利用拓扑排序来判断顶点是否在环中。
3. 为了找出安全节点，可以采取逆序建图的方式，将所有边进行反向。这样出度为 $0$ 的终点就变为了入度为 $0$ 的点。
4. 然后通过拓扑排序不断移除入度为 $0$ 的点之后，如果不在「环」中的点，最后入度一定为 $0$，这些点也就是安全节点。而在「环」中的点，最后入度一定不为 $0$。
5. 最后将所有安全的起始节点存入数组作为答案返回。

### 思路 1：代码

```python
class Solution:
    # 拓扑排序，graph 中包含所有顶点的有向边关系（包括无边顶点）
    def topologicalSortingKahn(self, graph: dict):
        indegrees = {u: 0 for u in graph}   # indegrees 用于记录所有节点入度
        for u in graph:
            for v in graph[u]:
                indegrees[v] += 1           # 统计所有节点入度
        
        # 将入度为 0 的顶点存入集合 S 中
        S = collections.deque([u for u in indegrees if indegrees[u] == 0])
        
        while S:
            u = S.pop()                     # 从集合中选择一个没有前驱的顶点 0
            for v in graph[u]:              # 遍历顶点 u 的邻接顶点 v
                indegrees[v] -= 1           # 删除从顶点 u 出发的有向边
                if indegrees[v] == 0:       # 如果删除该边后顶点 v 的入度变为 0
                    S.append(v)             # 将其放入集合 S 中
        
        res = []
        for u in indegrees:
            if indegrees[u] == 0:
                res.append(u)
        
        return res
        
    def eventualSafeNodes(self, graph: List[List[int]]) -> List[int]:
        graph_dict = {u: [] for u in range(len(graph))}

        for u in range(len(graph)):
            for v in graph[u]:
                graph_dict[v].append(u)     # 逆序建图

        return self.topologicalSortingKahn(graph_dict)
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(n + m)$，其中 $n$ 是图中节点数目，$m$ 是图中边数目。
- **空间复杂度**：$O(n + m)$。

# [0803. 打砖块](https://leetcode.cn/problems/bricks-falling-when-hit/)

- 标签：并查集、数组、矩阵
- 难度：困难

## 题目链接

- [0803. 打砖块 - 力扣](https://leetcode.cn/problems/bricks-falling-when-hit/)

## 题目大意

**描述**：给定一个 $m \times n$ 大小的二元网格，其中 $1$ 表示砖块，$0$ 表示空白。砖块稳定（不会掉落）的前提是：

- 一块砖直接连接到网格的顶部。
- 或者至少有一块相邻（4 个方向之一）砖块稳定不会掉落时。

再给定一个数组 $hits$，这是需要依次消除砖块的位置。每当消除 $hits[i] = (row_i, col_i)$ 位置上的砖块时，对应位置的砖块（若存在）会消失，然后其他的砖块可能因为这一消除操作而掉落。一旦砖块掉落，它会立即从网格中消失（即，它不会落在其他稳定的砖块上）。

**要求**：返回一个数组 $result$，其中 $result[i]$ 表示第 $i$ 次消除操作对应掉落的砖块数目。

**说明**：

- 消除可能指向是没有砖块的空白位置，如果发生这种情况，则没有砖块掉落。
- $m == grid.length$。
- $n == grid[i].length$。
- $1 \le m, n \le 200$。
- $grid[i][j]$ 为 $0$ 或 $1$。
- $1 \le hits.length \le 4 \times 10^4$。
- $hits[i].length == 2$。
- $0 \le xi \le m - 1$。
- $0 \le yi \le n - 1$。
- 所有 $(xi, yi)$ 互不相同。

**示例**：

- 示例 1：

```python
输入：grid = [[1,0,0,0],[1,1,1,0]], hits = [[1,0]]
输出：[2]
解释：网格开始为：
[[1,0,0,0]，
 [1,1,1,0]]
消除 (1,0) 处加粗的砖块，得到网格：
[[1,0,0,0]
 [0,1,1,0]]
两个加粗的砖不再稳定，因为它们不再与顶部相连，也不再与另一个稳定的砖相邻，因此它们将掉落。得到网格：
[[1,0,0,0],
 [0,0,0,0]]
因此，结果为 [2]。
```

- 示例 2：

```python
输入：grid = [[1,0,0,0],[1,1,0,0]], hits = [[1,1],[1,0]]
输出：[0,0]
解释：网格开始为：
[[1,0,0,0],
 [1,1,0,0]]
消除 (1,1) 处加粗的砖块，得到网格：
[[1,0,0,0],
 [1,0,0,0]]
剩下的砖都很稳定，所以不会掉落。网格保持不变：
[[1,0,0,0], 
 [1,0,0,0]]
接下来消除 (1,0) 处加粗的砖块，得到网格：
[[1,0,0,0],
 [0,0,0,0]]
剩下的砖块仍然是稳定的，所以不会有砖块掉落。
因此，结果为 [0,0]。
```

## 解题思路

### 思路 1：并查集

一个很直观的想法：

- 将所有砖块放入一个集合中。
- 根据 $hits$ 数组的顺序，每敲掉一块砖。则将这块砖与相邻（4 个方向）的砖块断开集合。
- 然后判断哪些砖块会掉落，从集合中删除会掉落的砖块，并统计掉落砖块的数量。
  - **掉落砖块的数目 = 击碎砖块之前与屋顶相连的砖块数目 - 击碎砖块之后与屋顶相连的砖块数目 - 1**。

涉及集合问题，很容易想到用并查集来做。但是并查集主要用于合并查找集合，不适合断开集合。我们可以反向思考问题：

- 先将 $hits$ 中的所有位置上的砖块敲掉。
- 将剩下的砖块建立并查集。
- 逆序填回被敲掉的砖块，并与相邻（4 个方向）的砖块合并。这样问题就变为了 **补上砖块会新增多少个砖块粘到屋顶**。

整个算法步骤具体如下：

1. 先将二维数组 $grid$ 复制一份到二维数组 $copy\underline{\hspace{0.5em}}gird$ 上。这是因为遍历 $hits$ 元素时需要判断原网格是空白还是被打碎的砖块。
2. 在 $copy\underline{\hspace{0.5em}}grid$ 中将 $hits$ 中打碎的砖块赋值为 $0$。
3. 建立并查集，将房顶上的砖块合并到一个集合中。
4. 逆序遍历 $hits$，将 $hits$ 中的砖块补到 $copy\underline{\hspace{0.5em}}grid$ 中，并计算每一步中有多少个砖块粘到屋顶上（与屋顶砖块在一个集合中），并存入答案数组对应位置。
5. 最后输出答案数组。

### 思路 1：代码

```python
class UnionFind:
    def __init__(self, n):
        self.parent = [i for i in range(n)]
        self.size = [1 for _ in range(n)]

    def find(self, x):
        while x != self.parent[x]:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x == root_y:
            return False
        self.parent[root_x] = root_y
        self.size[root_y] += self.size[root_x]
        return True

    def is_connected(self, x, y):
        return self.find(x) == self.find(y)

    def get_size(self, x):
        root_x = self.find(x)
        return self.size[root_x]

class Solution:
    def hitBricks(self, grid: List[List[int]], hits: List[List[int]]) -> List[int]:
        directions = {(0, 1), (1, 0), (-1, 0), (0, -1)}
        rows, cols = len(grid), len(grid[0])

        def is_area(x, y):
            return 0 <= x < rows and 0 <= y < cols

        def get_index(x, y):
            return x * cols + y

        copy_grid = [[grid[i][j] for j in range(cols)] for i in range(rows)]

        for hit in hits:
            copy_grid[hit[0]][hit[1]] = 0

        union_find = UnionFind(rows * cols + 1)

        for j in range(cols):
            if copy_grid[0][j] == 1:
                union_find.union(j, rows * cols)

        for i in range(1, rows):
            for j in range(cols):
                if copy_grid[i][j] == 1:
                    if copy_grid[i - 1][j] == 1:
                        union_find.union(get_index(i - 1, j), get_index(i, j))
                    if j > 0 and copy_grid[i][j - 1] == 1:
                        union_find.union(get_index(i, j - 1), get_index(i, j))

        size_hits = len(hits)
        res = [0 for _ in range(size_hits)]
        for i in range(size_hits - 1, -1, -1):
            x, y = hits[i][0], hits[i][1]
            if grid[x][y] == 0:
                continue
            origin = union_find.get_size(rows * cols)
            if x == 0:
                union_find.union(y, rows * cols)
            for direction in directions:
                new_x = x + direction[0]
                new_y = y + direction[1]
                if is_area(new_x, new_y) and copy_grid[new_x][new_y] == 1:
                    union_find.union(get_index(x, y), get_index(new_x, new_y))
            curr = union_find.get_size(rows * cols)
            res[i] = max(0, curr - origin - 1)
            copy_grid[x][y] = 1
        return res
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(m \times n \times \alpha(m \times n))$，其中 $\alpha$ 是反 Ackerman 函数。
- **空间复杂度**：$O(m \times n)$。

# [0804. 唯一摩尔斯密码词](https://leetcode.cn/problems/unique-morse-code-words/)

- 标签：数组、哈希表、字符串
- 难度：简单

## 题目链接

- [0804. 唯一摩尔斯密码词 - 力扣](https://leetcode.cn/problems/unique-morse-code-words/)

## 题目大意

**描述**：国际摩尔斯密码定义一种标准编码方式，将每个字母对应于一个由一系列点和短线组成的字符串， 比如:

- `'a'` 对应 `".-"`，
- `'b'` 对应 `"-..."`，
- `'c'` 对应 `"-.-."` ，以此类推。

为了方便，所有 $26$ 个英文字母的摩尔斯密码表如下：

`[".-","-...","-.-.","-..",".","..-.","--.","....","..",".---","-.-",".-..","--","-.","---",".--.","--.-",".-.","...","-","..-","...-",".--","-..-","-.--","--.."]`

给定一个字符串数组 $words$，每个单词可以写成每个字母对应摩尔斯密码的组合。

- 例如，`"cab"` 可以写成 `"-.-..--..."` ，(即 `"-.-."` + `".-"` + `"-..."` 字符串的结合)。我们将这样一个连接过程称作单词翻译。

**要求**：对 $words$ 中所有单词进行单词翻译，返回不同单词翻译的数量。

**说明**：

- $1 \le words.length \le 100$。
- $1 \le words[i].length \le 12$。
- $words[i]$ 由小写英文字母组成。

**示例**：

- 示例 1：

```python
输入: words = ["gin", "zen", "gig", "msg"]
输出: 2
解释: 
各单词翻译如下:
"gin" -> "--...-."
"zen" -> "--...-."
"gig" -> "--...--."
"msg" -> "--...--."

共有 2 种不同翻译, "--...-." 和 "--...--.".
```

- 示例 2：

```python
输入：words = ["a"]
输出：1
```

## 解题思路

### 思路 1：模拟 + 哈希表

1. 根据题目要求，将所有单词都转换为对应摩斯密码。
2. 使用哈希表存储所有转换后的摩斯密码。
3. 返回哈希表中不同的摩斯密码个数（脊哈希表的长度）作为答案。

### 思路 1：代码

```Python
class Solution:
    def uniqueMorseRepresentations(self, words: List[str]) -> int:
        table = [".-","-...","-.-.","-..",".","..-.","--.","....","..",".---","-.-",".-..","--","-.","---",".--.","--.-",".-.","...","-","..-","...-",".--","-..-","-.--","--.."]
        word_set = set()

        for word in words:
            word_mose = ""
            for ch in word:
                word_mose += table[ord(ch) - ord('a')]
            word_set.add(word_mose)

        return len(word_set)
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(s)$，其中 $s$ 为数组 $words$ 中所有单词的长度之和。
- **空间复杂度**：$O(s)$。

# [0806. 写字符串需要的行数](https://leetcode.cn/problems/number-of-lines-to-write-string/)

- 标签：数组、字符串
- 难度：简单

## 题目链接

- [0806. 写字符串需要的行数 - 力扣](https://leetcode.cn/problems/number-of-lines-to-write-string/)

## 题目大意

**描述**：给定一个数组 $widths$，其中 $words[0]$ 代表 `'a'` 需要的单位，$words[1]$ 代表 `'b'` 需要的单位，…，$words[25]$ 代表 `'z'` 需要的单位。再给定一个字符串 $s$，现在需要将字符串 $s$ 从左到右写到每一行上，每一行的最大宽度为 $100$ 个单位，如果在写某个字符的时候使改行超过了 $100$ 个单位，那么我们应该将这个字母写到下一行。

**要求**：计算出能放下 $s$ 的最少行数，以及最后一行使用的宽度单位。

**说明**：

- 字符串 $s$ 的长度在 $[1, 1000]$ 的范围。
- $s$ 只包含小写字母。
- $widths$ 是长度为 $26$ 的数组。
- $widths[i]$ 值的范围在 $[2, 10]$。

**示例**：

- 示例 1：

```python
输入: 
widths = [10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10]
S = "abcdefghijklmnopqrstuvwxyz"
输出: [3, 60]
解释: 
所有的字符拥有相同的占用单位10。所以书写所有的26个字母，
我们需要2个整行和占用60个单位的一行。
```

- 示例 2：

```python
输入: 
widths = [4,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10]
S = "bbbcccdddaaa"
输出: [2, 4]
解释: 
除去字母'a'所有的字符都是相同的单位10，并且字符串 "bbbcccdddaa" 将会覆盖 9 * 10 + 2 * 4 = 98 个单位.
最后一个字母 'a' 将会被写到第二行，因为第一行只剩下2个单位了。
所以，这个答案是2行，第二行有4个单位宽度。
```

## 解题思路

### 思路 1：模拟

1. 使用变量 $line\underline{\hspace{0.5em}}cnt$ 记录行数，使用变量 $last\underline{\hspace{0.5em}}cnt$ 记录最后一行使用的单位数。
2. 遍历字符串，如果当前最后一行使用的单位数 + 当前字符需要的单位超过了 $100$，则：
   1. 另起一行填充字符。（即行数加 $1$，最后一行使用的单位数为当前字符宽度）。
3. 如果当前最后一行使用的单位数 + 当前字符需要的单位没有超过 $100$，则：
   1. 在当前行填充字符。（即最后一行使用的单位数累加上当前字符宽度）。

### 思路 1：代码

```python
class Solution:
    def numberOfLines(self, widths: List[int], s: str) -> List[int]:
        line_cnt, last_cnt = 1, 0
        for ch in s:
            width = widths[ord(ch) - ord('a')]
            if last_cnt + width > 100:
                line_cnt += 1
                last_cnt = width
            else:
                last_cnt += width
                
        return [line_cnt, last_cnt]
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(n)$。
- **空间复杂度**：$O(1)$。

# [0811. 子域名访问计数](https://leetcode.cn/problems/subdomain-visit-count/)

- 标签：数组、哈希表、字符串、计数
- 难度：中等

## 题目链接

- [0811. 子域名访问计数 - 力扣](https://leetcode.cn/problems/subdomain-visit-count/)

## 题目大意

**描述**：网站域名是由多个子域名构成的。

- 例如 `"discuss.leetcode.com"` 的顶级域名为 `"com"`，二级域名为 `"leetcode.com"`，三级域名为 `"discuss.leetcode.com"`。

当访问 `"discuss.leetcode.com"` 时，也会隐式访问其父域名 `"leetcode.com"` 以及 `"com"`。

计算机配对域名的格式为 `"rep d1.d2.d3"` 或 `"rep d1.d2"`。其中 `rep` 表示访问域名的次数，`d1.d2.d3` 或 `d1.d2` 为域名本身。

- 例如：`"9001 discuss.leetcode.com"` 就是一个 计数配对域名 ，表示 `discuss.leetcode.com` 被访问了 `9001` 次。

现在给定一个由计算机配对域名组成的数组 `cpdomains`。

**要求**：解析每一个计算机配对域名，计算出所有域名的访问次数，并以数组形式返回。可以按任意顺序返回答案。

## 解题思路

这道题求解的是不同层级的域名的次数汇总，很容易想到使用哈希表。我们可以使用哈希表来统计不同层级的域名访问次数。具体做如下：

1. 如果数组 `cpdomains` 为空，直接返回空数组。
2. 使用哈希表 `times_dict` 存储不同层级的域名访问次数。
3. 遍历数组 `cpdomains`。对于每一个计算机配对域名 `cpdomain`：
    1. 先将计算机配对域名的访问次数 `times` 和域名 `domain` 进行分割。
    2. 然后将域名转为子域名数组 `domain_list`，逆序拼接不同等级的子域名 `sub_domain`。
    3. 如果子域名 `sub_domain` 没有出现在哈希表 `times_dict` 中，则在哈希表中存入 `sub_domain` 和访问次数 `times` 的键值对。
    4. 如果子域名 `sub_domain` 曾经出现在哈希表 `times_dict` 中，则在哈希表对应位置加上 `times`。
4. 遍历完之后，遍历哈希表 `times_dict`，将所有域名和访问次数拼接为字符串，存入答案数组中。
5. 最后返回答案数组。

## 代码

```python
class Solution:
    def subdomainVisits(self, cpdomains: List[str]) -> List[str]:
        if not cpdomains:
            return []

        times_dict = dict()
        for cpdomain in cpdomains:
            tiems, domain = cpdomain.split()
            tiems = int(tiems)
            
            domain_list = domain.split('.')
            for i in range(len(domain_list) - 1, -1, -1):
                sub_domain = '.'.join(domain_list[i:])
                if sub_domain not in times_dict:
                    times_dict[sub_domain] = tiems
                else:
                    times_dict[sub_domain] += tiems
        
        res = []
        for key in times_dict.keys():
            res.append(str(times_dict[key]) + ' ' + key)
        return res
```

# [0814. 二叉树剪枝](https://leetcode.cn/problems/binary-tree-pruning/)

- 标签：树、深度优先搜索、二叉树
- 难度：中等

## 题目链接

- [0814. 二叉树剪枝 - 力扣](https://leetcode.cn/problems/binary-tree-pruning/)

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

    def pruneTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        if not root:
            return root
        if self.containsOnlyZero(root):
            return None

        root.left = self.pruneTree(root.left)
        root.right = self.pruneTree(root.right)
        return root
```

# [0819. 最常见的单词](https://leetcode.cn/problems/most-common-word/)

- 标签：哈希表、字符串、计数
- 难度：简单

## 题目链接

- [0819. 最常见的单词 - 力扣](https://leetcode.cn/problems/most-common-word/)

## 题目大意

**描述**：给定一个字符串 $paragraph$ 表示段落，再给定搞一个禁用单词列表 $banned$。

**要求**：返回出现次数最多，同时不在禁用列表中的单词。

**说明**：

- 题目保证至少有一个词不在禁用列表中，而且答案唯一。
- 禁用列表 $banned$ 中的单词用小写字母表示，不含标点符号。
- 段落 $paragraph$ 只包含字母、空格和下列标点符号`!?',;.`
- 段落中的单词不区分大小写。
- $1 \le \text{段落长度} \le 1000$。
- $0 \le \text{禁用单词个数} \le 100$。
- $1 \le \text{禁用单词长度} \le 10$。
- 答案是唯一的，且都是小写字母（即使在 $paragraph$ 里是大写的，即使是一些特定的名词，答案都是小写的）。
- 不存在没有连字符或者带有连字符的单词。
- 单词里只包含字母，不会出现省略号或者其他标点符号。

**示例**：

- 示例 1：

```python
输入: 
paragraph = "Bob hit a ball, the hit BALL flew far after it was hit."
banned = ["hit"]
输出: "ball"
解释: 
"hit" 出现了3次，但它是一个禁用的单词。
"ball" 出现了2次 (同时没有其他单词出现2次)，所以它是段落里出现次数最多的，且不在禁用列表中的单词。 
注意，所有这些单词在段落里不区分大小写，标点符号需要忽略（即使是紧挨着单词也忽略， 比如 "ball,"）， 
"hit"不是最终的答案，虽然它出现次数更多，但它在禁用单词列表中。
```

- 示例 2：

```python
输入：
paragraph = "a."
banned = []
输出："a"
```

## 解题思路

### 思路 1：哈希表

1. 将禁用词列表转为集合 $banned\underline{\hspace{0.5em}}set$。
2. 遍历段落 $paragraph$，获取段落中的所有单词。
3. 判断当前单词是否在禁用词集合中，如果不在禁用词集合中，则使用哈希表对该单词进行计数。
4. 遍历完，找出哈希表中频率最大的单词，将该单词作为答案进行返回。

### 思路 1：代码

```python
class Solution:
    def mostCommonWord(self, paragraph: str, banned: List[str]) -> str:
        banned_set = set(banned)
        cnts = Counter()

        word = ""
        for ch in paragraph:
            if ch.isalpha():
                word += ch.lower()
            else:
                if word and word not in banned_set:
                    cnts[word] += 1
                word = ""
        if word and word not in banned_set:
            cnts[word] += 1

        max_cnt, ans = 0, ""
        for word, cnt in cnts.items():
            if cnt > max_cnt:
                max_cnt = cnt
                ans = word
        
        return ans
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(n + m)$，其中 $n$ 为段落 $paragraph$ 的长度，$m$ 是禁用词 $banned$ 的长度。
- **空间复杂度**：$O(n + m)$。

