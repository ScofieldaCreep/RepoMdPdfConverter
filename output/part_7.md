# [1984. 学生分数的最小差值](https://leetcode.cn/problems/minimum-difference-between-highest-and-lowest-of-k-scores/)

- 标签：数组、排序、滑动窗口
- 难度：简单

## 题目链接

- [1984. 学生分数的最小差值 - 力扣](https://leetcode.cn/problems/minimum-difference-between-highest-and-lowest-of-k-scores/)

## 题目大意

**描述**：给定一个下标从 $0$ 开始的整数数组 $nums$，其中 $nums[i]$ 表示第 $i$ 名学生的分数。另给定一个整数 $k$。

**要求**：从数组中选出任意 $k$ 名学生的分数，使这 $k$ 个分数间最高分和最低分的差值达到最小化。返回可能的最小差值 。

**说明**：

- $1 \le k \le nums.length \le 1000$。
- $0 \le nums[i] \le 10^5$。

**示例**：

- 示例 1：

```python
输入：nums = [90], k = 1
输出：0
解释：选出 1 名学生的分数，仅有 1 种方法：
- [90] 最高分和最低分之间的差值是 90 - 90 = 0
可能的最小差值是 0
```

- 示例 2：

```python
输入：nums = [9,4,1,7], k = 2
输出：2
解释：选出 2 名学生的分数，有 6 种方法：
- [9,4,1,7] 最高分和最低分之间的差值是 9 - 4 = 5
- [9,4,1,7] 最高分和最低分之间的差值是 9 - 1 = 8
- [9,4,1,7] 最高分和最低分之间的差值是 9 - 7 = 2
- [9,4,1,7] 最高分和最低分之间的差值是 4 - 1 = 3
- [9,4,1,7] 最高分和最低分之间的差值是 7 - 4 = 3
- [9,4,1,7] 最高分和最低分之间的差值是 7 - 1 = 6
可能的最小差值是 2
```

## 解题思路

### 思路 1：排序 + 滑动窗口

如果想要最小化选择的 $k$ 名学生中最高分与最低分的差值，我们应该在排序后的数组中连续选择 $k$ 名学生。这是因为如果将连续 $k$ 名学生中的某位学生替换成不连续的学生，其最高分 / 最低分一定会发生变化，并且一定会使最高分变得最高 / 最低分变得最低。从而导致差值增大。

因此，最优方案一定是在排序后的数组中连续选择 $k$ 名学生中的所有情况中的其中一种。

这样，我们可以先对数组 $nums$ 进行升序排序。然后使用一个固定长度为 $k$ 的滑动窗口计算连续选择 $k$ 名学生的最高分与最低分的差值。并记录下最小的差值 $ans$，最后作为答案并返回结果。

### 思路 1：代码

```Python
class Solution:
    def minimumDifference(self, nums: List[int], k: int) -> int:
        nums.sort()
        ans = float('inf')
        for i in range(k - 1, len(nums)):
            ans = min(ans, nums[i] - nums[i - k + 1])

        return ans
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(n \times \log n)$，其中 $n$ 为数组 $nums$ 的长度。
- **空间复杂度**：$O(1)$。

### 思路 2：

### 思路 2：代码

```python
```

### 思路 2：复杂度分析

- **时间复杂度**：
- **空间复杂度**：

# [1986. 完成任务的最少工作时间段](https://leetcode.cn/problems/minimum-number-of-work-sessions-to-finish-the-tasks/)

- 标签：位运算、数组、动态规划、回溯、状态压缩
- 难度：中等

## 题目链接

- [1986. 完成任务的最少工作时间段 - 力扣](https://leetcode.cn/problems/minimum-number-of-work-sessions-to-finish-the-tasks/)

## 题目大意

**描述**：给定一个整数数组 $tasks$ 代表需要完成的任务。 其中 $tasks[i]$ 表示第 $i$ 个任务需要花费的时长（单位为小时）。再给定一个整数 $sessionTime$，代表在一个工作时段中，最多可以连续工作的小时数。在连续工作至多 $sessionTime$ 小时后，需要进行休息。

现在需要按照如下条件完成给定任务：

1. 如果你在某一个时间段开始一个任务，你需要在同一个时间段完成它。
2. 完成一个任务后，你可以立马开始一个新的任务。
3. 你可以按任意顺序完成任务。

**要求**：按照上述要求，返回完成所有任务所需要的最少数目的工作时间段。

**说明**：

- $n == tasks.length$。
- $1 \le n \le 14$。
- $1 \le tasks[i] \le 10$。
- $max(tasks[i]) \le sessionTime \le 15$。

**示例**：

- 示例 1：

```python
输入：tasks = [1,2,3], sessionTime = 3
输出：2
解释：你可以在两个工作时间段内完成所有任务。
- 第一个工作时间段：完成第一和第二个任务，花费 1 + 2 = 3 小时。
- 第二个工作时间段：完成第三个任务，花费 3 小时。
```

- 示例 2：

```python
输入：tasks = [3,1,3,1,1], sessionTime = 8
输出：2
解释：你可以在两个工作时间段内完成所有任务。
- 第一个工作时间段：完成除了最后一个任务以外的所有任务，花费 3 + 1 + 3 + 1 = 8 小时。
- 第二个工作时间段，完成最后一个任务，花费 1 小时。
```

## 解题思路

### 思路 1：状压 DP

### 思路 1：代码

```python
class Solution:
    def minSessions(self, tasks: List[int], sessionTime: int) -> int:
        size = len(tasks)
        states = 1 << size
        
        prefix_sum = [0 for _ in range(states)]
        for state in range(states):
            for i in range(size):
                if (state >> i) & 1:
                    prefix_sum[state] = prefix_sum[state ^ (1 << i)] + tasks[i]
                    break
        
        dp = [float('inf') for _ in range(states)]
        dp[0] = 0
        for state in range(states):
            sub = state
            while sub > 0:
                if prefix_sum[sub] <= sessionTime:
                    dp[state] = min(dp[state], dp[state ^ sub] + 1)
                sub = (sub - 1) & state

        return dp[states - 1]
```

### 思路 1：复杂度分析

- **时间复杂度**：
- **空间复杂度**：

# [1991. 找到数组的中间位置](https://leetcode.cn/problems/find-the-middle-index-in-array/)

- 标签：数组、前缀和
- 难度：简单

## 题目链接

- [1991. 找到数组的中间位置 - 力扣](https://leetcode.cn/problems/find-the-middle-index-in-array/)

## 题目大意

**描述**：给定一个下标从 $0$ 开始的整数数组 $nums$。

**要求**：返回最左边的中间位置 $middleIndex$（也就是所有可能中间位置下标做小的一个）。如果找不到这样的中间位置，则返回 $-1$。

**说明**：

- **中间位置 $middleIndex$**：满足 $nums[0] + nums[1] + … + nums[middleIndex - 1] == nums[middleIndex + 1] + nums[middleIndex + 2] + … + nums[nums.length - 1]$ 的数组下标。
- 如果 $middleIndex == 0$，左边部分的和定义为 $0$。类似的，如果 $middleIndex == nums.length - 1$，右边部分的和定义为 $0$。

**示例**：

- 示例 1：

```python
输入：nums = [2,3,-1,8,4]
输出：3
解释：
下标 3 之前的数字和为：2 + 3 + -1 = 4
下标 3 之后的数字和为：4 = 4
```

- 示例 2：

```python
输入：nums = [1,-1,4]
输出：2
解释：
下标 2 之前的数字和为：1 + -1 = 0
下标 2 之后的数字和为：0
```

## 解题思路

### 思路 1：前缀和

1. 先遍历一遍数组，求出数组中全部元素和为 $total$。
2. 再遍历一遍数组，使用变量 $prefix\underline{\hspace{0.5em}}sum$ 为前 $i$ 个元素和。
3. 当遍历到第 $i$ 个元素时，其数组左侧元素之和为 $prefix\underline{\hspace{0.5em}}sum$，右侧元素和为 $total - prefix\underline{\hspace{0.5em}}sum - nums[i]$。
   1. 如果左右元素之和相等，即 $prefix\underline{\hspace{0.5em}}sum == total - prefix\underline{\hspace{0.5em}}sum - nums[i]$（$2 \times prefix\underline{\hspace{0.5em}}sum + nums[i] == total$） 时，$i$ 为中间位置。此时返回 $i$。
   2. 如果不满足，则继续累加当前元素到 $prefix\underline{\hspace{0.5em}}sum$ 中，继续向后遍历。
4. 如果找不到符合要求的中间位置，则返回 $-1$。

### 思路 1：代码

```python
class Solution:
    def findMiddleIndex(self, nums: List[int]) -> int:
        total = sum(nums)

        prefix_sum = 0
        for i in range(len(nums)):
            if 2 * prefix_sum + nums[i] == total:
                return i
            prefix_sum += nums[i]
        
        return -1
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(n)$。
- **空间复杂度**：$O(1)$。

# [1994. 好子集的数目](https://leetcode.cn/problems/the-number-of-good-subsets/)

- 标签：位运算、数组、数学、动态规划、状态压缩
- 难度：困难

## 题目链接

- [1994. 好子集的数目 - 力扣](https://leetcode.cn/problems/the-number-of-good-subsets/)

## 题目大意

**描述**：给定一个整数数组 $nums$。

**要求**：返回 $nums$ 中不同的好子集的数目对 $10^9 + 7$ 取余的结果。

**说明**：

- **子集**：通过删除 $nums$ 中一些（可能一个都不删除，也可能全部都删除）元素后剩余元素组成的数组。如果两个子集删除的下标不同，那么它们被视为不同的子集。
  
- **好子集**：如果 $nums$ 的一个子集中，所有元素的乘积可以表示为一个或多个互不相同的质数的乘积，那么我们称它为好子集。
  - 比如，如果 $nums = [1, 2, 3, 4]$：
    - $[2, 3]$，$[1, 2, 3]$ 和 $[1, 3]$ 是好子集，乘积分别为 $6 = 2 \times 3$ ，$6 = 2 \times 3$ 和 $3 = 3$。
    - $[1, 4]$ 和 $[4]$ 不是好子集，因为乘积分别为 $4 = 2 \times 2$ 和 $4 = 2 \times 2$。

- $1 \le nums.length \le 10^5$。
- $1 \le nums[i] \le 30$。

**示例**：

- 示例 1：

```python
输入：nums = [1,2,3,4]
输出：6
解释：好子集为：
- [1,2]：乘积为 2，可以表示为质数 2 的乘积。
- [1,2,3]：乘积为 6，可以表示为互不相同的质数 2 和 3 的乘积。
- [1,3]：乘积为 3，可以表示为质数 3 的乘积。
- [2]：乘积为 2，可以表示为质数 2 的乘积。
- [2,3]：乘积为 6，可以表示为互不相同的质数 2 和 3 的乘积。
- [3]：乘积为 3，可以表示为质数 3 的乘积。
```

- 示例 2：

```python
输入：nums = [4,2,3,15]
输出：5
解释：好子集为：
- [2]：乘积为 2，可以表示为质数 2 的乘积。
- [2,3]：乘积为 6，可以表示为互不相同质数 2 和 3 的乘积。
- [2,15]：乘积为 30，可以表示为互不相同质数 2，3 和 5 的乘积。
- [3]：乘积为 3，可以表示为质数 3 的乘积。
- [15]：乘积为 15，可以表示为互不相同质数 3 和 5 的乘积。
```

## 解题思路

### 思路 1：状态压缩 DP

根据题意可以看出：

1. 虽然 $nums$ 的长度是 $[1, 10^5]$，但是其值域范围只有 $[1, 30]$，则我们可以将 $[1, 30]$ 的数分为 $3$ 类：
   1. 质数：$[2, 3, 5, 7, 11, 13, 17, 19, 23, 29]$（共 $10$ 个数）。由于好子集的乘积拆解后的质因子只能包含这 $10$ 个，我们可以使用一个数组 $primes$ 记录下这 $10$ 个质数，将好子集的乘积拆解为质因子后，每个 $primes[i]$ 最多出现一次。
   2. 非质数：$[4, 6, 8, 9, 10, 12, 14, 16, 18, 20, 21, 22, 24, 25, 26, 27, 28, 30]$。非质数肯定不会出现在好子集的乘积拆解后的质因子中。
   3. 特殊的数：$[1]$。对于一个好子集而言，无论向中间添加多少个 $1$，得到的新子集仍是好子集。
2. 分类完成后，由于 $[1, 30]$ 中只有 $10$ 个质数，因此我们可以使用一个长度为 $10$ 的二进制数  $state$ 来表示 $primes$ 中质因数的选择情况。其中，如果 $state$ 第 $i$ 位为 $1$，则说明第 $i$ 个质因数 $primes[i]$ 被使用过；如果 $state$ 第 $i$ 位为 $0$，则说明第 $i$ 个质因数 $primes[i]$ 没有被使用过。
3. 题目规定值相同，但是下标不同的子集视为不同子集，那么我们可先统计出 $nums$ 中每个数 $nums[i]$ 的出现次数，将其存入 $cnts$ 数组中，其中 $cnts[num]$ 表示 $num$ 出现的次数。这样在统计方案时，直接计算出 $num$ 的方案数，再乘以 $cnts[num]$ 即可。

接下来，我们就可以使用「动态规划」的方式来解决这道题目了。

###### 1. 划分阶段

按照质因数的选择情况进行阶段划分。

###### 2. 定义状态

定义状态 $dp[state]$ 表示为：当质因数选择的情况为 $state$ 时，好子集的数目。

###### 3. 状态转移方程

对于 $nums$ 中的每个数 $num$，其对应出现次数为 $cnt$。我们可以通过试除法，将 $num$ 分解为不同的质因数，并使用「状态压缩」的方式，用一个二进制数 $cur\underline{\hspace{0.5em}}state$ 来表示当前数 $num$ 中使用了哪些质因数。然后枚举所有状态，找到与 $cur\underline{\hspace{0.5em}}state$ 不冲突的状态 $state$（也就是除了 $cur\underline{\hspace{0.5em}}state$ 中选择的质因数外，选择的其他质因数情况，比如 $cur\underline{\hspace{0.5em}}state$ 选择了 $2$ 和 $5$，则枚举不选择 $2$ 和 $5$ 的状态）。

此时，状态转移方程为：$dp[state | cur\underline{\hspace{0.5em}}state] = \sum (dp[state] \times cnt) \mod MOD , \quad state \text{ \& } cur\underline{\hspace{0.5em}}state == 0$

###### 4. 初始条件

- 当 $state == 0$，所选质因数为空时，空集为好子集，则 $dp[0] = 1$。同时，对于一个好子集而言，无论向中间添加多少个 $1$，得到的新子集仍是好子集，所以对于空集来说，可以对应出 $2^{cnts[1]}$ 个方案，则最终 $dp[0] = 2^{cnts[1]}$。

###### 5. 最终结果

根据我们之前定义的状态，$dp[state]$ 表示为：当质因数的选择的情况为 $state$ 时，好子集的数目。 所以最终结果为所有状态下的好子集数目累积和。所以我们可以枚举所有状态，并记录下所有好子集的数目和，就是最终结果。

### 思路 1：代码

```python
class Solution:
    def numberOfGoodSubsets(self, nums: List[int]) -> int:
        MOD = 10 ** 9 + 7
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

        cnts = Counter(nums)
        dp = [0 for _ in range(1 << len(primes))]
        dp[0] = pow(2, cnts[1], MOD)            # 计算 1
		
        # num 分解质因数
        for num, cnt in cnts.items():           # 遍历 nums 中所有数及其频数
            if num == 1:                        # 跳过 1
                continue
                
            flag = True                         # 检查 num 的质因数是否都不超过 1
            cur_num = num                       
            cur_state = 0
            for i, prime in enumerate(primes):  # 对 num 进行试除
                cur_cnt = 0
                while cur_num % prime == 0:
                    cur_cnt += 1
                    cur_state |= 1 << i
                    cur_num //= prime
                if cur_cnt > 1:                 # 当前质因数超过 1，则 num 不能添加到子集中，跳过
                    flag = False
                    break
            if not flag:
                continue
            
            for state in range(1 << len(primes)):
                if state & cur_state == 0:      # 只有当前选择状态与前一状态不冲突时，才能进行动态转移
                    dp[state | cur_state] = (dp[state | cur_state] + dp[state] * cnt) % MOD
            
        ans = 0                                 # 统计所有非空集合的方案数
        for i in range(1, 1 << len(primes)):
            ans = (ans + dp[i]) % MOD

        return ans
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(n + m \times 2^p)$，其中 $n$ 为数组 $nums$ 的元素个数，$m$ 为 $nums$ 的最大值，$p$ 为 $[1, 30]$ 中的质数个数。
- **空间复杂度**：$O(2^p)$。
# [2011. 执行操作后的变量值](https://leetcode.cn/problems/final-value-of-variable-after-performing-operations/)

- 标签：数组、字符串、模拟
- 难度：简单

## 题目链接

- [2011. 执行操作后的变量值 - 力扣](https://leetcode.cn/problems/final-value-of-variable-after-performing-operations/)

## 题目大意

存在一种支持 `4` 种操作和 `1` 个变量 `X` 的编程语言：

- `++X` 和 `x++` 使得变量 `X` 值加 `1`。
- `--X` 和 `X--` 使得变脸 `X ` 值减 `1`。

`X` 的初始值是 `0`。现在给定一个字符串数组 `operations`，这是由操作组成的一个列表。

要求：返回执行所有操作后，`X` 的最终值。

## 解题思路

思路很简单，初始答案 `res` 赋值为 `0`。

然后遍历操作列表 `operations`，判断每一个操作 `operation` 的符号。如果操作中含有 `+`，则让答案加 `1`，否则，则让答案减 `1`。最后输出答案。

## 代码

```python
def finalValueAfterOperations(self, operations):
        """
        :type operations: List[str]
        :rtype: int
        """
        res = 0

        for opration in operations:
            res += 1 if '+' in opration else -1

        return res
```

# [2023. 连接后等于目标字符串的字符串对](https://leetcode.cn/problems/number-of-pairs-of-strings-with-concatenation-equal-to-target/)

- 标签：数组、字符串
- 难度：中等

## 题目链接

- [2023. 连接后等于目标字符串的字符串对 - 力扣](https://leetcode.cn/problems/number-of-pairs-of-strings-with-concatenation-equal-to-target/)

## 题目大意

**描述**：给定一个数字字符串数组 `nums` 和一个数字字符串 `target`。

**要求**：返回 `nums[i] + nums[j]` （两个字符串连接，其中 `i != j`）结果等于 `target` 的下标 `(i, j)` 的数目。

**说明**：

- $2 \le nums.length \le 100$。
- $1 \le nums[i].length \le 100$。
- $2 \le target.length \le 100$。
- `nums[i]` 和 `target` 只包含数字。
- `nums[i]` 和 `target` 不含有任何前导 $0$。

**示例**：

- 示例 1：

```python
输入：nums = ["777","7","77","77"], target = "7777"
输出：4
解释：符合要求的下标对包括：
- (0, 1)："777" + "7"
- (1, 0)："7" + "777"
- (2, 3)："77" + "77"
- (3, 2)："77" + "77"
```

- 示例 2：

```python
输入：nums = ["123","4","12","34"], target = "1234"
输出：2
解释：符合要求的下标对包括
- (0, 1)："123" + "4"
- (2, 3)："12" + "34"
```

## 解题思路

### 思路 1：暴力枚举

1. 双重循环遍历所有的 `i` 和 `j`，满足 `i != j` 并且 `nums[i] + nums[j] == target` 时，记入到答案数目中。
2. 遍历完，返回答案数目。

### 思路 1：代码

```python
class Solution:
    def numOfPairs(self, nums: List[str], target: str) -> int:
        res = 0
        for i in range(len(nums)):
            for j in range(len(nums)):
                if i != j and nums[i] + nums[j] == target:
                    res += 1
        
        return res
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(n^2)$。
- **空间复杂度**：$O(1)$。

### 思路 2：哈希表

1. 使用哈希表记录字符串数组 `nums` 中所有数字字符串的数量。
2. 遍历哈希表中的键 `num`。
3. 将 `target` 根据 `num` 的长度分为前缀 `prefix` 和 `suffix`。
4. 如果 `num` 等于 `prefix`，则判断后缀 `suffix` 是否在哈希表中，如果在哈希表中，则说明 `prefix` 和 `suffix` 能够拼接为 `target`。
   1. 如果 `num` 等于 `suffix`，此时 `perfix == suffix`，则答案数目累积为 `table[prefix] * (table[suffix] - 1)`。
   2. 如果 `num` 不等于 `suffix`，则答案数目累积为 `table[prefix] * table[suffix]`。
5. 最后输出答案数目。

### 思路 2：代码

```python
class Solution:
    def numOfPairs(self, nums: List[str], target: str) -> int:
        res = 0
        table = collections.defaultdict(int)
        for num in nums:
            table[num] += 1

        for num in table:
            size = len(num)
            prefix, suffix = target[ :size], target[size: ]
            if num == prefix and suffix in table:
                if num == suffix:
                    res += table[prefix] * (table[suffix] - 1)
                else:
                    res += table[prefix] * table[suffix]
        
        return res
```

### 思路 2：复杂度分析

- **时间复杂度**：$O(n)$。
- **空间复杂度**：$O(n)$。# [2050. 并行课程 III](https://leetcode.cn/problems/parallel-courses-iii/)

- 标签：图、拓扑排序、数组、动态规划
- 难度：困难

## 题目链接

- [2050. 并行课程 III - 力扣](https://leetcode.cn/problems/parallel-courses-iii/)

## 题目大意

**描述**：给定一个整数 $n$，表示有 $n$ 节课，课程编号为 $1 \sim n$。

再给定一个二维整数数组 $relations$，其中 $relations[j] = [prevCourse_j, nextCourse_j]$，表示课程 $prevCourse_j$ 必须在课程 $nextCourse_j$ 之前完成（先修课的关系）。

再给定一个下标从 $0$ 开始的整数数组 $time$，其中 $time[i]$ 表示完成第 $(i + 1)$ 门课程需要花费的月份数。

现在根据以下规则计算完成所有课程所需要的最少月份数：

- 如果一门课的所有先修课都已经完成，则可以在任意时间开始这门课程。
- 可以同时上任意门课程。

**要求**：返回完成所有课程所需要的最少月份数。

**说明**：

- $1 \le n \le 5 * 10^4$。
- $0 \le relations.length \le min(n * (n - 1) / 2, 5 \times 10^4)$。
- $relations[j].length == 2$。
- $1 \le prevCourse_j, nextCourse_j \le n$。
- $prevCourse_j != nextCourse_j$。
- 所有的先修课程对 $[prevCourse_j, nextCourse_j]$ 都是互不相同的。
- $time.length == n$。
- $1 \le time[i] \le 10^4$。
- 先修课程图是一个有向无环图。

**示例**：

- 示例 1：

![](https://assets.leetcode.com/uploads/2021/10/07/ex1.png)

```python
输入：n = 3, relations = [[1,3],[2,3]], time = [3,2,5]
输出：8
解释：上图展示了输入数据所表示的先修关系图，以及完成每门课程需要花费的时间。
你可以在月份 0 同时开始课程 1 和 2 。
课程 1 花费 3 个月，课程 2 花费 2 个月。
所以，最早开始课程 3 的时间是月份 3 ，完成所有课程所需时间为 3 + 5 = 8 个月。
```

- 示例 2：

![](https://assets.leetcode.com/uploads/2021/10/07/ex2.png)

```python
输入：n = 5, relations = [[1,5],[2,5],[3,5],[3,4],[4,5]], time = [1,2,3,4,5]
输出：12
解释：上图展示了输入数据所表示的先修关系图，以及完成每门课程需要花费的时间。
你可以在月份 0 同时开始课程 1 ，2 和 3 。
在月份 1，2 和 3 分别完成这三门课程。
课程 4 需在课程 3 之后开始，也就是 3 个月后。课程 4 在 3 + 4 = 7 月完成。
课程 5 需在课程 1，2，3 和 4 之后开始，也就是在 max(1,2,3,7) = 7 月开始。
所以完成所有课程所需的最少时间为 7 + 5 = 12 个月。
```

## 解题思路

### 思路 1：拓扑排序 + 动态规划

1. 使用邻接表 $graph$ 存放课程关系图，并统计每门课程节点的入度，存入入度列表 $indegrees$。定义 $dp[i]$ 为完成第 $i$ 门课程所需要的最少月份数。使用 $ans$ 表示完成所有课程所需要的最少月份数。
2. 借助队列 $queue$，将所有入度为 $0$ 的节点入队。
3. 将队列中入度为 $0$ 的节点依次取出。对于取出的每个节点 $u$：
   1. 遍历该节点的相邻节点 $v$，更新相邻节点 $v$ 所需要的最少月份数，即：$dp[v] = max(dp[v], dp[u] + time[v - 1])$。
   2. 更新完成所有课程所需要的最少月份数 $ans$，即：$ans = max(ans, dp[v])$。
   3. 相邻节点 $v$ 的入度减 $1$，如果入度减 $1$ 后的节点入度为 0，则将其加入队列 $queue$。
4. 重复 $3$ 的步骤，直到队列中没有节点。
5. 最后返回 $ans$。

### 思路 1：代码

```python
class Solution:
    def minimumTime(self, n: int, relations: List[List[int]], time: List[int]) -> int:
        graph = [[] for _ in range(n + 1)]
        indegrees = [0 for _ in range(n + 1)]

        for u, v in relations:
            graph[u].append(v)
            indegrees[v] += 1

        queue = collections.deque()
        dp = [0 for _ in range(n + 1)]

        ans = 0
        for i in range(1, n + 1):
            if indegrees[i] == 0:
                queue.append(i)
                dp[i] = time[i - 1]
                ans = max(ans, time[i - 1])

        while queue:
            u = queue.popleft()
            for v in graph[u]:
                dp[v] = max(dp[v], dp[u] + time[v - 1])
                ans = max(ans, dp[v])
                indegrees[v] -= 1
                if indegrees[v] == 0:
                    queue.append(v)

        return ans
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(m + n)$，其中 $m$ 为数组 $relations$ 的长度。
- **空间复杂度**：$O(m + n)$。
# [2156. 查找给定哈希值的子串](https://leetcode.cn/problems/find-substring-with-given-hash-value/)

- 标签：字符串、滑动窗口、哈希函数、滚动哈希
- 难度：困难

## 题目链接

- [2156. 查找给定哈希值的子串 - 力扣](https://leetcode.cn/problems/find-substring-with-given-hash-value/)

## 题目大意

**描述**：如果给定整数 `p` 和 `m`，一个长度为 `k` 且下标从 `0` 开始的字符串 `s` 的哈希值按照如下函数计算：

- $hash(s, p, m) = (val(s[0]) * p^0 + val(s[1]) * p^1 + ... + val(s[k-1]) * p^{k-1}) mod m$.

其中 `val(s[i])` 表示 `s[i]` 在字母表中的下标，从 `val('a') = 1` 到 `val('z') = 26`。

现在给定一个字符串 `s` 和整数 `power`，`modulo`，`k` 和 `hashValue` 。

**要求**：返回 `s` 中 第一个 长度为 `k` 的 子串 `sub`，满足 `hash(sub, power, modulo) == hashValue`。

**说明**：

- 子串：定义为一个字符串中连续非空字符组成的序列。
- $1 \le k \le s.length \le 2 * 10^4$。
- $1 \le power, modulo \le 10^9$。
- $0 \le hashValue < modulo$。
- `s` 只包含小写英文字母。
- 测试数据保证一定存在满足条件的子串。

**示例**：

- 示例 1：

```python
输入：s = "leetcode", power = 7, modulo = 20, k = 2, hashValue = 0
输出："ee"
解释："ee" 的哈希值为 hash("ee", 7, 20) = (5 * 1 + 5 * 7) mod 20 = 40 mod 20 = 0 。
"ee" 是长度为 2 的第一个哈希值为 0 的子串，所以我们返回 "ee" 。
```

## 解题思路

### 思路 1：Rabin Karp 算法、滚动哈希算法

这道题目的思想和 Rabin Karp 字符串匹配算法中用到的滚动哈希思想是一样的。不过两者计算的公式是相反的。

- 本题目中的子串哈希计算公式：$hash(s, p, m) = (val(s[i]) * p^0 + val(s[i+1]) * p^1 + ... + val(s[i+k-1]) * p^{k-1}) \mod m$.

- RK 算法中的子串哈希计算公式：$hash(s, p, m) = (val(s[i]) * p^{k-1} + val(s[i+1]) * p^{k-2} + ... + val(s[i+k-1]) * p^0) \mod m$.

可以看出两者的哈希计算公式是反的。

在 RK 算法中，下一个子串的哈希值计算方式为：$Hash(s_{[i + 1, i + k]}) = \{[Hash(s_{[i, i + k - 1]}) - s_i \times d^{k - 1}] \times d + s_{i + k} \times d^{0} \} \mod m$。其中 $Hash(s_{[i, i + k - 1]}$ 为当前子串的哈希值，$Hash(s_{[i + 1, i + k]})$ 为下一个子串的哈希值。

这个公式也可以用文字表示为：**在计算完当前子串的哈希值后，向右滚动字符串，即移除当前子串中最左侧字符的哈希值（$val(s[i]) * p^{k-1}$）之后，再将整体乘以 $p$，再移入最右侧字符的哈希值 $val(s[i+k])$**。

我们可以参考 RK 算法中滚动哈希的计算方式，将其应用到本题中。

因为两者的哈希计算公式相反，所以本题中，我们可以从右侧想左侧逆向遍历字符串，当计算完当前子串的哈希值后，移除当前子串最右侧字符的哈希值（$ val(s[i+k-1]) * p^{k-1}$）之后，再整体乘以 $p$，再移入最左侧字符的哈希值 $val(s[i - 1])$。

在本题中，对应的下一个逆向子串的哈希值计算方式为：$Hash(s_{[i - 1, i + k - 2]}) = \{ [Hash(s_{[i, i + k - 1]}) - s_{i + k - 1} \times d^{k - 1}] \times d + s_{i - 1} \times d^{0} \} \mod m$。其中 $Hash(s_{[i, i + k - 1]})$ 为当前子串的哈希值，$Hash(s_{[i - 1, i + k - 2]})$ 是下一个逆向子串的哈希值。

利用取模运算的两个公式：

- $(a \times b) \mod m = ((a \mod m) \times (b \mod m)) \mod m$
- $(a + b) \mod m = (a \mod m + b \mod m) \mod m$

我们可以把上面的式子转变为：

$$\begin{aligned} Hash(s_{[i - 1, i + k - 2]}) &=  \{[Hash(s_{[i, i + k - 1]}) - s_{i + k - 1} \times d^{k - 1}] \times d + s_{i - 1} \times d^{0} \} \mod m  \cr &= \{[Hash(s_{[i, i + k - 1]}) - s_{i + k - 1} \times d^{k - 1}] \times d \mod m + s_{i - 1} \times d^{0} \mod m \} \mod m \cr &= \{[Hash(s_{[i, i + k - 1]}) - s_{i + k - 1} \times d^{k - 1}] \mod m \times d \mod m + s_{i - 1} \times d^{0} \mod m \} \mod m \end{aligned}$$

> 注意：这里之所以用了「反向迭代」而不是「正向迭代」是因为如果使用了正向迭代，那么每次移除的最左侧字符哈希值为 $val(s[i]) * p^0$，之后整体需要除以 $p$，再移入最右侧字符哈希值为（$val(s[i+k]) * p^{k-1})$）。
>
> 这样就用到了「除法」。而除法是不满足取模运算对应的公式的，所以这里不能用这种方法进行迭代。
>
> 而反向迭代，用到的是乘法。在整个过程中是满足取模运算相关的公式。乘法取余不影响最终结果。

### 思路 1：代码

```python
class Solution:
    def subStrHash(self, s: str, power: int, modulo: int, k: int, hashValue: int) -> str:
        hash_t = 0
        n = len(s)
        for i in range(n - 1, n - k - 1, -1):
            hash_t = (hash_t * power + (ord(s[i]) - ord('a') + 1)) % modulo # 计算最后一个子串的哈希值
    
        h = pow(power, k - 1) % modulo                                      # 计算最高位项，方便后续移除操作
        ans = ""
        if hash_t == hashValue:
            ans = s[n - k: n]
        for i in range(n - k - 1, -1, -1):                                   # 反向迭代，滚动计算子串的哈希值
            hash_t = (hash_t - h * (ord(s[i + k]) - ord('a') + 1)) % modulo  # 移除 s[i + k] 的哈希值
            hash_t = (hash_t * power % modulo + (ord(s[i]) - ord('a') + 1) % modulo) % modulo  # 添加 s[i] 的哈希值
            if hash_t == hashValue:                                          # 如果子串哈希值等于 hashValue，则为答案
                ans = s[i: i + k]
        return ans
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(n)$。其中字符串 $s$ 的长度为 $n$。
- **空间复杂度**：$O(1)$。
# [2172. 数组的最大与和](https://leetcode.cn/problems/maximum-and-sum-of-array/)

- 标签：位运算、数组、动态规划、状态压缩
- 难度：困难

## 题目链接

- [2172. 数组的最大与和 - 力扣](https://leetcode.cn/problems/maximum-and-sum-of-array/)

## 题目大意

**描述**：给定一个长度为 $n$ 的整数数组 $nums$ 和一个整数 $numSlots$ 满足 $2 \times numSlots \ge n$。一共有 $numSlots$ 个篮子，编号为 $1 \sim numSlots$。

现在需要将所有 $n$ 个整数分到这些篮子中，且每个篮子最多有 $2$ 个整数。

**要求**：返回将 $nums$ 中所有数放入 $numSlots$ 个篮子中的最大与和。

**说明**：

- **与和**：当前方案中，每个数与它所在篮子编号的按位与运算结果之和。
  - 比如，将数字 $[1, 3]$ 放入篮子 $1$ 中，$[4, 6]$ 放入篮子 $2$ 中，这个方案的与和为 $(1 \text{ AND } 1) + (3 \text{ AND } 1) + (4 \text{ AND } 2) + (6 \text{ AND } 2) = 1 + 1 + 0 + 2 = 4$。
- $n == nums.length$。
- $1 \le numSlots \le 9$。
- $1 \le n \le 2 \times numSlots$。
- $1 \le nums[i] \le 15$。

**示例**：

- 示例 1：

```python
输入：nums = [1,2,3,4,5,6], numSlots = 3
输出：9
解释：一个可行的方案是 [1, 4] 放入篮子 1 中，[2, 6] 放入篮子 2 中，[3, 5] 放入篮子 3 中。
最大与和为 (1 AND 1) + (4 AND 1) + (2 AND 2) + (6 AND 2) + (3 AND 3) + (5 AND 3) = 1 + 0 + 2 + 2 + 3 + 1 = 9。
```

- 示例 2：

```python
输入：nums = [1,3,10,4,7,1], numSlots = 9
输出：24
解释：一个可行的方案是 [1, 1] 放入篮子 1 中，[3] 放入篮子 3 中，[4] 放入篮子 4 中，[7] 放入篮子 7 中，[10] 放入篮子 9 中。
最大与和为 (1 AND 1) + (1 AND 1) + (3 AND 3) + (4 AND 4) + (7 AND 7) + (10 AND 9) = 1 + 1 + 3 + 4 + 7 + 8 = 24 。
注意，篮子 2 ，5 ，6 和 8 是空的，这是允许的。
```

## 解题思路

### 思路 1：状压 DP

每个篮子最多可分 $2$ 个整数，则我们可以将 $1$ 个篮子分成两个篮子，这样总共有 $2 \times numSlots$ 个篮子，每个篮子中最多可以装 $1$ 个整数。

同时因为 $numSlots$ 的范围为 $[1, 9]$，$2 \times numSlots$ 的范围为 $[2, 19]$，范围不是很大，所以我们可以用「状态压缩」的方式来表示每个篮子中的整数放取情况。

即使用一个 $n \times numSlots$ 位的二进制数 $state$ 来表示每个篮子中的整数放取情况。如果 $state$ 的第 $i$ 位为 $1$，表示第 $i$ 个篮子里边放了整数，如果 $state$ 的第 $i$ 位为 $0$，表示第 $i$ 个篮子为空。

这样，我们就可以通过动态规划的方式来解决这道题。

###### 1. 划分阶段

按照 $2 \times numSlots$ 个篮子中的整数放取情况进行阶段划分。

###### 2. 定义状态

定义当前每个篮子中的整数放取情况为 $state$，$state$ 对应选择的整数个数为 $count(state)$。

则可以定义状态 $dp[state]$ 表示为：将前 $count(state)$ 个整数放到篮子里，并且每个篮子中的整数放取情况为 $state$ 时，可以获得的最大与和。

###### 3. 状态转移方程

对于当前状态 $dp[state]$，肯定是从比 $state$ 少选一个元素的状态中递推而来。我们可以枚举少选一个元素的状态，找到可以获得的最大与和，赋值给 $dp[state]$。

即状态转移方程为：$dp[state] = min(dp[state], dp[state \oplus (1 \text{ <}\text{< } i)] + (i // 2 + 1) \text{ \& } nums[one\underline{\hspace{0.5em}}cnt - 1])$，其中：

1. $state$ 第 $i$ 位一定为 $1$。
2. $state \oplus (1 \text{ <}\text{< } i)$ 为比 $state$ 少选一个元素的状态。
3. $i // 2 + 1$ 为篮子对应编号
4. $nums[one\underline{\hspace{0.5em}}cnt - 1]$ 为当前正在考虑的数组元素。

###### 4. 初始条件

- 初始每个篮子中都没有放整数的情况下，可以获得的最大与和为 $0$，即 $dp[0] = 0$。

###### 5. 最终结果

根据我们之前定义的状态，$dp[state]$ 表示为：将前 $count(state)$ 个整数放到篮子里，并且每个篮子中的整数放取情况为 $state$ 时，可以获得的最大与和。所以最终结果为 $max(dp)$。

> 注意：当 $one\underline{\hspace{0.5em}}cnt > len(nums)$ 时，无法通过递推得到 $dp[state]$，需要跳过。

### 思路 1：代码

```python
class Solution:
    def maximumANDSum(self, nums: List[int], numSlots: int) -> int:
        states = 1 << (numSlots * 2)
        dp = [0 for _ in range(states)]

        for state in range(states):
            one_cnt = bin(state).count('1')
            if one_cnt > len(nums):
                continue
            for i in range(numSlots * 2):
                if (state >> i) & 1:
                    dp[state] = max(dp[state], dp[state ^ (1 << i)] + ((i // 2 + 1) & nums[one_cnt - 1]))
        
        return max(dp)
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(2^m \times m)$，其中 $m = 2 \times numSlots$。
- **空间复杂度**：$O(2^m)$。

# [2235. 两整数相加](https://leetcode.cn/problems/add-two-integers/)

- 标签：数学
- 难度：简单

## 题目链接

- [2235. 两整数相加 - 力扣](https://leetcode.cn/problems/add-two-integers/)

## 题目大意

**描述**：给定两个整数 $num1$ 和 $num2$。

**要求**：返回这两个整数的和。

**说明**：

- $-100 \le num1, num2 \le 100$。

**示例**：

- 示例 1：

```python
示例 1：
输入：num1 = 12, num2 = 5
输出：17
解释：num1 是 12，num2 是 5，它们的和是 12 + 5 = 17，因此返回 17。
```

- 示例 2：

```python
输入：num1 = -10, num2 = 4
输出：-6
解释：num1 + num2 = -6，因此返回 -6。
```

## 解题思路

### 思路 1：直接计算

1. 直接计算整数 $num1$ 与 $num2$ 的和，返回 $num1 + num2$ 即可。

### 思路 1：代码

```python
class Solution:
    def sum(self, num1: int, num2: int) -> int:
        return num1 + num2
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(1)$。
- **空间复杂度**：$O(1)$。
# [2246. 相邻字符不同的最长路径](https://leetcode.cn/problems/longest-path-with-different-adjacent-characters/)

- 标签：树、深度优先搜索、图、拓扑排序、数组、字符串
- 难度：困难

## 题目链接

- [2246. 相邻字符不同的最长路径 - 力扣](https://leetcode.cn/problems/longest-path-with-different-adjacent-characters/)

## 题目大意

**描述**：给定一个长度为 $n$ 的数组 $parent$ 来表示一棵树（即一个连通、无向、无环图）。该树的节点编号为 $0 \sim n - 1$，共 $n$ 个节点，其中根节点的编号为 $0$。其中 $parent[i]$ 表示节点 $i$ 的父节点，由于节点 $0$ 是根节点，所以 $parent[0] == -1$。再给定一个长度为 $n$ 的字符串，其中 $s[i]$ 表示分配给节点 $i$ 的字符。

**要求**：找出路径上任意一对相邻节点都没有分配到相同字符的最长路径，并返回该路径的长度。

**说明**：

- $n == parent.length == s.length$。
- $1 \le n \le 10^5$。
- 对所有 $i \ge 1$ ，$0 \le parent[i] \le n - 1$ 均成立。
- $parent[0] == -1$。
- $parent$ 表示一棵有效的树。
- $s$ 仅由小写英文字母组成。

**示例**：

- 示例 1：

![](https://assets.leetcode.com/uploads/2022/03/25/testingdrawio.png)

```python
输入：parent = [-1,0,0,1,1,2], s = "abacbe"
输出：3
解释：任意一对相邻节点字符都不同的最长路径是：0 -> 1 -> 3 。该路径的长度是 3 ，所以返回 3。
可以证明不存在满足上述条件且比 3 更长的路径。
```

- 示例 2：

![](https://assets.leetcode.com/uploads/2022/03/25/graph2drawio.png)

```python
输入：parent = [-1,0,0,0], s = "aabc"
输出：3
解释：任意一对相邻节点字符都不同的最长路径是：2 -> 0 -> 3 。该路径的长度为 3 ，所以返回 3。
```

## 解题思路

### 思路 1：树形 DP + 深度优先搜索

因为题目给定的是表示父子节点的 $parent$  数组，为了方便递归遍历相邻节点，我们可以根据 $partent$ 数组，建立一个由父节点指向子节点的有向图 $graph$。

如果不考虑相邻节点是否为相同字符这一条件，那么这道题就是在求树的直径（树的最长路径长度）中的节点个数。

对于根节点为 $u$ 的树来说：

1. 如果其最长路径经过根节点 $u$，则 **最长路径长度 = 某子树中的最长路径长度 + 另一子树中的最长路径长度 + 1**。
2. 如果其最长路径不经过根节点 $u$，则 **最长路径长度 = 某个子树中的最长路径长度**。

即：**最长路径长度 = max(某子树中的最长路径长度 + 另一子树中的最长路径长度 + 1，某个子树中的最长路径长度)**。

对此，我们可以使用深度优先搜索递归遍历 $u$ 的所有相邻节点 $v$，并在递归遍历的同时，维护一个全局最大路径和变量 $ans$，以及当前节点 $u$ 的最大路径长度变量 $u\underline{\hspace{0.5em}}len$。

1. 先计算出从相邻节点 $v$ 出发的最长路径长度 $v\underline{\hspace{0.5em}}len$。
2. 更新维护全局最长路径长度为 $self.ans = max(self.ans, \quad u\underline{\hspace{0.5em}}len + v\underline{\hspace{0.5em}}len + 1)$。
3. 更新维护当前节点 $u$ 的最长路径长度为 $u\underline{\hspace{0.5em}}len = max(u\underline{\hspace{0.5em}}len, \quad v\underline{\hspace{0.5em}}len + 1)$。

因为题目限定了「相邻节点字符不同」，所以在更新全局最长路径长度和当前节点 $u$ 的最长路径长度时，我们需要判断一下节点 $u$ 与相邻节点 $v$ 的字符是否相同，只有在字符不同的条件下，才能够更新维护。

最后，因为题目要求的是树的直径（树的最长路径长度）中的节点个数，而：**路径的节点 = 路径长度 + 1**，所以最后我们返回 $self.ans + 1$ 作为答案。

### 思路 1：代码

```python
class Solution:
    def longestPath(self, parent: List[int], s: str) -> int:
        size = len(parent)

        # 根据 parent 数组，建立有向图
        graph = [[] for _ in range(size)]
        for i in range(1, size):
            graph[parent[i]].append(i)

        ans = 0
        def dfs(u):
            nonlocal ans
            u_len = 0                                   # u 节点的最大路径长度
            for v in graph[u]:                          # 遍历 u 节点的相邻节点
                v_len = dfs(v)                          # 相邻节点的最大路径长度
                if s[u] != s[v]:                        # 相邻节点字符不同
                    ans = max(ans, u_len + v_len + 1)   # 维护最大路径长度
                    u_len = max(u_len, v_len + 1)       # 更新 u 节点的最大路径长度
            return u_len                                # 返回 u 节点的最大路径长度

        dfs(0)
        return ans + 1
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(n)$，其中 $n$ 是树的节点数目。
- **空间复杂度**：$O(n)$。
# [2249. 统计圆内格点数目](https://leetcode.cn/problems/count-lattice-points-inside-a-circle/)

- 标签：几何、数组、哈希表、数学、枚举
- 难度：中等

## 题目链接

- [2249. 统计圆内格点数目 - 力扣](https://leetcode.cn/problems/count-lattice-points-inside-a-circle/)

## 题目大意

**描述**：给定一个二维整数数组 `circles`。其中 `circles[i] = [xi, yi, ri]` 表示网格上圆心为 `(xi, yi)` 且半径为 `ri` 的第 $i$ 个圆。

**要求**：返回出现在至少一个圆内的格点数目。

**说明**：

- **格点**：指的是整数坐标对应的点。
- 圆周上的点也被视为出现在圆内的点。
- $1 \le circles.length \le 200$。
- $circles[i].length == 3$。
- $1 \le xi, yi \le 100$。
- $1 \le ri \le min(xi, yi)$。

**示例**：

- 示例 1：

![](https://assets.leetcode.com/uploads/2022/03/02/exa-11.png)

```python
输入：circles = [[2,2,1]]
输出：5
解释：
给定的圆如上图所示。
出现在圆内的格点为 (1, 2)、(2, 1)、(2, 2)、(2, 3) 和 (3, 2)，在图中用绿色标识。
像 (1, 1) 和 (1, 3) 这样用红色标识的点，并未出现在圆内。
因此，出现在至少一个圆内的格点数目是 5。
```

- 示例 2：

```python
输入：circles = [[2,2,2],[3,4,1]]
输出：16
解释：
给定的圆如上图所示。
共有 16 个格点出现在至少一个圆内。
其中部分点的坐标是 (0, 2)、(2, 0)、(2, 4)、(3, 2) 和 (4, 4)。
```

## 解题思路

### 思路 1：枚举算法

题目要求中 $1 \le xi, yi \le 100$，$1 \le ri \le min(xi, yi)$。则圆中点的范围为 $1 \le x, y \le 200$。

我们可以枚举所有坐标和所有圆，检测该坐标是否在圆中。

为了优化枚举范围，我们可以先遍历一遍所有圆，计算最小、最大的 $x$、$y$ 范围，再枚举所有坐标和所有圆，并进行检测。

### 思路 1：代码

```python
class Solution:
    def countLatticePoints(self, circles: List[List[int]]) -> int:
        min_x, min_y = 200, 200
        max_x, max_y = 0, 0
        for circle in circles:
            if circle[0] + circle[2] > max_x:
                max_x = circle[0] + circle[2]
            if circle[0] - circle[2] < min_x:
                min_x = circle[0] - circle[2]
            if circle[1] + circle[2] > max_y:
                max_y = circle[1] + circle[2]
            if circle[1] - circle[2] < min_y:
                min_y = circle[1] - circle[2]
        
        ans = 0
        for x in range(min_x, max_x + 1):
            for y in range(min_y, max_y + 1):
                for xi, yi, ri in circles:
                    if (xi - x) * (xi - x) + (yi - y) * (yi - y) <= ri * ri:
                        ans += 1
                        break
        
        return ans
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(x \times y)$，其中 $x$、$y$ 分别为横纵坐标的个数。
- **空间复杂度**：$O(1)$。
# [2276. 统计区间中的整数数目](https://leetcode.cn/problems/count-integers-in-intervals/)

- 标签：设计、线段树、有序集合
- 难度：困难

## 题目链接

- [2276. 统计区间中的整数数目 - 力扣](https://leetcode.cn/problems/count-integers-in-intervals/)

## 题目大意

**描述**：给定一个区间的空集。

**要求**：设计并实现满足要求的数据结构：

- 新增：添加一个区间到这个区间集合中。
- 统计：计算出现在 至少一个 区间中的整数个数。

实现 CountIntervals 类：

- `CountIntervals()` 使用区间的空集初始化对象
- `void add(int left, int right)` 添加区间 `[left, right]` 到区间集合之中。
- `int count()` 返回出现在 至少一个 区间中的整数个数。

**说明**：

- 区间 `[left, right]` 表示满足 $left \le x \le right$ 的所有整数 `x`。
- $1 \le left \le right \le 10^9$。
- 最多调用 `add` 和 `count` 方法 **总计** $10^5$ 次。
- 调用 `count` 方法至少一次。

**示例**：

- 示例 1：

```python
输入：
["CountIntervals", "add", "add", "count", "add", "count"]
[[], [2, 3], [7, 10], [], [5, 8], []]
输出：
[null, null, null, 6, null, 8]

解释：
CountIntervals countIntervals = new CountIntervals(); // 用一个区间空集初始化对象
countIntervals.add(2, 3);  // 将 [2, 3] 添加到区间集合中
countIntervals.add(7, 10); // 将 [7, 10] 添加到区间集合中
countIntervals.count();    // 返回 6
                           // 整数 2 和 3 出现在区间 [2, 3] 中
                           // 整数 7、8、9、10 出现在区间 [7, 10] 中
countIntervals.add(5, 8);  // 将 [5, 8] 添加到区间集合中
countIntervals.count();    // 返回 8
                           // 整数 2 和 3 出现在区间 [2, 3] 中
                           // 整数 5 和 6 出现在区间 [5, 8] 中
                           // 整数 7 和 8 出现在区间 [5, 8] 和区间 [7, 10] 中
                           // 整数 9 和 10 出现在区间 [7, 10] 中
```

## 解题思路

### 思路 1：动态开点线段树

这道题可以使用线段树来做。

因为区间的范围是 $[1, 10^9]$，普通数组构成的线段树不满足要求。需要用到动态开点线段树。具体做法如下：

- 初始化方法，构建一棵线段树。每个线段树的节点类存储当前区间中保存的元素个数。

- 在 `add` 方法中，将区间 `[left, right]` 上的每个元素值赋值为 `1`，则区间值为 `right - left + 1`。

- 在 `count` 方法中，返回区间 $[0, 10^9]$ 的区间值（即区间内元素个数）。

### 思路 1：动态开点线段树代码

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
    
    # 区间更新接口：将区间为 [q_left, q_right] 上的元素值修改为 val
    def update_interval(self, q_left, q_right, val):
        self.__update_interval(q_left, q_right, val, self.tree)
    
    # 区间查询接口：查询区间为 [q_left, q_right] 的区间值
    def query_interval(self, q_left, q_right):
        return self.__query_interval(q_left, q_right, self.tree)
            
    
    # 以下为内部实现方法
    
    # 区间更新实现方法
    def __update_interval(self, q_left, q_right, val, node):
        if node.left >= q_left and node.right <= q_right:  # 节点所在区间被 [q_left, q_right] 所覆盖
            node.lazy_tag = val                     # 将当前节点的延迟标记标记为 val
            interval_size = (node.right - node.left + 1)    # 当前节点所在区间大小
            node.val = val * interval_size          # 当前节点所在区间每个元素值改为 val
            return
        if node.right < q_left or node.left > q_right:  # 节点所在区间与 [q_left, q_right] 无关
            return
    
        self.__pushdown(node)                       # 向下更新节点所在区间的左右子节点的值和懒惰标记
    
        if q_left <= node.mid:                      # 在左子树中更新区间值
            self.__update_interval(q_left, q_right, val, node.leftNode)
        if q_right > node.mid:                      # 在右子树中更新区间值
            self.__update_interval(q_left, q_right, val, node.rightNode)
            
        self.__pushup(node)
    
    # 区间查询实现方法：在线段树的 [left, right] 区间范围中搜索区间为 [q_left, q_right] 的区间值
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
    
    # 向上更新实现方法：更新 node 节点区间值 等于 该节点左右子节点元素值的聚合计算结果
    def __pushup(self, node):
        if node.leftNode and node.rightNode:
            node.val = self.function(node.leftNode.val, node.rightNode.val)
            
    # 向下更新实现方法：更新 node 节点所在区间的左右子节点的值和懒惰标记
    def __pushdown(self, node):
        if node.leftNode is None:
            node.leftNode = SegTreeNode(node.left, node.mid)
        if node.rightNode is None:
            node.rightNode = SegTreeNode(node.mid + 1, node.right)
            
        lazy_tag = node.lazy_tag
        if node.lazy_tag is None:
            return
            
        node.leftNode.lazy_tag = lazy_tag           # 更新左子节点懒惰标记
        left_size = (node.leftNode.right - node.leftNode.left + 1)
        node.leftNode.val = lazy_tag * left_size    # 更新左子节点值
        
        node.rightNode.lazy_tag = lazy_tag          # 更新右子节点懒惰标记
        right_size = (node.rightNode.right - node.rightNode.left + 1)
        node.rightNode.val = lazy_tag * right_size  # 更新右子节点值
        
        node.lazy_tag = None                        # 更新当前节点的懒惰标记
    
    
class CountIntervals:

    def __init__(self):
        self.STree = SegmentTree(lambda x, y: x + y)
        self.left = 10 ** 9
        self.right = 0


    def add(self, left: int, right: int) -> None:
        self.STree.update_interval(left, right, 1) 


    def count(self) -> int:
        return self.STree.query_interval(0, int(1e9))



# Your CountIntervals object will be instantiated and called as such:
# obj = CountIntervals()
# obj.add(left,right)
# param_2 = obj.count()
```

# [2376. 统计特殊整数](https://leetcode.cn/problems/count-special-integers/)

- 标签：数学、动态规划
- 难度：困难

## 题目链接

- [2376. 统计特殊整数 - 力扣](https://leetcode.cn/problems/count-special-integers/)

## 题目大意

**描述**：给定一个正整数 $n$。

**要求**：求区间 $[1, n]$ 内的所有整数中，特殊整数的数目。

**说明**：

- **特殊整数**：如果一个正整数的每一个数位都是互不相同的，则称它是特殊整数。
- $1 \le n \le 2 \times 10^9$。

**示例**：

- 示例 1：

```python
输入：n = 20
输出：19
解释：1 到 20 之间所有整数除了 11 以外都是特殊整数。所以总共有 19 个特殊整数。
```

- 示例 2：

```python
输入：n = 5
输出：5
解释：1 到 5 所有整数都是特殊整数。
```

## 解题思路

### 思路 1：动态规划 + 数位 DP

将 $n$ 转换为字符串 $s$，定义递归函数 `def dfs(pos, state, isLimit, isNum):` 表示构造第 $pos$ 位及之后所有数位的合法方案数。接下来按照如下步骤进行递归。

1. 从 `dfs(0, 0, True, False)` 开始递归。 `dfs(0, 0, True, False)` 表示：
      1. 从位置 $0$ 开始构造。
      2. 初始没有使用数字（即前一位所选数字集合为 $0$）。
      3. 开始时受到数字 $n$ 对应最高位数位的约束。
      4. 开始时没有填写数字。

2. 如果遇到  $pos == len(s)$，表示到达数位末尾，此时：
      1. 如果 $isNum == True$，说明当前方案符合要求，则返回方案数 $1$。
      2. 如果 $isNum == False$，说明当前方案不符合要求，则返回方案数 $0$。

3. 如果 $pos \ne len(s)$，则定义方案数 $ans$，令其等于 $0$，即：`ans = 0`。
4. 如果遇到 $isNum == False$，说明之前位数没有填写数字，当前位可以跳过，这种情况下方案数等于 $pos + 1$ 位置上没有受到 $pos$ 位的约束，并且之前没有填写数字时的方案数，即：`ans = dfs(i + 1, state, False, False)`。
5. 如果 $isNum == True$，则当前位必须填写一个数字。此时：
      1. 根据 $isNum$ 和 $isLimit$ 来决定填当前位数位所能选择的最小数字（$minX$）和所能选择的最大数字（$maxX$），
      2. 然后根据 $[minX, maxX]$ 来枚举能够填入的数字 $d$。
      3. 如果之前没有选择 $d$，即 $d$ 不在之前选择的数字集合 $state$ 中，则方案数累加上当前位选择 $d$ 之后的方案数，即：`ans += dfs(pos + 1, state | (1 << d), isLimit and d == maxX, True)`。
            1. `state | (1 << d)` 表示之前选择的数字集合 $state$ 加上 $d$。
            2. `isLimit and d == maxX` 表示 $pos + 1$ 位受到之前 $pos$ 位限制。
            3. $isNum == True$ 表示 $pos$ 位选择了数字。

6. 最后的方案数为 `dfs(0, 0, True, False)`，将其返回即可。

### 思路 1：代码

```python
class Solution:
    def countSpecialNumbers(self, n: int) -> int:
        # 将 n 转换为字符串 s
        s = str(n)
        
        @cache
        # pos: 第 pos 个数位
        # state: 之前选过的数字集合。
        # isLimit: 表示是否受到选择限制。如果为真，则第 pos 位填入数字最多为 s[pos]；如果为假，则最大可为 9。
        # isNum: 表示 pos 前面的数位是否填了数字。如果为真，则当前位不可跳过；如果为假，则当前位可跳过。
        def dfs(pos, state, isLimit, isNum):
            if pos == len(s):
                # isNum 为 True，则表示当前方案符合要求
                return int(isNum)
            
            ans = 0
            if not isNum:
                # 如果 isNum 为 False，则可以跳过当前数位
                ans = dfs(pos + 1, state, False, False)
            
            # 如果前一位没有填写数字，则最小可选择数字为 0，否则最少为 1（不能含有前导 0）。
            minX = 0 if isNum else 1
            # 如果受到选择限制，则最大可选择数字为 s[pos]，否则最大可选择数字为 9。
            maxX = int(s[pos]) if isLimit else 9
            
            # 枚举可选择的数字
            for d in range(minX, maxX + 1): 
                # d 不在选择的数字集合中，即之前没有选择过 d
                if (state >> d) & 1 == 0:
                    ans += dfs(pos + 1, state | (1 << d), isLimit and d == maxX, True)
            return ans
    
        return dfs(0, 0, True, False)
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(\log n \times 10 \times 2^{10})$，其中 $n$ 为给定整数。
- **空间复杂度**：$O(\log n \times 2^{10})$。
# [2427. 公因子的数目](https://leetcode.cn/problems/number-of-common-factors/)

- 标签：数学、枚举、数论
- 难度：简单

## 题目链接

- [2427. 公因子的数目 - 力扣](https://leetcode.cn/problems/number-of-common-factors/)

## 题目大意

**描述**：给定两个正整数 $a$ 和 $b$。

**要求**：返回 $a$ 和 $b$ 的公因子数目。

**说明**：

- **公因子**：如果 $x$ 可以同时整除 $a$ 和 $b$，则认为 $x$ 是 $a$ 和 $b$ 的一个公因子。
- $1 \le a, b \le 1000$。

**示例**：

- 示例 1：

```python
输入：a = 12, b = 6
输出：4
解释：12 和 6 的公因子是 1、2、3、6。
```

- 示例 2：

```python
输入：a = 25, b = 30
输出：2
解释：25 和 30 的公因子是 1、5。
```

## 解题思路

### 思路 1：枚举算法

最直接的思路就是枚举所有 $[1, min(a, b)]$ 之间的数，并检查是否能同时整除 $a$ 和 $b$。

当然，因为 $a$ 与 $b$ 的公因子肯定不会超过 $a$ 与 $b$ 的最大公因数，则我们可以直接枚举 $[1, gcd(a, b)]$ 之间的数即可，其中 $gcd(a, b)$ 是 $a$ 与 $b$ 的最大公约数。

### 思路 1：代码

```python
class Solution:
    def commonFactors(self, a: int, b: int) -> int:
        ans = 0
        for i in range(1, math.gcd(a, b) + 1):
            if a % i == 0 and b % i == 0:
                ans += 1
        return ans
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(\sqrt{min(a, b)})$。
- **空间复杂度**：$O(1)$。
# [2538. 最大价值和与最小价值和的差值](https://leetcode.cn/problems/difference-between-maximum-and-minimum-price-sum/)

- 标签：树、深度优先搜索、数组、动态规划
- 难度：困难

## 题目链接

- [2538. 最大价值和与最小价值和的差值 - 力扣](https://leetcode.cn/problems/difference-between-maximum-and-minimum-price-sum/)

## 题目大意

**描述**：给定一个整数 $n$ 和一个长度为 $n - 1$ 的二维整数数组 $edges$ 用于表示一个 $n$ 个节点的无向无根图，节点编号为 $0 \sim n - 1$。其中 $edges[i] = [ai, bi]$ 表示树中节点 $ai$ 和 $bi$ 之间有一条边。再给定一个整数数组 $price$，其中 $price[i]$ 表示图中节点 $i$ 的价值。

一条路径的价值和是这条路径上所有节点的价值之和。

你可以选择树中任意一个节点作为根节点 $root$。选择 $root$ 为根的开销是以 $root$ 为起点的所有路径中，价值和最大的一条路径与最小的一条路径的差值。

**要求**：返回所有节点作为根节点的选择中，最大的开销为多少。

**说明**：

- $1 \le n \le 10^5$。
- $edges.length == n - 1$。
- $0 \le ai, bi \le n - 1$。
- $edges$ 表示一棵符合题面要求的树。
- $price.length == n$。
- $1 \le price[i] \le 10^5$。

**示例**：

- 示例 1：

![](https://assets.leetcode.com/uploads/2022/12/01/example14.png)

```python
输入：n = 6, edges = [[0,1],[1,2],[1,3],[3,4],[3,5]], price = [9,8,7,6,10,5]
输出：24
解释：上图展示了以节点 2 为根的树。左图（红色的节点）是最大价值和路径，右图（蓝色的节点）是最小价值和路径。
- 第一条路径节点为 [2,1,3,4]：价值为 [7,8,6,10] ，价值和为 31 。
- 第二条路径节点为 [2] ，价值为 [7] 。
最大路径和与最小路径和的差值为 24 。24 是所有方案中的最大开销。
```

- 示例 2：

![](https://assets.leetcode.com/uploads/2022/11/24/p1_example2.png)

```python
输入：n = 3, edges = [[0,1],[1,2]], price = [1,1,1]
输出：2
解释：上图展示了以节点 0 为根的树。左图（红色的节点）是最大价值和路径，右图（蓝色的节点）是最小价值和路径。
- 第一条路径包含节点 [0,1,2]：价值为 [1,1,1] ，价值和为 3 。
- 第二条路径节点为 [0] ，价值为 [1] 。
最大路径和与最小路径和的差值为 2 。2 是所有方案中的最大开销。
```

## 解题思路

### 思路 1：树形 DP + 深度优先搜索

1. 因为 $price$ 数组中元素都为正数，所以价值和最小的一条路径一定为「单个节点」，也就是根节点 $root$ 本身。
2. 因为价值和最大的路径是从根节点 $root$ 出发的价值和最大的一条路径，所以「最大的开销」等于「从根节点 $root$ 出发的价值和最大的一条路径」与「路径中一个端点值」 的差值。
3. 价值和最大的路径的两个端点中，一个端点为根节点 $root$，另一个节点为叶子节点。

这样问题就变为了求树中一条路径，使得路径的价值和减去其中一个端点值的权值最大。

对此我们可以使用深度优先搜索递归遍历二叉树，并在递归遍历的同时，维护一个最大开销变量 $ans$。

然后定义函数 ` def dfs(self, u, father):` 计算以节点 $u$ 为根节点的子树中，带端点的最大路径和 $max\underline{\hspace{0.5em}}s1$，以及去掉端点的最大路径和 $max\underline{\hspace{0.5em}}s2$，其中 $father$ 表示节点 $u$ 的根节点，用于遍历邻接节点的过程中过滤父节点，避免重复遍历。

初始化带端点的最大路径和 $max\underline{\hspace{0.5em}}s1$ 为 $price[u]$，表示当前只有一个节点，初始化去掉端点的最大路径和 $max\underline{\hspace{0.5em}}s2$ 为 $0$，表示当前没有节点。

然后在遍历节点 $u$ 的相邻节点 $v$ 时，递归调用 $dfs(v, u)$，获取以节点 $v$ 为根节点的子树中，带端点的最大路径和 $s1$，以及去掉端点的最大路径和 $s2$。此时最大开销变量 $self.ans$ 有两种情况：

1. $u$ 的子树中带端点的最大路径和，加上 $v$ 的子树中不带端点的最大路径和，即：$max\underline{\hspace{0.5em}}s1 + s2$。
2. $u$ 的子树中去掉端点的最大路径和，加上 $v$ 的子树中带端点的最大路径和，即：$max\underline{\hspace{0.5em}}s2 + s1$。

此时我们更新最大开销变量 $self.ans$，即：$self.ans = max(self.ans, \quad max\underline{\hspace{0.5em}}s1 + s2, \quad  max\underline{\hspace{0.5em}}s2 + s1)$。

然后更新 $u$ 的子树中带端点的最大路径和 $max\underline{\hspace{0.5em}}s1$，即：$max\underline{\hspace{0.5em}}s1= max(max\underline{\hspace{0.5em}}s1, \quad s1 + price[u])$。

再更新 $u$ 的子树中去掉端点的最大路径和 $max\underline{\hspace{0.5em}}s2$，即：$max\underline{\hspace{0.5em}}s2 = max(max\underline{\hspace{0.5em}}s2, \quad s2 + price[u])$。

最后返回带端点 $u$ 的最大路径和 $max\underline{\hspace{0.5em}}s1$，以及去掉端点 $u$ 的最大路径和 $。

最终，最大开销变量 $self.ans$ 即为答案。

### 思路 1：代码

```python
class Solution:
    def __init__(self):
        self.ans = 0
        
    def dfs(self, graph, price, u, father):
        max_s1 = price[u]
        max_s2 = 0
        for v in graph[u]:
            if v == father:    # 过滤父节点，避免重复遍历
                continue
            s1, s2 = self.dfs(graph, price, v, u)
            self.ans = max(self.ans, max_s1 + s2, max_s2 + s1)
            max_s1 = max(max_s1, s1 + price[u])
            max_s2 = max(max_s2, s2 + price[u])
        return max_s1, max_s2

    def maxOutput(self, n: int, edges: List[List[int]], price: List[int]) -> int:
        graph = [[] for _ in range(n)]
        for u, v in edges:
            graph[u].append(v)
            graph[v].append(u)

        self.dfs(graph, price, 0, -1)
        return self.ans
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(n)$，其中 $n$ 为树中节点个数。
- **空间复杂度**：$O(n)$。

## 参考链接

- 【题解】[二维差分模板 双指针 树形DP 树的直径【力扣周赛 328】](https://www.bilibili.com/video/BV1QT41127kJ/)
- 【题解】[2538. 最大价值和与最小价值和的差值 题解](https://github.com/doocs/leetcode/blob/main/solution/2500-2599/2538.Difference Between Maximum and Minimum Price Sum/README.md)
# [2585. 获得分数的方法数](https://leetcode.cn/problems/number-of-ways-to-earn-points/)

- 标签：数组、动态规划
- 难度：困难

## 题目链接

- [2585. 获得分数的方法数 - 力扣](https://leetcode.cn/problems/number-of-ways-to-earn-points/)

## 题目大意

**描述**：考试中有 $n$ 种类型的题目。给定一个整数 $target$ 和一个下标从 $0$ 开始的二维整数数组 $types$，其中 $types[i] = [count_i, marks_i]$ 表示第 $i$ 种类型的题目有 $count_i$ 道，每道题目对应 $marks_i$ 分。

**要求**：返回你在考试中恰好得到 $target$ 分的方法数。由于答案可能很大，结果需要对 $10^9 + 7$ 取余。

**说明**：

- 同类型题目无法区分。比如说，如果有 $3$ 道同类型题目，那么解答第 $1$ 和第 $2$ 道题目与解答第 $1$ 和第 $3$ 道题目或者第 $2$ 和第 $3$ 道题目是相同的。
- $1 \le target \le 1000$。
- $n == types.length$。
- $1 \le n \le 50$。
- $types[i].length == 2$。
- $1 \le counti, marksi \le 50$。

**示例**：

- 示例 1：

```python
输入：target = 6, types = [[6,1],[3,2],[2,3]]
输出：7
解释：要获得 6 分，你可以选择以下七种方法之一：
- 解决 6 道第 0 种类型的题目：1 + 1 + 1 + 1 + 1 + 1 = 6
- 解决 4 道第 0 种类型的题目和 1 道第 1 种类型的题目：1 + 1 + 1 + 1 + 2 = 6
- 解决 2 道第 0 种类型的题目和 2 道第 1 种类型的题目：1 + 1 + 2 + 2 = 6
- 解决 3 道第 0 种类型的题目和 1 道第 2 种类型的题目：1 + 1 + 1 + 3 = 6
- 解决 1 道第 0 种类型的题目、1 道第 1 种类型的题目和 1 道第 2 种类型的题目：1 + 2 + 3 = 6
- 解决 3 道第 1 种类型的题目：2 + 2 + 2 = 6
- 解决 2 道第 2 种类型的题目：3 + 3 = 6
```

- 示例 2：

```python
输入：target = 5, types = [[50,1],[50,2],[50,5]]
输出：4
解释：要获得 5 分，你可以选择以下四种方法之一：
- 解决 5 道第 0 种类型的题目：1 + 1 + 1 + 1 + 1 = 5
- 解决 3 道第 0 种类型的题目和 1 道第 1 种类型的题目：1 + 1 + 1 + 2 = 5
- 解决 1 道第 0 种类型的题目和 2 道第 1 种类型的题目：1 + 2 + 2 = 5
- 解决 1 道第 2 种类型的题目：5
```

## 解题思路

### 思路 1：动态规划

###### 1. 划分阶段

按照进行阶段划分。

###### 2. 定义状态

定义状态 $dp[i][w]$ 表示为：前 $i$ 种题目恰好组成 $w$ 分的方案数。

###### 3. 状态转移方程

前 $i$ 种题目恰好组成 $w$ 分的方案数，等于前 $i - 1$ 种问题恰好组成 $w - k \times marks_i$ 分的方案数总和，即状态转移方程为：$dp[i][w] = \sum_{k = 0} dp[i - 1][w - k \times marks_i]$。

###### 4. 初始条件

- 前 $0$ 种题目恰好组成 $0$ 分的方案数为 $1$。

###### 5. 最终结果

根据我们之前定义的状态， $dp[i][w]$ 表示为：前 $i$ 种题目恰好组成 $w$ 分的方案数。 所以最终结果为 $dp[size][target]$。

### 思路 1：代码

```python
class Solution:    
    def waysToReachTarget(self, target: int, types: List[List[int]]) -> int:
        size = len(types)
        group_count = [types[i][0] for i in range(len(types))]
        weight = [[(types[i][1] * k) for k in range(types[i][0] + 1)] for i in range(len(types))]
        mod = 1000000007
            
        dp = [[0 for _ in range(target + 1)] for _ in range(size + 1)]
        dp[0][0] = 1
        
        # 枚举前 i 组物品
        for i in range(1, size + 1):
            # 枚举背包装载重量
            for w in range(target + 1):
                # 枚举第 i 组物品能取个数
                dp[i][w] = dp[i - 1][w]
                for k in range(1, group_count[i - 1] + 1):
                    if w >= weight[i - 1][k]:
                        dp[i][w] += dp[i - 1][w - weight[i - 1][k]]
                        dp[i][w] %= mod
        
        return dp[size][target]
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(n \times target \times m)$，其中 $n$ 为题目种类数，$target$ 为目标分数，$m$ 为每种题目的最大分数。
- **空间复杂度**：$O(n \times target)$。

# [2719. 统计整数数目](https://leetcode.cn/problems/count-of-integers/)

- 标签：数学、字符串、动态规划
- 难度：困难

## 题目链接

- [2719. 统计整数数目 - 力扣](https://leetcode.cn/problems/count-of-integers/)

## 题目大意

**描述**：给定两个数字字符串 $num1$ 和 $num2$，以及两个整数 $max\underline{\hspace{0.5em}}sum$ 和 $min\underline{\hspace{0.5em}}sum$。

**要求**：返回好整数的数目。答案可能很大，请返回答案对 $10^9 + 7$ 取余后的结果。

**说明**：

- **好整数**：如果一个整数 $x$ 满足一下条件，我们称它是一个好整数：
  - $num1 \le x \le num2$。
  - $num\underline{\hspace{0.5em}}sum \le digit\underline{\hspace{0.5em}}sum(x) \le max\underline{\hspace{0.5em}}sum$。

- $digit\underline{\hspace{0.5em}}sum(x)$ 表示 $x$ 各位数字之和。
- $1 \le num1 \le num2 \le 10^{22}$。
- $1 \le min\underline{\hspace{0.5em}}sum \le max\underline{\hspace{0.5em}}sum \le 400$。

**示例**：

- 示例 1：

```python
输入：num1 = "1", num2 = "12", min_num = 1, max_num = 8
输出：11
解释：总共有 11 个整数的数位和在 1 到 8 之间，分别是 1,2,3,4,5,6,7,8,10,11 和 12 。所以我们返回 11。
```

- 示例 2：

```python
输入：num1 = "1", num2 = "5", min_num = 1, max_num = 5
输出：5
解释：数位和在 1 到 5 之间的 5 个整数分别为 1,2,3,4 和 5 。所以我们返回 5。
```

## 解题思路

### 思路 1：动态规划 + 数位 DP

将 $num1$ 补上前导 $0$，补到和 $num2$ 长度一致，定义递归函数 `def dfs(pos, total, isMaxLimit, isMinLimit):` 表示构造第 $pos$ 位及之后所有数位的合法方案数。接下来按照如下步骤进行递归。

1. 从 `dfs(0, 0, True, True)` 开始递归。 `dfs(0, 0, True, True)` 表示：
	1. 从位置 $0$ 开始构造。
	2. 初始数位和为 $0$。
	3. 开始时当前数位最大值受到最高位数位的约束。
	4. 开始时当前数位最小值受到最高位数位的约束。
2. 如果 $total > max\underline{\hspace{0.5em}}sum$，说明当前方案不符合要求，则返回方案数 $0$。
3. 如果遇到  $pos == len(s)$，表示到达数位末尾，此时：
	1. 如果 $min\underline{\hspace{0.5em}}sum \le total \le max\underline{\hspace{0.5em}}sum$，说明当前方案符合要求，则返回方案数 $1$。
	2. 如果不满足，则当前方案不符合要求，则返回方案数 $0$。
4. 如果 $pos \ne len(s)$，则定义方案数 $ans$，令其等于 $0$，即：`ans = 0`。
5. 根据 $isMaxLimit$ 和 $isMinLimit$ 来决定填当前位数位所能选择的最小数字（$minX$）和所能选择的最大数字（$maxX$）。
6. 然后根据 $[minX, maxX]$ 来枚举能够填入的数字 $d$。
7. 方案数累加上当前位选择 $d$ 之后的方案数，即：`ans += dfs(pos + 1, total + d, isMaxLimit and d == maxX, isMinLimit and d == minX)`。
	1. `total + d` 表示当前数位和 $total$ 加上 $d$。
	2. `isMaxLimit and d == maxX` 表示 $pos + 1$ 位最大值受到之前 $pos$ 位限制。
	3. `isMinLimit and d == maxX` 表示 $pos + 1$ 位最小值受到之前 $pos$ 位限制。
8. 最后的方案数为 `dfs(0, 0, True, True) % MOD`，将其返回即可。

### 思路 1：代码

```python
class Solution:
    def count(self, num1: str, num2: str, min_sum: int, max_sum: int) -> int:
        MOD = 10 ** 9 + 7
        # 将 num1 补上前导 0，补到和 num2 长度一致
        m, n = len(num1), len(num2)
        if m < n:
            num1 = '0' * (n - m) + num1
        
        @cache
        # pos: 第 pos 个数位
        # total: 表示数位和
        # isMaxLimit: 表示是否受到上限选择限制。如果为真，则第 pos 位填入数字最多为 s[pos]；如果为假，则最大可为 9。
        # isMaxLimit: 表示是否受到下限选择限制。如果为真，则第 pos 位填入数字最小为 s[pos]；如果为假，则最小可为 0。
        def dfs(pos, total, isMaxLimit, isMinLimit):
            if total > max_sum:
                return 0
            
            if pos == n:
                # 当 min_sum <= total <= max_sum 时，当前方案符合要求
                return int(total >= min_sum)
            
            ans = 0
            # 如果受到选择限制，则最小可选择数字为 num1[pos]，否则最大可选择数字为 0。
            minX = int(num1[pos]) if isMinLimit else 0
            # 如果受到选择限制，则最大可选择数字为 num2[pos]，否则最大可选择数字为 9。
            maxX = int(num2[pos]) if isMaxLimit else 9
            
            # 枚举可选择的数字
            for d in range(minX, maxX + 1): 
                ans += dfs(pos + 1, total + d, isMaxLimit and d == maxX, isMinLimit and d == minX)
            return ans % MOD
    
        return dfs(0, 0, True, True) % MOD
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(n \times 10)$，其中 $n$ 为数组 $nums2$ 的长度。
- **空间复杂度**：$O(n \times max\underline{\hspace{0.5em}}sum)$。

# 题目相关

- 标签：
- 难度：

## 题目链接

- 题目相关

## 题目大意

**描述**：

**要求**：

**说明**：

- 

**示例**：

- 示例 1：

```python
```

- 示例 2：

```python
```

## 解题思路

### 思路 1：动态规划

###### 1. 划分阶段

按照 进行阶段划分。

###### 2. 定义状态

定义状态 $dp[i]$ 表示为：。

###### 3. 状态转移方程



###### 4. 初始条件



###### 5. 最终结果

根据我们之前定义的状态，$dp[i]$ 表示为：。 所以最终结果为 $dp[size]$。

### 思路 1：代码

```python

```

### 思路 1：复杂度分析

- **时间复杂度**：
- **空间复杂度**：
# 题目相关

- 标签：
- 难度：

## 题目链接

- 题目相关

## 题目大意

**描述**：

**要求**：

**说明**：

- 

**示例**：

- 示例 1：

```python
```

- 示例 2：

```python
```

## 解题思路

### 思路 1：



### 思路 1：代码

```python

```

### 思路 1：复杂度分析

- **时间复杂度**：
- **空间复杂度**：

### 思路 2：



### 思路 2：代码

```python

```

### 思路 2：复杂度分析

- **时间复杂度**：
- **空间复杂度**：# [剑指 Offer 03. 数组中重复的数字](https://leetcode.cn/problems/shu-zu-zhong-zhong-fu-de-shu-zi-lcof/)

- 标签：数组、哈希表、排序
- 难度：简单

## 题目链接

- [剑指 Offer 03. 数组中重复的数字 - 力扣](https://leetcode.cn/problems/shu-zu-zhong-zhong-fu-de-shu-zi-lcof/)

## 题目大意

给定一个包含 `n + 1` 个整数的数组 `nums`，里边包含的值都在 `1 ~ n` 之间。假设 `nums` 中只存在一个重复的整数，要求找出这个重复的数。

## 解题思路

使用哈希表存储数组每个元素，遇到重复元素则直接返回该元素。

## 代码

```python
class Solution:
    def findRepeatNumber(self, nums: List[int]) -> int:
        nums_dict = dict()
        for num in nums:
            if num in nums_dict:
                return num
            nums_dict[num] = 1
        return -1
```

# [剑指 Offer 04. 二维数组中的查找](https://leetcode.cn/problems/er-wei-shu-zu-zhong-de-cha-zhao-lcof/)

- 标签：数组、二分查找、分治、矩阵
- 难度：中等

## 题目链接

- [剑指 Offer 04. 二维数组中的查找 - 力扣](https://leetcode.cn/problems/er-wei-shu-zu-zhong-de-cha-zhao-lcof/)

## 题目大意

给定一个 `m * n` 大小的有序整数矩阵 `matrix`。每行元素从左到右升序排列，每列元素从上到下升序排列。再给定一个目标值 `target`。

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

    def findNumberIn2DArray(self, matrix: List[List[int]], target: int) -> bool:
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

# [剑指 Offer 05. 替换空格](https://leetcode.cn/problems/ti-huan-kong-ge-lcof/)

- 标签：字符串
- 难度：简单

## 题目链接

- [剑指 Offer 05. 替换空格 - 力扣](https://leetcode.cn/problems/ti-huan-kong-ge-lcof/)

## 题目大意

给定一个字符串 `s`。

要求：将字符串 `s` 中的每个空格换成 `%20`。

## 解题思路

Python 的字符串是不可变类型，所以需要先用数组存储答案，再将其转为字符串返回。具体操作如下。

- 定义数组 `res`，遍历字符串 `s`。
  - 如果当前字符 `ch` 为空格，则将 ` %20` 加入到数组中。
  - 如果当前字符 `ch` 不为空格，则直接加入到数组中。
- 遍历完之后，通过 `join` 将其转为字符串返回。

## 代码

```python
class Solution:
    def replaceSpace(self, s: str) -> str:
        res = []
        for ch in s:
            if ch == ' ':
                res.append("%20")
            else:
                res.append(ch)
        return "".join(res)
```

# [剑指 Offer 06. 从尾到头打印链表](https://leetcode.cn/problems/cong-wei-dao-tou-da-yin-lian-biao-lcof/)

- 标签：栈、递归、链表、双指针
- 难度：简单

## 题目链接

- [剑指 Offer 06. 从尾到头打印链表 - 力扣](https://leetcode.cn/problems/cong-wei-dao-tou-da-yin-lian-biao-lcof/)

## 题目大意

给定一个链表的头节点 `head`。

要求：从尾到头反过来返回每个节点的值（用数组返回）。

## 解题思路

- 定义数组 `res`，从头到尾遍历链表。
- 将每个节点值存入数组中。
- 直接返回倒序数组。

## 代码

```python
class Solution:
    def reversePrint(self, head: ListNode) -> List[int]:
        res = []
        while head:
            res.append(head.val)
            head = head.next
        return res[::-1]
```

# [剑指 Offer 07. 重建二叉树](https://leetcode.cn/problems/zhong-jian-er-cha-shu-lcof/)

- 标签：树、数组、哈希表、分治、二叉树
- 难度：中等

## 题目链接

- [剑指 Offer 07. 重建二叉树 - 力扣](https://leetcode.cn/problems/zhong-jian-er-cha-shu-lcof/)

## 题目大意

给定一棵二叉树的前序遍历结果和中序遍历结果。

要求：构建该二叉树，并返回其根节点。假设树中没有重复的元素。

## 解题思路

前序遍历的顺序是：根 -> 左 -> 右。中序遍历的顺序是：左 -> 根 -> 右。根据前序遍历的顺序，可以找到根节点位置。然后在中序遍历的结果中可以找到对应的根节点位置，就可以从根节点位置将二叉树分割成左子树、右子树。同时能得到左右子树的节点个数。此时构建当前节点，并递归建立左右子树，在左右子树对应位置继续递归遍历进行上述步骤，直到节点为空，具体操作步骤如下：

- 从前序遍历顺序中当前根节点的位置在 `postorder[0]`。
- 通过在中序遍历中查找上一步根节点对应的位置 `inorder[k]`，从而将二叉树的左右子树分隔开，并得到左右子树节点的个数。
- 从上一步得到的左右子树个数将前序遍历结果中的左右子树分开。
- 构建当前节点，并递归建立左右子树，在左右子树对应位置继续递归遍历并执行上述三步，直到节点为空。

## 代码

```python
class Solution:
    def buildTree(self, preorder: List[int], inorder: List[int]) -> TreeNode:
        def createTree(preorder, inorder, n):
            if n == 0:
                return None
            k = 0
            while preorder[0] != inorder[k]:
                k += 1
            node = TreeNode(inorder[k])
            node.left = createTree(preorder[1: k + 1], inorder[0: k], k)
            node.right = createTree(preorder[k + 1:], inorder[k + 1:], n - k - 1)
            return node

        return createTree(preorder, inorder, len(inorder))
```

# [剑指 Offer 09. 用两个栈实现队列](https://leetcode.cn/problems/yong-liang-ge-zhan-shi-xian-dui-lie-lcof/)

- 标签：栈、设计、队列
- 难度：简单

## 题目链接

- [剑指 Offer 09. 用两个栈实现队列 - 力扣](https://leetcode.cn/problems/yong-liang-ge-zhan-shi-xian-dui-lie-lcof/)

## 题目大意

要求：使用两个栈实现先入先出队列。需要实现对应的两个函数：

- `appendTail`：在队列尾部插入整数。
- `deleteHead`：在队列头部删除整数（如果队列中没有元素，`deleteHead` 返回 -1）。

## 解题思路

使用两个栈，inStack 用于输入，outStack 用于输出。

- `appendTail` 操作：将元素压入 inStack 中
- `deleteHead` 操作：
  - 先判断  `inStack` 和 `outStack` 是否都为空，如果都为空则说明队列中没有元素，直接返回 `-1`。
  - 如果 `outStack` 输出栈为空，将 `inStack` 输入栈元素依次取出，按顺序压入 `outStack` 栈。这样 `outStack` 栈的元素顺序和之前 `inStack` 元素顺序相反，`outStack` 顶层元素就是要取出的队头元素，将其移出，并返回该元素。如果 `outStack` 输出栈不为空，则直接取出顶层元素。

## 代码

```python
class CQueue:

    def __init__(self):
        self.inStack = []
        self.outStack = []


    def appendTail(self, value: int) -> None:
        self.inStack.append(value)


    def deleteHead(self) -> int:
        if len(self.outStack) == 0 and len(self.inStack) == 0:
            return -1
        if (len(self.outStack) == 0):
            while (len(self.inStack) != 0):
                self.outStack.append(self.inStack[-1])
                self.inStack.pop()
        top = self.outStack[-1]
        self.outStack.pop()
        return top
```

# [剑指 Offer 10- I. 斐波那契数列](https://leetcode.cn/problems/fei-bo-na-qi-shu-lie-lcof/)

- 标签：记忆化搜索、数学、动态规划
- 难度：简单

## 题目链接

- [剑指 Offer 10- I. 斐波那契数列 - 力扣](https://leetcode.cn/problems/fei-bo-na-qi-shu-lie-lcof/)

## 题目大意

给定一个整数 `n`。

要求：计算斐波那契数列的第 `n` 项。

注意：答案需对 `1000000007` 进行取余操作。

## 解题思路

斐波那契的递推公式为：`F(n) = F(n-1) + F(n-2)`。

直接根据递推公式求解即可。注意答案需要取余。

## 代码

```python
class Solution:
    def fib(self, n: int) -> int:
        if n < 2:
            return n
        f1 = 0
        f2 = 0
        f3 = 1
        for i in range(2, n + 1):
            f1, f2 = f2, f3
            f3 = (f1 + f2) % 1000000007
        return f3
```

# [剑指 Offer 10- II. 青蛙跳台阶问题](https://leetcode.cn/problems/qing-wa-tiao-tai-jie-wen-ti-lcof/)

- 标签：记忆化搜索、数学、动态规划
- 难度：简单

## 题目链接

- [剑指 Offer 10- II. 青蛙跳台阶问题 - 力扣](https://leetcode.cn/problems/qing-wa-tiao-tai-jie-wen-ti-lcof/)

## 题目大意

一直青蛙一次可以跳上 `1` 级台阶，也可以跳上 `2` 级台阶。

要求：求该青蛙跳上 `n` 级台阶共有多少中跳法。答案需要对 `1000000007` 取余。

## 解题思路

先来看一下规律：

第 0 级台阶：1 种方法（比较特殊）

第 1 级台阶：1 种方法（从 0 阶爬 1 阶）

第 2 阶台阶：2 种方法（从 0 阶爬 2 阶，从 1 阶爬 1 阶）

第 i 阶台阶：从第 i-1 阶台阶爬 1 阶，或者从第 i-2 阶台阶爬 2 阶。

则推出递推公式为：

- 当 `n = 0` 时，`F(i) = 1`。
- 当 `n > 0` 时，`F(i) = F(i-1) + F(i-2)`。

## 代码

```python
class Solution:
    def numWays(self, n: int) -> int:
        if n == 0:
            return 1

        f1, f2, f3 = 0, 1, 1
        for i in range(2, n + 1):
            f1, f2 = f2, f3
            f3 = (f1 + f2) % 1000000007
        return f3
```

# [剑指 Offer 11. 旋转数组的最小数字](https://leetcode.cn/problems/xuan-zhuan-shu-zu-de-zui-xiao-shu-zi-lcof/)

- 标签：数组、二分查找
- 难度：简单

## 题目链接

- [剑指 Offer 11. 旋转数组的最小数字 - 力扣](https://leetcode.cn/problems/xuan-zhuan-shu-zu-de-zui-xiao-shu-zi-lcof/)

## 题目大意

给定一个数组 `numbers`，`numbers` 是有升序数组经过「旋转」得到的。但是旋转次数未知。数组中可能存在重复元素。

要求：找出数组中的最小元素。

- 旋转：将数组整体右移。

## 解题思路

数组经过「旋转」之后，会有两种情况，第一种就是原先的升序序列，另一种是两段升序的序列。

第一种的最小值在最左边。第二种最小值在第二段升序序列的第一个元素。

```
          *
        *
      *
    *
  *
*
```



```
    *
  *
*
          *
        *
      *
```

最直接的办法就是遍历一遍，找到最小值。但是还可以有更好的方法。考虑用二分查找来降低算法的时间复杂度。

创建两个指针 left、right，分别指向数组首尾。让后计算出两个指针中间值 mid。将 mid 与右边界进行比较。

1. 如果 `numbers[mid] > numbers[right]`，则最小值不可能在 `mid` 左侧，一定在 `mid` 右侧，则将 `left` 移动到 `mid + 1` 位置，继续查找右侧区间。
2. 如果 `numbers[mid] < numbers[right]`，则最小值一定在 `mid` 左侧，将 `right` 移动到 `mid` 位置上，继续查找左侧区间。
3. 当 `numbers[mid] == numbers[right]`，无法判断在 `mid` 的哪一侧，可以采用 `right = right - 1` 逐步缩小区域。

## 代码

```python
class Solution:
    def minArray(self, numbers: List[int]) -> int:
        left = 0
        right = len(numbers) - 1
        while left < right:
            mid = left + (right - left) // 2
            if numbers[mid] > numbers[right]:
                left = mid + 1
            elif numbers[mid] < numbers[right]:
                right = mid
            else:
                right = right - 1
        return numbers[left]
```

# [剑指 Offer 12. 矩阵中的路径](https://leetcode.cn/problems/ju-zhen-zhong-de-lu-jing-lcof/)

- 标签：数组、回溯、矩阵
- 难度：中等

## 题目链接

- [剑指 Offer 12. 矩阵中的路径 - 力扣](https://leetcode.cn/problems/ju-zhen-zhong-de-lu-jing-lcof/)

## 题目大意

给定一个 `m * n` 大小的二维字符矩阵 `board` 和一个字符串单词 `word`。如果 `word` 存在于网格中，返回 `True`，否则返回 `False`。

- 单词必须按照字母顺序通过上下左右相邻的单元格字母构成。且同一个单元格内的字母不允许被重复使用。

## 解题思路

回溯算法在二维矩阵 `board` 中按照上下左右四个方向递归搜索。设函数 `backtrack(i, j, index)` 表示从 `board[i][j]` 出发，能否搜索到单词字母 `word[index]`，以及 `index` 位置之后的后缀子串。如果能搜索到，则返回 `True`，否则返回 `False`。`backtrack(i, j, index)` 执行步骤如下：

- 如果 $board[i][j] = word[index]$，而且 `index` 已经到达 `word` 字符串末尾，则返回 `True`。
- 如果 $board[i][j] = word[index]$，而且 `index` 未到达 `word` 字符串末尾，则遍历当前位置的所有相邻位置。如果从某个相邻位置能搜索到后缀子串，则返回 `True`，否则返回 `False`。
- 如果 $board[i][j] \ne word[index]$，则当前字符不匹配，返回 `False`。

## 代码

```python
class Solution:
    def exist(self, board: List[List[str]], word: str) -> bool:
        directs = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        rows = len(board)
        if rows == 0:
            return False
        cols = len(board[0])
        visited = [[False for _ in range(cols)] for _ in range(rows)]

        def backtrack(i, j, index):
            if index == len(word) - 1:
                return board[i][j] == word[index]

            if board[i][j] == word[index]:
                visited[i][j] = True
                for direct in directs:
                    new_i = i + direct[0]
                    new_j = j + direct[1]
                    if 0 <= new_i < rows and 0 <= new_j < cols and visited[new_i][new_j] == False:
                        if backtrack(new_i, new_j, index + 1):
                            return True
                visited[i][j] = False
            return False

        for i in range(rows):
            for j in range(cols):
                if backtrack(i, j, 0):
                    return True
        return False
```

# [剑指 Offer 13. 机器人的运动范围](https://leetcode.cn/problems/ji-qi-ren-de-yun-dong-fan-wei-lcof/)

- 标签：深度优先搜索、广度优先搜索、动态规划
- 难度：中等

## 题目链接

- [剑指 Offer 13. 机器人的运动范围 - 力扣](https://leetcode.cn/problems/ji-qi-ren-de-yun-dong-fan-wei-lcof/)

## 题目大意

**描述**：有一个 `m * n` 大小的方格，坐标从 `(0, 0)` 到 `(m - 1, n - 1)`。一个机器人从 `(0, 0)` 处的格子开始移动，每次可以向上、下、左、右移动一格（不能移动到方格外），也不能移动到行坐标和列坐标的数位之和大于 `k` 的格子。现在给定 `3` 个整数 `m`、`n`、`k`。

**要求**：计算并输出该机器人能够达到多少个格子。

**说明**：

- $1 \le n, m \le 100$。
- $0 \le k \le 20$。

**示例**：

- 示例 1：

```python
输入：m = 2, n = 3, k = 1
输出：3
```

- 示例 2：

```python
输入：m = 3, n = 1, k = 0
输出：1
```

## 解题思路

### 思路 1：广度优先搜索

先定义一个计算数位和的方法 `digitsum`，该方法输入一个整数，返回该整数各个数位的总和。

然后我们使用广度优先搜索方法，具体步骤如下：

- 将 `(0, 0)` 加入队列 `queue` 中。
- 当队列不为空时，每次将队首坐标弹出，加入访问集合 `visited` 中。
- 再将满足行列坐标的数位和不大于 `k` 的格子位置加入到队列中，继续弹出队首位置。
- 直到队列为空时停止。输出访问集合的长度。

### 思路 1：代码

```python
import collections

class Solution:
    def digitsum(self, n: int):
        ans = 0
        while n:
            ans += n % 10
            n //= 10
        return ans

    def movingCount(self, m: int, n: int, k: int) -> int:
        queue = collections.deque([(0, 0)])
        visited = set()

        while queue:
            x, y = queue.popleft()
            if (x, y) not in visited and self.digitsum(x) + self.digitsum(y) <= k:
                visited.add((x, y))
                for dx, dy in [(1, 0), (0, 1)]:
                    nx = x + dx
                    ny = y + dy
                    if 0 <= nx < m and 0 <= ny < n:
                        queue.append((nx, ny))
        return len(visited)
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(m \times n)$。其中 $m$ 为方格的行数，$n$ 为方格的列数。
- **空间复杂度**：$O(m \times n)$。

# [剑指 Offer 14- I. 剪绳子](https://leetcode.cn/problems/jian-sheng-zi-lcof/)

- 标签：数学、动态规划
- 难度：中等

## 题目链接

- [剑指 Offer 14- I. 剪绳子 - 力扣](https://leetcode.cn/problems/jian-sheng-zi-lcof/)

## 题目大意

给定一根长度为 `n` 的绳子，将绳子剪成整数长度的 `m` 段，每段绳子长度即为 `k[0]`、`k[1]`、...、`k[m - 1]`。

要求：计算出 `k[0] * k[1] * ... * k[m - 1]` 可能的最大乘积。

## 解题思路

可以使用动态规划求解。

定义状态 `dp[i]` 为：拆分长度为 `i` 的绳子，可以获得的最大乘积为 `dp[i]`。

将 `j` 从 `1` 遍历到 `i - 1`，通过两种方式得到 `dp[i]`：

- `(i - j) * j` ，直接将长度为 `i` 的绳子分割为 `i - j` 和 `j`，获取两者乘积。
- `dp[i - j] * j`，将长度为 `i`的绳子 中的 `i - j` 部分拆分，得到 `dp[i - j]`，和 `j` ，获取乘积。

则 `dp[i]` 取两者中的最大值。遍历 `j`，得到 `dp[i]` 的最大值。

则状态转移方程为：`dp[i] = max(dp[i], (i - j) * j, dp[i - j] * j)`。

最终输出 `dp[n]`。

## 代码

```python
class Solution:
    def cuttingRope(self, n: int) -> int:
        dp = [0 for _ in range(n + 1)]
        dp[1] = 1
        for i in range(2, n + 1):
            for j in range(1, i):
                dp[i] = max(dp[i], dp[i - j] * j, (i - j) * j)
        return dp[n]
```

# [剑指 Offer 15. 二进制中1的个数](https://leetcode.cn/problems/er-jin-zhi-zhong-1de-ge-shu-lcof/)

- 标签：位运算
- 难度：简单

## 题目链接

- [剑指 Offer 15. 二进制中1的个数 - 力扣](https://leetcode.cn/problems/er-jin-zhi-zhong-1de-ge-shu-lcof/)

## 题目大意

给定一个无符号整数 `n`。

要求：统计其对应二进制表达式中 `1` 的个数。

## 解题思路

### 1. 循环按位计算

对整数 n 的每一位进行按位与运算，并统计结果。

### 2. 改进位运算

利用 $n \text{ \& } (n-1)$ 。这个运算刚好可以将 n 的二进制中最低位的 $1$ 变为 $0$。 比如 $n = 6$ 时，$6 = (110)_2$，$6 - 1 = (101)_2$，$(110)_2 \text{ \& } (101)_2 = (100)_2$ 。

利用这个位运算，不断的将 $n$ 中最低位的 $1$ 变为 $0$，直到 $n$ 变为 $0$ 即可，其变换次数就是我们要求的结果。

## 代码

1. 循环按位计算

```python
class Solution:
    def hammingWeight(self, n: int) -> int:
        ans = 0
        while n:
            ans += (n & 1)
            n = n >> 1
        return ans
```

2. 改进位运算

```python
class Solution:
    def hammingWeight(self, n: int) -> int:
        ans = 0
        while n:
            n &= n-1
            ans += 1
        return ans
```



# [剑指 Offer 16. 数值的整数次方](https://leetcode.cn/problems/shu-zhi-de-zheng-shu-ci-fang-lcof/)

- 标签：递归、数学
- 难度：中等

## 题目链接

- [剑指 Offer 16. 数值的整数次方 - 力扣](https://leetcode.cn/problems/shu-zhi-de-zheng-shu-ci-fang-lcof/)

## 题目大意

给定浮点数 `x` 和整数 `n`。

要求：实现 `pow(x, n)`，即计算 $x^n$，不能使用库函数，不需要考虑大数问题。

## 解题思路

常规方法是直接将 x 累乘 n 次得出结果，时间复杂度为 $O(n)$。可以利用快速幂来减少时间复杂度。

如果 n 为偶数，$x^n = x^{n/2} * x^{n/2}$。如果 n 为奇数，$x^n = x * x^{(n-1)/2} * x^{(n-1)/2}$。

$x^(n/2)$ 又可以继续向下递归划分。则我们可以利用低纬度的幂计算结果，来得到高纬度的幂计算结果。

这样递归求解，时间复杂度为 $O(logn)$，并且递归也可以转为递推来做。

需要注意如果 n 为负数，可以转换为 $\frac{1}{x} ^{(-n)}$。

## 代码

```python
class Solution:
    def myPow(self, x: float, n: int) -> float:
        if x == 0.0:
            return 0.0
        res = 1
        if n < 0:
            x = 1 / x
            n = -n
        while n:
            if n & 1:
                res *= x
            x *= x
            n >>= 1
        return res
```

# [剑指 Offer 17. 打印从1到最大的n位数](https://leetcode.cn/problems/da-yin-cong-1dao-zui-da-de-nwei-shu-lcof/)

- 标签：数组、数学
- 难度：简单

## 题目链接

- [剑指 Offer 17. 打印从1到最大的n位数 - 力扣](https://leetcode.cn/problems/da-yin-cong-1dao-zui-da-de-nwei-shu-lcof/)

## 题目大意

给定一个数字 `n`。

要求：按顺序打印从 `1` 到最大 `n` 位的十进制数。

## 解题思路

直接枚举 $1 \sim 10^{n} - 1$，生成列表并返回。

## 代码

```python
class Solution:
    def printNumbers(self, n: int) -> List[int]:
        return [i for i in range(1, 10 ** n)]
```

# [剑指 Offer 18. 删除链表的节点](https://leetcode.cn/problems/shan-chu-lian-biao-de-jie-dian-lcof/)

- 标签：链表
- 难度：简单

## 题目链接

- [剑指 Offer 18. 删除链表的节点 - 力扣](https://leetcode.cn/problems/shan-chu-lian-biao-de-jie-dian-lcof/)

## 题目大意

给定一个链表。

要求：删除链表中值为 `val` 的节点，并返回新的链表头节点。

## 解题思路

用两个指针 `prev` 和 `curr`。`prev` 指向前一节点和当前节点，`curr` 指向当前节点。从前向后遍历链表，遇到值为 `val` 的节点时，将 `prev` 指向当前节点的下一个节点，继续递归遍历。遇不到则更新 `prev` 指针，并继续遍历。

需要注意的是要删除的节点可能包含了头节点。我们可以考虑在遍历之前，新建一个头节点，让其指向原来的头节点。这样，最终如果删除的是头节点，则删除原头节点即可。返回结果的时候，可以直接返回新建头节点的下一位节点。

## 代码

```python
class Solution:
    def deleteNode(self, head: ListNode, val: int) -> ListNode:
        newHead = ListNode(0, head)
        newHead.next = head

        prev, curr = newHead, head
        while curr:
            if curr.val == val:
                prev.next = curr.next
            else:
                prev = curr
            curr = curr.next
        return newHead.next
```

# [剑指 Offer 21. 调整数组顺序使奇数位于偶数前面](https://leetcode.cn/problems/diao-zheng-shu-zu-shun-xu-shi-qi-shu-wei-yu-ou-shu-qian-mian-lcof/)

- 标签：数组、双指针、排序
- 难度：简单

## 题目链接

- [剑指 Offer 21. 调整数组顺序使奇数位于偶数前面 - 力扣](https://leetcode.cn/problems/diao-zheng-shu-zu-shun-xu-shi-qi-shu-wei-yu-ou-shu-qian-mian-lcof/)

## 题目大意

**描述**：给定一个整数数组 $nums$。

**要求**：将奇数元素位于数组的前半部分，偶数元素位于数组的后半部分。

**说明**：

- $0 \le nums.length \le 50000$。
- $0 \le nums[i] \le 10000$。

**示例**：

- 示例 1：

```python
输入：nums = [1,2,3,4,5]
输出：[1,3,5,2,4] 
解释：为正确答案之一
```

## 解题思路

### 思路 1：快慢指针

定义快慢指针 $slow$、$fast$，开始时都指向 $0$。

- $fast$ 向前搜索奇数位置，$slow$ 指向下一个奇数应当存放的位置。
- $fast$ 不断进行右移，当遇到奇数时，将该奇数与 $slow$ 指向的元素进行交换，并将 $slow$ 进行右移。
- 重复上面操作，直到 $fast$ 指向数组末尾。

### 思路 1：代码

```python
class Solution:
    def exchange(self, nums: List[int]) -> List[int]:
        slow, fast = 0, 0
        while fast < len(nums):
            if nums[fast] % 2 == 1:
                nums[slow], nums[fast] = nums[fast], nums[slow]
                slow += 1
            fast += 1

        return nums
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(n)$，其中 $n$ 为数组 $nums$ 中的元素个数。
- **空间复杂度**：$O(1)$。

# [剑指 Offer 22. 链表中倒数第k个节点](https://leetcode.cn/problems/lian-biao-zhong-dao-shu-di-kge-jie-dian-lcof/)

- 标签：链表、双指针
- 难度：简单

## 题目链接

- [剑指 Offer 22. 链表中倒数第k个节点 - 力扣](https://leetcode.cn/problems/lian-biao-zhong-dao-shu-di-kge-jie-dian-lcof/)

## 题目大意

给定一个链表的头节点 `head`，以及一个整数 `k`。

要求返回链表的倒数第 `k` 个节点。

## 解题思路

常规思路是遍历一遍链表，求出链表长度，再遍历一遍到对应位置，返回该位置上的节点。

如果用一次遍历实现的话，可以使用快慢指针。让快指针先走 `k` 步，然后快慢指针、慢指针再同时走，每次一步，这样等快指针遍历到链表尾部的时候，慢指针就刚好遍历到了倒数第 `k` 个节点位置。返回该该位置上的节点即可。

## 代码

```python
class Solution:
    def getKthFromEnd(self, head: ListNode, k: int) -> ListNode:
        slow = head
        fast = head
        for _ in range(k):
            if fast == None:
                return fast
            fast = fast.next
        while fast:
            slow = slow.next
            fast = fast.next
        return slow
```

# [剑指 Offer 24. 反转链表](https://leetcode.cn/problems/fan-zhuan-lian-biao-lcof/)

- 标签：递归、链表
- 难度：简单

## 题目链接

- [剑指 Offer 24. 反转链表 - 力扣](https://leetcode.cn/problems/fan-zhuan-lian-biao-lcof/)

## 题目大意

**描述**：给定一个链表的头节点 `head`。

**要求**：将该链表反转并输出反转后链表的头节点。

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
- 【题解】[【反转链表】：双指针，递归，妖魔化的双指针 - 反转链表 - 力扣（LeetCode）](https://leetcode.cn/problems/reverse-linked-list/solution/fan-zhuan-lian-biao-shuang-zhi-zhen-di-gui-yao-mo-/)

# [剑指 Offer 25. 合并两个排序的链表](https://leetcode.cn/problems/he-bing-liang-ge-pai-xu-de-lian-biao-lcof/)

- 标签：递归、链表
- 难度：简单

## 题目链接

- [剑指 Offer 25. 合并两个排序的链表 - 力扣](https://leetcode.cn/problems/he-bing-liang-ge-pai-xu-de-lian-biao-lcof/)

## 题目大意

给定两个升序链表。

要求：将其合并为一个升序链表。

## 解题思路

利用归并排序的思想。

创建一个新的链表节点作为头节点（记得保存），然后判断 l1和 l2 头节点的值，将较小值的节点添加到新的链表中。

当一个节点添加到新的链表中之后，将对应的 l1 或 l2 链表向后移动一位。

然后继续判断当前 l1 节点和当前 l2 节点的值，继续将较小值的节点添加到新的链表中，然后将对应的链表向后移动一位。

这样，当 l1 或 l2 遍历到最后，最多有一个链表还有节点未遍历，则直接将该节点链接到新的链表尾部即可。

## 代码

```python
class Solution:
    def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
        newHead = ListNode(-1)

        curr = newHead
        while l1 and l2:
            if l1.val <= l2.val:
                curr.next = l1
                l1 = l1.next
            else:
                curr.next = l2
                l2 = l2.next
            curr = curr.next

        curr.next = l1 if l1 is not None else l2

        return newHead.next
```

# [剑指 Offer 26. 树的子结构](https://leetcode.cn/problems/shu-de-zi-jie-gou-lcof/)

- 标签：树、深度优先搜索、二叉树
- 难度：中等

## 题目链接

- [剑指 Offer 26. 树的子结构 - 力扣](https://leetcode.cn/problems/shu-de-zi-jie-gou-lcof/)

## 题目大意

给定两棵二叉树的根节点 `A`、`B`。

要求：判断 `B` 是不是 `A` 的子结构。（空树不是任意一棵树的子结构）。

- `B` 是 `A` 的子结构：`A` 中有出现和 `B` 相同的结构和节点值。

## 解题思路

深度优先搜索。

- 先判断特例，如果 `A`、`B` 都为空树，则直接返回 `False`。
- 然后递归判断 `A`、`B` 是否相等。
    - 如果 `A`、`B` 相等，则返回 `True`。
    - 如果 `A`、`B` 不相等，则递归判断 `B` 是否是  `A` 的左子树的子结构，或者 `B` 是否是 `A` 的右子树的子结构，如果有一种满足，则返回 `True`，如果都不满足，则返回 `False`。

递归判断 `A`、`B` 是否相等的具体方法如下：

- 如果 `B` 为空树，则直接返回 `False`，因为空树不是任意一棵树的子结构。
- 如果 `A` 为空树或者 `A` 节点的值不等于 `B` 节点的值，则返回 `False`。
- 如果 `A`、`B` 都不为空，且节点值相同，则递归判断 `A` 的左子树和 `B` 的左子树是否相等，判断 `A` 的右子树和 `B` 的右子树是否相等。如果都相等，则返回 `True`，否则返回 `False`。

## 代码

```python
class Solution:
    def hasSubStructure(self, A: TreeNode, B: TreeNode) -> bool:
        if not B:
            return True
        if not A or A.val != B.val:
            return False
        return self.hasSubStructure(A.left, B.left) and self.hasSubStructure(A.right, B.right)

    def isSubStructure(self, A: TreeNode, B: TreeNode) -> bool:
        if not A or not B:
            return False
        if self.hasSubStructure(A, B):
            return True
        return self.isSubStructure(A.left, B) or self.isSubStructure(A.right, B)
```

# [剑指 Offer 27. 二叉树的镜像](https://leetcode.cn/problems/er-cha-shu-de-jing-xiang-lcof/)

- 标签：树、深度优先搜索、广度优先搜索、二叉树
- 难度：简单

## 题目链接

- [剑指 Offer 27. 二叉树的镜像 - 力扣](https://leetcode.cn/problems/er-cha-shu-de-jing-xiang-lcof/)

## 题目大意

给定一个二叉树的根节点 `root`。

要求：将其进行左右翻转。

## 解题思路

从根节点开始遍历，然后从叶子节点向上递归交换左右子树位置。

## 代码

```python
class Solution:
    def mirrorTree(self, root: TreeNode) -> TreeNode:
        if not root:
            return root
        left = self.mirrorTree(root.left)
        right = self.mirrorTree(root.right)
        root.left = right
        root.right = left
        return root
```

# [剑指 Offer 28. 对称的二叉树](https://leetcode.cn/problems/dui-cheng-de-er-cha-shu-lcof/)

- 标签：树、深度优先搜索、广度优先搜索、二叉树
- 难度：简单

## 题目链接

- [剑指 Offer 28. 对称的二叉树 - 力扣](https://leetcode.cn/problems/dui-cheng-de-er-cha-shu-lcof/)

## 题目大意

给定一个二叉树的根节点 `root`。

要求：检查这课二叉树是否是左右对称的。

## 解题思路

递归遍历左右子树， 然后判断当前节点的左右子节点。如果可以直接判断的情况，则跳出递归，直接返回结果。如果无法直接判断结果，则递归检测左右子树的外侧节点是否相等，同理再递归检测左右子树的内侧节点是否相等。

## 代码

```python
class Solution:
    def isSymmetric(self, root: TreeNode) -> bool:
        if not root:
            return True
        return self.check(root.left, root.right)

    def check(self, left: TreeNode, right: TreeNode):
        if not left and not right:
            return True
        elif not left and right:
            return False
        elif left and not right:
            return False
        elif left.val != right.val:
            return False

        return self.check(left.left, right.right) and self.check(left.right, right.left)
```

# [剑指 Offer 29. 顺时针打印矩阵](https://leetcode.cn/problems/shun-shi-zhen-da-yin-ju-zhen-lcof/)

- 标签：数组、矩阵、模拟
- 难度：简单

## 题目链接

- [剑指 Offer 29. 顺时针打印矩阵 - 力扣](https://leetcode.cn/problems/shun-shi-zhen-da-yin-ju-zhen-lcof/)

## 题目大意

给定一个 `m * n` 大小的二维矩阵 `matrix`。

要求：按照顺时针旋转的顺序，返回矩阵中的所有元素。

## 解题思路

按照题意进行模拟。可以实现定义一下上、下、左、右的边界，然后按照逆时针的顺序从边界上依次访问元素。

当访问完当前边界之后，要更新一下边界位置，缩小范围，方便下一轮进行访问。

## 代码

```python
class Solution:
    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
        size_m = len(matrix)
        if size_m == 0:
            return []
        size_n = len(matrix[0])
        if size_n == 0:
            return []

        up, down, left, right = 0, size_m - 1, 0, size_n - 1
        ans = []
        while True:
            for i in range(left, right + 1):
                ans.append(matrix[up][i])
            up += 1
            if up > down:
                break
            for i in range(up, down + 1):
                ans.append(matrix[i][right])
            right -= 1
            if right < left:
                break
            for i in range(right, left - 1, -1):
                ans.append(matrix[down][i])
            down -= 1
            if down < up:
                break
            for i in range(down, up - 1, -1):
                ans.append(matrix[i][left])
            left += 1
            if left > right:
                break
        return ans
```

# [剑指 Offer 30. 包含min函数的栈](https://leetcode.cn/problems/bao-han-minhan-shu-de-zhan-lcof/)

- 标签：栈、设计
- 难度：简单

## 题目链接

- [剑指 Offer 30. 包含min函数的栈 - 力扣](https://leetcode.cn/problems/bao-han-minhan-shu-de-zhan-lcof/)

## 题目大意

要求：设计一个「栈」，实现  `push` ，`pop` ，`top` ，`min` 操作，并且操作时间复杂度都是 `O(1)`。

## 解题思路

使用一个栈，栈元素中除了保存当前值之外，再保存一个当前最小值。

-  `push` 操作：如果栈不为空，则判断当前值与栈顶元素所保存的最小值，并更新当前最小值，将新元素保存到栈中。
-  `pop`操作：正常出栈
-  `top` 操作：返回栈顶元素保存的值。
-  `min` 操作：返回栈顶元素保存的最小值。

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


    def min(self) -> int:
        return self.stack[-1].min
```

# [剑指 Offer 31. 栈的压入、弹出序列](https://leetcode.cn/problems/zhan-de-ya-ru-dan-chu-xu-lie-lcof/)

- 标签：栈、数组、模拟
- 难度：中等

## 题目链接

- [剑指 Offer 31. 栈的压入、弹出序列 - 力扣](https://leetcode.cn/problems/zhan-de-ya-ru-dan-chu-xu-lie-lcof/)

## 题目大意

给定连个整数序列 `pushed` 和 `popped`，其中 `pushed` 表示栈的压入顺序。

要求：判断第二个序列 `popped` 是否为栈的压出序列。

## 解题思路

借助一个栈来模拟压入、压出的操作。检测最后是否能模拟成功。

## 代码

```python
class Solution:
    def validateStackSequences(self, pushed: List[int], popped: List[int]) -> bool:
        stack = []
        index = 0
        for item in pushed:
            stack.append(item)
            while(stack and stack[-1] == popped[index]):
                stack.pop()
                index += 1

        return len(stack) == 0
```

# [剑指 Offer 32 - I. 从上到下打印二叉树](https://leetcode.cn/problems/cong-shang-dao-xia-da-yin-er-cha-shu-lcof/)

- 标签：树、广度优先搜索、二叉树
- 难度：中等

## 题目链接

- [剑指 Offer 32 - I. 从上到下打印二叉树 - 力扣](https://leetcode.cn/problems/cong-shang-dao-xia-da-yin-er-cha-shu-lcof/)

## 题目大意

给定一棵二叉树的根节点 `root`。

要求：从上到下打印出二叉树的每个节点，同一层的节点按照从左到右的顺序打印。

## 解题思路

广度优先搜索。

具体步骤如下：

- 根节点入队。
- 当队列不为空时，求出当前队列长度 $s_i$。
    - 依次从队列中取出这 $s_i$ 个元素，将其加入答案数组，并将其左右子节点入队，然后继续迭代。
- 当队列为空时，结束。

## 代码

```python
class Solution:
    def levelOrder(self, root: TreeNode) -> List[int]:
        if not root:
            return []
        queue = [root]
        order = []
        while queue:
            size = len(queue)
            for _ in range(size):
                curr = queue.pop(0)
                order.append(curr.val)
                if curr.left:
                    queue.append(curr.left)
                if curr.right:
                    queue.append(curr.right)
        return order
```

# [剑指 Offer 32 - II. 从上到下打印二叉树 II](https://leetcode.cn/problems/cong-shang-dao-xia-da-yin-er-cha-shu-ii-lcof/)

- 标签：树、广度优先搜索、二叉树
- 难度：简单

## 题目链接

- [剑指 Offer 32 - II. 从上到下打印二叉树 II - 力扣](https://leetcode.cn/problems/cong-shang-dao-xia-da-yin-er-cha-shu-ii-lcof/)

## 题目大意

给定一棵二叉树的根节点 `root`。

要求：从上到下按层打印二叉树，同一层的节点按从左到右的顺序打印，每一层打印到一行。

## 解题思路

广度优先搜索，需要增加一些变化。普通广度优先搜索只取一个元素，变化后的广度优先搜索每次取出第 i 层上所有元素。

具体步骤如下：

- 根节点入队。
- 当队列不为空时，求出当前队列长度 $s_i$。
    - 依次从队列中取出这 $s_i$ 个元素，并将其左右子节点入队，遍历完之后将这层节点数组加入答案数组中，然后继续迭代。
- 当队列为空时，结束。

## 代码

```python
class Solution:
    def levelOrder(self, root: TreeNode) -> List[List[int]]:
        if not root:
            return []
        queue = [root]
        order = []
        while queue:
            level = []
            size = len(queue)
            for _ in range(size):
                curr = queue.pop(0)
                level.append(curr.val)
                if curr.left:
                    queue.append(curr.left)
                if curr.right:
                    queue.append(curr.right)
            if level:
                order.append(level)
        return order
```

# [剑指 Offer 32 - III. 从上到下打印二叉树 III](https://leetcode.cn/problems/cong-shang-dao-xia-da-yin-er-cha-shu-iii-lcof/)

- 标签：树、广度优先搜索、二叉树
- 难度：中等

## 题目链接

- [剑指 Offer 32 - III. 从上到下打印二叉树 III - 力扣](https://leetcode.cn/problems/cong-shang-dao-xia-da-yin-er-cha-shu-iii-lcof/)

## 题目大意

给定一个二叉树的根节点 `root`。

要求：返回其之字形层序遍历。

- 之字形层序遍历：从左往右，再从右往左进行下一层遍历，以此类推，层与层之间交替进行。

## 解题思路

广度优先搜索，在二叉树的层序遍历的基础上需要增加一些变化。

普通广度优先搜索只取一个元素，变化后的广度优先搜索每次取出第 i 层上所有元素。

新增一个变量 odd，用于判断当前层数是奇数层，还是偶数层。从而判断元素遍历方向。

存储每层元素的 level 列表改用双端队列，如果是奇数层，则从末尾添加元素。如果是偶数层，则从头部添加元素。

具体步骤如下：

- 根节点入队。
- 当队列不为空时，求出当前队列长度 $s_i$，并判断当前层数的奇偶性。
- 依次从队列中取出这 $s_i$ 个元素。
    - 如果为奇数层，如果是奇数层，则从 level 末尾添加元素。
    - 如果是偶数层，则从 level头部添加元素。
- 然后保存将其左右子节点入队，然后继续迭代。
- 当队列为空时，结束。

## 代码

```python
import collections

class Solution:
    def levelOrder(self, root: TreeNode) -> List[List[int]]:
        if not root:
            return []
        queue = [root]
        order = []
        odd = True
        while queue:
            level = collections.deque()
            size = len(queue)
            for _ in range(size):
                curr = queue.pop(0)
                if odd:
                    level.append(curr.val)
                else:
                    level.appendleft(curr.val)
                if curr.left:
                    queue.append(curr.left)
                if curr.right:
                    queue.append(curr.right)
            if level:
                order.append(list(level))
            odd = not odd
        return order
```

# [剑指 Offer 33. 二叉搜索树的后序遍历序列](https://leetcode.cn/problems/er-cha-sou-suo-shu-de-hou-xu-bian-li-xu-lie-lcof/)

- 标签：栈、树、二叉搜索树、递归、二叉树、单调栈
- 难度：中等

## 题目链接

- [剑指 Offer 33. 二叉搜索树的后序遍历序列 - 力扣](https://leetcode.cn/problems/er-cha-sou-suo-shu-de-hou-xu-bian-li-xu-lie-lcof/)

## 题目大意

**描述**：给定一个整数数组 $postorder$。数组的任意两个数字都互不相同。

**要求**：判断该数组是不是某二叉搜索树的后序遍历结果。如果是，则返回 `True`，否则返回 `False`。

**说明**：

- 数组长度 <= 1000。
- $postorder$ 中无重复数字。

**示例**：

- 示例 1：

![](https://pic.leetcode.cn/1694762751-fwHhWX-%E5%89%91%E6%8C%8733%E7%A4%BA%E4%BE%8B1.png)

```python
输入: postorder = [4,9,6,9,8]
输出: false 
解释：从上图可以看出这不是一颗二叉搜索树
```

- 示例 2：

![](https://pic.leetcode.cn/1694762510-vVpTic-%E5%89%91%E6%8C%8733.png)

```python
输入: postorder = [4,6,5,9,8]
输出: true 
解释：可构建的二叉搜索树如上图
```

## 解题思路

### 思路 1：递归分治

后序遍历的顺序为：左 -> 右 -> 根。而二叉搜索树的定义是：左子树所有节点值 < 根节点值，右子树所有节点值 > 根节点值。

所以，可以把数组最右侧元素作为二叉搜索树的根节点值。然后判断数组的左右两侧是否符合左侧值都小于该节点值，右侧值都大于该节点值。如果不满足，则说明不是某二叉搜索树的后序遍历结果。

找到左右分界线位置，然后递归左右数组继续查找。

终止条件为数组 开始位置 > 结束位置，此时该树的子节点数目小于等于 $1$，直接返回 `True` 即可。

### 思路 1：代码

```python
class Solution:
    def verifyPostorder(self, postorder: List[int]) -> bool:
        def verify(left, right):
            if left >= right:
                return True
            index = left
            while postorder[index] < postorder[right]:
                index += 1
            mid = index
            while postorder[index] > postorder[right]:
                index += 1

            return index == right and verify(left, mid - 1) and verify(mid, right - 1)
        if len(postorder) <= 2:
            return True
        return verify(0, len(postorder) - 1)
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(n^2)$。
- **空间复杂度**：$O(n)$。



# [剑指 Offer 34. 二叉树中和为某一值的路径](https://leetcode.cn/problems/er-cha-shu-zhong-he-wei-mou-yi-zhi-de-lu-jing-lcof/)

- 标签：树、深度优先搜索、回溯、二叉树
- 难度：中等

## 题目链接

- [剑指 Offer 34. 二叉树中和为某一值的路径 - 力扣](https://leetcode.cn/problems/er-cha-shu-zhong-he-wei-mou-yi-zhi-de-lu-jing-lcof/)

## 题目大意

给定一棵二叉树的根节点 `root` 和一个整数 `target`。

要求：打印出二叉树中各节点的值的和为 `target` 的所有路径。从根节点开始往下一直到叶节点所经过的节点形成一条路径。

## 解题思路

回溯求解。在回溯的同时，记录下当前路径。同时维护 `target`，每遍历到一个节点，就减去该节点值。如果遇到叶子节点，并且 `target == 0` 时，将当前路径加入答案数组中。然后递归遍历左右子树，并回退当前节点，继续遍历。

## 代码

```python
class Solution:
    def pathSum(self, root: TreeNode, target: int) -> List[List[int]]:
        res = []
        path = []
        def dfs(root: TreeNode, target: int):
            if not root:
                return
            path.append(root.val)
            target -= root.val
            if not root.left and not root.right and target == 0:
                res.append(path[:])
            dfs(root.left, target)
            dfs(root.right, target)
            path.pop()
        dfs(root, target)
        return res

```

# [剑指 Offer 35. 复杂链表的复制](https://leetcode.cn/problems/fu-za-lian-biao-de-fu-zhi-lcof/)

- 标签：哈希表、链表
- 难度：中等

## 题目链接

- [剑指 Offer 35. 复杂链表的复制 - 力扣](https://leetcode.cn/problems/fu-za-lian-biao-de-fu-zhi-lcof/)

## 题目大意

给定一个链表，每个节点除了 `next` 指针之后，还包含一个随机指针 `random`，该指针可以指向链表中的任何节点或者空节点。

要求：将该链表进行深拷贝。

## 解题思路

遍历链表，利用哈希表，以旧节点：新节点为映射关系，将节点关系存储下来。

再次遍历链表，将新链表的 `next` 和 `random` 指针设置好。

## 代码

```python
class Solution:
    def copyRandomList(self, head: 'Node') -> 'Node':
        if not head:
            return None
        node_dict = dict()
        curr = head
        while curr:
            new_node = Node(curr.val, None, None)
            node_dict[curr] = new_node
            curr = curr.next
        curr = head
        while curr:
            if curr.next:
                node_dict[curr].next = node_dict[curr.next]
            if curr.random:
                node_dict[curr].random = node_dict[curr.random]
            curr = curr.next
        return node_dict[head]
```

# [剑指 Offer 36. 二叉搜索树与双向链表](https://leetcode.cn/problems/er-cha-sou-suo-shu-yu-shuang-xiang-lian-biao-lcof/)

- 标签：栈、树、深度优先搜索、二叉搜索树、链表、二叉树、双向链表
- 难度：中等

## 题目链接

- [剑指 Offer 36. 二叉搜索树与双向链表 - 力扣](https://leetcode.cn/problems/er-cha-sou-suo-shu-yu-shuang-xiang-lian-biao-lcof/)

## 题目大意

给定一棵二叉树的根节点 `root`。

要求：将这棵二叉树转换为一个排序的循环双向链表。要求不能创建新的节点，只能调整树中节点指针的指向。

## 解题思路

通过中序递归遍历可以将二叉树升序排列输出。这道题需要在中序遍历的同时，将节点的左右指向进行改变。使用 `head`、`tail` 存放双向链表的头尾节点，然后从根节点开始，进行中序递归遍历。

具体做法如下：

- 如果当前节点为空，直接返回。
- 如果当前节点不为空：
  - 递归遍历左子树。
  - 如果尾节点不为空，则将尾节点与当前节点进行连接。
  - 如果尾节点为空，则初始化头节点。
  - 将当前节点标记为尾节点。
  - 递归遍历右子树。
- 最后将头节点和尾节点进行连接。

## 代码

```python
class Solution:
    def treeToDoublyList(self, root: 'Node') -> 'Node':
        def dfs(node: 'Node'):
            if not node:
                return

            dfs(node.left)
            if self.tail:
                self.tail.right = node
                node.left = self.tail
            else:
                self.head = node
            self.tail = node
            dfs(node.right)

        if not root:
            return None

        self.head, self.tail = None, None
        dfs(root)
        self.head.left = self.tail
        self.tail.right = self.head
        return self.head
```

# [剑指 Offer 37. 序列化二叉树](https://leetcode.cn/problems/xu-lie-hua-er-cha-shu-lcof/)

- 标签：树、深度优先搜索、广度优先搜索、设计、字符串、二叉树
- 难度：困难

## 题目链接

- [剑指 Offer 37. 序列化二叉树 - 力扣](https://leetcode.cn/problems/xu-lie-hua-er-cha-shu-lcof/)

## 题目大意

给定一棵二叉树的根节点 `root`。

要求：设计一个算法，来实现二叉树的序列化与反序列化。

## 解题思路

1. 序列化：将二叉树转为字符串数据表示

按照前序递归遍历二叉树，并将根节点跟左右子树的值链接起来（中间用 `,` 隔开）。

注意：如果遇到空节点，则标记为 'None'，这样在反序列化时才能唯一确定一棵二叉树。

2. 反序列化：将字符串数据转为二叉树结构

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

# [剑指 Offer 38. 字符串的排列](https://leetcode.cn/problems/zi-fu-chuan-de-pai-lie-lcof/)

- 标签：字符串、回溯
- 难度：中等

## 题目链接

- [剑指 Offer 38. 字符串的排列 - 力扣](https://leetcode.cn/problems/zi-fu-chuan-de-pai-lie-lcof/)

## 题目大意

给定一个字符串 `s`。

要求：打印出该字符串中字符的所有排列。可以以任意顺序返回这个字符串数组，但里边不能有重复元素。

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

    def permutation(self, s: str) -> List[str]:
        self.res.clear()
        self.path.clear()
        ls = list(s)
        ls.sort()
        visited = [False for _ in range(len(s))]
        self.backtrack(ls, visited)
        return self.res
```

# [剑指 Offer 39. 数组中出现次数超过一半的数字](https://leetcode.cn/problems/shu-zu-zhong-chu-xian-ci-shu-chao-guo-yi-ban-de-shu-zi-lcof/)

- 标签：数组、哈希表、分治、计数、排序
- 难度：简单

## 题目链接

- [剑指 Offer 39. 数组中出现次数超过一半的数字 - 力扣](https://leetcode.cn/problems/shu-zu-zhong-chu-xian-ci-shu-chao-guo-yi-ban-de-shu-zi-lcof/)

## 题目大意

给定一个数组 `nums`，其中有一个数字出现次数超过数组长度一半。

要求：找到出现次数超过数组长度一半的数字。

## 解题思路

可以利用哈希表。遍历一遍数组 `nums`，用哈希表统计每个元素 `num` 出现的次数，再遍历一遍哈希表，找出元素个数最多的元素即可。

## 代码

```python
class Solution:
    def majorityElement(self, nums: List[int]) -> int:
        numDict = dict()
        for num in nums:
            if num in numDict:
                numDict[num] += 1
            else:
                numDict[num] = 1
        max = 0
        max_index = -1
        for num in numDict:
            if numDict[num] > max:
                max = numDict[num]
                max_index = num
        return max_index
```

# [剑指 Offer 40. 最小的k个数](https://leetcode.cn/problems/zui-xiao-de-kge-shu-lcof/)

- 标签：数组、分治、快速选择、排序、堆（优先队列）
- 难度：简单

## 题目链接

- [剑指 Offer 40. 最小的k个数 - 力扣](https://leetcode.cn/problems/zui-xiao-de-kge-shu-lcof/)

## 题目大意

**描述**：给定整数数组 $arr$，再给定一个整数 $k$。

**要求**：返回数组 $arr$ 中最小的 $k$ 个数。

**说明**：

- $0 \le k \le arr.length \le 10000$。
- $0 \le arr[i] \le 10000$。

**示例**：

- 示例 1：

```python
输入：arr = [3,2,1], k = 2
输出：[1,2] 或者 [2,1]
```

- 示例 2：

```python
输入：arr = [0,1,2,1], k = 1
输出：[0]
```

## 解题思路

直接可以想到的思路是：排序后输出数组上对应的最小的 k 个数。所以问题关键在于排序方法的复杂度。

冒泡排序、选择排序、插入排序时间复杂度 $O(n^2)$ 太高了，解答会超时。

可考虑堆排序、归并排序、快速排序。

### 思路 1：堆排序（基于大顶堆）

具体做法如下：

1. 使用数组前 $k$ 个元素，维护一个大小为 $k$ 的大顶堆。
2. 遍历数组 $[k, size - 1]$ 的元素，判断其与堆顶元素关系，如果遇到比堆顶元素小的元素，则将与堆顶元素进行交换。再将这 $k$ 个元素调整为大顶堆。
3. 最后输出大顶堆的 $k$ 个元素。

### 思路 1：代码

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

    def getLeastNumbers(self, arr: List[int], k: int) -> List[int]:
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

### 思路 1：复杂度分析

- **时间复杂度**：$O(n\log_2k)$。
- **空间复杂度**：$O(1)$。

### 思路 2：快速排序

使用快速排序在每次调整时，都会确定一个元素的最终位置，且以该元素为界限，将数组分成了左右两个子数组，左子数组中的元素都比该元素小，右子树组中的元素都比该元素大。

这样，只要某次划分的元素恰好是第 $k$ 个元素下标，就找到了数组中最小的 $k$ 个数所对应的区间，即 $[0, k - 1]$。 并且我们只需关注第 $k$ 个最小元素所在区间的排序情况，与第 $k$ 个最小元素无关的区间排序都可以忽略。这样进一步减少了执行步骤。

### 思路 2：代码

```python
import random

class Solution:
    # 从 arr[low: high + 1] 中随机挑选一个基准数，并进行移动排序
    def randomPartition(self, arr: [int], low: int, high: int):
        # 随机挑选一个基准数
        i = random.randint(low, high)
        # 将基准数与最低位互换
        arr[i], arr[low] = arr[low], arr[i]
        # 以最低位为基准数，然后将序列中比基准数大的元素移动到基准数右侧，比他小的元素移动到基准数左侧。最后将基准数放到正确位置上
        return self.partition(arr, low, high)
    
    # 以最低位为基准数，然后将序列中比基准数大的元素移动到基准数右侧，比他小的元素移动到基准数左侧。最后将基准数放到正确位置上
    def partition(self, arr: [int], low: int, high: int):
        pivot = arr[low]            # 以第 1 为为基准数
        i = low + 1                 # 从基准数后 1 位开始遍历，保证位置 i 之前的元素都小于基准数
        
        for j in range(i, high + 1):
            # 发现一个小于基准数的元素
            if arr[j] < pivot:
                # 将小于基准数的元素 arr[j] 与当前 arr[i] 进行换位，保证位置 i 之前的元素都小于基准数
                arr[i], arr[j] = arr[j], arr[i]
                # i 之前的元素都小于基准数，所以 i 向右移动一位
                i += 1
        # 将基准节点放到正确位置上
        arr[i - 1], arr[low] = arr[low], arr[i - 1]
        # 返回基准数位置
        return i - 1

    def quickSort(self, arr, low, high, k):
        size = len(arr)
        if low < high:
            # 按照基准数的位置，将序列划分为左右两个子序列
            pi = self.randomPartition(arr, low, high)
            if pi == k:
                return arr[:k]
            if pi > k:
                # 对左子序列进行递归快速排序
                self.quickSort(arr, low, pi - 1, k)
            if pi < k:
                # 对右子序列进行递归快速排序
                self.quickSort(arr, pi + 1, high, k)

        return arr[:k]

    def getLeastNumbers(self, arr: List[int], k: int) -> List[int]:
        size = len(arr)
        if k >= size:
            return arr
        return self.quickSort(arr, 0, size - 1, k)
```

### 思路 2：复杂度分析

- **时间复杂度**：$O(n)$。证明过程可参考「算法导论 9.2：期望为线性的选择算法」。
- **空间复杂度**：$O(\log n)$。递归使用栈空间的空间代价期望为 $O(\log n)$。

# [剑指 Offer 41. 数据流中的中位数](https://leetcode.cn/problems/shu-ju-liu-zhong-de-zhong-wei-shu-lcof/)

- 标签：设计、双指针、数据流、排序、堆（优先队列）
- 难度：困难

## 题目链接

- [剑指 Offer 41. 数据流中的中位数 - 力扣](https://leetcode.cn/problems/shu-ju-liu-zhong-de-zhong-wei-shu-lcof/)

## 题目大意

要求：设计一个支持一下两种操作的数组结构：

- `void addNum(int num)`：从数据流中添加一个整数到数据结构中。
- `double findMedian()`：返回目前所有元素的中位数。

## 解题思路

使用一个大顶堆 `queMax` 记录大于中位数的数，使用一个小顶堆 `queMin` 小于中位数的数。

- 当添加元素数量为偶数： `queMin` 和 `queMax` 中元素数量相同，则中位数为它们队头的平均值。
- 当添加元素数量为奇数：`queMin` 中的数比 `queMax` 多一个，此时中位数为 `queMin` 的队头。

为了满足上述条件，在进行 `addNum` 操作时，我们应当分情况处理：

- `num > max{queMin}`：此时 `num` 大于中位数，将该数添加到大顶堆 `queMax` 中。新的中位数将大于原来的中位数，所以可能需要将 `queMax` 中的最小数移动到 `queMin` 中。
- `num ≤ max{queMin}`：此时 `num` 小于中位数，将该数添加到小顶堆 `queMin` 中。新的中位数将小于等于原来的中位数，所以可能需要将 `queMin` 中最大数移动到 `queMax` 中。

## 代码

```python
import heapq

class MedianFinder:

    def __init__(self):
        """
        initialize your data structure here.
        """
        self.queMin = list()
        self.queMax = list()


    def addNum(self, num: int) -> None:
        if not self.queMin or num < -self.queMin[0]:
            heapq.heappush(self.queMin, -num)
            if len(self.queMax) + 1 < len(self.queMin):
                heapq.heappush(self.queMax, -heapq.heappop(self.queMin))
        else:
            heapq.heappush(self.queMax, num)
            if len(self.queMax) > len(self.queMin):
                heapq.heappush(self.queMin, -heapq.heappop(self.queMax))


    def findMedian(self) -> float:
        if len(self.queMin) > len(self.queMax):
            return -self.queMin[0]
        return (-self.queMin[0] + self.queMax[0]) / 2
```

# [剑指 Offer 42. 连续子数组的最大和](https://leetcode.cn/problems/lian-xu-zi-shu-zu-de-zui-da-he-lcof/)

- 标签：数组、分治、动态规划
- 难度：简单

## 题目链接

- [剑指 Offer 42. 连续子数组的最大和 - 力扣](https://leetcode.cn/problems/lian-xu-zi-shu-zu-de-zui-da-he-lcof/)

## 题目大意

给定一个整数数组 `nums` 。

要求：找到一个具有最大和的连续子数组（子数组最少包含一个元素），返回其最大和，要求时间复杂度为 `O(n)`。

## 解题思路

动态规划的方法，关键点是要找到状态转移方程。

假设 f(i) 表示第 i 个数结尾的「连续子数组的最大和」，那么 $max_{0 < i \le n-1} {f(i)} = max(f(i-1) + nums[i], nums[i])$

即将之前累加和加上当前值与当前值做比较。

- 如果将之前累加和加上当前值 > 当前值，那么加上当前值。
- 如果将之前累加和加上当前值 < 当前值，那么 $f(i) = nums[i]$。

## 代码

```python
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        max_ans = nums[0]
        ans = 0
        for num in nums:
            ans = max(ans + num, num)
            max_ans = max(max_ans, ans)
        return max_ans
```

# [剑指 Offer 44. 数字序列中某一位的数字](https://leetcode.cn/problems/shu-zi-xu-lie-zhong-mou-yi-wei-de-shu-zi-lcof/)

- 标签：数学、二分查找
- 难度：中等

## 题目链接

- [剑指 Offer 44. 数字序列中某一位的数字 - 力扣](https://leetcode.cn/problems/shu-zi-xu-lie-zhong-mou-yi-wei-de-shu-zi-lcof/)

## 题目大意

数字以 `0123456789101112131415…` 的格式序列化到一个字符序列中。在这个序列中，第 `5` 位（从下标 `0` 开始计数）是 `5`，第 `13` 位是 `1`，第 `19` 位是 `4`，等等。

要求：返回任意第 `n` 位对应的数字。

## 解题思路

根据题意中的字符串，找数学规律：

- `123456789`：是 `9` 个 `1` 位数字。
- `10111213...9899`：是 `90` 个 `2` 位数字。
- `100...999`：是 `900` 个 `3` 位数字。
- `1000...9999` 是 `9000` 个 `4` 位数字。

- 我们可以先找到对应的数字对应的位数 `digits`。
- 然后找到该位数 `digits` 的起始数字 `start`。
- 再计算出 `n` 所在的数字 `number`。`number` 等于从起始数字 `start` 开始的第 $\lfloor(n - 1) / digits\rfloor$ 个数字。即 `number = start + (n - 1) // digits`。
- 然后确定 `n` 对应的是数字 `number` 中的哪一位。即 `idx = (n - 1) % digits`。
- 最后返回结果。

## 代码

```python
class Solution:
    def findNthDigit(self, n: int) -> int:
        digits = 1
        start = 1
        base = 9
        while n > base:
            n -= base
            digits += 1
            start *= 10
            base = start * digits * 9

        number = start + (n - 1) // digits
        idx = (n - 1) % digits
        return int(str(number)[idx])
```

## 参考资料

- 【题解】[面试题44. 数字序列中某一位的数字（迭代 + 求整 / 求余，清晰图解） - 数字序列中某一位的数字 - 力扣](https://leetcode.cn/problems/shu-zi-xu-lie-zhong-mou-yi-wei-de-shu-zi-lcof/solution/mian-shi-ti-44-shu-zi-xu-lie-zhong-mou-yi-wei-de-6/)
# [剑指 Offer 45. 把数组排成最小的数](https://leetcode.cn/problems/ba-shu-zu-pai-cheng-zui-xiao-de-shu-lcof/)

- 标签：贪心、字符串、排序
- 难度：中等

## 题目链接

- [剑指 Offer 45. 把数组排成最小的数 - 力扣](https://leetcode.cn/problems/ba-shu-zu-pai-cheng-zui-xiao-de-shu-lcof/)

## 题目大意

**描述**：给定一个非负整数数组 $nums$。

**要求**：将数组中的数字拼接起来排成一个数，打印能拼接出的所有数字中的最小的一个。

**说明**：

- $0 < nums.length \le 100$。
- 输出结果可能非常大，所以你需要返回一个字符串而不是整数。
- 拼接起来的数字可能会有前导 $0$，最后结果不需要去掉前导 $0$。

**示例**：

- 示例 1：

```python
输入: [10,2]
输出: "102"
```

- 示例 2：

```python
输入：[3,30,34,5,9]
输出："3033459"
```

## 解题思路

### 思路 1：自定义排序

本质上是给数组进行排序。假设 $x$、$y$ 是数组 $nums$ 中的两个元素。则排序的判断规则如下所示：

- 如果拼接字符串 $x + y > y + x$，则 $x$ 大于 $y$，$y$ 应该排在 $x$ 前面，从而使拼接起来的数字尽可能的小。
- 反之，如果拼接字符串 $x + y < y + x$，则 $x$ 小于 $y$，$x$ 应该排在 $y$ 前面，从而使拼接起来的数字尽可能的小。

按照上述规则，对原数组进行排序。这里使用了 `functools.cmp_to_key` 自定义排序函数。

### 思路 1：自定义排序代码

```python
import functools

class Solution:
    def minNumber(self, nums: List[int]) -> str:
        def cmp(a, b):
            if a + b == b + a:
                return 0
            elif a + b > b + a:
                return 1
            else:
                return -1

        nums_s = list(map(str, nums))
        nums_s.sort(key=functools.cmp_to_key(cmp))
        return ''.join(nums_s)
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(n \times \log n)$。排序算法的时间复杂度为 $O(n \times \log n)$。
- **空间复杂度**：$O(1)$。

## 参考资料

- 【题解】[剑指 Offer 45. 把数组排成最小的数（自定义排序，清晰图解） - 把数组排成最小的数 - 力扣](https://leetcode.cn/problems/ba-shu-zu-pai-cheng-zui-xiao-de-shu-lcof/solution/mian-shi-ti-45-ba-shu-zu-pai-cheng-zui-xiao-de-s-4/)
# [剑指 Offer 46. 把数字翻译成字符串](https://leetcode.cn/problems/ba-shu-zi-fan-yi-cheng-zi-fu-chuan-lcof/)

- 标签：字符串、动态规划
- 难度：中等

## 题目链接

- [剑指 Offer 46. 把数字翻译成字符串 - 力扣](https://leetcode.cn/problems/ba-shu-zi-fan-yi-cheng-zi-fu-chuan-lcof/)

## 题目大意

给定一个数字 `num`，按照如下规则将其翻译为字符串：`0` 翻译为 `a`，`1` 翻译为 `b`，…，`11` 翻译为 `l`，…，`25` 翻译为 `z`。

要求：计算出共有多少种可能的翻译方案。

## 解题思路

可用动态规划来做。

将数字 `nums` 转为字符串 `s`。设 `dp[i]` 表示字符串 `s` 前 `i` 个数字 `s[0: i]` 的翻译方案数。`dp[i]` 的来源有两种情况：

1. 第 `i - 1`、`i - 2` 构成的数字在 `[10, 25]`之间，则 `dp[i]` 来源于： `s[i - 1]` 单独翻译的方案数（即 `dp[i - 1]`） +  `s[i - 2]` 和 `s[i - 1]` 连起来进行翻译的方案数（即 `dp[i - 2]`）。
2. 第 `i - 1`、`i - 2` 构成的数字在 `[10, 25]`之外，则 `dp[i]` 来源于：`s[i]` 单独翻译的方案数。

## 代码

```python
class Solution:
    def translateNum(self, num: int) -> int:
        s = str(num)
        size = len(s)
        dp = [0 for _ in range(size + 1)]
        dp[0] = 1
        dp[1] = 1
        for i in range(2, size + 1):
            temp = int(s[i-2:i])
            if temp >= 10 and temp <= 25:
                dp[i] = dp[i - 1] + dp[i - 2]
            else:
                dp[i] = dp[i - 1]
        return dp[size]
```

# [剑指 Offer 47. 礼物的最大价值](https://leetcode.cn/problems/li-wu-de-zui-da-jie-zhi-lcof/)

- 标签：数组、动态规划、矩阵
- 难度：中等

## 题目链接

- [剑指 Offer 47. 礼物的最大价值 - 力扣](https://leetcode.cn/problems/li-wu-de-zui-da-jie-zhi-lcof/)

## 题目大意

给定一个 `m * n` 大小的二维矩阵 `grid` 代表棋盘，棋盘的每一格都放有一个礼物，每个礼物有一定的价值（价值大于 `0`）。`grid[i][j]` 表示棋盘第 `i` 行第 `j` 列的礼物价值。我们可以从左上角的格子开始拿礼物，每次只能向右或者向下移动一格，直到到达棋盘的右下角。

要求：计算出最多能拿多少价值的礼物。 

## 解题思路

可以用动态规划求解，设 `dp[i][j]` 是从 `(0, 0)` 到 `(i - 1, j - 1)` 能得礼物的最大价值。

显然 `dp[i][j] = max(dp[i - 1][j] + dp[i][j - 1]) + grid[i][j]`。

因为是自上而下递推 `dp[i-1][j]` 可以用 `dp[j]` 来表示，所以也可以将二维改为一位。状态转移公式为： `dp[j] = max(dp[j], dp[j - 1]) + grid[i][j]`。

## 代码

```python
class Solution:
    def maxValue(self, grid: List[List[int]]) -> int:
        if not grid:
            return 0
        size_m = len(grid)
        size_n = len(grid[0])
        dp = [0 for _ in range(size_n + 1)]
        for i in range(size_m):
            for j in range(size_n):
                dp[j + 1] = max(dp[j], dp[j + 1]) + grid[i][j]
        return dp[size_n]
```

# [剑指 Offer 48. 最长不含重复字符的子字符串](https://leetcode.cn/problems/zui-chang-bu-han-zhong-fu-zi-fu-de-zi-zi-fu-chuan-lcof/)

- 标签：哈希表、字符串、滑动窗口
- 难度：中等

## 题目链接

- [剑指 Offer 48. 最长不含重复字符的子字符串 - 力扣](https://leetcode.cn/problems/zui-chang-bu-han-zhong-fu-zi-fu-de-zi-zi-fu-chuan-lcof/)

## 题目大意

给定一个字符串 `s`。

要求：找出其中不含有重复字符的最长子串的长度。

## 解题思路

利用集合来存储不重复的字符。用两个指针分别指向最长子串的左右节点。遍历字符串，右指针不断右移，利用集合来判断有没有重复的字符，如果没有，就持续向右扩大右边界。如果出现重复字符，就缩小左侧边界。每次移动终止，都要计算一下当前不含重复字符的子串长度，并判断一下是否需要更新最大长度。

## 代码

```python
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        if not s:
            return 0

        letterSet = set()
        right = 0
        ans = 0
        for i in range(len(s)):
            if i != 0:
                letterSet.remove(s[i - 1])
            while right < len(s) and s[right] not in letterSet:
                letterSet.add(s[right])
                right += 1
            ans = max(ans, right - i)
        return ans
```

# [剑指 Offer 49. 丑数](https://leetcode.cn/problems/chou-shu-lcof/)

- 标签：哈希表、数学、动态规划、堆（优先队列）
- 难度：中等

## 题目链接

- [剑指 Offer 49. 丑数 - 力扣](https://leetcode.cn/problems/chou-shu-lcof/)

## 题目大意

给定一个整数 `n`。

要求：找出并返回第 `n` 个丑数。

- 丑数：只包含质因数 `2`、`3`、`5` 的正整数。

## 解题思路

动态规划求解。

定义状态 `dp[i]` 表示第 `i` 个丑数。

状态转移方程为：`dp[i] = min(dp[p2] * 2, dp[p3] * 3, dp[p5] * 5)` ，其中 `p2`、`p3`、`p5` 分别表示当前 `i` 中  `2`、`3`、`5` 的质因子数量。

## 代码

```python
class Solution:
    def nthUglyNumber(self, n: int) -> int:
        dp = [1 for _ in range(n)]
        p2, p3, p5 = 0, 0, 0
        for i in range(1, n):
            dp[i] = min(dp[p2] * 2, dp[p3] * 3, dp[p5] * 5)
            if dp[i] == dp[p2] * 2:
                p2 += 1
            if dp[i] == dp[p3] * 3:
                p3 += 1
            if dp[i] == dp[p5] * 5:
                p5 += 1
        return dp[n - 1]
```

# [剑指 Offer 50. 第一个只出现一次的字符](https://leetcode.cn/problems/di-yi-ge-zhi-chu-xian-yi-ci-de-zi-fu-lcof/)

- 标签：队列、哈希表、字符串、计数
- 难度：简单

## 题目链接

- [剑指 Offer 50. 第一个只出现一次的字符 - 力扣](https://leetcode.cn/problems/di-yi-ge-zhi-chu-xian-yi-ci-de-zi-fu-lcof/)

## 题目大意

给定一个字符串 `s`。

要求：从字符串 `s` 中找到第一个只出现一次的字符。如果没有，则返回空格 ` `。

## 解题思路

遍历字符串 `s`，使用哈希表存储每个字符频数。

再次遍历字符串 `s`，返回第一个频数为 `1` 的字符。

## 代码

```python
class Solution:
    def firstUniqChar(self, s: str) -> str:
        dic = dict()
        for ch in s:
            if ch in dic:
                dic[ch] += 1
            else:
                dic[ch] = 1

        for ch in s:
            if ch in dic and dic[ch] == 1:
                return ch
        return ' '
```

# [剑指 Offer 51. 数组中的逆序对](https://leetcode.cn/problems/shu-zu-zhong-de-ni-xu-dui-lcof/)

- 标签：树状数组、线段树、数组、二分查找、分治、有序集合、归并排序
- 难度：困难

## 题目链接

- [剑指 Offer 51. 数组中的逆序对 - 力扣](https://leetcode.cn/problems/shu-zu-zhong-de-ni-xu-dui-lcof/)

## 题目大意

**描述**：给定一个数组 $nums$。

**要求**：计算出数组中的逆序对的总数。

**说明**：

- **逆序对**：在数组中的两个数字，如果前面一个数字大于后面的数字，则这两个数字组成一个逆序对。
- $0 \le nums.length \le 50000$。

**示例**：

- 示例 1：

```python
输入: [7,5,6,4]
输出: 5
```

## 解题思路

### 思路 1：归并排序

归并排序主要分为：「分解过程」和「合并过程」。其中「合并过程」实质上是两个有序数组的合并过程。

![](https://qcdn.itcharge.cn/images/20220414204405.png)

每当遇到 左子数组当前元素 > 右子树组当前元素时，意味着「左子数组从当前元素开始，一直到左子数组末尾元素」与「右子树组当前元素」构成了若干个逆序对。

比如上图中的左子数组 $[0, 3, 5, 7]$ 与右子树组 $[1, 4, 6, 8]$，遇到左子数组中元素 $3$ 大于右子树组中元素 $1$。则左子数组从 $3$ 开始，经过 $5$ 一直到 $7$，与右子数组当前元素 $1$ 都构成了逆序对。即 $[3, 1]$、$[5, 1]$、$[7, 1]$ 都构成了逆序对。

因此，我们可以在合并两个有序数组的时候计算逆序对。具体做法如下：

1. 使用全局变量 $cnt$ 来存储逆序对的个数。然后进行归并排序。
2. **分割过程**：先递归地将当前序列平均分成两半，直到子序列长度为 $1$。
   1. 找到序列中心位置 $mid$，从中心位置将序列分成左右两个子序列 $left\underline{\hspace{0.5em}}arr$、$right\underline{\hspace{0.5em}}arr$。
   2. 对左右两个子序列 $left\underline{\hspace{0.5em}}arr$、$right\underline{\hspace{0.5em}}arr$ 分别进行递归分割。
   3. 最终将数组分割为 $n$ 个长度均为 $1$ 的有序子序列。
3. **归并过程**：从长度为 $1$ 的有序子序列开始，依次进行两两归并，直到合并成一个长度为 $n$ 的有序序列。
   1. 使用数组变量 $arr$ 存放归并后的有序数组。
   2. 使用两个指针 $left\underline{\hspace{0.5em}}i$、$right\underline{\hspace{0.5em}}i$ 分别指向两个有序子序列 $left\underline{\hspace{0.5em}}arr$、$right\underline{\hspace{0.5em}}arr$ 的开始位置。
   3. 比较两个指针指向的元素：
      1. 如果 $left\underline{\hspace{0.5em}}arr[left\underline{\hspace{0.5em}}i] \le right\underline{\hspace{0.5em}}arr[right\underline{\hspace{0.5em}}i]$，则将 $left\underline{\hspace{0.5em}}arr[left\underline{\hspace{0.5em}}i]$ 存入到结果数组 $arr$ 中，并将指针移动到下一位置。
      2. 如果 $left\underline{\hspace{0.5em}}arr[left\underline{\hspace{0.5em}}i] > right\underline{\hspace{0.5em}}arr[right\underline{\hspace{0.5em}}i]$，则 **记录当前左子序列中元素与当前右子序列元素所形成的逆序对的个数，并累加到 $cnt$ 中，即 `self.cnt += len(left_arr) - left_i`**，然后将 $right\underline{\hspace{0.5em}}arr[right\underline{\hspace{0.5em}}i]$ 存入到结果数组 $arr$ 中，并将指针移动到下一位置。
   4. 重复步骤 $3$，直到某一指针到达子序列末尾。
   5. 将另一个子序列中的剩余元素存入到结果数组 $arr$ 中。
   6. 返回归并后的有序数组 $arr$。
4. 返回数组中的逆序对的总数，即 $self.cnt$。

### 思路 1：代码

```python
class Solution:
    cnt = 0
    def merge(self, left_arr, right_arr):           # 归并过程
        arr = []
        left_i, right_i = 0, 0
        while left_i < len(left_arr) and right_i < len(right_arr):
            # 将两个有序子序列中较小元素依次插入到结果数组中
            if left_arr[left_i] <= right_arr[right_i]:
                arr.append(left_arr[left_i])
                left_i += 1
            else:
                self.cnt += len(left_arr) - left_i
                arr.append(right_arr[right_i])
                right_i += 1
        
        while left_i < len(left_arr):
            # 如果左子序列有剩余元素，则将其插入到结果数组中
            arr.append(left_arr[left_i])
            left_i += 1
            
        while right_i < len(right_arr):
            # 如果右子序列有剩余元素，则将其插入到结果数组中
            arr.append(right_arr[right_i])
            right_i += 1
        
        return arr                                  # 返回排好序的结果数组

    def mergeSort(self, arr):                       # 分割过程
        if len(arr) <= 1:                           # 数组元素个数小于等于 1 时，直接返回原数组
            return arr
        
        mid = len(arr) // 2                         # 将数组从中间位置分为左右两个数组。
        left_arr = self.mergeSort(arr[0: mid])      # 递归将左子序列进行分割和排序
        right_arr =  self.mergeSort(arr[mid:])      # 递归将右子序列进行分割和排序
        return self.merge(left_arr, right_arr)      # 把当前序列组中有序子序列逐层向上，进行两两合并。

    def reversePairs(self, nums: List[int]) -> int:
        self.cnt = 0
        self.mergeSort(nums)
        return self.cnt
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(n \times \log n)$。
- **空间复杂度**：$O(n)$。

### 思路 2：树状数组

数组 $tree[i]$ 表示数字 $i$ 是否在序列中出现过，如果数字 $i$ 已经存在于序列中，$tree[i] = 1$，否则 $tree[i] = 0$。

1. 按序列从左到右将值为 $nums[i]$ 的元素当作下标为$nums[i]$，赋值为 $1$ 插入树状数组里，这时，比 $nums[i]$ 大的数个数就是 $i + 1 - query(a)$。
2. 将全部结果累加起来就是逆序数了。

### 思路 2：代码

```python
import bisect

class BinaryIndexTree:

    def __init__(self, n):
        self.size = n
        self.tree = [0 for _ in range(n + 1)]

    def lowbit(self, index):
        return index & (-index)

    def update(self, index, delta):
        while index <= self.size:
            self.tree[index] += delta
            index += self.lowbit(index)

    def query(self, index):
        res = 0
        while index > 0:
            res += self.tree[index]
            index -= self.lowbit(index)
        return res

class Solution:
    def reversePairs(self, nums: List[int]) -> int:
        size = len(nums)
        sort_nums = sorted(nums)
        for i in range(size):
            nums[i] = bisect.bisect_left(sort_nums, nums[i]) + 1

        bit = BinaryIndexTree(size)
        ans = 0
        for i in range(size):
            bit.update(nums[i], 1)
            ans += (i + 1 - bit.query(nums[i]))
        return ans
```

### 思路 2：复杂度分析

- **时间复杂度**：$O(n \times \log n)$。
- **空间复杂度**：$O(n)$。

# [剑指 Offer 52. 两个链表的第一个公共节点](https://leetcode.cn/problems/liang-ge-lian-biao-de-di-yi-ge-gong-gong-jie-dian-lcof/)

- 标签：哈希表、链表、双指针
- 难度：简单

## 题目链接

- [剑指 Offer 52. 两个链表的第一个公共节点 - 力扣](https://leetcode.cn/problems/liang-ge-lian-biao-de-di-yi-ge-gong-gong-jie-dian-lcof/)

## 题目大意

给定 A、B 两个链表，判断两个链表是否相交，返回相交的起始点。如果不相交，则返回 None。

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
        if not headA or not headB:
            return None
        pA = headA
        pB = headB
        while pA != pB:
            pA = pA.next if pA else headB
            pB = pB.next if pB else headA
        return pA
```

# [剑指 Offer 53 - I. 在排序数组中查找数字 I](https://leetcode.cn/problems/zai-pai-xu-shu-zu-zhong-cha-zhao-shu-zi-lcof/)

- 标签：数组、二分查找
- 难度：简单

## 题目链接

- [剑指 Offer 53 - I. 在排序数组中查找数字 I - 力扣](https://leetcode.cn/problems/zai-pai-xu-shu-zu-zhong-cha-zhao-shu-zi-lcof/)

## 题目大意

给定一个排序数组 `nums`，以及一个整数 `target`。

要求：统计 `target` 在排序数组 `nums` 中出现的次数。

## 解题思路

两次二分查找。

- 先查找 `target` 第一次出现的位置（下标）：`left`。
- 再查找 `target` 最后一次出现的位置（下标）：`right`。
- 最终答案为 `right - left + 1`。

## 代码

```python
class Solution:
    def searchLeft(self, nums, target):
        left, right = 0, len(nums) - 1
        while left < right:
            mid = left + (right - left) // 2
            if nums[mid] < target:
                left = mid + 1
            else:
                right = mid
        if nums[left] == target:
            return left
        else:
            return -1

    def searchRight(self, nums, target):
        left, right = 0, len(nums) - 1
        while left < right:
            mid = left + (right - left + 1) // 2
            if nums[mid] <= target:
                left = mid
            else:
                right = mid - 1
        return left

    def search(self, nums: List[int], target: int) -> int:
        if len(nums) == 0:
            return 0
        left = self.searchLeft(nums, target)
        right = self.searchRight(nums, target)

        if left == -1:
            return 0

        return right - left + 1
```

# [剑指 Offer 53 - II. 0～n-1中缺失的数字](https://leetcode.cn/problems/que-shi-de-shu-zi-lcof/)

- 标签：位运算、数组、哈希表、数学、二分查找
- 难度：简单

## 题目链接

- [剑指 Offer 53 - II. 0～n-1中缺失的数字 - 力扣](https://leetcode.cn/problems/que-shi-de-shu-zi-lcof/)

## 题目大意

给定一个 `n - 1` 个数的升序数组，数组中元素值都在 `0 ~ n - 1` 之间。 `nums` 中有且只有一个数字不在该数组中。

要求：找出这个缺失的数字。

## 解题思路

可以用二分查找解决。

对于中间值，判断元素值与索引值是否一致，如果一致，则说明缺失数字在索引的右侧。如果不一致，则可能为当前索引或者索引的左侧。

## 代码

```python
class Solution:
    def missingNumber(self, nums: List[int]) -> int:
        if len(nums) == 0:
            return 0
        left, right = 0, len(nums) - 1
        while left < right:
            mid = left + (right - left) // 2
            if mid == nums[mid]:
                left = mid + 1
            else:
                right = mid
        if left == nums[left]:
            return left + 1
        else:
            return left
```

# [剑指 Offer 54. 二叉搜索树的第k大节点](https://leetcode.cn/problems/er-cha-sou-suo-shu-de-di-kda-jie-dian-lcof/)

- 标签：树、深度优先搜索、二叉搜索树、二叉树
- 难度：简单

## 题目链接

- [剑指 Offer 54. 二叉搜索树的第k大节点 - 力扣](https://leetcode.cn/problems/er-cha-sou-suo-shu-de-di-kda-jie-dian-lcof/)

## 题目大意

**描述**：给定一棵二叉搜索树的根节点 $root$，以及一个整数 $k$。

**要求**：找出二叉搜索树书第 $k$ 大的节点。

**说明**：

- 

**示例**：

- 示例 1：

![](https://pic.leetcode.cn/1695101634-kzHKZW-image.png)

```python
输入：root = [7, 3, 9, 1, 5], cnt = 2
       7
      / \
     3   9
    / \
   1   5
输出：7
```

- 示例 2：

![](https://pic.leetcode.cn/1695101636-ESZtLa-image.png)

```python
输入: root = [10, 5, 15, 2, 7, null, 20, 1, null, 6, 8], cnt = 4
       10
      / \
     5   15
    / \    \
   2   7    20
  /   / \ 
 1   6   8
输出: 8
```

## 解题思路

### 思路 1：遍历

已知中序遍历「左 -> 根 -> 右」能得到递增序列。逆中序遍历「右 -> 根 -> 左」可以得到递减序列。

则根据「右 -> 根 -> 左」递归遍历 k 次，找到第 $k$ 个节点位置，并记录答案。

### 思路 1：代码

```python
class Solution:
    res = 0
    k = 0
    def dfs(self, root):
        if not root:
            return
        self.dfs(root.right)
        if self.k == 0:
            return
        self.k -= 1
        if self.k == 0:
            self.res = root.val
            return
        self.dfs(root.left)

    def kthLargest(self, root: TreeNode, k: int) -> int:
        self.res = 0
        self.k = k
        self.dfs(root)
        return self.res
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(n)$，其中 $n$ 为树中节点数量。
- **空间复杂度**：$O(n)$。

# [剑指 Offer 55 - I. 二叉树的深度](https://leetcode.cn/problems/er-cha-shu-de-shen-du-lcof/)

- 标签：树、深度优先搜索、广度优先搜索、二叉树
- 难度：简单

## 题目链接

- [剑指 Offer 55 - I. 二叉树的深度 - 力扣](https://leetcode.cn/problems/er-cha-shu-de-shen-du-lcof/)

## 题目大意

给定一个二叉树的根节点 `root`。

要求：找出树的深度。

- 深度：从根节点到叶节点一次经过的节点形成一条路径，最长路径的长度为树的深度。

## 解题思路

递归遍历，先递归遍历左右子树，返回左右子树的高度，则当前节点的高度为左右子树最大深度 + 1。即 `max(left_height, right_height) + 1`。

## 代码

```python
class Solution:
    def maxDepth(self, root: TreeNode) -> int:
        if root == None:
            return 0

        left_height = self.maxDepth(root.left)
        right_height = self.maxDepth(root.right)
        return max(left_height, right_height) + 1
```

# [剑指 Offer 55 - II. 平衡二叉树](https://leetcode.cn/problems/ping-heng-er-cha-shu-lcof/)

- 标签：树、深度优先搜索、二叉树
- 难度：简单

## 题目链接

- [剑指 Offer 55 - II. 平衡二叉树 - 力扣](https://leetcode.cn/problems/ping-heng-er-cha-shu-lcof/)

## 题目大意

给定一棵二叉树的根节点 `root`。

要求：判断该树是不是平衡二叉树。如果是平衡二叉树，返回 `True`，否则，返回 `False`。

- 平衡二叉树：任意节点的左右子树深度不超过 `1`。

## 解题思路

递归遍历二叉树。先递归遍历左右子树，判断左右子树是否平衡，再判断以当前节点为根节点的左右子树是否平衡。

如果遍历的子树是平衡的，则返回它的高度，否则返回 `-1`。

只要出现不平衡的子树，则该二叉树一定不是平衡二叉树。

## 代码

```python
class Solution:
    def isBalanced(self, root: TreeNode) -> bool:
        def height(root: TreeNode) -> int:
            if root == None:
                return False
            leftHeight = height(root.left)
            rightHeight = height(root.right)
            if leftHeight == -1 or rightHeight == -1 or abs(leftHeight - rightHeight) > 1:
                return -1
            else:
                return max(leftHeight, rightHeight) + 1

        return height(root) >= 0
```

# [剑指 Offer 56 - I. 数组中数字出现的次数](https://leetcode.cn/problems/shu-zu-zhong-shu-zi-chu-xian-de-ci-shu-lcof/)

- 标签：位运算、数组
- 难度：中等

## 题目链接

- [剑指 Offer 56 - I. 数组中数字出现的次数 - 力扣](https://leetcode.cn/problems/shu-zu-zhong-shu-zi-chu-xian-de-ci-shu-lcof/)

## 题目大意

给定一个整型数组 `nums` 。`nums` 里除两个数字之外，其他数字都出现了两次。

要求：找出这两个只出现一次的数字。要求时间复杂度是 $O(n)$，空间复杂度是 $O(1)$。

## 解题思路

- 求解这道题之前，我们先来看看如何求解「一个数组中除了某个元素只出现一次以外，其余每个元素均出现两次。」即「[136. 只出现一次的数字](https://leetcode.cn/problems/single-number/)」问题。我们可以对所有数不断进行异或操作，最终可得到单次出现的元素。

- 如果数组中有两个数字只出现一次，其余每个元素均出现两次。那么经过全部异或运算。我们可以得到只出现一次的两个数字的异或结果。
- 根据异或结果的性质，异或运算中如果某一位上为 `1`，则说明异或的两个数在该位上是不同的。根据这个性质，我们将数字分为两组：一组是和该位为 `0` 的数字，另一组是该位为 `1` 的数字。然后将这两组分别进行异或运算，就可以得到最终要求的两个数字。

## 代码

```python
class Solution:
    def singleNumbers(self, nums: List[int]) -> List[int]:
        all_xor = 0
        for num in nums:
            all_xor ^= num
        # 获取所有异或中最低位的 1
        mask = 1
        while all_xor & mask == 0:
            mask <<= 1

        a_xor, b_xor = 0, 0
        for num in nums:
            if num & mask == 0:
                a_xor ^= num
            else:
                b_xor ^= num

        return a_xor, b_xor
```

# [剑指 Offer 57 - II. 和为s的连续正数序列](https://leetcode.cn/problems/he-wei-sde-lian-xu-zheng-shu-xu-lie-lcof/)

- 标签：数学、双指针、枚举
- 难度：简单

## 题目链接

- [剑指 Offer 57 - II. 和为s的连续正数序列 - 力扣](https://leetcode.cn/problems/he-wei-sde-lian-xu-zheng-shu-xu-lie-lcof/)

## 题目大意

**描述**：给定一个正整数 `target`。

**要求**：输出所有和为 `target` 的连续正整数序列（至少含有两个数）。序列中的数字由小到大排列，不同序列按照首个数字从小到大排列。

**说明**：

- $1 \le target \le 10^5$。

**示例**：

- 示例 1：

```python
输入：target = 9
输出：[[2,3,4],[4,5]]
```

- 示例 2：

```python
输入：target = 15
输出：[[1,2,3,4,5],[4,5,6],[7,8]]
```

## 解题思路

### 思路 1：枚举算法

连续正整数序列中元素的最小值大于等于 `1`，而最大值不会超过 `target`。所以我们可以枚举可行的区间，并计算出区间和，将其与 `target` 进行比较，如果相等则将对应的区间元素加入答案数组中，最终返回答案数组。

因为题目要求至少含有两个数，则序列开始元素不会超过 `target` 的一半，所以序列开始元素可以从 `1` 开始，枚举到 `target // 2` 即可。

具体步骤如下：

1. 使用列表变量 `res` 作为答案数组。
2. 使用一重循环 `i`，用于枚举序列开始位置，枚举范围为 `[1, target // 2]`。
3. 使用变量 `cur_sum` 维护当前区间的区间和，`cur_sum` 初始为 `0`。
4. 使用第 `2` 重循环 `j`，用于枚举序列的结束位置，枚举范围为 `[i, target - 1]`，并累积计算当前区间的区间和，即 `cur_sum += j`。
   1. 如果当前区间的区间和大于 `target`，则跳出循环。
   2. 如果当前区间的区间和等于 `target`，则将区间上的元素保存为列表，并添加到答案数组中，然后跳出第 `2` 重循环。
5. 遍历完返回答案数组。

### 思路 1：代码

```python
class Solution:
    def findContinuousSequence(self, target: int) -> List[List[int]]:
        res = []
        for i in range(1, target // 2 + 1):
            cur_sum = 0
            for j in range(i, target):
                cur_sum += j
                if cur_sum > target:
                    break
                if cur_sum == target:
                    cur_res = []
                    for k in range(i, j + 1):
                        cur_res.append(k)
                    res.append(cur_res)
                    break
        return res
```

### 思路 1：复杂度分析

- **时间复杂度**：$target \times \sqrt{target}$。
- **空间复杂度**：$O(1)$。

### 思路 2：滑动窗口

具体做法如下：

- 初始化窗口，令 `left = 1`，`right = 2`。
- 计算 `sum = (left + right) * (right - left + 1) // 2`。
- 如果 `sum == target`，时，将其加入答案数组中。
- 如果 `sum < target` 时，说明需要扩大窗口，则 `right += 1`。
- 如果 `sum > target` 时，说明需要缩小窗口，则 `left += 1`。
- 直到 `left >= right` 时停止，返回答案数组。

### 思路 2：滑动窗口代码

```python
class Solution:
    def findContinuousSequence(self, target: int) -> List[List[int]]:
        left, right = 1, 2
        res = []
        while left < right:
            sum = (left + right) * (right - left + 1) // 2
            if sum == target:
                arr = []
                for i in range(0, right - left + 1):
                    arr.append(i + left)
                res.append(arr)
                left += 1
            elif sum < target:
                right += 1
            else:
                left += 1
        return res
```

### 思路 2：复杂度分析

- **时间复杂度**：$O(target)$。
- **空间复杂度**：$O(1)$。
# [剑指 Offer 57. 和为s的两个数字](https://leetcode.cn/problems/he-wei-sde-liang-ge-shu-zi-lcof/)

- 标签：数组、双指针、二分查找
- 难度：简单

## 题目链接

- [剑指 Offer 57. 和为s的两个数字 - 力扣](https://leetcode.cn/problems/he-wei-sde-liang-ge-shu-zi-lcof/)

## 题目大意

给定一个升序数组 `nums`，以及一个目标整数 `target`。

要求：在数组中查找两个数，使它们的和刚好等于 `target`。

## 解题思路

因为数组是升序的，可以使用双指针。`left`、`right` 分别指向数组首尾位置。

- 计算 `sum = nums[left] + nums[right]`。
- 如果 `sum > target`，则 `right` 进行左移。
- 如果 `sum < target`，则 `left` 进行右移。
- 如果 `sum == target`，则返回 `[nums[left], nums[right]]`。

## 代码

```python
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        left, right = 0, len(nums) - 1
        while left < right:
            sum = nums[left] + nums[right]
            if sum > target:
                right -= 1
            elif sum < target:
                left += 1
            else:
                return nums[left], nums[right]
        return []
```

# [剑指 Offer 58 - I. 翻转单词顺序](https://leetcode.cn/problems/fan-zhuan-dan-ci-shun-xu-lcof/)

- 标签：双指针、字符串
- 难度：简单

## 题目链接

- [剑指 Offer 58 - I. 翻转单词顺序 - 力扣](https://leetcode.cn/problems/fan-zhuan-dan-ci-shun-xu-lcof/)

## 题目大意

给定一个字符串 `s`。

要求：逐个翻转字符串中所有的单词。

说明：

- 数组字符串 `s` 可以再前面、后面或者单词间包含多余的空格。
- 翻转后的单词应当只有一个空格分隔。
- 翻转后的字符串不应该包含额外的空格。

## 解题思路

最简单的就是调用 API 进行切片，翻转。复杂一点的也可以根据 API 的思路写出模拟代码。

## 代码

```python
class Solution:
    def reverseWords(self, s: str) -> str:
        return " ".join(reversed(s.split()))
```

# [剑指 Offer 58 - II. 左旋转字符串](https://leetcode.cn/problems/zuo-xuan-zhuan-zi-fu-chuan-lcof/)

- 标签：数学、双指针、字符串
- 难度：简单

## 题目链接

- [剑指 Offer 58 - II. 左旋转字符串 - 力扣](https://leetcode.cn/problems/zuo-xuan-zhuan-zi-fu-chuan-lcof/)

## 题目大意

给定一个字符串 `s` 和一个整数 `n`。

要求：将字符串 `s` 每个字符向左旋转 `n` 位。

- 左旋转：将字符串前面的若干字符转移到字符串的尾部。

## 解题思路

- 使用数组 `res` 存放答案。
- 先遍历 `[n, len(s) - 1]` 范围的字符，将其存入数组。
- 再遍历 `[0, n - 1]` 范围的字符，将其存入数组。
- 将数组转为字符串返回。

## 代码

```python
class Solution:
    def reverseLeftWords(self, s: str, n: int) -> str:
        res = []
        for i in range(n, len(s)):
            res.append(s[i])
        for i in range(n):
            res.append(s[i])
        return "".join(res)
```

# [剑指 Offer 59 - I. 滑动窗口的最大值](https://leetcode.cn/problems/hua-dong-chuang-kou-de-zui-da-zhi-lcof/)

- 标签：队列、滑动窗口、单调队列、堆（优先队列）
- 难度：困难

## 题目链接

- [剑指 Offer 59 - I. 滑动窗口的最大值 - 力扣](https://leetcode.cn/problems/hua-dong-chuang-kou-de-zui-da-zhi-lcof/)

## 题目大意

给定一个整数数组 `nums` 和滑动窗口的大小 `k`。表示为大小为 `k` 的滑动窗口从数组的最左侧移动到数组的最右侧。我们只能看到滑动窗口内的 `k` 个数字，滑动窗口每次只能向右移动一位。

要求：返回滑动窗口中的最大值。

## 解题思路

暴力求解的话，二重循环，时间复杂度为 $O(n * k)$。

我们可以使用优先队列，每次窗口移动时想优先队列中增加一个节点，并删除一个节点。将窗口中的最大值加入到答案数组中。

## 代码

```python
class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        size = len(nums)
        if size == 0:
            return []

        q = [(-nums[i], i) for i in range(k)]
        heapq.heapify(q)
        res = [-q[0][0]]

        for i in range(k, size):
            heapq.heappush(q, (-nums[i], i))
            while q[0][1] <= i - k:
                heapq.heappop(q)
            res.append(-q[0][0])
        return res
```

# [剑指 Offer 59 - II. 队列的最大值](https://leetcode.cn/problems/dui-lie-de-zui-da-zhi-lcof/)

- 标签：设计、队列、单调队列
- 难度：中等

## 题目链接

- [剑指 Offer 59 - II. 队列的最大值 - 力扣](https://leetcode.cn/problems/dui-lie-de-zui-da-zhi-lcof/)

## 题目大意

要求：设计一个「队列」，实现 `max_value` 函数，可通过 `max_value` 得到大年队列的最大值。并且要求 `max_value`、`push_back`、`pop_front` 的均摊时间复杂度都是 `O(1)`。

## 解题思路

利用空间换时间，使用两个队列。其中一个为原始队列 `queue`，另一个为递减队列 `deque`，`deque` 用来保存队列的最大值，具体做法如下：

- `push_back` 操作：如果 `deque` 队尾元素小于即将入队的元素 `value`，则将小于 `value` 的元素全部出队，再将 `valuew` 入队。否则直接将 `value` 直接入队，这样 `deque` 队首元素保存的就是队列的最大值。
- `pop_front` 操作：先判断 `deque`、`queue` 是否为空，如果 `deque` 或者 `queue` 为空，则说明队列为空，直接返回 `-1`。如果都不为空，从 `queue` 中取出一个元素，并跟 `deque` 队首元素进行比较，如果两者相等则需要将 `deque` 队首元素弹出。
- `max_value` 操作：如果 `deque` 不为空，则返回 `deque` 队首元素。否则返回 `-1`。

## 代码

```python
import collections
import queue


class MaxQueue:

    def __init__(self):
        self.queue = queue.Queue()
        self.deque = collections.deque()


    def max_value(self) -> int:
        if self.deque:
            return self.deque[0]
        else:
            return -1


    def push_back(self, value: int) -> None:
        while self.deque and self.deque[-1] < value:
            self.deque.pop()
        self.deque.append(value)
        self.queue.put(value)


    def pop_front(self) -> int:
        if not self.deque or not self.queue:
            return -1
        ans = self.queue.get()
        if ans == self.deque[0]:
            self.deque.popleft()
        return ans
```

# [剑指 Offer 61. 扑克牌中的顺子](https://leetcode.cn/problems/bu-ke-pai-zhong-de-shun-zi-lcof/)

- 标签：数组、排序
- 难度：简单

## 题目链接

- [剑指 Offer 61. 扑克牌中的顺子 - 力扣](https://leetcode.cn/problems/bu-ke-pai-zhong-de-shun-zi-lcof/)

## 题目大意

给定一个 `5` 位数的数组 `nums` 代表扑克牌中的 `5` 张牌。其中 `2~10` 为数字本身，`A` 用 `1` 表示，`J` 用 `11` 表示，`Q` 用 `12` 表示，`K` 用 `13` 表示，大小王用 `0` 表示，且大小王可以替换任意数字。

要求：判断给定的 `5` 张牌是否是一个顺子，即是否为连续的`5` 个数。

## 解题思路

先不考虑牌中有大小王，如果 `5` 个数是连续的，则这 `5` 个数中最大值最小值的关系为：`最大值 - 最小值 = 4`。如果牌中有大小王可以替换这 `5` 个数中的任意数字，则除大小王之外剩下数的最大值最小值关系为 `最大值 - 最小值 <= 4`。而且剩余数不能有重复数字。于是可以这样进行判断。

遍历 `5` 张牌：

- 如果出现大小王，则跳过。
- 判断 `5` 张牌中是否有重复数，如果有则直接返回 `False`，如果没有则将其加入集合。
- 计算 `5` 张牌的最大值，最小值。

最后判断 `最大值 - 最小值  <= 4` 是否成立。如果成立，返回 `True`，否则返回 `False`。

## 代码

```python
class Solution:
    def isStraight(self, nums: List[int]) -> bool:
        max_num, min_num = 0, 14
        repeat = set()
        for num in nums:
            if num == 0:
                continue
            if num in repeat:
                return False
            repeat.add(num)
            max_num = max(max_num, num)
            min_num = min(min_num, num)
        return max_num - min_num <= 4
```

# [剑指 Offer 62. 圆圈中最后剩下的数字](https://leetcode.cn/problems/yuan-quan-zhong-zui-hou-sheng-xia-de-shu-zi-lcof/)

- 标签：递归、数学
- 难度：简单

## 题目链接

- [剑指 Offer 62. 圆圈中最后剩下的数字 - 力扣](https://leetcode.cn/problems/yuan-quan-zhong-zui-hou-sheng-xia-de-shu-zi-lcof/)

## 题目大意

**描述**：$0$、$1$、…、$n - 1$ 这 $n$ 个数字排成一个圆圈，从数字 $0$ 开始，每次从圆圈里删除第 $m$ 个数字。现在给定整数 $n$ 和 $m$。

**要求**：求出这个圆圈中剩下的最后一个数字。 

**说明**：

- $1 \le num \le 10^5$。
- $1 \le target \le 10^6$。

**示例**：

- 示例 1：

```python
输入：num = 7, target = 4
输出：1
```

- 示例 2：

```python
输入：num = 12, target = 5
输出：0
```

## 解题思路

### 思路 1：枚举 + 模拟

模拟循环删除，需要进行 $n - 1$ 轮，每轮需要对节点进行 $m$ 次访问操作。总体时间复杂度为 $O(n \times m)$。

可以通过找规律来做，以 $n = 5$、$m = 3$ 为例。

- 刚开始为 $0$、$1$、$2$、$3$、$4$。
- 第一次从 $0$ 开始数，数 $3$ 个数，于是 $2$ 出圈，变为 $3$、$4$、$0$、$1$。
- 第二次从 $3$ 开始数，数 $3$ 个数，于是 $0$ 出圈，变为 $1$、$3$、$4$。
- 第三次从 $1$ 开始数，数 $3$ 个数，于是 $4$ 出圈，变为 $1$、$3$。
- 第四次从 $1$ 开始数，数 $3$ 个数，于是 $1$ 出圈，变为 $3$。
- 所以最终为 $3$。

通过上面的流程可以发现：每隔 $m$ 个数就要删除一个数，那么被删除的这个数的下一个数就会成为新的起点。就相当于数组进行左移了 $m$ 位。反过来思考的话，从最后一步向前推，则每一步都向右移动了 $m$ 位（包括胜利者）。

如果用 $f(n, m)$ 表示： $n$ 个数构成环没删除 $m$ 个数后，最终胜利者的位置，则 $f(n, m) = f(n - 1, m) + m$。

即等于 $n - 1$ 个数构成的环没删除 $m$ 个数后最终胜利者的位置，像右移动 $m$ 次。

问题是现在并不是真的进行了右移，因为当前数组右移后超过数组容量的部分应该重新放到数组头部位置。所以公式应为：$f(n, m) = [f(n - 1, m) + m] \mod n$，$n$ 为反过来向前推的时候，每一步剩余的数字个数（比如第二步推回第一步，n $4$），则反过来递推公式为：

- $f(1, m) = 0$。
- $f(2, m) = [f(1, m) + m] \mod 2$。
- $f(3, m) = [f(2, m) + m] \mod 3$。
- 。。。。。。

- $f(n, m) = [f(n - 1, m) + m] \mod n $。

接下来就是递推求解了。

### 思路 1：代码

```python
class Solution:
    def lastRemaining(self, n: int, m: int) -> int:
        ans = 0
        for i in range(2, n + 1):
            ans = (m + ans) % i
        return ans
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(n)$。
- **空间复杂度**：$O(1)$。

## 参考资料：

- [字节题库 - #剑62 - 简单 - 圆圈中最后剩下的数字 - 1刷](https://leetcode.cn/problems/yuan-quan-zhong-zui-hou-sheng-xia-de-shu-zi-lcof/solution/zi-jie-ti-ku-jian-62-jian-dan-yuan-quan-3hlji/)
# [剑指 Offer 63. 股票的最大利润](https://leetcode.cn/problems/gu-piao-de-zui-da-li-run-lcof/)

- 标签：数组、动态规划
- 难度：中等

## 题目链接

- [剑指 Offer 63. 股票的最大利润 - 力扣](https://leetcode.cn/problems/gu-piao-de-zui-da-li-run-lcof/)

## 题目大意

给定一个数组 `nums`，`nums[i]` 表示一支给定股票第 `i` 天的价格。选择某一天买入这只股票，并选择在未来的某一个不同的日子卖出该股票。求能获取的最大利润。

## 解题思路

最简单的思路当然是两重循环暴力枚举，寻找不同天数下的最大利润。

但更好的做法是进行一次遍历。设置两个变量 `minprice`（用来记录买入的最小值）、`maxprofit`（用来记录可获取的最大利润）。

进行一次遍历，遇到当前价格比 `minprice` 还要小的，就更新 `minprice`。如果单签价格大于或者等于 `minprice`，则判断一下以当前价格卖出的话能卖多少，如果比 `maxprofit` 还要大，就更新 `maxprofit`。

## 代码

```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        minprice = 10010
        maxprofit = 0
        for price in prices:
            if price < minprice:
                minprice = price
            elif price - minprice > maxprofit:
                maxprofit = price - minprice
        return maxprofit
```

# [剑指 Offer 64. 求1+2+…+n](https://leetcode.cn/problems/qiu-12n-lcof/)

- 标签：位运算、递归、脑筋急转弯
- 难度：中等

## 题目链接

- [剑指 Offer 64. 求1+2+…+n - 力扣](https://leetcode.cn/problems/qiu-12n-lcof/)

## 题目大意

给定一个整数 `n`。

要求：计算 `1 + 2 + ... + n`，并且不能使用乘除法、for、while、if、else、switch、case 等关键字及条件判断语句（A?B:C）。

## 解题思路

Python 中的逻辑运算最终返回的是最后一个非空值。比如 `3 and 2 and 'a'` 最终返回的是 `'a'`。利用这个特性可以递归求解。

## 代码

```python
class Solution:
    def sumNums(self, n: int) -> int:
        return n and n + self.sumNums(n - 1)
```

# [剑指 Offer 65. 不用加减乘除做加法](https://leetcode.cn/problems/bu-yong-jia-jian-cheng-chu-zuo-jia-fa-lcof/)

- 标签：位运算、数学
- 难度：简单

## 题目链接

- [剑指 Offer 65. 不用加减乘除做加法 - 力扣](https://leetcode.cn/problems/bu-yong-jia-jian-cheng-chu-zuo-jia-fa-lcof/)

## 题目大意

给定两个整数 `a`、`b`。

要求：不能使用运算符 `+`、`-`、`*`、`/`，计算两整数 `a` 、`b` 之和。

## 解题思路

需要用到位运算的一些知识。

- 异或运算 a ^ b ：可以获得 a + b 无进位的加法结果。
- 与运算 a & b：对应位置为 1，说明 a、b 该位置上原来都为 1，则需要进位。
- 座椅运算 a << 1：将 a 对应二进制数左移 1 位。

这样，通过 a^b 运算，我们可以得到相加后无进位结果，再根据 (a&b) << 1，计算进位后结果。

进行 a^b 和 (a&b) << 1操作之后判断进位是否为 0，若不为 0，则继续上一步操作，直到进位为 0。

> 注意：
>
> Python 的整数类型是无限长整数类型，负数不确定符号位是第几位。所以我们可以将输入的数字手动转为 32 位无符号整数。
>
> 通过 a &= 0xFFFFFFFF 即可将 a 转为 32 位无符号整数。最后通过对 a 的范围判断，将其结果映射为有符号整数。

## 代码

```python
class Solution:
    def getSum(self, a: int, b: int) -> int:
        MAX_INT = 0x7FFFFFFF
        MASK = 0xFFFFFFFF
        a &= MASK
        b &= MASK
        while b:
            carry = ((a & b) << 1) & MASK
            a ^= b
            b = carry
        if a <= MAX_INT:
            return a
        else:
            return ~(a ^ MASK)
```

# [剑指 Offer 66. 构建乘积数组](https://leetcode.cn/problems/gou-jian-cheng-ji-shu-zu-lcof/)

- 标签：数组、前缀和
- 难度：中等

## 题目链接

- [剑指 Offer 66. 构建乘积数组 - 力扣](https://leetcode.cn/problems/gou-jian-cheng-ji-shu-zu-lcof/)

## 题目大意

给定一个数组 `A`。

要求：构建一个数组 `B`，其中 `B[i]` 为数组 `A` 中除了 `A[i]` 之外的其他所有元素乘积。

要求不能使用除法。

## 解题思路

构造一个答案数组 `B`，长度和数组 `A` 长度一致。先从左到右遍历一遍 `A` 数组，将 `A[i]` 左侧的元素乘积累积起来，存储到 `B` 数组中。再从右到左遍历一遍，将 `A[i]` 右侧的元素乘积累积起来，再乘以原本 `B[i]` 的值，即为 `A` 中除了 `A[i]` 之外的其他所有元素乘积。

## 代码

```python
class Solution:
    def constructArr(self, a: List[int]) -> List[int]:
        size = len(a)
        b = [1 for _ in range(size)]

        left = 1
        for i in range(size):
            b[i] *= left
            left *= a[i]

        right = 1
        for i in range(size - 1, -1, -1):
            b[i] *= right
            right *= a[i]
        return b
```

# [剑指 Offer 67. 把字符串转换成整数](https://leetcode.cn/problems/ba-zi-fu-chuan-zhuan-huan-cheng-zheng-shu-lcof/)

- 标签：字符串
- 难度：中等

## 题目链接

- [剑指 Offer 67. 把字符串转换成整数 - 力扣](https://leetcode.cn/problems/ba-zi-fu-chuan-zhuan-huan-cheng-zheng-shu-lcof/)

## 题目大意

给定一个字符串 `str`。

要求：使其能换成一个 32 位有符号整数。并且该方法满足以下要求：

- 丢弃开头无用的空格字符，直到找到第一个非空格字符为止。
- 当找到的第一个非空字符为正负号时，将该符号与后面尽可能多的连续数组组合起来，作为该整数的正负号。如果第一个非空字符为数字，则直接将其与之后连续的数字字符组合起来，形成整数。
- 该字符串中除了有效的整数部分之后也可能会存在多余字符，可直接将这些字符忽略，不会对函数造成影响。
- 如果第一个非空格字符不是一个有效整数字符、或者字符串为空、字符串仅包含空白字符时，函数不需要进行转换。
- 需要检测有效性，无法读取返回 0。
- 所有整数范围为 $[-2^{31}, 2^{31} - 1]$，超过这个范围，则返回 $2^{31} - 1$ 或者 $-2^{31}$。

## 解题思路

根据题意直接模拟即可。

1. 先去除前后空格。
2. 检测正负号。
3. 读入数字，并用字符串存储数字结果
4. 将数字字符串转为整数，并根据正负号转换整数结果。
5. 判断整数范围，并返回最终结果。

## 代码

```python
class Solution:
    def strToInt(self, str: str) -> int:
        num_str = ""
        positive = True
        start = 0

        s = str.lstrip()
        if not s:
            return 0

        if s[0] == '-':
            positive = False
            start = 1
        elif s[0] == '+':
            positive = True
            start = 1
        elif not s[0].isdigit():
            return 0

        for i in range(start, len(s)):
            if s[i].isdigit():
                num_str += s[i]
            else:
                break
        if not num_str:
            return 0
        num = int(num_str)
        if not positive:
            num = -num
            return max(num, -2 ** 31)
        else:
            return min(num, 2 ** 31 - 1)
```

# [剑指 Offer 68 - I. 二叉搜索树的最近公共祖先](https://leetcode.cn/problems/er-cha-sou-suo-shu-de-zui-jin-gong-gong-zu-xian-lcof/)

- 标签：树、深度优先搜索、二叉搜索树、二叉树
- 难度：简单

## 题目链接

- [剑指 Offer 68 - I. 二叉搜索树的最近公共祖先 - 力扣](https://leetcode.cn/problems/er-cha-sou-suo-shu-de-zui-jin-gong-gong-zu-xian-lcof/)

## 题目大意

给定一棵二叉搜索树的根节点 `root` 和两个指定节点 `p`、`q`。

要求：找到该树中两个指定节点 `p`、`q` 的最近公共祖先。

- 祖先：若节点 `p` 在节点 `node` 的左子树或右子树中，或者 `p == node`，则称 `node` 是 `p` 的祖先。
- 最近公共祖先：对于树的两个节点 `p`、`q`，最近公共祖先表示为一个节点 `lca_node`，满足 `lca_node` 是 `p`、`q` 的祖先且 `lca_node` 的深度尽可能大（一个节点也可以是自己的祖先）

## 解题思路

对于节点 `p`、节点 `q`，最近公共祖先就是从根节点分别到它们路径上的分岔点，也是路径中最后一个相同的节点，现在我们的问题就是求这个分岔点。

使用递归遍历查找最近公共祖先。

- 从根节点开始遍历；
  - 如果当前节点的值大于 `p`、`q` 的值，说明 `p` 和 `q`  应该在当前节点的左子树，因此将当前节点移动到它的左子节点，继续遍历；
  - 如果当前节点的值小于 `p`、`q` 的值，说明 `p` 和 `q`  应该在当前节点的右子树，因此将当前节点移动到它的右子节点，继续遍历；
  - 如果当前节点不满足上面两种情况，则说明 `p` 和 `q` 分别在当前节点的左右子树上，则当前节点就是分岔点，直接返回该节点即可。

## 代码

```python
class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        ancestor = root
        while True:
            if ancestor.val > p.val and ancestor.val > q.val:
                ancestor = ancestor.left
            elif ancestor.val < p.val and ancestor.val < q.val:
                ancestor = ancestor.right
            else:
                break
        return ancestor
```

# [剑指 Offer 68 - II. 二叉树的最近公共祖先](https://leetcode.cn/problems/er-cha-shu-de-zui-jin-gong-gong-zu-xian-lcof/)

- 标签：树、深度优先搜索、二叉树
- 难度：简单

## 题目链接

- [剑指 Offer 68 - II. 二叉树的最近公共祖先 - 力扣](https://leetcode.cn/problems/er-cha-shu-de-zui-jin-gong-gong-zu-xian-lcof/)

## 题目大意

给定一个二叉树的根节点 `root`，再给定两个指定节点 `p`、`q`。

要求：找到两个指定节点 `p`、`q` 的最近公共祖先。

- 祖先：若节点 `p` 在节点 `node` 的左子树或右子树中，或者 `p == node`，则称 `node` 是 `p` 的祖先。
- 最近公共祖先：对于树的两个节点 `p`、`q`，最近公共祖先表示为一个节点 `lca_node`，满足 `lca_node` 是 `p`、`q` 的祖先且 `lca_node` 的深度尽可能大（一个节点也可以是自己的祖先）

## 解题思路

设 `lca_node` 为节点 `p`、`q` 的最近公共祖先。则 `lca_node` 只能是下面几种情况：

- `p`、`q`  在 `lca_node` 的子树中，且分别在 `lca_node` 的两侧子树中。
- `p = lca_node`，且 `q` 在 `lca_node` 的左子树或右子树中。
- `q = lca_node`，且 `p` 在 `lca_node` 的左子树或右子树中。

下面递归求解 `lca_node`。递归需要满足以下条件：

- 如果 `p`、`q` 都不为空，则返回 `p`、`q` 的公共祖先。
- 如果 `p`、`q` 只有一个存在，则返回存在的一个。
- 如果 `p`、`q` 都不存在，则返回存在的一个。

具体思路为：

- 如果当前节点 `node` 为 `None`，则说明 `p`、`q` 不在 `node` 的子树中，不可能为公共祖先，直接返回 `None`。
- 如果当前节点 `node` 等于 `p` 或者 `q`，那么 `node` 就是 `p`、`q` 的最近公共祖先，直接返回 `node`
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

# [剑指 Offer II 001. 整数除法](https://leetcode.cn/problems/xoh6Oh/)

- 标签：位运算、数学
- 难度：简单

## 题目链接

- [剑指 Offer II 001. 整数除法 - 力扣](https://leetcode.cn/problems/xoh6Oh/)

## 题目大意

给定两个整数，被除数 dividend 和除数 divisor。要求返回两数相除的商，并且不能使用乘法，除法和取余运算。取值范围在 $[-2^{31}, 2^{31}-1]$。如果结果溢出，则返回 $2^{31} - 1$。

## 解题思路

题目要求不能使用乘法，除法和取余运算。

可以把被除数和除数当做二进制，这样进行运算的时候，就可以通过移位运算来实现二进制的乘除。

- 先将除数不断左移，移位到位数大于或等于被除数。记录其移位次数 count。

- 然后再将除数右移 count 次，模拟二进制除法运算。
  - 如果当前被除数大于等于除数，则将 1 左移 count 位，即为当前位的商，并将其累加答案上。再用除数减去被除数，进行下一次运算。

## 代码

```python

添加备注


class Solution:
    def divide(self, a: int, b: int) -> int:
        MIN_INT, MAX_INT = -2147483648, 2147483647
        symbol = True if (a ^ b) < 0 else False
        if a < 0:
            a = -a
        if b < 0:
            b = -b

        # 除数不断左移，移位到位数大于或等于被除数
        count = 0
        while a >= b:
            count += 1
            b <<= 1

        # 向右移位，不断模拟二进制除法运算
        res = 0
        while count > 0:
            count -= 1
            b >>= 1
            if a >= b:
                res += (1 << count)
                a -= b
        if symbol:
            res = -res
        if MIN_INT <= res <= MAX_INT:
            return res
        else:
            return MAX_INT
```

# [剑指 Offer II 002. 二进制加法](https://leetcode.cn/problems/JFETK5/)

- 标签：位运算、数学、字符串、模拟
- 难度：简单

## 题目链接

- [剑指 Offer II 002. 二进制加法 - 力扣](https://leetcode.cn/problems/JFETK5/)

## 题目大意

给定两个二进制数的字符串 `a`、`b`。

要求：计算 `a` 和 `b` 的和，返回结果也用二进制表示。

## 解题思路

这道题可以直接将 `a`、`b` 转换为十进制数，相加后再转换为二进制数。

也可以利用位运算的一些知识，直接求和。

因为 `a`、`b` 为二进制的字符串，先将其转换为二进制数。

本题用到的位运算知识：

- 异或运算 `x ^ y` ：可以获得 `x + y` 无进位的加法结果。
- 与运算 `x & y`：对应位置为 `1`，说明 `x`、`y` 该位置上原来都为 `1`，则需要进位。
- 座椅运算 `x << 1`：将 a 对应二进制数左移 `1` 位。

这样，通过 `x ^ y` 运算，我们可以得到相加后无进位结果，再根据 `(x & y) << 1`，计算进位后结果。

进行 `x ^ y` 和 `(x & y) << 1`操作之后判断进位是否为 `0`，若不为 `0`，则继续上一步操作，直到进位为 `0`。

最后将其结果转为 `2` 进制返回。

## 代码

```python
class Solution:
    def addBinary(self, a: str, b: str) -> str:
        x = int(a, 2)
        y = int(b, 2)
        while y:
            carry = ((x & y) << 1)
            x ^= y
            y = carry
        return bin(x)[2:]
```

# [剑指 Offer II 003. 前 n 个数字二进制中 1 的个数](https://leetcode.cn/problems/w3tCBm/)

- 标签：位运算、动态规划
- 难度：简单

## 题目链接

- [剑指 Offer II 003. 前 n 个数字二进制中 1 的个数 - 力扣](https://leetcode.cn/problems/w3tCBm/)

## 题目大意

给定一个整数 `n`。

要求：对于 `0 ≤ i ≤ n` 的每一个 `i`，计算其二进制表示中 `1` 的个数，返回一个长度为 `n + 1` 的数组 `ans` 作为答案。

## 解题思路

可以根据整数的二进制特点将其分为两类：

- 奇数：一定比前面相邻的偶数多一个 `1`。
- 偶数：一定和除以 `2` 之后的数一样多。
- 边界 `0`：`1` 的个数为 `0`。

于是可以根据规律，从 `0` 开始到 `n` 进行递推求解。

## 代码

```python
class Solution:
    def countBits(self, n: int) -> List[int]:
        dp = [0 for _ in range(n + 1)]
        for i in range(1, n + 1):
            if i % 2 == 1:
                dp[i] = dp[i - 1] + 1
            else:
                dp[i] = dp[i // 2]
        return dp
```

# [剑指 Offer II 004. 只出现一次的数字](https://leetcode.cn/problems/WGki4K/)

- 标签：位运算、数组
- 难度：中等

## 题目链接

- [剑指 Offer II 004. 只出现一次的数字 - 力扣](https://leetcode.cn/problems/WGki4K/)

## 题目大意

给定一个整数数组 `nums`，除了某个元素仅出现一次外，其余每个元素恰好出现三次。

要求：找到并返回那个只出现了一次的元素。

## 解题思路

### 1. 哈希表

朴素解法就是利用哈希表。统计出每个元素的出现次数。再遍历哈希表，找到仅出现一次的元素。

### 2. 位运算

将出现三次的元素换成二进制形式放在一起，其二进制对应位置上，出现 `1` 的个数一定是 `3` 的倍数（包括 `0`）。此时，如果在放进来只出现一次的元素，则某些二进制位置上出现 `1` 的个数就不是 `3` 的倍数了。

将这些二进制位置上出现 `1` 的个数不是 `3` 的倍数位置值置为 `1`，是 `3` 的倍数则置为 `0`。这样对应下来的二进制就是答案所求。

注意：因为 Python 的整数没有位数限制，所以不能通过最高位确定正负。所以 Python 中负整数的补码会被当做正整数。所以在遍历到最后 `31` 位时进行 `ans -= (1 << 31)` 操作，目的是将负数的补码转换为「负号 + 原码」的形式。这样就可以正常识别二进制下的负数。参考：[Two's Complement Binary in Python? - Stack Overflow](https://stackoverflow.com/questions/12946116/twos-complement-binary-in-python/12946226)

## 代码

1. 哈希表

```python
class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        nums_dict = dict()
        for num in nums:
            if num in nums_dict:
                nums_dict[num] += 1
            else:
                nums_dict[num] = 1
        for key in nums_dict:
            value = nums_dict[key]
            if value == 1:
                return key
        return 0
```

2. 位运算

```python
class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        ans = 0
        for i in range(32):
            count = 0
            for j in range(len(nums)):
                count += (nums[j] >> i) & 1
            if count % 3 != 0:
                if i == 31:
                    ans -= (1 << 31)
                else:
                    ans = ans | 1 << i
        return ans
```



# [剑指 Offer II 005. 单词长度的最大乘积](https://leetcode.cn/problems/aseY1I/)

- 标签：位运算、数组、字符串
- 难度：中等

## 题目链接

- [剑指 Offer II 005. 单词长度的最大乘积 - 力扣](https://leetcode.cn/problems/aseY1I/)

## 题目大意

给定一个字符串数组 `words`。字符串中只包含英语的小写字母。

要求：计算当两个字符串 `words[i]` 和 `words[j]` 不包含相同字符时，它们长度的乘积的最大值。如果没有不包含相同字符的一对字符串，返回 0。

## 解题思路

这道题的核心难点是判断任意两个字符串之间是否包含相同字符。最直接的做法是先遍历第一个字符串的每个字符，再遍历第二个字符串查看是否有相同字符。但是这样做的话，时间复杂度过高。考虑怎么样可以优化一下。

题目中说字符串中只包含英语的小写字母，也就是 `26` 种字符。一个 `32` 位的 `int` 整数每一个二进制位都可以表示一种字符的有无，那么我们就可以通过一个整数来表示一个字符串中所拥有的字符种类。延伸一下，我们可以用一个整数数组来表示一个字符串数组中，每个字符串所拥有的字符种类。

接下来事情就简单了，两重循环遍历整数数组，遇到两个字符串不包含相同字符的情况，就计算一下他们长度的乘积，并维护一个乘积最大值。最后输出最大值即可。

## 代码

```python
class Solution:
    def maxProduct(self, words: List[str]) -> int:
        size = len(words)
        arr = [0 for _ in range(size)]
        for i in range(size):
            word = words[i]
            len_word = len(word)
            for j in range(len_word):
                arr[i] |= 1 << (ord(word[j]) - ord('a'))
        ans = 0
        for i in range(size):
            for j in range(i + 1, size):
                if arr[i] & arr[j] == 0:
                    k = len(words[i]) * len(words[j])
                    ans = k if ans < k else ans
        return ans
```

# [剑指 Offer II 006. 排序数组中两个数字之和](https://leetcode.cn/problems/kLl5u1/)

- 标签：数组、双指针、二分查找
- 难度：简单

## 题目链接

- [剑指 Offer II 006. 排序数组中两个数字之和 - 力扣](https://leetcode.cn/problems/kLl5u1/)

## 题目大意

给定一个升序数组：`numbers` 和一个目标值 `target`。

要求：从数组中找出满足相加之和等于 `target` 的两个数，并返回两个数在数组中下的标值。

## 解题思路

因为数组是有序的，所以我们可以使用两个指针 low，high。low 指向数组开始较小元素位置，high 指向数组较大元素位置。判断两个位置上的元素和，如果和等于目标值，则返回两个元素位置。如果和大于目标值，则 high 左移，继续检测。如果和小于目标值，则 low 右移，继续检测。直到 low 和 high 移动到相同位置停止检测。若最终仍没找到，则返回 [0, 0]。

## 代码

```python
class Solution:
    def twoSum(self, numbers: List[int], target: int) -> List[int]:
        low = 0
        high = len(numbers) - 1
        while low < high:
            total = numbers[low] + numbers[high]
            if total == target:
                return [low, high]
            elif total < target:
                low += 1
            else:
                high -= 1
        return [0, 0]
```

# [剑指 Offer II 007. 数组中和为 0 的三个数](https://leetcode.cn/problems/1fGaJU/)

- 标签：数组、双指针、排序
- 难度：中等

## 题目链接

- [剑指 Offer II 007. 数组中和为 0 的三个数 - 力扣](https://leetcode.cn/problems/1fGaJU/)

## 题目大意

给定一个包含 `n` 个整数的数组 `nums`，判断 `nums` 中是否存在三个元素 `a`、`b`、`c`，满足 `a + b + c = 0`。

要求：找出所有满足要求的不重复的三元组。

## 解题思路

直接三重遍历查找 a、b、c 的时间复杂度是：$O(n^3)$。我们可以通过一些操作来降低复杂度。

先将数组进行排序，以保证按顺序查找 a、b、c 时，元素值为升序，从而保证所找到的三个元素是不重复的。同时也方便下一步使用双指针减少一重遍历。时间复杂度为：$O(nlogn)$

第一重循环遍历 a，对于每个 a 元素，从 a 元素的下一个位置开始，使用双指针 left，right。left 指向 a 元素的下一个位置，right 指向末尾位置。先将 left 右移、right 左移去除重复元素，再进行下边的判断。

- 若 `nums[a] + nums[left] + nums[right] = 0`，则得到一个解，将其加入答案数组中，并继续将 left 右移，right 左移；

- 若 `nums[a] + nums[left] + nums[right] > 0`，说明 nums[right] 值太大，将 right 向左移；
- 若 `nums[a] + nums[left] + nums[right] < 0`，说明 nums[left] 值太小，将 left 右移。

## 代码

```python
class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        n = len(nums)
        nums.sort()
        ans = []

        for i in range(n):
            if i > 0 and nums[i] == nums[i - 1]:
                continue
            left = i + 1
            right = n - 1
            while left < right:
                while left < right and left > i + 1 and nums[left] == nums[left - 1]:
                    left += 1
                while left < right and right < n - 1 and nums[right + 1] == nums[right]:
                    right -= 1
                if left < right and nums[i] + nums[left] + nums[right] == 0:
                    ans.append([nums[i], nums[left], nums[right]])
                    left += 1
                    right -= 1
                elif nums[i] + nums[left] + nums[right] > 0:
                    right -= 1
                else:
                    left += 1
        return ans
```

# [剑指 Offer II 008. 和大于等于 target 的最短子数组](https://leetcode.cn/problems/2VG8Kg/)

- 标签：数组、二分查找、前缀和、滑动窗口
- 难度：中等

## 题目链接

- [剑指 Offer II 008. 和大于等于 target 的最短子数组 - 力扣](https://leetcode.cn/problems/2VG8Kg/)

## 题目大意

给定一个只包含正整数的数组 `nums` 和一个正整数 `target`。

要求：找出数组中满足和大于等于 `target` 的长度最小的「连续子数组」，并返回其长度。

## 解题思路

最直接的做法是暴力枚举，时间复杂度为 $O(n^2)$。但是我们可以利用滑动窗口的方法，在时间复杂度为 $O(n)$ 的范围内解决问题。

定义两个指针 `start` 和 `end`。`start` 代表滑动窗口开始位置，`end` 代表滑动窗口结束位置。再定义一个变量 `sum` 用来存储滑动窗口中的元素和，一个变量 `ans` 来存储满足提议的最小长度。

先不断移动 `end`，直到 `sum ≥ target`，则更新最小长度值 `ans`。然后再将滑动窗口的起始位置从滑动窗口中移出去，直到 `sum ≤ target`，在移出的期间，同样要更新最小长度值 `ans`。

然后等满足 `sum ≤ target` 时，再移动 `end`，重复上一步，直到遍历到数组末尾。

## 代码

```python
class Solution:
    def minSubArrayLen(self, target: int, nums: List[int]) -> int:
        if not nums:
            return 0
        n = len(nums)

        start = 0
        end = 0
        sum = 0
        ans = n + 1
        while end < n:
            sum += nums[end]
            while sum >= target:
                ans = min(ans, end - start + 1)
                sum -= nums[start]
                start += 1
            end += 1
        if ans == n + 1:
            return 0
        else:
            return ans
```

# [剑指 Offer II 009. 乘积小于 K 的子数组](https://leetcode.cn/problems/ZVAVXX/)

- 标签：数组、滑动窗口
- 难度：中等

## 题目链接

- [剑指 Offer II 009. 乘积小于 K 的子数组 - 力扣](https://leetcode.cn/problems/ZVAVXX/)

## 题目大意

给定一个正整数数组 `nums` 和一个整数 `k`。

要求：找出该数组内乘积小于 `k` 的连续子数组的个数。

## 解题思路

滑动窗口求解。

设定两个指针：`left`、`right`，分别指向滑动窗口的左右边界，保证窗口内所有数的乘积 `product` 都小于 `k`。

- 一开始，`left`、`right` 都指向 `0`。

- 向右移动 `right`，将最右侧元素加入当前子数组乘积 `product` 中。

- 如果 `product >= k` ，则不断右移 `left`，缩小滑动窗口，并更新当前乘积值 `product`  直到 `product < k`。
- 累积答案个数 += 1，继续右移 `right`，直到 `right >= len(nums)` 结束。
- 输出累积答案个数。

## 代码

```python
class Solution:
    def numSubarrayProductLessThanK(self, nums: List[int], k: int) -> int:
        if k <= 1:
            return 0

        size = len(nums)
        left, right = 0, 0
        count = 0
        product = 1
        while right < size:
            product *= nums[right]
            right += 1
            while product >= k:
                product /= nums[left]
                left += 1
            count += (right - left)
        return count
```

# [剑指 Offer II 010. 和为 k 的子数组](https://leetcode.cn/problems/QTMn0o/)

- 标签：数组、哈希表、前缀和
- 难度：中等

## 题目链接

- [剑指 Offer II 010. 和为 k 的子数组 - 力扣](https://leetcode.cn/problems/QTMn0o/)

## 题目大意

给定一个整数数组 `nums` 和一个整数 `k`。

要求：找到该数组中和为 `k` 的连续子数组的个数。

## 解题思路

看到题目的第一想法是通过滑动窗口求解。但是做下来发现有些数据样例无法通过。发现这道题目中的整数不能保证都为正数，则无法通过滑动窗口进行求解。

先考虑暴力做法，外层两重循环，遍历所有连续子数组，然后最内层再计算一下子数组的和。部分代码如下：

```python
for i in range(len(nums)):
    for j in range(i + 1):
        sum = countSum(i, j)
```

这样下来时间复杂度就是 $O(n^3)$ 了。下一步是想办法降低时间复杂度。

先用一重循环遍历数组，计算出数组 `nums` 中前 i 个元素的和（前缀和），保存到一维数组 `pre_sum` 中，那么对于任意 `[j..i]` 的子数组 的和为 `pre_sum[i] - pre_sum[j - 1]`。这样计算子数组和的时间复杂度降为了 $O(1)$。总体时间复杂度为 $O(n^3)$。

但是还是超时了。。

由于我们只关心和为 `k` 出现的次数，不关心具体的解，可以使用哈希表来加速运算。

`pre_sum[i]` 的定义是前 `i` 个元素和，则 `pre_sum[i]` 可以由 `pre_sum[i - 1]` 递推而来，即：`pre_sum[i] = pre_sum[i - 1] + sum[i]`。 `[j..i]` 子数组和为 `k` 可以转换为：`pre_sum[i] - pre_sum[j - 1] == k`。

综合一下，可得：`pre_sum[j - 1] == pre_sum[i] - k `。

所以，当我们考虑以 `i` 结尾和为 `k` 的连续子数组个数时，只需要统计有多少个前缀和为 `pre_sum[i] - k` （即 `pre_sum[j - 1]`）的个数即可。具体做法如下：

- 使用 `pre_sum` 变量记录前缀和（代表 `pre_sum[i]`）。
- 使用哈希表 `pre_dic` 记录 `pre_sum[i]` 出现的次数。键值对为 `pre_sum[i] : pre_sum_count`。
- 从左到右遍历数组，计算当前前缀和 `pre_sum`。
- 如果 `pre_sum - k` 在哈希表中，则答案个数累加上 `pre_dic[pre_sum - k]`。
- 如果 `pre_sum` 在哈希表中，则前缀和个数累加 1，即 `pre_dic[pre_sum] += 1`。
- 最后输出答案个数。

## 代码

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

# [剑指 Offer II 011. 0 和 1 个数相同的子数组](https://leetcode.cn/problems/A1NYOS/)

- 标签：数组、哈希表、前缀和
- 难度：中等

## 题目链接

- [剑指 Offer II 011. 0 和 1 个数相同的子数组 - 力扣](https://leetcode.cn/problems/A1NYOS/)

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

# [剑指 Offer II 012. 左右两边子数组的和相等](https://leetcode.cn/problems/tvdfij/)

- 标签：数组、前缀和
- 难度：简单

## 题目链接

- [剑指 Offer II 012. 左右两边子数组的和相等 - 力扣](https://leetcode.cn/problems/tvdfij/)

## 题目大意

给定一个数组 `nums`。

要求：找到「左侧元素和」与「右侧元素和相等」的位置，若找不到，则返回 `-1`。

## 解题思路

两次遍历，第一次遍历先求出数组全部元素和。第二次遍历找到左侧元素和恰好为全部元素和一半的位置。

## 代码

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

# [剑指 Offer II 013. 二维子矩阵的和](https://leetcode.cn/problems/O4NDxx/)

- 标签：设计、数组、矩阵、前缀和
- 难度：中等

## 题目链接

- [剑指 Offer II 013. 二维子矩阵的和 - 力扣](https://leetcode.cn/problems/O4NDxx/)

## 题目大意

给定一个二维矩阵 `matrix`。

要求：满足以下多个请求：

- ` def sumRegion(self, row1: int, col1: int, row2: int, col2: int) -> int:`计算以 `(row1, col1)` 为左上角、`(row2, col2)` 为右下角的子矩阵中各个元素的和。
- `def __init__(self, matrix: List[List[int]]):` 对二维矩阵 `matrix` 进行初始化操作。

## 解题思路

在进行初始化的时候做预处理，这样在多次查询时可以减少重复计算，也可以减少时间复杂度。

在进行初始化的时候，使用一个二维数组 `pre_sum` 记录下以 `(0, 0)` 为左上角，以当前 `(row, col)` 为右下角的子数组各个元素和，即 `pre_sum[row + 1][col + 1]`。

则在查询时，以 `(row1, col1)` 为左上角、`(row2, col2)` 为右下角的子矩阵中各个元素的和就等于以 `(0, 0)` 到 `(row2, col2)` 的大子矩阵减去左边 `(0, 0)` 到 `(row2, col1 - 1)`的子矩阵，再减去上边 `(0, 0)` 到 `(row1 - 1, col2)` 的子矩阵，再加上左上角 `(0, 0)` 到 `(row1 - 1, col1 - 1)` 的子矩阵（因为之前重复减了）。即 `pre_sum[row2 + 1][col2 + 1] - self.pre_sum[row2 + 1][col1] - self.pre_sum[row1][col2 + 1] + self.pre_sum[row1][col1]`。

## 代码

```python
class NumMatrix:

    def __init__(self, matrix: List[List[int]]):
        rows = len(matrix)
        cols = len(matrix[0])
        self.pre_sum = [[0 for _ in range(cols + 1)] for _ in range(rows + 1)]
        for row in range(rows):
            for col in range(cols):
                self.pre_sum[row + 1][col + 1] = self.pre_sum[row + 1][col] + self.pre_sum[row][col + 1] - self.pre_sum[row][col] + matrix[row][col]


    def sumRegion(self, row1: int, col1: int, row2: int, col2: int) -> int:
        return self.pre_sum[row2 + 1][col2 + 1] - self.pre_sum[row2 + 1][col1] - self.pre_sum[row1][col2 + 1] + self.pre_sum[row1][col1]
```

# [剑指 Offer II 016. 不含重复字符的最长子字符串](https://leetcode.cn/problems/wtcaE1/)

- 标签：哈希表、字符串、滑动窗口
- 难度：中等

## 题目链接

- [剑指 Offer II 016. 不含重复字符的最长子字符串 - 力扣](https://leetcode.cn/problems/wtcaE1/)

## 题目大意

给定一个字符串 `s`。

要求：找出其中不含有重复字符的 最长子串 的长度。

## 解题思路

利用集合来存储不重复的字符。用两个指针分别指向最长子串的左右节点。遍历字符串，右指针不断右移，利用集合来判断有没有重复的字符，如果没有，就持续向右扩大右边界。如果出现重复字符，就缩小左侧边界。每次移动终止，都要计算一下当前不含重复字符的子串长度，并判断一下是否需要更新最大长度。

## 代码

```python
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        if not s:
            return 0

        letterSet = set()
        right = 0
        ans = 0
        for i in range(len(s)):
            if i != 0:
                letterSet.remove(s[i - 1])
            while right < len(s) and s[right] not in letterSet:
                letterSet.add(s[right])
                right += 1
            ans = max(ans, right - i)
        return ans
```

# [剑指 Offer II 017. 含有所有字符的最短字符串](https://leetcode.cn/problems/M1oyTv/)

- 标签：哈希表、字符串、滑动窗口
- 难度：困难

## 题目链接

- [剑指 Offer II 017. 含有所有字符的最短字符串 - 力扣](https://leetcode.cn/problems/M1oyTv/)

## 题目大意

给定一个字符串 `s`、一个字符串 `t`。

要求：返回 `s` 中涵盖 `t` 所有字符的最小子串。如果 `s` 中不存在涵盖 `t` 所有字符的子串，则返回空字符串 `""`。如果存在多个符合条件的子字符串，返回任意一个。

## 解题思路

使用滑动窗口求解。

`left`、`right` 表示窗口的边界，一开始都位于下标 `0` 处。`need` 用于记录短字符串需要的字符数。`window` 记录当前窗口内的字符数。

将 `right` 右移，直到出现了 `t` 中全部字符，开始右移 `left`，减少滑动窗口的大小，并记录下最小覆盖子串的长度和起始位置。最后输出结果。

## 代码

```python
class Solution:
    def minWindow(self, s: str, t: str) -> str:
        need = collections.defaultdict(int)
        window = collections.defaultdict(int)
        for ch in t:
            need[ch] += 1

        left, right = 0, 0
        valid = 0
        start = 0
        size = len(s) + 1

        while right < len(s):
            insert_ch = s[right]
            right += 1

            if insert_ch in need:
                window[insert_ch] += 1
                if window[insert_ch] == need[insert_ch]:
                    valid += 1

            while valid == len(need):
                if right - left < size:
                    start = left
                    size = right - left
                remove_ch = s[left]
                left += 1
                if remove_ch in need:
                    if window[remove_ch] == need[remove_ch]:
                        valid -= 1
                    window[remove_ch] -= 1
        if size == len(s) + 1:
            return ''
        return s[start:start + size]
```

# [剑指 Offer II 018. 有效的回文](https://leetcode.cn/problems/XltzEq/)

- 标签：双指针、字符串
- 难度：简单

## 题目链接

- [剑指 Offer II 018. 有效的回文 - 力扣](https://leetcode.cn/problems/XltzEq/)

## 题目大意

给定一个字符串 `s`。

要求：判断是否为回文串。（只考虑字符串中的字母和数字字符，并且忽略字母的大小写）

## 解题思路

左右两个指针 `start` 和 `end`，左指针 `start` 指向字符串头部，右指针 `end` 指向字符串尾部。先过滤掉除字母和数字字符以外的字符，在判断 `s[start]` 和 `s[end]` 是否相等。不相等返回 `False`，相等则继续过滤和判断。

## 代码

```python
class Solution:
    def isPalindrome(self, s: str) -> bool:
        n = len(s)
        start = 0
        end = n - 1
        while start < end:
            if not s[start].isalnum():
                start += 1
                continue
            if not s[end].isalnum():
                end -= 1
                continue
            if s[start].lower() == s[end].lower():
                start += 1
                end -= 1
            else:
                return False
        return True
```

# [剑指 Offer II 019. 最多删除一个字符得到回文](https://leetcode.cn/problems/RQku0D/)

- 标签：贪心、双指针、字符串
- 难度：简单

## 题目链接

- [剑指 Offer II 019. 最多删除一个字符得到回文 - 力扣](https://leetcode.cn/problems/RQku0D/)

## 题目大意

给定一个非空字符串 `s`。

要求：判断如果最多从字符串中删除一个字符能否得到一个回文字符串。

## 解题思路

双指针 + 贪心算法。

- 用两个指针 `left`、`right` 分别指向字符串的开始和结束位置。

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

# [剑指 Offer II 020. 回文子字符串的个数](https://leetcode.cn/problems/a7VOhD/)

- 标签：字符串、动态规划
- 难度：中等

## 题目链接

- [剑指 Offer II 020. 回文子字符串的个数 - 力扣](https://leetcode.cn/problems/a7VOhD/)

## 题目大意

给定一个字符串 `s`。

要求：计算 `s` 中有多少个回文子串。

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

