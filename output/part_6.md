# [1176. 健身计划评估](https://leetcode.cn/problems/diet-plan-performance/)

- 标签：数组、滑动窗口
- 难度：简单

## 题目链接

- [1176. 健身计划评估 - 力扣](https://leetcode.cn/problems/diet-plan-performance/)

## 题目大意

**描述**：好友给自己制定了一份健身计划。想请你帮他评估一下这份计划是否合理。

给定一个数组 $calories$，其中 $calories[i]$ 代表好友第 $i$ 天需要消耗的卡路里总量。再给定 $lower$ 代表较低消耗的卡路里，$upper$ 代表较高消耗的卡路里。再给定一个整数 $k$，代表连续 $k$ 天。

- 如果你的好友在这一天以及之后连续 $k$ 天内消耗的总卡路里 $T$ 小于 $lower$，则这一天的计划相对糟糕，并失去 $1$ 分。
- 如果你的好友在这一天以及之后连续 $k$ 天内消耗的总卡路里 $T$ 高于 $upper$，则这一天的计划相对优秀，并得到 $1$ 分。
- 如果你的好友在这一天以及之后连续 $k$ 天内消耗的总卡路里 $T$ 大于等于 $lower$，并且小于等于 $upper$，则这份计划普普通通，分值不做变动。

**要求**：输出最后评估的得分情况。

**说明**：

- $1 \le k \le calories.length \le 10^5$。
- $0 \le calories[i] \le 20000$。
- $0 \le lower \le upper$。 

**示例**：

- 示例 1：

```python
输入：calories = [1,2,3,4,5], k = 1, lower = 3, upper = 3
输出：0
解释：calories[0], calories[1] < lower 而 calories[3], calories[4] > upper, 总分 = 0.
```

- 示例 2：

```python
输入：calories = [3,2], k = 2, lower = 0, upper = 1
输出：1
解释：calories[0] + calories[1] > upper, 总分 = 1.
```

## 解题思路

### 思路 1：滑动窗口

固定长度为 $k$ 的滑动窗口题目。具体做法如下：

1. $score$ 用来维护得分情况，初始值为 $0$。$window\underline{\hspace{0.5em}}sum$ 用来维护窗口中卡路里总量。
2. $left$ 、$right$ 都指向数组的第一个元素，即：`left = 0`，`right = 0`。
3. 向右移动 $right$，先将 $k$ 个元素填入窗口中。
4. 当窗口元素个数为 $k$ 时，即：$right - left + 1 \ge k$ 时，计算窗口内的卡路里总量，并判断和 $upper$、$lower$ 的关系。同时维护得分情况。
5. 然后向右移动 $left$，从而缩小窗口长度，即 `left += 1`，使得窗口大小始终保持为 $k$。
6. 重复 $4 \sim 5$ 步，直到 $right$ 到达数组末尾。

最后输出得分情况 $score$。

### 思路 1：代码

```python
class Solution:
    def dietPlanPerformance(self, calories: List[int], k: int, lower: int, upper: int) -> int:
        left, right = 0, 0
        window_sum = 0
        score = 0
        while right < len(calories):
            window_sum += calories[right]

            if right - left + 1 >= k:
                if window_sum < lower:
                    score -= 1
                elif window_sum > upper:
                    score += 1
                window_sum -= calories[left]
                left += 1

            right += 1
        return score
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(n)$，其中 $n$ 为数组 $calories$ 的长度。
- **空间复杂度**：$O(1)$。

# [1184. 公交站间的距离](https://leetcode.cn/problems/distance-between-bus-stops/)

- 标签：数组
- 难度：简单

## 题目链接

- [1184. 公交站间的距离 - 力扣](https://leetcode.cn/problems/distance-between-bus-stops/)

## 题目大意

**描述**：环形公交路线上有 $n$ 个站，序号为 $0 \sim n - 1$。给定一个数组 $distance$ 表示每一对相邻公交站之间的距离，其中 $distance[i]$ 表示编号为 $i$ 的车站与编号为 $(i + 1) \mod n$ 的车站之间的距离。再给定乘客的出发点编号 $start$ 和目的地编号 $destination$。

**要求**：返回乘客从出发点 $start$ 到目的地 $destination$ 之间的最短距离。

**说明**：

- $1 \le n \le 10^4$。
- $distance.length == n$。
- $0 \le start, destination < n$。
- $0 \le distance[i] \le 10^4$。

**示例**：

- 示例 1：

![](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2019/09/08/untitled-diagram-1.jpg)

```python
输入：distance = [1,2,3,4], start = 0, destination = 1
输出：1
解释：公交站 0 和 1 之间的距离是 1 或 9，最小值是 1。
```

- 示例 2：

![](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2019/09/08/untitled-diagram-1-1.jpg)

```python
输入：distance = [1,2,3,4], start = 0, destination = 2
输出：3
解释：公交站 0 和 2 之间的距离是 3 或 7，最小值是 3。
```

## 解题思路

### 思路 1：简单模拟

1. 因为 $start$ 和 $destination$ 的先后顺序不影响结果，为了方便计算，我们先令 $start \le destination$。
2. 遍历数组 $distance$，计算出 $[start, destination]$ 之间的距离和 $dist$。
3. 计算出环形路线中 $[destination, start]$ 之间的距离和为 $sum(distance) - dist$。
4. 比较 $2 \sim 3$ 中两个距离的大小，将距离最小值作为答案返回。

### 思路 1：代码

```python
class Solution:
    def distanceBetweenBusStops(self, distance: List[int], start: int, destination: int) -> int:
        start, destination = min(start, destination), max(start, destination)
        dist = 0
        for i in range(len(distance)):
            if start <= i < destination:
                dist += distance[i]
        
        return min(dist, sum(distance) - dist)
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(n)$。
- **空间复杂度**：$O(1)$。
# [1202. 交换字符串中的元素](https://leetcode.cn/problems/smallest-string-with-swaps/)

- 标签：深度优先搜索、广度优先搜索、并查集、哈希表、字符串
- 难度：中等

## 题目链接

- [1202. 交换字符串中的元素 - 力扣](https://leetcode.cn/problems/smallest-string-with-swaps/)

## 题目大意

**描述**：给定一个字符串 `s`，再给定一个数组 `pairs`，其中 `pairs[i] = [a, b]` 表示字符串的第 `a` 个字符可以跟第 `b` 个字符交换。只要满足 `pairs` 中的交换关系，可以任意多次交换字符串中的字符。

**要求**：返回 `s` 经过若干次交换之后，可以变成的字典序最小的字符串。

**说明**：

- $1 \le s.length \le 10^5$。
- $0 \le pairs.length \le 10^5$。
- $0 \le pairs[i][0], pairs[i][1] < s.length$。
- `s` 中只含有小写英文字母。

**示例**：

- 示例 1：

```python
输入：s = "dcab", pairs = [[0,3],[1,2]]
输出："bacd"
解释： 
交换 s[0] 和 s[3], s = "bcad"
交换 s[1] 和 s[2], s = "bacd"
```

- 示例 2：

```python
输入：s = "dcab", pairs = [[0,3],[1,2],[0,2]]
输出："abcd"
解释：
交换 s[0] 和 s[3], s = "bcad"
交换 s[0] 和 s[2], s = "acbd"
交换 s[1] 和 s[2], s = "abcd"
```

## 解题思路

### 思路 1：并查集

如果第 `a` 个字符可以跟第 `b` 个字符交换，第 `b` 个字符可以跟第 `c` 个字符交换，那么第 `a` 个字符、第 `b` 个字符、第 `c` 个字符之间就可以相互交换。我们可以把可以相互交换的「位置」都放入一个集合中。然后对每个集合中的字符进行排序。然后将其放置回在字符串中原有位置即可。

### 思路 1：代码

```python
import collections

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
    def smallestStringWithSwaps(self, s: str, pairs: List[List[int]]) -> str:
        size = len(s)
        union_find = UnionFind(size)
        for pair in pairs:
            union_find.union(pair[0], pair[1])
        mp = collections.defaultdict(list)

        for i, ch in enumerate(s):
            mp[union_find.find(i)].append(ch)

        for vec in mp.values():
            vec.sort(reverse=True)

        ans = []
        for i in range(size):
            x = union_find.find(i)
            ans.append(mp[x][-1])
            mp[x].pop()

        return "".join(ans)
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(n \times \log_2 n + m * \alpha(n))$。其中 $n$ 是字符串的长度，$m$ 为 $pairs$ 的索引对数量，$\alpha$ 是反 `Ackerman` 函数。
- **空间复杂度**：$O(n)$。# [1208. 尽可能使字符串相等](https://leetcode.cn/problems/get-equal-substrings-within-budget/)

- 标签：字符串、二分查找、前缀和、滑动窗口
- 难度：中等

## 题目链接

- [1208. 尽可能使字符串相等 - 力扣](https://leetcode.cn/problems/get-equal-substrings-within-budget/)

## 题目大意

**描述**：给定两个长度相同的字符串，$s$ 和 $t$。将 $s$ 中的第 $i$ 个字符变到 $t$ 中的第 $i$ 个字符需要 $| s[i] - t[i] |$ 的开销（开销可能为 $0$），也就是两个字符的 ASCII 码值的差的绝对值。用于变更字符串的最大预算是 $maxCost$。在转化字符串时，总开销应当小于等于该预算，这也意味着字符串的转化可能是不完全的。

**要求**：如果你可以将 $s$ 的子字符串转化为它在 $t$ 中对应的子字符串，则返回可以转化的最大长度。如果 $s$ 中没有子字符串可以转化成 $t$ 中对应的子字符串，则返回 $0$。

**说明**：

- $1 \le s.length, t.length \le 10^5$。
- $0 \le maxCost \le 10^6$。
- $s$ 和 $t$ 都只含小写英文字母。

**示例**：

- 示例 1：

```python
输入：s = "abcd", t = "bcdf", maxCost = 3
输出：3
解释：s 中的 "abc" 可以变为 "bcd"。开销为 3，所以最大长度为 3。
```

- 示例 2：

```python
输入：s = "abcd", t = "cdef", maxCost = 3
输出：1
解释：s 中的任一字符要想变成 t 中对应的字符，其开销都是 2。因此，最大长度为 1。
```

## 解题思路

### 思路 1：滑动窗口

维护一个滑动窗口 $window\underline{\hspace{0.5em}}sum$ 用于记录窗口内的开销总和，保证窗口内的开销总和小于等于 $maxCost$。使用 $ans$ 记录可以转化的最大长度。具体做法如下：

使用两个指针 $left$、$right$。分别指向滑动窗口的左右边界，保证窗口内所有元素转化开销总和小于等于 $maxCost$。

- 先统计出 $s$ 中第 $i$ 个字符变为 $t$ 的第 $i$ 个字符的开销，用数组 $costs$ 保存。
- 一开始，$left$、$right$ 都指向 $0$。
- 将最右侧字符的转变开销填入窗口中，向右移动 $right$。
- 直到窗口内开销总和 $window\underline{\hspace{0.5em}}sum$ 大于 $maxCost$。则不断右移 $left$，缩小窗口长度。直到 $window\underline{\hspace{0.5em}}sum \le maxCost$ 时，更新可以转换的最大长度 $ans$。
- 向右移动 $right$，直到 $right \ge len(s)$ 为止。
- 输出答案 $ans$。

### 思路 1：代码

```python
class Solution:
    def equalSubstring(self, s: str, t: str, maxCost: int) -> int:
        size = len(s)
        costs = [0 for _ in range(size)]
        for i in range(size):
            costs[i] = abs(ord(s[i]) - ord(t[i]))

        left, right = 0, 0
        ans = 0
        window_sum = 0
        while right < size:
            window_sum += costs[right]
            while window_sum > maxCost:
                window_sum -= costs[left]
                left += 1
            ans = max(ans, right - left + 1)
            right += 1

        return ans
```

### 思路 1：复杂度分析

- **时间复杂度**：
- **空间复杂度**：

# [1217. 玩筹码](https://leetcode.cn/problems/minimum-cost-to-move-chips-to-the-same-position/)

- 标签：贪心、数组、数学
- 难度：简单

## 题目链接

- [1217. 玩筹码 - 力扣](https://leetcode.cn/problems/minimum-cost-to-move-chips-to-the-same-position/)

## 题目大意

**描述**：给定一个数组 $position$ 代表 $n$ 个筹码的位置，其中 $position[i]$ 代表第 $i$ 个筹码的位置。现在需要把所有筹码移到同一个位置。在一步中，我们可以将第 $i$ 个芯片的位置从 $position[i]$ 改变为:

- $position[i] + 2$ 或 $position[i] - 2$，此时 $cost = 0$；
- $position[i] + 1$ 或 $position[i] - 1$，此时 $cost = 1$。

即移动偶数位长度的代价为 $0$，移动奇数位长度的代价为 $1$。

**要求**：返回将所有筹码移动到同一位置上所需要的 最小代价 。

**说明**：

- $1 \le chips.length \le 100$。
- $1 \le chips[i] \le 10^9$。

**示例**：

- 示例 1：

```python
输入：position = [2,2,2,3,3]
输出：2
解释：我们可以把位置3的两个芯片移到位置 2。每一步的成本为 1。总成本 = 2。
```

## 解题思路

### 思路 1：贪心算法

题目中移动偶数位长度是不需要代价的，所以奇数位移动到奇数位不需要代价，偶数位移动到偶数位也不需要代价。

则我们可以想将所有偶数位都移动到下标为 $0$ 的位置，奇数位都移动到下标为 $1$ 的位置。

这样，所有的奇数位、偶数位上的人都到相同或相邻位置了。

我们只需要统计一下奇数位和偶数位的数字个数。将少的数移动到多的数上边就是最小代价。

则这道题就可以通过以下步骤求解：

- 遍历数组，统计数组中奇数个数和偶数个数。
- 返回奇数个数和偶数个数中较小的数即为答案。

### 思路 1：贪心算法代码

```python
class Solution:
    def minCostToMoveChips(self, position: List[int]) -> int:
        odd, even = 0, 0
        for p in position:
            if p & 1:
                odd += 1
            else:
                even += 1
        return min(odd, even)
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(n)$，其中 $n$ 为数组 $poition$ 的长度。
- **空间复杂度**：$O(1)$。
# [1220. 统计元音字母序列的数目](https://leetcode.cn/problems/count-vowels-permutation/)

- 标签：动态规划
- 难度：困难

## 题目链接

- [1220. 统计元音字母序列的数目 - 力扣](https://leetcode.cn/problems/count-vowels-permutation/)

## 题目大意

**描述**：给定一个整数 `n`，我们可以按照以下规则生成长度为 `n` 的字符串：

- 字符串中的每个字符都应当是小写元音字母（`'a'`、`'e'`、`'i'`、`'o'`、`'u'`）。
- 每个元音 `'a'` 后面都只能跟着 `'e'`。
- 每个元音 `'e'` 后面只能跟着 `'a'` 或者是 `'i'`。
- 每个元音 `'i'` 后面不能再跟着另一个 `'i'`。
- 每个元音 `'o'` 后面只能跟着 `'i'` 或者是 `'u'`。
- 每个元音 `'u'` 后面只能跟着 `'a'`。

**要求**：统计一下我们可以按上述规则形成多少个长度为 `n` 的字符串。由于答案可能会很大，所以请返回模 $10^9 + 7$ 之后的结果。

**说明**：

- $1 \le n \le 2 * 10^4$。

**示例**：

- 示例 1：

```python
输入：n = 2
输出：10
解释：所有可能的字符串分别是："ae", "ea", "ei", "ia", "ie", "io", "iu", "oi", "ou" 和 "ua"。
```

## 解题思路

### 思路 1：动态规划

根据题目给定的字符串规则，我们可以将其整理一下：

- 元音字母 `'a'` 前面只能跟着 `'e'`、`'i'`、`'u'`。
- 元音字母 `'e'` 前面只能跟着 `'a'`、`'i'`。
- 元音字母 `'i'` 前面只能跟着 `'e'`、`'o'`。
- 元音字母 `'o'` 前面只能跟着 `'i'`。
- 元音字母 `'u'` 前面只能跟着 `'o'`、`'i'`。

现在我们可以按照字符串的长度以及字符结尾进行阶段划分，并按照上述规则推导状态转移方程。

###### 1. 划分阶段

按照字符串的结尾位置和结尾位置上的字符进行阶段划分。

###### 2. 定义状态

定义状态 `dp[i][j]` 表示为：长度为 `i` 并且以字符 `j` 结尾的字符串数量。这里 $j = 0, 1, 2, 3, 4$ 分别代表元音字母 `'a'`、`'e'`、`'i'`、`'o'`、`'u'`。

###### 3. 状态转移方程

通过上面的字符规则，可以得到状态转移方程为：


$\begin{cases} dp[i][0] = dp[i - 1][1] + dp[i - 1][2] + dp[i - 1][4] \cr dp[i][1] = dp[i - 1][0] + dp[i - 1][2] \cr dp[i][2] = dp[i - 1][1] + dp[i - 1][3] \cr dp[i][3] = dp[i - 1][2] \cr dp[i][4] = dp[i - 1][2] + dp[i - 1][3] \end{cases}$

###### 4. 初始条件

- 长度为 `1` 并且以字符 `j` 结尾的字符串数量为 `1`，即 `dp[1][j] = 1`。

###### 5. 最终结果

根据我们之前定义的状态，`dp[i]` 表示为：长度为 `i` 并且以字符 `j` 结尾的字符串数量。则将 `dp[n]` 行所有列相加，就是长度为 `n` 的字符串数量。

### 思路 1：动态规划代码

```python
class Solution:
    def countVowelPermutation(self, n: int) -> int:
        mod = 10 ** 9 + 7
        dp = [[0 for _ in range(5)] for _ in range(n + 1)]

        for j in range(5):
            dp[1][j] = 1

        for i in range(2, n + 1):
            dp[i][0] = (dp[i - 1][1] + dp[i - 1][2] + dp[i - 1][4]) % mod
            dp[i][1] = (dp[i - 1][0] + dp[i - 1][2]) % mod
            dp[i][2] = (dp[i - 1][1] + dp[i - 1][3]) % mod
            dp[i][3] = dp[i - 1][2] % mod
            dp[i][4] = (dp[i - 1][2] + dp[i - 1][3]) % mod

        ans = 0
        for j in range(5):
            ans += dp[n][j] % mod
        ans %= mod
        
        return ans
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(n)$。
- **空间复杂度**：$O(n)$。
# [1227. 飞机座位分配概率](https://leetcode.cn/problems/airplane-seat-assignment-probability/)

- 标签：脑筋急转弯、数学、动态规划、概率与统计
- 难度：中等

## 题目链接

- [1227. 飞机座位分配概率 - 力扣](https://leetcode.cn/problems/airplane-seat-assignment-probability/)

## 题目大意

**描述**：给定一个整数 $n$，代表 $n$ 位乘客即将登飞机。飞机上刚好有 $n$ 个座位。第一位乘客的票丢了，他随便选择了一个座位坐下。则剩下的乘客将会：

- 如果自己的座位还空着，就坐到自己的座位上。
- 如果自己的座位被占用了，就随机选择其他座位。

**要求**：计算出第 $n$ 位乘客坐在自己座位上的概率是多少。

**说明**：

- $1 \le n \le 10^5$。

**示例**：

- 示例 1：

```python
输入：n = 1
输出：1.00000
解释：第一个人只会坐在自己的位置上。
```

- 示例 2：

```python
输入: n = 2
输出: 0.50000
解释：在第一个人选好座位坐下后，第二个人坐在自己的座位上的概率是 0.5。
```

## 解题思路

### 思路 1：数学

我们按照乘客的登机顺序为乘客编下号：$1 \sim n$，我们用 $f(n)$ 来表示第 $n$ 位乘客登机时，坐在自己座位上的概率。先从简单的情况开始考虑：

当 $n = 1$ 时：

- 第 $1$ 位乘客只能坐在第 $1$ 个座位上，$f(1) = 1$。

当 $n = 2$ 时：

- 第 $1$ 位乘客有 $\frac{1}{2}$ 的概率选中自己的位置，第 $2$ 位乘客一定能坐到自己的位置上，则第 $2$ 位乘客坐在自己座位上的概率为 $\frac{1}{2} * 1.0$。
- 第 $1$ 位乘客有 $\frac{1}{2}$ 的概率坐在第 $2$ 位乘客的位置上，第 $2$ 位乘客只能坐到第 $1$ 位乘客的位置上，那么第 $2$ 位乘客坐在自己座位上的概率为 $\frac{1}{2} * 0.0$。
- 综上，$f(2) =  \frac{1}{2} * 1.0 + \frac{1}{2} * 0.0 = 0.5$。

当 $n \ge 3$ 时：

- 先来考虑第 $1$ 位乘客登机情况：

  - 第 $1$ 位乘客有 $\frac{1}{n}$ 的概率选择坐在自己位置上，这样第 $1$ 位到第 $n - 1$ 位乘客的座位都不会被占，第 n 位乘客一定能坐到自己位置上。那么第 n 位乘客坐在自己座位上的概率为 $\frac{1}{n} * 1.0$。

  - 第 $1$ 位乘客有 $\frac{1}{n}$ 的概率选择坐在第 $n$ 位乘客的位置上，这样第 $2$ 位到第 $n - 1$ 位乘客的座位都不会被占，第 $n$ 位乘客只能坐到第 $1$ 位乘客的位置上，那么第 $n$ 位乘客坐在自己座位上的概率为 $\frac{1}{n} * 0.0$。

  - 第 $1$ 位乘客有 $\frac{n-2}{n}$ 的概率坐在第 $i$ 号座位上，$2 \le i \le n - 1$，每个座位被选中概率为 $\frac{1}{n}$。这样第 $2$ 位到第 $i - 1$ 位乘客的座位都不会被占。此时第 $i$ 位乘客，会在剩下的 $n - (i - 1)$ 个座位中进行选择：

    - 坐在第 $1$ 位乘客的位置上，这样后面的乘客座位都不会被占，第 $n$ 位乘客一定能坐到自己位置上。

    - 坐在第 $n$ 个乘客的位置上，这样第 $n$ 个乘客肯定无法坐到自己的位置上。

    - 在第 $[i + 1, n - 1]$ 之间找个位置坐。

- 再来考虑第 $i$ 位乘客登机情况：
  - 第 $i$ 为乘客所面临的情况跟第 $1$ 位乘客所面临的情况类似，只不过问题的规模数从 $n$ 减小到了  $n - (i - 1)$。

那么综合上面情况，可以得到 $f(n),(n \ge 3)$ 的递推式：

$\begin{aligned} f(n) & =  \frac{1}{n} * 1.0 + \frac{1}{n} * 0.0 + \frac{1}{n} * \sum_{i = 2}^{n-1} f(n - i + 1) \cr & = \frac{1}{n} (1.0 + \sum_{i = 2}^{n-1} f(n - i + 1)) \end{aligned}$

接下来我们从等式中寻找规律，消去 $\sum_{i = 2}^{n-1} f(n - i + 1)$ 部分。

将 $n$ 换为 $n - 1$，得：

$\begin{aligned} f(n - 1) & =  \frac{1}{n - 1} * 1.0 + \frac{1}{n - 1} * 0.0 + \frac{1}{n - 1} * \sum_{i = 2}^{n-2} f(n - i) \cr & = \frac{1}{n - 1} (1.0 + \sum_{i = 2}^{n-2} f(n - i)) \end{aligned} $

将 $f(n) * n$ 与 $f(n - 1) * (n - 1)$ 进行比较：

$\begin{aligned} f(n) * n & = 1.0 + \sum_{i = 2}^{n-1} f(n - i + 1) & (1) \cr f(n - 1) * (n - 1) & = 1.0 + \sum_{i = 2}^{n-2} f(n - i) & (2) \end{aligned}$

将上述 (1)、(2) 式相减得：

$\begin{aligned} & f(n) * n - f(n - 1) * (n - 1) & \cr = & \sum_{i = 2}^{n-1} f(n - i + 1) - \sum_{i = 2}^{n-2}  f(n - i) \cr = & f(n-1) \end{aligned}$

整理后得：$f(n) = f(n - 1)$。

已知 $f(1) = 1$，$f(2) = 0.5$，因此当 $n \ge 3$ 时，$f(n) = 0.5$。

所以可以得出结论：

$f(n) = \begin{cases} 1.0 & n = 1 \cr 0.5 & n \ge 2  \end{cases}$

### 思路 1：代码

```python
class Solution:
    def nthPersonGetsNthSeat(self, n: int) -> float:
        if n == 1:
            return 1.0
        else:
            return 0.5
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(1)$。
- **空间复杂度**：$O(1)$。

## 参考资料

- [飞机座位分配概率 - 力扣（LeetCode）](https://leetcode.cn/problems/airplane-seat-assignment-probability/solution/fei-ji-zuo-wei-fen-pei-gai-lu-by-leetcod-gyw4/)

# [1229. 安排会议日程](https://leetcode.cn/problems/meeting-scheduler/)

- 标签：数组、双指针、排序
- 难度：中等

## 题目链接

- [1229. 安排会议日程 - 力扣](https://leetcode.cn/problems/meeting-scheduler/)

## 题目大意

**描述**：给定两位客户的空闲时间表：$slots1$ 和 $slots2$，再给定会议的预计持续时间 $duration$。

其中 $slots1[i] = [start_i, end_i]$ 表示空闲时间第从 $start_i$ 开始，到 $end_i$ 结束。$slots2$ 也是如此。

**要求**：为他们安排合适的会议时间，如果有合适的会议时间，则返回该时间的起止时刻。如果没有满足要求的会议时间，就请返回一个 空数组。

**说明**：

- **会议时间**：两位客户都有空参加，并且持续时间能够满足预计时间 $duration$ 的最早的时间间隔。
- 题目保证数据有效。同一个人的空闲时间不会出现交叠的情况，也就是说，对于同一个人的两个空闲时间 $[start1, end1]$ 和 $[start2, end2]$，要么 $start1 > end2$，要么 $start2 > end1$。
- $1 \le slots1.length, slots2.length \le 10^4$。
- $slots1[i].length, slots2[i].length == 2$。
- $slots1[i][0] < slots1[i][1]$。
- $slots2[i][0] < slots2[i][1]$。
- $0 \le slots1[i][j], slots2[i][j] \le 10^9$。
- $1 \le duration \le 10^6$。

**示例**：

- 示例 1：

```python
输入：slots1 = [[10,50],[60,120],[140,210]], slots2 = [[0,15],[60,70]], duration = 8
输出：[60,68]
```

- 示例 2：

```python
输入：slots1 = [[10,50],[60,120],[140,210]], slots2 = [[0,15],[60,70]], duration = 12
输出：[]
```

## 解题思路

### 思路 1：分离双指针

题目保证了同一个人的空闲时间不会出现交叠。那么可以先直接对两个客户的空间时间表按照开始时间从小到大排序。然后使用分离双指针来遍历两个数组，求出重合部分，并判断重合区间是否大于等于 $duration$。具体做法如下：

1. 先对两个数组排序。
2. 然后使用两个指针 $left\underline{\hspace{0.5em}}1$、$left\underline{\hspace{0.5em}}2$。$left\underline{\hspace{0.5em}}1$ 指向第一个数组开始位置，$left\underline{\hspace{0.5em}}2$ 指向第二个数组开始位置。
3. 遍历两个数组。计算当前两个空闲时间区间的重叠范围。
   1. 如果重叠范围大于等于 $duration$，直接返回当前重叠范围开始时间和会议结束时间，即 $[start, start + duration]$，$start$ 为重叠范围开始时间。
   2. 如果第一个客户的空闲结束时间小于第二个客户的空闲结束时间，则令 $left\underline{\hspace{0.5em}}1$ 右移，即 `left_1 += 1`，继续比较重叠范围。
   3. 如果第一个客户的空闲结束时间大于等于第二个客户的空闲结束时间，则令 $left\underline{\hspace{0.5em}}2$ 右移，即 `left_2 += 1`，继续比较重叠范围。
4. 直到 $left\underline{\hspace{0.5em}}1 == len(slots1)$ 或者 $left\underline{\hspace{0.5em}}2 == len(slots2)$ 时跳出循环，返回空数组 $[]$。

### 思路 1：代码

```python
class Solution:
    def minAvailableDuration(self, slots1: List[List[int]], slots2: List[List[int]], duration: int) -> List[int]:
        slots1.sort()
        slots2.sort()
        size1 = len(slots1)
        size2 = len(slots2)
        left_1, left_2 = 0, 0
        while left_1 < size1 and left_2 < size2:
            start_1, end_1 = slots1[left_1]
            start_2, end_2 = slots2[left_2]
            start = max(start_1, start_2)
            end = min(end_1, end_2)
            if end - start >= duration:
                return [start, start + duration]
            if end_1 < end_2:
                left_1 += 1
            else:
                left_2 += 1
        return []
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(n \times \log n + m \times \log m)$，其中 $n$、$m$ 分别为数组 $slots1$、$slots2$  中的元素个数。
- **空间复杂度**：$O(\log n + \log m)$。

# [1232. 缀点成线](https://leetcode.cn/problems/check-if-it-is-a-straight-line/)

- 标签：几何、数组、数学
- 难度：简单

## 题目链接

- [1232. 缀点成线 - 力扣](https://leetcode.cn/problems/check-if-it-is-a-straight-line/)

## 题目大意

给定一系列的二维坐标点的坐标 `(xi, yi)`，判断这些点是否属于同一条直线。若属于同一条直线，则返回 True，否则返回 False。

## 解题思路

如果根据斜率来判断点是否处于同一条直线，需要处理斜率不存在（无穷大）的情况。我们可以使用叉乘来判断三个点构成的两个向量是否处于同一条直线上。

叉乘原理：

设向量 P 为 `(x1, y1)` 向量，Q 为 `(x2, y2)`，则向量 P、Q 的叉积定义为：$P × Q = x_1y_2 - x_2y_1$，其几何意义表示为如果以向量 P 和向量 Q 为边构成一个平行四边形，那么这两个向量叉乘的模长与这个平行四边形的正面积相等。

![向量叉积](https://img.geek-docs.com/mathematical-basis/linear-algebra/220px-Cross_product_parallelogram.png)

- 如果 `P × Q = 0`，则 P 与 Q 共线，有可能同向，也有可能反向。
- 如果 `P × Q > 0`，则 P 在 Q 的顺时针方向。
- 如果 `P × Q < 0`，则 P 在 Q 的逆时针方向。

具体求解方法：

- 先求出第一个坐标与第二个坐标构成的向量 P。
- 遍历所有坐标，求出所有坐标与第一个坐标构成的向量 Q。
  - 如果 `P × Q ≠ 0`，则返回 False。
- 如果遍历完仍没有发现 `P × Q ≠ 0`，则返回 True。

## 代码

```python
class Solution:
    def checkStraightLine(self, coordinates: List[List[int]]) -> bool:
        x1 = coordinates[1][0] - coordinates[0][0]
        y1 = coordinates[1][1] - coordinates[0][1]

        for i in range(len(coordinates)):
            x2 = coordinates[i][0] - coordinates[0][0]
            y2 = coordinates[i][1] - coordinates[0][1]
            if x1 * y2 != x2 * y1:
                return False
        return True
```

# [1245. 树的直径](https://leetcode.cn/problems/tree-diameter/)

- 标签：树、深度优先搜索、广度优先搜索、图、拓扑排序
- 难度：中等

## 题目链接

- [1245. 树的直径 - 力扣](https://leetcode.cn/problems/tree-diameter/)

## 题目大意

**描述**：给定一个数组 $edges$，用来表示一棵无向树。其中 $edges[i] = [u, v]$ 表示节点 $u$ 和节点 $v$ 之间的双向边。书上的节点编号为 $0 \sim edges.length$，共 $edges.length + 1$ 个节点。

**要求**：求出这棵无向树的直径。

**说明**：

- $0 \le edges.length < 10^4$。
- $edges[i][0] \ne edges[i][1]$。
- $0 \le edges[i][j] \le edges.length$。
- $edges$ 会形成一棵无向树。

**示例**：

- 示例 1：

![](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2019/10/31/1397_example_1.png)

```python
输入：edges = [[0,1],[0,2]]
输出：2
解释：
这棵树上最长的路径是 1 - 0 - 2，边数为 2。
```

- 示例 2：

![](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2019/10/31/1397_example_2.png)

```python
输入：edges = [[0,1],[1,2],[2,3],[1,4],[4,5]]
输出：4
解释： 
这棵树上最长的路径是 3 - 2 - 1 - 4 - 5，边数为 4。
```

## 解题思路

### 思路 1：树形 DP + 深度优先搜索

对于根节点为 $u$ 的树来说：

1. 如果其最长路径经过根节点 $u$，则：**最长路径长度 = 某子树中的最长路径长度 + 另一子树中的最长路径长度 + 1**。
2. 如果其最长路径不经过根节点 $u$，则：**最长路径长度 = 某个子树中的最长路径长度**。

即：**最长路径长度 = max(某子树中的最长路径长度 + 另一子树中的最长路径长度 + 1，某个子树中的最长路径长度)**。

对此，我们可以使用深度优先搜索递归遍历 $u$ 的所有相邻节点 $v$，并在递归遍历的同时，维护一个全局最大路径和变量 $ans$，以及当前节点 $u$ 的最大路径长度变量 $u\underline{\hspace{0.5em}}len$。

1. 先计算出从相邻节点 $v$ 出发的最长路径长度 $v\underline{\hspace{0.5em}}len$。
2. 更新维护全局最长路径长度为 $self.ans = max(self.ans, \quad u\underline{\hspace{0.5em}}len + v\underline{\hspace{0.5em}}len + 1)$。
3. 更新维护当前节点 $u$ 的最长路径长度为 $u\underline{\hspace{0.5em}}len = max(u\underline{\hspace{0.5em}}len, \quad v\underline{\hspace{0.5em}}len + 1)$。

> 注意：在遍历邻接节点的过程中，为了避免造成重复遍历，我们在使用深度优先搜索时，应过滤掉父节点。

### 思路 1：代码

```python
class Solution:
    def __init__(self):
        self.ans = 0

    def dfs(self, graph, u, fa):
        u_len = 0
        for v in graph[u]:
            if v != fa:
                v_len = self.dfs(graph, v, u)
                self.ans = max(self.ans, u_len + v_len + 1)
                u_len = max(u_len, v_len + 1)
        return u_len

    def treeDiameter(self, edges: List[List[int]]) -> int:
        size = len(edges) + 1

        graph = [[] for _ in range(size)]
        for u, v in edges:
            graph[u].append(v)
            graph[v].append(u)
        
        self.dfs(graph, 0, -1)
        return self.ans
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(n)$，其中 $n$ 为无向树中的节点个数。
- **空间复杂度**：$O(n)$。

# [1247. 交换字符使得字符串相同](https://leetcode.cn/problems/minimum-swaps-to-make-strings-equal/)

- 标签：贪心、数学、字符串
- 难度：中等

## 题目链接

- [1247. 交换字符使得字符串相同 - 力扣](https://leetcode.cn/problems/minimum-swaps-to-make-strings-equal/)

## 题目大意

**描述**：给定两个长度相同的字符串 $s1$ 和 $s2$，并且两个字符串中只含有字符 `'x'` 和 `'y'`。现在需要通过「交换字符」的方式使两个字符串相同。

- 每次「交换字符」，需要分别从两个字符串中各选一个字符进行交换。
- 「交换字符」只能发生在两个不同的字符串之间，不能发生在同一个字符串内部。

**要求**：返回使 $s1$ 和 $s2$ 相同的最小交换次数，如果没有方法能够使得这两个字符串相同，则返回 $-1$。

**说明**：

- $1 \le s1.length, s2.length \le 1000$。
- $s1$、$ s2$ 只包含 `'x'` 或 `'y'`。

**示例**：

- 示例 1：

```python
输入：s1 = "xy", s2 = "yx"
输出：2
解释：
交换 s1[0] 和 s2[0]，得到 s1 = "yy"，s2 = "xx" 。
交换 s1[0] 和 s2[1]，得到 s1 = "xy"，s2 = "xy" 。
注意，你不能交换 s1[0] 和 s1[1] 使得 s1 变成 "yx"，因为我们只能交换属于两个不同字符串的字符。
```

## 解题思路

### 思路 1：贪心算法

- 如果 $s1 == s2$，则不需要交换。
- 如果 `s1 = "xx"`，`s2 = "yy"`，则最少需要交换一次，才可以使两个字符串相等。
- 如果 `s1 = "yy"`，`s2 = "xx"`，则最少需要交换一次，才可以使两个字符串相等。
- 如果 `s1 = "xy"`，`s2 = "yx"`，则最少需要交换两次，才可以使两个字符串相等。
- 如果 `s1 = "yx"`，`s2 = "xy"`，则最少需要交换两次，才可以使两个字符串相等。

则可以总结为：

- `"xx"` 与 `"yy"`、`"yy"` 与 `"xx"` 只需要交换一次。
- `"xy"` 与 `"yx"`、`"yx"` 与 `"xy"` 需要交换两次。

我们把这两种情况分别进行统计。

- 当遇到 $s1[i] == s2[i]$ 时直接跳过。
- 当遇到 `s1[i] == 'x'`，`s2[i] == 'y'` 时，则统计数量到变量 $xyCnt$ 中。
- 当遇到 `s1[i] == 'y'`，`s2[i] == 'y'` 时，则统计数量到变量 $yxCnt$ 中。

则最后我们只需要判断 $xyCnt$ 和 $yxCnt$ 的个数即可。

- 如果 $xyCnt + yxCnt$ 是奇数，则说明最终会有一个位置上的两个字符无法通过交换相匹配。
- 如果 $xyCnt + yxCnt$ 是偶数，并且 $xyCnt$ 为偶数，则 $yxCnt$ 也为偶数。则优先交换 `"xx"` 与 `"yy"`、`"yy"` 与 `"xx"`。即每两个 $xyCnt$ 对应一次交换，每两个 $yxCnt$ 对应交换一次，则结果为 $xyCnt \div 2 + yxCnt \div 2$。
- 如果 $xyCnt + yxCnt$ 是偶数，并且 $xyCnt$ 为奇数，则 $yxCnt$ 也为奇数。则优先交换 `"xx"` 与 `"yy"`、`"yy"` 与 `"xx"`。即每两个 $xyCnt$ 对应一次交换，每两个 $yxCnt$ 对应交换一次，则结果为 $xyCnt \div 2 + yxCnt \div 2$。最后还剩一组 `"xy"` 与 `"yx"` 或者 `"yx"` 与 `"xy"`，则再交换一次，则结果为 $xyCnt \div 2 + yxCnt \div 2 + 2$。

以上结果可以统一写成 $xyCnt \div 2 + yxCnt \div 2 + xyCnt \mod 2 \times 2$。

### 思路 1：贪心算法代码

```python
class Solution:
    def minimumSwap(self, s1: str, s2: str) -> int:
        xyCnt, yxCnt = 0, 0
        for i in range(len(s1)):
            if s1[i] == s2[i]:
                continue
            if s1[i] == 'x':
                xyCnt += 1
            else:
                yxCnt += 1

        if (xyCnt + yxCnt) & 1:
            return -1
        return xyCnt // 2 + yxCnt // 2 + (xyCnt % 2 * 2)
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(n)$，其中 $n$ 为字符串的长度。
- **空间复杂度**：$O(1)$。
# [1253. 重构 2 行二进制矩阵](https://leetcode.cn/problems/reconstruct-a-2-row-binary-matrix/)

- 标签：贪心、数组、矩阵
- 难度：中等

## 题目链接

- [1253. 重构 2 行二进制矩阵 - 力扣](https://leetcode.cn/problems/reconstruct-a-2-row-binary-matrix/)

## 题目大意

**描述**：给定一个 $2$ 行 $n$ 列的二进制数组：

- 矩阵是一个二进制矩阵，这意味着矩阵中的每个元素不是 $0$ 就是 $1$。
- 第 $0$ 行的元素之和为 $upper$。
- 第 $1$ 行的元素之和为 $lowe$r。
- 第 $i$ 列（从 $0$ 开始编号）的元素之和为 $colsum[i]$，$colsum$ 是一个长度为 $n$ 的整数数组。

**要求**：你需要利用 $upper$，$lower$ 和 $colsum$ 来重构这个矩阵，并以二维整数数组的形式返回它。

**说明**：

- 如果有多个不同的答案，那么任意一个都可以通过本题。
- 如果不存在符合要求的答案，就请返回一个空的二维数组。
- $1 \le colsum.length \le 10^5$。
- $0 \le upper, lower \le colsum.length$。
- $0 \le colsum[i] \le 2$。

**示例**：

- 示例 1：

```python
输入：upper = 2, lower = 1, colsum = [1,1,1]
输出：[[1,1,0],[0,0,1]]
解释：[[1,0,1],[0,1,0]] 和 [[0,1,1],[1,0,0]] 也是正确答案。
```

- 示例 2：

```python
输入：upper = 2, lower = 3, colsum = [2,2,1,1]
输出：[]
```

## 解题思路

### 思路 1：贪心算法

1. 先构建一个 $2 \times n$ 的答案数组 $ans$，其中 $ans[0]$ 表示矩阵的第 $0$ 行，$ans[1]$ 表示矩阵的第 $1$​ 行。
2. 遍历数组 $colsum$，对于当前列的和 $colsum[i]$ 来说：
   1. 如果 $colsum[i] == 2$，则需要将 $ans[0][i]$ 和 $ans[1][i]$ 都置为 $1$，此时 $upper$ 和 $lower$ 各自减去 $1$。
   2. 如果 $colsum[i] == 1$，则需要将 $ans[0][i]$ 置为 $1$ 或将 $ans[1][i]$ 置为 $1$。我们优先使用元素和多的那一项。
      1. 如果 $upper > lower$，则优先使用 $upper$，将 $ans[0][i]$ 置为 $1$，并且令 $upper$ 减去 $1$。
      2. 如果 $upper \le lower$，则优先使用 $lower$，将 $ans[1][i]$ 置为 $1$，并且令 $lower$ 减去 $1$。
   3. 如果 $colsum[i] == 0$，则需要将 $ans[0][i]$ 和 $ans[1][i]$ 都置为 $0$。
3. 在遍历过程中，如果出现 $upper < 0$ 或者 $lower < 0$，则说明无法构造出满足要求的矩阵，则直接返回空数组。
4. 遍历结束后，如果 $upper$ 和 $lower$ 都为 $0$，则返回答案数组 $ans$；否则返回空数组。

### 思路 1：代码

```Python
class Solution:
    def reconstructMatrix(self, upper: int, lower: int, colsum: List[int]) -> List[List[int]]:
        size = len(colsum)
        ans = [[0 for _ in range(size)] for _ in range(2)]

        for i in range(size):
            if colsum[i] == 2:
                ans[0][i] = ans[1][i] = 1
                upper -= 1
                lower -= 1
            elif colsum[i] == 1:
                if upper > lower:
                    ans[0][i] = 1
                    upper -= 1
                else:
                    ans[1][i] = 1
                    lower -= 1
            if upper < 0 or lower < 0:
                return []
        if lower != 0 or upper != 0:
            return []
        return ans
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(n)$。
- **空间复杂度**：$O(n)$。

# [1254. 统计封闭岛屿的数目](https://leetcode.cn/problems/number-of-closed-islands/)

- 标签：深度优先搜索、广度优先搜索、并查集、数组、矩阵
- 难度：中等

## 题目链接

- [1254. 统计封闭岛屿的数目 - 力扣](https://leetcode.cn/problems/number-of-closed-islands/)

## 题目大意

**描述**：给定一个二维矩阵 `grid`，每个位置要么是陆地（记号为 `0`）要么是水域（记号为 `1`）。

我们从一块陆地出发，每次可以往上下左右 `4` 个方向相邻区域走，能走到的所有陆地区域，我们将其称为一座「岛屿」。

如果一座岛屿完全由水域包围，即陆地边缘上下左右所有相邻区域都是水域，那么我们将其称为「封闭岛屿」。

**要求**：返回封闭岛屿的数目。

**说明**：

- $1 \le grid.length, grid[0].length \le 100$。
- $0 \le grid[i][j] \le 1$。

**示例**：

- 示例 1：

![](https://assets.leetcode.com/uploads/2019/10/31/sample_3_1610.png)

```python
输入：grid = [[1,1,1,1,1,1,1,0],[1,0,0,0,0,1,1,0],[1,0,1,0,1,1,1,0],[1,0,0,0,0,1,0,1],[1,1,1,1,1,1,1,0]]
输出：2
解释：灰色区域的岛屿是封闭岛屿，因为这座岛屿完全被水域包围（即被 1 区域包围）。
```

- 示例 2：

![](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2019/11/07/sample_4_1610.png)

```python
输入：grid = [[0,0,1,0,0],[0,1,0,1,0],[0,1,1,1,0]]
输出：1
```

## 解题思路

### 思路 1：深度优先搜索

1. 从 `grid[i][j] == 0` 的位置出发，使用深度优先搜索的方法遍历上下左右四个方向上相邻区域情况。
   1. 如果上下左右都是 `grid[i][j] == 1`，则返回 `True`。
   2. 如果有一个以上方向的 `grid[i][j] == 0`，则返回 `False`。
   3. 遍历之后将当前陆地位置置为 `1`，表示该位置已经遍历过了。
2. 最后统计出上下左右都满足 `grid[i][j] == 1` 的情况数量，即为答案。

### 思路 1：代码

```python
class Solution:
    directs = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    def dfs(self, grid, i, j):
        n, m = len(grid), len(grid[0])
        if i < 0 or i >= n or j < 0 or j >= m:
            return False
        if grid[i][j] == 1:
            return True
        grid[i][j] = 1

        res = True
        for direct in self.directs:
            new_i = i + direct[0]
            new_j = j + direct[1]
            if not self.dfs(grid, new_i, new_j):
                res = False
        return res

    def closedIsland(self, grid: List[List[int]]) -> int:
        res = 0
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] == 0 and self.dfs(grid, i, j):
                    res += 1

        return res
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(m \times n)$。其中 $m$ 和 $n$ 分别为行数和列数。
- **空间复杂度**：$O(m \times n)$。# [1261. 在受污染的二叉树中查找元素](https://leetcode.cn/problems/find-elements-in-a-contaminated-binary-tree/)

- 标签：树、深度优先搜索、广度优先搜索、设计、哈希表、二叉树
- 难度：中等

## 题目链接

- [1261. 在受污染的二叉树中查找元素 - 力扣](https://leetcode.cn/problems/find-elements-in-a-contaminated-binary-tree/)

## 题目大意

**描述**：给出一满足下属规则的二叉树的根节点 $root$：

1. $root.val == 0$。
2. 如果 $node.val == x$ 且 $node.left \ne None$，那么 $node.left.val == 2 \times x + 1$。
3. 如果 $node.val == x$ 且 $node.right \ne None$，那么 $node.left.val == 2 \times x + 2$​。

现在这个二叉树受到「污染」，所有的 $node.val$ 都变成了 $-1$。

**要求**：请你先还原二叉树，然后实现 `FindElements` 类：

- `FindElements(TreeNode* root)` 用受污染的二叉树初始化对象，你需要先把它还原。
- `bool find(int target)` 判断目标值 $target$ 是否存在于还原后的二叉树中并返回结果。

**说明**：

- $node.val == -1$
- 二叉树的高度不超过 $20$。
- 节点的总数在 $[1, 10^4]$ 之间。
- 调用 `find()` 的总次数在 $[1, 10^4]$ 之间。
- $0 \le target \le 10^6$。

**示例**：

- 示例 1：

![](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2019/11/16/untitled-diagram-4-1.jpg)

```python
输入：
["FindElements","find","find"]
[[[-1,null,-1]],[1],[2]]
输出：
[null,false,true]
解释：
FindElements findElements = new FindElements([-1,null,-1]); 
findElements.find(1); // return False 
findElements.find(2); // return True 
```

- 示例 2：

![](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2019/11/16/untitled-diagram-4.jpg)

```python
输入：
["FindElements","find","find","find"]
[[[-1,-1,-1,-1,-1]],[1],[3],[5]]
输出：
[null,true,true,false]
解释：
FindElements findElements = new FindElements([-1,-1,-1,-1,-1]);
findElements.find(1); // return True
findElements.find(3); // return True
findElements.find(5); // return False
```

## 解题思路

### 思路 1：哈希表 + 深度优先搜索

1. 从根节点开始进行还原。
2. 然后使用深度优先搜索的方式，依次递归还原左右两个孩子节点。
3. 递归还原的同时，将还原之后的所有节点值，存入集合 $val\underline{\hspace{0.5em}}set$ 中。

这样就可以在 $O(1)$ 的时间复杂度内判断目标值 $target$ 是否在还原后的二叉树中了。

### 思路 1：代码

```Python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class FindElements:

    def __init__(self, root: Optional[TreeNode]):
        self.val_set = set()
        def dfs(node, val):
            if not node:
                return
            self.val_set.add(val)
            dfs(node.left, val * 2 + 1)
            dfs(node.right, val * 2 + 2)
        
        dfs(root, 0)


    def find(self, target: int) -> bool:
        return target in self.val_set



# Your FindElements object will be instantiated and called as such:
# obj = FindElements(root)
# param_1 = obj.find(target)
```

### 思路 1：复杂度分析

- **时间复杂度**：还原二叉树：$O(n)$，其中 $n$ 为二叉树中的节点个数。查找目标值：$O(1)$。
- **空间复杂度**：$O(n)$。

# [1266. 访问所有点的最小时间](https://leetcode.cn/problems/minimum-time-visiting-all-points/)

- 标签：几何、数组、数学
- 难度：简单

## 题目链接

- [1266. 访问所有点的最小时间 - 力扣](https://leetcode.cn/problems/minimum-time-visiting-all-points/)

## 题目大意

**描述**：给定 $n$ 个点的整数坐标数组 $points$。其中 $points[i] = [xi, yi]$，表示第 $i$ 个点坐标为 $(xi, yi)$。可以按照以下规则在平面上移动：

1. 每一秒内，可以：
   1. 沿着水平方向移动一个单位长度。
   2. 沿着竖直方向移动一个单位长度。
   3. 沿着对角线移动 $\sqrt 2$ 个单位长度（可看做在一秒内沿着水平方向和竖直方向各移动一个单位长度）。
2. 必须按照坐标数组 $points$ 中的顺序来访问这些点。
3. 在访问某个点时，可以经过该点后面出现的点，但经过的那些点不算作有效访问。

**要求**：计算出访问这些点需要的最小时间（以秒为单位）。

**说明**：

- $points.length == n$。
- $1 \le n \le 100$。
- $points[i].length == 2$。
- $-1000 \le points[i][0], points[i][1] \le 1000$。

**示例**：

- 示例 1：

![](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2019/11/24/1626_example_1.png)

```python
输入：points = [[1,1],[3,4],[-1,0]]
输出：7
解释：一条最佳的访问路径是： [1,1] -> [2,2] -> [3,3] -> [3,4] -> [2,3] -> [1,2] -> [0,1] -> [-1,0]   
从 [1,1] 到 [3,4] 需要 3 秒 
从 [3,4] 到 [-1,0] 需要 4 秒
一共需要 7 秒
```

```python
输入：points = [[3,2],[-2,2]]
输出：5
```

## 解题思路

### 思路 1：数学

根据题意，每一秒可以沿着水平方向移动一个单位长度、或者沿着竖直方向移动一个单位长度、或者沿着对角线移动 $\sqrt 2$ 个单位长度。而沿着对角线移动 $\sqrt 2$ 个单位长度可以看做是先沿着水平方向移动一个单位长度，又沿着竖直方向移动一个单位长度，算是一秒走了两步距离。

现在假设从 A 点（坐标为 $(x1, y1)$）移动到 B 点（坐标为 $(x2, y2)$）。

那么从 A 点移动到 B 点如果要想得到最小时间，我们应该计算出沿着水平方向走的距离为 $dx = |x2 - x1|$，沿着竖直方向走的距离为 $dy = |y2 - y1|$。

然后比较沿着水平方向的移动距离和沿着竖直方向的移动距离。

- 如果 $dx > dy$，则我们可以先沿着对角线移动 $dy$ 次，再水平移动 $dx - dy$ 次，总共 $dx$ 次。
- 如果 $dx == dy$，则我们可以直接沿着对角线移动 $dx$ 次，总共 $dx$ 次。
- 如果 $dx < dy$，则我们可以先沿着对角线移动 $dx$ 次，再水平移动 $dy - dx$ 次，，总共 $dy$ 次。

根据上面观察可以发现：最小时间取决于「走的步数较多的那个方向所走的步数」，即 $max(dx, dy)$。

根据题目要求，需要按照坐标数组 $points$ 中的顺序来访问这些点，则我们需要按顺序遍历整个数组，计算出相邻点之间的 $max(dx, dy)$，将其累加到答案中。

最后将答案输出即可。

### 思路 1：代码

```python
class Solution:
    def minTimeToVisitAllPoints(self, points: List[List[int]]) -> int:
        ans = 0
        x1, y1 = points[0]
        for point in points:
            x2, y2 = point
            ans += max(abs(x2 - x1), abs(y2 - y1))
            x1, y1 = point
        
        return ans    
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(n)$。
- **空间复杂度**：$O(1)$。
# [1268. 搜索推荐系统](https://leetcode.cn/problems/search-suggestions-system/)

- 标签：字典树、数组、字符串
- 难度：中等

## 题目链接

- [1268. 搜索推荐系统 - 力扣](https://leetcode.cn/problems/search-suggestions-system/)

## 题目大意

给定一个产品数组 `products` 和一个字符串 `searchWord` ，`products`  数组中每个产品都是一个字符串。

要求：设计一个推荐系统，在依次输入单词 `searchWord` 的每一个字母后，推荐 `products` 数组中前缀与 `searchWord` 相同的最多三个产品（如果前缀相同的可推荐产品超过三个，请按字典序返回最小的三个）。

- 请你以二维列表的形式，返回在输入 `searchWord` 每个字母后相应的推荐产品的列表。

## 解题思路

先将产品数组按字典序排序。

然后使用字典树结构存储每个产品，并在字典树中维护一个数组，用于表示当前前缀所对应的产品列表（只保存最多 3 个产品）。

在查询的时候，将不同前缀所对应的产品列表加入到答案数组中。

最后输出答案数组。

## 代码

```python
class Trie:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.children = dict()
        self.isEnd = False
        self.words = list()


    def insert(self, word: str) -> None:
        """
        Inserts a word into the trie.
        """
        cur = self
        for ch in word:
            if ch not in cur.children:
                cur.children[ch] = Trie()
            cur = cur.children[ch]
            if len(cur.words) < 3:
                cur.words.append(word)
        cur.isEnd = True


    def search(self, word: str) -> bool:
        """
        Returns if the word is in the trie.
        """
        cur = self
        res = []
        flag = False
        for ch in word:
            if flag or ch not in cur.children:
                res.append([])
                flag = True
            else:
                cur = cur.children[ch]
                res.append(cur.words)

        return res

class Solution:
    def suggestedProducts(self, products: List[str], searchWord: str) -> List[List[str]]:
        products.sort()
        trie_tree = Trie()
        for product in products:
            trie_tree.insert(product)

        return trie_tree.search(searchWord)
```

# [1281. 整数的各位积和之差](https://leetcode.cn/problems/subtract-the-product-and-sum-of-digits-of-an-integer/)

- 标签：数学
- 难度：简单

## 题目链接

- [1281. 整数的各位积和之差 - 力扣](https://leetcode.cn/problems/subtract-the-product-and-sum-of-digits-of-an-integer/)

## 题目大意

**描述**：给定一个整数 `n`。

**要求**：计算并返回该整数「各位数字之积」与「各位数字之和」的差。

**说明**：

- $1 <= n <= 10^5$。

**示例**：

- 示例 1：

```python
输入：n = 234
输出：15

解释：
各位数之积 2 * 3 * 4 = 24 
各位数之和 2 + 3 + 4 = 9 
结果 24 - 9 = 15
```

## 解题思路

### 思路 1：数学

- 通过取模运算得到 `n` 的最后一位，即 `n %= 10`。
- 然后去除  `n`  的最后一位，及`n //= 10`。
- 一次求出各位数字之积与各位数字之和，并返回其差值。

### 思路 1：数学代码

```python
class Solution:
    def subtractProductAndSum(self, n: int) -> int:
        product = 1
        total = 0
        while n:
            digit = n % 10
            product *= digit
            total += digit
            n //= 10
        return product - total
```

# [1296. 划分数组为连续数字的集合](https://leetcode.cn/problems/divide-array-in-sets-of-k-consecutive-numbers/)

- 标签：贪心、数组、哈希表、排序
- 难度：中等

## 题目链接

- [1296. 划分数组为连续数字的集合 - 力扣](https://leetcode.cn/problems/divide-array-in-sets-of-k-consecutive-numbers/)

## 题目大意

**描述**：给定一个整数数组 `nums` 和一个正整数 `k`。

**要求**：判断是否可以把这个数组划分成一些由 `k` 个连续数字组成的集合。如果可以，则返回 `True`；否则，返回 `False`。

**说明**：

- $1 \le k \le nums.length \le 10^5$。
- $1 \le nums[i] \le 10^9$。

**示例**：

- 示例 1：

```python
输入：nums = [1,2,3,3,4,4,5,6], k = 4
输出：True
解释：数组可以分成 [1,2,3,4] 和 [3,4,5,6]。
```

## 解题思路

### 思路 1：哈希表 + 排序

1. 使用哈希表存储每个数出现的次数。
2. 将哈希表中每个键从小到大排序。
3. 从哈希表中最小的数开始，以它作为当前连续数字的开头，然后依次判断连续的 `k` 个数是否在哈希表中，如果在的话，则将哈希表中对应数的数量减 `1`。不在的话，说明无法满足题目要求，直接返回 `False`。
4. 重复执行 2 ~ 3 步，直到哈希表为空。最后返回 `True`。

### 思路 1：哈希表 + 排序代码

```python
class Solution:
    def isPossibleDivide(self, nums: List[int], k: int) -> bool:
        hand_map = collections.defaultdict(int)
        for i in range(len(nums)):
            hand_map[nums[i]] += 1
        for key in sorted(hand_map.keys()):
            value = hand_map[key]
            if value == 0:
                continue
            count = 0
            for i in range(k):
                hand_map[key + count] -= value
                if hand_map[key + count] < 0:
                    return False
                count += 1
        return True
```
# [1300. 转变数组后最接近目标值的数组和](https://leetcode.cn/problems/sum-of-mutated-array-closest-to-target/)

- 标签：数组、二分查找、排序
- 难度：中等

## 题目链接

- [1300. 转变数组后最接近目标值的数组和 - 力扣](https://leetcode.cn/problems/sum-of-mutated-array-closest-to-target/)

## 题目大意

**描述**：给定一个整数数组 $arr$ 和一个目标值 $target$。

**要求**：返回一个整数 $value$，使得将数组中所有大于 $value$ 的值变成 $value$ 后，数组的和最接近 $target$（最接近表示两者之差的绝对值最小）。如果有多种使得和最接近 $target$ 的方案，请你返回这些整数中的最小值。

**说明**：

- 答案 $value$ 不一定是 $arr$ 中的数字。
- $1 \le arr.length \le 10^4$。
- $1 \le arr[i], target \le 10^5$。

**示例**：

- 示例 1：

```python
输入：arr = [4,9,3], target = 10
输出：3
解释：当选择 value 为 3 时，数组会变成 [3, 3, 3]，和为 9 ，这是最接近 target 的方案。
```

- 示例 2：

```python
输入：arr = [60864,25176,27249,21296,20204], target = 56803
输出：11361
```

## 解题思路

### 思路 1：二分查找

题目可以理解为：在 $[0, max(arr)]$ 的区间中，查找一个值 $value$。使得「转变后的数组和」与 $target$ 最接近。

- 转变规则：将数组中大于 $value$ 的值变为 $value$。

在 $[0, max(arr)]$ 的区间中，查找一个值 $value$ 可以使用二分查找答案的方式减少时间复杂度。但是这个最接近 $target$ 应该怎么理解，或者说怎么衡量接近程度。

最接近 $target$ 的肯定是数组和等于 $target$ 的时候。不过更可能是出现数组和恰好比 $target$ 大一点，或数组和恰好比 $target$ 小一点。我们可以将 $target$ 上下两个值相对应的数组和与 $target$ 进行比较，输出差值更小的那一个 $value$。

在根据查找的值 $value$ 计算数组和时，也可以通过二分查找方法查找出数组刚好大于等于 $value$ 元素下标。还可以根据事先处理过的前缀和数组，快速得到转变后的数组和。

最后输出使得数组和与 $target$ 差值更小的 $value$。

整个算法步骤如下：

- 先对数组排序，并计算数组的前缀和 $pre\underline{\hspace{0.5em}}sum$。
- 通过二分查找在 $[0, arr[-1]]$ 中查找使得转变后数组和刚好大于等于 $target$ 的值 $value$。
- 计算 $value$ 对应的数组和 $sum\underline{\hspace{0.5em}}1$，以及 $value - 1$ 对应的数组和 $sum\underline{\hspace{0.5em}}2$。并分别计算与 $target$ 的差值 $diff\underline{\hspace{0.5em}}1$、$diff\underline{\hspace{0.5em}}2$。
- 输出差值小的那个值。

### 思路 1：代码

```python
class Solution:
    # 计算 value 对应的转变后的数组
    def calc_sum(self, arr, value, pre_sum):
        size = len(arr)
        left, right = 0, size - 1
        while left < right:
            mid = left + (right - left) // 2
            if arr[mid] < value:
                left = mid + 1
            else:
                right = mid

        return pre_sum[left] + (size - left) * value

    # 查找使得转变后的数组和刚好大于等于 target 的 value
    def binarySearchValue(self, arr, target, pre_sum):
        left, right = 0, arr[-1]
        while left < right:
            mid = left + (right - left) // 2
            if self.calc_sum(arr, mid, pre_sum) < target:
                left = mid + 1
            else:
                right = mid
        return left

    def findBestValue(self, arr: List[int], target: int) -> int:
        size = len(arr)
        arr.sort()
        pre_sum = [0 for _ in range(size + 1)]

        for i in range(size):
            pre_sum[i + 1] = pre_sum[i] + arr[i]

        value = self.binarySearchValue(arr, target, pre_sum)

        sum_1 = self.calc_sum(arr, value, pre_sum)
        sum_2 = self.calc_sum(arr, value - 1, pre_sum)
        diff_1 = abs(sum_1 - target)
        diff_2 = abs(sum_2 - target)

        return value if diff_1 < diff_2 else value - 1
```

### 思路 1：复杂度分析

- **时间复杂度**：$O((n + k) \times \log n)$。其中 $n$ 是数组 $arr$ 的长度，$k$ 是数组 $arr$ 中的最大值。
- **空间复杂度**：$O(n)$。

# [1305. 两棵二叉搜索树中的所有元素](https://leetcode.cn/problems/all-elements-in-two-binary-search-trees/)

- 标签：树、深度优先搜索、二叉搜索树、二叉树、排序
- 难度：中等

## 题目链接

- [1305. 两棵二叉搜索树中的所有元素 - 力扣](https://leetcode.cn/problems/all-elements-in-two-binary-search-trees/)

## 题目大意

**描述**：给定两棵二叉搜索树的根节点 $root1$ 和 $root2$。

**要求**：返回一个列表，其中包含两棵树中所有整数并按升序排序。

**说明**：

- 每棵树的节点数在 $[0, 5000]$ 范围内。
- $-10^5 \le Node.val \le 10^5$。

**示例**：

- 示例 1：

![](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2019/12/29/q2-e1.png)

```python
输入：root1 = [2,1,4], root2 = [1,0,3]
输出：[0,1,1,2,3,4]
```

- 示例 2：

![](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2019/12/29/q2-e5-.png)

```python
输入：root1 = [1,null,8], root2 = [8,1]
输出：[1,1,8,8]
```

## 解题思路

### 思路 1：二叉树的中序遍历 + 快慢指针

根据二叉搜索树的特性，如果我们以中序遍历的方式遍历整个二叉搜索树时，就会得到一个有序递增列表。我们按照这样的方式分别对两个二叉搜索树进行中序遍历，就得到了两个有序数组，那么问题就变成了：两个有序数组的合并问题。

两个有序数组的合并可以参考归并排序中的归并过程，使用快慢指针将两个有序数组合并为一个有序数组。

具体步骤如下：

1. 分别使用中序遍历的方式遍历两个二叉搜索树，得到两个有序数组 $nums1$、$nums2$。
2. 使用两个指针 $index1$、$index2$ 分别指向两个有序数组的开始位置。
3. 比较两个指针指向的元素，将两个有序数组中较小元素依次存入结果数组 $nums$ 中，并将指针移动到下一个位置。
4. 重复步骤 $3$，直到某一指针到达数组末尾。
5. 将另一个数组中的剩余元素依次存入结果数组 $nums$ 中。
6. 返回结果数组 $nums$。

### 思路 1：代码

```python
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
    def getAllElements(self, root1: TreeNode, root2: TreeNode) -> List[int]:
        nums1 = self.inorderTraversal(root1)
        nums2 = self.inorderTraversal(root2)
        nums = []
        index1, index2 = 0, 0
        while index1 < len(nums1) and index2 < len(nums2):
            if nums1[index1] < nums2[index2]:
                nums.append(nums1[index1])
                index1 += 1
            else:
                nums.append(nums2[index2])
                index2 += 1
        
        while index1 < len(nums1):
            nums.append(nums1[index1])
            index1 += 1
    
        while index2 < len(nums2):
            nums.append(nums2[index2])
            index2 += 1
        
        return nums
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(n + m)$，其中 $n$ 和 $m$ 分别为两棵二叉搜索树的节点个数。
- **空间复杂度**：$O(n + m)$。
# [1310. 子数组异或查询](https://leetcode.cn/problems/xor-queries-of-a-subarray/)

- 标签：位运算、数组、前缀和
- 难度：中等

## 题目链接

- [1310. 子数组异或查询 - 力扣](https://leetcode.cn/problems/xor-queries-of-a-subarray/)

## 题目大意

**描述**：给定一个正整数数组 `arr`，再给定一个对应的查询数组 `queries`，其中 `queries[i] = [Li, Ri]`。

**要求**：对于每个查询 `queries[i]`，要求计算从 `Li` 到 `Ri` 的异或值（即 `arr[Li] ^ arr[Li+1] ^ ... ^ arr[Ri]`）作为本次查询的结果。并返回一个包含给定查询 `queries` 所有结果的数组。

**说明**：

- $1 \le arr.length \le 3 * 10^4$。
- $1 \le arr[i] \le 10^9$。
- $1 \le queries.length \le 3 * 10^4$。
- $queries[i].length == 2$。
- $0 \le queries[i][0] \le queries[i][1] < arr.length$。

**示例**：

- 示例 1：

```python
输入：arr = [1,3,4,8], queries = [[0,1],[1,2],[0,3],[3,3]]
输出：[2,7,14,8] 
解释

数组中元素的二进制表示形式是：
1 = 0001 
3 = 0011 
4 = 0100 
8 = 1000 

查询的 XOR 值为：
[0,1] = 1 xor 3 = 2 
[1,2] = 3 xor 4 = 7 
[0,3] = 1 xor 3 xor 4 xor 8 = 14 
[3,3] = 8
```

## 解题思路

### 思路 1：线段树

- 使用数组 `res` 作为答案数组，用于存放每个查询的结果值。
- 根据 `nums` 数组构建一棵线段树。
- 然后遍历查询数组 `queries`。对于每个查询 `queries[i]`，在线段树中查询对应区间的异或值，将其结果存入答案数组 `res` 中。
- 返回答案数组 `res` 即可。

这样构建线段树的时间复杂度为 $O(\log n)$，单次区间查询的时间复杂度为 $O(\log n)$。总体时间复杂度为 $O(k * \log n)$，其中 $k$ 是查询次数。

### 思路 1：线段树代码

```python
# 线段树的节点类
class SegTreeNode:
    def __init__(self, val=0):
        self.left = -1                              # 区间左边界
        self.right = -1                             # 区间右边界
        self.val = val                              # 节点值（区间值）
        self.lazy_tag = None                        # 区间和问题的延迟更新标记
        
        
# 线段树类
class SegmentTree:
    # 初始化线段树接口
    def __init__(self, nums, function):
        self.size = len(nums)
        self.tree = [SegTreeNode() for _ in range(4 * self.size)]  # 维护 SegTreeNode 数组
        self.nums = nums                            # 原始数据
        self.function = function                    # function 是一个函数，左右区间的聚合方法
        if self.size > 0:
            self.__build(0, 0, self.size - 1)
    
    # 单点更新接口：将 nums[i] 更改为 val
    def update_point(self, i, val):
        self.nums[i] = val
        self.__update_point(i, val, 0)
    
    # 区间更新接口：将区间为 [q_left, q_right] 上的所有元素值加上 val
    def update_interval(self, q_left, q_right, val):
        self.__update_interval(q_left, q_right, val, 0)
        
    # 区间查询接口：查询区间为 [q_left, q_right] 的区间值
    def query_interval(self, q_left, q_right):
        return self.__query_interval(q_left, q_right, 0)
    
    # 获取 nums 数组接口：返回 nums 数组
    def get_nums(self):
        for i in range(self.size):
            self.nums[i] = self.query_interval(i, i)
        return self.nums
        
        
    # 以下为内部实现方法
    
    # 构建线段树实现方法：节点的存储下标为 index，节点的区间为 [left, right]
    def __build(self, index, left, right):
        self.tree[index].left = left
        self.tree[index].right = right
        if left == right:                           # 叶子节点，节点值为对应位置的元素值
            self.tree[index].val = self.nums[left]
            return
    
        mid = left + (right - left) // 2            # 左右节点划分点
        left_index = index * 2 + 1                  # 左子节点的存储下标
        right_index = index * 2 + 2                 # 右子节点的存储下标
        self.__build(left_index, left, mid)         # 递归创建左子树
        self.__build(right_index, mid + 1, right)   # 递归创建右子树
        self.__pushup(index)                        # 向上更新节点的区间值
    
    
    # 区间查询实现方法：在线段树中搜索区间为 [q_left, q_right] 的区间值
    def __query_interval(self, q_left, q_right, index):
        left = self.tree[index].left
        right = self.tree[index].right
        
        if left >= q_left and right <= q_right:     # 节点所在区间被 [q_left, q_right] 所覆盖
            return self.tree[index].val             # 直接返回节点值
        if right < q_left or left > q_right:        # 节点所在区间与 [q_left, q_right] 无关
            return 0
    
        mid = left + (right - left) // 2            # 左右节点划分点
        left_index = index * 2 + 1                  # 左子节点的存储下标
        right_index = index * 2 + 2                 # 右子节点的存储下标
        res_left = 0                                # 左子树查询结果
        res_right = 0                               # 右子树查询结果
        if q_left <= mid:                           # 在左子树中查询
            res_left = self.__query_interval(q_left, q_right, left_index)
        if q_right > mid:                           # 在右子树中查询
            res_right = self.__query_interval(q_left, q_right, right_index)
        
        return self.function(res_left, res_right)   # 返回左右子树元素值的聚合计算结果
    
    # 向上更新实现方法：更新下标为 index 的节点区间值 等于 该节点左右子节点元素值的聚合计算结果
    def __pushup(self, index):
        left_index = index * 2 + 1                  # 左子节点的存储下标
        right_index = index * 2 + 2                 # 右子节点的存储下标
        self.tree[index].val = self.function(self.tree[left_index].val, self.tree[right_index].val)


class Solution:
    def xorQueries(self, arr: List[int], queries: List[List[int]]) -> List[int]:
        self.STree = SegmentTree(arr, lambda x, y: (x ^ y))
        res = []
        for query in queries:
            ans = self.STree.query_interval(query[0], query[1])
            res.append(ans)
        return res
```
# [1313. 解压缩编码列表](https://leetcode.cn/problems/decompress-run-length-encoded-list/)

- 标签：数组
- 难度：简单

## 题目链接

- [1313. 解压缩编码列表 - 力扣](https://leetcode.cn/problems/decompress-run-length-encoded-list/)

## 题目大意

**描述**：给定一个以行程长度编码压缩的整数列表 $nums$。

考虑每对相邻的两个元素 $[freq, val] = [nums[2 \times i], nums[2 \times i + 1]]$ （其中 $i \ge 0$ ），每一对都表示解压后子列表中有 $freq$ 个值为 $val$ 的元素，你需要从左到右连接所有子列表以生成解压后的列表。

**要求**：返回解压后的列表。

**说明**：

- $2 \le nums.length \le 100$。
- $nums.length \mod 2 == 0$。
- $1 \le nums[i] \le 100$。

**示例**：

- 示例 1：

```python
输入：nums = [1,2,3,4]
输出：[2,4,4,4]
解释：第一对 [1,2] 代表着 2 的出现频次为 1，所以生成数组 [2]。
第二对 [3,4] 代表着 4 的出现频次为 3，所以生成数组 [4,4,4]。
最后将它们串联到一起 [2] + [4,4,4] = [2,4,4,4]。
```

- 示例 2：

```python
输入：nums = [1,1,2,3]
输出：[1,3,3]
```

## 解题思路

### 思路 1：模拟

1. 以步长为 $2$，遍历数组 $nums$。
2. 对于遍历到的元素 $nums[i]$、$nnums[i + 1]$，将 $nums[i]$ 个 $nums[i + 1]$ 存入答案数组中。
3. 返回答案数组。

### 思路 1：代码

```Python
class Solution:
    def decompressRLElist(self, nums: List[int]) -> List[int]:
        res = []
        for i in range(0, len(nums), 2):
            cnts = nums[i]
            for cnt in range(cnts):
                res.append(nums[i + 1])
        
        return res
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(n + s)$，其中 $n$ 为数组 $nums$ 的长度，$s$ 是数组 $nums$  中所有偶数下标对应元素之和。
- **空间复杂度**：$O(s)$。

# [1317. 将整数转换为两个无零整数的和](https://leetcode.cn/problems/convert-integer-to-the-sum-of-two-no-zero-integers/)

- 标签：数学
- 难度：简单

## 题目链接

- [1317. 将整数转换为两个无零整数的和 - 力扣](https://leetcode.cn/problems/convert-integer-to-the-sum-of-two-no-zero-integers/)

## 题目大意

**描述**：给定一个整数 $n$。

**要求**：返回一个由两个整数组成的列表 $[A, B]$，满足：

- $A$ 和 $B$ 都是无零整数。
- $A + B = n$。

**说明**：

- **无零整数**：十进制表示中不含任何 $0$ 的正整数。
- 题目数据保证至少一个有效的解决方案。
- 如果存在多个有效解决方案，可以返回其中任意一个。
- $2 \le n \le 10^4$。

**示例**：

- 示例 1：

```python
输入：n = 2
输出：[1,1]
解释：A = 1, B = 1. A + B = n 并且 A 和 B 的十进制表示形式都不包含任何 0。
```

- 示例 2：

```python
输入：n = 11
输出：[2,9]
```

## 解题思路

### 思路 1：枚举

1. 由于给定的 $n$ 范围为 $[1, 10000]$，比较小，我们可以直接在 $[1, n)$ 的范围内枚举 $A$，并通过 $n - A$ 得到 $B$。
2. 在判断 $A$ 和 $B$ 中是否都不包含 $0$。如果都不包含 $0$，则返回 $[A, B]$。

### 思路 1：代码

```python
class Solution:
    def getNoZeroIntegers(self, n: int) -> List[int]:
        for A in range(1, n):
            B = n - A
            if '0' not in str(A) and '0' not in str(B):
                return [A, B]
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(n \times \log n)$。
- **空间复杂度**：$O(1)$。

# [1319. 连通网络的操作次数](https://leetcode.cn/problems/number-of-operations-to-make-network-connected/)

- 标签：深度优先搜索、广度优先搜索、并查集、图
- 难度：中等

## 题目链接

- [1319. 连通网络的操作次数 - 力扣](https://leetcode.cn/problems/number-of-operations-to-make-network-connected/)

## 题目大意

**描述**：$n$ 台计算机通过网线连接成一个网络，计算机的编号从 $0$ 到 $n - 1$。线缆用 $comnnections$ 表示，其中 $connections[i] = [a, b]$ 表示连接了计算机 $a$ 和 $b$。

给定这个计算机网络的初始布线 $connections$，可以拔除任意两台直接相连的计算机之间的网线，并用这根网线连接任意一对未直接连接的计算机。

**要求**：计算并返回使所有计算机都连通所需的最少操作次数。如果不可能，则返回 $-1$。

**说明**：

- $1 \le n \le 10^5$。
- $1 \le connections.length \le min( \frac{n \times (n-1)}{2}, 10^5)$。
- $connections[i].length == 2$。
- $0 \le connections[i][0], connections[i][1] < n$。
- $connections[i][0] != connections[i][1]$。
- 没有重复的连接。
- 两台计算机不会通过多条线缆连接。

**示例**：

- 示例 1：

![](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2020/01/11/sample_1_1677.png)

```python
输入：n = 4, connections = [[0,1],[0,2],[1,2]]
输出：1
解释：拔下计算机 1 和 2 之间的线缆，并将它插到计算机 1 和 3 上。
```

- 示例 2：

![](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2020/01/11/sample_2_1677.png)

```python
输入：n = 6, connections = [[0,1],[0,2],[0,3],[1,2],[1,3]]
输出：2
```

## 解题思路

### 思路 1：并查集

$n$ 台计算机至少需要 $n - 1$ 根线才能进行连接，如果网线的数量少于 $n - 1$，那么就不可能将其连接。接下来计算最少操作次数。

把 $n$ 台计算机看做是 $n$ 个节点，每条网线看做是一条无向边。维护两个变量：多余电线数 $removeCount$、需要电线数 $needConnectCount$。初始 $removeCount = 1, needConnectCount = n - 1$。

遍历网线数组，将相连的节点 $a$ 和 $b$ 利用并查集加入到一个集合中（调用 `union` 操作）。

- 如果 $a$ 和 $b$ 已经在同一个集合中，说明该连接线多余，多余电线数加 $1$。
- 如果 $a$ 和 $b$ 不在一个集合中，则将其合并，则 $a$ 和 $b$ 之间不再需要用额外的电线连接了，所以需要电线数减 $1$。

最后，判断多余的电线数是否满足需要电线数，不满足返回 $-1$，如果满足，则返回需要电线数。

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
    def makeConnected(self, n: int, connections: List[List[int]]) -> int:
        union_find = UnionFind(n)
        removeCount = 0
        needConnectCount = n - 1
        for connection in connections:
            if union_find.union(connection[0], connection[1]):
                needConnectCount -= 1
            else:
                removeCount += 1

        if removeCount < needConnectCount:
            return -1
        return needConnectCount
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(m \times \alpha(n))$，其中 $m$ 是数组 $connections$ 的长度，$\alpha$ 是反 `Ackerman` 函数。
- **空间复杂度**：$O(n)$。

# [1324. 竖直打印单词](https://leetcode.cn/problems/print-words-vertically/)

- 标签：数组、字符串、模拟
- 难度：中等

## 题目链接

- [1324. 竖直打印单词 - 力扣](https://leetcode.cn/problems/print-words-vertically/)

## 题目大意

**描述**：给定一个字符串 $s$。

**要求**：按照单词在 $s$ 中出现顺序将它们全部竖直返回。

**说明**：

- 单词应该以字符串列表的形式返回，必要时用空格补位，但输出尾部的空格需要删除（不允许尾随空格）。
- 每个单词只能放在一列上，每一列中也只能有一个单词。
- $1 \le s.length \le 200$。
- $s$ 仅含大写英文字母。
- 题目数据保证两个单词之间只有一个空格。

**示例**：

- 示例 1：

```python
输入：s = "HOW ARE YOU"
输出：["HAY","ORO","WEU"]
解释：每个单词都应该竖直打印。 
 "HAY"
 "ORO"
 "WEU"
```

- 示例 2：

```python
输入：s = "TO BE OR NOT TO BE"
输出：["TBONTB","OEROOE","   T"]
解释：题目允许使用空格补位，但不允许输出末尾出现空格。
"TBONTB"
"OEROOE"
"   T"
```

## 解题思路

### 思路 1：模拟

1. 将字符串 $s$ 按空格分割为单词数组 $words$。
2. 计算出单词数组 $words$ 中单词的最大长度 $max\underline{\hspace{0.5em}}len$。
3. 第一重循环遍历竖直单词的每个单词位置 $i$，第二重循环遍历当前第 $j$ 个单词。
   1. 如果当前单词没有第 $i$ 个字符（当前单词的长度超过了单词位置 $i$），则将空格插入到竖直单词中。
   2. 如果当前单词有第 $i$ 个字符，泽讲当前单词的第 $i$ 个字符插入到竖直单词中。
4. 第二重循环遍历完，将竖直单词去除尾随空格，并加入到答案数组中。
5. 第一重循环遍历完，则返回答案数组。

### 思路 1：代码

```Python
class Solution:
    def printVertically(self, s: str) -> List[str]:
        words = s.split(' ')
        max_len = 0
        for word in words:
            max_len = max(len(word), max_len)

        res = []
        for i in range(max_len):
            ans = ""
            for j in range(len(words)):
                if i + 1 > len(words[j]):
                    ans += ' '
                else:
                    ans += words[j][i]
            res.append(ans.rstrip())
        
        return res
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(n \times max(|word|))$，其中 $n$ 为字符串 $s$ 中的单词个数，$max(|word|)$ 是最长的单词长度。。
- **空间复杂度**：$O(n \times max(|word|))$。

- [1338. 数组大小减半](https://leetcode.cn/problems/reduce-array-size-to-the-half/)

- 标签：贪心、数组、哈希表、排序、堆（优先队列）
- 难度：中等

## 题目链接

- [1338. 数组大小减半 - 力扣](https://leetcode.cn/problems/reduce-array-size-to-the-half/)

## 题目大意

**描述**：给定过一个整数数组 $arr$。你可以从中选出一个整数集合，并在数组 $arr$ 删除所有整数集合对应的数。

**要求**：返回至少能删除数组中的一半整数的整数集合的最小大小。

**说明**：

- $1 \le arr.length \le 10^5$。
- $arr.length$ 为偶数。
- $1 \le arr[i] \le 10^5$。

**示例**：

- 示例 1：

```python
输入：arr = [3,3,3,3,5,5,5,2,2,7]
输出：2
解释：选择 {3,7} 使得结果数组为 [5,5,5,2,2]、长度为 5（原数组长度的一半）。
大小为 2 的可行集合有 {3,5},{3,2},{5,2}。
选择 {2,7} 是不可行的，它的结果数组为 [3,3,3,3,5,5,5]，新数组长度大于原数组的二分之一。
```

- 示例 2：

```python
输入：arr = [7,7,7,7,7,7]
输出：1
解释：我们只能选择集合 {7}，结果数组为空。
```

## 解题思路

### 思路 1：贪心算法

对于选出的整数集合中每一个数 $x$ 来说，我们会删除数组 $arr$ 中所有值为 $x$ 的整数。

因为题目要求我们选出的整数集合最小，所以在每一次选择整数 $x$ 加入整数集合时，我们都应该选择数组 $arr$ 中出现次数最多的数。

因此，我们可以统计出数组 $arr$ 中每个整数的出现次数，用哈希表存储，并依照出现次数进行降序排序。

然后，依次选择出现次数最多的数进行删除，并统计个数，直到删除了至少一半的数时停止。

最后，将统计个数作为答案返回。

### 思路 1：代码

```Python
class Solution:
    def minSetSize(self, arr: List[int]) -> int:
        cnts = Counter(arr)
        ans, cnt = 0, 0
        for num, freq in cnts.most_common():
            cnt += freq
            ans += 1
            if cnt * 2 >= len(arr):
                break

        return ans
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(n \times \log n)$，其中 $n$ 为数组 $arr$ 的长度。
- **空间复杂度**：$O(n)$。

# [1343. 大小为 K 且平均值大于等于阈值的子数组数目](https://leetcode.cn/problems/number-of-sub-arrays-of-size-k-and-average-greater-than-or-equal-to-threshold/)

- 标签：数组、滑动窗口
- 难度：中等

## 题目链接

- [1343. 大小为 K 且平均值大于等于阈值的子数组数目 - 力扣](https://leetcode.cn/problems/number-of-sub-arrays-of-size-k-and-average-greater-than-or-equal-to-threshold/)

## 题目大意

**描述**：给定一个整数数组 $arr$ 和两个整数 $k$ 和 $threshold$。

**要求**：返回长度为 $k$ 且平均值大于等于 $threshold$ 的子数组数目。

**说明**：

- $1 \le arr.length \le 10^5$。
- $1 \le arr[i] \le 10^4$。
- $1 \le k \le arr.length$。
- $0 \le threshold \le 10^4$。

**示例**：

- 示例 1：

```python
输入：arr = [2,2,2,2,5,5,5,8], k = 3, threshold = 4
输出：3
解释：子数组 [2,5,5],[5,5,5] 和 [5,5,8] 的平均值分别为 4，5 和 6 。其他长度为 3 的子数组的平均值都小于 4 （threshold 的值)。
```

- 示例 2：

```python
输入：arr = [11,13,17,23,29,31,7,5,2,3], k = 3, threshold = 5
输出：6
解释：前 6 个长度为 3 的子数组平均值都大于 5 。注意平均值不是整数。
```

## 解题思路

### 思路 1：滑动窗口（固定长度）

这道题目是典型的固定窗口大小的滑动窗口题目。窗口大小为 `k`。具体做法如下：

1. `ans` 用来维护答案数目。`window_sum` 用来维护窗口中元素的和。
2. `left` 、`right` 都指向序列的第一个元素，即：`left = 0`，`right = 0`。
3. 向右移动 `right`，先将 `k` 个元素填入窗口中。
4. 当窗口元素个数为 `k` 时，即：`right - left + 1 >= k` 时，判断窗口内的元素和平均值是否大于等于阈值 `threshold`。
   1. 如果满足，则答案数目 + 1。
   2. 然后向右移动 `left`，从而缩小窗口长度，即 `left += 1`，使得窗口大小始终保持为 `k`。
5. 重复 3 ~ 4 步，直到 `right` 到达数组末尾。
6. 最后输出答案数目。

### 思路 1：代码

```python
class Solution:
    def numOfSubarrays(self, arr: List[int], k: int, threshold: int) -> int:
        left = 0
        right = 0
        window_sum = 0
        ans = 0

        while right < len(arr):
            window_sum += arr[right]
            
            if right - left + 1 >= k:
                if window_sum >= k * threshold:
                    ans += 1
                window_sum -= arr[left]
                left += 1

            right += 1

        return ans
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(n)$。
- **空间复杂度**：$O(n)$。

# [1344. 时钟指针的夹角](https://leetcode.cn/problems/angle-between-hands-of-a-clock/)

- 标签：数学
- 难度：中等

## 题目链接

- [1344. 时钟指针的夹角 - 力扣](https://leetcode.cn/problems/angle-between-hands-of-a-clock/)

## 题目大意

**描述**：给定两个数 $hour$ 和 $minutes$。

**要求**：请你返回在时钟上，由给定时间的时针和分针组成的较小角的角度（$60$ 单位制）。

**说明**：

- $1 \le hour \le 12$。
- $0 \le minutes \le 59$。
- 与标准答案误差在 $10^{-5}$ 以内的结果都被视为正确结果。

**示例**：

- 示例 1：

![](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2020/02/08/sample_1_1673.png)

```python
输入：hour = 12, minutes = 30
输出：165
```

- 示例 2：

![](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2020/02/08/sample_2_1673.png)

```python
输入：hour = 3, minutes = 30
输出；75
```

## 解题思路

### 思路 1：数学

1. 我们以 $00:00$ 为基准，分别计算出分针与 $00:00$ 中垂线的夹角，以及时针与 $00:00$ 中垂线的夹角。
2. 然后计算出两者差值的绝对值 $diff$。当前差值可能为较小的角（小于 $180°$ 的角），也可能为较大的角（大于等于 $180°$ 的角）。
3. 将差值的绝对值 $diff$ 与 $360 - diff$ 进行比较，取较小值作为答案。

### 思路 1：代码

```Python
class Solution:
    def angleClock(self, hour: int, minutes: int) -> float:
        mins_angle = 6 * minutes
        hours_angle = (hour % 12 + minutes / 60) * 30

        diff = abs(hours_angle - mins_angle)
        return min(diff, 360 - diff)
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(1)$。
- **空间复杂度**：$O(1)$。

# [1347. 制造字母异位词的最小步骤数](https://leetcode.cn/problems/minimum-number-of-steps-to-make-two-strings-anagram/)

- 标签：哈希表、字符串、计数
- 难度：中等

## 题目链接

- [1347. 制造字母异位词的最小步骤数 - 力扣](https://leetcode.cn/problems/minimum-number-of-steps-to-make-two-strings-anagram/)

## 题目大意

**描述**：给定两个长度相等的字符串 $s$ 和 $t$。每一个步骤中，你可以选择将 $t$ 中任一个字符替换为另一个字符。

**要求**：返回使 $t$ 成为 $s$ 的字母异位词的最小步骤数。

**说明**：

- **字母异位词**：指字母相同，但排列不同（也可能相同）的字符串。
- $1 \le s.length \le 50000$。
- $s.length == t.length$。
- $s$ 和 $t$ 只包含小写英文字母。

**示例**：

- 示例 1：

```python
输出：s = "bab", t = "aba"
输出：1
提示：用 'b' 替换 t 中的第一个 'a'，t = "bba" 是 s 的一个字母异位词。
```

- 示例 2：

```python
输出：s = "leetcode", t = "practice"
输出：5
提示：用合适的字符替换 t 中的 'p', 'r', 'a', 'i' 和 'c'，使 t 变成 s 的字母异位词。
```

## 解题思路

### 思路 1：哈希表

题目要求使 $t$ 成为 $s$ 的字母异位词，则只需要 $t$ 和 $s$ 对应的每种字符数量相一致即可，无需考虑字符位置。

因为每一次转换都会减少一个字符，并增加另一个字符。

1. 我们使用两个哈希表 $cnts\underline{\hspace{0.5em}}s$、$cnts\underline{\hspace{0.5em}}t$ 分别对 $t$ 和 $s$ 中的字符进行计数，并求出两者的交集。
2. 遍历交集中的字符种类，以及对应的字符数量。
3. 对于当前字符 $key$，如果当前字符串 $s$ 中的字符 $key$ 的数量小于字符串 $t$ 中字符 $key$ 的数量，即 $cnts\underline{\hspace{0.5em}}s[key] < cnts\underline{\hspace{0.5em}}t[key]$。则 $s$ 中需要补齐的字符数量就是需要的最小步数，将其累加到答案中。
4.  遍历完返回答案。

### 思路 1：代码

```Python
class Solution:
    def minSteps(self, s: str, t: str) -> int:
        cnts_s, cnts_t = Counter(s), Counter(t)
        cnts = cnts_s | cnts_t

        ans = 0
        for key, cnt in cnts.items():
            if cnts_s[key] < cnts_t[key]:
                ans += cnts_t[key] - cnts_s[key]

        return ans
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(m + n)$，其中 $m$、$n$ 分别为字符串 $s$、$t$ 的长度。
- **空间复杂度**：$O(|\sum|)$，其中 $\sum$ 是字符集，本题中 $| \sum | = 26$。

# [1349. 参加考试的最大学生数](https://leetcode.cn/problems/maximum-students-taking-exam/)

- 标签：位运算、数组、动态规划、状态压缩、矩阵
- 难度：困难

## 题目链接

- [1349. 参加考试的最大学生数 - 力扣](https://leetcode.cn/problems/maximum-students-taking-exam/)

## 题目大意

**描述**：给定一个 $m \times n$ 大小的矩阵 $seats$ 表示教室中的座位分布，其中如果座位是坏的（不可用），就用 `'#'` 表示，如果座位是好的，就用 `'.'` 表示。

学生可以看到左侧、右侧、左上方、右上方这四个方向上紧邻他的学生答卷，但是看不到直接坐在他前面或者后面的学生答卷。

**要求**：计算并返回该考场可以容纳的一期参加考试且无法作弊的最大学生人数。

**说明**：

- 学生必须坐在状况良好的座位上。
- $seats$ 只包含字符 `'.'` 和 `'#'`。
- $m == seats.length$。
- $n == seats[i].length$。
- $1 \le m \le 8$。
- $1 \le n \le 8$。

**示例**：

- 示例 1：

![](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2020/02/09/image.png)

```python
输入：seats = [["#",".","#","#",".","#"],
              [".","#","#","#","#","."],
              ["#",".","#","#",".","#"]]
输出：4
解释：教师可以让 4 个学生坐在可用的座位上，这样他们就无法在考试中作弊。
```

- 示例 2：

```python
输入：seats = [[".","#"],
              ["#","#"],
              ["#","."],
              ["#","#"],
              [".","#"]]
输出：3
解释：让所有学生坐在可用的座位上。
```

## 解题思路

### 思路 1：状态压缩 DP

题目中给定的 $m$、$n$ 范围为 $1 \le m, n \le 8$，每一排最多有 $8$ 个座位，那么我们可以使用一个 $8$ 位长度的二进制数来表示当前排座位的选择情况（也就是「状态压缩」的方式）。

同时从题目中可以看出，当前排的座位与当前行左侧、右侧座位有关，并且也与上一排中左上方、右上方的座位有关，则我们可以使用一个二维数组来表示状态。其中第一维度为排数，第二维度为当前排的座位选择情况。

具体做法如下：

###### 1. 划分阶段

按照排数、当前排的座位选择情况进行阶段划分。

###### 2. 定义状态

定义状态 $dp[i][state]$ 表示为：前 $i$ 排，并且最后一排座位选择状态为 $state$ 时，可以参加考试的最大学生数。

###### 3. 状态转移方程

因为学生可以看到左侧、右侧、左上方、右上方这四个方向上紧邻他的学生答卷，所以对于当前排的某个座位来说，其左侧、右侧、左上方、右上方都不应有人坐。我们可以根据当前排的座位选取状态 $cur\underline{\hspace{0.5em}}state$，并通过枚举的方式，找出符合要求的上一排座位选取状态 $pre\underline{\hspace{0.5em}}state$，并计算出当前排座位选择个数，即 $f(cur\underline{\hspace{0.5em}}state)$，则状态转移方程为：

 $dp[i][state] = \max \lbrace dp[i - 1][pre\underline{\hspace{0.5em}}state] \rbrace  + f(state)$ 

因为所给座位中还有坏座位（不可用）的情况，我们可以使用一个 $8$ 位的二进制数 $bad\underline{\hspace{0.5em}}seat$ 来表示当前排的坏座位情况，如果 $cur\underline{\hspace{0.5em}}state  \text{ \& } bad\underline{\hspace{0.5em}}seat == 1$，则说明当前状态下，选择了坏椅子，则可直接跳过这种状态。

我们还可以通过 $cur\underline{\hspace{0.5em}}state  \text{ \& }  (cur\underline{\hspace{0.5em}}state \text{ <}\text{< } 1)$ 和 $cur\underline{\hspace{0.5em}}state \& (cur\underline{\hspace{0.5em}}state \text{ >}\text{> } 1)$ 来判断当前排选择状态下，左右相邻座位上是否有人，如果有人，则可直接跳过这种状态。

同理，我们还可以通过 $cur\underline{\hspace{0.5em}}state  \text{ \& } (pre\underline{\hspace{0.5em}}state \text{ <}\text{< } 1)$ 和 $cur\underline{\hspace{0.5em}}state \text{ \& } (pre\underline{\hspace{0.5em}}state \text{ >}\text{> } 1)$ 来判断当前排选择状态下，上一行左上、右上相邻座位上是否有人，如果有人，则可直接跳过这种状态。

###### 4. 初始条件

- 默认情况下，前 $0$ 排所有选择状态下，可以参加考试的最大学生数为 $0$。

###### 5. 最终结果

根据我们之前定义的状态，$dp[i][state]$ 表示为：前 $i$ 排，并且最后一排座位选择状态为 $state$ 时，可以参加考试的最大学生数。 所以最终结果为最后一排 $dp[rows]$ 中的最大值。

### 思路 1：代码

```python
class Solution:
    def maxStudents(self, seats: List[List[str]]) -> int:
        rows, cols = len(seats), len(seats[0])
        states = 1 << cols
        dp = [[0 for _ in range(states)] for _ in range(rows + 1)]

        for i in range(1, rows + 1):                    # 模拟 1 ~ rows 排分配座位
            bad_seat = 0                                # 当前排的坏座位情况
            for j in range(cols):
                if seats[i - 1][j] == '#':              # 记录坏座位情况
                    bad_seat |= 1 << j

            for cur_state in range(states):             # 枚举当前排的座位选取状态
                if cur_state & bad_seat:                # 当前排的座位选择了换座位，跳过
                    continue
                if cur_state & (cur_state << 1):        # 当前排左侧座位有人，跳过
                    continue
                if cur_state & (cur_state >> 1):        # 当前排右侧座位有人，跳过
                    continue

                count = bin(cur_state).count('1')       # 计算当前排最多可以坐多少人
                for pre_state in range(states):         # 枚举前一排情况
                    if cur_state & (pre_state << 1):    # 左上座位有人，跳过
                        continue
                    if cur_state & (pre_state >> 1):    # 右上座位有人，跳过
                        continue
                    # dp[i][cur_state] 取自上一排分配情况为 pre_state 的最大值 + 当前排最多可以坐的人数
                    dp[i][cur_state] = max(dp[i][cur_state], dp[i - 1][pre_state] + count)

        return max(dp[rows])
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(m \times 2^{2n})$，其中 $m$、$n$ 分别为所给矩阵的行数、列数。
- **空间复杂度**：$O(m \times 2^n)$。

# [1358. 包含所有三种字符的子字符串数目](https://leetcode.cn/problems/number-of-substrings-containing-all-three-characters/)

- 标签：哈希表、字符串、滑动窗口
- 难度：中等

## 题目链接

- [1358. 包含所有三种字符的子字符串数目 - 力扣](https://leetcode.cn/problems/number-of-substrings-containing-all-three-characters/)

## 题目大意

给你一个字符串 `s` ，`s` 只包含三种字符 `a`, `b` 和 `c`。

请你返回 `a`，`b` 和 `c` 都至少出现过一次的子字符串数目。

## 解题思路

只要找到首个 `a`、`b`、`c` 同时存在的子字符串，则在该子字符串后面追加字符构成的新字符串还是满足题意的。假设该子串末尾字母的位置为 `i`，则以此字符串构建的新字符串有 `len(s) - i`个。所以题目可以转换为找出 `a`、`b`、`c` 同时存在的最短子串，并记录所有满足题意的字符串数量。具体做法如下：

用滑动窗口 `window` 来记录各个字符个数，`window` 为哈希表类型。用 `ans` 来维护 `a`，`b` 和 `c` 都至少出现过一次的子字符串数目。

设定两个指针：`left`、`right`，分别指向滑动窗口的左右边界，保证窗口中不超过 `k` 种字符。

- 一开始，`left`、`right` 都指向 `0`。
- 将最右侧字符 `s[right]` 加入当前窗口 `window_counts` 中，记录该字符个数，向右移动 `right`。
- 如果该窗口中字符的种数大于等于 `3` 种，即 `len(window) >= 3`，则累积答案个数为 `len(s) - right`，并不断右移 `left`，缩小滑动窗口长度，并更新窗口中对应字符的个数，直到 `len(window) < 3`。
- 然后继续右移 `right`，直到 `right >= len(nums)` 结束。
- 输出答案 `ans`。

## 代码

```python
class Solution:
    def numberOfSubstrings(self, s: str) -> int:
        window = dict()
        ans = 0
        left, right = 0, 0

        while right < len(s):
            if s[right] in window:
                window[s[right]] += 1
            else:
                window[s[right]] = 1

            while len(window) >= 3:
                ans += len(s) - right
                window[s[left]] -= 1
                if window[s[left]] == 0:
                    del window[s[left]]
                left += 1
            right += 1
        return ans
```

# [1362. 最接近的因数](https://leetcode.cn/problems/closest-divisors/)

- 标签：数学
- 难度：中等

## 题目链接

- [1362. 最接近的因数 - 力扣](https://leetcode.cn/problems/closest-divisors/)

## 题目大意

**描述**：给定一个整数 $num$。

**要求**：找出同时满足下面全部要求的两个整数：

- 两数乘积等于 $num + 1$ 或 $num + 2$。
- 以绝对差进行度量，两数大小最接近。

你可以按照任意顺序返回这两个整数。

**说明**：

- $1 \le num \le 10^9$。

**示例**：

- 示例 1：

```python
输入：num = 8
输出：[3,3]
解释：对于 num + 1 = 9，最接近的两个因数是 3 & 3；对于 num + 2 = 10, 最接近的两个因数是 2 & 5，因此返回 3 & 3。
```

- 示例 2：

```python
输入：num = 123
输出：[5,25]
```

## 解题思路

### 思路 1：数学

对于整数的任意一个范围在 $[\sqrt{n}, n]$ 的因数而言，一定存在一个范围在 $[1, \sqrt{n}]$ 的因数与其对应。因此，我们在遍历整数因数时，我们只需遍历 $[1, \sqrt{n}]$ 范围内的因数即可。

则这道题的具体解题步骤如下：

1. 对于整数 $num + 1$、从 $\sqrt{num + 1}$ 的位置开始，到 $1$ 为止，以递减的顺序在 $[1, \sqrt{num + 1}]$ 范围内找到最接近的小因数 $a1$，并根据 $num // a1$ 获得另一个因数 $a2$。
2. 用同样的方式，对于整数 $num + 2$、从 $\sqrt{num + 2}$ 的位置开始，到 $1$ 为止，以递减的顺序在 $[1, \sqrt{num + 2}]$ 范围内找到最接近的小因数 $b1$，并根据 $num // b1$ 获得另一个因数 $b2$。
3. 判断 $abs(a1 - a2)$ 与 $abs(b1 - b2)$ 的大小，返回差值绝对值较小的一对因子数作为答案。

### 思路 1：代码

```Python
class Solution:
    def disassemble(self, num):
        for i in range(int(sqrt(num) + 1), 1, -1):
            if num % i == 0:
                return (i, num // i)
        return (1, num)

    def closestDivisors(self, num: int) -> List[int]:
        a1, a2 = self.disassemble(num + 1)
        b1, b2 = self.disassemble(num + 2)
        if abs(a1 - a2) <= abs(b1 - b2):
            return [a1, a2]
        return [b1, b2]
```

### 思路 1：复杂度分析

- **时间复杂度**：$(\sqrt{n})$。
- **空间复杂度**：$O(1)$。

# [1381. 设计一个支持增量操作的栈](https://leetcode.cn/problems/design-a-stack-with-increment-operation/)

- 标签：栈、设计、数组
- 难度：中等

## 题目链接

- [1381. 设计一个支持增量操作的栈 - 力扣](https://leetcode.cn/problems/design-a-stack-with-increment-operation/)

## 题目大意

**要求**：设计一个支持对其元素进行增量操作的栈。

实现自定义栈类 $CustomStack$：

- `CustomStack(int maxSize)`：用 $maxSize$ 初始化对象，$maxSize$ 是栈中最多能容纳的元素数量。
- `void push(int x)`：如果栈还未增长到 $maxSize$，就将 $x$ 添加到栈顶。
- `int pop()`：弹出栈顶元素，并返回栈顶的值，或栈为空时返回 $-1$。
- `void inc(int k, int val)`：栈底的 $k$ 个元素的值都增加 $val$。如果栈中元素总数小于 $k$，则栈中的所有元素都增加 $val$。

**说明**：

- $1 \le maxSize, x, k \le 1000$。
- $0 \le val \le 100$。
- 每种方法 `increment`，`push` 以及 `pop` 分别最多调用 $1000$ 次。

**示例**：

- 示例 1：

```python
输入：
["CustomStack","push","push","pop","push","push","push","increment","increment","pop","pop","pop","pop"]
[[3],[1],[2],[],[2],[3],[4],[5,100],[2,100],[],[],[],[]]
输出：
[null,null,null,2,null,null,null,null,null,103,202,201,-1]
解释：
CustomStack stk = new CustomStack(3); // 栈是空的 []
stk.push(1);                          // 栈变为 [1]
stk.push(2);                          // 栈变为 [1, 2]
stk.pop();                            // 返回 2 --> 返回栈顶值 2，栈变为 [1]
stk.push(2);                          // 栈变为 [1, 2]
stk.push(3);                          // 栈变为 [1, 2, 3]
stk.push(4);                          // 栈仍然是 [1, 2, 3]，不能添加其他元素使栈大小变为 4
stk.increment(5, 100);                // 栈变为 [101, 102, 103]
stk.increment(2, 100);                // 栈变为 [201, 202, 103]
stk.pop();                            // 返回 103 --> 返回栈顶值 103，栈变为 [201, 202]
stk.pop();                            // 返回 202 --> 返回栈顶值 202，栈变为 [201]
stk.pop();                            // 返回 201 --> 返回栈顶值 201，栈变为 []
stk.pop();                            // 返回 -1 --> 栈为空，返回 -1
```

## 解题思路

### 思路 1：模拟

1. 初始化：
   1. 使用空数组 $stack$ 用于表示栈。
   2. 使用 $size$ 用于表示当前栈中元素个数，
   3. 使用 $maxSize$ 用于表示栈中允许的最大元素个数。
   4. 使用另一个空数组 $increments$ 用于增量操作。
2. `push(x)` 操作：
   1. 判断当前元素个数与栈中允许的最大元素个数关系。
   2. 如果当前元素个数小于栈中允许的最大元素个数，则：
      1. 将 $x$ 添加到数组 $stack$ 中，即：`self.stack.append(x)`。
      2. 当前元素个数加 $1$，即：`self.size += 1`。
      3. 将 $0$ 添加到增量数组 $increments$  中，即：`self.increments.append(0)`。
3. `increment(k, val)` 操作：
   1. 如果增量数组不为空，则取 $k$ 与元素个数 `self.size` 的较小值，令增量数组对应位置加上 `val`（等 `pop()` 操作时，再计算出准确值）。
4. `pop()` 操作：
   1. 如果当前元素个数为 $0$，则直接返回 $-1$。
   2. 如果当前元素个数大于等于 $2$，则更新弹出元素后的增量数组（保证剩余元素弹出时能够正确计算出），即：`self.increments[-2] += self.increments[-1]`
   3. 令元素个数减 $1$，即：`self.size -= 1`。
   4. 弹出数组 $stack$ 中的栈顶元素和增量数组 $increments$ 中的栈顶元素，令其相加，即为弹出元素值，将其返回。

### 思路 1：代码

```python
class CustomStack:

    def __init__(self, maxSize: int):
        self.maxSize = maxSize
        self.stack = []
        self.increments = []
        self.size = 0


    def push(self, x: int) -> None:
        if self.size < self.maxSize:
            self.stack.append(x)
            self.increments.append(0)
            self.size += 1


    def pop(self) -> int:
        if self.size == 0:
            return -1
        if self.size >= 2:
            self.increments[-2] += self.increments[-1]
        self.size -= 1
        
        val = self.stack.pop() + self.increments.pop()
        return val


    def increment(self, k: int, val: int) -> None:
        if self.increments:
            self.increments[min(k, self.size) - 1] += val



# Your CustomStack object will be instantiated and called as such:
# obj = CustomStack(maxSize)
# obj.push(x)
# param_2 = obj.pop()
# obj.increment(k,val)
```

### 思路 1：复杂度分析

- **时间复杂度**：初始化、`push` 操作、`pop` 操作、`increment` 操作的时间复杂度为 $O(1)$。
- **空间复杂度**：$O(maxSize)$。
# [1400. 构造 K 个回文字符串](https://leetcode.cn/problems/construct-k-palindrome-strings/)

- 标签：贪心、哈希表、字符串、计数
- 难度：中等

## 题目链接

- [1400. 构造 K 个回文字符串 - 力扣](https://leetcode.cn/problems/construct-k-palindrome-strings/)

## 题目大意

**描述**：给定一个字符串 $s$ 和一个整数 $k$。

**要求**：用 $s$ 字符串中所有字符构造 $k$ 个非空回文串。如果可以用 $s$ 中所有字符构造 $k$ 个回文字符串，那么请你返回 `True`，否则返回 `False`。

**说明**：

- $1 \le s.length \le 10^5$。
- $s$ 中所有字符都是小写英文字母。
- $1 \le k \le 10^5$。

**示例**：

- 示例 1：

```python
输入：s = "annabelle", k = 2
输出：True
解释：可以用 s 中所有字符构造 2 个回文字符串。
一些可行的构造方案包括："anna" + "elble"，"anbna" + "elle"，"anellena" + "b"
```

## 解题思路

### 思路 1：贪心算法

- 用字符串 $s$ 中所有字符构造回文串最多可以构造 $len(s)$ 个（将每个字符当做一个回文串）。所以如果 $len(s) < k$，则说明字符数量不够，无法构成 $k$ 个回文串，直接返回 `False`。
- 如果 $len(s) == k$，则可以直接使用单个字符构建回文串，直接返回 `True`。
- 如果 $len(s) > k$，则需要判断一下字符串 $s$ 中每个字符的个数。因为当字符是偶数个时，可以直接构造成回文串。所以我们只需要考虑个数为奇数的字符即可。如果个位为奇数的字符种类小于等于 $k$，则说明可以构造 $k$ 个回文串，返回 `True`。如果个位为奇数的字符种类大于 $k$，则说明无法构造 $k$ 个回文串，返回 `Fasle`。

### 思路 1：贪心算法代码

```python
import collections

class Solution:
    def canConstruct(self, s: str, k: int) -> bool:
        size = len(s)
        if size < k:
            return False
        if size == k:
            return True
        letter_dict = dict()
        for i in range(size):
            if s[i] in letter_dict:
                letter_dict[s[i]] += 1
            else:
                letter_dict[s[i]] = 1

        odd = 0
        for key in letter_dict:
            if letter_dict[key] % 2 == 1:
               odd += 1
        return odd <= k
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(n + |\sum|)$，其中 $n$ 为字符串 $s$ 的长度，$\sum$ 是字符集，本题中 $|\sum| = 26$。
- **空间复杂度**：$O(|\sum|)$。
# [1408. 数组中的字符串匹配](https://leetcode.cn/problems/string-matching-in-an-array/)

- 标签：数组、字符串、字符串匹配
- 难度：简单

## 题目链接

- [1408. 数组中的字符串匹配 - 力扣](https://leetcode.cn/problems/string-matching-in-an-array/)

## 题目大意

**描述**：给定一个字符串数组 `words`，数组中的每个字符串都可以看作是一个单词。如果可以删除 `words[j]` 最左侧和最右侧的若干字符得到 `word[i]`，那么字符串 `words[i]` 就是 `words[j]` 的一个子字符串。

**要求**：按任意顺序返回 `words` 中是其他单词的子字符串的所有单词。

**说明**：

- $1 \le words.length \le 100$。
- $1 \le words[i].length \le 30$
- `words[i]` 仅包含小写英文字母。
- 题目数据保证每个 `words[i]` 都是独一无二的。

**示例**：

- 示例 1：

```python
输入：words = ["mass","as","hero","superhero"]
输出：["as","hero"]
解释："as" 是 "mass" 的子字符串，"hero" 是 "superhero" 的子字符串。此外，["hero","as"] 也是有效的答案。
```

## 解题思路

### 思路 1：KMP 算法

1. 先按照字符串长度从小到大排序，使用数组 `res` 保存答案。
2. 使用两重循环遍历，对于 `words[i]` 和 `words[j]`，使用 `KMP` 匹配算法，如果 `wrods[j]` 包含 `words[i]`，则将其加入到答案数组中，并跳出最里层循环。
3. 返回答案数组 `res`。

### 思路 1：代码

```python
class Solution:
    # 生成 next 数组
    # next[j] 表示下标 j 之前的模式串 p 中，最长相等前后缀的长度
    def generateNext(self, p: str):
        m = len(p)
        next = [0 for _ in range(m)]                # 初始化数组元素全部为 0
        
        left = 0                                    # left 表示前缀串开始所在的下标位置
        for right in range(1, m):                   # right 表示后缀串开始所在的下标位置
            while left > 0 and p[left] != p[right]: # 匹配不成功, left 进行回退, left == 0 时停止回退
                left = next[left - 1]               # left 进行回退操作
            if p[left] == p[right]:                 # 匹配成功，找到相同的前后缀，先让 left += 1，此时 left 为前缀长度
                left += 1
            next[right] = left                      # 记录前缀长度，更新 next[right], 结束本次循环, right += 1

        return next

    # KMP 匹配算法，T 为文本串，p 为模式串
    def kmp(self, T: str, p: str) -> int:
        n, m = len(T), len(p)
        
        next = self.generateNext(p)                      # 生成 next 数组
        
        j = 0                                       # j 为模式串中当前匹配的位置
        for i in range(n):                          # i 为文本串中当前匹配的位置
            while j > 0 and T[i] != p[j]:           # 如果模式串前缀匹配不成功, 将模式串进行回退, j == 0 时停止回退
                j = next[j - 1]
            if T[i] == p[j]:                        # 当前模式串前缀匹配成功，令 j += 1，继续匹配
                j += 1
            if j == m:                              # 当前模式串完全匹配成功，返回匹配开始位置
                return i - j + 1
        return -1                                   # 匹配失败，返回 -1
        
    def stringMatching(self, words: List[str]) -> List[str]:
        words.sort(key=lambda x:len(x))

        res = []
        for i in range(len(words) - 1):
            for j in range(i + 1, len(words)):
                if self.kmp(words[j], words[i]) != -1:
                    res.append(words[i])           
                    break
        return res
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(n^2 \times m)$，其中字符串数组长度为 $n$，字符串数组中最长字符串长度为 $m$。
- **空间复杂度**：$O(m)$。
# [1422. 分割字符串的最大得分](https://leetcode.cn/problems/maximum-score-after-splitting-a-string/)

- 标签：字符串
- 难度：简单

## 题目链接

- [1422. 分割字符串的最大得分 - 力扣](https://leetcode.cn/problems/maximum-score-after-splitting-a-string/)

## 题目大意

**描述**：给定一个由若干 $0$ 和 $1$ 组成的字符串。将字符串分割成两个非空子字符串的得分为：左子字符串中 $0$ 的数量 + 右子字符串中 $1$ 的数量。

**要求**：计算并返回该字符串分割成两个非空子字符串（即左子字符串和右子字符串）所能获得的最大得分。

**说明**：

- $2 \le s.length \le 500$。
- 字符串 $s$ 仅由字符 $0$ 和 $1$ 组成。

**示例**：

- 示例 1：

```python
输入：s = "011101"
输出：5 
解释：
将字符串 s 划分为两个非空子字符串的可行方案有：
左子字符串 = "0" 且 右子字符串 = "11101"，得分 = 1 + 4 = 5 
左子字符串 = "01" 且 右子字符串 = "1101"，得分 = 1 + 3 = 4 
左子字符串 = "011" 且 右子字符串 = "101"，得分 = 1 + 2 = 3 
左子字符串 = "0111" 且 右子字符串 = "01"，得分 = 1 + 1 = 2 
左子字符串 = "01110" 且 右子字符串 = "1"，得分 = 2 + 1 = 3
```

- 示例 2：

```python
输入：s = "00111"
输出：5
解释：当 左子字符串 = "00" 且 右子字符串 = "111" 时，我们得到最大得分 = 2 + 3 = 5
```

## 解题思路

### 思路 1：前缀和

1. 遍历字符串 $s$，使用前缀和数组来记录每个前缀子字符串中 $1$ 的个数。
2. 再次遍历字符串 $s$，枚举每个分割点，利用前缀和数组计算出当前分割出的左子字符串中 $1$ 的个数与右子字符串中 $0$ 的个数，并计算当前得分，然后更新最大得分。
3. 返回最大得分作为答案。

### 思路 1：代码

```python
class Solution:
    def maxScore(self, s: str) -> int:
        size = len(s)
        one_cnts = [0 for _ in range(size + 1)]

        for i in range(1, size + 1):
            if s[i - 1] == '1':
                one_cnts[i] = one_cnts[i - 1] + 1
            else:
                one_cnts[i] = one_cnts[i - 1]

        ans = 0
        for i in range(1, size):
            left_score = i - one_cnts[i]
            right_score = one_cnts[size] - one_cnts[i]
            ans = max(ans, left_score + right_score)
        
        return ans
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(n)$，其中 $n$ 为字符串 $s$ 的长度。
- **空间复杂度**：$O(n)$。
# [1423. 可获得的最大点数](https://leetcode.cn/problems/maximum-points-you-can-obtain-from-cards/)

- 标签：数组、前缀和、滑动窗口
- 难度：中等

## 题目链接

- [1423. 可获得的最大点数 - 力扣](https://leetcode.cn/problems/maximum-points-you-can-obtain-from-cards/)

## 题目大意

**描述**：将卡牌排成一行，给定每张卡片的点数数组 $cardPoints$，其中 $cardPoints[i]$ 表示第 $i$ 张卡牌对应点数。

每次行动，可以从行的开头或者末尾拿一张卡牌，最终保证正好拿到了 $k$ 张卡牌。所得点数就是你拿到手中的所有卡牌的点数之和。

现在给定一个整数数组 $cardPoints$ 和整数 $k$。

**要求**：返回可以获得的最大点数。

**说明**：

- $1 \le cardPoints.length \le 10^5$。
- $1 \le cardPoints[i] \le 10^4$
- $1 \le k \le cardPoints.length$。

**示例**：

- 示例 1：

```python
输入：cardPoints = [1,2,3,4,5,6,1], k = 3
输出：12
解释：第一次行动，不管拿哪张牌，你的点数总是 1 。但是，先拿最右边的卡牌将会最大化你的可获得点数。最优策略是拿右边的三张牌，最终点数为 1 + 6 + 5 = 12。
```

- 示例 2：

```python
输入：cardPoints = [2,2,2], k = 2
输出：4
解释：无论你拿起哪两张卡牌，可获得的点数总是 4。
```

## 解题思路

### 思路 1：滑动窗口

可以用固定长度的滑动窗口来做。

由于只能从开头或末尾位置拿 $k$ 张牌，则最后剩下的肯定是连续的 $len(cardPoints) - k$ 张牌。要求求出 $k$ 张牌可以获得的最大收益，我们可以反向先求出连续 $len(cardPoints) - k$ 张牌的最小点数。则答案为 $sum(cardPoints) - min\underline{\hspace{0.5em}}sum$。维护一个固定长度为 $len(cardPoints) - k$ 的滑动窗口，求最小和。具体做法如下：

1. $window\underline{\hspace{0.5em}}sum$ 用来维护窗口内的元素和，初始值为 $0$。$min\underline{\hspace{0.5em}}sum$ 用来维护滑动窗口元素的最小和。初始值为 $sum(cardPoints)$。滑动窗口的长度为 $window\underline{\hspace{0.5em}}size$，值为 $len(cardPoints) - k$。
2. 使用双指针 $left$、$right$。$left$ 、$right$ 都指向序列的第一个元素，即：`left = 0`，`right = 0`。
3. 向右移动 $right$，先将 $window\underline{\hspace{0.5em}}size$ 个元素填入窗口中。
4. 当窗口元素个数为 $window\underline{\hspace{0.5em}}size$ 时，即：$right - left + 1 \ge window\underline{\hspace{0.5em}}size$ 时，计算窗口内的元素和，并维护子数组最小和 $min\underline{\hspace{0.5em}}sum$。
5. 然后向右移动 $left$，从而缩小窗口长度，即 `left += 1`，使得窗口大小始终保持为 $k$。
6. 重复 4 ~ 5 步，直到 $right$ 到达数组末尾。
7. 最后输出 $sum(cardPoints) - min\underline{\hspace{0.5em}}sum$ 即为答案。

注意：如果 $window\underline{\hspace{0.5em}}size$ 为 $0$ 时需要特殊判断，此时答案为数组和 $sum(cardPoints)$。

### 思路 1：代码

```python
class Solution:
    def maxScore(self, cardPoints: List[int], k: int) -> int:
        window_size = len(cardPoints) - k
        window_sum = 0
        cards_sum = sum(cardPoints)
        min_sum = cards_sum

        left, right = 0, 0
        if window_size == 0:
            return cards_sum

        while right < len(cardPoints):
            window_sum += cardPoints[right]

            if right - left + 1 >= window_size:
                min_sum = min(window_sum, min_sum)
                window_sum -= cardPoints[left]
                left += 1

            right += 1

        return cards_sum - min_sum
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(n)$，其中 $n$ 为数组 $cardPoints$ 中的元素数量。
- **空间复杂度**：$O(1)$。

# [1438. 绝对差不超过限制的最长连续子数组](https://leetcode.cn/problems/longest-continuous-subarray-with-absolute-diff-less-than-or-equal-to-limit/)

- 标签：队列、数组、有序集合、滑动窗口、单调队列、堆（优先队列）
- 难度：中等

## 题目链接

- [1438. 绝对差不超过限制的最长连续子数组 - 力扣](https://leetcode.cn/problems/longest-continuous-subarray-with-absolute-diff-less-than-or-equal-to-limit/)

## 题目大意

给定一个整数数组 `nums`，和一个表示限制的整数 `limit`。

要求：返回最长连续子数组的长度，该子数组中的任意两个元素之间的绝对差必须小于或者等于 `limit`。

如果不存在满足条件的子数组，则返回 `0`。

## 解题思路

求最长连续子数组，可以使用滑动窗口来解决。这道题目的难点在于如何维护滑动窗口内的最大值和最小值的差值。遍历滑动窗口求最大值和最小值，每次计算的时间复杂度为 $O(k)$，时间复杂度过高。考虑使用特殊的数据结构来降低时间复杂度。可以使用堆（优先队列）来解决。这里使用 `Python` 中 `heapq` 实现。具体做法如下：

- 使用 `left`、`right` 两个指针，分别指向滑动窗口的左右边界，保证窗口中最大值和最小值的差值不超过 `limit`。
- 一开始，`left`、`right` 都指向 `0`。
- 向右移动 `right`，将最右侧元素加入当前窗口和大顶堆、小顶堆中。
- 如果大顶堆堆顶元素和小顶堆堆顶元素大于 `limit`，则不断右移 `left`，缩小滑动窗口长度，并更新窗口内的大顶堆、小顶堆。
- 如果大顶堆堆顶元素和小顶堆堆顶元素小于等于 `limit`，则更新最长连续子数组长度。
- 然后继续右移 `right`，直到 `right >= len(nums)` 结束。
- 输出答案。

## 代码

```python
import heapq

class Solution:
    def longestSubarray(self, nums: List[int], limit: int) -> int:
        size = len(nums)
        heap_max = []
        heap_min = []

        ans = 0
        left, right = 0, 0
        while right < size:
            heapq.heappush(heap_max, [-nums[right], right])
            heapq.heappush(heap_min, [nums[right], right])

            while -heap_max[0][0] - heap_min[0][0] > limit:
                while heap_min[0][1] <= left:
                    heapq.heappop(heap_min)
                while heap_max[0][1] <= left:
                    heapq.heappop(heap_max)
                left += 1
            ans = max(ans, right - left + 1)
            right += 1

        return ans
```

# [1446. 连续字符](https://leetcode.cn/problems/consecutive-characters/)

- 标签：字符串
- 难度：简单

## 题目链接

- [1446. 连续字符 - 力扣](https://leetcode.cn/problems/consecutive-characters/)

## 题目大意

给你一个字符串 `s` ，字符串的「能量」定义为：只包含一种字符的最长非空子字符串的长度。

要求：返回字符串的能量。

注意：

- `1 <= s.length <= 500`
- `s` 只包含小写英文字母。

## 解题思路

使用 `count` 统计连续不重复子串的长度，使用 `ans` 记录最长连续不重复子串的长度。

## 代码

```python
class Solution:
    def maxPower(self, s: str) -> int:
        ans = 1
        count = 1
        for i in range(1, len(s)):
            if s[i] == s[i - 1]:
                count += 1
            else:
                count = 1
            ans = max(ans, count)
        return ans
```

# [1447. 最简分数](https://leetcode.cn/problems/simplified-fractions/)

- 标签：数学、字符串、数论
- 难度：中等

## 题目链接

- [1447. 最简分数 - 力扣](https://leetcode.cn/problems/simplified-fractions/)

## 题目大意

**描述**：给定一个整数 $n$。

**要求**：返回所有 $0$ 到 $1$ 之间（不包括 $0$ 和 $1$）满足分母小于等于 $n$ 的最简分数。分数可以以任意顺序返回。

**说明**：

- $1 \le n \le 100$。

**示例**：

- 示例 1：

```python
输入：n = 2
输出：["1/2"]
解释："1/2" 是唯一一个分母小于等于 2 的最简分数。
```

- 示例 2：

```python
输入：n = 4
输出：["1/2","1/3","1/4","2/3","3/4"]
解释："2/4" 不是最简分数，因为它可以化简为 "1/2"。
```

## 解题思路

### 思路 1：数学

如果分子和分母的最大公约数为 $1$ 时，则当前分数为最简分数。

而 $n$ 的数据范围为 $(1, 100)$。因此我们可以使用两重遍历，分别枚举分子和分母，然后通过判断分子和分母是否为最大公约数，来确定当前分数是否为最简分数。

### 思路 1：代码

```python
class Solution:
    def simplifiedFractions(self, n: int) -> List[str]:
        res = []

        for i in range(1, n):
            for j in range(i + 1, n + 1):
                if math.gcd(i, j) == 1:
                    res.append(str(i) + "/" + str(j))

        return res
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(n^2 \times \log n)$。
- **空间复杂度**：$O(1)$。

# [1449. 数位成本和为目标值的最大数字](https://leetcode.cn/problems/form-largest-integer-with-digits-that-add-up-to-target/)

- 标签：数组、动态规划
- 难度：困难

## 题目链接

- [1449. 数位成本和为目标值的最大数字 - 力扣](https://leetcode.cn/problems/form-largest-integer-with-digits-that-add-up-to-target/)

## 题目大意

**描述**：给定一个整数数组 $cost$ 和一个整数 $target$。现在从 `""` 开始，不断通过以下规则得到一个新的整数：

1. 给当前结果添加一个数位（$i + 1$）的成本为 $cost[i]$（$cost$ 数组下标从 $0$ 开始）。
2. 总成本必须恰好等于 $target$。
3. 添加的数位中没有数字 $0$。

**要求**：找到按照上述规则可以得到的最大整数。

**说明**：

- 由于答案可能会很大，请你以字符串形式返回。
- 如果按照上述要求无法得到任何整数，请你返回 `"0"`。
- $cost.length == 9$。
- $1 \le cost[i] \le 5000$。
- $1 \le target \le 5000$。

**示例**：

- 示例 1：

```python
输入：cost = [4,3,2,5,6,7,2,5,5], target = 9
输出："7772"
解释：添加数位 '7' 的成本为 2 ，添加数位 '2' 的成本为 3 。所以 "7772" 的代价为 2*3+ 3*1 = 9 。 "977" 也是满足要求的数字，但 "7772" 是较大的数字。
 数字     成本
  1  ->   4
  2  ->   3
  3  ->   2
  4  ->   5
  5  ->   6
  6  ->   7
  7  ->   2
  8  ->   5
  9  ->   5
```

- 示例 2：

```python
输入：cost = [7,6,5,5,5,6,8,7,8], target = 12
输出："85"
解释：添加数位 '8' 的成本是 7 ，添加数位 '5' 的成本是 5 。"85" 的成本为 7 + 5 = 12。
 数字     成本
  1  ->   7
  2  ->   6
  3  ->   5
  4  ->   5
  5  ->   5
  6  ->   6
  7  ->   8
  8  ->   7
  9  ->   8
```

## 解题思路

把每个数位（$1 \sim 9$）看做是一件物品，$cost[i]$ 看做是物品的重量，一共有无数件物品可以使用，$target$ 看做是背包的载重上限，得到的最大整数可以看做是背包的最大价值。那么问题就变为了「完全背包问题」中的「恰好装满背包的最大价值问题」。

因为答案可能会很大，要求以字符串形式返回。这里我们可以直接令 $dp[w]$ 为字符串形式，然后定义一个 `def maxInt(a, b):`  方法用于判断两个字符串代表的数字大小。

### 思路 1：动态规划

###### 1. 划分阶段

按照背包载重上限进行阶段划分。

###### 2. 定义状态

定义状态 $dp[w]$ 表示为：将物品装入一个最多能装重量为 $w$ 的背包中，恰好装满背包的情况下，能装入背包的最大整数。

###### 3. 状态转移方程

$dp[w] = maxInt(dp[w], str(i) + dp[w - cost[i - 1]])$

###### 4. 初始条件

1. 只有载重上限为 $0$ 的背包，在不放入物品时，能够恰好装满背包（有合法解），此时背包所含物品的最大价值为空字符串，即 `dp[0] = ""`。
2. 其他载重上限下的背包，在放入物品的时，都不能恰好装满背包（都没有合法解），此时背包所含物品的最大价值属于未定义状态，值为自定义字符 `"#"`，即 ，`dp[w] = "#"`，$0 \le w \le target$。

###### 5. 最终结果

根据我们之前定义的状态，$dp[w]$ 表示为：将物品装入一个最多能装重量为 $w$ 的背包中，恰好装满背包的情况下，能装入背包的最大价值总和。 所以最终结果为 $dp[target]$。

### 思路 1：代码

```python
class Solution:
    def largestNumber(self, cost: List[int], target: int) -> str:
        def maxInt(a, b):
            if len(a) == len(b):
                return max(a, b)
            if len(a) > len(b):
                return a
            return b

        size = len(cost)
        dp = ["#" for _ in range(target + 1)]
        dp[0] = ""

        for i in range(1, size + 1):
            for w in range(cost[i - 1], target + 1):
                if dp[w - cost[i - 1]] != "#":
                    dp[w] = maxInt(dp[w], str(i) + dp[w - cost[i - 1]])
        if dp[target] == "#":
            return "0"
        return dp[target]
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(n \times target)$，其中 $n$ 为数组 $cost$ 的元素个数，$target$ 为所给整数。
- **空间复杂度**：$O(target)$。
# [1450. 在既定时间做作业的学生人数](https://leetcode.cn/problems/number-of-students-doing-homework-at-a-given-time/)

- 标签：数组
- 难度：简单

## 题目链接

- [1450. 在既定时间做作业的学生人数 - 力扣](https://leetcode.cn/problems/number-of-students-doing-homework-at-a-given-time/)

## 题目大意

**描述**：给你两个长度相等的整数数组，一个表示开始时间的数组 $startTime$ ，另一个表示结束时间的数组 $endTime$。再给定一个整数 $queryTime$ 作为查询时间。已知第 $i$ 名学生在 $startTime[i]$ 时开始写作业并于 $endTime[i]$ 时完成作业。

**要求**：返回在查询时间 $queryTime$ 时正在做作业的学生人数。即能够使 $queryTime$ 处于区间 $[startTime[i], endTime[i]]$ 的学生人数。

**说明**：

- $startTime.length == endTime.length$。
- $1\le startTime.length \le 100$。
- $1 \le startTime[i] \le endTime[i] \le 1000$。
- $1 \le queryTime \le 1000$。

**示例**：

- 示例 1：

```python
输入：startTime = [4], endTime = [4], queryTime = 4
输出：1
解释：在查询时间只有一名学生在做作业。
```

## 解题思路

### 思路 1：枚举算法

- 维护一个用于统计在查询时间 $queryTime$ 时正在做作业的学生人数的变量 $cnt$。然后遍历所有学生的开始时间和结束时间。
- 如果 $queryTime$ 在区间 $[startTime[i], endTime[i]]$ 之间，即 $startTime[i] <= queryTime <= endTime[i]$，则令 $cnt$ 加 $1$。
- 遍历完输出统计人数 $cnt$。

### 思路 1：枚举算法代码

```python
class Solution:
    def busyStudent(self, startTime: List[int], endTime: List[int], queryTime: int) -> int:
        cnt = 0
        size = len(startTime)
        for i in range(size):
            if startTime[i] <= queryTime <= endTime[i]:
                cnt += 1
        return cnt
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(n)$，其中 $n$ 为数组中的元素个数。
- **空间复杂度**：$O(1)$。

### 思路 2：线段树

- 因为 $1 \le startTime[i] \le endTime[i] \le 1000$，所以我们可以维护一个区间为 $[0, 1000]$ 的线段树，初始化所有区间值都为 $0$。
- 然后遍历所有学生的开始时间和结束时间，并将区间 $[startTime[i], endTime[i]]$ 值加 $1$。
- 在线段树中查询 $queryTime$ 对应的单点区间 $[queryTime, queryTime]$ 的最大值为多少。

### 思路 2：线段树代码

```python
# 线段树的节点类
class SegTreeNode:
    def __init__(self, val=0):
        self.left = -1                              # 区间左边界
        self.right = -1                             # 区间右边界
        self.val = val                              # 节点值（区间值）
        self.lazy_tag = None                        # 区间和问题的延迟更新标记
        
        
# 线段树类
class SegmentTree:
    # 初始化线段树接口
    def __init__(self, nums, function):
        self.size = len(nums)
        self.tree = [SegTreeNode() for _ in range(4 * self.size)]  # 维护 SegTreeNode 数组
        self.nums = nums                            # 原始数据
        self.function = function                    # function 是一个函数，左右区间的聚合方法
        if self.size > 0:
            self.__build(0, 0, self.size - 1)
    
    # 单点更新接口：将 nums[i] 更改为 val
    def update_point(self, i, val):
        self.nums[i] = val
        self.__update_point(i, val, 0)
    
    # 区间更新接口：将区间为 [q_left, q_right] 上的所有元素值加上 val
    def update_interval(self, q_left, q_right, val):
        self.__update_interval(q_left, q_right, val, 0)
        
    # 区间查询接口：查询区间为 [q_left, q_right] 的区间值
    def query_interval(self, q_left, q_right):
        return self.__query_interval(q_left, q_right, 0)
    
    # 获取 nums 数组接口：返回 nums 数组
    def get_nums(self):
        for i in range(self.size):
            self.nums[i] = self.query_interval(i, i)
        return self.nums
        
        
    # 以下为内部实现方法
    
    # 构建线段树实现方法：节点的存储下标为 index，节点的区间为 [left, right]
    def __build(self, index, left, right):
        self.tree[index].left = left
        self.tree[index].right = right
        if left == right:                           # 叶子节点，节点值为对应位置的元素值
            self.tree[index].val = self.nums[left]
            return
    
        mid = left + (right - left) // 2            # 左右节点划分点
        left_index = index * 2 + 1                  # 左子节点的存储下标
        right_index = index * 2 + 2                 # 右子节点的存储下标
        self.__build(left_index, left, mid)         # 递归创建左子树
        self.__build(right_index, mid + 1, right)   # 递归创建右子树
        self.__pushup(index)                        # 向上更新节点的区间值
    
    # 单点更新实现方法：将 nums[i] 更改为 val，节点的存储下标为 index
    def __update_point(self, i, val, index):
        left = self.tree[index].left
        right = self.tree[index].right
        
        if left == right:
            self.tree[index].val = val              # 叶子节点，节点值修改为 val
            return
        
        mid = left + (right - left) // 2            # 左右节点划分点
        left_index = index * 2 + 1                  # 左子节点的存储下标
        right_index = index * 2 + 2                 # 右子节点的存储下标
        if i <= mid:                                # 在左子树中更新节点值
            self.__update_point(i, val, left_index)
        else:                                       # 在右子树中更新节点值
            self.__update_point(i, val, right_index)
        
        self.__pushup(index)                        # 向上更新节点的区间值
    
    # 区间更新实现方法
    def __update_interval(self, q_left, q_right, val, index):
        left = self.tree[index].left
        right = self.tree[index].right
        
        if left >= q_left and right <= q_right:     # 节点所在区间被 [q_left, q_right] 所覆盖        
            if self.tree[index].lazy_tag is not None:
                self.tree[index].lazy_tag += val    # 将当前节点的延迟标记增加 val
            else:
                self.tree[index].lazy_tag = val     # 将当前节点的延迟标记增加 val
            interval_size = (right - left + 1)      # 当前节点所在区间大小
            self.tree[index].val += val * interval_size  # 当前节点所在区间每个元素值增加 val
            return
        
        if right < q_left or left > q_right:        # 节点所在区间与 [q_left, q_right] 无关
            return
    
        self.__pushdown(index)                      # 向下更新节点的区间值
    
        mid = left + (right - left) // 2            # 左右节点划分点
        left_index = index * 2 + 1                  # 左子节点的存储下标
        right_index = index * 2 + 2                 # 右子节点的存储下标
        if q_left <= mid:                           # 在左子树中更新区间值
            self.__update_interval(q_left, q_right, val, left_index)
        if q_right > mid:                           # 在右子树中更新区间值
            self.__update_interval(q_left, q_right, val, right_index)
        
        self.__pushup(index)                        # 向上更新节点的区间值
    
    # 区间查询实现方法：在线段树中搜索区间为 [q_left, q_right] 的区间值
    def __query_interval(self, q_left, q_right, index):
        left = self.tree[index].left
        right = self.tree[index].right
        
        if left >= q_left and right <= q_right:     # 节点所在区间被 [q_left, q_right] 所覆盖
            return self.tree[index].val             # 直接返回节点值
        if right < q_left or left > q_right:        # 节点所在区间与 [q_left, q_right] 无关
            return 0
    
        self.__pushdown(index)
    
        mid = left + (right - left) // 2            # 左右节点划分点
        left_index = index * 2 + 1                  # 左子节点的存储下标
        right_index = index * 2 + 2                 # 右子节点的存储下标
        res_left = 0                                # 左子树查询结果
        res_right = 0                               # 右子树查询结果
        if q_left <= mid:                           # 在左子树中查询
            res_left = self.__query_interval(q_left, q_right, left_index)
        if q_right > mid:                           # 在右子树中查询
            res_right = self.__query_interval(q_left, q_right, right_index)
        
        return self.function(res_left, res_right)   # 返回左右子树元素值的聚合计算结果
    
    # 向上更新实现方法：更新下标为 index 的节点区间值 等于 该节点左右子节点元素值的聚合计算结果
    def __pushup(self, index):
        left_index = index * 2 + 1                  # 左子节点的存储下标
        right_index = index * 2 + 2                 # 右子节点的存储下标
        self.tree[index].val = self.function(self.tree[left_index].val, self.tree[right_index].val)

    # 向下更新实现方法：更新下标为 index 的节点所在区间的左右子节点的值和懒惰标记
    def __pushdown(self, index):
        lazy_tag = self.tree[index].lazy_tag
        if lazy_tag is None: 
            return
        
        left_index = index * 2 + 1                  # 左子节点的存储下标
        right_index = index * 2 + 2                 # 右子节点的存储下标
        
        if self.tree[left_index].lazy_tag is not None:
            self.tree[left_index].lazy_tag += lazy_tag  # 更新左子节点懒惰标记
        else:
            self.tree[left_index].lazy_tag = lazy_tag
        left_size = (self.tree[left_index].right - self.tree[left_index].left + 1)
        self.tree[left_index].val += lazy_tag * left_size   # 左子节点每个元素值增加 lazy_tag
        
        if self.tree[right_index].lazy_tag is not None:
            self.tree[right_index].lazy_tag += lazy_tag # 更新右子节点懒惰标记
        else:
            self.tree[right_index].lazy_tag = lazy_tag
        right_size = (self.tree[right_index].right - self.tree[right_index].left + 1)
        self.tree[right_index].val += lazy_tag * right_size # 右子节点每个元素值增加 lazy_tag
        
        self.tree[index].lazy_tag = None            # 更新当前节点的懒惰标记


class Solution:
    def busyStudent(self, startTime: List[int], endTime: List[int], queryTime: int) -> int:
        nums = [0 for _ in range(1010)]
        self.STree = SegmentTree(nums, lambda x, y: max(x, y))
        size = len(startTime)
        for i in range(size):
            self.STree.update_interval(startTime[i], endTime[i], 1)

        return self.STree.query_interval(queryTime, queryTime)
```

### 思路 2：复杂度分析

- **时间复杂度**：$O(n \times \log n)$，其中 $n$ 为数组元素的个数。
- **空间复杂度**：$O(n)$。

### 思路 3：树状数组

- 因为 $1 \le startTime[i] \le endTime[i] \le 1000$，所以我们可以维护一个区间为 $[0, 1000]$ 的树状数组。
- 注意：
  - 树状数组中 $update(self, index, delta):$ 指的是将对应元素 $nums[index] $ 加上 $delta$。
  - $query(self, index):$ 指的是 $index$ 位置之前的元素和，即前缀和。
- 然后遍历所有学生的开始时间和结束时间，将树状数组上 $startTime[i]$ 的值增加 $1$，再将树状数组上$endTime[i]$ 的值减少 $1$。
- 则查询 $queryTime$ 位置的前缀和即为答案。

### 思路 3：树状数组代码

```python
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
    def busyStudent(self, startTime: List[int], endTime: List[int], queryTime: int) -> int:
        bit = BinaryIndexTree(1010)
        size = len(startTime)
        for i in range(size):
            bit.update(startTime[i], 1)
            bit.update(endTime[i] + 1, -1)
        return bit.query(queryTime)
```

### 思路 3：复杂度分析

- **时间复杂度**：$O(n \times \log n)$，其中 $n$ 为数组元素的个数。
- **空间复杂度**：$O(n)$。
# [1451. 重新排列句子中的单词](https://leetcode.cn/problems/rearrange-words-in-a-sentence/)

- 标签：字符串、排序
- 难度：中等

## 题目链接

- [1451. 重新排列句子中的单词 - 力扣](https://leetcode.cn/problems/rearrange-words-in-a-sentence/)

## 题目大意

**描述**：「句子」是一个用空格分隔单词的字符串。给定一个满足下述格式的句子 $text$:

- 句子的首字母大写。
- $text$ 中的每个单词都用单个空格分隔。

**要求**：重新排列 $text$ 中的单词，使所有单词按其长度的升序排列。如果两个单词的长度相同，则保留其在原句子中的相对顺序。

请同样按上述格式返回新的句子。

**说明**：

- $text$ 以大写字母开头，然后包含若干小写字母以及单词间的单个空格。
- $1 \le text.length \le 10^5$。

**示例**：

- 示例 1：

```python
输入：text = "Leetcode is cool"
输出："Is cool leetcode"
解释：句子中共有 3 个单词，长度为 8 的 "Leetcode" ，长度为 2 的 "is" 以及长度为 4 的 "cool"。
输出需要按单词的长度升序排列，新句子中的第一个单词首字母需要大写。
```

- 示例 2：

```python
输入：text = "Keep calm and code on"
输出："On and keep calm code"
解释：输出的排序情况如下：
"On" 2 个字母。
"and" 3 个字母。
"keep" 4 个字母，因为存在长度相同的其他单词，所以它们之间需要保留在原句子中的相对顺序。
"calm" 4 个字母。
"code" 4 个字母。
```

## 解题思路

### 思路 1：模拟

1. 将 $text$ 按照 `" "` 进行分割为单词数组 $words$。
2. 将单词数组按照「单词长度」进行升序排序。
3. 将单词数组用 `" "` 连接起来，并将首字母转为大写字母，其他字母转为小写字母，将结果存入答案字符串 $ans$ 中。
4. 返回答案字符串 $ans$。

### 思路 1：代码

```Python
class Solution:
    def arrangeWords(self, text: str) -> str:
        words = text.split(' ')
        words.sort(key=lambda word:len(word))
        ans = " ".join(words).capitalize()

        return ans
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(n \times \log n)$，其中 $n$ 为字符串 $text$ 的长度。
- **空间复杂度**：$O(n)$。

# [1456. 定长子串中元音的最大数目](https://leetcode.cn/problems/maximum-number-of-vowels-in-a-substring-of-given-length/)

- 标签：字符串、滑动窗口
- 难度：中等

## 题目链接

- [1456. 定长子串中元音的最大数目 - 力扣](https://leetcode.cn/problems/maximum-number-of-vowels-in-a-substring-of-given-length/)

## 题目大意

**描述**：给定字符串 $s$ 和整数 $k$。

**要求**：返回字符串 $s$ 中长度为 $k$ 的单个子字符串中可能包含的最大元音字母数。

**说明**：

- 英文中的元音字母为（$a$, $e$, $i$, $o$, $u$）。
- $1 <= s.length <= 10^5$。
- $s$ 由小写英文字母组成。
- $1 <= k <= s.length$。

**示例**：

- 示例 1：

```python
输入：s = "abciiidef", k = 3
输出：3
解释：子字符串 "iii" 包含 3 个元音字母。
```

- 示例 2：

```python
输入：s = "aeiou", k = 2
输出：2
解释：任意长度为 2 的子字符串都包含 2 个元音字母。
```

## 解题思路

### 思路 1：滑动窗口

固定长度的滑动窗口题目。维护一个长度为 $k$ 的窗口，并统计滑动窗口中最大元音字母数。具体做法如下：

1. $ans$ 用来维护长度为 $k$ 的单个字符串中最大元音字母数。$window\underline{\hspace{0.5em}}count$ 用来维护窗口中元音字母数。集合 $vowel\underline{\hspace{0.5em}}set$ 用来存储元音字母。
2. $left$ 、$right$ 都指向字符串 $s$ 的第一个元素，即：$left = 0$，$right = 0$。
3. 判断 $s[right]$ 是否在元音字母集合中，如果在则用 $window\underline{\hspace{0.5em}}count$ 进行计数。
4. 当窗口元素个数为 $k$ 时，即：$right - left + 1 \ge k$ 时，更新 $ans$。然后判断 $s[left]$ 是否为元音字母，如果是则 `window_count -= 1`，并向右移动 $left$，从而缩小窗口长度，即 `left += 1`，使得窗口大小始终保持为 $k$。
5. 重复 $3 \sim 4$ 步，直到 $right$ 到达数组末尾。
6. 最后输出 $ans$。

### 思路 1：代码

```python
class Solution:
    def maxVowels(self, s: str, k: int) -> int:
        left, right = 0, 0
        ans = 0
        window_count = 0
        vowel_set = ('a','e','i','o','u')

        while right < len(s):
            if s[right] in vowel_set:
                window_count += 1

            if right - left + 1 >= k:
                ans = max(ans, window_count)
                if s[left] in vowel_set:
                    window_count -= 1
                left += 1

            right += 1
        return ans
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(n)$，其中 $n$ 为字符串 $s$ 的长度。
- **空间复杂度**：$O(1)$。

# [1476. 子矩形查询](https://leetcode.cn/problems/subrectangle-queries/)

- 标签：设计、数组、矩阵
- 难度：中等

## 题目链接

- [1476. 子矩形查询 - 力扣](https://leetcode.cn/problems/subrectangle-queries/)

## 题目大意

**要求**：实现一个类 SubrectangleQueries，它的构造函数的参数是一个 $rows \times cols $的矩形（这里用整数矩阵表示），并支持以下两种操作：

1. `updateSubrectangle(int row1, int col1, int row2, int col2, int newValue)`：用 $newValue$ 更新以 $(row1,col1)$ 为左上角且以 $(row2,col2)$ 为右下角的子矩形。

2. `getValue(int row, int col)`：返回矩形中坐标 (row,col) 的当前值。

**说明**：

- 最多有 $500$ 次 `updateSubrectangle` 和 `getValue` 操作。
- $1 <= rows, cols <= 100$。
- $rows == rectangle.length$。
- $cols == rectangle[i].length$。
- $0 <= row1 <= row2 < rows$。
- $0 <= col1 <= col2 < cols$。
- $1 <= newValue, rectangle[i][j] <= 10^9$。
- $0 <= row < rows$。
- $0 <= col < cols$。

**示例**：

- 示例 1：

```python
输入：
["SubrectangleQueries","getValue","updateSubrectangle","getValue","getValue","updateSubrectangle","getValue","getValue"]
[[[[1,2,1],[4,3,4],[3,2,1],[1,1,1]]],[0,2],[0,0,3,2,5],[0,2],[3,1],[3,0,3,2,10],[3,1],[0,2]]
输出：
[null,1,null,5,5,null,10,5]
解释：
SubrectangleQueries subrectangleQueries = new SubrectangleQueries([[1,2,1],[4,3,4],[3,2,1],[1,1,1]]);  
// 初始的 (4x3) 矩形如下：
// 1 2 1
// 4 3 4
// 3 2 1
// 1 1 1
subrectangleQueries.getValue(0, 2); // 返回 1
subrectangleQueries.updateSubrectangle(0, 0, 3, 2, 5);
// 此次更新后矩形变为：
// 5 5 5
// 5 5 5
// 5 5 5
// 5 5 5 
subrectangleQueries.getValue(0, 2); // 返回 5
subrectangleQueries.getValue(3, 1); // 返回 5
subrectangleQueries.updateSubrectangle(3, 0, 3, 2, 10);
// 此次更新后矩形变为：
// 5   5   5
// 5   5   5
// 5   5   5
// 10  10  10 
subrectangleQueries.getValue(3, 1); // 返回 10
subrectangleQueries.getValue(0, 2); // 返回 5
```

- 示例 2：

```python
输入：
["SubrectangleQueries","getValue","updateSubrectangle","getValue","getValue","updateSubrectangle","getValue"]
[[[[1,1,1],[2,2,2],[3,3,3]]],[0,0],[0,0,2,2,100],[0,0],[2,2],[1,1,2,2,20],[2,2]]
输出：
[null,1,null,100,100,null,20]
解释：
SubrectangleQueries subrectangleQueries = new SubrectangleQueries([[1,1,1],[2,2,2],[3,3,3]]);
subrectangleQueries.getValue(0, 0); // 返回 1
subrectangleQueries.updateSubrectangle(0, 0, 2, 2, 100);
subrectangleQueries.getValue(0, 0); // 返回 100
subrectangleQueries.getValue(2, 2); // 返回 100
subrectangleQueries.updateSubrectangle(1, 1, 2, 2, 20);
subrectangleQueries.getValue(2, 2); // 返回 20

```

## 解题思路

### 思路 1：暴力

矩形最大为 $row \times col == 100 \times 100$，则每次更新最多需要更新 $10000$ 个值，更新次数最多为 $500$ 次。

用暴力更新的方法最多需要更新 $5000000$ 次，我们可以尝试一下用暴力更新的方法解决本题（提交后发现可以通过）。

### 思路 1：代码

```Python
class SubrectangleQueries:

    def __init__(self, rectangle: List[List[int]]):
        self.rectangle = rectangle


    def updateSubrectangle(self, row1: int, col1: int, row2: int, col2: int, newValue: int) -> None:
        for row in range(row1, row2 + 1):
            for col in range(col1, col2 + 1):
                self.rectangle[row][col] = newValue


    def getValue(self, row: int, col: int) -> int:
        return self.rectangle[row][col]
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(row \times col \times 500)$。
- **空间复杂度**：$O(row \times col)$。
# [1480. 一维数组的动态和](https://leetcode.cn/problems/running-sum-of-1d-array/)

- 标签：数组、前缀和
- 难度：简单

## 题目链接

- [1480. 一维数组的动态和 - 力扣](https://leetcode.cn/problems/running-sum-of-1d-array/)

## 题目大意

**描述**：给定一个数组 $nums$。

**要求**：返回数组 $nums$ 的动态和。

**说明**：

- **动态和**：数组前 $i$ 项元素和构成的数组，计算公式为 $runningSum[i] = \sum_{x = 0}^{x = i}(nums[i])$。
- $1 \le nums.length \le 1000$。
- $-10^6 \le nums[i] \le 10^6$。

**示例**：

- 示例 1：

```python
输入：nums = [1,2,3,4]
输出：[1,3,6,10]
解释：动态和计算过程为 [1, 1+2, 1+2+3, 1+2+3+4]。
```

- 示例 2：

```python
输入：nums = [1,1,1,1,1]
输出：[1,2,3,4,5]
解释：动态和计算过程为 [1, 1+1, 1+1+1, 1+1+1+1, 1+1+1+1+1]。
```

## 解题思路

### 思路 1：递推

根据动态和的公式 $runningSum[i] = \sum_{x = 0}^{x = i}(nums[i])$，可以推导出：

$runningSum = \begin{cases} nums[0], & i = 0 \cr runningSum[i - 1] + nums[i], & i > 0\end{cases}$

则解决过程如下：

1. 新建一个长度等于 $nums$ 的数组 $res$ 用于存放答案。
2. 初始化 $res[0] = nums[0]$。
3. 从下标 $1$ 开始遍历数组 $nums$，递推更新 $res$，即：`res[i] = res[i - 1] + nums[i]`。
4. 遍历结束，返回 $res$ 作为答案。

### 思路 1：代码

```python
class Solution:
    def runningSum(self, nums: List[int]) -> List[int]:
        size = len(nums)
        res = [0 for _ in range(size)]
        for i in range(size):
            if i == 0:
                res[i] = nums[i]
            else:
                res[i] = res[i - 1] + nums[i]
        return res
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(n)$。一重循环遍历的时间复杂度为 $O(n)$。
- **空间复杂度**：$O(n)$。如果算上答案数组的空间占用，则空间复杂度为 $O(n)$。不算上则空间复杂度为 $O(1)$。

# [1482. 制作 m 束花所需的最少天数](https://leetcode.cn/problems/minimum-number-of-days-to-make-m-bouquets/)

- 标签：数组、二分查找
- 难度：中等

## 题目链接

- [1482. 制作 m 束花所需的最少天数 - 力扣](https://leetcode.cn/problems/minimum-number-of-days-to-make-m-bouquets/)

## 题目大意

**描述**：给定一个整数数组 $bloomDay$，以及两个整数 $m$ 和 $k$。$bloomDay$ 代表花朵盛开的时间，$bloomDay[i]$ 表示第 $i$ 朵花的盛开时间。盛开后就可以用于一束花中。

现在需要制作 $m$ 束花。制作花束时，需要使用花园中相邻的 $k$ 朵花 。

**要求**：返回从花园中摘 $m$ 束花需要等待的最少的天数。如果不能摘到 $m$ 束花则返回 $-1$。

**说明**：

- $bloomDay.length == n$。
- $1 \le n \le 10^5$。
- $1 \le bloomDay[i] \le 10^9$。
- $1 \le m \le 10^6$。
- $1 \le k \le n$。

**示例**：

- 示例 1：

```python
输入：bloomDay = [1,10,3,10,2], m = 3, k = 1
输出：3
解释：让我们一起观察这三天的花开过程，x 表示花开，而 _ 表示花还未开。
现在需要制作 3 束花，每束只需要 1 朵。
1 天后：[x, _, _, _, _]   // 只能制作 1 束花
2 天后：[x, _, _, _, x]   // 只能制作 2 束花
3 天后：[x, _, x, _, x]   // 可以制作 3 束花，答案为 3
```

- 示例 2：

```python
输入：bloomDay = [1,10,3,10,2], m = 3, k = 2
输出：-1
解释：要制作 3 束花，每束需要 2 朵花，也就是一共需要 6 朵花。而花园中只有 5 朵花，无法满足制作要求，返回 -1。
```

## 解题思路

### 思路 1：二分查找算法

这道题跟「[0875. 爱吃香蕉的珂珂](https://leetcode.cn/problems/koko-eating-bananas/)」、「[1011. 在 D 天内送达包裹的能力](https://leetcode.cn/problems/capacity-to-ship-packages-within-d-days/)」有点相似。

根据题目可知：

- 制作花束最少使用时间跟花朵开花最短时间有关系，即 $min(bloomDay)$。
- 制作花束最多使用时间跟花朵开花最长时间有关系，即 $max(bloomDay)$。
- 则制作花束所需要的天数就变成了一个区间 $[min(bloomDay), max(bloomDay)]$。

那么，我们就可以根据这个区间，利用二分查找算法找到一个符合题意的最少天数。而判断某个天数下能否摘到 $m$ 束花则可以写个方法判断。具体步骤如下：

-  遍历数组 $bloomDay$。
   - 如果 $bloomDay[i] \le days$。就将花朵数量加 $1$。
     - 当能摘的花朵数等于 $k$ 时，能摘的花束数目加 $1$，花朵数量置为 $0$。
   - 如果 $bloomDay[i] > days$。就将花朵数置为 $0$。
-  最后判断能摘的花束数目是否大于等于 $m$。

整个算法的步骤如下：

- 如果 $m \times k > len(bloomDay)$，说明无法满足要求，直接返回 $-1$。
- 使用两个指针 $left$、$right$。令 $left$ 指向 $min(bloomDay)$，$right$ 指向 $max(bloomDay)$。代表待查找区间为 $[left, right]$。
- 取两个节点中心位置 $mid$，判断是否能在 $mid$ 天制作 $m$ 束花。
  - 如果不能，则将区间 $[left, mid]$ 排除掉，继续在区间 $[mid + 1, right]$ 中查找。
  - 如果能，说明天数还可以继续减少，则继续在区间 $[left, mid]$ 中查找。
- 当 $left == right$ 时跳出循环，返回 $left$。

### 思路 1：代码

```python
class Solution:
    def canMake(self, bloomDay, days, m, k):
        count = 0
        flower = 0
        for i in range(len(bloomDay)):
            if bloomDay[i] <= days:
                flower += 1
                if flower == k:
                    count += 1
                    flower = 0
            else:
                flower = 0
        return count >= m

    def minDays(self, bloomDay: List[int], m: int, k: int) -> int:
        if m > len(bloomDay) / k:
            return -1

        left, right = min(bloomDay), max(bloomDay)

        while left < right:
            mid = left + (right - left) // 2
            if not self.canMake(bloomDay, mid, m, k):
                left = mid + 1
            else:
                right = mid

        return left
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(n \times \log (max(bloomDay) - min(bloomDay)))$。
- **空间复杂度**：$O(1)$。

## 参考资料

- 【题解】[【赤小豆】为什么是二分法，思路及模板 python - 制作 m 束花所需的最少天数 - 力扣（LeetCode）](https://leetcode.cn/problems/minimum-number-of-days-to-make-m-bouquets/solution/chi-xiao-dou-python-wei-shi-yao-shi-er-f-24p7/)

# [1486. 数组异或操作](https://leetcode.cn/problems/xor-operation-in-an-array/)

- 标签：位运算、数学
- 难度：简单

## 题目链接

- [1486. 数组异或操作 - 力扣](https://leetcode.cn/problems/xor-operation-in-an-array/)

## 题目大意

给定两个整数 n、start。数组 nums 定义为：nums[i] = start + 2*i（下标从 0 开始）。n 为数组长度。返回数组 nums 中所有元素按位异或（XOR）后得到的结果。

## 解题思路

### 1. 模拟

直接按照题目要求模拟即可。

### 2. 规律

- $x \oplus x = 0$；
- $x \oplus y = y \oplus x$（交换律）；
- $(x \oplus y) \oplus z = x \oplus (y \oplus z)$（结合律）；
- $x \oplus y \oplus y = x$（自反性）；
- $\forall i \in Z$，有 $4i \oplus (4i+1) \oplus (4i+2) \oplus (4i+3) = 0$；
- $\forall i \in Z$，有 $2i \oplus (2i+1) = 1$；
- $\forall i \in Z$，有 $2i \oplus 1 = 2i+1$。

本题中计算的是 $start \oplus (start + 2) \oplus (start + 4) \oplus (start + 6) \oplus … \oplus (start+(2*(n-1)))$。

可以看出，若 start 为奇数，则 $start+2, start + 4, …, start + (2 \times(n - 1))$ 都为奇数。若 start 为偶数，则 $start + 2, start + 4, …, start + (2 \times(n - 1))$ 都为偶数。则它们对应二进制的最低位相同，则我们可以将最低位提取处理单独处理。从而将公式转换一下。

令 $s = \frac{start}{2}$，则等式变为 $(s) \oplus (s+1) \oplus (s+2) \oplus (s+3) \oplus … \oplus (s+(n-1)) * 2 + e$，e 表示运算结果的最低位。

根据自反性，$(s) \oplus (s+1) \oplus (s+2) \oplus (s+3) \oplus … \oplus (s+(n-1)) = \\ (1 \oplus 2 \oplus … \oplus (s-1)) \oplus (1 \oplus 2 \oplus … \oplus (s-1) \oplus (s) \oplus (s+1) \oplus … \oplus (s+(n-1)))$

例如： $3 \oplus 4 \oplus 5 \oplus 6 \oplus 7 = (1 \oplus 2) \oplus (1 \oplus 2 \oplus 3 \oplus 4 \oplus 5 \oplus 6 \oplus7)$

就变为了计算前 n 项序列的异或值。假设我们定义一个函数 sumXor(x) 用于计算前 n 项数的异或结果，通过观察可得出：

$sumXor(x) = \begin{cases} \begin{array} \ x, & x = 4i, k \in Z \cr (x-1) \oplus x, & x = 4i+1, k \in Z \cr (x-2) \oplus (x-1) \oplus x, & x = 4i+2, k \in Z \cr (x-3) \oplus (x-2) \oplus (x-3) \oplus x, & x = 4i+3, k \in Z \end{array} \end{cases}$

继续化简得：

$sumXor(x) = \begin{cases} \begin{array} \ x, & x = 4i, k \in Z \cr 1, & x = 4i+1, k \in Z \cr x+1, & x = 4i+2, k \in Z \cr 0, & x = 4i+3, k \in Z \end{array} \end{cases}$

则最终结果为 $sumXor(s-1) \oplus sumXor(s+n-1) * 2 + e$。

下面还有最后一位 e 的计算。

- 若 start 为偶数，则最后一位 e 为 0。
- 若 start 为奇数，最后一位 e 跟 n 有关，若 n 为奇数，则最后一位 e 为 1，若 n 为偶数，则最后一位 e 为 0。

总结下来就是 `e = start & n & 1`。

## 代码

1. 模拟

```python
class Solution:
    def xorOperation(self, n: int, start: int) -> int:
        ans = 0
        for i in range(n):
            ans ^= (start + i * 2)
        return ans
```

2. 规律

```python
class Solution:
    def sumXor(self, x):
        if x % 4 == 0:
            return x
        if x % 4 == 1:
            return 1
        if x % 4 == 2:
            return x + 1
        return 0
    def xorOperation(self, n: int, start: int) -> int:
        s = start >> 1
        e = n & start & 1
        ans = self.sumXor(s-1) ^ self.sumXor(s + n - 1)
        return ans << 1 | e
```

# [1491. 去掉最低工资和最高工资后的工资平均值](https://leetcode.cn/problems/average-salary-excluding-the-minimum-and-maximum-salary/)

- 标签：数组、排序
- 难度：简单

## 题目链接

- [1491. 去掉最低工资和最高工资后的工资平均值 - 力扣](https://leetcode.cn/problems/average-salary-excluding-the-minimum-and-maximum-salary/)

## 题目大意

**描述**：给定一个整数数组 `salary`，数组中的每一个数都是唯一的，其中 `salary[i]` 是第 `i` 个员工的工资。

**要求**：返回去掉最低工资和最高工资之后，剩下员工工资的平均值。

**说明**：

- $3 \le salary.length \le 100$。
- $10^3 \le salary[i] \le 10^6$。
- $salary[i]$ 是唯一的。
- 与真实值误差在 $10^{-5}$ 以内的结果都将视为正确答案。

**示例**：

- 示例 1：

```python
给定 salary = [1000,2000,3000]
输出 2000.00000
解释 最低工资为 1000，最高工资为 3000，去除最低工资和最高工资之后，剩下员工工资的平均值为 2000 / 1 = 2000
```

## 解题思路

### 思路 1：

因为给定 $salary.length \ge 3$，并且 $salary[i]$ 是唯一的，所以无需考虑最低工资和最高工资是同一个。接下来就是按照题意模拟过程：

- 计算出最小工资为 `min_s`，即 `min_s = min(salary)`。
- 计算出最大工资为 `max_s`，即 `max_s = max(salary)`。
- 计算出所有工资和之后再减去最小工资和最大工资，即 `total = sum(salary) - min_s - max_s`。
- 求剩下工资的平均值，并返回，即 `return total / (len(salary) - 2)`。

## 代码

### 思路 1 代码：

```python
class Solution:
    def average(self, salary: List[int]) -> float:
        min_s, max_s = min(salary), max(salary)
        total = sum(salary) - min_s - max_s
        return total / (len(salary) - 2)
```

# [1493. 删掉一个元素以后全为 1 的最长子数组](https://leetcode.cn/problems/longest-subarray-of-1s-after-deleting-one-element/)

- 标签：数组、动态规划、滑动窗口
- 难度：中等

## 题目链接

- [1493. 删掉一个元素以后全为 1 的最长子数组 - 力扣](https://leetcode.cn/problems/longest-subarray-of-1s-after-deleting-one-element/)

## 题目大意

**描述**：给定一个二进制数组 $nums$，需要从数组中删掉一个元素。

**要求**：返回最长的且只包含 $1$ 的非空子数组的长度。如果不存在这样的子数组，请返回 $0$。

**说明**：

- $1 \le nums.length \le 10^5$。
- $nums[i]$ 要么是 $0$ 要么是 $1$。

**示例**：

- 示例 1：

```python
输入：nums = [1,1,0,1]
输出：3
解释：删掉位置 2 的数后，[1,1,1] 包含 3 个 1。
```

- 示例 2：

```python
输入：nums = [0,1,1,1,0,1,1,0,1]
输出：5
解释：删掉位置 4 的数字后，[0,1,1,1,1,1,0,1] 的最长全 1 子数组为 [1,1,1,1,1]。
```

## 解题思路

### 思路 1：滑动窗口

维护一个元素值为 $0$ 的元素数量少于 $1$ 个的滑动窗口。则答案为滑动窗口长度减去窗口内 $0$ 的个数求最大值。具体做法如下：

设定两个指针：$left$、$right$，分别指向滑动窗口的左右边界，保证窗口 $0$ 的个数小于 $1$ 个。使用 $window\underline{\hspace{0.5em}}count$ 记录窗口中 $0$ 的个数，使用 $ans$ 记录删除一个元素后，最长的只包含 $1$ 的非空子数组长度。

- 一开始，$left$、$right$ 都指向 $0$。

- 如果最右侧元素等于 $0$，则 `window_count += 1` 。

- 如果 $window\underline{\hspace{0.5em}}count > 1$ ，则不断右移 $left$，缩小滑动窗口长度。并更新当前窗口中 $0$ 的个数，直到 $window\underline{\hspace{0.5em}}count \le 1$。
- 更新答案值，然后向右移动 $right$，直到 $right \ge len(nums)$ 结束。
- 输出答案 $ans$。

### 思路 1：代码

```python
class Solution:
    def longestSubarray(self, nums: List[int]) -> int:
        left, right = 0, 0
        window_count = 0
        ans = 0

        while right < len(nums):
            if nums[right] == 0:
                window_count += 1

            while window_count > 1:
                if nums[left] == 0:
                    window_count -= 1
                left += 1
            ans = max(ans, right - left + 1 - window_count)
            right += 1

        if ans == len(nums):
            return len(nums) - 1
        else:
            return ans
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(n)$，其中 $n$ 为数组 $nums$ 的长度。
- **空间复杂度**：$O(1)$。

# [1496. 判断路径是否相交](https://leetcode.cn/problems/path-crossing/)

- 标签：哈希表、字符串
- 难度：简单

## 题目链接

- [1496. 判断路径是否相交 - 力扣](https://leetcode.cn/problems/path-crossing/)

## 题目大意

**描述**：给定一个字符串 $path$，其中 $path[i]$ 的值可以是 `'N'`、`'S'`、`'E'` 或者 `'W'`，分别表示向北、向南、向东、向西移动一个单位。

你从二维平面上的原点 $(0, 0)$ 处开始出发，按 $path$ 所指示的路径行走。

**要求**：如果路径在任何位置上与自身相交，也就是走到之前已经走过的位置，请返回 $True$；否则，返回 $False$。

**说明**：

- $1 \le path.length \le 10^4$。
- $path[i]$ 为 `'N'`、`'S'`、`'E'` 或 `'W'`。

**示例**：

- 示例 1：

![](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2020/06/28/screen-shot-2020-06-10-at-123929-pm.png)

```python
输入：path = "NES"
输出：false 
解释：该路径没有在任何位置相交。
```

- 示例 2：

![](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2020/06/28/screen-shot-2020-06-10-at-123843-pm.png)

```python
输入：path = "NESWW"
输出：true
解释：该路径经过原点两次。
```

## 解题思路

### 思路 1：哈希表 + 模拟

1. 使用哈希表将 `'N'`、`'S'`、`'E'`、`'W'` 对应横纵坐标轴上的改变表示出来。
2. 使用集合 $visited$ 存储走过的坐标元组。
3. 遍历 $path$，按照 $path$ 所指示的路径模拟行走，并将所走过的坐标使用 $visited$ 存储起来。
4. 如果在 $visited$ 遇到已经走过的坐标，则返回 $True$。
5. 如果遍历完仍未发现已经走过的坐标，则返回 $False$。

### 思路 1：代码

```Python
class Solution:
    def isPathCrossing(self, path: str) -> bool:
        directions = {
            "N" : (-1, 0),
            "S" : (1, 0),
            "W" : (0, -1),
            "E" : (0, 1),
        }

        x, y = 0, 0
        
        visited = set()
        visited.add((x, y))
        
        for ch in path:
            x += directions[ch][0]
            y += directions[ch][1]
            if (x, y) in visited:
                return True
            visited.add((x, y))
        
        return False
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(n)$，其中 $n$ 为数组 $path$ 的长度。
- **空间复杂度**：$O(n)$。

# [1502. 判断能否形成等差数列](https://leetcode.cn/problems/can-make-arithmetic-progression-from-sequence/)

- 标签：数组、排序
- 难度：简单

## 题目链接

- [1502. 判断能否形成等差数列 - 力扣](https://leetcode.cn/problems/can-make-arithmetic-progression-from-sequence/)

## 题目大意

**描述**：给定一个数字数组 `arr`。如果一个数列中，任意相邻两项的差总等于同一个常数，那么这个数序就称为等差数列。

**要求**：如果数组 `arr` 通过重新排列可以形成等差数列，则返回 `True`；否则返回 `False`。

**说明**：

- $2 \le arr.length \le 1000$
- $-10^6 \le arr[i] \le 10^6$

**示例**：

- 示例 1：

```python
输入：arr = [3,5,1]
输出：True
解释：数组重新排序后得到 [1,3,5] 或者 [5,3,1]，任意相邻两项的差分别为 2 或 -2 ，可以形成等差数列。
```

## 解题思路

### 思路 1：

- 如果数组元素个数小于等于 `2`，则数组肯定可以形成等差数列，直接返回 `True`。
- 对数组进行排序。
- 从下标为 `2` 的元素开始，遍历相邻的 `3` 个元素 `arr[i]` 、`arr[i - 1]`、`arr[i - 2]`。判断 `arr[i] - arr[i - 1]` 是否等于 `arr[i - 1] - arr[i - 2]`。如果不等于，则数组无法形成等差数列，返回 `False`。
- 如果遍历完数组，则说明数组可以形成等差数列，返回 `True`。

## 代码

### 思路 1 代码：

```python
class Solution:
    def canMakeArithmeticProgression(self, arr: List[int]) -> bool:
        size = len(arr)
        if size <= 2:
            return True

        arr.sort()
        for i in range(2, size):
            if arr[i] - arr[i - 1] != arr[i - 1] - arr[i - 2]:
                return False
        return True
```

# [1507. 转变日期格式](https://leetcode.cn/problems/reformat-date/)

- 标签：字符串
- 难度：简单

## 题目链接

- [1507. 转变日期格式 - 力扣](https://leetcode.cn/problems/reformat-date/)

## 题目大意

**描述**：给定一个字符串 $date$，它的格式为 `Day Month Year` ，其中：

- $Day$ 是集合 `{"1st", "2nd", "3rd", "4th", ..., "30th", "31st"}` 中的一个元素。
- $Month$ 是集合 `{"Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"}` 中的一个元素。
- $Year$ 的范围在 $[1900, 2100]$ 之间。

**要求**：将字符串转变为 `YYYY-MM-DD` 的格式，其中：

- $YYYY$ 表示 $4$ 位的年份。
- $MM$ 表示 $2$ 位的月份。
- $DD$ 表示 $2$ 位的天数。

**说明**：

- 给定日期保证是合法的，所以不需要处理异常输入。

**示例**：

- 示例 1：

```python
输入：date = "20th Oct 2052"
输出："2052-10-20"
```

- 示例 2：

```python
输入：date = "6th Jun 1933"
输出："1933-06-06"
```

## 解题思路

### 思路 1：模拟

1. 将字符串分割为三部分，分别按照以下规则得到日、月、年：
   1. 日：去掉末尾两位英文字母，将其转为整型数字，并且进行补零操作，使其宽度为 $2$。
   2. 月：使用哈希表将其映射为对应两位数字。
   3. 年：直接赋值。
2. 将得到的年、月、日使用 `"-"` 进行链接并返回。

### 思路 1：代码

```python
class Solution:
    def reformatDate(self, date: str) -> str:
        months = {
            "Jan" : "01", "Feb" : "02", "Mar" : "03", "Apr" : "04", "May" : "05", "Jun" : "06", 
            "Jul" : "07", "Aug" : "08", "Sep" : "09", "Oct" : "10", "Nov" : "11", "Dec" : "12"
        }
        date_list = date.split(' ')
        day = "{:0>2d}".format(int(date_list[0][: -2]))
        month = months[date_list[1]]
        year = date_list[2]
        return year + "-" + month + "-" + day
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(1)$。
- **空间复杂度**：$O(1)$。

# [1523. 在区间范围内统计奇数数目](https://leetcode.cn/problems/count-odd-numbers-in-an-interval-range/)

- 标签：数学
- 难度：简单

## 题目链接

- [1523. 在区间范围内统计奇数数目 - 力扣](https://leetcode.cn/problems/count-odd-numbers-in-an-interval-range/)

## 题目大意

**描述**：给定两个非负整数 `low` 和 `high`。

**要求**：返回 `low` 与 `high` 之间（包括二者）的奇数数目。

**说明**：

- $0 \le low \le high \le 10^9$。

**示例**：

- 示例 1：

```python
输入：low = 3, high = 7
输出：3
解释：3 到 7 之间奇数数字为 [3,5,7]
```

## 解题思路

### 思路 1：

暴力枚举 `[low, high]` 之间的奇数可能会超时。我们可以通过公式直接计算出 `[0, low - 1]` 之间的奇数个数和 `[0, high]` 之间的奇数个数，然后将两者相减即为答案。

计算奇数个数的公式为：$pre(x) = \lfloor \frac{x + 1}{2} \rfloor$。

## 代码

### 思路 1 代码：

```python
class Solution:
    def pre(self, val):
        return (val + 1) >> 1

    def countOdds(self, low: int, high: int) -> int:
        return self.pre(high) - self.pre(low - 1)
```

# [1534. 统计好三元组](https://leetcode.cn/problems/count-good-triplets/)

- 标签：数组、枚举
- 难度：简单

## 题目链接

- [1534. 统计好三元组 - 力扣](https://leetcode.cn/problems/count-good-triplets/)

## 题目大意

**描述**：给定一个整数数组 $arr$，以及 $a$、$b$、$c$ 三个整数。

**要求**：统计其中好三元组的数量。

**说明**：

- **好三元组**：如果三元组（$arr[i]$、$arr[j]$、$arr[k]$）满足下列全部条件，则认为它是一个好三元组。
  - $0 \le i < j < k < arr.length$。
  - $| arr[i] - arr[j] | \le a$。
  - $| arr[j] - arr[k] | \le b$。
  - $| arr[i] - arr[k] | \le c$。

- $3 \le arr.length \le 100$。
- $0 \le arr[i] \le 1000$。
- $0 \le a, b, c \le 1000$。

**示例**：

- 示例 1：

```python
输入：arr = [3,0,1,1,9,7], a = 7, b = 2, c = 3
输出：4
解释：一共有 4 个好三元组：[(3,0,1), (3,0,1), (3,1,1), (0,1,1)]。
```

- 示例 2：

```python
输入：arr = [1,1,2,2,3], a = 0, b = 0, c = 1
输出：0
解释：不存在满足所有条件的三元组。
```

## 解题思路

### 思路 1：枚举

- 使用三重循环依次枚举所有的 $(i, j, k)$，判断对应 $arr[i]$、$arr[j]$、$arr[k]$ 是否满足条件。
- 然后统计出所有满足条件的三元组的数量。

### 思路 1：代码

```python
class Solution:
    def countGoodTriplets(self, arr: List[int], a: int, b: int, c: int) -> int:
        size = len(arr)
        ans = 0
        
        for i in range(size):
            for j in range(i + 1, size):
                for k in range(j + 1, size):
                    if abs(arr[i] - arr[j]) <= a and abs(arr[j] - arr[k]) <= b and abs(arr[i] - arr[k]) <= c:
                        ans += 1
        
        return ans
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(n^3)$，其中 $n$ 是数组 $arr$ 的长度。
- **空间复杂度**：$O(1)$。

### 思路 2：枚举优化 + 前缀和

我们可以先通过二重循环遍历二元组 $(j, k)$，找出所有满足 $| arr[j] - arr[k] | \le b$ 的二元组。

然后在 $| arr[j] - arr[k] | \le b$ 的条件下，我们需要找到满足以下要求的 $arr[i]$ 数量：

1. $i < j$。
2. $| arr[i] - arr[j] | \le a$。
3. $| arr[i] - arr[k] | \le c$。
4. $0 \le arr[i] \le 1000$。

其中 $2$、$3$ 去除绝对值之后可变为：

1. $arr[j] - a \le arr[i] \le arr[j] + a$。
2. $arr[k] - c \le arr[i] \le arr[k] + c$。

将这两个条件再结合第 $4$ 个条件综合一下就变为：$max(0, arr[j] - a, arr[k] - c) \le arr[i] \le min(arr[j] + a, arr[k] + c, 1000)$。

假如定义 $left = max(0, arr[j] - a, arr[k] - c)$，$right = min(arr[j] + a, arr[k] + c, 1000)$。

现在问题就转变了如何快速获取在值域区间 $[left, right]$ 中，有多少个 $arr[i]$。

我们可以利用前缀和数组，先计算出 $[0, 1000]$ 范围中，满足 $arr[i] < num$ 的元素个数，即为 $prefix\underline{\hspace{0.5em}}cnts[num]$。

然后对于区间 $[left, right]$，通过 $prefix\underline{\hspace{0.5em}}cnts[right] - prefix\underline{\hspace{0.5em}}cnts[left - 1]$ 即可快速求解出区间 $[left, right]$ 内 $arr[i]$ 的个数。

因为 $i < j < k$，所以我们可以在每次 $j$ 向右移动一位的时候，更新 $arr[j]$ 对应的前缀和数组，保证枚举到 $j$ 时，$prefix\underline{\hspace{0.5em}}cnts$ 存储对应元素值的个数足够正确。

### 思路 2：代码

```python
class Solution:
    def countGoodTriplets(self, arr: List[int], a: int, b: int, c: int) -> int:
        size = len(arr)
        ans = 0
        prefix_cnts = [0 for _ in range(1010)]

        for j in range(size):
            for k in range(j + 1, size):
                if abs(arr[j] - arr[k]) <= b:
                    left_j, right_j = arr[j] - a, arr[j] + a
                    left_k, right_k = arr[k] - c, arr[k] + c
                    left, right = max(0, left_j, left_k), min(1000, right_j, right_k)
                    if left <= right:
                        if left == 0:
                            ans += prefix_cnts[right]
                        else:
                            ans += prefix_cnts[right] - prefix_cnts[left - 1]

            for k in range(arr[j], 1001):
                prefix_cnts[k] += 1
        
        return ans
```

### 思路 2：复杂度分析

- **时间复杂度**：$O(n^2 + n \times S)$，其中 $n$ 是数组 $arr$ 的长度，$S$ 为数组的值域上限。
- **空间复杂度**：$O(S)$。

# [1547. 切棍子的最小成本](https://leetcode.cn/problems/minimum-cost-to-cut-a-stick/)

- 标签：数组、动态规划、排序
- 难度：困难

## 题目链接

- [1547. 切棍子的最小成本 - 力扣](https://leetcode.cn/problems/minimum-cost-to-cut-a-stick/)

## 题目大意

**描述**：给定一个整数 $n$，代表一根长度为 $n$ 个单位的木根，木棍从 $0 \sim n$ 标记了若干位置。例如，长度为 $6$ 的棍子可以标记如下：

![](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2020/08/09/statement.jpg)

再给定一个整数数组 $cuts$，其中 $cuts[i]$ 表示需要将棍子切开的位置。

我们可以按照顺序完成切割，也可以根据需要更改切割顺序。

每次切割的成本都是当前要切割的棍子的长度，切棍子的总成本是所有次切割成本的总和。对棍子进行切割将会把一根木棍分成两根较小的木棍（这两根小木棍的长度和就是切割前木棍的长度）。

**要求**：返回切棍子的最小总成本。

**说明**：

- $2 \le n \le 10^6$。
- $1 \le cuts.length \le min(n - 1, 100)$。
- $1 \le cuts[i] \le n - 1$。
- $cuts$ 数组中的所有整数都互不相同。

**示例**：

- 示例 1：

![](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2020/08/09/e1.jpg)

```python
输入：n = 7, cuts = [1,3,4,5]
输出：16
解释：按 [1, 3, 4, 5] 的顺序切割的情况如下所示。
第一次切割长度为 7 的棍子，成本为 7 。第二次切割长度为 6 的棍子（即第一次切割得到的第二根棍子），第三次切割为长度 4 的棍子，最后切割长度为 3 的棍子。总成本为 7 + 6 + 4 + 3 = 20 。而将切割顺序重新排列为 [3, 5, 1, 4] 后，总成本 = 16（如示例图中 7 + 4 + 3 + 2 = 16）。
```

![](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2020/08/09/e11.jpg)

- 示例 2：

```python
输入：n = 9, cuts = [5,6,1,4,2]
输出：22
解释：如果按给定的顺序切割，则总成本为 25。总成本 <= 25 的切割顺序很多，例如，[4, 6, 5, 2, 1] 的总成本 = 22，是所有可能方案中成本最小的。
```

## 解题思路

### 思路 1：动态规划

我们可以预先在数组 $cuts$ 种添加位置 $0$ 和位置 $n$，然后对数组 $cuts$ 进行排序。这样待切割的木棍就对应了数组中连续元素构成的「区间」。

###### 1. 划分阶段

按照区间长度进行阶段划分。

###### 2. 定义状态

定义状态 $dp[i][j]$ 表示为：切割区间为 $[i, j]$ 上的小木棍的最小成本。

###### 3. 状态转移方程

假设位置 $i$ 与位置 $j$ 之间最后一个切割的位置为 $k$，则 $dp[i][j]$ 取决与由 $k$ 作为切割点分割出的两个区间 $[i, k]$ 与 $[k, j]$ 上的最小成本 + 切割位置 $k$ 所带来的成本。

而切割位置 $k$ 所带来的成本是这段区间所代表的小木棍的长度，即 $cuts[j] - cuts[i]$。

则状态转移方程为：$dp[i][j] = min \lbrace dp[i][k] + dp[k][j] + cuts[j] - cuts[i] \rbrace, \quad i < k < j$

###### 4. 初始条件

- 相邻位置之间没有切割点，不需要切割，最小成本为 $0$，即 $dp[i - 1][i] = 0$。
- 其余位置默认为最小成本为一个极大值，即 $dp[i][j] = \infty, \quad i + 1 \ne j$。

###### 5. 最终结果

根据我们之前定义的状态，$dp[i][j]$ 表示为：切割区间为 $[i, j]$ 上的小木棍的最小成本。 所以最终结果为 $dp[0][size - 1]$。

### 思路 1：代码

```python
class Solution:
    def minCost(self, n: int, cuts: List[int]) -> int:
        cuts.append(0)
        cuts.append(n)
        cuts.sort()
        
        size = len(cuts)
        dp = [[float('inf') for _ in range(size)] for _ in range(size)]
        for i in range(1, size):
            dp[i - 1][i] = 0

        for l in range(3, size + 1):        # 枚举区间长度
            for i in range(size):           # 枚举区间起点
                j = i + l - 1               # 根据起点和长度得到终点                            
                if j >= size:      
                    continue
                dp[i][j] = float('inf')
                for k in range(i + 1, j):   # 枚举区间分割点
                    # 状态转移方程，计算合并区间后的最优值
                    dp[i][j] = min(dp[i][j], dp[i][k] + dp[k][j] + cuts[j] - cuts[i])
        return dp[0][size - 1]
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(m^3)$，其中 $m$ 为数组 $cuts$ 的元素个数。
- **空间复杂度**：$O(m^2)$。
# [1551. 使数组中所有元素相等的最小操作数](https://leetcode.cn/problems/minimum-operations-to-make-array-equal/)

- 标签：数学
- 难度：中等

## 题目链接

- [1551. 使数组中所有元素相等的最小操作数 - 力扣](https://leetcode.cn/problems/minimum-operations-to-make-array-equal/)

## 题目大意

**描述**：存在一个长度为 $n$ 的数组 $arr$，其中 $arr[i] = (2 \times i) + 1$，$(0 \le i < n)$。

在一次操作中，我们可以选出两个下标，记作 $x$ 和 $y$（$0 \le x, y < n$），并使 $arr[x]$ 减去 $1$，$arr[y]$ 加上 $1$）。最终目标是使数组中所有元素都相等。

现在给定一个整数 $n$，即数组 $arr$ 的长度。

**要求**：返回使数组 $arr$ 中所有元素相等所需要的最小操作数。

**说明**：

- 题目测试用例将会保证：在执行若干步操作后，数组中的所有元素最终可以全部相等。
- $1 \le n \le 10^4$。

**示例**：

- 示例 1：

```python
输入：n = 3
输出：2
解释：arr = [1, 3, 5]
第一次操作选出 x = 2 和 y = 0，使数组变为 [2, 3, 4]
第二次操作继续选出 x = 2 和 y = 0，数组将会变成 [3, 3, 3]
```

- 示例 2：

```python
输入：n = 6
输出：9
```

## 解题思路

### 思路 1：贪心

通过观察可以发现，数组中所有元素构成了一个等差数列，为了使所有元素相等，在每一次操作中，尽可能让较小值增大，让较大值减小，直到到达平均值为止，这样才能得到最小操作次数。

在一次操作中，我们可以同时让第 $i$ 个元素增大与第 $n - 1 - i$ 个元素减小。这样，我们只需要统计出数组前半部分元素变化幅度即可。

### 思路 1：代码

```python
class Solution:
    def minOperations(self, n: int) -> int:
        ans = 0
        for i in range(n // 2):
            ans += n - 1 - 2 * i
        return ans
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(n)$。
- **空间复杂度**：$O(1)$。

### 思路 2：贪心 + 优化

数组前半部分元素变化幅度的计算可以看做是一个等差数列求和，所以我们可以直接根据高斯求和公式求出结果。

$\lbrace n - 1 + [n - 1 - 2 * (n \div 2 - 1)]\rbrace \times (n \div 2) \div 2 = n \times n \div 4$

### 思路 2：代码

```python
class Solution:
    def minOperations(self, n: int) -> int:
        return n * n // 4
```

### 思路 2：复杂度分析

- **时间复杂度**：$O(1)$。
- **空间复杂度**：$O(1)$。
# [1556. 千位分隔数](https://leetcode.cn/problems/thousand-separator/)

- 标签：字符串
- 难度：简单

## 题目链接

- [1556. 千位分隔数 - 力扣](https://leetcode.cn/problems/thousand-separator/)

## 题目大意

**描述**：给定一个整数 $n$。

**要求**：每隔三位田间点（即 `"."` 符号）作为千位分隔符，并将结果以字符串格式返回。

**说明**：

- $0 \le n \le 2^{31}$。

**示例**：

- 示例 1：

```python
输入：n = 987
输出："987"
```

- 示例 2：

```python
输入：n = 123456789
输出："123.456.789"
```

## 解题思路

### 思路 1：模拟

1. 使用字符串变量 $ans$ 用于存储答案，使用一个计数器 $idx$ 来记录当前位数的个数。
2. 将 $n$ 转为字符串 $s$ 后，从低位向高位遍历。
3. 将当前数字 $s[i]$ 存入 $ans$ 中，计数器加 $1$，当计数器为 $3$ 的整数倍并且当前数字位不是最高位时，将 `"."` 存入 $ans$ 中。
4. 遍历完成后，将 $ans$ 翻转后返回。

### 思路 1：代码

```python
class Solution:
    def thousandSeparator(self, n: int) -> str:
        s = str(n)
        ans = ""

        idx = 0
        for i in range(len(s) - 1, -1, -1):
            ans += s[i]
            idx += 1
            if idx % 3 == 0 and i != 0:
                ans += "."

        return ''.join(reversed(ans))
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(\log n)$。
- **空间复杂度**：$O(\log n)$。

# [1561. 你可以获得的最大硬币数目](https://leetcode.cn/problems/maximum-number-of-coins-you-can-get/)

- 标签：贪心、数组、数学、博弈、排序
- 难度：中等

## 题目链接

- [1561. 你可以获得的最大硬币数目 - 力扣](https://leetcode.cn/problems/maximum-number-of-coins-you-can-get/)

## 题目大意

有 `3*n` 堆数目不一的硬币，三个人按照下面的规则分硬币：

- 每一轮选出任意 3 堆硬币。
- Alice 拿走硬币数量最多的那一堆。
- 我们自己拿走硬币数量第二多的那一堆。
- Bob 拿走最后一堆。
- 重复这个过程，直到没有更多硬币。

现在给定一个整数数组 `piles`，代表 `3*n` 堆硬币，其中 `piles[i]` 表示第 `i` 堆中硬币的数目。

## 解题思路

每次 `3` 堆，总共取 `n` 次。Bob 每次总是选择最少的一堆，所以最终 Bob 得到 `3*n` 堆中最少的 `n` 堆才能使得另外两个人获得更多。所以先对硬币堆进行排序。Bob 拿走最少的 `n` 堆。我们接着分剩下的 `2*n` 堆。

按照大小顺序，每次都选取硬币数目最多的两堆， Alice 取得较大的一堆，我们取较小的一堆。

然后继续在剩余堆中选取硬币数目最多的两堆，同样 Alice 取得较大的一堆，我们取较小的一堆。

只有这样才能在满足规则的情况下，使我们所获得硬币数最多。

最后统计我们所获取的硬币数，并返回结果。

## 代码

```python
class Solution:
    def maxCoins(self, piles: List[int]) -> int:
        piles.sort()
        ans = 0
        for i in range(len(piles) // 3, len(piles), 2):
            ans += piles[i]
        return ans
```

# [1567. 乘积为正数的最长子数组长度](https://leetcode.cn/problems/maximum-length-of-subarray-with-positive-product/)

- 标签：贪心、数组、动态规划
- 难度：中等

## 题目链接

- [1567. 乘积为正数的最长子数组长度 - 力扣](https://leetcode.cn/problems/maximum-length-of-subarray-with-positive-product/)

## 题目大意

给定一个整数数组 `nums`。

要求：求出乘积为正数的最长子数组的长度。

- 子数组：是由原数组中零个或者更多个连续数字组成的数组。

## 解题思路

使用动态规划来做。使用数组 `pos` 表示以下标 `i` 结尾的乘积为正数的最长子数组长度。使用数组 `neg` 表示以下标 `i` 结尾的乘积为负数的最长子数组长度。

- 先初始化 `pos[0]`、`neg[0]`。
  - 如果 `nums[0] == 0`，则 `pos[0] = 0, neg[0] = 0`。
  - 如果 `nums[0] > 0`，则 `pos[0] = 1, neg[0] = 0`。
  - 如果 `nums[0] < 0`，则 `pos[0] = 0, neg[0] = 1`。

- 然后从下标 `1` 开始递推遍历数组 `nums`，对于 `nums[i - 1]` 和 `nums[i]`：

  - 如果 `nums[i - 1] == 0`，显然有 `pos[i] = 0`，`neg[i] = 0`。表示：以`i` 结尾的乘积为正数的最长子数组长度为 `0`，以`i` 结尾的乘积为负数数的最长子数组长度也为 `0`。

  - 如果 `nums[i - 1] > 0`，则 `pos[i] = pos[i - 1] + 1`。而 `neg[i]` 需要进行判断，如果 `neg[i - 1] > 0`，则再乘以当前 `nums[i]` 后仍为负数，此时长度 +1，即 `neg[i] = neg[i - 1] + 1 `。而如果 `neg[i - 1] == 0`，则 `neg[i] = 0`。

  - 如果 `nums[i - 1] < 0`，则 `pos[i]` 需要进行判断，如果 `neg[i - 1] > 0`，再乘以当前 `nums[i]` 后变为正数，此时长度 +1，即 `pos[i] = neg[i - 1] + 1`。而如果 `neg[i - 1] = 0`，则 `pos[i] = 0`。
  - 更新 `ans` 答案为 `pos[i]` 最大值。

- 最后输出答案 `ans`。

## 代码

```python
class Solution:
    def getMaxLen(self, nums: List[int]) -> int:
        size = len(nums)
        pos = [0 for _ in range(size + 1)]
        neg = [0 for _ in range(size + 1)]

        if nums[0] == 0:
            pos[0], neg[0] = 0, 0
        elif nums[0] > 0:
            pos[0], neg[0] = 1, 0
        else:
            pos[0], neg[0] = 0, 1

        ans = pos[0]
        for i in range(1, size):
            if nums[i] == 0:
                pos[i] = 0
                neg[i] = 0
            elif nums[i] > 0:
                pos[i] = pos[i - 1] + 1
                neg[i] = neg[i - 1] + 1 if neg[i - 1] > 0 else 0
            elif nums[i] < 0:
                pos[i] = neg[i - 1] + 1 if neg[i - 1] > 0 else 0
                neg[i] = pos[i - 1] + 1
            ans = max(ans, pos[i])
        return ans
```

## 参考资料

- 【题解】[递推就完事了，巨好理解~ - 乘积为正数的最长子数组长度 - 力扣](https://leetcode.cn/problems/maximum-length-of-subarray-with-positive-product/solution/di-tui-jiu-wan-shi-liao-ju-hao-li-jie-by-time-limi/)
# [1582. 二进制矩阵中的特殊位置](https://leetcode.cn/problems/special-positions-in-a-binary-matrix/)

- 标签：数组、矩阵
- 难度：简单

## 题目链接

- [1582. 二进制矩阵中的特殊位置 - 力扣](https://leetcode.cn/problems/special-positions-in-a-binary-matrix/)

## 题目大意

**描述**：给定一个 $m \times n$ 的二进制矩阵 $mat$。

**要求**：返回矩阵 $mat$ 中特殊位置的数量。

**说明**：

- **特殊位置**：如果位置 $(i, j)$ 满足 $mat[i][j] == 1$ 并且行 $i$ 与列 $j$ 中的所有其他元素都是 $0$（行和列的下标从 $0$ 开始计数），那么它被称为特殊位置。
- $m == mat.length$。
- $n == mat[i].length$。
- $1 \le m, n \le 100$。
- $mat[i][j]$ 是 $0$ 或 $1$。

**示例**：

- 示例 1：

![](https://assets.leetcode.com/uploads/2021/12/23/special1.jpg)

```python
输入：mat = [[1,0,0],[0,0,1],[1,0,0]]
输出：1
解释：位置 (1, 2) 是一个特殊位置，因为 mat[1][2] == 1 且第 1 行和第 2 列的其他所有元素都是 0。
```

- 示例 2：

![img](https://assets.leetcode.com/uploads/2021/12/24/special-grid.jpg)

```python
输入：mat = [[1,0,0],[0,1,0],[0,0,1]]
输出：3
解释：位置 (0, 0)，(1, 1) 和 (2, 2) 都是特殊位置。
```

## 解题思路

### 思路 1：模拟

1. 按照行、列遍历二位数组 $mat$。
2. 使用数组 $row\underline{\hspace{0.5em}}cnts$、$col\underline{\hspace{0.5em}}cnts$ 分别记录每行和每列所含 $1$ 的个数。
3. 再次按照行、列遍历二维数组 $mat$。
4. 统计满足 $mat[row][col] == 1$ 并且 $row\underline{\hspace{0.5em}}cnts[row] == col\underline{\hspace{0.5em}}cnts[col] == 1$ 的位置个数。 
5. 返回答案。

### 思路 1：代码

```Python
class Solution:
    def numSpecial(self, mat: List[List[int]]) -> int:
        rows, cols = len(mat), len(mat[0])
        row_cnts = [0 for _ in range(rows)]
        col_cnts = [0 for _ in range(cols)]

        for row in range(rows):
            for col in range(cols):
                row_cnts[row] += mat[row][col]
                col_cnts[col] += mat[row][col]

        ans = 0
        for row in range(rows):
            for col in range(cols):
                if mat[row][col] == 1 and row_cnts[row] == 1 and col_cnts[col] == 1:
                    ans += 1
        
        return ans
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(m \times n)$，其中 $m$、$n$ 分别为数组 $mat$ 的行数和列数。
- **空间复杂度**：$O(m + n)$。

# [1584. 连接所有点的最小费用](https://leetcode.cn/problems/min-cost-to-connect-all-points/)

- 标签：并查集、图、数组、最小生成树
- 难度：中等

## 题目链接

- [1584. 连接所有点的最小费用 - 力扣](https://leetcode.cn/problems/min-cost-to-connect-all-points/)

## 题目大意

**描述**：给定一个 $points$ 数组，表示 2D 平面上的一些点，其中 $points[i] = [x_i, y_i]$。

链接点 $[x_i, y_i]$ 和点 $[x_j, y_j]$ 的费用为它们之间的 **曼哈顿距离**：$|x_i - x_j| + |y_i - y_j|$。其中 $|val|$ 表示 $val$ 的绝对值。

**要求**：返回将所有点连接的最小总费用。

**说明**：

- 只有任意两点之间有且仅有一条简单路径时，才认为所有点都已连接。
- $1 \le points.length \le 1000$。
- $-10^6 \le x_i, y_i \le 10^6$。
- 所有点 $(x_i, y_i)$ 两两不同。

**示例**：

- 示例 1：

![](https://assets.leetcode.com/uploads/2020/08/26/d.png)

![](https://assets.leetcode.com/uploads/2020/08/26/c.png)

```python
输入：points = [[0,0],[2,2],[3,10],[5,2],[7,0]]
输出：20
解释：我们可以按照上图所示连接所有点得到最小总费用，总费用为 20 。
注意到任意两个点之间只有唯一一条路径互相到达。
```

- 示例 2：

```python
输入：points = [[3,12],[-2,5],[-4,1]]
输出：18
```

## 解题思路

将所有点之间的费用看作是边，则所有点和边可以看作是一个无向图。每两个点之间都存在一条无向边，边的权重为两个点之间的曼哈顿距离。将所有点连接的最小总费用，其实就是求无向图的最小生成树。对此我们可以使用 Prim 算法或者 Kruskal 算法。

### 思路 1：Prim 算法

每次选择最短边来扩展最小生成树，从而保证生成树的总权重最小。算法通过不断扩展小生成树的顶点集合 $MST$，逐步构建出最小生成树。

### 思路 1：代码

```Python
class Solution:
    def distance(self, point1, point2):
        return abs(point1[0] - point2[0]) + abs(point1[1] - point2[1])

    def Prim(self, points, start):
        size = len(points)
        vis = set()
        dis = [float('inf') for _ in range(size)]

        ans = 0                     # 最小生成树的边权值
        dis[start] = 0              # 起始位置到起始位置的边权值初始化为 0

        for i in range(1, size):
            dis[i] = self.distance(points[start], points[i])
        vis.add(start)

        for _ in range(size - 1):       # 进行 n 轮迭代
            min_dis = float('inf')
            min_dis_i = -1
            for i in range(size):
                if i not in vis and dis[i] < min_dis:
                    min_dis = dis[i]
                    min_dis_i = i
            if min_dis_i == -1:
                return -1

            ans += min_dis
            vis.add(min_dis_i)
            

            for i in range(size):
                if i not in vis:
                    dis[i] = min(dis[i], self.distance(points[i], points[min_dis_i]))

        return ans

    def minCostConnectPoints(self, points: List[List[int]]) -> int:
        return self.Prim(points, 0)
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(n^2)$。
- **空间复杂度**：$O(n^2)$。

### 思路 2：Kruskal 算法

通过依次选择权重最小的边并判断其两个端点是否连接在同一集合中，从而逐步构建最小生成树。这个过程保证了最终生成的树是无环的，并且总权重最小。

### 思路 2：代码

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
    def Kruskal(self, edges, size):
        union_find = UnionFind(size)
        
        edges.sort(key=lambda x: x[2])
        
        ans, cnt = 0, 0
        for x, y, dist in edges:
            if union_find.is_connected(x, y):
                continue
            ans += dist
            cnt += 1
            union_find.union(x, y)
            if cnt == size - 1:
                return ans
        return ans
    
    def minCostConnectPoints(self, points: List[List[int]]) -> int:
        size = len(points)
        edges = []
        for i in range(size):
            xi, yi = points[i]
            for j in range(i + 1, size):
                xj, yj = points[j]
                dist = abs(xi - xj) + abs(yi - yj)
                edges.append([i, j, dist])
                
        ans = self.Kruskal(edges, size)
        return ans

```

### 思路 2：复杂度分析

- **时间复杂度**：$O(m \times \log(n))$。其中 $m$ 为边数，$n$ 为节点数，本题中 $m = n^2$。
- **空间复杂度**：$O(n^2)$。

# [1593. 拆分字符串使唯一子字符串的数目最大](https://leetcode.cn/problems/split-a-string-into-the-max-number-of-unique-substrings/)

- 标签：哈希表、字符串、回溯
- 难度：中等

## 题目链接

- [1593. 拆分字符串使唯一子字符串的数目最大 - 力扣](https://leetcode.cn/problems/split-a-string-into-the-max-number-of-unique-substrings/)

## 题目大意

**描述**：给定一个字符串 $s$。将字符串 $s$ 拆分后可以得到若干非空子字符串，这些子字符串连接后应当能够还原为原字符串。但是拆分出来的每个子字符串都必须是唯一的 。

**要求**：拆分该字符串，并返回拆分后唯一子字符串的最大数目。

**说明**：

- 子字符串是字符串中的一个连续字符序列。
- $1 \le s.length \le 16$。
- $s$ 仅包含小写英文字母。

**示例**：

- 示例 1：

```python
输入：s = "ababccc"
输出：5
解释：一种最大拆分方法为 ['a', 'b', 'ab', 'c', 'cc'] 。像 ['a', 'b', 'a', 'b', 'c', 'cc'] 这样拆分不满足题目要求，因为其中的 'a' 和 'b' 都出现了不止一次。
```

- 示例 2：

```python
输入：s = "aba"
输出：2
解释：一种最大拆分方法为 ['a', 'ba']。
```

## 解题思路

### 思路 1：回溯算法

维护一个全局变量 $ans$ 用于记录拆分后唯一子字符串的最大数目。并使用集合 $s\underline{\hspace{0.5em}}set$ 记录不重复的子串。

- 从下标为 $0$ 开头的子串回溯。
- 对于下标为 $index$ 开头的子串，我们可以在 $index + 1$ 开始到 $len(s) - 1$ 的位置上，分别进行子串拆分，将子串拆分为 $s[index: i + 1]$。

- 如果当前子串不在 $s\underline{\hspace{0.5em}}set$ 中，则将其存入 $s\underline{\hspace{0.5em}}set$ 中，然后记录当前拆分子串个数，并从 $i + 1$ 的位置进行下一层递归拆分。然后在拆分完，对子串进行回退操作。
- 如果拆到字符串 $s$ 的末尾，则记录并更新 $ans$。
- 在开始位置还可以进行以下剪枝：如果剩余字符个数 + 当前子串个数 <= 当前拆分后子字符串的最大数目，则直接返回。

最后输出 $ans$。

### 思路 1：代码

```python
class Solution:
    ans = 0
    def backtrack(self, s, index, count, s_set):
        if len(s) - index + count <= self.ans:
            return 
        if index >= len(s):
            self.ans = max(self.ans, count)
            return

        for i in range(index, len(s)):
            sub_s = s[index: i + 1]
            if sub_s not in s_set:
                s_set.add(sub_s)
                self.backtrack(s, i + 1, count + 1, s_set)
                s_set.remove(sub_s)


    def maxUniqueSplit(self, s: str) -> int:
        s_set = set()
        self.ans = 0
        self.backtrack(s, 0, 0, s_set)
        return self.ans
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(n \times 2^n)$，其中 $n$ 为字符串的长度。
- **空间复杂度**：$O(n)$。

# [1595. 连通两组点的最小成本](https://leetcode.cn/problems/minimum-cost-to-connect-two-groups-of-points/)

- 标签：位运算、数组、动态规划、状态压缩、矩阵
- 难度：困难

## 题目链接

- [1595. 连通两组点的最小成本 - 力扣](https://leetcode.cn/problems/minimum-cost-to-connect-two-groups-of-points/)

## 题目大意

**描述**：有两组点，其中一组中有 $size_1$ 个点，第二组中有 $size_2$ 个点，且 $size_1 \ge size_2$。现在给定一个大小为 $size_1 \times size_2$ 的二维数组 $cost$ 用于表示两组点任意两点之间的链接成本。其中 $cost[i][j]$ 表示第一组中第 $i$ 个点与第二组中第 $j$ 个点的链接成本。

如果两个组中每个点都与另一个组中的一个或多个点连接，则称这两组点是连通的。 

**要求**：返回连通两组点所需的最小成本。

**说明**：

- $size_1 == cost.length$。
- $size_2 == cost[i].length$。
- $1 \le size_1, size_2 \le 12$。
- $size_1 \ge size_2$。
- $0 \le cost[i][j] \le 100$。

**示例**：

- 示例 1：

![](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2020/09/20/ex1.jpg)

```python
输入：cost = [[15, 96], [36, 2]]
输出：17
解释：连通两组点的最佳方法是：
1--A
2--B
总成本为 17。
```

- 示例 2：

![](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2020/09/20/ex2.jpg)

```python
输入：cost = [[1, 3, 5], [4, 1, 1], [1, 5, 3]]
输出：4
解释：连通两组点的最佳方法是：
1--A
2--B
2--C
3--A
最小成本为 4。
请注意，虽然有多个点连接到第一组中的点 2 和第二组中的点 A ，但由于题目并不限制连接点的数目，所以只需要关心最低总成本。
```

## 解题思路

### 思路 1：状压 DP



### 思路 1：代码

```python
class Solution:
    def connectTwoGroups(self, cost: List[List[int]]) -> int:
        m, n = len(cost), len(cost[0])
        states = 1 << n
        dp = [[float('inf') for _ in range(states)] for _ in range(m + 1)]
        dp[0][0] = 0
        for i in range(1, m + 1):
            for state in range(states):
                for j in range(n):
                    dp[i][state | (1 << j)] = min(dp[i][state | (1 << j)], dp[i - 1][state] + cost[i - 1][j], dp[i][state] + cost[i - 1][j])

        return dp[m][states - 1]
```

### 思路 1：复杂度分析

- **时间复杂度**：
- **空间复杂度**：

# [1603. 设计停车系统](https://leetcode.cn/problems/design-parking-system/)

- 标签：设计、计数、模拟
- 难度：简单

## 题目链接

- [1603. 设计停车系统 - 力扣](https://leetcode.cn/problems/design-parking-system/)

## 题目大意

给一个停车场设计一个停车系统。停车场总共有三种尺寸的车位：大、中、小，每种尺寸的车位分别有固定数目。

现在要求实现 `ParkingSystem` 类：

-  `ParkingSystem(big, medium, small)`：初始化 ParkingSystem 类，三个参数分别对应三种尺寸车位的数目。
- `addCar(carType) -> bool:`：检测是否有 `carType` 对应的停车位，如果有，则将车停入车位，并返回 `True`，否则返回 `False`。

## 解题思路

使用不同成员变量存放车位数目。并根据给定操作进行判断。

## 代码

```python
class ParkingSystem:

    def __init__(self, big: int, medium: int, small: int):
        self.park = [0, big, medium, small]

    def addCar(self, carType: int) -> bool:
        if self.park[carType] == 0:
            return False
        self.park[carType] -= 1
        return True
```

# [1605. 给定行和列的和求可行矩阵](https://leetcode.cn/problems/find-valid-matrix-given-row-and-column-sums/)

- 标签：贪心、数组、矩阵
- 难度：中等

## 题目链接

- [1605. 给定行和列的和求可行矩阵 - 力扣](https://leetcode.cn/problems/find-valid-matrix-given-row-and-column-sums/)

## 题目大意

**描述**：给你两个非负整数数组 `rowSum` 和 `colSum` ，其中 `rowSum[i]` 是二维矩阵中第 `i` 行元素的和，`colSum[j]` 是第 `j` 列元素的和。换句话说，我们不知道矩阵里的每个元素，只知道每一行的和，以及每一列的和。

**要求**：找到并返回一个大小为 `rowSum.length * colSum.length` 的任意非负整数矩阵，且该矩阵满足 `rowSum` 和 `colSum` 的要求。

**说明**：

- 返回任意一个满足题目要求的二维矩阵即可，题目保证存在至少一个可行矩阵。
- $1 \le rowSum.length, colSum.length \le 500$。
- $0 \le rowSum[i], colSum[i] \le 10^8$。
- $sum(rows) == sum(columns)$。

**示例**：

- 示例 1：

```python
输入：rowSum = [3,8], colSum = [4,7]
输出：[[3,0],
        [1,7]]

解释
第 0 行：3 + 0 = 3 == rowSum[0]
第 1 行：1 + 7 = 8 == rowSum[1]
第 0 列：3 + 1 = 4 == colSum[0]
第 1 列：0 + 7 = 7 == colSum[1]
行和列的和都满足题目要求，且所有矩阵元素都是非负的。
另一个可行的矩阵为   [[1,2],
                   [3,5]]
```

## 解题思路

### 思路 1：贪心算法

题目要求找出一个满足要求的非负整数矩阵，矩阵中元素值可以为 `0`。所以我们可以尽可能将大的值填入前面的行和列中，然后剩余位置用 `0` 补齐即可。具体做法如下：

1. 使用二维数组 `board` 来保存答案，初始情况下，`board` 中元素全部赋值为 `0`。
2. 遍历二维数组的每一行，每一列。当前位置下的值为当前行的和与当前列的和的较小值，即 `board[row][col] = min(rowSum[row], colSum[col])`。
3. 更新当前行的和，将当前行的和减去 `board[row][col]`。
4. 更新当前列的和，将当前列的和减去 `board[row][col]`。
5. 遍历完返回二维数组 `board`。

### 思路 1：贪心算法代码

```python
class Solution:
    def restoreMatrix(self, rowSum: List[int], colSum: List[int]) -> List[List[int]]:
        rows, cols = len(rowSum), len(colSum)
        board = [[0 for _ in range(cols)] for _ in range(rows)]
        for row in range(rows):
            for col in range(cols):
                board[row][col] = min(rowSum[row], colSum[col])
                rowSum[row] -= board[row][col]
                colSum[col] -= board[row][col]
        return board
```
# [1614. 括号的最大嵌套深度](https://leetcode.cn/problems/maximum-nesting-depth-of-the-parentheses/)

- 标签：栈、字符串
- 难度：简单

## 题目链接

- [1614. 括号的最大嵌套深度 - 力扣](https://leetcode.cn/problems/maximum-nesting-depth-of-the-parentheses/)

## 题目大意

**描述**：给你一个有效括号字符串 $s$。

**要求**：返回该字符串 $s$ 的嵌套深度 。

**说明**：

- 如果字符串满足以下条件之一，则可以称之为 有效括号字符串（valid parentheses string，可以简写为 VPS）：
  - 字符串是一个空字符串 `""`，或者是一个不为 `"("` 或 `")"` 的单字符。
  - 字符串可以写为 $AB$（$A$ 与 B 字符串连接），其中 $A$ 和 $B$ 都是有效括号字符串 。
  - 字符串可以写为 ($A$)，其中 $A$ 是一个有效括号字符串。

- 类似地，可以定义任何有效括号字符串 $s$ 的 嵌套深度 $depth(s)$：

  - `depth("") = 0`。
  - `depth(C) = 0`，其中 $C$ 是单个字符的字符串，且该字符不是 `"("` 或者 `")"`。
  - `depth(A + B) = max(depth(A), depth(B))`，其中 $A$ 和 $B$ 都是 有效括号字符串。
  - `depth("(" + A + ")") = 1 + depth(A)`，其中 A 是一个 有效括号字符串。
- $1 \le s.length \le 100$。
- $s$ 由数字 $0 \sim 9$ 和字符 `'+'`、`'-'`、`'*'`、`'/'`、`'('`、`')'` 组成。
- 题目数据保证括号表达式 $s$ 是有效的括号表达式。

**示例**：

- 示例 1：

```python
输入：s = "(1+(2*3)+((8)/4))+1"
输出：3
解释：数字 8 在嵌套的 3 层括号中。
```

- 示例 2：

```python
输入：s = "(1)+((2))+(((3)))"
输出：3
```

## 解题思路

### 思路 1：模拟

我们可以使用栈来进行模拟括号匹配。遍历字符串 $s$，如果遇到左括号，则将其入栈，如果遇到右括号，则弹出栈中的左括号，与当前右括号进行匹配。在整个过程中栈的大小的最大值，就是我们要求的 $s$ 的嵌套深度，其实也是求最大的连续左括号的数量（跳过普通字符，并且与右括号匹配后）。具体步骤如下：

1. 使用 $ans$ 记录最大的连续左括号数量，使用 $cnt$ 记录当前栈中左括号的数量。
2. 遍历字符串 $s$：
   1. 如果遇到左括号，则令 $cnt$ 加 $1$。
   2. 如果遇到右括号，则令 $cnt$ 减 $1$。
   3. 将 $cnt$ 与答案进行比较，更新最大的连续左括号数量。
3. 遍历完字符串 $s$，返回答案 $ans$。

### 思路 1：代码

```Python
class Solution:
    def maxDepth(self, s: str) -> int:
        ans, cnt = 0, 0
        for ch in s:
            if ch == '(':
                cnt += 1
            elif ch == ')':
                cnt -= 1
            ans = max(ans, cnt)

        return ans
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(n)$，其中 $n$ 为字符串 $s$ 的长度。
- **空间复杂度**：$O(1)$。

# [1617. 统计子树中城市之间最大距离](https://leetcode.cn/problems/count-subtrees-with-max-distance-between-cities/)

- 标签：位运算、树、动态规划、状态压缩、枚举
- 难度：困难

## 题目链接

- [1617. 统计子树中城市之间最大距离 - 力扣](https://leetcode.cn/problems/count-subtrees-with-max-distance-between-cities/)

## 题目大意

**描述**：给定一个整数 $n$，代表 $n$ 个城市，城市编号为 $1 \sim n$。同时给定一个大小为 $n - 1$ 的数组 $edges$，其中 $edges[i] = [u_i, v_i]$ 表示城市 $u_i$ 和 $v_i$ 之间有一条双向边。题目保证任意城市之间只有唯一的一条路径。换句话说，所有城市形成了一棵树。

**要求**：返回一个大小为 $n - 1$ 的数组，其中第 $i$ 个元素（下标从 $1$ 开始）是城市间距离恰好等于 $i$ 的子树数目。

**说明**：

- **两个城市间距离**：定义为它们之间需要经过的边的数目。
- **一棵子树**：城市的一个子集，且子集中任意城市之间可以通过子集中的其他城市和边到达。两个子树被认为不一样的条件是至少有一个城市在其中一棵子树中存在，但在另一棵子树中不存在。
- $2 \le n \le 15$。
- $edges.length == n - 1$。
- $edges[i].length == 2$。
- $1 \le u_i, v_i \le n$。
- 题目保证 $(ui, vi)$ 所表示的边互不相同。

**示例**：

- 示例 1：

```python
输入：n = 4, edges = [[1,2],[2,3],[2,4]]
输出：[3,4,0]
解释：
子树 {1,2}, {2,3} 和 {2,4} 最大距离都是 1 。
子树 {1,2,3}, {1,2,4}, {2,3,4} 和 {1,2,3,4} 最大距离都为 2 。
不存在城市间最大距离为 3 的子树。
```

- 示例 2：

```python
输入：n = 2, edges = [[1,2]]
输出：[1]
```

## 解题思路

### 思路 1：树形 DP + 深度优先搜索

因为题目中给定 $n$ 的范围为 $2 \le n \le 15$，范围比较小，我们可以通过类似「[0078. 子集](https://leetcode.cn/problems/subsets/)」中二进制枚举的方式，得到所有子树的子集。

而对于一个确定的子树来说，求子树中两个城市间距离就是在求子树的直径，这就跟 [「1245. 树的直径」](https://leetcode.cn/problems/tree-diameter/) 和 [「2246. 相邻字符不同的最长路径」](https://leetcode.cn/problems/longest-path-with-different-adjacent-characters/) 一样了。

那么这道题的思路就变成了：

1. 通过二进制枚举的方式，得到所有子树。
2. 对于当前子树，通过树形 DP + 深度优先搜索的方式，计算出当前子树的直径。
3. 统计所有子树直径中经过的不同边数个数，将其放入答案数组中。

### 思路 1：代码

```python
class Solution:
    def countSubgraphsForEachDiameter(self, n: int, edges: List[List[int]]) -> List[int]:
        graph = [[] for _ in range(n)]                              # 建图
        for u, v in edges:
            graph[u - 1].append(v - 1)
            graph[v - 1].append(u - 1)

        def dfs(mask, u):
            nonlocal visited, diameter
            visited |= 1 << u                                       # 标记 u 访问过
            u_len = 0                                               # u 节点的最大路径长度
            for v in graph[u]:                                      # 遍历 u 节点的相邻节点
                if (visited >> v) & 1 == 0 and mask >> v & 1:       # v 没有访问过，且在子集中
                    v_len = dfs(mask, v)                            # 相邻节点的最大路径长度
                    diameter = max(diameter, u_len + v_len + 1)     # 维护最大路径长度
                    u_len = max(u_len, v_len + 1)                   # 更新 u 节点的最大路径长度
            return u_len
        
        ans = [0 for _ in range(n - 1)]

        for mask in range(3, 1 << n):                               # 二进制枚举子集
            if mask & (mask - 1) == 0:                              # 子集至少需要两个点
                continue
            visited = 0
            diameter = 0
            u = mask.bit_length() - 1        
            dfs(mask, u)                                            # 在子集 mask 中递归求树的直径
            if visited == mask:
                ans[diameter - 1] += 1
        return ans
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(n \times 2^n)$，其中 $n$ 为给定的城市数目。
- **空间复杂度**：$O(n)$。
# [1631. 最小体力消耗路径](https://leetcode.cn/problems/path-with-minimum-effort/)

- 标签：深度优先搜索、广度优先搜索、并查集、数组、二分查找、矩阵、堆（优先队列）
- 难度：中等

## 题目链接

- [1631. 最小体力消耗路径 - 力扣](https://leetcode.cn/problems/path-with-minimum-effort/)

## 题目大意

**描述**：给定一个 $rows \times cols$ 大小的二维数组 $heights$，其中 $heights[i][j]$ 表示为位置 $(i, j)$ 的高度。

现在要从左上角 $(0, 0)$ 位置出发，经过方格的一些点，到达右下角 $(n - 1, n - 1)$  位置上。其中所经过路径的花费为「这条路径上所有相邻位置的最大高度差绝对值」。

**要求**：计算从 $(0, 0)$ 位置到 $(n - 1, n - 1)$  的最优路径的花费。

**说明**：

- **最优路径**：路径上「所有相邻位置最大高度差绝对值」最小的那条路径。
- $rows == heights.length$。
- $columns == heights[i].length$。
- $1 \le rows, columns \le 100$。
- $1 \le heights[i][j] \le 10^6$。

**示例**：

- 示例 1：

![](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2020/10/25/ex1.png)

```python
输入：heights = [[1,2,2],[3,8,2],[5,3,5]]
输出：2
解释：路径 [1,3,5,3,5] 连续格子的差值绝对值最大为 2 。
这条路径比路径 [1,2,2,2,5] 更优，因为另一条路径差值最大值为 3。
```

- 示例 2：

![](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2020/10/25/ex2.png)

```python
输入：heights = [[1,2,3],[3,8,4],[5,3,5]]
输出：1
解释：路径 [1,2,3,4,5] 的相邻格子差值绝对值最大为 1 ，比路径 [1,3,5,3,5] 更优。
```

## 解题思路

### 思路 1：并查集

将整个网络抽象为一个无向图，每个点与相邻的点（上下左右）之间都存在一条无向边，边的权重为两个点之间的高度差绝对值。

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
    def minimumEffortPath(self, heights: List[List[int]]) -> int:
        row_size = len(heights)
        col_size = len(heights[0])
        size = row_size * col_size
        edges = []
        for row in range(row_size):
            for col in range(col_size):
                if row < row_size - 1:
                    x = row * col_size + col
                    y = (row + 1) * col_size + col
                    h = abs(heights[row][col] - heights[row + 1][col])
                    edges.append([x, y, h])
                if col < col_size - 1:
                    x = row * col_size + col
                    y = row * col_size + col + 1
                    h = abs(heights[row][col] - heights[row][col + 1])
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

# [1641. 统计字典序元音字符串的数目](https://leetcode.cn/problems/count-sorted-vowel-strings/)

- 标签：数学、动态规划、组合数学
- 难度：中等

## 题目链接

- [1641. 统计字典序元音字符串的数目 - 力扣](https://leetcode.cn/problems/count-sorted-vowel-strings/)

## 题目大意

**描述**：给定一个整数 $n$。

**要求**：返回长度为 $n$、仅由原音（$a$、$e$、$i$、$o$、$u$）组成且按字典序排序的字符串数量。

**说明**：

- 字符串 $a$ 按字典序排列需要满足：对于所有有效的 $i$，$s[i]$ 在字母表中的位置总是与 $s[i + 1]$ 相同或在 $s[i+1] $之前。
- $1 \le n \le 50$。

**示例**：

- 示例 1：

```python
输入：n = 1
输出：5
解释：仅由元音组成的 5 个字典序字符串为 ["a","e","i","o","u"]
```

- 示例 2：

```python
输入：n = 2
输出：15
解释：仅由元音组成的 15 个字典序字符串为
["aa","ae","ai","ao","au","ee","ei","eo","eu","ii","io","iu","oo","ou","uu"]
注意，"ea" 不是符合题意的字符串，因为 'e' 在字母表中的位置比 'a' 靠后
```

## 解题思路

### 思路 1：组和数学

题目要求按照字典序排列，则如果确定了每个元音的出现次数可以确定一个序列。

对于长度为 $n$ 的序列，$a$、$e$、$i$、$o$、$u$ 出现次数加起来为 $n$ 次，且顺序为  $a…a \rightarrow e…e \rightarrow i…i  \rightarrow o…o  \rightarrow u…u$。

我们可以看作是将 $n$ 分隔成了 $5$ 份，每一份对应一个原音字母的数量。

我们可以使用「隔板法」的方式，看作有 $n$ 个球，$4$ 个板子，将 $n$ 个球分隔成 $5$ 份。

则一共有 $n + 4$ 个位置可以放板子，总共需要放 $4$ 个板子，则答案为 $C_{n + 4}^4$，其中 $C$ 为组和数。

### 思路 1：代码

```Python
class Solution:
    def countVowelStrings(self, n: int) -> int:
        return comb(n + 4, 4)
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(| \sum |)$，其中 $\sum$ 为字符集，本题中 $| \sum | = 5$ 。
- **空间复杂度**：$O(1)$。

# [1646. 获取生成数组中的最大值](https://leetcode.cn/problems/get-maximum-in-generated-array/)

- 标签：数组、动态规划、模拟
- 难度：简单

## 题目链接

- [1646. 获取生成数组中的最大值 - 力扣](https://leetcode.cn/problems/get-maximum-in-generated-array/)

## 题目大意

**描述**：给定一个整数 $n$，按照下述规则生成一个长度为 $n + 1$ 的数组 $nums$：

- $nums[0] = 0$。
- $nums[1] = 1$。
- 当 $2 \le 2 \times i \le n$ 时，$nums[2 \times i] = nums[i]$。
- 当 $2 \le 2 \times i + 1 \le n$ 时，$nums[2 \times i + 1] = nums[i] + nums[i + 1]$。

**要求**：返回生成数组 $nums$ 中的最大值。

**说明**：

- $0 \le n \le 100$。

**示例**：

- 示例 1：

```python
输入：n = 7
输出：3
解释：根据规则：
  nums[0] = 0
  nums[1] = 1
  nums[(1 * 2) = 2] = nums[1] = 1
  nums[(1 * 2) + 1 = 3] = nums[1] + nums[2] = 1 + 1 = 2
  nums[(2 * 2) = 4] = nums[2] = 1
  nums[(2 * 2) + 1 = 5] = nums[2] + nums[3] = 1 + 2 = 3
  nums[(3 * 2) = 6] = nums[3] = 2
  nums[(3 * 2) + 1 = 7] = nums[3] + nums[4] = 2 + 1 = 3
因此，nums = [0,1,1,2,1,3,2,3]，最大值 3
```

- 示例 2：

```python
输入：n = 2
输出：1
解释：根据规则，nums[0]、nums[1] 和 nums[2] 之中的最大值是 1
```

## 解题思路

### 思路 1：模拟

1. 按照题目要求，定义一个长度为 $n + 1$ 的数组 $nums$。
2. 按照规则模拟生成对应的 $nums$ 数组元素。
3. 求出数组 $nums$ 中最大值，并作为答案返回。

### 思路 1：代码

```python
class Solution:
    def getMaximumGenerated(self, n: int) -> int:
        if n <= 1:
            return n
            
        nums = [0 for _ in range(n + 1)]
        nums[1] = 1

        for i in range(n):
            if 2 * i <= n:
                nums[2 * i] = nums[i]
            if 2 * i + 1 <= n:
                nums[2 * i + 1] = nums[i] + nums[i + 1]

        ans = max(nums)
        return ans
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(n)$。
- **空间复杂度**：$O(n)$。
# [1647. 字符频次唯一的最小删除次数](https://leetcode.cn/problems/minimum-deletions-to-make-character-frequencies-unique/)

- 标签：贪心、哈希表、字符串、排序
- 难度：中等

## 题目链接

- [1647. 字符频次唯一的最小删除次数 - 力扣](https://leetcode.cn/problems/minimum-deletions-to-make-character-frequencies-unique/)

## 题目大意

**描述**：给定一个字符串 $s$。

**要求**：返回使 $s$ 成为优质字符串需要删除的最小字符数。

**说明**：

- **频次**：指的是该字符在字符串中的出现次数。例如，在字符串 `"aab"` 中，`'a'` 的频次是 $2$，而 `'b'` 的频次是 $1$。
- **优质字符串**：如果字符串 $s$ 中不存在两个不同字符频次相同的情况，就称 $s$ 是优质字符串。
- $1 \le s.length \le 10^5$。
- $s$ 仅含小写英文字母。

**示例**：

- 示例 1：

```python
输入：s = "aab"
输出：0
解释：s 已经是优质字符串。
```

- 示例 2：

```python
输入：s = "aaabbbcc"
输出：2
解释：可以删除两个 'b' , 得到优质字符串 "aaabcc" 。
另一种方式是删除一个 'b' 和一个 'c' ，得到优质字符串 "aaabbc"。
```

## 解题思路

### 思路 1：贪心算法 + 哈希表

1. 使用哈希表 $cnts$ 统计每字符串中每个字符出现次数。
2. 然后使用集合 $s\underline{\hspace{0.5em}}set$ 保存不同的出现次数。
3. 遍历哈希表中所偶出现次数：
   1. 如果当前出现次数不在集合 $s\underline{\hspace{0.5em}}set$ 中，则将该次数添加到集合 $s\underline{\hspace{0.5em}}set$ 中。
   2. 如果当前出现次数在集合 $s\underline{\hspace{0.5em}}set$ 中，则不断减少该次数，直到该次数不在集合 $s\underline{\hspace{0.5em}}set$ 中停止，将次数添加到集合 $s\underline{\hspace{0.5em}}set$ 中，同时将减少次数累加到答案 $ans$ 中。
4. 遍历完哈希表后返回答案 $ans$。

### 思路 1：代码

```Python
class Solution:
    def minDeletions(self, s: str) -> int:
        cnts = Counter(s)
        s_set = set()

        ans = 0
        for key, value in cnts.items():
            while value > 0 and value in s_set:
                value -= 1
                ans += 1
            s_set.add(value)
        
        return ans
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(n)$。
- **空间复杂度**：$O(n)$。

# [1657. 确定两个字符串是否接近](https://leetcode.cn/problems/determine-if-two-strings-are-close/)

- 标签：哈希表、字符串、排序
- 难度：中等

## 题目链接

- [1657. 确定两个字符串是否接近 - 力扣](https://leetcode.cn/problems/determine-if-two-strings-are-close/)

## 题目大意

**描述**：如果可以使用以下操作从一个字符串得到另一个字符串，则认为两个字符串 接近 ：

- 操作 1：交换任意两个现有字符。
  - 例如，`abcde` -> `aecdb`。
- 操作 2：将一个 现有 字符的每次出现转换为另一个现有字符，并对另一个字符执行相同的操作。
  - 例如，`aacabb` -> `bbcbaa`（所有 `a` 转化为 `b`，而所有的 `b` 转换为 `a` ）。

给定两个字符串，$word1$ 和 $word2$。

**要求**：如果 $word1$ 和 $word2$ 接近 ，就返回 $True$；否则，返回 $False$。

**说明**：

- $1 \le word1.length, word2.length \le 10^5$。
- $word1$ 和 $word2$ 仅包含小写英文字母。

**示例**：

- 示例 1：

```python
输入：word1 = "abc", word2 = "bca"
输出：True
解释：2 次操作从 word1 获得 word2 。
执行操作 1："abc" -> "acb"
执行操作 1："acb" -> "bca"
```

- 示例 2：

```python
输入：word1 = "a", word2 = "aa"
输出：False
解释：不管执行多少次操作，都无法从 word1 得到 word2 ，反之亦然。
```

## 解题思路

### 思路 1：模拟

无论是操作 1，还是操作 2，只是对字符位置进行交换，而不会产生或者删除字符。

则我们只需要检查两个字符串的字符种类以及每种字符的个数是否相同即可。

具体步骤如下：

1. 分别使用哈希表 $cnts1$、$cnts2$ 统计每个字符串中的字符种类，每种字符的个数。
2. 判断两者的字符种类是否相等，并且判断每种字符的个数是否相同。
3. 如果字符种类相同，且每种字符的个数完全相同，则返回 $True$，否则，返回 $False$。

### 思路 1：代码

```Python
class Solution:
    def closeStrings(self, word1: str, word2: str) -> bool:
        cnts1 = Counter(word1)
        cnts2 = Counter(word2)

        return cnts1.keys() == cnts2.keys() and sorted(cnts1.values()) == sorted(cnts2.values())
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(max(n1, n2) + |\sum| \times \log | \sum |)$，其中 $n1$、$n2$ 分别为字符串 $word1$、$word2$ 的长度，$\sum$ 为字符集，本题中 $| \sum | = 26$。
- **空间复杂度**：$O(| \sum |)$。

# [1658. 将 x 减到 0 的最小操作数](https://leetcode.cn/problems/minimum-operations-to-reduce-x-to-zero/)

- 标签：数组、哈希表、二分查找、前缀和、滑动窗口
- 难度：中等

## 题目链接

- [1658. 将 x 减到 0 的最小操作数 - 力扣](https://leetcode.cn/problems/minimum-operations-to-reduce-x-to-zero/)

## 题目大意

**描述**：给定一个整数数组 $nums$ 和一个整数 $x$ 。每一次操作时，你应当移除数组 $nums$ 最左边或最右边的元素，然后从 $x$ 中减去该元素的值。请注意，需要修改数组以供接下来的操作使用。

**要求**：如果可以将 $x$ 恰好减到 $0$，返回最小操作数；否则，返回 $-1$。

**说明**：

- $1 \le nums.length \le 10^5$。
- $1 \le nums[i] \le 10^4$。
- $1 \le x \le 10^9$。

**示例**：

- 示例 1：

```python
输入：nums = [1,1,4,2,3], x = 5
输出：2
解释：最佳解决方案是移除后两个元素，将 x 减到 0。
```

- 示例 2：

```python
输入：nums = [3,2,20,1,1,3], x = 10
输出：5
解释：最佳解决方案是移除后三个元素和前两个元素（总共 5 次操作），将 x 减到 0。
```

## 解题思路

### 思路 1：滑动窗口

将 $x$ 减到 $0$ 的最小操作数可以转换为求和等于 $sum(nums) - x$ 的最长连续子数组长度。我们可以维护一个区间和为 $sum(nums) - x$ 的滑动窗口，求出最长的窗口长度。具体做法如下：

令 `target = sum(nums) - x`，使用 $max\underline{\hspace{0.5em}}len$ 维护和等于 $target$ 的最长连续子数组长度。然后用滑动窗口 $window\underline{\hspace{0.5em}}sum$ 来记录连续子数组的和，设定两个指针：$left$、$right$，分别指向滑动窗口的左右边界，保证窗口中的和刚好等于 $target$。

- 一开始，$left$、$right$ 都指向 $0$。
- 向右移动 $right$，将最右侧元素加入当前窗口和 $window\underline{\hspace{0.5em}}sum$ 中。
- 如果 $window\underline{\hspace{0.5em}}sum > target$，则不断右移 $left$，缩小滑动窗口长度，并更新窗口和的最小值，直到 $window\underline{\hspace{0.5em}}sum \le target$。
- 如果 $window\underline{\hspace{0.5em}}sum == target$，则更新最长连续子数组长度。
- 然后继续右移 $right$，直到 $right \ge len(nums)$ 结束。
- 输出 $len(nums) - max\underline{\hspace{0.5em}}len$ 作为答案。
- 注意判断题目中的特殊情况。

### 思路 1：代码

```python
class Solution:
    def minOperations(self, nums: List[int], x: int) -> int:
        target = sum(nums) - x
        size = len(nums)
        if target < 0:
            return -1
        if target == 0:
            return size
        left, right = 0, 0
        window_sum = 0
        max_len = float('-inf')

        while right < size:
            window_sum += nums[right]

            while window_sum > target:
                window_sum -= nums[left]
                left += 1
            if window_sum == target:
                max_len = max(max_len, right - left + 1)
            right += 1
        return len(nums) - max_len if max_len != float('-inf') else -1
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(n)$，其中 $n$ 为数组 $nums$ 的长度。
- **空间复杂度**：$O(1)$。

# [1672. 最富有客户的资产总量](https://leetcode.cn/problems/richest-customer-wealth/)

- 标签：数组、矩阵
- 难度：简单

## 题目链接

- [1672. 最富有客户的资产总量 - 力扣](https://leetcode.cn/problems/richest-customer-wealth/)

## 题目大意

**描述**：给定一个 $m \times n$ 的整数网格 $accounts$，其中 $accounts[i][j]$ 是第 $i$ 位客户在第 $j$ 家银行托管的资产数量。

**要求**：返回最富有客户所拥有的资产总量。

**说明**：

- 客户的资产总量：指的是他们在各家银行托管的资产数量之和。
- 最富有客户：资产总量最大的客户。
- $m == accounts.length$。
- $n == accounts[i].length$。
- $1 \le m, n \le 50$。
- $1 \le accounts[i][j] \le 100$。

**示例**：

- 示例 1：

```python
输入：accounts = [[1,2,3],[3,2,1]]
输出：6
解释：
第 1 位客户的资产总量 = 1 + 2 + 3 = 6
第 2 位客户的资产总量 = 3 + 2 + 1 = 6
两位客户都是最富有的，资产总量都是 6 ，所以返回 6。
```

- 示例 2：

```python
输入：accounts = [[1,5],[7,3],[3,5]]
输出：10
解释：
第 1 位客户的资产总量 = 6
第 2 位客户的资产总量 = 10 
第 3 位客户的资产总量 = 8
第 2 位客户是最富有的，资产总量是 10，随意返回 10。
```

## 解题思路

### 思路 1：直接模拟

1. 使用变量 $max\underline{\hspace{0.5em}}ans$ 存储最富有客户所拥有的资产总量。
2. 遍历所有客户，对于当前客户 $accounts[i]$，统计其拥有的资产总量。
3. 将当前客户的资产总量与 $max\underline{\hspace{0.5em}}ans$ 进行比较，如果大于 $max\underline{\hspace{0.5em}}ans$，则更新 $max\underline{\hspace{0.5em}}ans$ 的值。
4. 遍历完所有客户，最终返回 $max\underline{\hspace{0.5em}}ans$ 作为结果。

### 思路 1：代码

```python
class Solution:
    def maximumWealth(self, accounts: List[List[int]]) -> int:
        max_ans = 0
        for i in range(len(accounts)):
            total = 0
            for j in range(len(accounts[i])):
                total += accounts[i][j]
            if total > max_ans:
                max_ans = total
        return max_ans
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(m \times n)$。其中 $m$ 和 $n$ 分别为二维数组 $accounts$ 的行数和列数。两重循环遍历的时间复杂度为 $O(m * n)$ 。
- **空间复杂度**：$O(1)$。
# [1695. 删除子数组的最大得分](https://leetcode.cn/problems/maximum-erasure-value/)

- 标签：数组、哈希表、滑动窗口
- 难度：中等

## 题目链接

- [1695. 删除子数组的最大得分 - 力扣](https://leetcode.cn/problems/maximum-erasure-value/)

## 题目大意

**描述**：给定一个正整数数组 $nums$，从中删除一个含有若干不同元素的子数组。删除子数组的「得分」就是子数组各元素之和 。

**要求**：返回只删除一个子数组可获得的最大得分。

**说明**：

- **子数组**：如果数组 $b$ 是数组 $a$ 的一个连续子序列，即如果它等于 $a[l],a[l+1],...,a[r]$ ，那么它就是 $a$ 的一个子数组。
- $1 \le nums.length \le 10^5$。
- $1 \le nums[i] \le 10^4$。

**示例**：

- 示例 1：

```python
输入：nums = [4,2,4,5,6]
输出：17
解释：最优子数组是 [2,4,5,6]
```

- 示例 2：

```python
输入：nums = [5,2,1,2,5,2,1,2,5]
输出：8
解释：最优子数组是 [5,2,1] 或 [1,2,5]
```

## 解题思路

### 思路 1：滑动窗口

题目要求的是含有不同元素的连续子数组最大和，我们可以用滑动窗口来做，维护一个不包含重复元素的滑动窗口，计算最大的窗口和。具体方法如下：

- 用滑动窗口 $window$ 来记录不重复的元素个数，$window$ 为哈希表类型。用 $window\underline{\hspace{0.5em}}sum$ 来记录窗口内子数组元素和，$ans$ 用来维护最大子数组和。设定两个指针：$left$、$right$，分别指向滑动窗口的左右边界，保证窗口中没有重复元素。

- 一开始，$left$、$right$ 都指向 $0$。
- 将最右侧数组元素 $nums[right]$ 加入当前窗口 $window$ 中，记录该元素个数。
- 如果该窗口中该元素的个数多于 $1$ 个，即 $window[s[right]] > 1$，则不断右移 $left$，缩小滑动窗口长度，并更新窗口中对应元素的个数，直到 $window[s[right]] \le 1$。
- 维护更新无重复元素的最大子数组和。然后右移 $right$，直到 $right \ge len(nums)$ 结束。
- 输出无重复元素的最大子数组和。

### 思路 1：代码

```python
class Solution:
    def maximumUniqueSubarray(self, nums: List[int]) -> int:
        window_sum = 0
        left, right = 0, 0
        window = dict()
        ans = 0
        while right < len(nums):
            window_sum += nums[right]
            if nums[right] not in window:
                window[nums[right]] = 1
            else:
                window[nums[right]] += 1

            while window[nums[right]] > 1:
                window[nums[left]] -= 1
                window_sum -= nums[left]
                left += 1
            ans = max(ans, window_sum)
            right += 1
        return ans
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(n)$，其中 $n$ 为数组 $nums$ 的长度。
- **空间复杂度**：$O(n)$。

# [1698. 字符串的不同子字符串个数](https://leetcode.cn/problems/number-of-distinct-substrings-in-a-string/)

- 标签：字典树、字符串、后缀数组、哈希函数、滚动哈希
- 难度：中等

## 题目链接

- [1698. 字符串的不同子字符串个数 - 力扣](https://leetcode.cn/problems/number-of-distinct-substrings-in-a-string/)

## 题目大意

给定一个字符串 `s`。

要求：返回 `s` 的不同子字符串的个数。

注意：字符串的「子字符串」是由原字符串删除开头若干个字符（可能是 0 个）并删除结尾若干个字符（可能是 0 个）形成的字符串。

## 解题思路

构建一颗字典树。分别将原字符串删除开头若干个字符的子字符串依次插入到字典树中。

每次插入过程中碰到字典树中没有的字符节点时，说明此时插入的字符串可作为新的子字符串。

我们可以通过统计插入过程中新建字符节点的次数的方式来获取不同子字符串的个数。

## 代码

```python
class Trie:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.children = dict()
        self.isEnd = False


    def insert(self, word: str) -> int:
        """
        Inserts a word into the trie.
        """
        cur = self
        cnt = 0
        for ch in word:
            if ch not in cur.children:
                cur.children[ch] = Trie()
                cnt += 1
            cur = cur.children[ch]
        cur.isEnd = True
        return cnt


class Solution:
    def countDistinct(self, s: str) -> int:
        trie_tree = Trie()
        cnt = 0
        for i in range(len(s)):
            cnt += trie_tree.insert(s[i:])
        return cnt
```

# [1710. 卡车上的最大单元数](https://leetcode.cn/problems/maximum-units-on-a-truck/)

- 标签：贪心、数组、排序
- 难度：简单

## 题目链接

- [1710. 卡车上的最大单元数 - 力扣](https://leetcode.cn/problems/maximum-units-on-a-truck/)

## 题目大意

**描述**：现在需要将一些箱子装在一辆卡车上。给定一个二维数组 $boxTypes$，其中 $boxTypes[i] = [numberOfBoxesi, numberOfUnitsPerBoxi]$。

$numberOfBoxesi$ 是类型 $i$ 的箱子的数量。$numberOfUnitsPerBoxi$ 是类型 $i$ 的每个箱子可以装载的单元数量。

再给定一个整数 $truckSize$ 表示一辆卡车上可以装载箱子的最大数量。只要箱子数量不超过 $truckSize$，你就可以选择任意箱子装到卡车上。

**要求**：返回卡车可以装载的最大单元数量。

**说明**：

- $1 \le boxTypes.length \le 1000$。
- $1 \le numberOfBoxesi, numberOfUnitsPerBoxi \le 1000$。
- $1 \le truckSize \le 106$。

**示例**：

- 示例 1：

```python
输入：boxTypes = [[1,3],[2,2],[3,1]], truckSize = 4
输出：8
解释
箱子的情况如下：
- 1 个第一类的箱子，里面含 3 个单元。
- 2 个第二类的箱子，每个里面含 2 个单元。
- 3 个第三类的箱子，每个里面含 1 个单元。
可以选择第一类和第二类的所有箱子，以及第三类的一个箱子。
单元总数 = (1 * 3) + (2 * 2) + (1 * 1) = 8
```

- 示例 2：

```python
输入：boxTypes = [[5,10],[2,5],[4,7],[3,9]], truckSize = 10
输出：91
```

## 解题思路

### 思路 1：贪心算法

题目中，一辆卡车上可以装载箱子的最大数量是固定的（$truckSize$），那么如果想要使卡车上装载的单元数量最大，就应该优先选取装载单元数量多的箱子。

所以，从贪心算法的角度来考虑，我们应该按照每个箱子可以装载的单元数量对数组 $boxTypes$ 从大到小排序。然后优先选取装载单元数量多的箱子。 

下面我们使用贪心算法三步走的方法解决这道题。

1. **转换问题**：将原问题转变为，在 $truckSize$ 的限制下，当选取完装载单元数量最多的箱子 $box$ 之后，再解决剩下箱子（$truckSize - box[0]$）的选择问题（子问题）。
2. **贪心选择性质**：对于当前 $truckSize$，优先选取装载单元数量最多的箱子。
3. **最优子结构性质**：在上面的贪心策略下，当前 $truckSize$ 的贪心选择 + 剩下箱子的子问题最优解，就是全局最优解。也就是说在贪心选择的方案下，能够使得卡车可以装载的单元数量达到最大。

使用贪心算法的解决步骤描述如下：

1. 对数组 $boxTypes$ 按照每个箱子可以装载的单元数量从大到小排序。使用变量 $res$ 记录卡车可以装载的最大单元数量。
2. 遍历数组 $boxTypes$，对于当前种类的箱子 $box$：
   1. 如果 $truckSize > box[0]$，说明当前种类箱子可以全部装载。则答案数量加上该种箱子的单元总数，即 $box[0] \times box[1]$，并且最大数量 $truckSize$ 减去装载的箱子数。
   2. 如果 $truckSize \le box[0]$，说明当前种类箱子只能部分装载。则答案数量加上 $truckSize \times box[1]$，并跳出循环。
3. 最后返回答案 $res$。

### 思路 1：代码

```python
class Solution:
    def maximumUnits(self, boxTypes: List[List[int]], truckSize: int) -> int:
        boxTypes.sort(key=lambda x:x[1], reverse=True)
        res = 0
        for box in boxTypes:
            if truckSize > box[0]:
                res += box[0] * box[1]
                truckSize -= box[0]
            else:
                res += truckSize * box[1]
                break
        return res
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(n \times \log n)$，其中 $n$ 是数组 $boxTypes$ 的长度。
- **空间复杂度**：$O(\log n)$。
# [1716. 计算力扣银行的钱](https://leetcode.cn/problems/calculate-money-in-leetcode-bank/)

- 标签：数学
- 难度：简单

## 题目链接

- [1716. 计算力扣银行的钱 - 力扣](https://leetcode.cn/problems/calculate-money-in-leetcode-bank/)

## 题目大意

**描述**：Hercy 每天都往力扣银行里存钱。

最开始，他在周一的时候存入 $1$ 块钱。从周二到周日，他每天都比前一天多存入 $1$ 块钱。在接下来的每个周一，他都会比前一个周一多存入 $1$ 块钱。

给定一个整数 $n$。

**要求**：计算在第 $n$ 天结束的时候，Hercy 在力扣银行中总共存了多少块钱。

**说明**：

- $1 \le n \le 1000$。

**示例**：

- 示例 1：

```python
输入：n = 4
输出：10
解释：第 4 天后，总额为 1 + 2 + 3 + 4 = 10。
```

- 示例 2：

```python
输入：n = 10
输出：37
解释：第 10 天后，总额为 (1 + 2 + 3 + 4 + 5 + 6 + 7) + (2 + 3 + 4) = 37 。注意到第二个星期一，Hercy 存入 2 块钱。
```

## 解题思路

### 思路 1：暴力模拟

1. 记录当前周 $week$ 和当前周的当前天数 $day$。
2. 按照题目要求，每天增加 $1$ 块钱，每周一比上周一增加 $1$ 块钱。这样，每天存钱数为 $week + day - 1$。
3. 将每天存的钱数累加起来即为答案。

### 思路 1：代码

```python
class Solution:
    def totalMoney(self, n: int) -> int:
        weak, day = 1, 1
        ans = 0
        for i in range(n):
            ans += weak + day - 1
            day += 1
            if day == 8:
                day = 1
                weak += 1
        
        return ans
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(n)$。
- **空间复杂度**：$O(1)$。

### 思路 2：等差数列计算优化

每周一比上周一增加 $1$ 块钱，则每周七天存钱总数比上一周多 $7$ 块钱。所以每周存的钱数是一个等差数列。我们可以通过高斯求和公式求出所有整周存的钱数，再计算出剩下天数存的钱数，两者相加即为答案。

### 思路 2：代码

```python
class Solution:
    def totalMoney(self, n: int) -> int:
        week_cnt = n // 7
        weak_first_money = (1 + 7) * 7 // 2
        weak_last_money = weak_first_money + 7 * (week_cnt - 1)
        week_ans =  (weak_first_money + weak_last_money) * week_cnt // 2

        day_cnt = n % 7
        day_first_money = 1 + week_cnt
        day_last_money = day_first_money + day_cnt - 1
        day_ans = (day_first_money + day_last_money) * day_cnt // 2
        
        return week_ans + day_ans
```

### 思路 2：复杂度分析

- **时间复杂度**：$O(1)$。
- **空间复杂度**：$O(1)$。
# [1720. 解码异或后的数组](https://leetcode.cn/problems/decode-xored-array/)

- 标签：位运算、数组
- 难度：简单

## 题目链接

- [1720. 解码异或后的数组 - 力扣](https://leetcode.cn/problems/decode-xored-array/)

## 题目大意

n 个非负整数构成数组 arr，经过编码后变为长度为 n-1 的整数数组 encoded，其中 `encoded[i] = arr[i] XOR arr[i+1]`。例如 arr = [1, 0, 2, 1] 经过编码后变为 encoded = [1, 2, 3]。

现在给定编码后的数组 encoded 和原数组 arr 的第一个元素 arr[0]。要求返回原数组 arr。

## 解题思路

首先要了解异或的性质：

- 异或运算满足交换律和结合律。
  - 交换律：`a^b = b^a`
  - 结合律：`(a^b)^c = a^(b^c)`
- 任何整数和自身做异或运算结果都为 0，即 `x^x = 0`。
- 任何整数和 0 做异或运算结果都为其本身，即 `x^0 = 0`。

已知当 $1 \le i \le n$ 时，有 `encoded[i-1] = arr[i-1] XOR arr[i]`。两边同时「异或」上 arr[i-1]。得：

- `encoded[i-1] XOR arr[i-1] = arr[i-1] XOR arr[i] XOR arr[i-1]`
- `encoded[i-1] XOR arr[i-1] = arr[i] XOR 0`
- `encoded[i-1] XOR arr[i-1] = arr[i]`

所以就可以根据所得结论 `arr[i] = encoded[i-1] XOR arr[i-1]` 模拟得出原数组 arr。

## 代码

```python
class Solution:
    def decode(self, encoded: List[int], first: int) -> List[int]:
        n = len(encoded) + 1
        arr = [0] * n
        arr[0] = first
        for i in range(1, n):
            arr[i] = encoded[i-1] ^ arr[i-1]
        return arr
```

# [1726. 同积元组](https://leetcode.cn/problems/tuple-with-same-product/)

- 标签：数组、哈希表
- 难度：中等

## 题目链接

- [1726. 同积元组 - 力扣](https://leetcode.cn/problems/tuple-with-same-product/)

## 题目大意

**描述**：给定一个由不同正整数组成的数组 $nums$。

**要求**：返回满足 $a \times b = c \times d$ 的元组 $(a, b, c, d)$ 的数量。其中 $a$、$b$、$c$ 和 $d$ 都是 $nums$ 中的元素，且 $a \ne b \ne c \ne d$。

**说明**：

- $1 \le nums.length \le 1000$。
- $1 \le nums[i] \le 10^4$。
- $nums$ 中的所有元素互不相同。

**示例**：

- 示例 1：

```python
输入：nums = [2,3,4,6]
输出：8
解释：存在 8 个满足题意的元组：
(2,6,3,4) , (2,6,4,3) , (6,2,3,4) , (6,2,4,3)
(3,4,2,6) , (4,3,2,6) , (3,4,6,2) , (4,3,6,2)
```

- 示例 2：

```python
输入：nums = [1,2,4,5,10]
输出：16
解释：存在 16 个满足题意的元组：
(1,10,2,5) , (1,10,5,2) , (10,1,2,5) , (10,1,5,2)
(2,5,1,10) , (2,5,10,1) , (5,2,1,10) , (5,2,10,1)
(2,10,4,5) , (2,10,5,4) , (10,2,4,5) , (10,2,5,4)
(4,5,2,10) , (4,5,10,2) , (5,4,2,10) , (5,4,10,2)
```

## 解题思路

### 思路 1：哈希表 + 数学

1. 二重循环遍历数组 $nums$，使用哈希表 $cnts$ 记录下所有不同 $nums[i] \times nums[j]$ 的结果。
2. 因为满足 $a \times b = c \times d$ 的元组 $(a, b, c, d)$ 可以按照不同顺序进行组和，所以对于 $x$ 个 $nums[i] \times nums[j]$，就有 $C_x^2$ 种组和方法。
3. 遍历哈希表 $cnts$ 中所有值 $value$，将不同组和的方法数累积到答案 $ans$ 中。
4. 遍历完返回答案 $ans$。

### 思路 1：代码

```Python
class Solution:
    def tupleSameProduct(self, nums: List[int]) -> int:
        cnts = Counter()
        size = len(nums)
        for i in range(size):
            for j in range(i + 1, size):
                product = nums[i] * nums[j]
                cnts[product] += 1
        
        ans = 0
        for key, value in cnts.items():
            ans += value * (value - 1) * 4
        
        return ans
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(n^2)$，其中 $n$ 表示数组 $nums$ 的长度。
- **空间复杂度**：$O(n^2)$。

# [1736. 替换隐藏数字得到的最晚时间](https://leetcode.cn/problems/latest-time-by-replacing-hidden-digits/)

- 标签：贪心、字符串
- 难度：简单

## 题目链接

- [1736. 替换隐藏数字得到的最晚时间 - 力扣](https://leetcode.cn/problems/latest-time-by-replacing-hidden-digits/)

## 题目大意

**描述**：给定一个字符串 $time$，格式为 `hh:mm`（小时：分钟），其中某几位数字被隐藏（用 `?` 表示）。

**要求**：替换 $time$ 中隐藏的数字，返回你可以得到的最晚有效时间。

**说明**：

- **有效时间**： `00:00` 到 `23:59` 之间的所有时间，包括 `00:00` 和 `23:59`。
- $time$ 的格式为 `hh:mm`。
- 题目数据保证你可以由输入的字符串生成有效的时间。

**示例**：

- 示例 1：

```python
输入：time = "2?:?0"
输出："23:50"
解释：以数字 '2' 开头的最晚一小时是 23 ，以 '0' 结尾的最晚一分钟是 50。
```

- 示例 2：

```python
输入：time = "0?:3?"
输出："09:39"
```

## 解题思路

### 思路 1：贪心算法

为了使有效时间尽可能晚，我们可以从高位到低位依次枚举所有符号为 `?` 的字符。在保证时间有效的前提下，每一位上取最大值，并进行保存。具体步骤如下：

- 如果第 $1$ 位为 `?`：
  - 如果第 $2$ 位已经确定，并且范围在 $[4, 9]$ 中间，则第 $1$ 位最大为 $1$；
  - 否则第 $1$ 位最大为 $2$。
- 如果第 $2$ 位为 `?`：
  - 如果第 $1$ 位上值为 $2$，则第 $2$ 位最大可以为 $3$；
  - 否则第 $2$ 位最大为 $9$。
- 如果第 $3$ 位为 `?`：
  - 第 $3$ 位最大可以为 $5$。
- 如果第 $4$ 位为 `?`：
  - 第 $4$ 位最大可以为 $9$。

### 思路 1：代码

```python
class Solution:
    def maximumTime(self, time: str) -> str:
        time_list = list(time)
        if time_list[0] == '?':
            if '4' <= time_list[1] <= '9':
                time_list[0] = '1'
            else:
                time_list[0] = '2'

        if time_list[1] == '?':
            if time_list[0] == '2':
                time_list[1] = '3'
            else:
                time_list[1] = '9'

        if time_list[3] == '?':
            time_list[3] = '5'
        
        if time_list[4] == '?':
            time_list[4] = '9'
            
        return "".join(time_list)
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(1)$。
- **空间复杂度**：$O(1)$。

# [1742. 盒子中小球的最大数量](https://leetcode.cn/problems/maximum-number-of-balls-in-a-box/)

- 标签：哈希表、数学、计数
- 难度：简单

## 题目链接

- [1742. 盒子中小球的最大数量 - 力扣](https://leetcode.cn/problems/maximum-number-of-balls-in-a-box/)

## 题目大意

**描述**：给定两个整数 $lowLimit$ 和 $highLimt$，代表 $n$ 个小球的编号（包括 $lowLimit$ 和 $highLimit$，即 $n == highLimit = lowLimit + 1$）。另外有无限个盒子。

现在的工作是将每个小球放入盒子中，其中盒子的编号应当等于小球编号上每位数字的和。例如，编号 $321$ 的小球应当放入编号 $3 + 2 + 1 = 6$ 的盒子，而编号 $10$ 的小球应当放入编号 $1 + 0 = 1$ 的盒子。

**要求**：返回放有最多小球的盒子中的小球数量。如果有多个盒子都满足放有最多小球，只需返回其中任一盒子的小球数量。

**说明**：

- $1 \le lowLimit \le highLimit \le 10^5$。

**示例**：

- 示例 1：

```python
输入：lowLimit = 1, highLimit = 10
输出：2
解释：
盒子编号：1 2 3 4 5 6 7 8 9 10 11 ...
小球数量：2 1 1 1 1 1 1 1 1 0  0  ...
编号 1 的盒子放有最多小球，小球数量为 2。
```

- 示例 2：

```python
输入：lowLimit = 5, highLimit = 15
输出：2
解释：
盒子编号：1 2 3 4 5 6 7 8 9 10 11 ...
小球数量：1 1 1 1 2 2 1 1 1 0  0  ...
编号 5 和 6 的盒子放有最多小球，每个盒子中的小球数量都是 2。
```

## 解题思路

### 思路 1：动态规划 + 数位 DP

将 $lowLimit$、$highLimit$ 转为字符串 $s1$、$s2$，并将 $s1$ 补上前导 $0$，令其与 $s2$ 长度一致。定义递归函数 `def dfs(pos, remainTotal, isMaxLimit, isMinLimit):` 表示构造第 $pos$ 位及之后剩余数位和为 $remainTotal$ 的合法方案数。

因为数据范围为 $[1, 10^5]$，对应数位和范围为 $[1, 45]$。因此我们可以枚举所有的数位和，并递归调用 `dfs(i, remainTotal, isMaxLimit, isMinLimit)`，求出不同数位和对应的方案数，并求出最大方案数。

接下来按照如下步骤进行递归。

1. 从 `dfs(0, i, True, True)` 开始递归。 `dfs(0, i, True, True)` 表示：
	1. 从位置 $0$ 开始构造。
	2. 剩余数位和为 $i$。
	3. 开始时当前数位最大值受到最高位数位的约束。
	4. 开始时当前数位最小值受到最高位数位的约束。

2. 如果剩余数位和小于 $0$，说明当前方案不符合要求，则返回方案数 $0$。

3. 如果遇到  $pos == len(s)$，表示到达数位末尾，此时：
	1. 如果剩余数位和 $remainTotal$ 等于 $0$，说明当前方案符合要求，则返回方案数 $1$。
	2. 如果剩余数位和 $remainTotal$ 不等于 $0$，说明当前方案不符合要求，则返回方案数 $0$。

4. 如果 $pos \ne len(s)$，则定义方案数 $ans$，令其等于 $0$，即：`ans = 0`。
5. 如果遇到 $isNum == False$，说明之前位数没有填写数字，当前位可以跳过，这种情况下方案数等于 $pos + 1$ 位置上没有受到 $pos$ 位的约束，并且之前没有填写数字时的方案数，即：`ans = dfs(i + 1, state, False, False)`。
6. 根据 $isMaxLimit$ 和 $isMinLimit$ 来决定填当前位数位所能选择的最小数字（$minX$）和所能选择的最大数字（$maxX$）。

7. 然后根据 $[minX, maxX]$ 来枚举能够填入的数字 $d$。
8. 方案数累加上当前位选择 $d$ 之后的方案数，即：`ans += dfs(pos + 1, remainTotal - d, isMaxLimit and d == maxX, isMinLimit and d == minX)`。
	1. `remainTotal - d` 表示当前剩余数位和减去 $d$。
	2. `isMaxLimit and d == maxX` 表示 $pos + 1$ 位最大值受到之前 $pos$ 位限制。
	3. `isMinLimit and d == maxX` 表示 $pos + 1$ 位最小值受到之前 $pos$ 位限制。
9. 最后返回所有 `dfs(0, i, True, True)` 中最大的方案数即可。

### 思路 1：代码

```python
class Solution:
    def countBalls(self, lowLimit: int, highLimit: int) -> int:
        s1, s2 = str(lowLimit), str(highLimit)

        m, n = len(s1), len(s2)
        if m < n:
            s1 = '0' * (n - m) + s1
        
        @cache
        # pos: 第 pos 个数位
        # remainTotal: 表示剩余数位和
        # isMaxLimit: 表示是否受到上限选择限制。如果为真，则第 pos 位填入数字最多为 s2[pos]；如果为假，则最大可为 9。
        # isMinLimit: 表示是否受到下限选择限制。如果为真，则第 pos 位填入数字最小为 s1[pos]；如果为假，则最小可为 0。
        def dfs(pos, remainTotal, isMaxLimit, isMinLimit):
            if remainTotal < 0:
                return 0
            if pos == n:
                # remainTotal 为 0，则表示当前方案符合要求
                return int(remainTotal == 0)
            
            ans = 0
            # 如果前一位没有填写数字，或受到选择限制，则最小可选择数字为 s1[pos]，否则最少为 0（可以含有前导 0）。
            minX = int(s1[pos]) if isMinLimit else 0
            # 如果受到选择限制，则最大可选择数字为 s[pos]，否则最大可选择数字为 9。
            maxX = int(s2[pos]) if isMaxLimit else 9
            
            # 枚举可选择的数字
            for d in range(minX, maxX + 1): 
                ans += dfs(pos + 1, remainTotal - d, isMaxLimit and d == maxX, isMinLimit and d == minX)
            return ans

        ans = 0
        for i in range(46):
            ans = max(ans, dfs(0, i, True, True))
        return ans
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(n \times \log n \times 45)$。
- **空间复杂度**：$O(\log n)$。
# [1749. 任意子数组和的绝对值的最大值](https://leetcode.cn/problems/maximum-absolute-sum-of-any-subarray/)

- 标签：数组、动态规划
- 难度：中等

## 题目链接

- [1749. 任意子数组和的绝对值的最大值 - 力扣](https://leetcode.cn/problems/maximum-absolute-sum-of-any-subarray/)

## 题目大意

**描述**：给定一个整数数组 $nums$。

**要求**：找出 $nums$ 中「和的绝对值」最大的任意子数组（可能为空），并返回最大值。

**说明**：

- **子数组 $[nums_l, nums_{l+1}, ..., nums_{r-1}, nums_{r}]$ 的和的绝对值**：$abs(nums_l + nums_{l+1} + ... + nums_{r-1} + nums_{r})$。
- $abs(x)$ 定义如下：
  - 如果 $x$ 是负整数，那么 $abs(x) = -x$。
  - 如果 $x$ 是非负整数，那么 $abs(x) = x$。

- $1 \le nums.length \le 10^5$。
- $-10^4 \le nums[i] \le 10^4$。

**示例**：

- 示例 1：

```python
输入：nums = [1,-3,2,3,-4]
输出：5
解释：子数组 [2,3] 和的绝对值最大，为 abs(2+3) = abs(5) = 5。
```

- 示例 2：

```python
输入：nums = [2,-5,1,-4,3,-2]
输出：8
解释：子数组 [-5,1,-4] 和的绝对值最大，为 abs(-5+1-4) = abs(-8) = 8。
```

## 解题思路

### 思路 1：动态规划

子数组和的绝对值的最大值，可能来自于「连续子数组的最大和」，也可能来自于「连续子数组的最小和」。

而求解「连续子数组的最大和」，我们可以参考「[0053. 最大子数组和](https://leetcode.cn/problems/maximum-subarray/)」的做法，使用一个变量 $mmax$ 来表示以第 $i$ 个数结尾的连续子数组的最大和。使用另一个变量 $mmin$ 来表示以第 $i$ 个数结尾的连续子数组的最小和。然后取两者绝对值的最大值为答案 $ans$。

具体步骤如下：

1. 遍历数组 $nums$，对于当前元素 $nums[i]$：
   1. 如果 $mmax < 0$，则「第 $i - 1$ 个数结尾的连续子数组的最大和」+「第 $i$  个数的值」<「第 $i$ 个数的值」，所以 $mmax$ 应取「第 $i$ 个数的值」，即：$mmax = nums[i]$。
   2. 如果 $mmax \ge 0$ ，则「第 $i - 1$ 个数结尾的连续子数组的最大和」 +「第 $i$  个数的值」 >= 第 $i$ 个数的值，所以 $mmax$ 应取「第 $i - 1$ 个数结尾的连续子数组的最大和」 +「第 $i$  个数的值」，即：$mmax = mmax + nums[i]$。
   3. 如果 $mmin > 0$，则「第 $i - 1$ 个数结尾的连续子数组的最大和」+「第 $i$  个数的值」>「第 $i$ 个数的值」，所以 $mmax$ 应取「第 $i$ 个数的值」，即：$mmax = nums[i]$。
   4. 如果 $mmin \le 0$ ，则「第 $i - 1$ 个数结尾的连续子数组的最大和」 +「第 $i$  个数的值」 <= 第 $i$ 个数的值，所以 $mmax$ 应取「第 $i - 1$ 个数结尾的连续子数组的最大和」 +「第 $i$  个数的值」，即：$mmin = mmin + nums[i]$。
   5. 维护答案 $ans$，将 $mmax$ 和 $mmin$ 绝对值的最大值与 $ans$ 进行比较，并更新 $ans$。
2. 遍历完返回答案 $ans$。

### 思路 1：代码

```python
class Solution:
    def maxAbsoluteSum(self, nums: List[int]) -> int:
        ans = 0
        mmax, mmin = 0, 0
        for num in nums:
            mmax = max(mmax, 0) + num
            mmin = min(mmin, 0) + num
            ans = max(ans, mmax, -mmin)

        return ans
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(n)$。
- **空间复杂度**：$O(1)$。

# [1763. 最长的美好子字符串](https://leetcode.cn/problems/longest-nice-substring/)

- 标签：位运算、哈希表、字符串、分治、滑动窗口
- 难度：简单

## 题目链接

- [1763. 最长的美好子字符串 - 力扣](https://leetcode.cn/problems/longest-nice-substring/)

## 题目大意

**描述**： 给定一个字符串 $s$。

**要求**：返回 $s$ 最长的美好子字符串。 

**说明**：

- **美好字符串**：当一个字符串 $s$ 包含的每一种字母的大写和小写形式同时出现在 $s$ 中，就称这个字符串 $s$ 是美好字符串。
- $1 \le s.length \le 100$。

**示例**：

- 示例 1：

```python
输入：s = "YazaAay"
输出："aAa"
解释："aAa" 是一个美好字符串，因为这个子串中仅含一种字母，其小写形式 'a' 和大写形式 'A' 也同时出现了。
"aAa" 是最长的美好子字符串。
```

- 示例 2：

```python
输入：s = "Bb"
输出："Bb"
解释："Bb" 是美好字符串，因为 'B' 和 'b' 都出现了。整个字符串也是原字符串的子字符串。
```

## 解题思路

### 思路 1：枚举

字符串 $s$ 的范围为 $[1, 100]$，长度较小，我们可以枚举所有的子串，判断该子串是否为美好字符串。

由于大小写英文字母各有 $26$ 位，则我们可以利用二进制来标记某字符是否在子串中出现过，我们使用 $lower$ 标记子串中出现过的小写字母，使用 $upper$ 标记子串中出现过的大写字母。如果满足 $lower == upper$，则说明该子串为美好字符串。

具体解法步骤如下：

1. 使用二重循环遍历字符串。对于子串 $s[i]…s[j]$，使用 $lower$ 标记子串中出现过的小写字母，使用 $upper$ 标记子串中出现过的大写字母。
2. 如果 $s[j]$ 为小写字母，则 $lower$ 对应位置标记为出现过该小写字母，即：`lower |= 1 << (ord(s[j]) - ord('a'))`。
3. 如果 $s[j]$ 为大写字母，则 $upper$ 对应位置标记为出现过该小写字母，即：`upper |= 1 << (ord(s[j]) - ord('A'))`。
4. 判断当前子串对应 $lower$ 和 $upper$ 是否相等，如果相等，并且子串长度大于记录的最长美好字符串长度，则更新最长美好字符串长度。
5. 遍历完返回记录的最长美好字符串长度。

### 思路 1：代码

```Python
class Solution:
    def longestNiceSubstring(self, s: str) -> str:
        size = len(s)
        max_pos, max_len = 0, 0
        for i in range(size):
            lower, upper = 0, 0
            for j in range(i, size):
                if s[j].islower():
                    lower |= 1 << (ord(s[j]) - ord('a'))
                else:
                    upper |= 1 << (ord(s[j]) - ord('A'))
                if lower == upper and j - i + 1 > max_len:
                    max_len = j - i + 1
                    max_pos = i
        return s[max_pos: max_pos + max_len]
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(n^2)$，其中 $n$ 为字符串 $s$ 的长度。
- **空间复杂度**：$O(1)$。

# [1779. 找到最近的有相同 X 或 Y 坐标的点](https://leetcode.cn/problems/find-nearest-point-that-has-the-same-x-or-y-coordinate/)

- 标签：数组
- 难度：简单

## 题目链接

- [1779. 找到最近的有相同 X 或 Y 坐标的点 - 力扣](https://leetcode.cn/problems/find-nearest-point-that-has-the-same-x-or-y-coordinate/)

## 题目大意

**描述**：给定两个整数 `x` 和 `y`，表示笛卡尔坐标系下的 `(x, y)` 点。再给定一个数组 `points`，其中 `points[i] = [ai, bi]`，表示在 `(ai, bi)` 处有一个点。当一个点与 `(x, y)` 拥有相同的 `x` 坐标或者拥有相同的 `y` 坐标时，我们称这个点是有效的。

**要求**：返回数组中距离 `(x, y)` 点出曼哈顿距离最近的有效点在 `points` 中的下标位置。如果有多个最近的有效点，则返回下标最小的一个。如果没有有效点，则返回 `-1`。

**说明**：

- **曼哈顿距离**：`(x1, y1)` 和 `(x2, y2)` 之间的曼哈顿距离为 `abs(x1 - x2) + abs(y1 - y2)` 。
- $1 \le points.length \le 10^4$。
- $points[i].length == 2$。
- $1 \le x, y, ai, bi \le 10^4$。

**示例**：

- 示例 1：

```python
输入：x = 3, y = 4, points = [[1, 2], [3, 1], [2, 4], [2, 3], [4, 4]]
输出：2
解释：在所有点中 [3, 1]、[2, 4]、[4, 4] 为有效点。其中 [2, 4]、[4, 4] 距离 [3, 4] 曼哈顿距离最近，都为 1。[2, 4] 下标最小，所以返回 2。
```

## 解题思路

### 思路 1：

- 使用 `min_dist` 记录下有效点中最近的曼哈顿距离，初始化为 `float('inf')`。使用 `min_index` 记录下符合要求的最小下标。
- 遍历 `points` 数组，遇到有效点之后计算一下当前有效点与 `(x, y)` 的曼哈顿距离，并判断更新一下有效点中最近的曼哈顿距离 `min_dist` 和符合要求的最小下标 `min_index`。
- 遍历完之后，判断一下 `min_dist` 是否等于 `float('inf')`。如果等于，说明没有找到有效点，则返回 `-1`。如果不等于，则返回符合要求的最小下标 `min_index`。

## 代码

### 思路 1 代码：

```python
class Solution:
    def nearestValidPoint(self, x: int, y: int, points: List[List[int]]) -> int:
        min_dist = float('inf')
        min_index = 0
        for i in range(len(points)):
            if points[i][0] == x or points[i][1] == y:
                dist = abs(points[i][0] - x) + abs(points[i][1] - y)
                if dist < min_dist:
                    min_dist = dist
                    min_index = i

        if min_dist == float('inf'):
            return -1
        return min_index
```

# [1790. 仅执行一次字符串交换能否使两个字符串相等](https://leetcode.cn/problems/check-if-one-string-swap-can-make-strings-equal/)

- 标签：哈希表、字符串、计数
- 难度：简单

## 题目链接

- [1790. 仅执行一次字符串交换能否使两个字符串相等 - 力扣](https://leetcode.cn/problems/check-if-one-string-swap-can-make-strings-equal/)

## 题目大意

**描述**：给定两个长度相等的字符串 `s1` 和 `s2`。

已知一次「字符串交换操作」步骤如下：选出某个字符串中的两个下标（不一定要相同），并交换这两个下标所对应的字符。

**要求**：如果对其中一个字符串执行最多一次字符串交换可以使两个字符串相等，则返回 `True`；否则返回 `False`。

**说明**：

- $1 \le s1.length, s2.length \le 100$。
- $s1.length == s2.length$。
- `s1` 和 `s2` 仅由小写英文字母组成。

**示例**：

- 示例 1：

```python
给定：s1 = "bank", s2 = "kanb"
输出：True
解释：交换 s1 中的第一个和最后一个字符可以得到 "kanb"，与 s2 相同
```

## 解题思路

### 思路 1：

- 用一个变量 `diff_cnt` 记录两个字符串中对应位置上出现不同字符的次数。用 `c1`、`c2` 记录第一次出现不同字符时两个字符串对应位置上的字符。
- 遍历两个字符串，对于第 `i` 个位置的字符 `s1[i]` 和 `s2[i]`：
  - 如果 `s1[i] == s2[i]`，继续判断下一个位置。
  - 如果 `s1[i] != s2[i]`，则出现不同字符的次数加 `1`。
  - 如果出现不同字符的次数等于 `1`，则记录第一次出现不同字符时两个字符串对应位置上的字符。
  - 如果出现不同字符的次数等于 `2`，则判断第一次出现不同字符时两个字符串对应位置上的字符与当前位置字符交换之后是否相等。如果不等，则说明交换之后 `s1` 和 `s2` 不相等，返回 `False`。如果相等，则继续判断下一个位置。
  - 如果出现不同字符的次数超过 `2`，则不符合最多一次字符串交换的要求，返回 `False`。
- 如果遍历完，出现不同字符的次数为 `0` 或者 `2`，为 `0` 说明无需交换，本身 `s1` 和 `s2` 就是相等的，为 `2` 说明交换一次字符串之后  `s1` 和 `s2`  相等，此时返回 `True`。否则返回 `False`。

## 代码

### 思路 1 代码：

```python
class Solution:
    def areAlmostEqual(self, s1: str, s2: str) -> bool:
        size = len(s1)
        diff_cnt = 0
        c1, c2 = None, None
        for i in range(size):
            if s1[i] == s2[i]:
                continue
            diff_cnt += 1
            if diff_cnt == 1:
                c1 = s1[i]
                c2 = s2[i]
            elif diff_cnt == 2:
                if c1 != s2[i] or c2 != s1[i]:
                    return False
            else:
                return False

        return diff_cnt == 0 or diff_cnt == 2
```

# [1791. 找出星型图的中心节点](https://leetcode.cn/problems/find-center-of-star-graph/)

- 标签：图
- 难度：简单

## 题目链接

- [1791. 找出星型图的中心节点 - 力扣](https://leetcode.cn/problems/find-center-of-star-graph/)

## 题目大意

**描述**：有一个无向的行型图，由 $n$ 个编号 $1 \sim n$  的节点组成。星型图有一个中心节点，并且恰好有 $n - 1$ 条边将中心节点与其他每个节点连接起来。

给定一个二维整数数组 $edges$，其中 $edges[i] = [u_i, v_i]$ 表示节点 $u_i$ 与节点 $v_i$ 之间存在一条边。

**要求**：找出并返回该星型图的中心节点。

**说明**：

- $3 \le n \le 10^5$。
- $edges.length == n - 1$。
- $edges[i].length == 2$。
- $1 \le ui, vi \le n$。
- $ui \ne vi$。
- 题目数据给出的 $edges$ 表示一个有效的星型图。

**示例**：

- 示例 1：

![](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2021/03/14/star_graph.png)

```python
输入：edges = [[1,2],[2,3],[4,2]]
输出：2
解释：如上图所示，节点 2 与其他每个节点都相连，所以节点 2 是中心节点。
```

- 示例 2：

```python
输入：edges = [[1,2],[5,1],[1,3],[1,4]]
输出：1
```

## 解题思路

### 思路 1：求度数

根据题意可知：中心节点恰好有 $n - 1$ 条边将中心节点与其他每个节点连接起来，那么中心节点的度数一定为 $n - 1$。则我们可以遍历边集数组 $edges$，统计出每个节点 $u$ 的度数 $degrees[u]$。最后返回度数为 $n - 1$ 的节点编号。

### 思路 1：代码

```python
class Solution:
    def findCenter(self, edges: List[List[int]]) -> int:
        n = len(edges) + 1
        degrees = collections.Counter()

        for u, v in edges:
            degrees[u] += 1
            degrees[v] += 1

        for i in range(1, n + 1):
            if degrees[i] == n - 1:
                return i
        return -1
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(n)$。
- **空间复杂度**：$O(n)$。
# [1822. 数组元素积的符号](https://leetcode.cn/problems/sign-of-the-product-of-an-array/)

- 标签：数组、数学
- 难度：简单

## 题目链接

- [1822. 数组元素积的符号 - 力扣](https://leetcode.cn/problems/sign-of-the-product-of-an-array/)

## 题目大意

**描述**：已知函数 `signFunc(x)` 会根据 `x` 的正负返回特定值：

- 如果 `x` 是正数，返回 `1`。
- 如果 `x` 是负数，返回 `-1`。
- 如果 `x` 等于 `0`，返回 `0`。

现在给定一个整数数组 `nums`。令 `product` 为数组 `nums` 中所有元素值的乘积。

**要求**：返回 `signFun(product)` 的值。

**说明**：

- $1 \le nums.length \le 1000$。
- $-100 \le nums[i] \le 100$。

**示例**：

- 示例 1：

```python
输入 nums = [-1,-2,-3,-4,3,2,1]
输出 1
解释 数组中所有值的乘积是 144，且 signFunc(144) = 1
```

## 解题思路

### 思路 1：

题目要求的是数组所有值乘积的正负性，但是我们没必要将所有数乘起来再判断正负性。只需要统计出数组中负数的个数，再加以判断即可。

- 使用变量 `minus_count` 记录数组中负数个数。
- 然后遍历数组 `nums`，对于当前元素 `num`：
  - 如果为 `0`，则最终乘积肯定为 `0`，直接返回 `0`。
  - 如果小于 `0`，负数个数加 `1`。
- 最终统计出数组中负数的个数为 `minus_count`。
- 如果 `minus_count` 是 `2` 的倍数，则说明最终乘积为正数，返回 `1`。
- 如果 `minus_count` 不是 `2` 的倍数，则说明最终乘积为负数，返回 `-1`。

## 代码

### 思路 1 代码：

```python
class Solution:
    def arraySign(self, nums: List[int]) -> int:
        minus_count = 0
        for num in nums:
            if num < 0:
                minus_count += 1
            elif num == 0:
                return 0

        if minus_count % 2 == 0:
            return 1
        else:
            return -1
```

# [1827. 最少操作使数组递增](https://leetcode.cn/problems/minimum-operations-to-make-the-array-increasing/)

- 标签：贪心、数组
- 难度：简单

## 题目链接

- [1827. 最少操作使数组递增 - 力扣](https://leetcode.cn/problems/minimum-operations-to-make-the-array-increasing/)

## 题目大意

**描述**：给定一个整数数组 $nums$（下标从 $0$ 开始）。每一次操作中，你可以选择数组中的一个元素，并将它增加 $1$。

- 比方说，如果 $nums = [1,2,3]$，你可以选择增加 $nums[1]$ 得到 $nums = [1,3,3]$。

**要求**：请你返回使 $nums$ 严格递增的最少操作次数。

**说明**：

- 我们称数组 $nums$ 是严格递增的，当它满足对于所有的 $0 \le i < nums.length - 1$ 都有 $nums[i] < nums[i + 1]$。一个长度为 $1$ 的数组是严格递增的一种特殊情况。
- $1 \le nums.length \le 5000$。
- $1 \le nums[i] \le 10^4$。

**示例**：

- 示例 1：

```python
输入：nums = [1,1,1]
输出：3
解释：你可以进行如下操作：
1) 增加 nums[2] ，数组变为 [1,1,2]。
2) 增加 nums[1] ，数组变为 [1,2,2]。
3) 增加 nums[2] ，数组变为 [1,2,3]。
```

- 示例 2：

```python
输入：nums = [1,5,2,4,1]
输出：14
```

## 解题思路

### 思路 1：贪心算法

题目要求使 $nums$ 严格递增的最少操作次数。当遇到 $nums[i - 1] \ge nums[i]$ 时，我们应该在满足要求的同时，尽可能使得操作次数最少，则 $nums[i]$ 应增加到 $nums[i - 1] + 1$ 时，此时操作次数最少，并且满足 $nums[i - 1] < nums[i]$。

具体操作步骤如下：

1. 从左到右依次遍历数组元素。
2. 如果遇到 $nums[i - 1] \ge nums[i]$ 时：
   1. 本次增加的最少操作次数为 $nums[i - 1] + 1 - nums[i]$，将其计入答案中。
   2. 将 $nums[i]$ 变为 $nums[i - 1] + 1$。
3. 遍历完返回答案 $ans$。

### 思路 1：代码

```Python
class Solution:
    def minOperations(self, nums: List[int]) -> int:
        ans = 0
        for i in range(1, len(nums)):
            if nums[i - 1] >= nums[i]:
                ans += nums[i - 1] + 1 - nums[i]
                nums[i] = nums[i - 1] + 1
        
        return ans
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(n)$，其中 $n$ 为数组 $nums$ 的长度。
- **空间复杂度**：$O(1)$。
# [1833. 雪糕的最大数量](https://leetcode.cn/problems/maximum-ice-cream-bars/)

- 标签：贪心、数组、排序
- 难度：中等

## 题目链接

- [1833. 雪糕的最大数量 - 力扣](https://leetcode.cn/problems/maximum-ice-cream-bars/)

## 题目大意

**描述**：给定一个数组 $costs$ 表示不同雪糕的定价，其中 $costs[i]$ 表示第 $i$ 支雪糕的定价。再给定一个整数 $coins$ 表示 Tony 一共有的现金数量。

**要求**：计算并返回 Tony 用 $coins$ 现金能够买到的雪糕的最大数量。

**说明**：

- $costs.length == n$。
- $1 \le n \le 10^5$。
- $1 \le costs[i] \le 10^5$。
- $1 \le coins \le 10^8$。

**示例**：

- 示例 1：

```python
输入：costs = [1,3,2,4,1], coins = 7
输出：4
解释：Tony 可以买下标为 0、1、2、4 的雪糕，总价为 1 + 3 + 2 + 1 = 7
```

- 示例 2：

```python
输入：costs = [10,6,8,7,7,8], coins = 5
输出：0
解释：Tony 没有足够的钱买任何一支雪糕。
```

## 解题思路

### 思路 1：排序 + 贪心

贪心思路，如果想尽可能买到多的雪糕，就应该优先选择价格便宜的雪糕。具体步骤如下：

1. 对数组 $costs$ 进行排序。
2. 按照雪糕价格从低到高开始买雪糕，并记录下购买雪糕的数量，知道现有钱买不起雪糕为止。
3. 输出购买雪糕的数量作为答案。

### 思路 1：代码

```python
class Solution:
    def maxIceCream(self, costs: List[int], coins: int) -> int:
        costs.sort()
        ans = 0
        for cost in costs:
            if coins >= cost:
                ans += 1
                coins -= cost
        return ans
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(n \times \log_2n)$。
- **空间复杂度**：$O(1)$。


# [1844. 将所有数字用字符替换](https://leetcode.cn/problems/replace-all-digits-with-characters/)

- 标签：字符串
- 难度：简单

## 题目链接

- [1844. 将所有数字用字符替换 - 力扣](https://leetcode.cn/problems/replace-all-digits-with-characters/)

## 题目大意

**描述**：给定一个下标从 $0$ 开始的字符串 $s$。字符串 $s$ 的偶数下标处为小写英文字母，奇数下标处为数字。

定义一个函数 `shift(c, x)`，其中 $c$ 是一个字符且 $x$ 是一个数字，函数返回字母表中 $c$ 后边第 $x$ 个字符。

- 比如，`shift('a', 5) = 'f'`，`shift('x', 0) = 'x'`。

对于每个奇数下标 $i$，我们需要将数字 $s[i]$ 用 `shift(s[i - 1], s[i])` 替换。

**要求**：替换字符串 $s$ 中所有数字以后，将字符串 $s$ 返回。

**说明**：

- 题目保证 `shift(s[i - 1], s[i])` 不会超过 `'z'`。
- $1 \le s.length \le 100$。
- $s$ 只包含小写英文字母和数字。
- 对所有奇数下标处的 $i$，满足 `shift(s[i - 1], s[i]) <= 'z'` 。

**示例**：

- 示例 1：

```python
输入：s = "a1c1e1"
输出："abcdef"
解释：数字被替换结果如下：
- s[1] -> shift('a',1) = 'b'
- s[3] -> shift('c',1) = 'd'
- s[5] -> shift('e',1) = 'f'
```

- 示例 2：

```python
输入：s = "a1b2c3d4e"
输出："abbdcfdhe"
解释：数字被替换结果如下：
- s[1] -> shift('a',1) = 'b'
- s[3] -> shift('b',2) = 'd'
- s[5] -> shift('c',3) = 'f'
- s[7] -> shift('d',4) = 'h'
```

## 解题思路

### 思路 1：模拟

1. 先定义一个 `shift(ch, x)` 用于替换 `s[i]`。
2. 将字符串转为字符串列表，定义为 $res$。
3. 以两个字符为一组遍历字符串，对 $res[i]$ 进行修改。
4. 将字符串列表连接起来，作为答案返回。

### 思路 1：代码

```python
class Solution:
    def replaceDigits(self, s: str) -> str:
        def shift(ch, x):
            return chr(ord(ch) + x) 
        
        res = list(s)
        for i in range(1, len(s), 2):
            res[i] = shift(res[i - 1], int(res[i]))
        
        return "".join(res)
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(n)$。
- **空间复杂度**：$O(n)$。

# [1858. 包含所有前缀的最长单词](https://leetcode.cn/problems/longest-word-with-all-prefixes/)

- 标签：深度优先搜索、字典树
- 难度：中等

## 题目链接

- [1858. 包含所有前缀的最长单词 - 力扣](https://leetcode.cn/problems/longest-word-with-all-prefixes/)

## 题目大意

给定一个字符串数组 `words`。

要求：找出 `words` 中所有前缀从都在 `words` 中的最长字符串。如果存在多个符合条件相同长度的字符串，则输出字典序中最小的字符串。如果不存在这样的字符串，返回 `' '`。

- 例如：令 `words = ["a", "app", "ap"]`。字符串 `"app"` 含前缀 `"ap"` 和 `"a"` ，都在 `words` 中。

## 解题思路

使用字典树存储所有单词，再将字典中单词按照长度从大到小、字典序从小到大排序。

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
            if not cur.isEnd:
                return False
        return True


class Solution:
    def longestWord(self, words: List[str]) -> str:
        tire_tree = Trie()
        for word in words:
            tire_tree.insert(word)
        words.sort(key=lambda x:(-len(x), x))
        for word in words:
            if tire_tree.search(word):
                return word
        return ''
```

# [1859. 将句子排序](https://leetcode.cn/problems/sorting-the-sentence/)

- 标签：字符串、排序
- 难度：简单

## 题目链接

- [1859. 将句子排序 - 力扣](https://leetcode.cn/problems/sorting-the-sentence/)

## 题目大意

**描述**：给定一个句子 $s$，句子中包含的单词不超过 $9$ 个。并且句子 $s$ 中每个单词末尾添加了「从 $1$ 开始的单词位置索引」，并且将句子中所有单词打乱顺序。

举个例子，句子 `"This is a sentence"` 可以被打乱顺序得到 `"sentence4 a3 is2 This1"` 或者 `"is2 sentence4 This1 a3"` 。

**要求**：重新构造并得到原本顺序的句子。

**说明**：

- **一个句子**：指的是一个序列的单词用单个空格连接起来，且开头和结尾没有任何空格。每个单词都只包含小写或大写英文字母。
- $2 \le s.length \le 200$。
- $s$ 只包含小写和大写英文字母、空格以及从 $1$ 到 $9$ 的数字。
- $s$ 中单词数目为 $1$ 到 $9$ 个。
- $s$ 中的单词由单个空格分隔。
- $s$ 不包含任何前导或者后缀空格。

**示例**：

- 示例 1：

```python
输入：s = "is2 sentence4 This1 a3"
输出："This is a sentence"
解释：将 s 中的单词按照初始位置排序，得到 "This1 is2 a3 sentence4" ，然后删除数字。
```

- 示例 2：

```python
输入：s = "Myself2 Me1 I4 and3"
输出："Me Myself and I"
解释：将 s 中的单词按照初始位置排序，得到 "Me1 Myself2 and3 I4" ，然后删除数字。
```

## 解题思路

### 思路 1：模拟

1. 将句子 $s$ 按照空格分隔成数组 $s\underline{\hspace{0.5em}}list$。
2. 遍历数组 $s\underline{\hspace{0.5em}}list$ 中的单词：
   1. 从单词中分割出对应单词索引 $idx$ 和对应单词 $word$。
   2. 将单词 $word$ 存入答案数组 $res$ 对应位置 $idx - 1$ 上，即：$res[int(idx) - 1] = word$。
3. 将答案数组用空格拼接成句子字符串，并返回。

### 思路 1：代码

```python
class Solution:
    def sortSentence(self, s: str) -> str:
        s_list = s.split()
        size = len(s_list)
        res = ["" for _ in range(size)]
        for sub in s_list:
            idx = ""
            word = ""
            for ch in sub:
                if '1' <= ch <= '9':
                    idx += ch
                else:
                    word += ch
            res[int(idx) - 1] = word

        return " ".join(res)
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(m)$，其中 $m$ 为给定句子 $s$ 的长度。
- **空间复杂度**：$O(m)$。

# [1876. 长度为三且各字符不同的子字符串](https://leetcode.cn/problems/substrings-of-size-three-with-distinct-characters/)

- 标签：哈希表、字符串、计数、滑动窗口
- 难度：简单

## 题目链接

- [1876. 长度为三且各字符不同的子字符串 - 力扣](https://leetcode.cn/problems/substrings-of-size-three-with-distinct-characters/)

## 题目大意

**描述**：给定搞一个字符串 $s$。

**要求**：返回 $s$ 中长度为 $3$ 的好子字符串的数量。如果相同的好子字符串出现多次，则每一次都应该被记入答案之中。

**说明**：

- **子字符串**：指的是一个字符串中连续的字符序列。
- **好子字符串**：如果一个字符串中不含有任何重复字符，则称这个字符串为好子字符串。
- $1 \le s.length \le 100$。
- $s$ 只包含小写英文字母。

**示例**：

- 示例 1：

```python
输入：s = "xyzzaz"
输出：1
解释：总共有 4 个长度为 3 的子字符串："xyz"，"yzz"，"zza" 和 "zaz" 。
唯一的长度为 3 的好子字符串是 "xyz" 。
```

- 示例 2：

```python
输入：s = "aababcabc"
输出：4
解释：总共有 7 个长度为 3 的子字符串："aab"，"aba"，"bab"，"abc"，"bca"，"cab" 和 "abc" 。
好子字符串包括 "abc"，"bca"，"cab" 和 "abc" 。
```

## 解题思路

### 思路 1：模拟

1. 遍历字符串 $s$ 中长度为 3 的子字符串。
2. 判断子字符串中的字符是否有重复。如果没有重复，则答案进行计数。
3. 遍历完输出答案。

### 思路 1：代码

```python
class Solution:
    def countGoodSubstrings(self, s: str) -> int:
        ans = 0
        for i in range(2, len(s)):
            if s[i - 2] != s[i - 1] and s[i - 1] != s[i] and s[i - 2] != s[i]:
                ans += 1
        return ans
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(n)$。
- **空间复杂度**：$O(1)$。# [1877. 数组中最大数对和的最小值](https://leetcode.cn/problems/minimize-maximum-pair-sum-in-array/)

- 标签：贪心、数组、双指针、排序
- 难度：中等

## 题目链接

- [1877. 数组中最大数对和的最小值 - 力扣](https://leetcode.cn/problems/minimize-maximum-pair-sum-in-array/)

## 题目大意

**描述**：一个数对 $(a, b)$ 的数对和等于 $a + b$。最大数对和是一个数对数组中最大的数对和。

- 比如，如果我们有数对 $(1, 5)$，$(2, 3)$ 和 $(4, 4)$，最大数对和为 $max(1 + 5, 2 + 3, 4 + 4) = max(6, 5, 8) = 8$。

给定一个长度为偶数 $n$ 的数组 $nums$，现在将 $nums$ 中的元素分为 $n / 2$ 个数对，使得：

- $nums$ 中每个元素恰好在一个数对中。
- 最大数对和的值最小。

**要求**：在最优数对划分的方案下，返回最小的最大数对和。

**说明**：

- $n == nums.length$。
- $2 \le n \le 10^5$。
- $n$ 是偶数。
- $1 \le nums[i] \le 10^5$。

**示例**：

- 示例 1：

```python
输入：nums = [3,5,2,3]
输出：7
解释：数组中的元素可以分为数对 (3,3) 和 (5,2)。
最大数对和为 max(3+3, 5+2) = max(6, 7) = 7。
```

- 示例 2：

```python
输入：nums = [3,5,4,2,4,6]
输出：8
解释：数组中的元素可以分为数对 (3,5)，(4,4) 和 (6,2)。
最大数对和为 max(3+5, 4+4, 6+2) = max(8, 8, 8) = 8。
```

## 解题思路

### 思路 1：排序 + 贪心

为了使最大数对和的值尽可能的小，我们应该尽可能的让数组中最大值与最小值组成一对，次大值与次小值组成一对。而其他任何方案都会使得最大数对和的值更大。

那么，我们可以先将数组进行排序，然后首尾依次进行组对，并计算这种方案下的最大数对和即为答案。

### 思路 1：代码

```python
class Solution:
    def minPairSum(self, nums: List[int]) -> int:
        nums.sort()
        ans, size = 0, len(nums)
        for i in range(len(nums) // 2):
            ans = max(ans, nums[i] + nums[size - 1 - i])
        
        return ans
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(n \times \log n)$。
- **空间复杂度**：$O(\log n)$。

# [1879. 两个数组最小的异或值之和](https://leetcode.cn/problems/minimum-xor-sum-of-two-arrays/)

- 标签：位运算、数组、动态规划、状态压缩
- 难度：困难

## 题目链接

- [1879. 两个数组最小的异或值之和 - 力扣](https://leetcode.cn/problems/minimum-xor-sum-of-two-arrays/)

## 题目大意

**描述**：给定两个整数数组 $nums1$ 和 $nums2$，两个数组长度都为 $n$。

**要求**：将 $nums2$ 中的元素重新排列，使得两个数组的异或值之和最小。并返回重新排列之后的异或值之和。

**说明**：

- **两个数组的异或值之和**：$(nums1[0] \oplus nums2[0]) + (nums1[1] \oplus nums2[1]) + ... + (nums1[n - 1] \oplus nums2[n - 1])$（下标从 $0$ 开始）。
- 举个例子，$[1, 2, 3]$ 和 $[3,2,1]$ 的异或值之和 等于 $(1 \oplus 3) + (2 \oplus 2) + (3 \oplus 1) + (3 \oplus 1) = 2 + 0 + 2 = 4$。
- $n == nums1.length$。
- $n == nums2.length$。
- $1 \le n \le 14$。
- $0 \le nums1[i], nums2[i] \le 10^7$。

**示例**：

- 示例 1：

```python
输入：nums1 = [1,2], nums2 = [2,3]
输出：2
解释：将 nums2 重新排列得到 [3,2] 。
异或值之和为 (1 XOR 3) + (2 XOR 2) = 2 + 0 = 2。
```

- 示例 2：

```python
输入：nums1 = [1,0,3], nums2 = [5,3,4]
输出：8
解释：将 nums2 重新排列得到 [5,4,3] 。
异或值之和为 (1 XOR 5) + (0 XOR 4) + (3 XOR 3) = 4 + 4 + 0 = 8。
```

## 解题思路

### 思路 1：状态压缩 DP

由于数组 $nums2$ 可以重新排列，所以我们可以将数组 $nums1$ 中的元素顺序固定，然后将数组 $nums1$ 中第 $i$ 个元素与数组 $nums2$ 中所有还没被选择的元素进行组合，找到异或值之和最小的组合。

同时因为两个数组长度 $n$ 的大小范围只有 $[1, 14]$，所以我们可以采用「状态压缩」的方式来表示 $nums2$ 中当前元素的选择情况。

「状态压缩」指的是使用一个 $n$ 位的二进制数 $state$ 来表示排列中数的选取情况。

如果二进制数 $state$ 的第 $i$ 位为 $1$，说明数组 $nums2$ 第 $i$ 个元素在该状态中被选取。反之，如果该二进制的第 $i$ 位为 $0$，说明数组 $nums2$ 中第 $i$ 个元素在该状态中没有被选取。

举个例子：

1. $nums2 = \lbrace 1, 2, 3, 4 \rbrace$，$state = (1001)_2$，表示选择了第 $1$ 个元素和第 $4$ 个元素，也就是 $1$、$4$。
2. $nums2 = \lbrace 1, 2, 3, 4, 5, 6 \rbrace$，$state = (011010)_2$，表示选择了第 $2$ 个元素、第 $4$ 个元素、第 $5$ 个元素，也就是  $2$、$4$、$5$。

这样，我们就可以通过动态规划的方式来解决这道题。

###### 1. 划分阶段

按照数组 $nums$ 中元素选择情况进行阶段划分。

###### 2. 定义状态

定义当前数组 $nums2$ 中元素选择状态为 $state$，$state$ 对应选择的元素个数为 $count(state)$。

则可以定义状态 $dp[state]$ 表示为：当前数组 $nums2$ 中元素选择状态为 $state$，并且选择了 $nums1$ 中前 $count(state)$ 个元素的情况下，可以组成的最小异或值之和。

###### 3. 状态转移方程

对于当前状态 $dp[state]$，肯定是从比 $state$ 少选一个元素的状态中递推而来。我们可以枚举少选一个元素的状态，找到可以组成的异或值之和最小值，赋值给 $dp[state]$。

举个例子 $nums2 = \lbrace 1, 2, 3, 4 \rbrace$，$state = (1001)_2$，表示选择了第 $1$ 个元素和第 $4$ 个元素，也就是 $1$、$4$。那么 $state$ 只能从 $(1000)_2$ 和 $(0001)_2$ 这两个状态转移而来，我们只需要枚举这两种状态，并求出转移过来的异或值之和最小值。

即状态转移方程为：$dp[state] = min(dp[state], \quad dp[state \oplus (1 \text{ <}\text{< } i)] + (nums1[i] \oplus nums2[one\underline{\hspace{0.5em}}cnt - 1]))$，其中 $state$ 第 $i$ 位一定为 $1$，$one\underline{\hspace{0.5em}}cnt$ 为 $state$ 中 $1$ 的个数。

###### 4. 初始条件

- 既然是求最小值，不妨将所有状态初始为最大值。
- 未选择任何数时，异或值之和为 $0$，所以初始化 $dp[0] = 0$。

###### 5. 最终结果

根据我们之前定义的状态，$dp[state]$ 表示为：当前数组 $nums2$ 中元素选择状态为 $state$，并且选择了 $nums1$ 中前 $count(state)$ 个元素的情况下，可以组成的最小异或值之和。 所以最终结果为 $dp[states - 1]$，其中 $states = 1 \text{ <}\text{< } n$。

### 思路 1：代码

```python
class Solution:
    def minimumXORSum(self, nums1: List[int], nums2: List[int]) -> int:
        ans = float('inf')
        size = len(nums1)
        states = 1 << size

        dp = [float('inf') for _ in range(states)]
        dp[0] = 0
        for state in range(states):
            one_cnt = bin(state).count('1')
            for i in range(size):
                if (state >> i) & 1:
                    dp[state] = min(dp[state], dp[state ^ (1 << i)] + (nums1[i] ^ nums2[one_cnt - 1]))
        
        return dp[states - 1]
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(2^n \times n)$，其中 $n$ 是数组 $nums1$、$nums2$ 的长度。
- **空间复杂度**：$O(2^n)$。

# [1893. 检查是否区域内所有整数都被覆盖](https://leetcode.cn/problems/check-if-all-the-integers-in-a-range-are-covered/)

- 标签：数组、哈希表、前缀和
- 难度：简单

## 题目链接

- [1893. 检查是否区域内所有整数都被覆盖 - 力扣](https://leetcode.cn/problems/check-if-all-the-integers-in-a-range-are-covered/)

## 题目大意

**描述**：给定一个二维整数数组 $ranges$ 和两个整数 $left$ 和 $right$。每个 $ranges[i] = [start_i, end_i]$ 表示一个从 $start_i$ 到 $end_i$ 的 闭区间 。

**要求**：如果闭区间 $[left, right]$ 内每个整数都被 $ranges$ 中至少一个区间覆盖，那么请你返回 $True$ ，否则返回 $False$。

**说明**：

- $1 \le ranges.length \le 50$。
- $1 \le start_i \le end_i \le 50$。
- $1 \le left \le right \le 50$。

**示例**：

- 示例 1：

```python
输入：ranges = [[1,2],[3,4],[5,6]], left = 2, right = 5
输出：True
解释：2 到 5 的每个整数都被覆盖了：
- 2 被第一个区间覆盖。
- 3 和 4 被第二个区间覆盖。
- 5 被第三个区间覆盖。
```

- 示例 2：

```python
输入：ranges = [[1,10],[10,20]], left = 21, right = 21
输出：False
解释：21 没有被任何一个区间覆盖。
```

## 解题思路

### 思路 1：暴力

区间的范围为 $[1, 50]$，所以我们可以使用一个长度为 $51$ 的标志数组 $flags$ 用于标记区间内的所有整数。

1. 遍历数组 $ranges$ 中的所有区间 $[l, r]$。
2. 对于区间 $[l, r]$ 和区间 $[left, right]$，将两区间相交部分标记为 $True$。
3. 遍历区间 $[left, right]$ 上的所有整数，判断对应标志位是否为 $False$。
4. 如果对应标志位出现 $False$，则返回 $False$。
5. 如果遍历完所有标志位都为 $True$，则返回 $True$。

### 思路 1：代码

```Python
class Solution:
    def isCovered(self, ranges: List[List[int]], left: int, right: int) -> bool:
        flags = [False for _ in range(51)]
        for l, r in ranges:
            for i in range(max(l, left), min(r, right) + 1):
                flags[i] = True
            
        for i in range(left, right + 1):
            if not flags[i]:
                return False
        
        return True
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(50 \times n)$。
- **空间复杂度**：$O(50)$。
# [1897. 重新分配字符使所有字符串都相等](https://leetcode.cn/problems/redistribute-characters-to-make-all-strings-equal/)

- 标签：哈希表、字符串、计数
- 难度：简单

## 题目链接

- [1897. 重新分配字符使所有字符串都相等 - 力扣](https://leetcode.cn/problems/redistribute-characters-to-make-all-strings-equal/)

## 题目大意

**描述**：给定一个字符串数组 $words$（下标从 $0$ 开始计数）。

在一步操作中，需先选出两个 不同 下标 $i$ 和 $j$，其中 $words[i]$ 是一个非空字符串，接着将 $words[i]$ 中的任一字符移动到 $words[j]$ 中的 任一 位置上。

**要求**：如果执行任意步操作可以使 $words$ 中的每个字符串都相等，返回 $True$；否则，返回 $False$。

**说明**：

- $1 <= words.length <= 100$。
- $1 <= words[i].length <= 100$
- $words[i]$ 由小写英文字母组成。

**示例**：

- 示例 1：

```python
输入：words = ["abc","aabc","bc"]
输出：true
解释：将 words[1] 中的第一个 'a' 移动到 words[2] 的最前面。
使 words[1] = "abc" 且 words[2] = "abc"。
所有字符串都等于 "abc" ，所以返回 True。
```

- 示例 2：

```python
输入：words = ["ab","a"]
输出：False
解释：执行操作无法使所有字符串都相等。
```

## 解题思路

### 思路 1：哈希表

如果通过重新分配字符能够使所有字符串都相等，则所有字符串的字符需要满足：

1. 每个字符串中字符种类相同，
2. 每个字符串中各种字符的个数相同。

则我们可以使用哈希表来统计字符串中字符种类及个数。具体步骤如下：

1. 遍历单词数组 $words$ 中的所有单词 $word$。
2. 遍历所有单词 $word$ 中的所有字符 $ch$。
3. 使用哈希表 $cnts$ 统计字符种类及个数。
4. 如果所有字符个数都是单词个数的倍数，则说明通过重新分配字符能够使所有字符串都相等，则返回 $True$。
5. 否则返回 $False$。

### 思路 1：代码

```Python
class Solution:
    def makeEqual(self, words: List[str]) -> bool:
        size = len(words)
        cnts = Counter()
        for word in words:
            for ch in word:
                cnts[ch] += 1

        return all(value % size == 0 for key, value in cnts.items())
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(s + |\sum|)$，其中 $s$ 为数组 $words$ 中所有单词的长度之和，$\sum$ 是字符集，本题中 $|\sum| = 26$。
- **空间复杂度**：$O(|\sum|)$。
# [1903. 字符串中的最大奇数](https://leetcode.cn/problems/largest-odd-number-in-string/)

- 标签：贪心、数学、字符串
- 难度：简单

## 题目链接

- [1903. 字符串中的最大奇数 - 力扣](https://leetcode.cn/problems/largest-odd-number-in-string/)

## 题目大意

**描述**：给定一个字符串 $num$，表示一个大整数。

**要求**：在字符串 $num$ 的所有非空子字符串中找出值最大的奇数，并以字符串形式返回。如果不存在奇数，则返回一个空字符串 `""`。

**说明**：

- **子字符串**：指的是字符串中一个连续的字符序列。
- $1 \le num.length \le 10^5$
- $num$ 仅由数字组成且不含前导零。

**示例**：

- 示例 1：

```python
输入：num = "52"
输出："5"
解释：非空子字符串仅有 "5"、"2" 和 "52" 。"5" 是其中唯一的奇数。
```

- 示例 2：

```python
输入：num = "4206"
输出：""
解释：在 "4206" 中不存在奇数。
```

## 解题思路

### 思路 1：贪心算法

如果某个数 $x$ 为奇数，则 $x$ 末尾位上的数字一定为奇数。那么我们只需要在末尾为奇数的字符串中考虑最大的奇数即可。显而易见的是，最大的奇数一定是长度最长的那个。所以我们只需要逆序遍历字符串，找到第一个奇数，从整个字符串开始位置到该奇数位置所代表的整数，就是最大的奇数。具体步骤如下：

1. 逆序遍历字符串 $s$。
2. 找到第一个奇数位置 $i$，则 $num[0: i + 1]$ 为最大的奇数，将其作为答案返回。

### 思路 1：代码

```python
class Solution:
    def largestOddNumber(self, num: str) -> str:
        for i in range(len(num) - 1, -1, -1):
            if int(num[i]) % 2 == 1:
                return num[0: i + 1]
        return ""
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(n)$。
- **空间复杂度**：$O(1)$。
# [1921. 消灭怪物的最大数量](https://leetcode.cn/problems/eliminate-maximum-number-of-monsters/)

- 标签：贪心、数组、排序
- 难度：中等

## 题目链接

- [1921. 消灭怪物的最大数量 - 力扣](https://leetcode.cn/problems/eliminate-maximum-number-of-monsters/)

## 题目大意

**描述**：你正在玩一款电子游戏，在游戏中你需要保护城市免受怪物侵袭。给定一个下标从 $0$ 开始且大小为 $n$ 的整数数组 $dist$，其中 $dist[i]$ 是第 $i$ 个怪物与城市的初始距离（单位：米）。

怪物以恒定的速度走向城市。每个怪物的速度都以一个长度为 $n$ 的整数数组 $speed$ 表示，其中 $speed[i]$ 是第 $i$ 个怪物的速度（单位：千米/分）。

你有一种武器，一旦充满电，就可以消灭 一个 怪物。但是，武器需要 一分钟 才能充电。武器在游戏开始时是充满电的状态，怪物从 第 $0$ 分钟时开始移动。

一旦任一怪物到达城市，你就输掉了这场游戏。如果某个怪物 恰好 在某一分钟开始时到达城市（距离表示为 $0$），这也会被视为输掉 游戏，在你可以使用武器之前，游戏就会结束。

**要求**：返回在你输掉游戏前可以消灭的怪物的最大数量。如果你可以在所有怪物到达城市前将它们全部消灭，返回  $n$。

**说明**：

- 

**示例**：

- 示例 1：

```python
输入：dist = [1,3,4], speed = [1,1,1]
输出：3
解释：
第 0 分钟开始时，怪物的距离是 [1,3,4]，你消灭了第一个怪物。
第 1 分钟开始时，怪物的距离是 [X,2,3]，你消灭了第二个怪物。
第 3 分钟开始时，怪物的距离是 [X,X,2]，你消灭了第三个怪物。
所有 3 个怪物都可以被消灭。
```

- 示例 2：

```python
输入：dist = [1,1,2,3], speed = [1,1,1,1]
输出：1
解释：
第 0 分钟开始时，怪物的距离是 [1,1,2,3]，你消灭了第一个怪物。
第 1 分钟开始时，怪物的距离是 [X,0,1,2]，所以你输掉了游戏。
你只能消灭 1 个怪物。
```

## 解题思路

### 思路 1：排序 + 贪心算法

对于第 $i$ 个怪物，最晚可被消灭的时间为 $times[i] = \lfloor \frac{dist[i] - 1}{speed[i]} \rfloor$。我们可以根据以上公式，将所有怪物最晚可被消灭时间存入数组 $times$ 中，然后对 $times$ 进行升序排序。

然后遍历数组 $times$，对于第 $i$ 个怪物：

1. 如果 $times[i] < i$，则说明第 $i$ 个怪物无法被消灭，直接返回 $i$ 即可。
2. 如果 $times[i] \ge i$，则说明第 $i$ 个怪物可以被消灭，继续向下遍历。

如果遍历完数组 $times$，则说明所有怪物都可以被消灭，则返回 $n$。

### 思路 1：代码

```Python
class Solution:
    def eliminateMaximum(self, dist: List[int], speed: List[int]) -> int:
        times = []
        for d, s in zip(dist, speed):
            time = (d - 1) // s
            times.append(time)
        times.sort()

        size = len(times)
        for i in range(size):
            if times[i] < i:
                return i

        return size
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(n \times \log n)$，其中 $n$ 为数组 $dist$ 的长度。
- **空间复杂度**：$O(n)$。

# [1925. 统计平方和三元组的数目](https://leetcode.cn/problems/count-square-sum-triples/)

- 标签：数学、枚举
- 难度：简单

## 题目链接

- [1925. 统计平方和三元组的数目 - 力扣](https://leetcode.cn/problems/count-square-sum-triples/)

## 题目大意

**描述**：给你一个整数 $n$。

**要求**：请你返回满足 $1 \le a, b, c \le n$ 的平方和三元组的数目。

**说明**：

- **平方和三元组**：指的是满足 $a^2 + b^2 = c^2$ 的整数三元组 $(a, b, c)$。
- $1 \le n \le 250$。

**示例**：

- 示例 1：

```python
输入 n = 5
输出 2
解释 平方和三元组为 (3,4,5) 和 (4,3,5)。
```

- 示例 2：

```python
输入：n = 10
输出：4
解释：平方和三元组为 (3,4,5)，(4,3,5)，(6,8,10) 和 (8,6,10)。
```

## 解题思路

### 思路 1：枚举算法

我们可以在 $[1, n]$ 区间中枚举整数三元组 $(a, b, c)$ 中的 $a$ 和 $b$。然后判断 $a^2 + b^2$ 是否小于等于 $n$，并且是完全平方数。

在遍历枚举的同时，我们维护一个用于统计平方和三元组数目的变量 `cnt`。如果符合要求，则将计数 `cnt` 加 $1$。最终，我们返回该数目作为答案。

利用枚举算法统计平方和三元组数目的时间复杂度为 $O(n^2)$。

- 注意：在计算中，为了防止浮点数造成的误差，并且两个相邻的完全平方正数之间的距离一定大于 $1$，所以我们可以用 $\sqrt{a^2 + b^2 + 1}$ 来代替 $\sqrt{a^2 + b^2}$。

### 思路 1：代码

```python
class Solution:
    def countTriples(self, n: int) -> int:
        cnt = 0
        for a in range(1, n + 1):
            for b in range(1, n + 1):
                c = int(sqrt(a * a + b * b + 1))
                if c <= n and a * a + b * b == c * c:
                    cnt += 1
        return cnt
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(n^2)$。
- **空间复杂度**：$O(1)$。
# [1929. 数组串联](https://leetcode.cn/problems/concatenation-of-array/)

- 标签：数组
- 难度：简单

## 题目链接

- [1929. 数组串联 - 力扣](https://leetcode.cn/problems/concatenation-of-array/)

## 题目大意

**描述**：给定一个长度为 $n$ 的整数数组 $nums$。

**要求**：构建一个长度为 $2 \times n$ 的答案数组 $ans$，答案数组下标从 $0$ 开始计数 ，对于所有 $0 \le i < n$ 的 $i$ ，满足下述所有要求：

- $ans[i] == nums[i]$。
- $ans[i + n] == nums[i]$。

具体而言，$ans$ 由两个 $nums$ 数组「串联」形成。

**说明**：

- $n == nums.length$。
- $1 \le n \le 1000$。
- $1 \le nums[i] \le 1000$。

**示例**：

- 示例 1：

```python
输入：nums = [1,2,1]
输出：[1,2,1,1,2,1]
解释：数组 ans 按下述方式形成：
- ans = [nums[0],nums[1],nums[2],nums[0],nums[1],nums[2]]
- ans = [1,2,1,1,2,1]
```

- 示例 2：

```python
输入：nums = [1,3,2,1]
输出：[1,3,2,1,1,3,2,1]
解释：数组 ans 按下述方式形成：
- ans = [nums[0],nums[1],nums[2],nums[3],nums[0],nums[1],nums[2],nums[3]]
- ans = [1,3,2,1,1,3,2,1]
```

## 解题思路

### 思路 1：按要求模拟

1. 定义一个数组变量（列表）$ans$ 作为答案数组。
2. 然后按顺序遍历两次数组 $nums$ 中的元素，并依次添加到 $ans$ 的尾部。最后返回 $ans$。

### 思路 1：代码

```python
class Solution:
    def getConcatenation(self, nums: List[int]) -> List[int]:
        ans = []
        for num in nums:
            ans.append(num)
        for num in nums:
            ans.append(num)
        return ans
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(n)$，其中 $n$ 为数组 $nums$ 的长度。
- **空间复杂度**：$O(n)$。如果算上答案数组的空间占用，则空间复杂度为 $O(n)$。不算上则空间复杂度为 $O(1)$。

### 思路 2：利用运算符

Python 中可以直接利用 `+` 号运算符将两个列表快速进行串联。即 `return nums + nums`。

### 思路 2：代码

```python
class Solution:
    def getConcatenation(self, nums: List[int]) -> List[int]:
        return nums + nums
```

### 思路 2：复杂度分析

- **时间复杂度**：$O(n)$，其中 $n$ 为数组 $nums$ 的长度。
- **空间复杂度**：$O(n)$。如果算上答案数组的空间占用，则空间复杂度为 $O(n)$。不算上则空间复杂度为 $O(1)$。
# [1930. 长度为 3 的不同回文子序列](https://leetcode.cn/problems/unique-length-3-palindromic-subsequences/)

- 标签：哈希表、字符串、前缀和
- 难度：中等

## 题目链接

- [1930. 长度为 3 的不同回文子序列 - 力扣](https://leetcode.cn/problems/unique-length-3-palindromic-subsequences/)

## 题目大意

**描述**：给定一个人字符串 $s$。

**要求**：返回 $s$ 中长度为 $s$ 的不同回文子序列的个数。即便存在多种方法来构建相同的子序列，但相同的子序列只计数一次。

**说明**：

- **回文**：指正着读和反着读一样的字符串。
- **子序列**：由原字符串删除其中部分字符（也可以不删除）且不改变剩余字符之间相对顺序形成的一个新字符串。
  - 例如，`"ace"` 是 `"abcde"` 的一个子序列。

- $3 \le s.length \le 10^5$。
- $s$ 仅由小写英文字母组成。

**示例**：

- 示例 1：

```python
输入：s = "aabca"
输出：3
解释：长度为 3 的 3 个回文子序列分别是：
- "aba" ("aabca" 的子序列)
- "aaa" ("aabca" 的子序列)
- "aca" ("aabca" 的子序列)
```

- 示例 2：

```python
输入：s = "bbcbaba"
输出：4
解释：长度为 3 的 4 个回文子序列分别是：
- "bbb" ("bbcbaba" 的子序列)
- "bcb" ("bbcbaba" 的子序列)
- "bab" ("bbcbaba" 的子序列)
- "aba" ("bbcbaba" 的子序列)
```

## 解题思路

### 思路 1：枚举 + 哈希表

字符集只包含 $26$ 个小写字母，所以我们可以枚举这 $26$ 个小写字母。

对于每个小写字母，使用对撞双指针，找到字符串 $s$ 首尾两侧与小写字母相同的最左位置和最右位置。

如果两个位置不同，则我们可以将两个位置中间不重复的字符当作是长度为 $3$ 的子序列最中间的那个字符。

则我们可以统计出两个位置中间不重复字符的个数，将其累加到答案中。

遍历完，返回答案。

### 思路 1：代码

```Python
class Solution:
    def countPalindromicSubsequence(self, s: str) -> int:
        size = len(s)
        ans = 0

        for i in range(26):
            left, right = 0, size - 1
            
            while left < size and ord(s[left]) - ord('a') != i:
                left += 1
            
            while right >= 0 and ord(s[right]) - ord('a') != i:
                right -= 1

            if right - left < 2:
                continue

            char_set = set()
            for j in range(left + 1, right):
                char_set.add(s[j])
            ans += len(char_set)
        
        return ans
```

### 思路 1：复杂度分析

- **时间复杂度**：$n \times | \sum | + | \sum |^2$，其中 $n$ 为字符串 $s$ 的长度，$\sum$ 为字符集，本题中 $| \sum | = 26$。
- **空间复杂度**：$O(| \sum |)$。
# [1936. 新增的最少台阶数](https://leetcode.cn/problems/add-minimum-number-of-rungs/)

- 标签：贪心、数组
- 难度：中等

## 题目链接

- [1936. 新增的最少台阶数 - 力扣](https://leetcode.cn/problems/add-minimum-number-of-rungs/)

## 题目大意

**描述**：给定一个严格递增的整数数组 $rungs$，用于表示梯子上每一台阶的高度。当前你正站在高度为 $0$ 的地板上，并打算爬到最后一个台阶。

另给定一个整数 $dist$。每次移动中，你可以到达下一个距离当前位置（地板或台阶）不超过 $dist$ 高度的台阶。当前，你也可以在任何正整数高度插入尚不存在的新台阶。

**要求**：返回爬到最后一阶时必须添加到梯子上的最少台阶数。

**说明**：

- 

**示例**：

- 示例 1：

```python
输入：rungs = [1,3,5,10], dist = 2
输出：2
解释：
现在无法到达最后一阶。
在高度为 7 和 8 的位置增设新的台阶，以爬上梯子。 
梯子在高度为 [1,3,5,7,8,10] 的位置上有台阶。
```

- 示例 2：

```python
输入：rungs = [3,4,6,7], dist = 2
输出：1
解释：
现在无法从地板到达梯子的第一阶。 
在高度为 1 的位置增设新的台阶，以爬上梯子。 
梯子在高度为 [1,3,4,6,7] 的位置上有台阶。
```

## 解题思路

### 思路 1：贪心算法 + 模拟

1. 遍历梯子的每一层台阶。
2. 计算每一层台阶与上一层台阶之间的差值 $diff$。
3. 每层最少需要新增的台阶数为 $\lfloor \frac{diff - 1}{dist} \rfloor$，将其计入答案 $ans$ 中。
4. 遍历完返回答案。

### 思路 1：代码

```Python
class Solution:
    def addRungs(self, rungs: List[int], dist: int) -> int:
        ans, cur = 0, 0
        for h in rungs:
            diff = h - cur
            ans += (diff - 1) // dist
            cur = h
        
        return ans
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(n)$，其中 $n$ 为数组 $rungs$ 的长度。
- **空间复杂度**：$O(1)$。

# [1941. 检查是否所有字符出现次数相同](https://leetcode.cn/problems/check-if-all-characters-have-equal-number-of-occurrences/)

- 标签：哈希表、字符串、计数
- 难度：简单

## 题目链接

- [1941. 检查是否所有字符出现次数相同 - 力扣](https://leetcode.cn/problems/check-if-all-characters-have-equal-number-of-occurrences/)

## 题目大意

**描述**：给定一个字符串 $s$。如果 $s$ 中出现过的所有字符的出现次数相同，那么我们称字符串 $s$ 是「好字符串」。

**要求**：如果 $s$ 是一个好字符串，则返回 `True`，否则返回 `False`。

**说明**：

- $1 \le s.length \le 1000$。
- $s$ 只包含小写英文字母。

**示例**：

- 示例 1：

```python
输入：s = "abacbc"
输出：true
解释：s 中出现过的字符为 'a'，'b' 和 'c' 。s 中所有字符均出现 2 次。
```

- 示例 2：

```python
输入：s = "aaabb"
输出：false
解释：s 中出现过的字符为 'a' 和 'b' 。
'a' 出现了 3 次，'b' 出现了 2 次，两者出现次数不同。
```

## 解题思路

### 思路 1：哈希表

1. 使用哈希表记录字符串 $s$ 中每个字符的频数。
2. 然后遍历哈希表中的键值对，检测每个字符的频数是否相等。
3. 如果发现频数不相等，则直接返回 `False`。
4. 如果检查完发现所有频数都相等，则返回 `True`。

### 思路 1：代码

```python
class Solution:
    def areOccurrencesEqual(self, s: str) -> bool:
        counter = Counter(s)
        flag = -1
        for key in counter:
            if flag == -1:
                flag = counter[key]
            else:
                if flag != counter[key]:
                    return False
        return True
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(n)$。
- **空间复杂度**：$O(n)$。
# [1947. 最大兼容性评分和](https://leetcode.cn/problems/maximum-compatibility-score-sum/)

- 标签：位运算、数组、动态规划、回溯、状态压缩
- 难度：中等

## 题目链接

- [1947. 最大兼容性评分和 - 力扣](https://leetcode.cn/problems/maximum-compatibility-score-sum/)

## 题目大意

**描述**：有一份由 $n$ 个问题组成的调查问卷，每个问题的答案只有 $0$ 或 $1$。将这份调查问卷分发给 $m$ 名学生和 $m$ 名老师，学生和老师的编号都是 $0 \sim m - 1$。现在给定一个二维整数数组 $students$ 表示 $m$ 名学生给出的答案，其中 $studuents[i][j]$ 表示第 $i$ 名学生第 $j$ 个问题给出的答案。再给定一个二维整数数组 $mentors$ 表示 $m$ 名老师给出的答案，其中 $mentors[i][j]$ 表示第 $i$ 名导师第 $j$ 个问题给出的答案。

每个学生要和一名导师互相配对。配对的学生和导师之间的兼容性评分等于学生和导师答案相同的次数。

- 例如，学生答案为 $[1, 0, 1]$，而导师答案为 $[0, 0, 1]$，那么他们的兼容性评分为 $2$，因为只有第 $2$ 个和第 $3$ 个答案相同。

**要求**：找出最优的学生与导师的配对方案，以最大程度上提高所有学生和导师的兼容性评分和。然后返回可以得到的最大兼容性评分和。

**说明**：

- $m == students.length == mentors.length$。
- $n == students[i].length == mentors[j].length$。
- $1 \le m, n \le 8$。
- $students[i][k]$ 为 $0$ 或 $1$。
- $mentors[j][k]$ 为 $0$ 或 $1$。

**示例**：

- 示例 1：

```python
输入：students = [[1,1,0],[1,0,1],[0,0,1]], mentors = [[1,0,0],[0,0,1],[1,1,0]]
输出：8
解释：按下述方式分配学生和导师：
- 学生 0 分配给导师 2 ，兼容性评分为 3。
- 学生 1 分配给导师 0 ，兼容性评分为 2。
- 学生 2 分配给导师 1 ，兼容性评分为 3。
最大兼容性评分和为 3 + 2 + 3 = 8。
```

- 示例 2：

```python
输入：students = [[0,0],[0,0],[0,0]], mentors = [[1,1],[1,1],[1,1]]
输出：0
解释：任意学生与导师配对的兼容性评分都是 0。
```

## 解题思路

### 思路 1：状压 DP

因为 $m$、$n$ 的范围都是 $[1, 8]$，所以我们可以使用「状态压缩」的方式来表示学生的分配情况。即使用一个 $m$ 位长度的二进制数 $state$ 来表示每一位老师是否被分配了学生。如果 $state$ 的第 $i$ 位为 $1$，表示第 $i$ 位老师被分配了学生，如果 $state$ 的第 $i$ 位为 $0$，则表示第 $i$ 位老师没有分配到学生。

这样，我们就可以通过动态规划的方式来解决这道题。

###### 1. 划分阶段

按照学生的分配情况进行阶段划分。

###### 2. 定义状态

定义当前学生的分配情况为 $state$，$state$ 中包含 $count(state)$ 个 $1$，表示有 $count(state)$ 个老师被分配了学生。

则可以定义状态 $dp[state]$ 表示为：当前老师被分配学生的状态为 $state$，其中有 $count(state)$ 个老师被分配了学生的情况下，可以得到的最大兼容性评分和。

###### 3. 状态转移方程

对于当前状态 $state$，肯定是从比 $state$ 少选一个老师被分配的状态中递推而来。我们可以枚举少选一个元素的状态，找到可以得到的最大兼容性评分和，赋值给 $dp[state]$。

即状态转移方程为：$dp[state] = max(dp[state], \quad dp[state \oplus (1 \text{ <}\text{< } i)] + score[i][one\underline{\hspace{0.5em}}cnt - 1])$，其中：

1. $state$ 第 $i$ 位一定为 $1$。
2. $state \oplus (1 \text{ <}\text{< } i)$ 为比 $state$ 少选一个元素的状态。
3. $scores[i][one\underline{\hspace{0.5em}}cnt - 1]$ 为第 $i$ 名老师分配到第 $one\underline{\hspace{0.5em}}cnt - 1$ 名学生的兼容性评分。

关于每位老师与每位同学之间的兼容性评分，我们可以事先通过一个 $m \times m \times n$ 的三重循环计算得出，并且存入到 $m \times m$ 大小的二维矩阵 $scores$ 中。

###### 4. 初始条件

- 初始每个老师都没有分配到学生的状态下，可以得到的最兼容性评分和为 $0$，即 $dp[0] = 0$。

###### 5. 最终结果

根据我们之前定义的状态，$dp[state]$ 表示为：当前老师被分配学生的状态为 $state$，其中有 $count(state)$ 个老师被分配了学生的情况下，可以得到的最大兼容性评分和。所以最终结果为 $dp[states - 1]$，其中 $states = 1 \text{ <}\text{< } m$。

### 思路 1：代码

```python
class Solution:
    def maxCompatibilitySum(self, students: List[List[int]], mentors: List[List[int]]) -> int:
        m, n = len(students), len(students[0])
        scores = [[0 for _ in range(m)] for _ in range(m)]

        for i in range(m):
            for j in range(m):
                for k in range(n):
                    scores[i][j] += (students[i][k] == mentors[j][k])

        states = 1 << m
        dp = [0 for _ in range(states)]

        for state in range(states):
            one_cnt = bin(state).count('1')
            for i in range(m):
                if (state >> i) & 1:
                    dp[state] = max(dp[state], dp[state ^ (1 << i)] + scores[i][one_cnt - 1])
        return dp[states - 1]
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(m^2 \times n + m \times 2^m)$。
- **空间复杂度**：$O(2^m)$。

