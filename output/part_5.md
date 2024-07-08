# [0820. 单词的压缩编码](https://leetcode.cn/problems/short-encoding-of-words/)

- 标签：字典树、数组、哈希表、字符串
- 难度：中等

## 题目链接

- [0820. 单词的压缩编码 - 力扣](https://leetcode.cn/problems/short-encoding-of-words/)

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

# [0821. 字符的最短距离](https://leetcode.cn/problems/shortest-distance-to-a-character/)

- 标签：数组、双指针、字符串
- 难度：简单

## 题目链接

- [0821. 字符的最短距离 - 力扣](https://leetcode.cn/problems/shortest-distance-to-a-character/)

## 题目大意

**描述**：给定一个字符串 $s$ 和一个字符 $c$，并且 $c$ 是字符串 $s$ 中出现过的字符。

**要求**：返回一个长度与字符串 $s$ 想通的整数数组 $answer$，其中 $answer[i]$ 是字符串 $s$ 中从下标 $i$ 到离下标 $i$ 最近的字符 $c$ 的距离。

**说明**：

- 两个下标 $i$ 和 $j$ 之间的 **距离** 为 $abs(i - j)$ ，其中 $abs$ 是绝对值函数。
- $1 \le s.length \le 10^4$。
- $s[i]$ 和 $c$ 均为小写英文字母
- 题目数据保证 $c$ 在 $s$ 中至少出现一次。

**示例**：

- 示例 1：

```python
输入：s = "loveleetcode", c = "e"
输出：[3,2,1,0,1,0,0,1,2,2,1,0]
解释：字符 'e' 出现在下标 3、5、6 和 11 处（下标从 0 开始计数）。
距下标 0 最近的 'e' 出现在下标 3，所以距离为 abs(0 - 3) = 3。
距下标 1 最近的 'e' 出现在下标 3，所以距离为 abs(1 - 3) = 2。
对于下标 4，出现在下标 3 和下标 5 处的 'e' 都离它最近，但距离是一样的 abs(4 - 3) == abs(4 - 5) = 1。
距下标 8 最近的 'e' 出现在下标 6，所以距离为 abs(8 - 6) = 2。
```

- 示例 2：

```python
输入：s = "aaab", c = "b"
输出：[3,2,1,0]
```

## 解题思路

### 思路 1：两次遍历

第一次从左到右遍历，记录每个 $i$ 左边最近的 $c$ 的位置，并将其距离记录到 $answer[i]$ 中。

第二次从右到左遍历，记录每个 $i$ 右侧最近的 $c$ 的位置，并将其与第一次遍历左侧最近的 $c$ 的位置相比较，并将较小的距离记录到 $answer[i]$ 中。

### 思路 1：代码

```python
class Solution:
    def shortestToChar(self, s: str, c: str) -> List[int]:
        size = len(s)
        ans = [size + 1 for _ in range(size)]

        pos = -1
        for i in range(size):
            if s[i] == c:
                pos = i
            if pos != -1:
                ans[i] = i - pos
        
        pos = -1
        for i in range(size - 1, -1, -1):
            if s[i] == c:
                pos = i
            if pos != -1:
                ans[i] = min(ans[i], pos - i)

        return ans

```

### 思路 1：复杂度分析

- **时间复杂度**：$O(n)$。
- **空间复杂度**：$O(1)$。

# [0824. 山羊拉丁文](https://leetcode.cn/problems/goat-latin/)

- 标签：字符串
- 难度：简单

## 题目链接

- [0824. 山羊拉丁文 - 力扣](https://leetcode.cn/problems/goat-latin/)

## 题目大意

**描述**：给定一个由若干单词组成的句子 $sentence$，单词之间由空格分隔。每个单词仅由大写和小写字母组成。

**要求**：将句子转换为「山羊拉丁文（Goat Latin）」，并返回将 $sentence$ 转换为山羊拉丁文后的句子。

**说明**：

- 山羊拉丁文的规则如下：
  - 如果单词以元音开头（`a`，`e`，`i`，`o`，`u`），在单词后添加 `"ma"`。
    - 例如，单词 `"apple"` 变为 `"applema"`。

  - 如果单词以辅音字母开头（即，非元音字母），移除第一个字符并将它放到末尾，之后再添加 `"ma"`。
    - 例如，单词 `"goat"` 变为 `"oatgma"`。

  - 根据单词在句子中的索引，在单词最后添加与索引相同数量的字母 `a`，索引从 $1$ 开始。
    - 例如，在第一个单词后添加 `"a"` ，在第二个单词后添加 `"aa"`，以此类推。

- $1 \le sentence.length \le 150$。
- $sentence$ 由英文字母和空格组成。
- $sentence$ 不含前导或尾随空格。
- $sentence$ 中的所有单词由单个空格分隔。

**示例**：

- 示例 1：

```python
输入：sentence = "I speak Goat Latin"
输出："Imaa peaksmaaa oatGmaaaa atinLmaaaaa"
```

- 示例 2：

```python
输入：sentence = "The quick brown fox jumped over the lazy dog"
输出："heTmaa uickqmaaa rownbmaaaa oxfmaaaaa umpedjmaaaaaa overmaaaaaaa hetmaaaaaaaa azylmaaaaaaaaa ogdmaaaaaaaaaa"
```

## 解题思路

### 思路 1：模拟

1. 使用集合 $vowels$ 存储元音字符，然后将 $sentence$ 按照空格分隔成单词数组 $words$。
2. 遍历单词数组 $words$，对于当前单词 $word$，根据山羊拉丁文的规则，将其转为山羊拉丁文的单词，并存入答案数组 $res$ 中。
3. 遍历完之后将答案数组拼接为字符串并返回。

### 思路 1：代码

```python
class Solution:
    def toGoatLatin(self, sentence: str) -> str:
        vowels = set(['a', 'e', 'i', 'o', 'u', 'A', 'E', 'I', 'O', 'U'])
        words = sentence.split(' ')
        res = []
        for i in range(len(words)):
            word = words[i]
            ans = ""
            if word[0] in vowels:
                ans += word + "ma"
            else:
                ans += word[1:] + word[0] + "ma"
            ans += 'a' * (i + 1)
            res.append(ans)

        return " ".join(res)
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(n)$。
- **空间复杂度**：$O(1)$。

# [0830. 较大分组的位置](https://leetcode.cn/problems/positions-of-large-groups/)

- 标签：字符串
- 难度：简单

## 题目链接

- [0830. 较大分组的位置 - 力扣](https://leetcode.cn/problems/positions-of-large-groups/)

## 题目大意

**描述**：给定由小写字母构成的字符串 $s$。字符串 $s$ 包含一些连续的相同字符所构成的分组。

**要求**：找到每一个较大分组的区间，按起始位置下标递增顺序排序后，返回结果。

**说明**：

- **较大分组**：我们称所有包含大于或等于三个连续字符的分组为较大分组。

**示例**：

- 示例 1：

```python
输入：s = "abbxxxxzzy"
输出：[[3,6]]
解释："xxxx" 是一个起始于 3 且终止于 6 的较大分组。
```

- 示例 2：

```python
输入：s = "abc"
输出：[]
解释："a","b" 和 "c" 均不是符合要求的较大分组。
```

## 解题思路

### 思路 1：简单模拟

遍历字符串 $s$，统计出所有大于等于 $3$ 个连续字符的子字符串的开始位置与结束位置。具体步骤如下：

1. 令 $cnt = 1$，然后从下标 $1$ 位置开始遍历字符串 $s$。
	1. 如果 $s[i - 1] == s[i]$，则令 $cnt$ 加 $1$。
	2. 如果 $s[i - 1] \ne s[i]$，说明出现了不同字符，则判断之前连续字符个数 $cnt$ 是否大于等于 $3$。
	3. 如果 $cnt \ge 3$，则将对应包含 $cnt$ 个连续字符的子字符串的开始位置与结束位置存入答案数组中。
	4. 令 $cnt = 1$，重新开始记录连续字符个数。
2. 遍历完字符串 $s$，输出答案数组。

### 思路 1：代码

```python
class Solution:
    def largeGroupPositions(self, s: str) -> List[List[int]]:
        res = []
        cnt = 1
        size = len(s)
        for i in range(1, size):
            if s[i] == s[i - 1]:
                cnt += 1
            else:
                if cnt >= 3:
                    res.append([i - cnt, i - 1])
                cnt = 1
        if cnt >= 3:
            res.append([size - cnt, size - 1])
        return res
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(n)$。
- **空间复杂度**：$O(1)$。
# [0832. 翻转图像](https://leetcode.cn/problems/flipping-an-image/)

- 标签：数组、双指针、矩阵、模拟
- 难度：简单

## 题目链接

- [0832. 翻转图像 - 力扣](https://leetcode.cn/problems/flipping-an-image/)

## 题目大意

给定一个二进制矩阵 `A` 代表图像，先将矩阵进行水平翻转，再进行翻转（将 0 变为 1，1 变为 0）。

## 解题思路

两重 for 循环，第二层 for 循环遍历到一半即可。对于 `image[i][j]`、`image[i][n-1-j]` 先水平翻转操作，再进行翻转。

## 代码

```python
class Solution:
    def flipAndInvertImage(self, image: List[List[int]]) -> List[List[int]]:
        n = len(image)
        for i in range(n):
            for j in range((n+1)//2):
                image[i][j], image[i][n-1-j] = image[i][n-1-j], image[i][j]
                image[i][j] = 0 if image[i][j] == 1 else 1
                if j != n-1-j:
                    image[i][n-1-j] = 0 if image[i][n-1-j] == 1 else 1
        return image
```

# [0834. 树中距离之和](https://leetcode.cn/problems/sum-of-distances-in-tree/)

- 标签：树、深度优先搜索、图、动态规划
- 难度：困难

## 题目链接

- [0834. 树中距离之和 - 力扣](https://leetcode.cn/problems/sum-of-distances-in-tree/)

## 题目大意

**描述**：给定一个无向、连通的树。树中有 $n$ 个标记为 $0 \sim n - 1$ 的节点以及 $n - 1$ 条边 。

给定整数 $n$ 和数组 $edges$，其中 $edges[i] = [ai, bi]$ 表示树中的节点 $ai$ 和 $bi$ 之间有一条边。

**要求**：返回长度为 $n$ 的数组 $answer$，其中 $answer[i]$ 是树中第 $i$ 个节点与所有其他节点之间的距离之和。

**说明**：

- $1 \le n \le 3 \times 10^4$。
- $edges.length == n - 1$。
- $edges[i].length == 2$。
- $0 \le ai, bi < n$。
- $ai \ne bi$。
- 给定的输入保证为有效的树。

**示例**：

- 示例 1：

![](https://assets.leetcode.com/uploads/2021/07/23/lc-sumdist1.jpg)

```python
输入: n = 6, edges = [[0,1],[0,2],[2,3],[2,4],[2,5]]
输出: [8,12,6,10,10,10]
解释: 树如图所示。
我们可以计算出 dist(0,1) + dist(0,2) + dist(0,3) + dist(0,4) + dist(0,5) 
也就是 1 + 1 + 2 + 2 + 2 = 8。 因此，answer[0] = 8，以此类推。
```

- 示例 2：

![](https://assets.leetcode.com/uploads/2021/07/23/lc-sumdist3.jpg)

```python
输入: n = 2, edges = [[1,0]]
输出: [1,1]
```

## 解题思路

### 思路 1：树形 DP + 二次遍历换根法

最容易想到的做法是：枚举 $n$ 个节点，以每个节点为根节点进行树形 DP。

对于节点 $u$，定义 $dp[u]$ 为：以节点 $u$ 为根节点的树，它的所有子节点到它的距离之和。

然后进行一轮深度优先搜索，在搜索的过程中得到以节点 $v$ 为根节点的树，节点 $v$ 与所有其他子节点之间的距离之和 $dp[v]$。还能得到子树的节点个数 $sizes[v]$。

对于节点 $v$ 来说，其对 $dp[u]$ 的贡献为：节点 $v$ 与所有其他子节点之间的距离之和，再加上需要经过 $u \rightarrow v$ 这条边的节点个数，即 $dp[v] + sizes[v]$。

可得到状态转移方程为：$dp[u] = \sum_{v \in graph[u]}(dp[v] + sizes[v])$。

这样，对于 $n$ 个节点来说，需要进行 $n$ 次树形 DP，这种做法的时间复杂度为 $O(n^2)$，而 $n$ 的范围为 $[1, 3 \times 10^4]$，这样做会导致超时，因此需要进行优化。

我们可以使用「二次遍历换根法」进行优化，从而在 $O(n)$ 的时间复杂度内解决这道题。

以编号为 $0$ 的节点为根节点，进行两次深度优先搜索。

1. 第一次遍历：从编号为 $0$ 的根节点开始，自底向上地计算出节点 $0$ 到其他的距离之和，记录在 $ans[0]$ 中。并且统计出以子节点为根节点的子树节点个数 $sizes[v]$。
2. 第二次遍历：从编号为 $0$ 的根节点开始，自顶向下地枚举每个点，计算出将每个点作为新的根节点时，其他节点到根节点的距离之和。如果当前节点为 $v$，其父节点为 $u$，则自顶向下计算出 $ans[u]$ 之后，我们将根节点从 $u$ 换为节点 $v$，子树上的点到新根节点的距离比原来都小了 $1$，非子树上剩下所有点到新根节点的距离比原来都大了 $1$。则可以据此计算出节点 $v$ 与其他节点的距离和为：$ans[v] = ans[u] + n - 2 \times sizes[u]$。

### 思路 1：代码

```python
class Solution:
    def sumOfDistancesInTree(self, n: int, edges: List[List[int]]) -> List[int]:
        graph = [[] for _ in range(n)]

        for u, v in edges:
            graph[u].append(v)
            graph[v].append(u)


        ans = [0 for _ in range(n)]

        sizes = [1 for _ in range(n)]
        def dfs(u, fa, depth):
            ans[0] += depth
            for v in graph[u]:
                if v == fa:
                    continue
                dfs(v, u, depth + 1)
                sizes[u] += sizes[v]

        def reroot(u, fa):
            for v in graph[u]:
                if v == fa:
                    continue
                ans[v] = ans[u] + n - 2 * size[v]
                reroot(v, u)

        dfs(0, -1, 0)
        reroot(0, -1)
        return ans
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(n)$，其中 $n$ 为树的节点个数。
- **空间复杂度**：$O(n)$。

# [0836. 矩形重叠](https://leetcode.cn/problems/rectangle-overlap/)

- 标签：几何、数学
- 难度：简单

## 题目链接

- [0836. 矩形重叠 - 力扣](https://leetcode.cn/problems/rectangle-overlap/)

## 题目大意

给定两个矩形的左下角、右上角坐标：[x1, y1, x2, y2]。[x1, y1] 表示左下角坐标，[x2, y2] 表示右上角坐标。如果两个矩形相交面积大于 0，则称两矩形重叠。

要求：根据给定的矩形 rec1 和 rec2 的左下角、右上角坐标，如果重叠，则返回 True，否则返回 False。

## 解题思路

如果两个矩形重叠，则两个矩形的水平边投影到 x 轴上的线段会有交集，同理竖直边投影到 y 轴上的线段也会有交集。因此我们可以把问题看做是：判断两条线段是否有交集。

矩形 rec1 和 rec2 水平边投影到 x 轴上的线段为 `(rec1[0], rec1[2])` 和 `(rec2[0], rec2[2])`。如果两条线段有交集，则 `min(rec1[2], rec2[2]) > max(rec1[0], rec2[0])`。

矩形 rec1 和 rec2 竖直边投影到 y 轴上的线段为 `(rec1[1], rec1[3])` 和 `(rec2[1], rec2[3])`。如果两条线段有交集，则 `min(rec1[3], rec2[3]) > max(rec1[1], rec2[1])`。

判断是否满足上述条件，若满足则说明两个矩形重叠，返回 True，若不满足则返回 False。

## 代码

```python
class Solution:
    def isRectangleOverlap(self, rec1: List[int], rec2: List[int]) -> bool:
         return min(rec1[2], rec2[2]) > max(rec1[0], rec2[0]) and min(rec1[3], rec2[3]) > max(rec1[1], rec2[1])
```

# [0841. 钥匙和房间](https://leetcode.cn/problems/keys-and-rooms/)

- 标签：深度优先搜索、广度优先搜索、图
- 难度：中等

## 题目链接

- [0841. 钥匙和房间 - 力扣](https://leetcode.cn/problems/keys-and-rooms/)

## 题目大意

**描述**：有 `n` 个房间，编号为 `0` ~ `n - 1`，每个房间都有若干把钥匙，每把钥匙上都有一个编号，可以开启对应房间号的门。最初，除了 `0` 号房间外其他房间的门都是锁着的。

现在给定一个二维数组 `rooms`，`rooms[i][j]` 表示第 `i` 个房间的第 `j` 把钥匙所能开启的房间号。

**要求**：判断是否能开启所有房间的门。如果能开启，则返回 `True`。否则返回 `False`。

**说明**：

- $n == rooms.length$。
- $2 \le n \le 1000$。
- $0 \le rooms[i].length \le 1000$。
- $1 \le sum(rooms[i].length) \le 3000$。
- $0 \le rooms[i][j] < n$。
- 所有 $rooms[i]$ 的值互不相同。

**示例**：

- 示例 1：

```python
输入：rooms = [[1],[2],[3],[]]
输出：True
解释：
我们从 0 号房间开始，拿到钥匙 1。
之后我们去 1 号房间，拿到钥匙 2。
然后我们去 2 号房间，拿到钥匙 3。
最后我们去了 3 号房间。
由于我们能够进入每个房间，我们返回 true。
```

- 示例 2：

```python
输入：rooms = [[1,3],[3,0,1],[2],[0]]
输出：False
解释：我们不能进入 2 号房间。
```

## 解题思路

### 思路 1：深度优先搜索

当 `x` 号房间有 `y` 号房间的钥匙时，就可以认为我们可以通过 `x` 号房间去往 `y` 号房间。现在把 `n` 个房间看做是拥有 `n` 个节点的图，则上述关系可以看做是 `x` 与 `y` 点之间有一条有向边。

那么问题就变为了给定一张有向图，从 `0` 节点开始出发，问是否能到达所有的节点。

我们可以使用深度优先搜索的方式来解决这道题，具体做法如下：

1. 使用 set 集合变量 `visited` 来统计遍历到的节点个数。
2. 从 `0` 节点开始，使用深度优先搜索的方式遍历整个图。
3. 将当前节点 `x` 加入到集合 `visited` 中，遍历当前节点的邻接点。
   1. 如果邻接点不再集合 `visited` 中，则继续递归遍历。
4. 最后深度优先搜索完毕，判断一下遍历到的节点个数是否等于图的节点个数（即集合 `visited` 中的元素个数是否等于节点个数）。
   1. 如果等于，则返回 `True`
   2. 如果不等于，则返回 `False`。

### 思路 1：代码

```python
class Solution:
    def canVisitAllRooms(self, rooms: List[List[int]]) -> bool:
        def dfs(x):
            visited.add(x)
            for key in rooms[x]:
                if key not in visited:
                    dfs(key)
        visited = set()
        dfs(0)
        return len(visited) == len(rooms)
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(n + m)$，其中 $n$ 是房间的数量，$m$ 是所有房间中的钥匙数量的总数。
- **空间复杂度**：$O(n)$，递归调用的栈空间深度不超过 $n$。# [0844. 比较含退格的字符串](https://leetcode.cn/problems/backspace-string-compare/)

- 标签：栈、双指针、字符串、模拟
- 难度：简单

## 题目链接

- [0844. 比较含退格的字符串 - 力扣](https://leetcode.cn/problems/backspace-string-compare/)

## 题目大意

**描述**：给定 $s$ 和 $t$ 两个字符串。字符串中的 `#` 代表退格字符。

**要求**：当它们分别被输入到空白的文本编辑器后，判断二者是否相等。如果相等，返回 $True$；否则，返回 $False$。

**说明**：

- 如果对空文本输入退格字符，文本继续为空。
- $1 \le s.length, t.length \le 200$。
- $s$ 和 $t$ 只含有小写字母以及字符 `#`。

**示例**：

- 示例 1：

```python
输入：s = "ab#c", t = "ad#c"
输出：true
解释：s 和 t 都会变成 "ac"。
```

- 示例 2：

```python
输入：s = "ab##", t = "c#d#"
输出：true
解释：s 和 t 都会变成 ""。
```

## 解题思路

这道题的第一个思路是用栈，第二个思路是使用分离双指针。

### 思路 1：栈

- 定义一个构建方法，用来将含有退格字符串构建为删除退格的字符串。构建方法如下。
  - 使用一个栈存放删除退格的字符串。
  - 遍历字符串，如果遇到的字符不是 `#`，则将其插入到栈中。
  - 如果遇到的字符是 `#`，且当前栈不为空，则将当前栈顶元素弹出。
- 分别使用构建方法处理字符串 $s$ 和 $t$，如果处理完的字符串 $s$ 和 $t$ 相等，则返回 $True$，否则返回 $False$。

### 思路 1：代码

```python
class Solution:
    def build(self, s: str):
        stack = []
        for ch in s:
            if ch != '#':
                stack.append(ch)
            elif stack:
                stack.pop()
        return stack
        
    def backspaceCompare(self, s: str, t: str) -> bool:
        return self.build(s) == self.build(t)
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(n + m)$，其中 $n$ 和 $m$ 分别为字符串 $s$、$t$ 的长度。
- **空间复杂度**：$O(n + m)$。

### 思路 2：分离双指针

由于 `#` 会消除左侧字符，而不会影响右侧字符，所以我们选择从字符串尾端遍历 $s$、$t$ 字符串。具体做法如下：

- 使用分离双指针 $left\underline{\hspace{0.5em}}1$、$left\underline{\hspace{0.5em}}2$。$left\underline{\hspace{0.5em}}1$ 指向字符串 $s$ 末尾，$left\underline{\hspace{0.5em}}2$ 指向字符串 $t$ 末尾。使用 $sign\underline{\hspace{0.5em}}1$、$sign\underline{\hspace{0.5em}}2$ 标记字符串 $s$、$t$ 中当前退格字符个数。
- 从后到前遍历字符串 $s$、$t$。
  - 先来循环处理字符串 $s$ 尾端 `#` 的影响，具体如下：
    - 如果当前字符是 `#`，则更新 $s$ 当前退格字符个数，即 `sign_1 += 1`。同时将 $left\underline{\hspace{0.5em}}1$ 左移。 
    - 如果 $s$ 当前退格字符个数大于 $0$，则退格数减一，即 `sign_1 -= 1`。同时将 $left\underline{\hspace{0.5em}}1$ 左移。 
    - 如果 $s$ 当前为普通字符，则跳出循环。
  - 同理再来处理字符串 $t$ 尾端 `#` 的影响，具体如下：
    - 如果当前字符是 `#`，则更新 $t$ 当前退格字符个数，即 `sign_2 += 1`。同时将 $left\underline{\hspace{0.5em}}2$ 左移。 
    - 如果 $t$ 当前退格字符个数大于 $0$，则退格数减一，即 `sign_2 -= 1`。同时将 $left\underline{\hspace{0.5em}}2$ 左移。 
    - 如果 $t$ 当前为普通字符，则跳出循环。
  - 处理完，如果两个字符串为空，则说明匹配，直接返回 $True$。
  - 再先排除长度不匹配的情况，直接返回 $False$。
  - 最后判断 $s[left\underline{\hspace{0.5em}}1]$ 是否等于 $s[left\underline{\hspace{0.5em}}2]$。不等于则直接返回 $False$，等于则令 $left\underline{\hspace{0.5em}}1$、$left\underline{\hspace{0.5em}}2$ 左移，继续遍历。
- 遍历完没有出现不匹配的情况，则返回 $True$。

### 思路 2：代码

```python
class Solution:
    def backspaceCompare(self, s: str, t: str) -> bool:
        left_1, left_2 = len(s) - 1, len(t) - 1
        sign_1, sign_2 = 0, 0
        while left_1 >= 0 or left_2 >= 0:
            while left_1 >= 0:
                if s[left_1] == '#':
                    sign_1 += 1
                    left_1 -= 1
                elif sign_1 > 0:
                    sign_1 -= 1
                    left_1 -= 1
                else:
                    break

            while left_2 >= 0:
                if t[left_2] == '#':
                    sign_2 += 1
                    left_2 -= 1
                elif sign_2 > 0:
                    sign_2 -= 1
                    left_2 -= 1
                else:
                    break

            if left_1 < 0 and left_2 < 0:
                return True
            if left_1 >= 0 and left_2 < 0:
                return False
            if left_1 < 0 and left_2 >= 0:
                return False
            if s[left_1] != t[left_2]:
                return False

            left_1 -= 1
            left_2 -= 1

        return True
```

### 思路 2：复杂度分析

- **时间复杂度**：$O(n + m)$，其中 $n$ 和 $m$ 分别为字符串 $s$、$t$ 的长度。
- **空间复杂度**：$O(1)$。

# [0845. 数组中的最长山脉](https://leetcode.cn/problems/longest-mountain-in-array/)

- 标签：数组、双指针、动态规划、枚举
- 难度：中等

## 题目链接

- [0845. 数组中的最长山脉 - 力扣](https://leetcode.cn/problems/longest-mountain-in-array/)

## 题目大意

**描述**：给定一个整数数组 $arr$。

**要求**：返回最长山脉子数组的长度。如果不存在山脉子数组，返回 $0$。

**说明**：

- **山脉数组**：符合下列属性的数组 $arr$ 称为山脉数组。
  - $arr.length \ge 3$。
  - 存在下标 $i(0 < i < arr.length - 1)$ 满足：
    - $arr[0] < arr[1] < … < arr[i]$
    - $arr[i] > arr[i + 1] > … > arr[arr.length - 1]$

- $1 \le arr.length \le 10^4$。
- $0 \le arr[i] \le 10^4$。

**示例**：

- 示例 1：

```python
输入：arr = [2,1,4,7,3,2,5]
输出：5
解释：最长的山脉子数组是 [1,4,7,3,2]，长度为 5。
```

- 示例 2：

```python
输入：arr = [2,2,2]
输出：0
解释：不存在山脉子数组。
```

## 解题思路

### 思路 1：快慢指针

1. 使用变量 $ans$ 保存最长山脉长度。
2. 遍历数组，假定当前节点为山峰。
3. 使用双指针 $left$、$right$ 分别向左、向右查找山脉的长度。
4. 如果当前山脉的长度比最长山脉长度更长，则更新最长山脉长度。
5. 最后输出 $ans$。

### 思路 1：代码

```python
class Solution:
    def longestMountain(self, arr: List[int]) -> int:
        size = len(arr)
        res = 0
        for i in range(1, size - 1):
            if arr[i] > arr[i - 1] and arr[i] > arr[i + 1]:
                left = i - 1
                right = i + 1

                while left > 0 and arr[left - 1] < arr[left]:
                    left -= 1
                while right < size - 1 and arr[right + 1] < arr[right]:
                    right += 1
                if right - left + 1 > res:
                    res = right - left + 1
        return res
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(n)$，其中 $n$ 为数组 $arr$ 中的元素数量。
- **空间复杂度**：$O(1)$。

# [0846. 一手顺子](https://leetcode.cn/problems/hand-of-straights/)

- 标签：贪心、数组、哈希表、排序
- 难度：中等

## 题目链接

- [0846. 一手顺子 - 力扣](https://leetcode.cn/problems/hand-of-straights/)

## 题目大意

**描述**：`Alice` 手中有一把牌，她想要重新排列这些牌，分成若干组，使每一组的牌都是顺子（即由连续的牌构成），并且每一组的牌数都是 `groupSize`。现在给定一个整数数组 `hand`，其中 `hand[i]` 是表示第 `i` 张牌的数值，和一个整数 `groupSize`。

**要求**：如果 `Alice` 能将这些牌重新排列成若干组、并且每组都是 `goupSize` 张牌的顺子，则返回 `True`；否则，返回 `False`。

**说明**：

- $1 \le hand.length \le 10^4$。
- $0 \le hand[i] \le 10^9$。
- $1 \le groupSize \le hand.length$。

**示例**：

- 示例 1：

```python
输入：hand = [1,2,3,6,2,3,4,7,8], groupSize = 3
输出：True
解释：Alice 手中的牌可以被重新排列为 [1,2,3]，[2,3,4]，[6,7,8]。
```

## 解题思路

### 思路 1：哈希表 + 排序

1. 使用哈希表存储每个数出现的次数。
2. 将哈希表中每个键从小到大排序。
3. 从哈希表中最小的数开始，以它作为当前顺子的开头，然后依次判断顺子里的数是否在哈希表中，如果在的话，则将哈希表中对应数的数量减 `1`。不在的话，说明无法满足题目要求，直接返回 `False`。
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
# [0847. 访问所有节点的最短路径](https://leetcode.cn/problems/shortest-path-visiting-all-nodes/)

- 标签：位运算、广度优先搜索、图、动态规划、状态压缩
- 难度：困难

## 题目链接

- [0847. 访问所有节点的最短路径 - 力扣](https://leetcode.cn/problems/shortest-path-visiting-all-nodes/)

## 题目大意

**描述**：存在一个由 $n$ 个节点组成的无向连通图，图中节点编号为 $0 \sim n - 1$。现在给定一个数组 $graph$ 表示这个图。其中，$graph[i]$ 是一个列表，由所有与节点 $i$ 直接相连的节点组成。

**要求**：返回能够访问所有节点的最短路径长度。可以在任一节点开始和停止，也可以多次重访节点，并且可以重用边。

**说明**：

- $n == graph.length$。
- $1 \le n \le 12$。
- $0 \le graph[i].length < n$。
- $graph[i]$ 不包含 $i$。
- 如果 $graph[a]$ 包含 $b$，那么 $graph[b]$ 也包含 $a$。
- 输入的图总是连通图。

**示例**：

- 示例 1：

![](https://assets.leetcode.com/uploads/2021/05/12/shortest1-graph.jpg)

```python
输入：graph = [[1,2,3],[0],[0],[0]]
输出：4
解释：一种可能的路径为 [1,0,2,0,3]
```

- 示例 2：

![](https://assets.leetcode.com/uploads/2021/05/12/shortest2-graph.jpg)

```python
输入：graph = [[1],[0,2,4],[1,3,4],[2],[1,2]]
输出：4
解释：一种可能的路径为 [0,1,4,2,3]
```

## 解题思路

### 思路 1：状态压缩 + 广度优先搜索

 题目需要求解的是「能够访问所有节点的最短路径长度」，并且每个节点都可以作为起始点。

如果对于一个特定的起点，我们可以将该起点放入队列中，然后对其进行广度优先搜索，并使用访问数组 $visited$ 标记访问过的节点，直到所有节点都已经访问过时，返回路径长度即为「从某点开始出发，所能够访问所有节点的最短路径长度」。

而本题中，每个节点都可以作为起始点，则我们可以直接将所有节点放入队列中，然后对所有节点进行广度优先搜索。

因为本题中节点数目 $n$ 的范围为 $[1, 12]$，所以我们可以采用「状态压缩」的方式，标记节点的访问情况。每个点的初始状态可以表示为 `(u, 1 << u)`。当状态 $state == 1 \text{ <}\text{< } n - 1$ 时，表示所有节点都已经访问过了，此时返回其对应路径长度即为「能够访问所有节点的最短路径长度」。

为了方便在广度优先搜索的同事，记录当前的「路径长度」以及「节点的访问情况」。我们可以使用一个三元组 $(u, state, dist)$ 来表示当前节点情况，其中：

- $u$：表示当前节点编号。
- $state$：一个 $n$ 位的二进制数，表示 $n$ 个节点的访问情况。$state$ 第 $i$ 位为 $0$ 时表示未访问过，$state$ 第 $i$ 位为 $1$ 时表示访问过。
- $dist$ 表示当前的「路径长度」。

同时为了避免重复搜索同一个节点 $u$ 以及相同节点的访问情况，我们可以使用集合记录 $(u, state)$ 是否已经被搜索过。

整个算法步骤如下：

1. 将所有节点的 `(节点编号, 起始状态, 路径长度)` 作为三元组存入队列，并使用集合 $visited$ 记录所有节点的访问情况。
2. 对所有点开始进行广度优先搜索：
   1. 从队列中弹出队头节点。
   2. 判断节点的当前状态，如果所有节点都已经访问过，则返回答案。
   3. 如果没有全访问过，则遍历当前节点的邻接节点。
   4. 将邻接节点的访问状态标记为访问过。
   5. 如果节点即当前路径没有访问过，则加入队列继续遍历，并标记为访问过。
3. 重复进行第 $2$ 步，直到队列为空。

### 思路 1：代码

```python
import collections


class Solution:
    def shortestPathLength(self, graph: List[List[int]]) -> int:
        size = len(graph)

        queue = collections.deque([])
        visited = set()
        for u in range(size):
            queue.append((u, 1 << u, 0))            # 将 (节点编号, 起始状态, 路径长度) 存入队列
            visited.add((u, 1 << u))                # 标记所有节点的节点编号，以及当前状态

        while queue:                                # 对所有点开始进行广度优先搜索
            u, state, dist = queue.popleft()        # 弹出队头节点
            if state == (1 << size) - 1:            # 所有节点都访问完，返回答案
                return dist
            for v in graph[u]:                      # 遍历邻接节点
                next_state = state | (1 << v)       # 标记邻接节点的访问状态
                if (v, next_state) not in visited:  # 如果节点即当前路径没有访问过，则加入队列继续遍历，并标记为访问过
                    queue.append((v, next_state, dist + 1))
                    visited.add((v, next_state))

        return 0
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(n^2 \times 2^n)$，其中 $n$ 为图的节点数量。
- **空间复杂度**：$O(n \times 2^n)$。

# [0850. 矩形面积 II](https://leetcode.cn/problems/rectangle-area-ii/)

- 标签：线段树、数组、有序集合、扫描线
- 难度：困难

## 题目链接

- [0850. 矩形面积 II - 力扣](https://leetcode.cn/problems/rectangle-area-ii/)

## 题目大意

**描述**：给定一个二维矩形列表 `rectangles`，其中 `rectangle[i] = [x1, y1, x2, y2]` 表示第 `i` 个矩形，`(x1, y1)` 是第 `i` 个矩形左下角的坐标，`(x2, y2)` 是第 `i` 个矩形右上角的坐标。。

**要求**：计算 `rectangles` 中所有矩形所覆盖的总面积，并返回总面积。

**说明**：

- 任何被两个或多个矩形覆盖的区域应只计算一次 。
- 因为答案可能太大，返回 $10^9 + 7$ 的模。
- $1 \le rectangles.length \le 200$。
- $rectanges[i].length = 4$。
- $0 \le x_1, y_1, x_2, y_2 \le 10^9$。
- 矩形叠加覆盖后的总面积不会超越 $2^63 - 1$，这意味着可以用一个 $64$ 位有符号整数来保存面积结果。

**示例**：

- 示例 1：

![](https://s3-lc-upload.s3.amazonaws.com/uploads/2018/06/06/rectangle_area_ii_pic.png)

```python
输入：rectangles = [[0,0,2,2],[1,0,2,3],[1,0,3,1]]
输出：6
解释：如图所示，三个矩形覆盖了总面积为6的区域。
从 (1,1) 到 (2,2)，绿色矩形和红色矩形重叠。
从 (1,0) 到 (2,3)，三个矩形都重叠。
```

## 解题思路

### 思路 1：扫描线 + 动态开点线段树



### 思路 1：扫描线 + 动态开点线段树代码

```python
# 线段树的节点类
class SegTreeNode:
    def __init__(self, left=-1, right=-1, cnt=0, height=0, leftNode=None, rightNode=None):
        self.left = left                            # 区间左边界
        self.right = right                          # 区间右边界
        self.mid = left + (right - left) // 2
        self.leftNode = leftNode                    # 区间左节点
        self.rightNode = rightNode                  # 区间右节点
        self.cnt = cnt                              # 节点值（区间值）
        self.height = height                        # 区间问题的延迟更新标记
        
        
# 线段树类
class SegmentTree:
    # 初始化线段树接口
    def __init__(self):
        self.tree = SegTreeNode(0, int(1e9))
        
    # 区间更新接口：将区间为 [q_left, q_right] 上的元素值修改为 val
    def update_interval(self, q_left, q_right, val):
        self.__update_interval(q_left, q_right, val, self.tree)
        
    # 区间查询接口：查询区间为 [q_left, q_right] 的区间值
    def query_interval(self, q_left, q_right):
        return self.__query_interval(q_left, q_right, self.tree)
    
    
    # 以下为内部实现方法
        
    # 区间更新实现方法
    def __update_interval(self, q_left, q_right, val, node):
    
        if node.right < q_left or node.left > q_right:  # 节点所在区间与 [q_left, q_right] 无关
            return
        
        if node.left >= q_left and node.right <= q_right:  # 节点所在区间被 [q_left, q_right] 所覆盖
            node.cnt += val                              # 当前节点所在区间每个元素值改为 val
            self.__pushup(node)
            return

        
        self.__pushdown(node) 
        
        if q_left <= node.mid:                      # 在左子树中更新区间值
            self.__update_interval(q_left, q_right, val, node.leftNode)
        if q_right > node.mid:                      # 在右子树中更新区间值
            self.__update_interval(q_left, q_right, val, node.rightNode)
    
        self.__pushup(node)
    
    # 区间查询实现方法：在线段树的 [left, right] 区间范围中搜索区间为 [q_left, q_right] 的区间值
    def __query_interval(self, q_left, q_right, node):
        if node.right < q_left or node.left > q_right:  # 节点所在区间与 [q_left, q_right] 无关
            return 0
        
        if node.left >= q_left and node.right <= q_right:   # 节点所在区间被 [q_left, q_right] 所覆盖
            return node.height                      # 直接返回节点值
        
        self.__pushdown(node) 
        
        res_left = 0                                # 左子树查询结果
        res_right = 0                               # 右子树查询结果
        if q_left <= node.mid:                      # 在左子树中查询
            res_left = self.__query_interval(q_left, node.mid, node.leftNode)
        if q_right > node.mid:                      # 在右子树中查询
            res_right = self.__query_interval(node.mid + 1, q_right, node.rightNode)
            
        
        return res_left + res_right                 # 返回左右子树元素值的聚合计算结果
    
    # 向上更新实现方法：更新 node 节点区间值 等于 该节点左右子节点元素值的聚合计算结果
    def __pushup(self, node):
        if node.cnt > 0:
            node.height = node.right - node.left + 1
        else:
            if node.leftNode and node.rightNode:
                node.height = node.leftNode.height + node.rightNode.height
            else:
                node.height = 0
    
    # 向下更新实现方法：更新 node 节点所在区间的左右子节点的值和懒惰标记
    def __pushdown(self, node):
        if node.leftNode is None:
            node.leftNode = SegTreeNode(node.left, node.mid)
        if node.rightNode is None:
            node.rightNode = SegTreeNode(node.mid + 1, node.right)
            
class Solution:
    def rectangleArea(self, rectangles) -> int:
        # lines 存储每个矩阵的上下两条边
        lines = []
        
        for rectangle in rectangles:
            x1, y1, x2, y2 = rectangle
            lines.append([x1, y1 + 1, y2, 1])
            lines.append([x2, y1 + 1, y2, -1])
            
        lines.sort(key=lambda line: line[0])
            
        # 建立线段树
        self.STree = SegmentTree()
        
        ans = 0
        mod = 10 ** 9 + 7
        prev_x = lines[0][0]
        for i in range(len(lines)):
            x, y1, y2, val = lines[i]
            height = self.STree.query_interval(0, int(1e9))
            ans += height * (x - prev_x)
            ans %= mod
            self.STree.update_interval(y1, y2, val)
            prev_x = x
            
        return ans
```

## 参考资料

- 【文章】[【hdu1542】线段树求矩形面积并 - 拦路雨偏似雪花](https://www.cnblogs.com/KonjakJuruo/p/6024266.html)
# [0851. 喧闹和富有](https://leetcode.cn/problems/loud-and-rich/)

- 标签：深度优先搜索、图、拓扑排序、数组
- 难度：中等

## 题目链接

- [0851. 喧闹和富有 - 力扣](https://leetcode.cn/problems/loud-and-rich/)

## 题目大意

**描述**：有一组 `n` 个人作为实验对象，从 `0` 到 `n - 1` 编号，其中每个人都有不同数目的钱，以及不同程度的安静值 `quietness`。

现在给定一个数组 `richer`，其中 `richer[i] = [ai, bi]` 表示第 `ai` 个人比第 `bi` 个人更有钱。另给你一个整数数组 `quiet`，其中 `quiet[i]` 是第 `i` 个人的安静值。数组 `richer` 中所给出的数据逻辑自洽（也就是说，在第 `ai` 个人比第 `bi` 个人更有钱的同时，不会出现第 `bi` 个人比第 `ai` 个人更有钱的情况 ）。

**要求**：返回一个长度为 `n` 的整数数组 `answer` 作为答案，其中 `answer[i]` 表示在所有比第 `i` 个人更有钱或者和他一样有钱的人中，安静值最小的那个人的编号。 

**说明**：

- $n == quiet.length$
- $1 \le n \le 500$。
- $0 \le quiet[i] \le n$。
- $quiet$ 的所有值互不相同。
- $0 \le richer.length \le n * (n - 1) / 2$。
- $0 \le ai, bi < n$。
- $ai != bi$。
- $richer$ 中的所有数对 互不相同。
- 对 $richer$ 的观察在逻辑上是一致的。

**示例**：

- 示例 1：

```python
输入：richer = [[1,0],[2,1],[3,1],[3,7],[4,3],[5,3],[6,3]], quiet = [3,2,5,4,6,1,7,0]
输出：[5,5,2,5,4,5,6,7]

解释：
answer[0] = 5，
person 5 比 person 3 有更多的钱，person 3 比 person 1 有更多的钱，person 1 比 person 0 有更多的钱。
唯一较为安静（有较低的安静值 quiet[x]）的人是 person 7，
但是目前还不清楚他是否比 person 0 更有钱。
answer[7] = 7，
在所有拥有的钱肯定不少于 person 7 的人中（这可能包括 person 3，4，5，6 以及 7），
最安静（有较低安静值 quiet[x]）的人是 person 7。
其他的答案也可以用类似的推理来解释。
```

## 解题思路

### 思路 1：拓扑排序

对于第 `i` 个人，我们要求解的是比第 `i` 个人更有钱或者和他一样有钱的人中，安静值最小的那个人的编号。 

我们可以建立一张有向无环图，由富人指向穷人。这样，对于任意一点来说（比如 `x`），通过有向边链接的点（比如 `y`），拥有的钱都没有 `x` 多。则我们可以根据 `answer[x]` 去更新所有 `x` 能连接到的点的 `answer` 值。

我们可以先将数组 `answer`  元素初始化为当前元素编号。然后对建立的有向无环图进行拓扑排序，按照拓扑排序的顺序去更新 `x` 能连接到的点的 `answer` 值。

### 思路 1：拓扑排序代码

```python
import collections

class Solution:
    def loudAndRich(self, richer: List[List[int]], quiet: List[int]) -> List[int]:
        
        size = len(quiet)
        indegrees = [0 for _ in range(size)]
        edges = collections.defaultdict(list)

        for x, y in richer:
            edges[x].append(y)
            indegrees[y] += 1

        res = [i for i in range(size)]
        queue = collections.deque([])
        for i in range(size):
            if not indegrees[i]:
                queue.append(i)

        while queue:
            x = queue.popleft()
            size -= 1
            for y in edges[x]:
                if quiet[res[x]] < quiet[res[y]]:
                    res[y] = res[x]
                indegrees[y] -= 1
                if not indegrees[y]:
                    queue.append(y)
        return res
```
# [0852. 山脉数组的峰顶索引](https://leetcode.cn/problems/peak-index-in-a-mountain-array/)

- 标签：数组、二分查找
- 难度：中等

## 题目链接

- [0852. 山脉数组的峰顶索引 - 力扣](https://leetcode.cn/problems/peak-index-in-a-mountain-array/)

## 题目大意

**描述**：给定由整数组成的山脉数组 $arr$。

**要求**：返回任何满足 $arr[0] < arr[1] < ... arr[i - 1] < arr[i] > arr[i + 1] > ... > arr[len(arr) - 1] $ 的下标 $i$。

**说明**：

- **山脉数组**：满足以下属性的数组：
  1. $len(arr) \ge 3$；
  2. 存在 $i$（$0 < i < len(arr) - 1$），使得：
     1. $arr[0] < arr[1] < ... arr[i-1] < arr[i]$；
     2. $arr[i] > arr[i+1] > ... > arr[len(arr) - 1]$。
- $3 <= arr.length <= 105$
- $0 <= arr[i] <= 106$
- 题目数据保证 $arr$ 是一个山脉数组

**示例**：

- 示例 1：

```python
输入：arr = [0,1,0]
输出：1
```

- 示例 2：

```python
输入：arr = [0,2,1,0]
输出：1
```

## 解题思路

### 思路 1：二分查找

1. 使用两个指针 $left$、$right$ 。$left$ 指向数组第一个元素，$right$ 指向数组最后一个元素。
2. 取区间中间节点 $mid$，并比较 $nums[mid]$ 和 $nums[mid + 1]$ 的值大小。
   1. 如果 $nums[mid]< nums[mid + 1]$，则右侧存在峰值，令 `left = mid + 1`。
   2. 如果 $nums[mid] \ge nums[mid + 1]$，则左侧存在峰值，令 `right = mid`。
3. 最后，当 $left == right$ 时，跳出循环，返回 $left$。

### 思路 1：代码

```python
class Solution:
    def peakIndexInMountainArray(self, arr: List[int]) -> int:
        left = 0
        right = len(arr) - 1
        while left < right:
            mid = left + (right - left) // 2
            if arr[mid] < arr[mid + 1]:
                left = mid + 1
            else:
                right = mid
        return left
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(\log n)$。
- **空间复杂度**：$O(1)$。

# [0860. 柠檬水找零](https://leetcode.cn/problems/lemonade-change/)

- 标签：贪心、数组
- 难度：简单

## 题目链接

- [0860. 柠檬水找零 - 力扣](https://leetcode.cn/problems/lemonade-change/)

## 题目大意

**描述**：一杯柠檬水的售价是 $5$ 美元。现在有 $n$ 个顾客排队购买柠檬水，每人只能购买一杯。顾客支付的钱面额有 $5$ 美元、$10$ 美元、$20$ 美元。必须给每个顾客正确找零（就是每位顾客需要向你支付 $5$ 美元，多出的钱要找还回顾客）。

现在给定 $n$ 个顾客支付的钱币面额数组 `bills`。

**要求**：如果能给每位顾客正确找零，则返回 `True`，否则返回 `False`。

**说明**：

- 一开始的时候手头没有任何零钱。
- $1 \le bills.length \le 10^5$。
- `bills[i]` 不是 $5$ 就是 $10$ 或是 $20$。

**示例**：

- 示例 1：

```python
输入：bills = [5,5,5,10,20]
输出：True
解释：
前 3 位顾客那里，我们按顺序收取 3 张 5 美元的钞票。
第 4 位顾客那里，我们收取一张 10 美元的钞票，并返还 5 美元。
第 5 位顾客那里，我们找还一张 10 美元的钞票和一张 5 美元的钞票。
由于所有客户都得到了正确的找零，所以我们输出 True。
```

- 示例 2：

```python
输入：bills = [5,5,10,10,20]
输出：False
解释：
前 2 位顾客那里，我们按顺序收取 2 张 5 美元的钞票。
对于接下来的 2 位顾客，我们收取一张 10 美元的钞票，然后返还 5 美元。
对于最后一位顾客，我们无法退回 15 美元，因为我们现在只有两张 10 美元的钞票。
由于不是每位顾客都得到了正确的找零，所以答案是 False。
```

## 解题思路

### 思路 1：贪心算法

由于顾客只能给我们 $5$、$10$、$20$ 三种面额的钞票，且一开始我们手头没有任何钞票，所以我们手中所能拥有的钞票面额只能是 $5$、$10$、$20$。因此可以采取下面的策略：

1. 如果顾客支付 $5$ 美元，直接收下。
2. 如果顾客支付 $10$ 美元，如果我们手头有 $5$ 美元面额的钞票，则找给顾客，否则无法正确找零，返回 `False`。
3. 如果顾客支付 $20$ 美元，如果我们手头有 $1$ 张 $10$ 美元和 $1$ 张 $5$ 美元的钞票，或者有 $3$ 张 $5$ 美元的钞票，则可以找给顾客。如果两种组合方式同时存在，倾向于第 $1$ 种方式找零，因为使用 $5$ 美元的场景比使用 $10$ 美元的场景多，要尽可能的保留 $5$ 美元的钞票。如果这两种组合方式都不通知，则无法正确找零，返回 `False`。

所以，我们可以使用两个变量 `five` 和 `ten` 来维护手中 $5$ 美元、$10$ 美团的钞票数量， 然后遍历一遍根据上述条件分别判断即可。

### 思路 1：代码

```python
class Solution:
    def lemonadeChange(self, bills: List[int]) -> bool:
        five, ten, twenty = 0, 0, 0
        for bill in bills:
            if bill == 5:
                five += 1
            if bill == 10:
                if five <= 0:
                    return False
                ten += 1
                five -= 1
            if bill == 20:
                if five > 0 and ten > 0:
                    five -= 1
                    ten -= 1
                    twenty += 1
                elif five >= 3:
                    five -= 3
                    twenty += 1
                else:
                    return False

        return True
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(n)$，其中 $n$ 是数组 `bill` 的长度。
- **空间复杂度**：$O(1)$。

# [0861. 翻转矩阵后的得分](https://leetcode.cn/problems/score-after-flipping-matrix/)

- 标签：贪心、位运算、数组、矩阵
- 难度：中等

## 题目链接

- [0861. 翻转矩阵后的得分 - 力扣](https://leetcode.cn/problems/score-after-flipping-matrix/)

## 题目大意

**描述**：给定一个二维矩阵 `A`，其中每个元素的值为 `0` 或 `1`。

我们可以选择任一行或列，并转换该行或列中的每一个值：将所有 `0` 都更改为 `1`，将所有 `1` 都更改为 `0`。

在做出任意次数的移动后，将该矩阵的每一行都按照二进制数来解释，矩阵的得分就是这些数字的总和。

**要求**：返回尽可能高的分数。

**说明**：

- $1 \le A.length \le 20$。
- $1 \le A[0].length \le 20$。
- `A[i][j]` 值为 `0` 或 `1`。

**示例**：

- 示例 1：

```python
输入：[[0,0,1,1],[1,0,1,0],[1,1,0,0]]
输出：39
解释：
转换为  [[1,1,1,1],[1,0,0,1],[1,1,1,1]]
0b1111 + 0b1001 + 0b1111 = 15 + 9 + 15 = 39
```

## 解题思路

### 思路 1：贪心算法

对于一个二进制数来说，应该优先保证高位（靠前的列）尽可能的大，也就是保证高位尽可能值为 `1`。

- 我们先来看矩阵的第一列数，只要第一列的某一行为 `0`，则将这一行的值进行翻转。这样就保证了最高位一定为 `1`。
- 接下来，我们再来关注除了第一列的其他列，这里因为有最高位限制，所以我们不能随意再将某一行的值进行翻转，只能选择某一列进行翻转。
- 为了保证当前位上有尽可能多的 `1`。我们可以用两个变量 `one_cnt`、`zeo_cnt` 来记录当前列上 `1` 的个数和 `0` 的个数。如果 `0` 的个数多于 `1` 的个数，那么我们就将当前列进行翻转。从而保证当前位上有尽可能多的 `1`。
- 当所有列都遍历完成后，我们会得到加和最大的情况。

### 思路 1：贪心算法代码

```python
class Solution:
    def matrixScore(self, grid: List[List[int]]) -> int:
        zero_cnt, one_cnt = 0, 0
        res = 0
        rows, cols = len(grid), len(grid[0])

        for col in range(cols):
            for row in range(rows):
                if col == 0 and grid[row][col] == 0:
                    for j in range(cols):
                        grid[row][j] = 1 - grid[row][j]
                else:
                    if grid[row][col] == 1:
                        one_cnt += 1
                    else:
                        zero_cnt += 1
            if zero_cnt > one_cnt:
                for row in range(rows):
                    grid[row][col] = 1 - grid[row][col]

            for row in range(rows):
                if grid[row][col] == 1:
                    res += pow(2, cols - col - 1)
            zero_cnt = 0
            one_cnt = 0
        return res
```
# [0862. 和至少为 K 的最短子数组](https://leetcode.cn/problems/shortest-subarray-with-sum-at-least-k/)

- 标签：队列、数组、二分查找、前缀和、滑动窗口、单调队列、堆（优先队列）
- 难度：困难

## 题目链接

- [0862. 和至少为 K 的最短子数组 - 力扣](https://leetcode.cn/problems/shortest-subarray-with-sum-at-least-k/)

## 题目大意

**描述**：给定一个整数数组 $nums$ 和一个整数 $k$。

**要求**：找出 $nums$ 中和至少为 $k$ 的最短非空子数组，并返回该子数组的长度。如果不存在这样的子数组，返回 $-1$。

**说明**：

- **子数组**：数组中连续的一部分。
- $1 \le nums.length \le 10^5$。
- $-10^5 \le nums[i] \le 10^5$。
- $1 \le k \le 10^9$。

**示例**：

- 示例 1：

```python
输入：nums = [1], k = 1
输出：1
```

- 示例 2：

```python
输入：nums = [1,2], k = 4
输出：-1
```

## 解题思路

### 思路 1：前缀和 + 单调队列

题目要求得到满足和至少为 $k$ 的子数组的最短长度。

先来考虑暴力做法。如果使用两重循环分别遍历子数组的开始和结束位置，则可以直接求出所有满足条件的子数组，以及对应长度。但是这种做法的时间复杂度为 $O(n^2)$。我们需要对其进行优化。

#### 1. 前缀和优化

首先对于子数组和，我们可以使用「前缀和」的方式，方便快速的得到某个子数组的和。

对于区间 $[left, right]$，通过 $pre\underline{\hspace{0.5em}}sum[right + 1] - prefix\underline{\hspace{0.5em}}cnts[left]$  即可快速求解出区间 $[left, right]$ 的子数组和。

此时问题就转变为：是否能找到满足 $i > j$ 且 $pre\underline{\hspace{0.5em}}sum[i] - pre\underline{\hspace{0.5em}}sum[j] \ge k$ 两个条件的子数组 $[j, i)$？如果能找到，则找出 $i - j$ 差值最小的作为答案。

#### 2. 单调队列优化

对于区间 $[j, i)$ 来说，我们应该尽可能的减少不成立的区间枚举。

1. 对于某个区间 $[j, i)$ 来说，如果 $pre\underline{\hspace{0.5em}}sum[i] - pre\underline{\hspace{0.5em}}sum[j] \ge k$，那么大于 $i$ 的索引值就不用再进行枚举了，不可能比 $i - j$ 的差值更优了。此时我们应该尽可能的向右移动 $j$，从而使得 $i - j$ 更小。
2. 对于某个区间 $[j, i)$ 来说，如果 $pre\underline{\hspace{0.5em}}sum[j] \ge pre\underline{\hspace{0.5em}}sum[i]$，对于任何大于等于 $i$ 的索引值 $r$ 来说，$pre\underline{\hspace{0.5em}}sum[r] - pre\underline{\hspace{0.5em}}sum[i]$ 一定比 $pre\underline{\hspace{0.5em}}sum[i] - pre\underline{\hspace{0.5em}}sum[j]$ 更小且长度更小，此时 $pre\underline{\hspace{0.5em}}sum[j]$ 可以直接忽略掉。

因此，我们可以使用单调队列来维护单调递增的前缀数组 $pre\underline{\hspace{0.5em}}sum$。其中存放了下标 $x:x_0, x_1, …$，满足 $pre\underline{\hspace{0.5em}}sum[x_0] < pre\underline{\hspace{0.5em}}sum[x_1] < …$ 单调递增。

1. 使用一重循环遍历位置 $i$，将当前位置 $i$ 存入倒掉队列中。
2. 对于每一个位置 $i$，如果单调队列不为空，则可以判断其之前存入在单调队列中的 $pre\underline{\hspace{0.5em}}sum[j]$ 值，如果 $pre\underline{\hspace{0.5em}}sum[i] - pre\underline{\hspace{0.5em}}sum[j] \ge k$，则更新答案，并将 $j$ 从队头位置弹出。直到不再满足 $pre\underline{\hspace{0.5em}}sum[i] - pre\underline{\hspace{0.5em}}sum[j] \ge k$ 时为止（即 $pre\underline{\hspace{0.5em}}sum[i] - pre\underline{\hspace{0.5em}}sum[j] < k$）。
3. 如果队尾 $pre\underline{\hspace{0.5em}}sum[j] \ge pre\underline{\hspace{0.5em}}sum[i]$，那么说明以后无论如何都不会再考虑 $pre\underline{\hspace{0.5em}}sum[j]$ 了，则将其从队尾弹出。
4. 最后遍历完返回答案。

### 思路 1：代码

```Python
class Solution:
    def shortestSubarray(self, nums: List[int], k: int) -> int:
        size = len(nums)
        
        # 优化 1
        pre_sum = [0 for _ in range(size + 1)]
        for i in range(size):
            pre_sum[i + 1] = pre_sum[i] + nums[i]

        ans = float('inf')
        queue = collections.deque()

        for i in range(size + 1):            
          	# 优化 2
            while queue and pre_sum[i] - pre_sum[queue[0]] >= k:
                ans = min(ans, i - queue.popleft())
            while queue and pre_sum[queue[-1]] >= pre_sum[i]:
                queue.pop()
            queue.append(i)

        if ans == float('inf'):
            return -1
        return ans
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(n)$，其中 $n$ 为数组 $nums$ 的长度。
- **空间复杂度**：$O(n)$。

## 参考资料

- 【题解】[862. 和至少为 K 的最短子数组 - 力扣](https://leetcode.cn/problems/shortest-subarray-with-sum-at-least-k/solutions/1925036/liang-zhang-tu-miao-dong-dan-diao-dui-li-9fvh/)
- 【题解】[Leetcode 862：和至少为 K 的最短子数组 - 掘金](https://juejin.cn/post/7076316608460750856)
- 【题解】[LeetCode 862. 和至少为 K 的最短子数组 - AcWing](https://www.acwing.com/solution/leetcode/content/612/)
- 【题解】[0862. Shortest Subarray With Sum at Least K | LeetCode Cookbook](https://books.halfrost.com/leetcode/ChapterFour/0800~0899/0862.Shortest-Subarray-with-Sum-at-Least-K/)
# [0867. 转置矩阵](https://leetcode.cn/problems/transpose-matrix/)

- 标签：数组、矩阵、模拟
- 难度：简单

## 题目链接

- [0867. 转置矩阵 - 力扣](https://leetcode.cn/problems/transpose-matrix/)

## 题目大意

给定一个二维数组 matrix。返回 matrix 的转置矩阵。

## 解题思路

直接模拟求解即可。先求出 matrix 的规模。若 matrix 是 m * n 的矩阵。则创建一个 n * m 大小的矩阵 transposed。根据转置的规则对 transposed 的每个元素进行赋值。最终返回 transposed。

## 代码

```python
class Solution:
    def transpose(self, matrix: List[List[int]]) -> List[List[int]]:
        m = len(matrix)
        n = len(matrix[0])
        transposed = [[0 for _ in range(m)] for _ in range(n)]
        for i in range(m):
            for j in range(n):
                transposed[j][i] = matrix[i][j]
        return transposed
```

# [0868. 二进制间距](https://leetcode.cn/problems/binary-gap/)

- 标签：位运算
- 难度：简单

## 题目链接

- [0868. 二进制间距 - 力扣](https://leetcode.cn/problems/binary-gap/)

## 题目大意

**描述**：给定一个正整数 $n$。

**要求**：找到并返回 $n$ 的二进制表示中两个相邻 $1$ 之间的最长距离。如果不存在两个相邻的 $1$，返回 $0$。

**说明**：

- $1 \le n \le 10^9$。

**示例**：

- 示例 1：

```python
输入：n = 22
输出：2
解释：22 的二进制是 "10110"。
在 22 的二进制表示中，有三个 1，组成两对相邻的 1。
第一对相邻的 1 中，两个 1 之间的距离为 2。
第二对相邻的 1 中，两个 1 之间的距离为 1。
答案取两个距离之中最大的，也就是 2。
```

- 示例 2：

```python
输入：n = 8
输出：0
解释：8 的二进制是 "1000"。
在 8 的二进制表示中没有相邻的两个 1，所以返回 0。
```

## 解题思路

### 思路 1：遍历

1. 将正整数 $n$ 转为二进制字符串形式 $bin\underline{\hspace{0.5em}}n$。
2. 使用变量 $pre$ 记录二进制字符串中上一个 $1$ 的位置，使用变量 $ans$ 存储两个相邻 $1$ 之间的最长距离。
3. 遍历二进制字符串形式 $bin\underline{\hspace{0.5em}}n$ 的每一位，遇到 $1$ 时判断并更新两个相邻 $1$ 之间的最长距离。
4. 遍历完返回两个相邻 $1$ 之间的最长距离，即 $ans$。

### 思路 1：代码

```Python
class Solution:
    def binaryGap(self, n: int) -> int:
        bin_n = bin(n)
        pre, ans = 2, 0
        
        for i in range(2, len(bin_n)):
            if bin_n[i] == '1':
                ans = max(ans, i - pre)
                pre = i
            
        return ans
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(\log n)$。
- **空间复杂度**：$O(1)$。

# [0872. 叶子相似的树](https://leetcode.cn/problems/leaf-similar-trees/)

- 标签：树、深度优先搜索、二叉树
- 难度：简单

## 题目链接

- [0872. 叶子相似的树 - 力扣](https://leetcode.cn/problems/leaf-similar-trees/)

## 题目大意

将一棵二叉树树上所有的叶子，按照从左到右的顺序排列起来就形成了一个「叶值序列」。如果两棵二叉树的叶值序列是相同的，我们就认为它们是叶相似的。

现在给定两棵二叉树的根节点 `root1`、`root2`。如果两棵二叉是叶相似的，则返回 `True`，否则返回 `False`。

## 解题思路

分别 DFS 遍历两棵树，得到对应的叶值序列，判断两个叶值序列是否相等。

## 代码

```python
class Solution:
    def leafSimilar(self, root1: TreeNode, root2: TreeNode) -> bool:
        def dfs(node: TreeNode, res: List[int]):
            if not node:
                return
            if not node.left and not node.right:
                res.append(node.val)
            dfs(node.left, res)
            dfs(node.right, res)

        res1 = []
        dfs(root1, res1)
        res2 = []
        dfs(root2, res2)
        return res1 == res2
```

# [0873. 最长的斐波那契子序列的长度](https://leetcode.cn/problems/length-of-longest-fibonacci-subsequence/)

- 标签：数组、哈希表、动态规划
- 难度：中等

## 题目链接

- [0873. 最长的斐波那契子序列的长度 - 力扣](https://leetcode.cn/problems/length-of-longest-fibonacci-subsequence/)

## 题目大意

**描述**：给定一个严格递增的正整数数组 $arr$。

**要求**：从数组 $arr$ 中找出最长的斐波那契式的子序列的长度。如果不存斐波那契式的子序列，则返回 0。

**说明**：

- **斐波那契式序列**：如果序列 $X_1, X_2, ..., X_n$ 满足：

  - $n \ge 3$；
  - 对于所有 $i + 2 \le n$，都有 $X_i + X_{i+1} = X_{i+2}$。

  则称该序列为斐波那契式序列。

- **斐波那契式子序列**：从序列 $A$ 中挑选若干元素组成子序列，并且子序列满足斐波那契式序列，则称该序列为斐波那契式子序列。例如：$A = [3, 4, 5, 6, 7, 8]$。则 $[3, 5, 8]$ 是 $A$ 的一个斐波那契式子序列。

- $3 \le arr.length \le 1000$。

- $1 \le arr[i] < arr[i + 1] \le 10^9$。

**示例**：

- 示例 1：

```python
输入: arr = [1,2,3,4,5,6,7,8]
输出: 5
解释: 最长的斐波那契式子序列为 [1,2,3,5,8]。
```

- 示例 2：

```python
输入: arr = [1,3,7,11,12,14,18]
输出: 3
解释: 最长的斐波那契式子序列有 [1,11,12]、[3,11,14] 以及 [7,11,18]。
```

## 解题思路

### 思路 1： 暴力枚举（超时）

假设 $arr[i]$、$arr[j]$、$arr[k]$ 是序列 $arr$ 中的 $3$ 个元素，且满足关系：$arr[i] + arr[j] == arr[k]$，则 $arr[i]$、$arr[j]$、$arr[k]$ 就构成了 $arr$ 的一个斐波那契式子序列。

通过  $arr[i]$、$arr[j]$，我们可以确定下一个斐波那契式子序列元素的值为 $arr[i] + arr[j]$。

因为给定的数组是严格递增的，所以对于一个斐波那契式子序列，如果确定了 $arr[i]$、$arr[j]$，则可以顺着 $arr$ 序列，从第 $j + 1$ 的元素开始，查找值为 $arr[i] + arr[j]$ 的元素 。找到 $arr[i] + arr[j]$ 之后，然后再顺着查找子序列的下一个元素。

简单来说，就是确定了 $arr[i]$、$arr[j]$，就能尽可能的得到一个长的斐波那契式子序列，此时我们记录下子序列长度。然后对于不同的  $arr[i]$、$arr[j]$，统计不同的斐波那契式子序列的长度。

最后将这些长度进行比较，其中最长的长度就是答案。

### 思路 1：代码

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

### 思路 1：复杂度分析

- **时间复杂度**：$O(n^3)$，其中 $n$ 为数组 $arr$ 的元素个数。
- **空间复杂度**：$O(1)$。

### 思路 2：哈希表

对于 $arr[i]$、$arr[j]$，要查找的元素 $arr[i] + arr[j]$ 是否在 $arr$ 中，我们可以预先建立一个反向的哈希表。键值对关系为 $value : idx$，这样就能在 $O(1)$ 的时间复杂度通过 $arr[i] + arr[j]$ 的值查找到对应的 $arr[k]$，而不用像原先一样线性查找 $arr[k]$ 了。

### 思路 2：代码

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

### 思路 2：复杂度分析

- **时间复杂度**：$O(n^2)$，其中 $n$ 为数组 $arr$ 的元素个数。
- **空间复杂度**：$O(n)$。

### 思路 3：动态规划 + 哈希表

###### 1. 划分阶段

按照斐波那契式子序列相邻两项的结尾位置进行阶段划分。

###### 2. 定义状态

定义状态 $dp[i][j]$ 表示为：以 $arr[i]$、$arr[j]$ 为结尾的斐波那契式子序列的最大长度。

###### 3. 状态转移方程

以 $arr[j]$、$arr[k]$ 结尾的斐波那契式子序列的最大长度 = 满足 $arr[i] + arr[j] = arr[k]$ 条件下，以 $arr[i]$、$arr[j]$ 结尾的斐波那契式子序列的最大长度加 $1$。即状态转移方程为：$dp[j][k] = max_{(A[i] + A[j] = A[k], i < j < k)}(dp[i][j] + 1)$。

###### 4. 初始条件

默认状态下，数组中任意相邻两项元素都可以作为长度为 $2$ 的斐波那契式子序列，即 $dp[i][j] = 2$。

###### 5. 最终结果

根据我们之前定义的状态，$dp[i][j]$ 表示为：以 $arr[i]$、$arr[j]$ 为结尾的斐波那契式子序列的最大长度。那为了计算出最大的最长递增子序列长度，则需要在进行状态转移时，求出最大值 $ans$ 即为最终结果。

因为题目定义中，斐波那契式中 $n \ge 3$，所以只有当 $ans \ge 3$ 时，返回 $ans$。如果 $ans < 3$，则返回 $0$。

> **注意**：在进行状态转移的同时，我们应和「思路 2：哈希表」一样采用哈希表优化的方式来提高效率，降低算法的时间复杂度。

### 思路 3：代码

```python
class Solution:
    def lenLongestFibSubseq(self, arr: List[int]) -> int:
        size = len(arr)
        
        dp = [[0 for _ in range(size)] for _ in range(size)]
        ans = 0

        # 初始化 dp
        for i in range(size):
            for j in range(i + 1, size):
                dp[i][j] = 2

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

        if ans >= 3:
            return ans
        return 0
```

### 思路 3：复杂度分析

- **时间复杂度**：$O(n^2)$，其中 $n$ 为数组 $arr$ 的元素个数。
- **空间复杂度**：$O(n)$。

# [0875. 爱吃香蕉的珂珂](https://leetcode.cn/problems/koko-eating-bananas/)

- 标签：数组、二分查找
- 难度：中等

## 题目链接

- [0875. 爱吃香蕉的珂珂 - 力扣](https://leetcode.cn/problems/koko-eating-bananas/)

## 题目大意

**描述**：给定一个数组 $piles$ 代表 $n$ 堆香蕉。其中 $piles[i]$ 表示第 $i$ 堆香蕉的个数。再给定一个整数 $h$ ，表示最多可以在 $h$ 小时内吃完所有香蕉。珂珂决定以速度每小时 $k$（未知）根的速度吃香蕉。每一个小时，她讲选择其中一堆香蕉，从中吃掉 $k$ 根。如果这堆香蕉少于 $k$ 根，珂珂将在这一小时吃掉这堆的所有香蕉，并且这一小时不会再吃其他堆的香蕉。  

**要求**：返回珂珂可以在 $h$ 小时内吃掉所有香蕉的最小速度 $k$（$k$ 为整数）。

**说明**：

- $1 \le piles.length \le 10^4$。
- $piles.length \le h \le 10^9$。
- $1 \le piles[i] \le 10^9$。

**示例**：

- 示例 1：

```python
输入：piles = [3,6,7,11], h = 8
输出：4
```

- 示例 2：

```python
输入：piles = [30,11,23,4,20], h = 5
输出：30
```

## 解题思路

### 思路 1：二分查找算法

先来看 $k$ 的取值范围，因为 $k$ 是整数，且速度肯定不能为 $0$ 吧，为 $0$ 的话就永远吃不完了。所以$k$ 的最小值可以取 $1$。$k$ 的最大值根香蕉中最大堆的香蕉个数有关，因为 $1$ 个小时内只能选择一堆吃，不能再吃其他堆的香蕉，则 $k$ 的最大值取香蕉堆的最大值即可。即 $k$ 的最大值为 $max(piles)$。

我们的目标是求出 $h$ 小时内吃掉所有香蕉的最小速度 $k$。现在有了区间「$[1, max(piles)]$」，有了目标「最小速度 $k$」。接下来使用二分查找算法来查找「最小速度 $k$」。至于计算 $h$ 小时内能否以 $k$ 的速度吃完香蕉，我们可以再写一个方法 $canEat$ 用于判断。如果能吃完就返回 $True$，不能吃完则返回 $False$。下面说一下算法的具体步骤。

- 使用两个指针 $left$、$right$。令 $left$ 指向 $1$，$right$ 指向 $max(piles)$。代表待查找区间为 $[left, right]$

- 取两个节点中心位置 $mid$，判断是否能在 $h$ 小时内以 $k$ 的速度吃完香蕉。
  - 如果不能吃完，则将区间 $[left, mid]$ 排除掉，继续在区间 $[mid + 1, right]$ 中查找。
  - 如果能吃完，说明 $k$ 还可以继续减小，则继续在区间 $[left, mid]$ 中查找。
- 当 $left == right$ 时跳出循环，返回 $left$。

### 思路 1：代码

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

### 思路 1：复杂度分析

- **时间复杂度**：$O(n \times \log max(piles))$，$n$ 表示数组 $piles$ 中的元素个数。
- **空间复杂度**：$O(1)$。

# [0876. 链表的中间结点](https://leetcode.cn/problems/middle-of-the-linked-list/)

- 标签：链表、双指针
- 难度：简单

## 题目链接

- [0876. 链表的中间结点 - 力扣](https://leetcode.cn/problems/middle-of-the-linked-list/)

## 题目大意

**描述**：给定一个单链表的头节点 `head`。

**要求**：返回链表的中间节点。如果有两个中间节点，则返回第二个中间节点。

**说明**：

- 给定链表的结点数介于 `1` 和 `100` 之间。

**示例**：

- 示例 1：

```python
输入：[1,2,3,4,5]
输出：此列表中的结点 3 (序列化形式：[3,4,5])
解释：返回的结点值为 3 。
注意，我们返回了一个 ListNode 类型的对象 ans，这样：
ans.val = 3, ans.next.val = 4, ans.next.next.val = 5, 以及 ans.next.next.next = NULL.
```

- 示例 2：

```python
输入：[1,2,3,4,5,6]
输出：此列表中的结点 4 (序列化形式：[4,5,6])
解释：由于该列表有两个中间结点，值分别为 3 和 4，我们返回第二个结点。
```

## 解题思路

### 思路 1：单指针

先遍历一遍链表，统计一下节点个数为 `n`，再遍历到 `n / 2` 的位置，返回中间节点。

### 思路 1：代码

```python
class Solution:
    def middleNode(self, head: ListNode) -> ListNode:
        n = 0
        curr = head
        while curr:
            n += 1
            curr = curr.next
        k = 0
        curr = head
        while k < n // 2:
            k += 1
            curr = curr.next
        return curr
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(n)$。
- **空间复杂度**：$O(1)$。

### 思路 2：快慢指针

使用步长不一致的快慢指针进行一次遍历找到链表的中间节点。具体做法如下：

1. 使用两个指针 `slow`、`fast`。`slow`、`fast` 都指向链表的头节点。
2. 在循环体中将快、慢指针同时向右移动。其中慢指针每次移动 `1` 步，即 `slow = slow.next`。快指针每次移动 `2` 步，即 `fast = fast.next.next`。
3. 等到快指针移动到链表尾部（即 `fast == Node`）时跳出循环体，此时 `slow` 指向链表中间位置。
4. 返回 `slow` 指针。

### 思路 2：代码

```python
class Solution:
    def middleNode(self, head: ListNode) -> ListNode:
        fast = head
        slow = head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
        return slow
```

### 思路 2：复杂度分析

- **时间复杂度**：$O(n)$。
- **空间复杂度**：$O(1)$。# [0877. 石子游戏](https://leetcode.cn/problems/stone-game/)

- 标签：数组、数学、动态规划、博弈
- 难度：中等

## 题目链接

- [0877. 石子游戏 - 力扣](https://leetcode.cn/problems/stone-game/)

## 题目大意

亚历克斯和李在玩石子游戏。总共有偶数堆石子，每堆都有正整数颗石子 `piles[i]`，总共的石子数为奇数 。每回合，玩家从开始位置或者结束位置取走一整堆石子。直到没有石子堆为止结束游戏，最终手中石子颗数多的玩家获胜。假设亚历克斯和李每回合都能发挥出最佳水平，并且亚历克斯先开始。

给定代表每个位置石子颗数的数组 `piles`。

要求：判断亚历克斯是否能赢得比赛。如果亚历克斯赢得比赛，则返回 `True`。如果李赢得比赛返回 `False`。

## 解题思路

能取的次数是偶数个，总数是奇数个。

- 如果亚历克斯开始取了开始偶数位置 `0`，那么李只能取奇数位置 `1` 或者末尾位置 `len(piles) - 1`。然后亚历克斯可以j接着取偶数位。
- 或者亚历克斯开始取了最后奇数位置 `len(piles) - 1`，那么李只能取偶数位置 `0` 或 `len(piles) - 2`。然后亚历克斯可以接着取奇数位。
- 这样亚历克斯只要一开始计算好奇数位置上的石子总数多，还是偶数位置上的石子总数多，然后就可以选择一开始取奇数位置还是偶数位置。所以最后肯定会赢
- 游戏一开始，其实就没李啥事了。。。

## 代码

```python
class Solution:
    def stoneGame(self, piles: List[int]) -> bool:
        return True
```

# [0881. 救生艇](https://leetcode.cn/problems/boats-to-save-people/)

- 标签：贪心、数组、双指针、排序
- 难度：中等

## 题目链接

- [0881. 救生艇 - 力扣](https://leetcode.cn/problems/boats-to-save-people/)

## 题目大意

**描述**：给定一个整数数组 `people` 代表每个人的体重，其中第 `i` 个人的体重为 `people[i]`。再给定一个整数 `limit`，代表每艘船可以承载的最大重量。每艘船最多可同时载两人，但条件是这些人的重量之和最多为 `limit`。

**要求**：返回载到每一个人所需的最小船数（保证每个人都能被船载）。

**说明**：

- $1 \le people.length \le 5 \times 10^4$。
- $1 \le people[i] \le limit \le 3 \times 10^4$。

**示例**：

- 示例 1：

```python
输入：people = [1,2], limit = 3
输出：1
解释：1 艘船载 (1, 2)
```

- 示例 2：

```python
输入：people = [3,2,2,1], limit = 3
输出：3
解释：3 艘船分别载 (1, 2), (2) 和 (3)
```

## 解题思路

### 思路 1：贪心算法 + 双指针

暴力枚举的时间复杂度为 $O(n^2)$。使用双指针可以减少循环内的时间复杂度。

我们可以利用贪心算法的思想，让最重的和最轻的人一起走。这样一只船就可以尽可能的带上两个人。

具体做法如下：

1. 先对数组进行升序排序，使用 `ans` 记录所需最小船数。
2. 使用两个指针 `left`、`right`。`left` 指向数组开始位置，`right` 指向数组结束位置。
3. 判断 `people[left]` 和 `people[right]` 加一起是否超重。
   1. 如果 `people[left] + people[right] > limit`，则让重的人上船，船数量 + 1，令 `right` 左移，继续判断。
   2. 如果 `people[left] + people[right] <= limit`，则两个人都上船，船数量 + 1，并令 `left` 右移，`right` 左移，继续判断。
4. 如果 `lefft == right`，则让最后一个人上船，船数量 + 1。并返回答案。

### 思路 1：代码

```python
class Solution:
    def numRescueBoats(self, people: List[int], limit: int) -> int:
        people.sort()
        size = len(people)
        left, right = 0, size - 1
        ans = 0
        while left < right:
            if people[left] + people[right] > limit:
                right -= 1
            else:
                left += 1
                right -= 1
            ans += 1
        if left == right:
            ans += 1
        return ans
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(n \times \log n)$，其中 $n$ 是数组 `people` 的长度。
- **空间复杂度**：$O(\log n)$。

# [0884. 两句话中的不常见单词](https://leetcode.cn/problems/uncommon-words-from-two-sentences/)

- 标签：哈希表、字符串
- 难度：简单

## 题目链接

- [0884. 两句话中的不常见单词 - 力扣](https://leetcode.cn/problems/uncommon-words-from-two-sentences/)

## 题目大意

**描述**：给定两个字符串 $s1$ 和 $s2$ ，分别表示两个句子。

**要求**：返回所有不常用单词的列表。返回列表中单词可以按任意顺序组织。

**说明**：

- **句子**：是一串由空格分隔的单词。
- **单词**：仅由小写字母组成的子字符串。
- **不常见单词**：如果某个单词在其中一个句子中恰好出现一次，在另一个句子中却没有出现，那么这个单词就是不常见的。
- $1 \le s1.length, s2.length \le 200$。
- $s1$ 和 $s2$ 由小写英文字母和空格组成。
- $s1$ 和 $s2$ 都不含前导或尾随空格。
- $s1$ 和 $s2$ 中的所有单词间均由单个空格分隔。

**示例**：

- 示例 1：

```python
输入：s1 = "this apple is sweet", s2 = "this apple is sour"
输出：["sweet","sour"]
```

- 示例 2：

```python
输入：s1 = "apple apple", s2 = "banana"
输出：["banana"]
```

## 解题思路

### 思路 1：哈希表

题目要求找出在其中一个句子中恰好出现一次，在另一个句子中却没有出现的单词，其实就是找出在两个句子中只出现过一次的单词，我们可以用哈希表统计两个句子中每个单词的出现频次，然后将出现频次为 $1$ 的单词就是不常见单词，将其加入答案数组即可。

具体步骤如下：

1.  遍历字符串 $s1$、$s2$，使用哈希表 $table$ 统计字符串 $s1$、$s2$ 各个单词的出现频次。
2. 遍历哈希表，找出出现频次为 $1$ 的单词，将其加入答案数组 $res$ 中。
3. 遍历完返回答案数组 $res$。

### 思路 1：代码

```python
class Solution:
    def uncommonFromSentences(self, s1: str, s2: str) -> List[str]:
        table = dict()
        for word in s1.split(' '):
            if word not in table:
                table[word] = 1
            else:
                table[word] += 1
        
        for word in s2.split(' '):
            if word not in table:
                table[word] = 1
            else:
                table[word] += 1
       
        res = []
        for word in table:
            if table[word] == 1:
                res.append(word)
        
        return res
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(m + n)$，其中 $m$、$n$ 分别为字符串 $s1$、$s2$ 的长度。
- **空间复杂度**：$O(m + n)$。
# [0886. 可能的二分法](https://leetcode.cn/problems/possible-bipartition/)

- 标签：深度优先搜索、广度优先搜索、并查集、图
- 难度：中等

## 题目链接

- [0886. 可能的二分法 - 力扣](https://leetcode.cn/problems/possible-bipartition/)

## 题目大意

把 n 个人（编号为 1, 2, ... , n）分为任意大小的两组。每个人都可能不喜欢其他人，那么他们不应该属于同一组。

给定表示不喜欢关系的数组 `dislikes`，其中 `dislikes[i] = [a, b]` 表示 `a` 和 `b` 互相不喜欢，不允许将编号 `a` 和 `b` 的人归入同一组。

要求：如果可以以这种方式将所有人分为两组，则返回 `True`；如果不能则返回 `False`。

## 解题思路

先构建图，对于 `dislikes[i] = [a, b]`，在节点 `a` 和 `b` 之间建立一条无向边，然后判断该图是否为二分图。具体做法如下：

- 找到一个没有染色的节点 `u`，将其染成红色。
- 然后遍历该节点直接相连的节点 `v`，如果该节点没有被染色，则将该节点直接相连的节点染成蓝色，表示两个节点不是同一集合。如果该节点已经被染色并且颜色跟 `u` 一样，则说明该图不是二分图，直接返回 `False`。
- 从上面染成蓝色的节点 `v` 出发，遍历该节点直接相连的节点。。。依次类推的递归下去。
- 如果所有节点都顺利染上色，则说明该图为二分图，可以将所有人分为两组，返回 `True`。否则，如果在途中不能顺利染色，不能将所有人分为两组，则返回 `False`。

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

    def possibleBipartition(self, n: int, dislikes: List[List[int]]) -> bool:
        graph = [[] for _ in range(n + 1)]
        colors = [0 for _ in range(n + 1)]

        for x, y in dislikes:
            graph[x].append(y)
            graph[y].append(x)

        for i in range(1, n + 1):
            if colors[i] == 0 and not self.dfs(graph, colors, i, 1):
                return False
        return True
```

# [0887. 鸡蛋掉落](https://leetcode.cn/problems/super-egg-drop/)

- 标签：数学、二分查找、动态规划
- 难度：困难

## 题目链接

- [0887. 鸡蛋掉落 - 力扣](https://leetcode.cn/problems/super-egg-drop/)

## 题目大意

**描述**：给定一个整数 `k` 和整数 `n`，分别代表 `k` 枚鸡蛋和可以使用的一栋从第 `1` 层到第 `n` 层楼的建筑。

已知存在楼层 `f`，满足 `0 <= f <= n`，任何从高于 `f` 的楼层落下的鸡蛋都会碎，从 `f` 楼层或比它低的楼层落下的鸡蛋都不会碎。

每次操作，你可以取一枚没有碎的鸡蛋并把它从任一楼层 `x` 扔下（满足 `1 <= x <= n`），如果鸡蛋碎了，就不能再次使用它。如果鸡蛋没碎，则可以再次使用。

**要求**：计算并返回要确定 `f` 确切值的最小操作次数是多少。

**说明**：

- $1 \le k \le 100$。
- $1 \le n \le 10^4$。

**示例**：

- 示例 1：

```python
输入：k = 1, n = 2
输入：2
解释：鸡蛋从 1 楼掉落。如果它碎了，肯定能得出 f = 0。否则，鸡蛋从 2 楼掉落。如果它碎了，肯定能得出 f = 1。如果它没碎，那么肯定能得出 f = 2。因此，在最坏的情况下我们需要移动 2 次以确定 f 是多少。
```

## 解题思路

这道题目的题意不是很容易理解，我们先把题目简化一下，忽略一些限制条件，理解简单情况下的题意。然后再一步步增加限制条件，从而弄明白这道题目的意思，以及思考清楚这道题的解题思路。

我们先忽略 `k` 个鸡蛋这个条件，假设有无限个鸡蛋。

现在有 `1` ~ `n` 一共 `n` 层楼。已知存在楼层 `f`，低于等于 `f` 层的楼层扔下去的鸡蛋都不会碎，高于 `f` 的楼层扔下去的鸡蛋都会碎。

当然这个楼层 `f` 的确切值题目没有给出，需要我们一次次去测试鸡蛋最高会在哪一层不会摔碎。

在每次操作中，我们可以选定一个楼层，将鸡蛋扔下去：

- 如果鸡蛋没摔碎，则可以继续选择其他楼层进行测试。
- 如果鸡蛋摔碎了，则该鸡蛋无法继续测试。

现在题目要求：**已知有 `n` 层楼，无限个鸡蛋，求出至少需要扔几次鸡蛋，才能保证无论 `f` 是多少层，都能将 `f` 找出来？**

最简单且直观的想法：

1. 从第 `1` 楼开始扔鸡蛋。`1` 楼不碎，再去 `2` 楼扔。
2. `2` 楼还不碎，就去 `3` 楼扔。
3. …… 
4. 直到鸡蛋碎了，也就找到了鸡蛋不会摔碎的最高层 `f`。

用这种方法，最坏情况下，鸡蛋在第 `n` 层也没摔碎。这种情况下我们总共试了 `n` 次才确定鸡蛋不会摔碎的最高楼层 `f`。

下面再来说一下比 `n` 次要少的情况。

如果我们可以通过二分查找的方法，先从 `1` ~ `n` 层的中间层开始扔鸡蛋。

- 如果鸡蛋碎了，则从第 `1` 层到中间层这个区间中去扔鸡蛋。
- 如果鸡蛋没碎，则从中间层到第 `n` 层这个区间中去扔鸡蛋。

每次扔鸡蛋都从区间的中间层去扔，这样每次都能排除当前区间一半的答案，从而最终确定鸡蛋不会摔碎的最高楼层 `f`。

通过这种二分查找的方法，可以优化到 $\log n$ 次就能确定鸡蛋不会摔碎的最高楼层 `f`。

因为 $\log n \le n$，所以通过二分查找的方式，「至少」比线性查找的次数要少。

同样，我们还可以通过三分查找、五分查找等等方式减少次数。

这是在不限制鸡蛋个数的情况下，现在我们来限制一下鸡蛋个数为 `k`。

现在题目要求：**已知有 `n` 层楼，`k` 个鸡蛋，求出至少需要扔几次鸡蛋，才能保证无论 `f` 是多少层，都能将 `f` 找出来？**

如果鸡蛋足够多（大于等于 $\log_2 n$ 个），可以通过二分查找的方法来测试。如果鸡蛋不够多，可能二分查找过程中，鸡蛋就用没了，则不能通过二分查找的方法来测试。

那么这时候为了找出 `f` ，我们应该如何求出最少的扔鸡蛋次数？

### 思路 1：动态规划（超时）

可以这样考虑。题目限定了 `n` 层楼，`k` 个鸡蛋。

如果我们尝试在 `1` ~ `n` 层中的任意一层 `x` 扔鸡蛋：

1. 如果鸡蛋没碎，则说明 `1` ~ `x` 层都不用再考虑了，我们需要用 `k` 个鸡蛋去考虑剩下的 `n - x` 层，问题就从 `(n, k)` 转变为了 `(n - x, k)`。
2. 如果鸡蛋碎了，则说明 `x + 1` ~ `n` 层都不用再考虑了，我们需要去剩下的 `k - 1` 个鸡蛋考虑剩下的 `x - 1` 层，问题就从 `(n, k)` 转变为了 `(x - 1, k - 1)`。

这样一来，我们就可以根据上述关系使用动态规划方法来解决这道题目了。具体步骤如下：

###### 1. 划分阶段

按照楼层数量、剩余鸡蛋个数进行阶段划分。

###### 2. 定义状态

定义状态 `dp[i][j]` 表示为：一共有 `i` 层楼，`j` 个鸡蛋的条件下，为了找出 `f` ，最坏情况下的最少扔鸡蛋次数。

###### 3. 状态转移方程

根据之前的描述，`dp[i][j]` 有两个来源，其状态转移方程为：

$dp[i][j] = min_{1 \le x \le n} (max(dp[i - x][j], dp[x - 1][j - 1])) + 1$ 

###### 4. 初始条件

给定鸡蛋 `k` 的取值范围为 `[1, 100]`，`f` 值取值范围为 `[0, n]`，初始化时，可以考虑将所有值设置为当前拥有的楼层数。

- 当鸡蛋数为 `1` 时，`dp[i][1] = i`。这是如果唯一的蛋碎了，则无法测试了。只能从低到高，一步步进行测试，最终最少测试数为当前拥有的楼层数（如果刚开始初始化时已经将所有值设置为当前拥有的楼层数，其实这一步可省略）。
- 当楼层为 `1` 时，在 `1` 层扔鸡蛋，`dp[1][j] = 1`。这是因为：
  - 如果在 `1` 层扔鸡蛋碎了，则 `f < 1`。同时因为 `f` 的取值范围为 `[0, n]`。所以能确定 `f = 0`。
  - 如果在 `1` 层扔鸡蛋没碎，则 `f >= 1`。同时因为 `f` 的取值范围为 `[0, n]`。所以能确定 `f = 0`。 

###### 5. 最终结果

根据我们之前定义的状态，`dp[i][j]` 表示为：一共有 `i` 层楼，`j` 个鸡蛋的条件下，为了找出 `f` ，最坏情况下的最少扔鸡蛋次数。则最终结果为 `dp[n][k]`。

### 思路 1：代码

```python
class Solution:
    def superEggDrop(self, k: int, n: int) -> int:
        dp = [[0 for _ in range(k + 1)] for i in range(n + 1)]
        
        for i in range(1, n + 1):
            for j in range(1, k + 1):
                dp[i][j] = i

        # for i in range(1, n + 1):
        #     dp[i][1] = i

        for j in range(1, k + 1):
            dp[1][j] = 1

        for i in range(2, n + 1):
            for j in range(2, k + 1):
                for x in range(1, i + 1):
                    dp[i][j] = min(dp[i][j], max(dp[i - x][j], dp[x - 1][j - 1]) + 1)
                
        return dp[n][k]
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(n^2 \times k)$。三重循环的时间复杂度为 $O(n^2 \times k)$。
- **空间复杂度**：$O(n \times k)$。

### 思路 2：动态规划优化

上一步中时间复杂度为 $O(n^2 \times k)$。根据 $n$ 的规模，提交上去不出意外的超时了。

我们可以观察一下上面的状态转移方程：$dp[i][j] = min_{1 \le x \le n} (max(dp[i - x][j], dp[x - 1][j - 1])) + 1$ 。

这里最外两层循环的 `i`、`j` 分别为状态的阶段，可以先将 `i`、`j` 看作固定值。最里层循环的 `x` 代表选择的任意一层 `x` ，值从 `1` 遍历到 `i`。

此时我们把 `dp[i - x][j]` 和 `dp[x - 1][j - 1]` 分别单独来看。可以看出：

- 对于 `dp[i - x][j]`：当 `x` 增加时，`i - x` 的值减少，`dp[i - x][j]` 的值跟着减小。自变量 `x` 与函数 `dp[i - x][j]` 是一条单调非递增函数。
- 对于 `dp[x - 1][j - 1]`：当 `x` 增加时， `x - 1` 的值增加，`dp[x - 1][j - 1]` 的值跟着增加。自变量 `x` 与函数 `dp[x - 1][j - 1]` 是一条单调非递减函数。

两条函数的交点处就是两个函数较大值的最小值位置。即 `dp[i][j]` 所取位置。而这个位置可以通过二分查找满足 `dp[x - 1][j - 1] >= dp[i - x][j]` 最大的那个 `x`。这样时间复杂度就从 $O(n^2 \times k)$ 优化到了 $O(n  \log n \times k)$。

### 思路 2：代码

```python
class Solution:
    def superEggDrop(self, k: int, n: int) -> int:
        dp = [[0 for _ in range(k + 1)] for i in range(n + 1)]
        
        for i in range(1, n + 1):
            for j in range(1, k + 1):
                dp[i][j] = i

        # for i in range(1, n + 1):
        #     dp[i][1] = i

        for j in range(1, k + 1):
            dp[1][j] = 1

        for i in range(2, n + 1):
            for j in range(2, k + 1):
                left, right = 1, i
                while left < right:
                    mid = left + (right - left) // 2
                    if dp[mid - 1][j - 1] < dp[i - mid][j]:
                        left = mid + 1
                    else:
                        right = mid
                dp[i][j] = max(dp[left - 1][j - 1], dp[i - left][j]) + 1
                    
        return dp[n][k]
```

### 思路 2：复杂度分析

- **时间复杂度**：$O(n  \log n \times k)$。两重循环的时间复杂度为 $O(n \times k)$，二分查找的时间复杂度为 $O(\log n)$。
- **空间复杂度**：$O(n \times k)$。

### 思路 3：动态规划 + 逆向思维

再看一下我们现在的题目要求：已知有 `n` 层楼，`k` 个鸡蛋，求出至少需要扔几次鸡蛋，才能保证无论 `f` 是多少层，都能将 `f` 找出来？

我们可以逆向转换一下思维，将题目转变为：**已知有 `k` 个鸡蛋，最多扔 `x` 次鸡蛋（碎没碎都算 `1` 次），求最多可以检测的多少层？**

我们把未知条件「扔鸡蛋的次数」变为了已知条件，将「检测的楼层个数」变为了未知条件。

这样如果求出来的「检测的楼层个数」大于等于 `n`，则说明 `1` ~ `n` 层楼都考虑全了，`f` 值也就明确了。我们只需要从符合条件的情况中，找出「扔鸡蛋次数」最少的次数即可。

动态规划的具体步骤如下：

###### 1. 划分阶段

按照鸡蛋个数、扔鸡蛋的次数进行阶段划分。

###### 2. 定义状态

定义状态 `dp[i][j]` 表示为：一共有 `i` 个鸡蛋，最多扔 `j` 次鸡蛋（碎没碎都算 `1` 次）的条件下，最多可以检测的楼层个数。

###### 3. 状态转移方程

我们现在有 `i` 个鸡蛋，`j` 次扔鸡蛋的机会，现在尝试在 `1` ~ `n` 层中的任意一层 `x` 扔鸡蛋：

1. 如果鸡蛋没碎，剩下 `i` 个鸡蛋，还有 `j - 1` 次扔鸡蛋的机会，最多可以检测 `dp[i][j - 1]` 层楼层。
2. 如果鸡蛋碎了，剩下 `i - 1` 个鸡蛋，还有 `j - 1` 次扔鸡蛋的机会，最多可以检测 `dp[i - 1][j - 1]` 层楼层。
3. 再加上我们扔鸡蛋的第 `x` 层，`i` 个鸡蛋，`j` 次扔鸡蛋的机会最多可以检测 `dp[i][j - 1] + dp[i - 1][j - 1] + 1` 层。

则状态转移方程为：$dp[i][j] = dp[i][j - 1] + dp[i - 1][j - 1] + 1$。

###### 4. 初始条件

- 当鸡蛋数为 `1` 时，只有 `1` 次扔鸡蛋的机会时，最多可以检测 `1` 层，即 `dp[1][1] = 1`。

###### 5. 最终结果

根据我们之前定义的状态，`dp[i][j]` 表示为：一共有 `i` 个鸡蛋，最多扔 `j` 次鸡蛋（碎没碎都算 `1` 次）的条件下，最多可以检测的楼层个数。则我们需要从满足 `i == k` 并且 `dp[i][j] >= n`（即 `k` 个鸡蛋，`j` 次扔鸡蛋，一共检测出 `n` 层楼）的情况中，找出最小的 ` j`，将其返回。

### 思路 3：代码

```python
class Solution:
    def superEggDrop(self, k: int, n: int) -> int:
        dp = [[0 for _ in range(n + 1)] for i in range(k + 1)]
        dp[1][1] = 1

        for i in range(1, k + 1):
            for j in range(1, n + 1):
                dp[i][j] = dp[i][j - 1] + dp[i - 1][j - 1] + 1
                if i == k and dp[i][j] >= n:
                    return j
        return n
```

### 思路 3：复杂度分析

- **时间复杂度**：$O(n \times k)$。两重循环的时间复杂度为 $O(n \times k)$。
- **空间复杂度**：$O(n \times k)$。

## 参考资料

- 【题解】[题目理解 + 基本解法 + 进阶解法 - 鸡蛋掉落 - 力扣](https://leetcode.cn/problems/super-egg-drop/solution/ji-ben-dong-tai-gui-hua-jie-fa-by-labuladong/)
- 【题解】[动态规划（只解释官方题解方法一）（Java） - 鸡蛋掉落 - 力扣](https://leetcode.cn/problems/super-egg-drop/solution/dong-tai-gui-hua-zhi-jie-shi-guan-fang-ti-jie-fang/)
- 【题解】[动态规划 & 记忆化搜索 2000ms -> 32ms 的过程 - 鸡蛋掉落 - 力扣](https://leetcode.cn/problems/super-egg-drop/solution/python-dong-tai-gui-hua-ji-yi-hua-sou-su-hnj9/)
# [0889. 根据前序和后序遍历构造二叉树](https://leetcode.cn/problems/construct-binary-tree-from-preorder-and-postorder-traversal/)

- 标签：树、数组、哈希表、分治、二叉树
- 难度：中等

## 题目链接

- [0889. 根据前序和后序遍历构造二叉树 - 力扣](https://leetcode.cn/problems/construct-binary-tree-from-preorder-and-postorder-traversal/)

## 题目大意

**描述**：给定一棵无重复值二叉树的前序遍历结果 `preorder` 和后序遍历结果 `postorder`。

**要求**：构造出该二叉树并返回其根节点。如果存在多个答案，则可以返回其中任意一个。

**说明**：

- $1 \le preorder.length \le 30$。
- $1 \le preorder[i] \le preorder.length$。
- `preorder` 中所有值都不同。
- `postorder.length == preorder.length`。
- $1 \le postorder[i] \le postorder.length$。
- `postorder` 中所有值都不同。
- 保证 `preorder` 和 `postorder` 是同一棵二叉树的前序遍历和后序遍历。

**示例**：

- 示例 1：

![](https://assets.leetcode.com/uploads/2021/07/24/lc-prepost.jpg)

```python
输入：preorder = [1,2,4,5,3,6,7], postorder = [4,5,2,6,7,3,1]
输出：[1,2,3,4,5,6,7]
```

- 示例 2：

```python
输入: preorder = [1], postorder = [1]
输出: [1]
```

## 解题思路

### 思路 1：递归

如果已知二叉树的前序遍历序列和后序遍历序列，是不能唯一地确定一棵二叉树的。这是因为没有中序遍历序列无法确定左右部分，也就无法进行子序列的分割。

只有二叉树中每个节点度为 `2` 或者 `0` 的时候，已知前序遍历序列和后序遍历序列，才能唯一地确定一颗二叉树，如果二叉树中存在度为 `1` 的节点时是无法唯一地确定一棵二叉树的，这是因为我们无法判断该节点是左子树还是右子树。

而这道题说明了，如果存在多个答案，则可以返回其中任意一个。

我们可以默认指定前序遍历序列的第 `2` 个值为左子树的根节点，由此递归划分左右子序列。具体操作步骤如下：

1. 从前序遍历序列中可知当前根节点的位置在 `preorder[0]`。

2. 前序遍历序列的第 `2` 个值为左子树的根节点，即 `preorder[1]`。通过在后序遍历中查找上一步根节点对应的位置 `postorder[k]`（该节点右侧为右子树序列），从而将二叉树的左右子树分隔开，并得到左右子树节点的个数。

3. 从上一步得到的左右子树个数将后序遍历结果中的左右子树分开。

4. 构建当前节点，并递归建立左右子树，在左右子树对应位置继续递归遍历并执行上述三步，直到节点为空。

### 思路 1：代码

```python
class Solution:
    def constructFromPrePost(self, preorder: List[int], postorder: List[int]) -> TreeNode:
        def createTree(preorder, postorder, n):
            if n == 0:
                return None
            node = TreeNode(preorder[0])
            if n == 1:
                return node
            k = 0
            while postorder[k] != preorder[1]:
                k += 1
            node.left = createTree(preorder[1: k + 2], postorder[: k + 1], k + 1)
            node.right = createTree(preorder[k + 2: ], postorder[k + 1: -1], n - k - 2)
            return node
        return createTree(preorder, postorder, len(preorder))
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(n^2)$。其中 $n$ 是二叉树的节点数目。
- **空间复杂度**：$O(n^2)$。# [0892. 三维形体的表面积](https://leetcode.cn/problems/surface-area-of-3d-shapes/)

- 标签：几何、数组、数学、矩阵
- 难度：简单

## 题目链接

- [0892. 三维形体的表面积 - 力扣](https://leetcode.cn/problems/surface-area-of-3d-shapes/)

## 题目大意

**描述**：给定一个 $n \times n$ 的网格 $grid$，上面放置着一些 $1 \times 1 \times 1$ 的正方体。每个值 $v = grid[i][j]$ 表示 $v$ 个正方体叠放在对应单元格 $(i, j)$ 上。

放置好正方体后，任何直接相邻的正方体都会互相粘在一起，形成一些不规则的三维形体。

**要求**：返回最终这些形体的总面积。

**说明**：

- 每个形体的底面也需要计入表面积中。

**示例**：

- 示例 1：

![](https://assets.leetcode.com/uploads/2021/01/08/tmp-grid2.jpg)

```python
输入：grid = [[1,2],[3,4]]
输出：34
```

- 示例 2：

![](https://assets.leetcode.com/uploads/2021/01/08/tmp-grid4.jpg)

```python
输入：grid = [[1,1,1],[1,0,1],[1,1,1]]
输出：32
```

## 解题思路

### 思路 1：模拟

使用二重循环遍历所有的正方体，计算每一个正方体所贡献的表面积，将其累积起来即为答案。

而每一个正方体所贡献的表面积，可以通过枚举当前正方体前后左右相邻四个方向上的正方体的个数，从而通过判断计算得出。

- 如果当前位置 $(row, col)$ 存在正方体，则正方体在上下位置上起码贡献了 $2$ 的表面积。
- 如果当前位置 $(row, col)$ 的相邻位置 $(new\underline{\hspace{0.5em}}row, new\underline{\hspace{0.5em}}col)$ 上不存在正方体，说明当前正方体在该方向为最外侧，则 $(row, col)$ 位置所贡献的表面积为当前位置上的正方体个数，即 $grid[row][col]$。
- 如果当前位置 $(row, col)$ 的相邻位置 $(new\underline{\hspace{0.5em}}row, new\underline{\hspace{0.5em}}col)$ 上存在正方体：
	- 如果 $grid[row][col] > grid[new\underline{\hspace{0.5em}}row][new\underline{\hspace{0.5em}}col]$，说明 $grid[row][col]$ 在该方向上底面一部分被 $grid[new\underline{\hspace{0.5em}}row][new\underline{\hspace{0.5em}}col]$ 遮盖了，则 $(row, col)$ 位置所贡献的表面积为 $grid[row][col] - grid[new_row][new_col]$。
	- 如果 $grid[row][col] \le grid[new\underline{\hspace{0.5em}}row][new\underline{\hspace{0.5em}}col]$，说明 $grid[row][col]$ 在该方向上完全被 $grid[new\underline{\hspace{0.5em}}row][new\underline{\hspace{0.5em}}col]$ 遮盖了，则 $(row, col)$ 位置所贡献的表面积为 $0$。

### 思路 1：代码

```Python
class Solution:
    def surfaceArea(self, grid: List[List[int]]) -> int:
        directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        size = len(grid)

        ans = 0
        for row in range(size):
            for col in range(size):
                if grid[row][col]:
                    # 底部、顶部贡献表面积
                    ans += 2
                    for direction in directions:
                        new_row = row + direction[0]
                        new_col = col + direction[1]
                        if 0 <= new_row < size and 0 <= new_col < size:
                            if grid[row][col] > grid[new_row][new_col]:
                                add = grid[row][col] - grid[new_row][new_col]
                            else:
                                add = 0
                        else:
                            add = grid[row][col]
                        ans += add
        return ans
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(n^2)$，其中 $n$ 为二位数组 $grid$ 的行数或列数。
- **空间复杂度**：$O(1)$。

# [0897. 递增顺序搜索树](https://leetcode.cn/problems/increasing-order-search-tree/)

- 标签：栈、树、深度优先搜索、二叉搜索树、二叉树
- 难度：简单

## 题目链接

- [0897. 递增顺序搜索树 - 力扣](https://leetcode.cn/problems/increasing-order-search-tree/)

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



# [0900. RLE 迭代器](https://leetcode.cn/problems/rle-iterator/)

- 标签：设计、数组、计数、迭代器
- 难度：中等

## 题目链接

- [0900. RLE 迭代器 - 力扣](https://leetcode.cn/problems/rle-iterator/)

## 题目大意

**描述**：我们可以使用游程编码（即 RLE）来编码一个整数序列。在偶数长度 $encoding$ ( 从 $0$ 开始 )的游程编码数组中，对于所有偶数 $i$，$encoding[i]$ 告诉我们非负整数 $encoding[i + 1]$ 在序列中重复的次数。

- 例如，序列 $arr = [8,8,8,5,5]$ 可以被编码为 $encoding =[3,8,2,5]$。$encoding =[3,8,0,9,2,5]$ 和 $encoding =[2,8,1,8,2,5]$ 也是 $arr$ 有效的 RLE。

给定一个游程长度的编码数组 $encoding$。

**要求**：设计一个迭代器来遍历它。

实现 `RLEIterator` 类:

- `RLEIterator(int[] encoded)` 用编码后的数组初始化对象。
- `int next(int n)` 以这种方式耗尽后 $n$ 个元素并返回最后一个耗尽的元素。如果没有剩余的元素要耗尽，则返回 $-1$。

**说明**：

- $2 \le encoding.length \le 1000$。
- $encoding.length$ 为偶。
- $0 \le encoding[i] \le 10^9$。
- $1 \le n \le 10^9$。
- 每个测试用例调用 `next` 不高于 $1000$ 次。

**示例**：

- 示例 1：

```python
输入：
["RLEIterator","next","next","next","next"]
[[[3,8,0,9,2,5]],[2],[1],[1],[2]]
输出：
[null,8,8,5,-1]
解释：
RLEIterator rLEIterator = new RLEIterator([3, 8, 0, 9, 2, 5]); // 这映射到序列 [8,8,8,5,5]。
rLEIterator.next(2); // 耗去序列的 2 个项，返回 8。现在剩下的序列是 [8, 5, 5]。
rLEIterator.next(1); // 耗去序列的 1 个项，返回 8。现在剩下的序列是 [5, 5]。
rLEIterator.next(1); // 耗去序列的 1 个项，返回 5。现在剩下的序列是 [5]。
rLEIterator.next(2); // 耗去序列的 2 个项，返回 -1。 这是由于第一个被耗去的项是 5，
但第二个项并不存在。由于最后一个要耗去的项不存在，我们返回 -1。
```

## 解题思路

### 思路 1：模拟

1. 初始化时：
   1. 保存数组 $encoding$ 作为成员变量。
   2. 保存当前位置 $index$，表示当前迭代器指向元素 $encoding[index + 1]$。初始化赋值为 $0$。
   3. 保存当前指向元素 $encoding[index + 1]$ 已经被删除的元素个数 $d\underline{\hspace{0.5em}}cnt$。初始化赋值为 $0$。
2. 调用 `next(n)` 时：
   1. 对于当前元素，先判断当前位置是否超出 $encoding$ 范围，超过则直接返回 $-1$。
   2. 如果未超过，再判断当前元素剩余个数 $encoding[index] - d\underline{\hspace{0.5em}}cnt$ 是否小于 $n$ 个。
      1. 如果小于 $n$ 个，则删除当前元素剩余所有个数，并指向下一位置继续删除剩余元素。
      2. 如果等于大于等于 $n$ 个，则令当前指向元素 $encoding[index + 1]$ 已经被删除的元素个数 $d\underline{\hspace{0.5em}}cnt$ 加上 $n$。

### 思路 1：代码

```Python
class RLEIterator:

    def __init__(self, encoding: List[int]):
        self.encoding = encoding
        self.index = 0
        self.d_cnt = 0

    def next(self, n: int) -> int:
        while self.index < len(self.encoding):
            if self.d_cnt + n > self.encoding[self.index]:
                n -= self.encoding[self.index] - self.d_cnt
                self.d_cnt = 0
                self.index += 2
            else:
                self.d_cnt += n
                return self.encoding[self.index + 1]
        return -1
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(n + m)$，其中 $n$ 为数组 $encoding$ 的长度，$m$ 是调用 `next(n)` 的次数。
- **空间复杂度**：$O(n)$。

# [0901. 股票价格跨度](https://leetcode.cn/problems/online-stock-span/)

- 标签：栈、设计、数据流、单调栈
- 难度：中等

## 题目链接

- [0901. 股票价格跨度 - 力扣](https://leetcode.cn/problems/online-stock-span/)

## 题目大意

要求：编写一个 `StockSpanner` 类，用于收集某些股票的每日报价，并返回该股票当日价格的跨度。

- 今天股票价格的跨度：股票价格小于或等于今天价格的最大连续日数（从今天开始往回数，包括今天）。

例如：如果未来 7 天股票的价格是 `[100, 80, 60, 70, 60, 75, 85]`，那么股票跨度将是 `[1, 1, 1, 2, 1, 4, 6]`。

## 解题思路

「求解小于或等于今天价格的最大连续日」等价于「求出左侧第一个比当前股票价格大的股票，并计算距离」。求出左侧第一个比当前股票价格大的股票我们可以使用「单调递减栈」来做。具体步骤如下：

- 初始化方法：初始化一个空栈，即 `self.stack = []`

- 求解今天股票价格的跨度：

  - 初始化跨度 `span` 为 `1`。
  - 如果今日股票价格 `price` 大于等于栈顶元素 `self.stack[-1][0]`，则：
    - 将其弹出，即 `top = self.stack.pop()`。
    - 跨度累加上弹出栈顶元素的跨度，即 `span += top[1]`。
    - 继续判断，直到遇到一个今日股票价格 `price` 小于栈顶元素的元素位置，再将 `[price, span]` 压入栈中。
  - 如果今日股票价格 `price` 小于栈顶元素 `self.stack[-1][0]`，则直接将 `[price, span]` 压入栈中。

  - 最后输出今天股票价格的跨度 `span`。    

## 代码

```python
class StockSpanner:

    def __init__(self):
        self.stack = []

    def next(self, price: int) -> int:
        span = 1
        while self.stack and price >= self.stack[-1][0]:
            top = self.stack.pop()
            span += top[1]
        self.stack.append([price, span])
        return span
```

# [0902. 最大为 N 的数字组合](https://leetcode.cn/problems/numbers-at-most-n-given-digit-set/)

- 标签：数组、数学、字符串、二分查找、动态规划
- 难度：困难

## 题目链接

- [0902. 最大为 N 的数字组合 - 力扣](https://leetcode.cn/problems/numbers-at-most-n-given-digit-set/)

## 题目大意

**描述**：给定一个按非递减序列排列的数字数组 $digits$。我们可以使用任意次数的 $digits[i]$ 来写数字。例如，如果 `digits = ["1", "3", "5"]`，我们可以写数字，如 `"13"`, `"551"`, 和 `"1351315"`。

**要求**：返回可以生成的小于等于给定整数 $n$ 的正整数个数。

**说明**：

- $1 \le digits.length \le 9$。
- $digits[i].length == 1$。
- $digits[i]$ 是从 `'1'` 到 `'9'` 的数。
- $digits$ 中的所有值都不同。
- $digits$ 按非递减顺序排列。
- $1 \le n \le 10^9$。

**示例**：

- 示例 1：

```python
输入：digits = ["1","3","5","7"], n = 100
输出：20
解释：
可写出的 20 个数字是：
1, 3, 5, 7, 11, 13, 15, 17, 31, 33, 35, 37, 51, 53, 55, 57, 71, 73, 75, 77。
```

- 示例 2：

```python
输入：digits = ["1","4","9"], n = 1000000000
输出：29523
解释：
我们可以写 3 个一位数字，9 个两位数字，27 个三位数字，
81 个四位数字，243 个五位数字，729 个六位数字，
2187 个七位数字，6561 个八位数字和 19683 个九位数字。
总共，可以使用D中的数字写出 29523 个整数。
```

## 解题思路

### 思路 1：动态规划 + 数位 DP

数位 DP 模板的应用。因为这道题目中可以使用任意次数的 $digits[i]$，所以不需要用状态压缩的方式来表示数字集合。

这道题的具体步骤如下：

将 $n$ 转换为字符串 $s$，定义递归函数 `def dfs(pos, isLimit, isNum):` 表示构造第 $pos$ 位及之后所有数位的合法方案数。接下来按照如下步骤进行递归。

1. 从 `dfs(0, True, False)` 开始递归。 `dfs(0, True, False)` 表示：
   1. 从位置 $0$ 开始构造。
   2. 开始时受到数字 $n$ 对应最高位数位的约束。
   3. 开始时没有填写数字。
2. 如果遇到  $pos == len(s)$，表示到达数位末尾，此时：
   1. 如果 $isNum == True$，说明当前方案符合要求，则返回方案数 $1$。
   2. 如果 $isNum == False$，说明当前方案不符合要求，则返回方案数 $0$。
3. 如果 $pos \ne len(s)$，则定义方案数 $ans$，令其等于 $0$，即：`ans = 0`。
4. 如果遇到 $isNum == False$，说明之前位数没有填写数字，当前位可以跳过，这种情况下方案数等于 $pos + 1$ 位置上没有受到 $pos$ 位的约束，并且之前没有填写数字时的方案数，即：`ans = dfs(i + 1, False, False)`。
5. 如果 $isNum == True$，则当前位必须填写一个数字。此时：
   1. 根据 $isNum$ 和 $isLimit$ 来决定填当前位数位所能选择的最大数字（$maxX$）。
   2. 然后枚举 $digits$ 数组中所有能够填入的数字 $d$。
   3. 如果 $d$ 超过了所能选择的最大数字 $maxX$ 则直接跳出循环。
   4. 如果 $d$ 是合法数字，则方案数累加上当前位选择 $d$ 之后的方案数，即：`ans += dfs(pos + 1, isLimit and d == maxX, True)`。
      1. `isLimit and d == maxX` 表示 $pos + 1$ 位受到之前位限制和 $pos$ 位限制。
      2. $isNum == True$ 表示 $pos$ 位选择了数字。
6. 最后的方案数为 `dfs(0, True, False)`，将其返回即可。

### 思路 1：代码

```python
class Solution:
    def atMostNGivenDigitSet(self, digits: List[str], n: int) -> int:
        # 将 n 转换为字符串 s
        s = str(n)
        
        @cache
        # pos: 第 pos 个数位
        # isLimit: 表示是否受到选择限制。如果为真，则第 pos 位填入数字最多为 s[pos]；如果为假，则最大可为 9。
        # isNum: 表示 pos 前面的数位是否填了数字。如果为真，则当前位不可跳过；如果为假，则当前位可跳过。
        def dfs(pos, isLimit, isNum):
            if pos == len(s):
                # isNum 为 True，则表示当前方案符合要求
                return int(isNum)
            
            ans = 0
            if not isNum:
                # 如果 isNumb 为 False，则可以跳过当前数位
                ans = dfs(pos + 1, False, False)
            
            # 如果受到选择限制，则最大可选择数字为 s[pos]，否则最大可选择数字为 9。
            maxX = s[pos] if isLimit else '9'
            
            # 枚举可选择的数字
            for d in digits:
                if d > maxX:
                    break
                ans += dfs(pos + 1, isLimit and d == maxX, True)

            return ans
        
        return dfs(0, True, False)
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(m \times \log n)$，其中 $m$ 是数组 $digits$ 的长度，$\log n$ 是 $n$ 转为字符串之后的位数长度。
- **空间复杂度**：$O(\log n)$。

# [0904. 水果成篮](https://leetcode.cn/problems/fruit-into-baskets/)

- 标签：数组、哈希表、滑动窗口
- 难度：中等

## 题目链接

- [0904. 水果成篮 - 力扣](https://leetcode.cn/problems/fruit-into-baskets/)

## 题目大意

给定一个数组 `fruits`。其中 `fruits[i]` 表示第 `i` 棵树会产生 `fruits[i]` 型水果。

你可以从你选择的任何树开始，然后重复执行以下步骤：

- 把这棵树上的水果放进你的篮子里。如果你做不到，就停下来。
- 移动到当前树右侧的下一棵树。如果右边没有树，就停下来。
- 请注意，在选择一棵树后，你没有任何选择：你必须执行步骤 1，然后执行步骤 2，然后返回步骤 1，然后执行步骤 2，依此类推，直至停止。

你有 `2` 个篮子，每个篮子可以携带任何数量的水果，但你希望每个篮子只携带一种类型的水果。

要求：返回你能收集的水果树的最大总量。

## 解题思路

只有 `2` 个篮子，要求在连续子数组中装最多 `2` 种不同水果。可以理解为维护一个水果种类数为 `2` 的滑动数组，求窗口中最大的水果树数目。具体做法如下：

- 用滑动窗口 `window` 来维护不同种类水果树数目。`window` 为哈希表类型。`ans` 用来维护能收集的水果树的最大总量。设定两个指针：`left`、`right`，分别指向滑动窗口的左右边界，保证窗口中水果种类数不超过 `2` 种。
- 一开始，`left`、`right` 都指向 `0`。
- 将最右侧数组元素 `fruits[right]` 加入当前窗口 `window` 中，该水果树数目 +1。
- 如果该窗口中该水果树种类多于 `2` 种，即 `len(window) > 2`，则不断右移 `left`，缩小滑动窗口长度，并更新窗口中对应水果树的个数，直到 `len(window) <= 2`。
- 维护更新能收集的水果树的最大总量。然后右移 `right`，直到 `right >= len(fruits)` 结束。
- 输出能收集的水果树的最大总量。

## 代码

```python
class Solution:
    def totalFruit(self, fruits: List[int]) -> int:
        window = dict()
        window_size = 2
        ans = 0
        left, right = 0, 0
        while right < len(fruits):
            if fruits[right] in window:
                window[fruits[right]] += 1
            else:
                window[fruits[right]] = 1

            while len(window) > window_size:
                window[fruits[left]] -= 1
                if window[fruits[left]] == 0:
                    del window[fruits[left]]
                left += 1
            ans = max(ans, right - left + 1)
            right += 1
        return ans
```

# [0908. 最小差值 I](https://leetcode.cn/problems/smallest-range-i/)

- 标签：数组、数学
- 难度：简单

## 题目链接

- [0908. 最小差值 I - 力扣](https://leetcode.cn/problems/smallest-range-i/)

## 题目大意

**描述**：给定一个整数数组 `nums`，和一个整数 `k`。给数组中的每个元素 `nums[i]` 都加上一个任意数字 `x` （`-k <= x <= k`），从而得到一个新数组 `result`。

**要求**：返回数组 `result` 的最大值和最小值之间可能存在的最小差值。

**说明**：

- $1 \le nums.length \le 10^4$。
- $0 \le nums[i] \le 10^4$。
- $0 \le k \le 10^4$。

**示例**：

- 示例 1：

```python
输入：nums = [1], k = 0
输出：0
解释：分数是 max(nums) - min(nums) = 1 - 1 = 0。
```

- 示例 2：

```python
输入：nums = [0,10], k = 2
输出：6
解释：将 nums 改为 [2,8]。分数是 max(nums) - min(nums) = 8 - 2 = 6。
```

## 解题思路

### 思路 1：数学

`nums` 中的每个元素可以波动 `[-k, k]`。最小的差值就是「最大值减去 `k`」和「最小值加上 `k`」之间的差值。而如果差值小于 `0`，则说明每个数字都可以波动成相等的数字，此时直接返回 `0` 即可。

### 思路 1：代码

```python
class Solution:
    def smallestRangeI(self, nums: List[int], k: int) -> int:
        return max(0, max(nums) - min(nums) - 2*k)
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(n)$。
- **空间复杂度**：$O(1)$。

# [0912. 排序数组](https://leetcode.cn/problems/sort-an-array/)

- 标签：数组、分治、桶排序、计数排序、基数排序、排序、堆（优先队列）、归并排序
- 难度：中等

## 题目链接

- [0912. 排序数组 - 力扣](https://leetcode.cn/problems/sort-an-array/)

## 题目大意

**描述**：给定一个整数数组 $nums$。

**要求**：将该数组升序排列。

**说明**：

- $1 \le nums.length \le 5 * 10^4$。
- $-5 * 10^4 \le nums[i] \le 5 * 10^4$。

**示例**：

- 示例 1：

```python
输入：nums = [5,2,3,1]
输出：[1,2,3,5]
```

- 示例 2：

```python
输入：nums = [5,1,1,2,0,0]
输出：[0,0,1,1,2,5]
```

## 解题思路

这道题是一道用来复习排序算法，测试算法时间复杂度的好题。我试过了十种排序算法。得到了如下结论：

- 超时算法（时间复杂度为 $O(n^2)$）：冒泡排序、选择排序、插入排序。
- 通过算法（时间复杂度为 $O(n \times \log n)$）：希尔排序、归并排序、快速排序、堆排序。
- 通过算法（时间复杂度为 $O(n)$）：计数排序、桶排序。
- 解答错误算法（普通基数排序只适合非负数）：基数排序。

### 思路 1：冒泡排序（超时）

> **冒泡排序（Bubble Sort）基本思想**：经过多次迭代，通过相邻元素之间的比较与交换，使值较小的元素逐步从后面移到前面，值较大的元素从前面移到后面。

假设数组的元素个数为 $n$ 个，则冒泡排序的算法步骤如下：

1. 第 $1$ 趟「冒泡」：对前 $n$ 个元素执行「冒泡」，从而使第 $1$ 个值最大的元素放置在正确位置上。
	1. 先将序列中第 $1$ 个元素与第 $2$ 个元素进行比较，如果前者大于后者，则两者交换位置，否则不交换。
	2. 然后将第 $2$ 个元素与第 $3$ 个元素比较，如果前者大于后者，则两者交换位置，否则不交换。
	3. 依次类推，直到第 $n - 1$ 个元素与第 $n$ 个元素比较（或交换）为止。
	4. 经过第 $1$ 趟排序，使得 $n$ 个元素中第 $i$ 个值最大元素被安置在第 $n$ 个位置上。
2. 第 $2$ 趟「冒泡」：对前 $n - 1$ 个元素执行「冒泡」，从而使第 $2$ 个值最大的元素放置在正确位置上。
	1. 先将序列中第 $1$ 个元素与第 $2$ 个元素进行比较，若前者大于后者，则两者交换位置，否则不交换。
	2. 然后将第 $2$ 个元素与第 $3$ 个元素比较，若前者大于后者，则两者交换位置，否则不交换。
	3. 依次类推，直到第 $n - 2$ 个元素与第 $n - 1$ 个元素比较（或交换）为止。
	4. 经过第 $2$ 趟排序，使得数组中第 $2$ 个值最大元素被安置在第 $n$ 个位置上。
3. 依次类推，重复上述「冒泡」过程，直到某一趟排序过程中不出现元素交换位置的动作，则排序结束。

### 思路 1：代码

```python
class Solution:
    def bubbleSort(self, nums: [int]) -> [int]:
        # 第 i 趟「冒泡」
        for i in range(len(nums) - 1):
            flag = False    # 是否发生交换的标志位
            # 对数组未排序区间 [0, n - i - 1] 的元素执行「冒泡」
            for j in range(len(nums) - i - 1):
                # 相邻两个元素进行比较，如果前者大于后者，则交换位置
                if nums[j] > nums[j + 1]:
                    nums[j], nums[j + 1] = nums[j + 1], nums[j]
                    flag = True
            if not flag:    # 此趟遍历未交换任何元素，直接跳出
                break
        
        return nums
    
    def sortArray(self, nums: [int]) -> [int]:
        return self.bubbleSort(nums)
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(n^2)$。
- **空间复杂度**：$O(1)$。

### 思路 2：选择排序（超时）

>**选择排序（Selection Sort）基本思想**：将数组分为两个区间，左侧为已排序区间，右侧为未排序区间。每趟从未排序区间中选择一个值最小的元素，放到已排序区间的末尾，从而将该元素划分到已排序区间。

假设数组的元素个数为 $n$ 个，则选择排序的算法步骤如下：

1. 初始状态下，无已排序区间，未排序区间为 $[0, n - 1]$。
2. 第 $1$ 趟选择：
	1. 遍历未排序区间 $[0, n - 1]$，使用变量 $min\underline{\hspace{0.5em}}i$ 记录区间中值最小的元素位置。
	2. 将 $min\underline{\hspace{0.5em}}i$ 与下标为 $0$ 处的元素交换位置。如果下标为 $0$ 处元素就是值最小的元素位置，则不用交换。
	3. 此时，$[0, 0]$ 为已排序区间，$[1, n - 1]$（总共 $n - 1$ 个元素）为未排序区间。
3. 第 $2$ 趟选择：
	1. 遍历未排序区间 $[1, n - 1]$，使用变量 $min\underline{\hspace{0.5em}}i$ 记录区间中值最小的元素位置。
	2. 将 $min\underline{\hspace{0.5em}}i$ 与下标为 $1$ 处的元素交换位置。如果下标为 $1$ 处元素就是值最小的元素位置，则不用交换。
	3. 此时，$[0, 1]$ 为已排序区间，$[2, n - 1]$（总共 $n - 2$ 个元素）为未排序区间。
4. 依次类推，对剩余未排序区间重复上述选择过程，直到所有元素都划分到已排序区间，排序结束。

### 思路 2：代码

```python
class Solution:
    def selectionSort(self, nums: [int]) -> [int]:
        for i in range(len(nums) - 1):
            # 记录未排序区间中最小值的位置
            min_i = i
            for j in range(i + 1, len(nums)):
                if nums[j] < nums[min_i]:
                    min_i = j
            # 如果找到最小值的位置，将 i 位置上元素与最小值位置上的元素进行交换
            if i != min_i:
                nums[i], nums[min_i] = nums[min_i], nums[i]
        return nums

    def sortArray(self, nums: [int]) -> [int]:
        return self.selectionSort(nums)
```

### 思路 2：复杂度分析

- **时间复杂度**：$O(n^2)$。
- **空间复杂度**：$O(1)$。

### 思路 3：插入排序（超时）

>**插入排序（Insertion Sort）基本思想**：将数组分为两个区间，左侧为有序区间，右侧为无序区间。每趟从无序区间取出一个元素，然后将其插入到有序区间的适当位置。

假设数组的元素个数为 $n$ 个，则插入排序的算法步骤如下：

1. 初始状态下，有序区间为 $[0, 0]$，无序区间为 $[1, n - 1]$。
2. 第 $1$ 趟插入：
	1. 取出无序区间 $[1, n - 1]$ 中的第 $1$ 个元素，即 $nums[1]$。
	2. 从右到左遍历有序区间中的元素，将比 $nums[1]$ 小的元素向后移动 $1$ 位。
	3. 如果遇到大于或等于 $nums[1]$ 的元素时，说明找到了插入位置，将 $nums[1]$ 插入到该位置。
	4. 插入元素后有序区间变为 $[0, 1]$，无序区间变为 $[2, n - 1]$。
3. 第 $2$ 趟插入：
	1. 取出无序区间 $[2, n - 1]$ 中的第 $1$ 个元素，即 $nums[2]$。
	2. 从右到左遍历有序区间中的元素，将比 $nums[2]$ 小的元素向后移动 $1$ 位。
	3. 如果遇到大于或等于 $nums[2]$ 的元素时，说明找到了插入位置，将 $nums[2]$ 插入到该位置。
	4. 插入元素后有序区间变为 $[0, 2]$，无序区间变为 $[3, n - 1]$。
4. 依次类推，对剩余无序区间中的元素重复上述插入过程，直到所有元素都插入到有序区间中，排序结束。

### 思路 3：代码

```python
class Solution:
    def insertionSort(self, nums: [int]) -> [int]:
        # 遍历无序区间
        for i in range(1, len(nums)):
            temp = nums[i]
            j = i
            # 从右至左遍历有序区间
            while j > 0 and nums[j - 1] > temp:
                # 将有序区间中插入位置右侧的所有元素依次右移一位
                nums[j] = nums[j - 1]
                j -= 1
            # 将该元素插入到适当位置
            nums[j] = temp

        return nums

    def sortArray(self, nums: [int]) -> [int]:
        return self.insertionSort(nums)
```

### 思路 3：复杂度分析

- **时间复杂度**：$O(n^2)$。
- **空间复杂度**：$O(1)$。

### 思路 4：希尔排序（通过）

> **希尔排序（Shell Sort）基本思想**：将整个数组切按照一定的间隔取值划分为若干个子数组，每个子数组分别进行插入排序。然后逐渐缩小间隔进行下一轮划分子数组和对子数组进行插入排序。直至最后一轮排序间隔为 $1$，对整个数组进行插入排序。

假设数组的元素个数为 $n$ 个，则希尔排序的算法步骤如下：

1. 确定一个元素间隔数 $gap$。
2. 将参加排序的数组按此间隔数从第 $1$ 个元素开始一次分成若干个子数组，即分别将所有位置相隔为 $gap$ 的元素视为一个子数组。
3. 在各个子数组中采用某种排序算法（例如插入排序算法）进行排序。
4. 减少间隔数，并重新将整个数组按新的间隔数分成若干个子数组，再分别对各个子数组进行排序。
5. 依次类推，直到间隔数 $gap$ 值为 $1$，最后进行一次排序，排序结束。

### 思路 4：代码

```python
class Solution:
    def shellSort(self, nums: [int]) -> [int]:
        size = len(nums)
        gap = size // 2
        # 按照 gap 分组
        while gap > 0:
            # 对每组元素进行插入排序
            for i in range(gap, size):
                # temp 为每组中无序数组第 1 个元素
                temp = nums[i]
                j = i
                # 从右至左遍历每组中的有序数组元素
                while j >= gap and nums[j - gap] > temp:
                    # 将每组有序数组中插入位置右侧的元素依次在组中右移一位
                    nums[j] = nums[j - gap]
                    j -= gap
                # 将该元素插入到适当位置
                nums[j] = temp
            # 缩小 gap 间隔
            gap = gap // 2
        return nums

    def sortArray(self, nums: [int]) -> [int]:
        return self.shellSort(nums)
```

### 思路 4：复杂度分析

- **时间复杂度**：介于 $O(n \times \log n)$ 与 $O(n^2)$ 之间。
- **空间复杂度**：$O(1)$。

### 思路 5：归并排序（通过）

> **归并排序（Merge Sort）基本思想**：采用经典的分治策略，先递归地将当前数组平均分成两半，然后将有序数组两两合并，最终合并成一个有序数组。

假设数组的元素个数为 $n$ 个，则归并排序的算法步骤如下：

1. **分解过程**：先递归地将当前数组平均分成两半，直到子数组长度为 $1$。
	1. 找到数组中心位置 $mid$，从中心位置将数组分成左右两个子数组 $left\underline{\hspace{0.5em}}nums$、$right\underline{\hspace{0.5em}}nums$。
	2. 对左右两个子数组 $left\underline{\hspace{0.5em}}nums$、$right\underline{\hspace{0.5em}}nums$ 分别进行递归分解。
	3. 最终将数组分解为 $n$ 个长度均为 $1$ 的有序子数组。
2. **归并过程**：从长度为 $1$ 的有序子数组开始，依次将有序数组两两合并，直到合并成一个长度为 $n$ 的有序数组。
	1. 使用数组变量 $nums$ 存放合并后的有序数组。
	2. 使用两个指针 $left\underline{\hspace{0.5em}}i$、$right\underline{\hspace{0.5em}}i$ 分别指向两个有序子数组 $left\underline{\hspace{0.5em}}nums$、$right\underline{\hspace{0.5em}}nums$ 的开始位置。
	3. 比较两个指针指向的元素，将两个有序子数组中较小元素依次存入到结果数组 $nums$ 中，并将指针移动到下一位置。
	4. 重复步骤 $3$，直到某一指针到达子数组末尾。
	5. 将另一个子数组中的剩余元素存入到结果数组 $nums$ 中。
	6. 返回合并后的有序数组 $nums$。

### 思路 5：代码

```python
class Solution:
    # 合并过程
    def merge(self, left_nums: [int], right_nums: [int]):
        nums = []
        left_i, right_i = 0, 0
        while left_i < len(left_nums) and right_i < len(right_nums):
            # 将两个有序子数组中较小元素依次插入到结果数组中
            if left_nums[left_i] < right_nums[right_i]:
                nums.append(left_nums[left_i])
                left_i += 1
            else:
                nums.append(right_nums[right_i])
                right_i += 1
        
        # 如果左子数组有剩余元素，则将其插入到结果数组中
        while left_i < len(left_nums):
            nums.append(left_nums[left_i])
            left_i += 1
        
        # 如果右子数组有剩余元素，则将其插入到结果数组中
        while right_i < len(right_nums):
            nums.append(right_nums[right_i])
            right_i += 1
        
        # 返回合并后的结果数组
        return nums

    # 分解过程
    def mergeSort(self, nums: [int]) -> [int]:
        # 数组元素个数小于等于 1 时，直接返回原数组
        if len(nums) <= 1:
            return nums
        
        mid = len(nums) // 2                        # 将数组从中间位置分为左右两个数组
        left_nums = self.mergeSort(nums[0: mid])    # 递归将左子数组进行分解和排序
        right_nums =  self.mergeSort(nums[mid:])    # 递归将右子数组进行分解和排序
        return self.merge(left_nums, right_nums)    # 把当前数组组中有序子数组逐层向上，进行两两合并

    def sortArray(self, nums: [int]) -> [int]:
        return self.mergeSort(nums)
```

### 思路 5：复杂度分析

- **时间复杂度**：$O(n \times \log n)$。
- **空间复杂度**：$O(n)$。

### 思路 6：快速排序（通过）

> **快速排序（Quick Sort）基本思想**：采用经典的分治策略，选择数组中某个元素作为基准数，通过一趟排序将数组分为独立的两个子数组，一个子数组中所有元素值都比基准数小，另一个子数组中所有元素值都比基准数大。然后再按照同样的方式递归的对两个子数组分别进行快速排序，以达到整个数组有序。

假设数组的元素个数为 $n$ 个，则快速排序的算法步骤如下：

1. **哨兵划分**：选取一个基准数，将数组中比基准数大的元素移动到基准数右侧，比他小的元素移动到基准数左侧。
	1. 从当前数组中找到一个基准数 $pivot$（这里以当前数组第 $1$ 个元素作为基准数，即 $pivot = nums[low]$）。
	2. 使用指针 $i$ 指向数组开始位置，指针 $j$  指向数组末尾位置。
	3. 从右向左移动指针 $j$，找到第 $1$ 个小于基准值的元素。
	4. 从左向右移动指针 $i$，找到第 $1$ 个大于基准数的元素。
	5. 交换指针 $i$、指针 $j$ 指向的两个元素位置。
	6. 重复第 $3 \sim 5$ 步，直到指针 $i$ 和指针 $j$ 相遇时停止，最后将基准数放到两个子数组交界的位置上。
2. **递归分解**：完成哨兵划分之后，对划分好的左右子数组分别进行递归排序。
	1. 按照基准数的位置将数组拆分为左右两个子数组。
	2. 对每个子数组分别重复「哨兵划分」和「递归分解」，直到各个子数组只有 $1$ 个元素，排序结束。

### 思路 6：代码

```python
import random

class Solution:
    # 随机哨兵划分：从 nums[low: high + 1] 中随机挑选一个基准数，并进行移位排序
    def randomPartition(self, nums: [int], low: int, high: int) -> int:
        # 随机挑选一个基准数
        i = random.randint(low, high)
        # 将基准数与最低位互换
        nums[i], nums[low] = nums[low], nums[i]
        # 以最低位为基准数，然后将数组中比基准数大的元素移动到基准数右侧，比他小的元素移动到基准数左侧。最后将基准数放到正确位置上
        return self.partition(nums, low, high)
    
    # 哨兵划分：以第 1 位元素 nums[low] 为基准数，然后将比基准数小的元素移动到基准数左侧，将比基准数大的元素移动到基准数右侧，最后将基准数放到正确位置上
    def partition(self, nums: [int], low: int, high: int) -> int:        
        # 以第 1 位元素为基准数
        pivot = nums[low]
        
        i, j = low, high
        while i < j:
            # 从右向左找到第 1 个小于基准数的元素
            while i < j and nums[j] >= pivot:
                j -= 1
            # 从左向右找到第 1 个大于基准数的元素
            while i < j and nums[i] <= pivot:
                i += 1
            # 交换元素
            nums[i], nums[j] = nums[j], nums[i]
        
        # 将基准数放到正确位置上
        nums[j], nums[low] = nums[low], nums[j]
        return j

    def quickSort(self, nums: [int], low: int, high: int) -> [int]:
        if low < high:
            # 按照基准数的位置，将数组划分为左右两个子数组
            pivot_i = self.partition(nums, low, high)
            # 对左右两个子数组分别进行递归快速排序
            self.quickSort(nums, low, pivot_i - 1)
            self.quickSort(nums, pivot_i + 1, high)

        return nums

    def sortArray(self, nums: [int]) -> [int]:
        return self.quickSort(nums, 0, len(nums) - 1)
```

### 思路 6：复杂度分析

- **时间复杂度**：$O(n \times \log n)$。
- **空间复杂度**：$O(n)$。

### 思路 7：堆排序（通过）

> **堆排序（Heap sort）基本思想**：借用「堆结构」所设计的排序算法。将数组转化为大顶堆，重复从大顶堆中取出数值最大的节点，并让剩余的堆结构继续维持大顶堆性质。

假设数组的元素个数为 $n$ 个，则堆排序的算法步骤如下：

1. **构建初始大顶堆**：
	1. 定义一个数组实现的堆结构，将原始数组的元素依次存入堆结构的数组中（初始顺序不变）。
	2. 从数组的中间位置开始，从右至左，依次通过「下移调整」将数组转换为一个大顶堆。

2. **交换元素，调整堆**：
	1. 交换堆顶元素（第 $1$ 个元素）与末尾（最后 $1$ 个元素）的位置，交换完成后，堆的长度减 $1$。
	2. 交换元素之后，由于堆顶元素发生了改变，需要从根节点开始，对当前堆进行「下移调整」，使其保持堆的特性。

3. **重复交换和调整堆**：
	1. 重复第 $2$ 步，直到堆的大小为 $1$ 时，此时大顶堆的数组已经完全有序。

### 思路 7：代码

```python
class Solution:
    # 调整为大顶堆
    def heapify(self, arr, index, end):
        left = index * 2 + 1
        right = left + 1
        while left <= end:
            # 当前节点为非叶子节点
            max_index = index
            if arr[left] > arr[max_index]:
                max_index = left
            if right <= end and arr[right] > arr[max_index]:
                max_index = right
            if index == max_index:
                # 如果不用交换，则说明已经交换结束
                break
            arr[index], arr[max_index] = arr[max_index], arr[index]
            # 继续调整子树
            index = max_index
            left = index * 2 + 1
            right = left + 1

    # 初始化大顶堆
    def buildMaxHeap(self, arr):
        size = len(arr)
        # (size-2) // 2 是最后一个非叶节点，叶节点不用调整
        for i in range((size - 2) // 2, -1, -1):
            self.heapify(arr, i, size - 1)
        return arr

    # 升序堆排序，思路如下：
    # 1. 先建立大顶堆
    # 2. 让堆顶最大元素与最后一个交换，然后调整第一个元素到倒数第二个元素，这一步获取最大值
    # 3. 再交换堆顶元素与倒数第二个元素，然后调整第一个元素到倒数第三个元素，这一步获取第二大值
    # 4. 以此类推，直到最后一个元素交换之后完毕。
    def maxHeapSort(self, arr):
        self.buildMaxHeap(arr)
        size = len(arr)
        for i in range(size):
            arr[0], arr[size-i-1] = arr[size-i-1], arr[0]
            self.heapify(arr, 0, size-i-2)
        return arr

    def sortArray(self, nums: List[int]) -> List[int]:
        return self.maxHeapSort(nums)
```

### 思路 7：复杂度分析

- **时间复杂度**：$O(n \times \log n)$。
- **空间复杂度**：$O(1)$。

### 思路 8：计数排序（通过）

> **计数排序（Counting Sort）基本思想**：通过统计数组中每个元素在数组中出现的次数，根据这些统计信息将数组元素有序的放置到正确位置，从而达到排序的目的。

假设数组的元素个数为 $n$ 个，则计数排序的算法步骤如下：

1. **计算排序范围**：遍历数组，找出待排序序列中最大值元素 $nums\underline{\hspace{0.5em}}max$ 和最小值元素 $nums\underline{\hspace{0.5em}}min$，计算出排序范围为 $nums\underline{\hspace{0.5em}}max - nums\underline{\hspace{0.5em}}min + 1$。
2. **定义计数数组**：定义一个大小为排序范围的计数数组 $counts$，用于统计每个元素的出现次数。其中：
	1. 数组的索引值 $num - nums\underline{\hspace{0.5em}}min$ 表示元素的值为 $num$。
	2. 数组的值 $counts[num - nums\underline{\hspace{0.5em}}min]$ 表示元素 $num$ 的出现次数。

3. **对数组元素进行计数统计**：遍历待排序数组 $nums$，对每个元素在计数数组中进行计数，即将待排序数组中「每个元素值减去最小值」作为索引，将「对计数数组中的值」加 $1$，即令 $counts[num - nums\underline{\hspace{0.5em}}min]$ 加 $1$。
4. **生成累积计数数组**：从 $counts$ 中的第 $1$ 个元素开始，每一项累家前一项和。此时 $counts[num - nums\underline{\hspace{0.5em}}min]$ 表示值为 $num$ 的元素在排序数组中最后一次出现的位置。
5. **逆序填充目标数组**：逆序遍历数组 $nums$，将每个元素 $num$ 填入正确位置。
  6. 将其填充到结果数组 $res$ 的索引 $counts[num - nums\underline{\hspace{0.5em}}min]$ 处。
  7. 放入后，令累积计数数组中对应索引减 $1$，从而得到下个元素 $num$ 的放置位置。

### 思路 8：代码

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

    def sortArray(self, nums: [int]) -> [int]:
        return self.countingSort(nums)
```

### 思路 8：复杂度分析

- **时间复杂度**：$O(n + k)$。其中 $k$ 代表待排序序列的值域。
- **空间复杂度**：$O(k)$。其中 $k$ 代表待排序序列的值域。

### 思路 9：桶排序（通过）

> **桶排序（Bucket Sort）基本思想**：将待排序数组中的元素分散到若干个「桶」中，然后对每个桶中的元素再进行单独排序。

假设数组的元素个数为 $n$ 个，则桶排序的算法步骤如下：

1. **确定桶的数量**：根据待排序数组的值域范围，将数组划分为 $k$ 个桶，每个桶可以看做是一个范围区间。
2. **分配元素**：遍历待排序数组元素，将每个元素根据大小分配到对应的桶中。
3. **对每个桶进行排序**：对每个非空桶内的元素单独排序（使用插入排序、归并排序、快排排序等算法）。
4. **合并桶内元素**：将排好序的各个桶中的元素按照区间顺序依次合并起来，形成一个完整的有序数组。

### 思路 9：代码

```python
class Solution:
    def insertionSort(self, nums: [int]) -> [int]:
        # 遍历无序区间
        for i in range(1, len(nums)):
            temp = nums[i]
            j = i
            # 从右至左遍历有序区间
            while j > 0 and nums[j - 1] > temp:
                # 将有序区间中插入位置右侧的元素依次右移一位
                nums[j] = nums[j - 1]
                j -= 1
            # 将该元素插入到适当位置
            nums[j] = temp
            
        return nums

    def bucketSort(self,  nums: [int], bucket_size=5) -> [int]:
        # 计算待排序序列中最大值元素 nums_max、最小值元素 nums_min
        nums_min, nums_max = min(nums), max(nums)
        # 定义桶的个数为 (最大值元素 - 最小值元素) // 每个桶的大小 + 1
        bucket_count = (nums_max - nums_min) // bucket_size + 1
        # 定义桶数组 buckets
        buckets = [[] for _ in range(bucket_count)]

        # 遍历待排序数组元素，将每个元素根据大小分配到对应的桶中
        for num in nums:
            buckets[(num - nums_min) // bucket_size].append(num)

        # 对每个非空桶内的元素单独排序，排序之后，按照区间顺序依次合并到 res 数组中
        res = []
        for bucket in buckets:
            self.insertionSort(bucket)
            res.extend(bucket)
        
        # 返回结果数组
        return res

    def sortArray(self, nums: [int]) -> [int]:
        return self.bucketSort(nums)
```

### 思路 9：复杂度分析

- **时间复杂度**：$O(n)$。
- **空间复杂度**：$O(n + m)$。$m$ 为桶的个数。

### 思路 10：基数排序（提交解答错误，普通基数排序只适合非负数）

> **基数排序（Radix Sort）基本思想**：将整数按位数切割成不同的数字，然后从低位开始，依次到高位，逐位进行排序，从而达到排序的目的。

我们以最低位优先法为例，讲解一下基数排序的算法步骤。

1. **确定排序的最大位数**：遍历数组元素，获取数组最大值元素，并取得对应位数。
2. **从最低位（个位）开始，到最高位为止，逐位对每一位进行排序**：
	1. 定义一个长度为 $10$ 的桶数组 $buckets$，每个桶分别代表 $0 \sim 9$ 中的 $1$ 个数字。
	2. 按照每个元素当前位上的数字，将元素放入对应数字的桶中。
	3. 清空原始数组，然后按照桶的顺序依次取出对应元素，重新加入到原始数组中。

### 思路 10：代码

```python
class Solution:
    def radixSort(self, nums: [int]) -> [int]:
        # 桶的大小为所有元素的最大位数
        size = len(str(max(nums)))
        
        # 从最低位（个位）开始，逐位遍历每一位
        for i in range(size):
            # 定义长度为 10 的桶数组 buckets，每个桶分别代表 0 ~ 9 中的 1 个数字。
            buckets = [[] for _ in range(10)]
            # 遍历数组元素，按照每个元素当前位上的数字，将元素放入对应数字的桶中。
            for num in nums:
                buckets[num // (10 ** i) % 10].append(num)
            # 清空原始数组
            nums.clear()
            # 按照桶的顺序依次取出对应元素，重新加入到原始数组中。
            for bucket in buckets:
                for num in bucket:
                    nums.append(num)
                    
        # 完成排序，返回结果数组
        return nums
    
    def sortArray(self, nums: [int]) -> [int]:
        return self.radixSort(nums)
```

### 思路 10：复杂度分析

- **时间复杂度**：$O(n \times k)$。其中 $n$ 是待排序元素的个数，$k$ 是数字位数。$k$ 的大小取决于数字位的选择（十进制位、二进制位）和待排序元素所属数据类型全集的大小。
- **空间复杂度**：$O(n + k)$。

# [0918. 环形子数组的最大和](https://leetcode.cn/problems/maximum-sum-circular-subarray/)

- 标签：队列、数组、分治、动态规划、单调队列
- 难度：中等

## 题目链接

- [0918. 环形子数组的最大和 - 力扣](https://leetcode.cn/problems/maximum-sum-circular-subarray/)

## 题目大意

给定一个环形整数数组 nums，数组 nums 的尾部和头部是相连状态。求环形数组 nums 的非空子数组的最大和（子数组中每个位置元素最多出现一次）。

## 解题思路

构成环形整数数组 nums 的非空子数组的最大和的子数组有两种情况：

- 最大和的子数组为一个子区间：$nums[i] + nums[i+1] + nums[i+2] + ... + num[j]$。
- 最大和的子数组为首尾的两个子区间：$(nums[0] + nums[1] + ... + nums[i]) + (nums[j] + nums[j+1] + ... + num[N-1])$。

第一种情况其实就是无环情况下的整数数组的非空子数组最大和问题，跟「[53. 最大子序和](https://leetcode.cn/problems/maximum-subarray/)」问题是一致的，我们假设求解结果为 `max_num`。

下来来思考第二种情况，第二种情况下，要使首尾两个子区间的和尽可能的大，则中间的子区间的和应该尽可能的小。

使得中间子区间的和尽可能小的问题，可以转变为求解：整数数组 nums 的非空子数组最小和问题。求解思路跟上边是相似的，只不过最大变为了最小。我们假设求解结果为 `min_num`。

而首尾两个区间和尽可能大的结果为数组 nums 的和减去中间最小子数组和，即 `sum(nums) - min_num`。

 最终的结果就是比较 `sum(nums) - min_num` 和 `max_num`的大小，返回较大值即可。

## 代码

```python
class Solution:
    def maxSubarraySumCircular(self, nums: List[int]) -> int:
        size = len(nums)

        dp_max, dp_min = nums[0], nums[0]
        max_num, min_num = nums[0], nums[0]
        for i in range(1, size):
            dp_max = max(dp_max + nums[i], nums[i])
            dp_min = min(dp_min + nums[i], nums[i])
            max_num = max(dp_max, max_num)
            min_num = min(dp_min, min_num)
        sum_num = sum(nums)
        if max_num < 0:
            return max_num
        return max(sum_num - min_num, max_num)
```

# [0919. 完全二叉树插入器](https://leetcode.cn/problems/complete-binary-tree-inserter/)

- 标签：树、广度优先搜索、设计、二叉树
- 难度：中等

## 题目链接

- [0919. 完全二叉树插入器 - 力扣](https://leetcode.cn/problems/complete-binary-tree-inserter/)

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

# [0921. 使括号有效的最少添加](https://leetcode.cn/problems/minimum-add-to-make-parentheses-valid/)

- 标签：栈、贪心、字符串
- 难度：中等

## 题目链接

- [0921. 使括号有效的最少添加 - 力扣](https://leetcode.cn/problems/minimum-add-to-make-parentheses-valid/)

## 题目大意

**描述**：给定一个括号字符串 `s`，可以在字符串的任何位置插入一个括号。

**要求**：返回为使结果字符串 `s` 有效而必须添加的最少括号数。

**说明**：

- $1 \le s.length \le 1000$。
- `s` 只包含 `'('` 和 `')'` 字符。

只有满足下面几点之一，括号字符串才是有效的：

- 它是一个空字符串，或者
- 它可以被写成 AB （A 与 B 连接）, 其中 A 和 B 都是有效字符串，或者
- 它可以被写作 (A)，其中 A 是有效字符串。

例如，如果 `s = "()))"`，你可以插入一个开始括号为 `"(()))"` 或结束括号为 `"())))"`。

**示例**：

- 示例 1：

```python
输入：s = "())"
输出：1
```

## 解题思路

### 思路 1：贪心算法

为了最终添加的最少括号数，我们应该尽可能将当前能够匹配的括号先进行配对。则剩余的未完成配对的括号数量就是答案。

我们使用变量 `left_cnt` 来记录当前左括号的数量。使用 `res` 来记录添加的最少括号数量。

- 遍历字符串，判断当前字符。
- 如果当前字符为左括号 `(`，则令 `left_cnt` 加 `1`。
- 如果当前字符为右括号 `)`，则令 `left_cnt` 减 `1`。如果 `left_cnt` 减到 `-1`，说明当前有右括号不能完成匹配，则答案数量 `res` 加 `1`，并令 `left_cnt` 重新赋值为 `0`。
- 遍历完之后，令 `res` 加上剩余不匹配的 `left_cnt` 数量。
- 最后输出 `res`。

### 思路 1：贪心算法代码

```python
class Solution:
    def minAddToMakeValid(self, s: str) -> int:
        res = 0
        left_cnt = 0
        for ch in s:
            if ch == '(':
                left_cnt += 1
            elif ch == ')':
                left_cnt -= 1
                if left_cnt == -1:
                    left_cnt = 0
                    res += 1
        res += left_cnt
        return res
```
# [0925. 长按键入](https://leetcode.cn/problems/long-pressed-name/)

- 标签：双指针、字符串
- 难度：简单

## 题目链接

- [0925. 长按键入 - 力扣](https://leetcode.cn/problems/long-pressed-name/)

## 题目大意

**描述**：你的朋友正在使用键盘输入他的名字 $name$。偶尔，在键入字符时，按键可能会被长按，而字符可能被输入 $1$ 次或多次。

现在给定代表名字的字符串 $name$，以及实际输入的字符串 $typed$。

**要求**：检查键盘输入的字符 $typed$。如果它对应的可能是你的朋友的名字（其中一些字符可能被长按），就返回 `True`。否则返回 `False`。

**说明**：

- $1 \le name.length, typed.length \le 1000$。
- $name$ 和 $typed$ 的字符都是小写字母。

**示例**：

- 示例 1：

```python
输入：name = "alex", typed = "aaleex"
输出：true
解释：'alex' 中的 'a' 和 'e' 被长按。
```

- 示例 2：

```python
输入：name = "saeed", typed = "ssaaedd"
输出：false
解释：'e' 一定需要被键入两次，但在 typed 的输出中不是这样。
```

## 解题思路

### 思路 1：分离双指针

这道题目的意思是在 $typed$ 里边匹配 $name$，同时要考虑字符重复问题，以及不匹配的情况。可以使用分离双指针来做。具体做法如下：

1. 使用两个指针 $left\underline{\hspace{0.5em}}1$、$left\underline{\hspace{0.5em}}2$，$left\underline{\hspace{0.5em}}1$ 指向字符串 $name$ 开始位置，$left\underline{\hspace{0.5em}}2$ 指向字符串 $type$ 开始位置。
2. 如果 $name[left\underline{\hspace{0.5em}}1] == name[left\underline{\hspace{0.5em}}2]$，则将 $left\underline{\hspace{0.5em}}1$、$left\underline{\hspace{0.5em}}2$ 同时右移。
3. 如果 $nmae[left\underline{\hspace{0.5em}}1] \ne name[left\underline{\hspace{0.5em}}2]$，则：
   1. 如果 $typed[left\underline{\hspace{0.5em}}2]$ 和前一个位置元素 $typed[left\underline{\hspace{0.5em}}2 - 1]$ 相等，则说明出现了重复元素，将 $left\underline{\hspace{0.5em}}2$ 右移，过滤重复元素。
   2. 如果 $typed[left\underline{\hspace{0.5em}}2]$ 和前一个位置元素 $typed[left\underline{\hspace{0.5em}}2 - 1]$ 不等，则说明出现了多余元素，不匹配。直接返回 `False` 即可。

4. 当 $left\underline{\hspace{0.5em}}1 == len(name)$ 或者 $left\underline{\hspace{0.5em}}2 == len(typed)$ 时跳出循环。然后过滤掉 $typed$ 末尾的重复元素。
5. 最后判断，如果 $left\underline{\hspace{0.5em}}1 == len(name)$ 并且 $left\underline{\hspace{0.5em}}2 == len(typed)$，则说明匹配，返回 `True`，否则返回 `False`。

### 思路 1：代码

```python
class Solution:
    def isLongPressedName(self, name: str, typed: str) -> bool:
        left_1, left_2 = 0, 0

        while left_1 < len(name) and left_2 < len(typed):
            if name[left_1] == typed[left_2]:
                left_1 += 1
                left_2 += 1
            elif left_2 > 0 and typed[left_2 - 1] == typed[left_2]:
                left_2 += 1
            else:
                return False
        while 0 < left_2 < len(typed) and typed[left_2] == typed[left_2 - 1]:
            left_2 += 1

        if left_1 == len(name) and left_2 == len(typed):
            return True
        else:
            return False
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(n + m)$。其中 $n$、$m$ 分别为字符串 $name$、$typed$ 的长度。
- **空间复杂度**：$O(1)$。

# [0932. 漂亮数组](https://leetcode.cn/problems/beautiful-array/)

- 标签：数组、数学、分治
- 难度：中等

## 题目链接

- [0932. 漂亮数组 - 力扣](https://leetcode.cn/problems/beautiful-array/)

## 题目大意

**描述**：给定一个整数 $n$。

**要求**：返回长度为 $n$ 的任一漂亮数组。

**说明**：

- **漂亮数组**（长度为 $n$ 的数组 $nums$ 满足下述条件）：
  - $nums$ 是由范围 $[1, n]$ 的整数组成的一个排列。
  - 对于每个 $0 \le i < j < n$，均不存在下标 $k$（$i < k < j$）使得 $2 \times nums[k] == nums[i] + nums[j]$。
- $1 \le n \le 1000$。
- 本题保证对于给定的 $n$ 至少存在一个有效答案。

**示例**：

- 示例 1：

```python
输入：n = 4
输出：[2,1,4,3]
```

- 示例 2：

```python
输入：n = 5
输出：[3,1,2,5,4]
```

## 解题思路

### 思路 1：分治算法

根据题目要求，我们可以得到以下信息：

1. 题目要求 $2 \times nums[k] == nums[i] + nums[j], (0 \le i < k < j < n)$ 不能成立，可知：等式左侧必为偶数，只要右侧和为奇数则等式不成立。
2. 已知：奇数 + 偶数 = 奇数，则令 $nums[i]$ 和 $nums[j]$ 其中一个为奇数，另一个为偶数，即可保证 $nums[i] + nums[j]$ 一定为奇数。这里我们不妨令 $nums[i]$ 为奇数，令 $nums[j]$ 为偶数。
3. 如果数组 $nums$ 是漂亮数组，那么对数组 $nums$ 的每一位元素乘以一个常数或者加上一个常数之后，$nums$ 仍是漂亮数组。
   - 即如果 $[a_1, a_2, ..., a_n]$ 是一个漂亮数组，那么 $[k \times a_1 + b, k \times a_2 + b, ..., k \times a_n + b]$ 也是漂亮数组。

那么，我们可以按照下面的规则构建长度为 $n$ 的漂亮数组。

1. 当 $n = 1$ 时，返回 $[1]$。此时数组 $nums$ 中仅有 $1$ 个元素，并且满足漂亮数组的条件。
2. 当 $n > 1$ 时，我们将 $nums$ 分解为左右两个部分：`left_nums`、`right_nums`。如果左右两个部分满足：
   1. 数组 `left_nums` 中元素全为奇数（可以通过 `nums[i] * 2 - 1` 将 `left_nums` 中元素全部映射为奇数）。
   2. 数组 `right_nums` 中元素全为偶数（可以通过 `nums[i] * 2` 将 `right_nums` 中元素全部映射为偶数）。
   3. `left_nums` 和 `right_nums` 都是漂亮数组。
3. 那么 `left_nums + right_nums` 构成的数组一定也是漂亮数组，即 $nums$ 为漂亮数组，将 $nums$ 返回即可。

### 思路 1：代码

```python
class Solution:
    def beautifulArray(self, n: int) -> List[int]:
        if n == 1:
            return [1]

        nums = [0 for _ in range(n)]
        left_cnt = (n + 1) // 2
        right_cnt = n - left_cnt
        left_nums = self.beautifulArray(left_cnt)
        right_nums = self.beautifulArray(right_cnt)

        for i in range(left_cnt):
            nums[i] = 2 * left_nums[i] - 1
        
        for i in range(right_cnt):
            nums[left_cnt + i] = 2 * right_nums[i]
        
        return nums
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(n \times \log n)$，其中 $n$ 为数组 $nums$ 的长度。
- **空间复杂度**：$O(n \times \log n)$。
# [0933. 最近的请求次数](https://leetcode.cn/problems/number-of-recent-calls/)

- 标签：设计、队列、数据流
- 难度：简单

## 题目链接

- [0933. 最近的请求次数 - 力扣](https://leetcode.cn/problems/number-of-recent-calls/)

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

# [0935. 骑士拨号器](https://leetcode.cn/problems/knight-dialer/)

- 标签：动态规划
- 难度：中等

## 题目链接

- [0935. 骑士拨号器 - 力扣](https://leetcode.cn/problems/knight-dialer/)

## 题目大意

**描述**：象棋骑士可以垂直移动两个方格，水平移动一个方格，或者水平移动两个方格，垂直移动一个方格（两者都形成一个 $L$ 的形状），如下图所示。

![](https://assets.leetcode.com/uploads/2020/08/18/chess.jpg)

现在我们有一个象棋其实和一个电话垫，如下图所示，骑士只能站在一个数字单元格上（$0 \sim 9$）。

![](https://assets.leetcode.com/uploads/2020/08/18/phone.jpg)

现在给定一个整数 $n$。

**要求**：返回我们可以拨多少个长度为 $n$ 的不同电话号码。因为答案可能很大，所以最终答案需要对 $10^9 + 7$ 进行取模。

**说明**：

- 可以将骑士放在任何数字单元格上，然后执行 $n - 1$ 次移动来获得长度为 $n$ 的电话号码。
- $1 \le n \le 5000$。

**示例**：

- 示例 1：

```python
输入：n = 1
输出：10
解释：我们需要拨一个长度为1的数字，所以把骑士放在10个单元格中的任何一个数字单元格上都能满足条件。
```

- 示例 2：

```python
输入：n = 2
输出：20
解释：我们可以拨打的所有有效号码为[04, 06, 16, 18, 27, 29, 34, 38, 40, 43, 49, 60, 61, 67, 72, 76, 81, 83, 92, 94]
```

## 解题思路

### 思路 1：动态规划

根据象棋骑士的跳跃规则，以及电话键盘的样式，我们可以预先处理一下象棋骑士当前位置与下一步能跳跃到的位置关系，将其存入哈希表中，方便查询。

接下来我们可以用动态规划的方式，计算出跳跃 $n - 1$ 次总共能得到多少个长度为 $n$ 的不同电话号码。

###### 1. 划分阶段

按照步数、所处数字位置进行阶段划分。

###### 2. 定义状态

定义状态 $dp[i][v]$ 表示为：第 $i$ 步到达键位 $u$ 总共能到的长度为 $i + 1$ 的不同电话号码个数。

###### 3. 状态转移方程

第 $i$ 步到达键位 $v$ 所能得到的不同电话号码个数，取决于 $i - 1$ 步中所有能到达 $v$ 的键位 $u$ 的不同电话号码个数总和。

呢状态转移方程为：$dp[i][v] = \sum dp[i - 1][u]$（可以从 $u$ 跳到 $v$）。

###### 4. 初始条件

- 第 $0$ 步（位于开始位置）所能得到的电话号码个数为 $1$，因为开始时可以将骑士放在任何数字单元格上，所以所有的 $dp[0][v] = 1$。

###### 5. 最终结果

根据我们之前定义的状态，$dp[i][v]$ 表示为：第 $i$ 步到达键位 $u$ 总共能到的长度为 $i + 1$ 的不同电话号码个数。 所以最终结果为第 $n - 1$ 行所有的 $dp[n - 1][v]$ 的总和。

###  思路 1：代码

```python
class Solution:
    def knightDialer(self, n: int) -> int:
        graph = {
            0: [4, 6],
            1: [6, 8],
            2: [7, 9],
            3: [4, 8],
            4: [0, 3, 9],
            5: [],
            6: [0, 1, 7],
            7: [2, 6],
            8: [1, 3],
            9: [2, 4]
        }

        MOD = 10 ** 9 + 7
        dp = [[0 for _ in range(10)] for _ in range(n)]
        for v in range(10):
            dp[0][v] = 1

        for i in range(1, n):
            for u in range(10):
                for v in graph[u]:
                    dp[i][v] = (dp[i][v] + dp[i - 1][u]) % MOD
        
        return sum(dp[n - 1]) % MOD
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(n \times 10)$，其中 $n$ 为给定整数。
- **空间复杂度**：$O(n \times 10)$。

# [0938. 二叉搜索树的范围和](https://leetcode.cn/problems/range-sum-of-bst/)

- 标签：树、深度优先搜索、二叉搜索树、二叉树
- 难度：简单

## 题目链接

- [0938. 二叉搜索树的范围和 - 力扣](https://leetcode.cn/problems/range-sum-of-bst/)

## 题目大意

给定一个二叉搜索树，和一个范围 [low, high]。求范围 [low, high] 之间所有节点的值的和。

## 解题思路

二叉搜索树的定义：

- 若左子树不为空，则左子树上所有节点值均小于它的根节点值；
- 若右子树不为空，则右子树上所有节点值均大于它的根节点值；
- 任意节点的左、右子树也分别为二叉搜索树。

这道题求解 [low, high] 之间所有节点的值的和，需要递归求解。

- 当前节点为 None 时返回 0；
- 当前节点值 val > high 时，则返回左子树之和；
- 当前节点值 val < low 时，则返回右子树之和；
- 当前节点 val <= high，且 val >= low 时，则返回当前节点值 + 左子树之和 + 右子树之和。

## 代码

```python
class Solution:
    def rangeSumBST(self, root: TreeNode, low: int, high: int) -> int:
        if not root:
            return 0
        if root.val > high:
            return self.rangeSumBST(root.left, low, high)
        if root.val < low:
            return self.rangeSumBST(root.right, low, high)
        return root.val + self.rangeSumBST(root.left, low, high) + self.rangeSumBST(root.right, low, high)
```

# [0946. 验证栈序列](https://leetcode.cn/problems/validate-stack-sequences/)

- 标签：栈、数组、模拟
- 难度：中等

## 题目链接

- [0946. 验证栈序列 - 力扣](https://leetcode.cn/problems/validate-stack-sequences/)

## 题目大意

**描述**：给定两个整数序列 `pushed` 和 `popped`，每个序列中的值都不重复。

**要求**：如果第一个序列为空栈的压入顺序，而第二个序列 `popped` 为该栈的压出序列，则返回 `True`，否则返回 `False`。

**说明**：

- $1 \le pushed.length \le 1000$。
- $0 \le pushed[i] \le 1000$。
- $pushed$ 的所有元素互不相同。
- $popped.length == pushed.length$。
- $popped$ 是 $pushed$ 的一个排列。

**示例**：

- 示例 1：

```python
输入：pushed = [1,2,3,4,5], popped = [4,5,3,2,1]
输出：true
解释：我们可以按以下顺序执行：
push(1), push(2), push(3), push(4), pop() -> 4,
push(5), pop() -> 5, pop() -> 3, pop() -> 2, pop() -> 1
```

- 示例 2：

```python
输入：pushed = [1,2,3,4,5], popped = [4,3,5,1,2]
输出：false
解释：1 不能在 2 之前弹出。
```

## 解题思路

### 思路 1：栈

借助一个栈来模拟压入、压出的操作。检测最后是否能模拟成功。

### 思路 1：代码

```python
class Solution:
    def validateStackSequences(self, pushed: List[int], popped: List[int]) -> bool:
        stack = []
        index = 0
        for item in pushed:
            stack.append(item)
            while (stack and stack[-1] == popped[index]):
                stack.pop()
                index += 1

        return len(stack) == 0
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(n)$。
- **空间复杂度**：$O(n)$。

# [0947. 移除最多的同行或同列石头](https://leetcode.cn/problems/most-stones-removed-with-same-row-or-column/)

- 标签：深度优先搜索、并查集、图
- 难度：中等

## 题目链接

- [0947. 移除最多的同行或同列石头 - 力扣](https://leetcode.cn/problems/most-stones-removed-with-same-row-or-column/)

## 题目大意

**描述**：二维平面中有 $n$ 块石头，每块石头都在整数坐标点上，且每个坐标点上最多只能有一块石头。如果一块石头的同行或者同列上有其他石头存在，那么就可以移除这块石头。

给你一个长度为 $n$ 的数组 $stones$ ，其中 $stones[i] = [xi, yi]$ 表示第 $i$ 块石头的位置。

**要求**：返回可以移除的石子的最大数量。

**说明**：

- $1 \le stones.length \le 1000$。
- $0 \le xi, yi \le 10^4$。
- 不会有两块石头放在同一个坐标点上。

**示例**：

- 示例 1：

```python
输入：stones = [[0,0],[0,1],[1,0],[1,2],[2,1],[2,2]]
输出：5
解释：一种移除 5 块石头的方法如下所示：
1. 移除石头 [2,2] ，因为它和 [2,1] 同行。
2. 移除石头 [2,1] ，因为它和 [0,1] 同列。
3. 移除石头 [1,2] ，因为它和 [1,0] 同行。
4. 移除石头 [1,0] ，因为它和 [0,0] 同列。
5. 移除石头 [0,1] ，因为它和 [0,0] 同行。
石头 [0,0] 不能移除，因为它没有与另一块石头同行/列。
```

- 示例 2：

```python
输入：stones = [[0,0],[0,2],[1,1],[2,0],[2,2]]
输出：3
解释：一种移除 3 块石头的方法如下所示：
1. 移除石头 [2,2] ，因为它和 [2,0] 同行。
2. 移除石头 [2,0] ，因为它和 [0,0] 同列。
3. 移除石头 [0,2] ，因为它和 [0,0] 同行。
石头 [0,0] 和 [1,1] 不能移除，因为它们没有与另一块石头同行/列。
```

## 解题思路

### 思路 1：并查集

题目「求最多可以移走的石头数目」也可以换一种思路：「求最少留下的石头数目」。

- 如果两个石头 $A$、$B$ 处于同一行或者同一列，我们就可以删除石头 $A$  或 $B$，最少留下 $1$ 个石头。
- 如果三个石头 $A$、$B$、$C$，其中 $A$、$B$ 处于同一行，$B$、$C$ 处于同一列，则我们可以先删除石头 $A$，再删除石头 $C$，最少留下 $1$ 个石头。
- 如果有 $n$ 个石头，其中每个石头都有一个同行或者同列的石头，则我们可以将 $n - 1$ 个石头都删除，最少留下 $1$ 个石头。

通过上面的分析，我们可以利用并查集，将同行、同列的石头都加入到一个集合中。这样「最少可以留下的石头」就是并查集中集合的个数。

则答案为：**最多可以移走的石头数目 = 所有石头个数 - 最少可以留下的石头（并查集的集合个数）**。

因为石子坐标是二维的，在使用并查集的时候要区分横纵坐标，因为 $0 <= xi, yi <= 10^4$，可以取 $n = 10010$，将纵坐标映射到 $[n, n + 10000]$ 的范围内，这样就可以得到所有节点的标号。

最后计算集合个数，可以使用 set 集合去重，然后统计数量。

整体步骤如下：

1. 定义一个 $10010 \times 2$ 大小的并查集。
2. 遍历每块石头的横纵坐标：
   1. 将纵坐标映射到 $[10010, 10010 + 10000]$ 的范围内。
   2. 然后将当前石头的横纵坐标相连接（加入到并查集中）。
3. 建立一个 set 集合，查找每块石头横坐标所在集合对应的并查集编号，将编号加入到 set 集合中。
4. 最后，返回「所有石头个数 - 并查集集合个数」即为答案。

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
    def removeStones(self, stones: List[List[int]]) -> int:
        size = len(stones)
        n = 10010
        union_find = UnionFind(n * 2)
        for i in range(size):
            union_find.union(stones[i][0], stones[i][1] + n)

        stones_set = set()
        for i in range(size):
            stones_set.add(union_find.find(stones[i][0]))

        return size - len(stones_set)
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(n \times \alpha(n))$。其中 $n$ 是石子个数。$\alpha$ 是反 Ackerman 函数。
- **空间复杂度**：$O(n)$。# [0953. 验证外星语词典](https://leetcode.cn/problems/verifying-an-alien-dictionary/)

- 标签：数组、哈希表、字符串
- 难度：简单

## 题目链接

- [0953. 验证外星语词典 - 力扣](https://leetcode.cn/problems/verifying-an-alien-dictionary/)

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

# [0958. 二叉树的完全性检验](https://leetcode.cn/problems/check-completeness-of-a-binary-tree/)

- 标签：树、广度优先搜索、二叉树
- 难度：中等

## 题目链接

- [0958. 二叉树的完全性检验 - 力扣](https://leetcode.cn/problems/check-completeness-of-a-binary-tree/)

## 题目大意

**描述**：给定一个二叉树的根节点 `root`。

**要求**：判断该二叉树是否是一个完全二叉树。

**说明**：

- **完全二叉树**：
- 树的结点数在范围 $[1, 100]$ 内。
- $1 \le Node.val \le 1000$。

**示例**：

- 示例 1：

![](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2018/12/15/complete-binary-tree-1.png)

```python
输入：root = [1,2,3,4,5,6]
输出：true
解释：最后一层前的每一层都是满的（即，结点值为 {1} 和 {2,3} 的两层），且最后一层中的所有结点（{4,5,6}）都尽可能地向左。
```

- 示例 2：

![](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2018/12/15/complete-binary-tree-2.png)

```python
输入：root = [1,2,3,4,5,null,7]
输出：false
解释：值为 7 的结点没有尽可能靠向左侧。
```

## 解题思路

### 思路 1：广度优先搜索

对于一个完全二叉树，按照「层序遍历」的顺序进行广度优先搜索，在遇到第一个空节点之后，整个完全二叉树的遍历就已结束了。不应该在后续遍历过程中再次出现非空节点。

如果在遍历过程中在遇到第一个空节点之后，又出现了非空节点，则该二叉树不是完全二叉树。

利用这一点，我们可以在广度优先搜索的过程中，维护一个布尔变量 `is_empty` 用于标记是否遇见了空节点。

### 思路 1：代码

```python
class Solution:
    def isCompleteTree(self, root: Optional[TreeNode]) -> bool:
        if not root:
            return False

        queue = collections.deque([root])
        is_empty = False
        while queue:
            size = len(queue)
            for _ in range(size):
                cur = queue.popleft()
                if not cur:
                    is_empty = True
                else:
                    if is_empty:
                        return False
                    queue.append(cur.left)
                    queue.append(cur.right)
        return True
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(n)$，其中 $n$ 为二叉树的节点数。
- **空间复杂度**：$O(n)$。
# [0959. 由斜杠划分区域](https://leetcode.cn/problems/regions-cut-by-slashes/)

- 标签：深度优先搜索、广度优先搜索、并查集、图
- 难度：中等

## 题目链接

- [0959. 由斜杠划分区域 - 力扣](https://leetcode.cn/problems/regions-cut-by-slashes/)

## 题目大意

**描述**：在由 $1 \times 1$ 方格组成的 $n \times n$ 网格 $grid$ 中，每个 $1 \times 1$ 方块由 `'/'`、`'\'` 或 `' '` 构成。这些字符会将方块划分为一些共边的区域。

现在给定代表网格的二维数组 $grid$。

**要求**：返回区域的数目。

**说明**：

- 反斜杠字符是转义的，因此 `'\'` 用 `'\\'` 表示。
- $n == grid.length == grid[i].length$。
- $1 \le n \le 30$。
- $grid[i][j]$ 是 `'/'`、`'\'` 或 `' '`。

**示例**：

- 示例 1：

![](https://assets.leetcode.com/uploads/2018/12/15/1.png)

```python
输入：grid = [" /","/ "]
输出：2
```

- 示例 2：

![](https://assets.leetcode.com/uploads/2018/12/15/4.png)

```python
输入：grid = ["/\\","\\/"]
输出：5
解释：回想一下，因为 \ 字符是转义的，所以 "/\\" 表示 /\，而 "\\/" 表示 \/。
```

## 解题思路

### 思路 1：并查集

我们把一个 $1 \times 1$ 的单元格分割成逻辑上的 $4$ 个部分，则 `' '`、`'/'`、`'\'`  可以将 $1 \times 1$ 的方格分割为以下三种形态：

![](http://qcdn.itcharge.cn/images/20210827142447.png)

在进行遍历的时候，需要将联通的部分进行合并，并统计出联通的块数。这就需要用到了并查集。

遍历二维数组 $gird$，然后在「单元格内」和「单元格间」进行合并。

现在我们为单元格的每个小三角部分按顺时针方向都编上编号，起始位置为左边。然后单元格间的编号按照从左到右，从上到下的位置进行编号，如下图所示：

![](http://qcdn.itcharge.cn/images/20210827143836.png)

假设当前单元格的起始位置为 $index$，则合并策略如下：

- 如果是单元格内：
  - 如果是空格：合并 $index$、$index + 1$、$index + 2$、$index + 3$。
  - 如果是 `'/'`：合并 $index$ 和 $index + 1$，合并 $index + 2$ 和 $index + 3$。
  - 如果是 `'\'`：合并 $index$ 和 $index + 3$，合并 $index + 1$ 和 $index + 2$。
- 如果是单元格间，则向下向右进行合并：
  - 向下：合并 $index + 3$ 和 $index + 4 * size + 1 $。
  - 向右：合并 $index + 2$ 和 $index + 4$。

最后合并完成之后，统计并查集中连通分量个数即为答案。

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
    def regionsBySlashes(self, grid: List[str]) -> int:
        size = len(grid)
        m = 4 * size * size
        union_find = UnionFind(m)
        for i in range(size):
            for j in range(size):
                index = 4 * (i * size + j)
                ch = grid[i][j]
                if ch == '/':
                    union_find.union(index, index + 1)
                    union_find.union(index + 2, index + 3)
                elif ch == '\\':
                    union_find.union(index, index + 3)
                    union_find.union(index + 1, index + 2)
                else:
                    union_find.union(index, index + 1)
                    union_find.union(index + 1, index + 2)
                    union_find.union(index + 2, index + 3)
                if j + 1 < size:
                    union_find.union(index + 2, index + 4)
                if i + 1 < size:
                    union_find.union(index + 3, index + 4 * size + 1)

        return union_find.count
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(n^2 \times \alpha(n^2))$，其中 $\alpha$ 是反 `Ackerman` 函数。
- **空间复杂度**：$O(n^2)$。

# [0968. 监控二叉树](https://leetcode.cn/problems/binary-tree-cameras/)

- 标签：树、深度优先搜索、动态规划、二叉树
- 难度：困难

## 题目链接

- [0968. 监控二叉树 - 力扣](https://leetcode.cn/problems/binary-tree-cameras/)

## 题目大意

给定一个二叉树，需要在树的节点上安装摄像头。节点上的每个摄影头都可以监视其父节点、自身及其直接子节点。

计算监控树的所有节点所需的最小摄像头数量。

- 示例 1：



![img](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2018/12/29/bst_cameras_01.png)

```
输入：[0,0,null,0,0]
输出：1
解释：如图所示，一台摄像头足以监控所有节点。
```

- 示例 2：

![img](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2018/12/29/bst_cameras_02.png)

```
输入：[0,0,null,0,null,0,null,null,0]
输出：2
解释：需要至少两个摄像头来监视树的所有节点。 上图显示了摄像头放置的有效位置之一。
```

## 解题思路

根据题意可知，一个摄像头的有效范围为 3 层：父节点、自身及其直接子节点。而约是下层的节点就越多，所以摄像头应该优先满足下层节点。可以使用后序遍历的方式遍历二叉树的节点，这样就可以优先遍历叶子节点。

对于每个节点，利用贪心思想，可以确定三种状态：

- 第一种状态：该节点无覆盖
- 第二种状态：该节点已经装上了摄像头
- 第三种状态：该节点已经覆盖

为了让摄像头数量最少，我们要尽量让叶⼦节点的⽗节点安装摄像头，这样才能摄像头的数量最少。对此我们应当分析当前节点和左右两侧子节点的覆盖情况。

先来考虑空节点，空节点应该算作已经覆盖状态。

再来考虑左右两侧子覆盖情况：

- 如果左节点或者右节点都无覆盖，则当前节点需要装上摄像头，答案 res 需要 + 1。
- 如果左节点已经覆盖或者右节点已经装上了摄像头，则当前节点已经覆盖。
- 如果左节点右节点都已经覆盖，则当前节点无覆盖。

根据以上条件就可以写出对应的后序遍历代码。

## 代码

```python
class Solution:
    res = 0
    def traversal(self, cur: TreeNode) -> int:
        if not cur:
            return 3

        left = self.traversal(cur.left)
        right = self.traversal(cur.right)

        if left == 1 or right == 1:
            self.res += 1
            return 2

        if left == 2 or right == 2:
            return 3

        if left == 3 and right == 3:
            return 1
        return -1

    def minCameraCover(self, root: TreeNode) -> int:
        self.res = 0
        if self.traversal(root) == 1:
            self.res += 1
        return self.res
```

# [0973. 最接近原点的 K 个点](https://leetcode.cn/problems/k-closest-points-to-origin/)

- 标签：几何、数组、数学、分治、快速选择、排序、堆（优先队列）
- 难度：中等

## 题目链接

- [0973. 最接近原点的 K 个点 - 力扣](https://leetcode.cn/problems/k-closest-points-to-origin/)

## 题目大意

给定一个由由平面上的点组成的列表 `points`，再给定一个整数 `K`。

要求：从中找出 `K` 个距离原点` (0, 0)` 最近的点。（这里，平面上两点之间的距离是欧几里德距离。）可以按任何顺序返回答案。除了点坐标的顺序之外，答案确保是唯一的。

## 解题思路

1. 使用二叉堆构建优先队列，优先级为距离原点的距离。此时堆顶元素即为距离原点最近的元素。
2. 将堆顶元素加入到答案数组中，进行出队操作。时间复杂度 $O(log{n})$。
   - 出队操作：交换堆顶元素与末尾元素，将末尾元素已移出堆。继续调整大顶堆。
3. 不断重复第 2 步，直到 `K` 次结束。

## 代码

```python
class Heapq:
    def compare(self, a, b):
        dist_a = a[0] * a[0] + a[1] * a[1]
        dist_b = b[0] * b[0] + b[1] * b[1]
        if dist_a < dist_b:
            return -1
        elif dist_a == dist_b:
            return 0
        else:
            return 1
    # 堆调整方法：调整为小顶堆
    def heapAdjust(self, nums: [int], index: int, end: int):
        left = index * 2 + 1
        right = left + 1
        while left <= end:
            # 当前节点为非叶子结点
            max_index = index
            if self.compare(nums[left], nums[max_index]) == -1:
                max_index = left
            if right <= end and self.compare(nums[right], nums[max_index]) == -1:
                max_index = right
            if index == max_index:
                # 如果不用交换，则说明已经交换结束
                break
            nums[index], nums[max_index] = nums[max_index], nums[index]
            # 继续调整子树
            index = max_index
            left = index * 2 + 1
            right = left + 1

    # 将数组构建为二叉堆
    def heapify(self, nums: [int]):
        size = len(nums)
        # (size - 2) // 2 是最后一个非叶节点，叶节点不用调整
        for i in range((size - 2) // 2, -1, -1):
            # 调用调整堆函数
            self.heapAdjust(nums, i, size - 1)

    # 入队操作
    def heappush(self, nums: list, value):
        nums.append(value)
        size = len(nums)
        i = size - 1
        # 寻找插入位置
        while (i - 1) // 2 >= 0:
            cur_root = (i - 1) // 2
            # value 大于当前根节点，则插入到当前位置
            if self.compare(nums[cur_root], value) == -1:
                break
            # 继续向上查找
            nums[i] = nums[cur_root]
            i = cur_root
        # 找到插入位置或者到达根位置，将其插入
        nums[i] = value

    # 出队操作
    def heappop(self, nums: list) -> int:
        size = len(nums)
        nums[0], nums[-1] = nums[-1], nums[0]
        # 得到最小值（堆顶元素）然后调整堆
        top = nums.pop()
        if size > 0:
            self.heapAdjust(nums, 0, size - 2)

        return top

    # 升序堆排序
    def heapSort(self, nums: [int]):
        self.heapify(nums)
        size = len(nums)
        for i in range(size):
            nums[0], nums[size - i - 1] = nums[size - i - 1], nums[0]
            self.heapAdjust(nums, 0, size - i - 2)
        return nums

class Solution:
    def kClosest(self, points: List[List[int]], k: int) -> List[List[int]]:
        heap = Heapq()
        queue = []
        for point in points:
            heap.heappush(queue, point)

        res = []
        for i in range(k):
            res.append(heap.heappop(queue))

        return res
```

# [974. 和可被 K 整除的子数组](https://leetcode.cn/problems/subarray-sums-divisible-by-k/)

- 标签：数组、哈希表、前缀和
- 难度：中等

## 题目链接

- [974. 和可被 K 整除的子数组 - 力扣](https://leetcode.cn/problems/subarray-sums-divisible-by-k/)

## 题目大意

给定一个整数数组 `nums` 和一个整数 `k`。

要求：返回其中元素之和可被 `k` 整除的（连续、非空）子数组的数目。

## 解题思路

先考虑暴力计算子数组和，外层两重循环，遍历所有连续子数组，然后最内层再计算一下子数组的和。部分代码如下：

```python
for i in range(len(nums)):
    for j in range(i + 1):
        sum = countSum(i, j)
```

这样下来时间复杂度就是 $O(n^3)$ 了。下一步是想办法降低时间复杂度。

先用一重循环遍历数组，计算出数组 `nums` 中前 i 个元素的和（前缀和），保存到一维数组 `pre_sum` 中，那么对于任意 `[j..i]` 的子数组 的和为 `pre_sum[i] - pre_sum[j - 1]`。这样计算子数组和的时间复杂度降为了 $O(1)$。总体时间复杂度为 $O(n^2)$。

由于我们只关心和为 `k` 出现的次数，不关心具体的解，可以使用哈希表来加速运算。

`pre_sum[i]` 的定义是前 `i` 个元素和，则 `[j..i]` 子数组和可以被 `k` 整除可以转换为：`（pre_sum[i] - pre_sum[j - 1]）% k == 0`。再转换一下：`pre_sum[i] % k == pre_sum[j - 1] % k`。

所以，我们只需要统计满足 `pre_sum[i] % k == pre_sum[j - 1] % k` 条件的组合个数。具体做法如下：

使用 `pre_sum` 变量记录前缀和（代表 `pre_sum[i]`）。使用哈希表 `pre_dic` 记录 `pre_sum[i] % k` 出现的次数。键值对为 `pre_sum[i] : count`。

- 从左到右遍历数组，计算当前前缀和并对 `k`  取余，即 `pre_sum = (pre_sum + nums[i]) % k`。
  - 如果 `pre_sum` 在哈希表中，则答案个数累加上 `pre_dic[pre_sum]`。同时 `pre_sum` 个数累加 1，即 `pre_dic[pre_sum] += 1`。
  - 如果 `pre_sum` 不在哈希表中，则 `pre_sum` 个数记为 1，即 `pre_dic[pre_sum] += 1`。
- 最后输出答案个数。

## 代码

```python
class Solution:
    def subarraysDivByK(self, nums: List[int], k: int) -> int:
        pre_sum = 0
        ans = 0
        nums_dict = {0: 1}
        for i in range(len(nums)):
            pre_sum = (pre_sum + nums[i]) % k
            if pre_sum < 0:
                pre_sum += k
            if pre_sum in nums_dict:
                ans += nums_dict[pre_sum]
                nums_dict[pre_sum] += 1
            else:
                nums_dict[pre_sum] = 1
        return ans
```

# [0976. 三角形的最大周长](https://leetcode.cn/problems/largest-perimeter-triangle/)

- 标签：贪心、数组、数学、排序
- 难度：简单

## 题目链接

- [0976. 三角形的最大周长 - 力扣](https://leetcode.cn/problems/largest-perimeter-triangle/)

## 题目大意

**描述**：给定一些由正数（代表长度）组成的数组 `nums`。

**要求**：返回由其中 `3` 个长度组成的、面积不为 `0` 的三角形的最大周长。如果不能形成任何面积不为 `0` 的三角形，则返回 `0`。

**说明**：

- $3 \le nums.length \le 10^4$。
- $1 \le nums[i] \le 10^6$。

**示例**：

- 示例 1：

```python
输入：nums = [2,1,2]
输出：5
解释：长度为 2, 1, 2 的边组成的三角形周长为 5，为最大周长
```

## 解题思路

### 思路 1：

要想三角形的周长最大，则每一条边都要尽可能的长，并且还要满足三角形的边长条件，即 `a + b > c`，其中 `a`、`b`、`c` 分别是三角形的 `3` 条边长。

所以，我们可以先对所有边长进行排序。然后倒序枚举最长边 `nums[i]`，判断前两个边长相加是否大于最长边，即 `nums[i - 2] + nums[i - 1] > nums[i]`。如果满足，则返回 `3` 条边长的和，否则的话继续枚举最长边。

## 代码

### 思路 1 代码：

```python
class Solution:
    def largestPerimeter(self, nums: List[int]) -> int:
        nums.sort()
        for i in range(len(nums) - 1, 1, -1):
            if nums[i - 2] + nums[i - 1] > nums[i]:
                return nums[i - 2] + nums[i - 1] + nums[i]
        return 0
```

# [0977. 有序数组的平方](https://leetcode.cn/problems/squares-of-a-sorted-array/)

- 标签：数组、双指针、排序
- 难度：简单

## 题目链接

- [0977. 有序数组的平方 - 力扣](https://leetcode.cn/problems/squares-of-a-sorted-array/)

## 题目大意

**描述**：给定一个按「非递减顺序」排序的整数数组 $nums$。

**要求**：返回「每个数字的平方」组成的新数组，要求也按「非递减顺序」排序。

**说明**：

- 要求使用时间复杂度为 $O(n)$ 的算法解决本问题。
- $1 \le nums.length \le 10^4$。
- $-10^4 \le nums[i] \le 10^4$。
- $nums$ 已按非递减顺序排序。

**示例**：

- 示例 1：

```python
输入：nums = [-4,-1,0,3,10]
输出：[0,1,9,16,100]
解释：平方后，数组变为 [16,1,0,9,100]
排序后，数组变为 [0,1,9,16,100]
```

- 示例 2：

```python
输入：nums = [-7,-3,2,3,11]
输出：[4,9,9,49,121]
```

## 解题思路

### 思路 1：对撞指针

原数组是按「非递减顺序」排序的，可能会存在负数元素。但是无论是否存在负数，数字的平方最大值一定在原数组的两端。题目要求返回的新数组也要按照「非递减顺序」排序。那么，我们可以利用双指针，从两端向中间移动，然后不断将数的平方最大值填入数组。具体做法如下：

- 使用两个指针 $left$、$right$。$left$ 指向数组第一个元素位置，$right$ 指向数组最后一个元素位置。再定义 $index = len(nums) - 1$ 作为答案数组填入顺序的索引值。$res$ 作为答案数组。

- 比较 $nums[left]$ 与 $nums[right]$ 的绝对值大小。大的就是平方最大的的那个数。

  - 如果 $abs(nums[right])$ 更大，则将其填入答案数组对应位置，并令 `right -= 1`。

  - 如果 $abs(nums[left])$ 更大，则将其填入答案数组对应位置，并令 `left += 1`。

  - 令 $index -= 1$。

- 直到 $left == right$，最后将 $nums[left]$ 填入答案数组对应位置。

返回答案数组 $res$。

### 思路 1：代码

```python
class Solution:
    def sortedSquares(self, nums: List[int]) -> List[int]:
        size = len(nums)
        left, right = 0, size - 1
        index = size - 1
        res = [0 for _ in range(size)]

        while left < right:
            if abs(nums[left]) < abs(nums[right]):
                res[index] = nums[right] * nums[right]
                right -= 1
            else:
                res[index] = nums[left] * nums[left]
                left += 1
            index -= 1
        res[index] = nums[left] * nums[left]

        return res
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(n)$，其中 $n$ 为数组 $nums$ 中的元素数量。
- **空间复杂度**：$O(1)$，不考虑最终返回值的空间占用。

### 思路 2：排序算法

可以通过各种排序算法来对平方后的数组进行排序。以快速排序为例，具体步骤如下：

1. 遍历数组，将数组中各个元素变为平方项。
2. 从数组中找到一个基准数。
3. 然后将数组中比基准数大的元素移动到基准数右侧，比他小的元素移动到基准数左侧，从而把数组拆分为左右两个部分。
4. 再对左右两个部分分别重复第 2、3 步，直到各个部分只有一个数，则排序结束。

### 思路 2：代码

```python
import random

class Solution:
    def randomPartition(self, arr: [int], low: int, high: int):
        i = random.randint(low, high)
        arr[i], arr[high] = arr[high], arr[i]
        return self.partition(arr, low, high)

    def partition(self, arr: [int], low: int, high: int):
        i = low - 1
        pivot = arr[high]

        for j in range(low, high):
            if arr[j] <= pivot:
                i += 1
                arr[i], arr[j] = arr[j], arr[i]
        arr[i + 1], arr[high] = arr[high], arr[i + 1]
        return i + 1

    def quickSort(self, arr, low, high):
        if low < high:
            pi = self.randomPartition(arr, low, high)
            self.quickSort(arr, low, pi - 1)
            self.quickSort(arr, pi + 1, high)

        return arr

    def sortedSquares(self, nums: List[int]) -> List[int]:
        for i in range(len(nums)):
            nums[i] = nums[i] * nums[i]

        return self.quickSort(nums, 0, len(nums) - 1)
```

### 思路 2：复杂度分析

- **时间复杂度**：$O(n \log n)$，其中 $n$ 为数组 $nums$ 中的元素数量。
- **空间复杂度**：$O(\log n)$。

# [0978. 最长湍流子数组](https://leetcode.cn/problems/longest-turbulent-subarray/)

- 标签：数组、动态规划、滑动窗口
- 难度：中等

## 题目链接

- [0978. 最长湍流子数组 - 力扣](https://leetcode.cn/problems/longest-turbulent-subarray/)

## 题目大意

**描述**：给定一个数组 $arr$。当 $arr$ 的子数组 $arr[i]$，$arr[i + 1]$，$...$， $arr[j]$ 满足下列条件时，我们称其为湍流子数组：

- 如果 $i \le k < j$，当 $k$ 为奇数时， $arr[k] > arr[k + 1]$，且当 $k$ 为偶数时，$arr[k] < arr[k + 1]$；
- 或如果 $i \le k < j$，当 $k$ 为偶数时，$arr[k] > arr[k + 1]$ ，且当 $k$ 为奇数时，$arr[k] < arr[k + 1]$。
- 也就是说，如果比较符号在子数组中的每个相邻元素对之间翻转，则该子数组是湍流子数组。

**要求**：返回给定数组 $arr$ 的最大湍流子数组的长度。

**说明**：

- $1 \le arr.length \le 4 \times 10^4$。
- $0 \le arr[i] \le 10^9$。

**示例**：

- 示例 1：

```python
输入：arr = [9,4,2,10,7,8,8,1,9]
输出：5
解释：arr[1] > arr[2] < arr[3] > arr[4] < arr[5]
```

- 示例 2：

```python
输入：arr = [4,8,12,16]
输出：2
```

## 解题思路

### 思路 1：快慢指针

湍流子数组实际上像波浪一样，比如 $arr[i - 2] > arr[i - 1] < arr[i] > arr[i + 1] < arr[i + 2]$。所以我们可以使用双指针的做法。具体做法如下：

- 使用两个指针 $left$、$right$。$left$ 指向湍流子数组的左端，$right$ 指向湍流子数组的右端。
- 如果 $arr[right - 1] == arr[right]$，则更新 `left = right`，重新开始计算最长湍流子数组大小。
- 如果 $arr[right - 2] < arr[right - 1] < arr[right]$，此时为递增数组，则 $left$ 从 $right - 1$ 开始重新计算最长湍流子数组大小。
- 如果 $arr[right - 2] > arr[right - 1] > arr[right]$，此时为递减数组，则 $left$ 从 $right - 1$ 开始重新计算最长湍流子数组大小。
- 其他情况（即 $arr[right - 2] < arr[right - 1] > arr[right]$ 或 $arr[right - 2] > arr[right - 1] < arr[right]$）时，不用更新 $left$值。
- 更新最大湍流子数组的长度，并向右移动 $right$。直到 $right \ge len(arr)$ 时，返回答案 $ans$。

### 思路 1：代码

```python
class Solution:
    def maxTurbulenceSize(self, arr: List[int]) -> int:
        left, right = 0, 1
        ans = 1

        while right < len(arr):
            if arr[right - 1] == arr[right]:
                left = right
            elif right != 1 and arr[right - 2] < arr[right - 1] and arr[right - 1] < arr[right]:
                left = right - 1
            elif right != 1 and arr[right - 2] > arr[right - 1] and arr[right - 1] > arr[right]:
                left = right - 1
            ans = max(ans, right - left + 1)
            right += 1

        return ans
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(n)$，其中 $n$ 为数组 $arr$ 中的元素数量。
- **空间复杂度**：$O(1)$。

# [0982. 按位与为零的三元组](https://leetcode.cn/problems/triples-with-bitwise-and-equal-to-zero/)

- 标签：位运算、数组、哈希表
- 难度：困难

## 题目链接

- [0982. 按位与为零的三元组 - 力扣](https://leetcode.cn/problems/triples-with-bitwise-and-equal-to-zero/)

## 题目大意

**描述**：给定一个整数数组 $nums$。

**要求**：返回其中「按位与三元组」的数目。

**说明**：

- **按位与三元组**：由下标 $(i, j, k)$ 组成的三元组，并满足下述全部条件：
  - $0 \le i < nums.length$。
  - $0 \le j < nums.length$。
  - $0 \le k < nums.length$。
  - $nums[i] \text{ \& } nums[j] \text{ \& } nums[k] == 0$ ，其中 $\text{ \& }$ 表示按位与运算符。

- $1 \le nums.length \le 1000$。
- $0 \le nums[i] < 2^{16}$。

**示例**：

- 示例 1：

```python
输入：nums = [2,1,3]
输出：12
解释：可以选出如下 i, j, k 三元组：
(i=0, j=0, k=1) : 2 & 2 & 1
(i=0, j=1, k=0) : 2 & 1 & 2
(i=0, j=1, k=1) : 2 & 1 & 1
(i=0, j=1, k=2) : 2 & 1 & 3
(i=0, j=2, k=1) : 2 & 3 & 1
(i=1, j=0, k=0) : 1 & 2 & 2
(i=1, j=0, k=1) : 1 & 2 & 1
(i=1, j=0, k=2) : 1 & 2 & 3
(i=1, j=1, k=0) : 1 & 1 & 2
(i=1, j=2, k=0) : 1 & 3 & 2
(i=2, j=0, k=1) : 3 & 2 & 1
(i=2, j=1, k=0) : 3 & 1 & 2
```

- 示例 2：

```python
输入：nums = [0,0,0]
输出：27
```

## 解题思路

### 思路 1：枚举

最直接的方法是使用三重循环直接枚举 $(i, j, k)$，然后再判断 $nums[i] \text{ \& } nums[j] \text{ \& } nums[k]$ 是否为 $0$。但是这样做的时间复杂度为 $O(n^3)$。

从题目中可以看出 $nums[i]$ 的值域范围为 $[0, 2^{16}]$，而 $2^{16} = 65536$。所以我们可以按照下面步骤优化时间复杂度：

1. 先使用两重循环枚举 $(i, j)$，计算出 $nums[i] \text{ \& } nums[j]$ 的值，将其存入一个大小为 $2^{16}$ 的数组或者哈希表 $cnts$ 中，并记录每个 $nums[i] \text{ \& } nums[j]$ 值出现的次数。
2. 然后遍历该数组或哈希表，再使用一重循环遍历 $k$，找出所有满足 $nums[k] \text{ \& } x == 0$ 的 $x$，并将其对应数量 $cnts[x]$ 累积到答案 $ans$ 中。
3. 最后返回答案 $ans$ 即可。

### 思路 1：代码

```python
class Solution:
    def countTriplets(self, nums: List[int]) -> int:
        states = 1 << 16
        cnts = [0 for _ in range(states)]

        for num_x in nums:
            for num_y in nums:
                cnts[num_x & num_y] += 1
        
        ans = 0
        for num in nums:
            for x in range(states):
                if num & x == 0:
                    ans += cnts[x]
        
        return ans
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(n^2 + 2^{16} \times n)$，其中 $n$ 为数组 $nums$ 的长度。
- **空间复杂度**：$O(2^{16})$。

### 思路 2：枚举 + 优化

第一步跟思路 1 一样，我们先使用两重循环枚举 $(i, j)$，计算出 $nums[i] \text{ \& } nums[j]$ 的值，将其存入一个大小为 $2^{16}$ 的数组或者哈希表 $cnts$ 中，并记录每个 $nums[i] \text{ \& } nums[j]$ 值出现的次数。

接下来我们对思路 1 中的第二步进行优化，在思路 1 中，我们是通过枚举数组或哈希表的方式得到 $x$ 的，这里我们换一种方法。

使用一重循环遍历 $k$，对于 $nums[k]$，我们先计算出 $nums[k]$ 的补集，即将 $nums[k]$ 与 $2^{16} - 1$（二进制中 $16$ 个 $1$）进行按位异或操作，得到 $nums[k]$ 的补集 $com$。如果 $nums[k] \text{ \& } x == 0$，则 $x$ 一定是 $com$ 的子集。

换句话说，$x$ 中 $1$ 的位置一定与 $nums[k]$ 中 $1$ 的位置不同，如果 $nums[k]$ 中第 $m$ 位为 $1$，则 $x$ 中第 $m$ 位一定为 $0$。

接下来我们通过下面的方式来枚举子集：

1. 定义子集为 $sub$，初始时赋值为 $com$，即：$sub = com$。
2. 令 $sub$ 减 $1$，然后与 $com$ 做按位与操作，得到下一个子集，即：$sub = (sub - 1) \text{ \& } com$。
3. 不断重复第 $2$ 步，直到 $sub$ 为空集时为止。

这种方法能枚举子集的原理是：$sub$ 减 $1$ 会将最低位的 $1$ 改为 $0$，而比这个 $1$ 更低位的 $0$ 都改为了 $1$。此时再与 $com$ 做按位与操作，就会过保留原本高位上的 $1$，滤掉当前最低位的 $1$，并且保留比这个 $1$ 更低位上的原有的 $1$，也就得到嘞下一个子集。

举个例子，比如补集 $com$ 为 $(00010110)_2$：

1. 初始 $sub = (00010110)_2$。
2. 令其减 $1$ 后为 $(00010101)_2$，然后与 $com$ 做按位与操作，得到下一个子集 $sub = (00010100)_2$，即：$(00010101)_2 \text{ \& } (00010110)_2$）。
3. 令其减 $1$ 后为 $(00010011)_2$，然后与 $com$ 做按位与操作，得到下一个子集 $sub = (00010010)_2$，即： $(00010011)_2 \text{ \& } (00010110)_2$。
4. 令其减 $1$ 后为 $(00010001)_2$，然后与 $com$ 做按位与操作，得到下一个子集 $sub = (00010000)_2$，即：$(00010001)_2 \text{ \& } (00010110)_2$。
5. 令其减 $1$ 后为 $(00001111)_2$，然后与 $com$ 做按位与操作，得到下一个子集 $sub = (00000110)_2$，即：$(00001111)_2 \text{ \& } (00010110)_2$。
6. 令其减 $1$ 后为 $(00000101)_2$，然后与 $com$ 做按位与操作，得到下一个子集 $sub = (00000100)_2$，即：$(00000101)_2 \text{ \& } (00010110)_2$。
7. 令其减 $1$ 后为 $(00000011)_2$，然后与 $com$ 做按位与操作，得到下一个子集 $sub = (00000010)_2$，即：$(00000011)_2 \text{ \& } (00010110)_2$。
8. 令其减 $1$ 后为 $(00000001)_2$，然后与 $com$ 做按位与操作，得到下一个子集 $sub = (00000000)_2$，即：$(00000001)_2 \text{ \& } (00010110)_2$。
9. $sub$ 变为了空集。

### 思路 2：代码

```python
class Solution:
    def countTriplets(self, nums: List[int]) -> int:
        states = 1 << 16
        cnts = [0 for _ in range(states)]

        for num_x in nums:
            for num_y in nums:
                cnts[num_x & num_y] += 1
        
        ans = 0
        for num in nums:
            com = num ^ 0xffff			# com: num 的补集
            sub = com					# sub: 子集
            while True:
                ans += cnts[sub]
                if sub == 0:
                    break
                sub = (sub - 1) & com
        
        return ans
```

### 思路 2：复杂度分析

- **时间复杂度**：$O(n^2 + 2^{16} \times n)$，其中 $n$ 为数组 $nums$ 的长度。
- **空间复杂度**：$O(2^{16})$。

## 参考资料

- 【题解】[按位与为零的三元组 - 按位与为零的三元组](https://leetcode.cn/problems/triples-with-bitwise-and-equal-to-zero/solution/an-wei-yu-wei-ling-de-san-yuan-zu-by-lee-gjud/)
- 【题解】[有技巧的枚举 + 常数优化（Python/Java/C++/Go） - 按位与为零的三元组](https://leetcode.cn/problems/triples-with-bitwise-and-equal-to-zero/solution/you-ji-qiao-de-mei-ju-chang-shu-you-hua-daxit/)
# [0990. 等式方程的可满足性](https://leetcode.cn/problems/satisfiability-of-equality-equations/)

- 标签：并查集、图、数组、字符串
- 难度：中等

## 题目链接

- [0990. 等式方程的可满足性 - 力扣](https://leetcode.cn/problems/satisfiability-of-equality-equations/)

## 题目大意

**描述**：给定一个由字符串方程组成的数组 `equations`，每个字符串方程 `equations[i]` 的长度为 `4`，有以下两种形式组成：`a==b` 或 `a!=b`。`a` 和 `b` 是小写字母，表示单字母变量名。

**要求**：判断所有的字符串方程是否能同时满足，如果能同时满足，返回 `True`，否则返回 `False`。

**说明**：

- $1 \le equations.length \le 500$。
- $equations[i].length == 4$。
- $equations[i][0]$ 和 $equations[i][3]$ 是小写字母。
- $equations[i][1]$ 要么是 `'='`，要么是 `'!'`。
- `equations[i][2]` 是 `'='`。

**示例**：

- 示例 1：

```python
输入：["a==b","b!=a"]
输出：False
解释：如果我们指定，a = 1 且 b = 1，那么可以满足第一个方程，但无法满足第二个方程。没有办法分配变量同时满足这两个方程。
```

## 解题思路

### 思路 1：并查集

字符串方程只有 `==` 或者 `!=`，可以考虑将相等的遍历划分到相同集合中，然后再遍历所有不等式方程，看方程的两个变量是否在之前划分的相同集合中，如果在则说明不满足。

这就需要用到并查集，具体操作如下：

- 遍历所有等式方程，将等式两边的单字母变量顶点进行合并。
- 遍历所有不等式方程，检查不等式两边的单字母遍历是不是在一个连通分量中，如果在则返回 `False`，否则继续扫描。如果所有不等式检查都没有矛盾，则返回 `True`。

### 思路 1：并查集代码

```python
class UnionFind:
    def __init__(self, n):                          # 初始化
        self.fa = [i for i in range(n)]             # 每个元素的集合编号初始化为数组 fa 的下标索引
    
    def __find(self, x):                            # 查找元素根节点的集合编号内部实现方法
        while self.fa[x] != x:                      # 递归查找元素的父节点，直到根节点
            self.fa[x] = self.fa[self.fa[x]]        # 隔代压缩优化
            x = self.fa[x]
        return x                                    # 返回元素根节点的集合编号

    def union(self, x, y):                          # 合并操作：令其中一个集合的树根节点指向另一个集合的树根节点
        root_x = self.__find(x)
        root_y = self.__find(y)
        if root_x == root_y:                        # x 和 y 的根节点集合编号相同，说明 x 和 y 已经同属于一个集合
            return False
        
        self.fa[root_x] = root_y                    # x 的根节点连接到 y 的根节点上，成为 y 的根节点的子节点
        return True

    def is_connected(self, x, y):                   # 查询操作：判断 x 和 y 是否同属于一个集合
        return self.__find(x) == self.__find(y)

class Solution:
    def equationsPossible(self, equations: List[str]) -> bool:
        union_find = UnionFind(26)
        for eqation in equations:
            if eqation[1] == "=":
                index1 = ord(eqation[0]) - 97
                index2 = ord(eqation[3]) - 97
                union_find.union(index1, index2)

        for eqation in equations:
            if eqation[1] == "!":
                index1 = ord(eqation[0]) - 97
                index2 = ord(eqation[3]) - 97
                if union_find.is_connected(index1, index2):
                    return False
        return True
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(n + C \times \log C)$。其中 $n$ 是方程组 $equations$ 中的等式数量。$C$ 是字母变量的数量。本题中变量都是小写字母，即 $C \le 26$。
- **空间复杂度**：$O(C)$。# [0992. K 个不同整数的子数组](https://leetcode.cn/problems/subarrays-with-k-different-integers/)

- 标签：数组、哈希表、计数、滑动窗口
- 难度：困难

## 题目链接

- [0992. K 个不同整数的子数组 - 力扣](https://leetcode.cn/problems/subarrays-with-k-different-integers/)

## 题目大意

给定一个正整数数组 `nums`，再给定一个整数 `k`。如果 `nums` 的某个子数组中不同整数的个数恰好为 `k`，则称 `nums` 的这个连续、不一定不同的子数组为「好子数组」。

- 例如，`[1, 2, 3, 1, 2]` 中有 3 个不同的整数：`1`，`2` 以及 `3`。

要求：返回 `nums` 中好子数组的数目。

## 解题思路

这道题转换一下思路会更简单。

恰好包含 `k` 个不同整数的连续子数组数量 = 包含小于等于 `k` 个不同整数的连续子数组数量 - 包含小于等于 `k - 1` 个不同整数的连续子数组数量

可以专门写一个方法计算包含小于等于 `k` 个不同整数的连续子数组数量。

计算包含小于等于 `k` 个不同整数的连续子数组数量的方法具体步骤如下：

用滑动窗口 `windows` 来记录不同的整数个数，`windows` 为哈希表类型。

设定两个指针：`left`、`right`，分别指向滑动窗口的左右边界，保证窗口内不超过 `k` 个不同整数。

- 一开始，`left`、`right` 都指向 `0`。
- 将最右侧整数 `nums[right]` 加入当前窗口 `windows` 中，记录该整数个数。
- 如果该窗口中该整数的个数多于 `k` 个，即 `len(windows) > k`，则不断右移 `left`，缩小滑动窗口长度，并更新窗口中对应整数的个数，直到 `len(windows) <= k`。
- 维护更新包含小于等于 `k` 个不同整数的连续子数组数量。每次累加数量为 `right - left + 1`，表示以 `nums[right]` 为结尾的小于等于 `k` 个不同整数的连续子数组数量。
- 然后右移 `right`，直到 `right >= len(nums)` 结束。
- 返回包含小于等于 `k` 个不同整数的连续子数组数量。

## 代码

```python
class Solution:
    def subarraysMostKDistinct(self, nums, k):
        windows = dict()
        left, right = 0, 0
        ans = 0
        while right < len(nums):
            if nums[right] in windows:
                windows[nums[right]] += 1
            else:
                windows[nums[right]] = 1
            while len(windows) > k:
                windows[nums[left]] -= 1
                if windows[nums[left]] == 0:
                    del windows[nums[left]]
                left += 1
            ans += right - left + 1
            right += 1
        return ans

    def subarraysWithKDistinct(self, nums: List[int], k: int) -> int:
        return self.subarraysMostKDistinct(nums, k) - self.subarraysMostKDistinct(nums, k - 1)
```

# [0993. 二叉树的堂兄弟节点](https://leetcode.cn/problems/cousins-in-binary-tree/)

- 标签：树、深度优先搜索、广度优先搜索、二叉树
- 难度：简单

## 题目链接

- [0993. 二叉树的堂兄弟节点 - 力扣](https://leetcode.cn/problems/cousins-in-binary-tree/)

## 题目大意

给定一个二叉树，和两个值 x，y。从二叉树中找出 x 和 y 对应的节点 node_x，node_y。如果两个节点是堂兄弟节点，则返回 True，否则返回 False。

- 堂兄弟节点：两个节点的深度相同，父节点不同。

## 解题思路

广度优先搜索或者深度优先搜索都可。以深度优先搜索为例，递归遍历查找节点值为 x，y 的两个节点。在递归的同时，需要传入递归函数当前节点的深度和父节点信息。若找到对应的节点，则保存两节点对应深度和父节点信息。最后判断两个节点是否是深度相同，父节点不同。如果是，则返回 True，不是则返回 False。

## 代码

```python
class Solution:
    def isCousins(self, root: TreeNode, x: int, y: int) -> bool:
        depths = [0, 0]
        parents = [None, None]

        def dfs(node, depth, parent):
            if not node:
                return
            if node.val == x:
                depths[0] = depth
                parents[0] = parent
            elif node.val == y:
                depths[1] = depth
                parents[1] = parent
            dfs(node.left, depth+1, node)
            dfs(node.right, depth+1, node)

        dfs(root, 0, None)
        return depths[0] == depths[1] and parents[0] != parents[1]
```

# [0995. K 连续位的最小翻转次数](https://leetcode.cn/problems/minimum-number-of-k-consecutive-bit-flips/)

- 标签：位运算、队列、数组、前缀和、滑动窗口
- 难度：困难

## 题目链接

- [0995. K 连续位的最小翻转次数 - 力扣](https://leetcode.cn/problems/minimum-number-of-k-consecutive-bit-flips/)

## 题目大意

**描述**：给定一个仅包含 $0$ 和 $1$ 的数组 $nums$，再给定一个整数 $k$。进行一次 $k$ 位翻转包括选择一个长度为 $k$ 的（连续）子数组，同时将子数组中的每个 $0$ 更改为 $1$，而每个 $1$ 更改为 $0$。

**要求**：返回所需的 $k$ 位翻转的最小次数，以便数组没有值为 $0$ 的元素。如果不可能，返回 $-1$。

**说明**：

- **子数组**：数组的连续部分。
- $1 <= nums.length <= 105$。
- $1 <= k <= nums.length$。

**示例**：

- 示例 1：

```python
输入：nums = [0,1,0], K = 1
输出：2
解释：先翻转 A[0]，然后翻转 A[2]。
```

- 示例 2：

```python
输入：nums = [0,0,0,1,0,1,1,0], K = 3
输出：3
解释：
翻转 A[0],A[1],A[2]: A变成 [1,1,1,1,0,1,1,0]
翻转 A[4],A[5],A[6]: A变成 [1,1,1,1,1,0,0,0]
翻转 A[5],A[6],A[7]: A变成 [1,1,1,1,1,1,1,1]
```

## 解题思路

### 思路 1：滑动窗口

每次需要翻转的起始位置肯定是遇到第一个元素为 $0$ 的位置开始反转，如果能够使得整个数组不存在 $0$，即返回 $ans$ 作为反转次数。

同时我们还可以发现：

- 如果某个元素反转次数为奇数次，元素会由 $0 \rightarrow 1$，$1 \rightarrow 0$。
- 如果某个元素反转次数为偶数次，元素不会发生变化。

每个第 $i$ 位置上的元素只会被前面 $[i - k + 1, i - 1]$ 的元素影响。所以我们只需要知道前面 $k - 1$ 个元素翻转次数的奇偶性就可以了。

同时如果我们知道了前面 $k - 1$ 个元素的翻转次数就可以直接修改 $nums[i]$ 了。

我们使用 $flip\underline{\hspace{0.5em}}count$ 记录第 $i$ 个元素之前 $k - 1$ 个位置总共被反转了多少次，或者 $flip\underline{\hspace{0.5em}}count$ 是大小为 $k - 1$ 的滑动窗口。

- 如果前面第 $k - 1$ 个元素翻转了奇数次，则如果 $nums[i] == 1$，则 $nums[i]$ 也被翻转成了 $0$，需要再翻转 $1$ 次。
- 如果前面第 $k - 1$ 个元素翻转了偶数次，则如果 $nums[i] == 0$，则 $nums[i]$ 也被翻转成为了 $0$，需要再翻转 $1$ 次。

这两句写成判断语句可以写为：`if (flip_count + nums[i]) % 2 == 0:`。

因为 $0 <= nums[i] <= 1$，所以我们可以用 $0$ 和 $1$ 以外的数，比如 $2$ 来标记第 $i$ 个元素发生了翻转，即 `nums[i] = 2`。这样在遍历到第 $i$ 个元素时，如果有 $nums[i - k] == 2$，则说明 $nums[i - k]$ 发生了翻转。同时根据 $flip\underline{\hspace{0.5em}}count$ 和 $nums[i]$ 来判断第 $i$ 位是否需要进行翻转。

整个算法的具体步骤如下：

- 使用 $res$ 记录最小翻转次数。使用 $flip\underline{\hspace{0.5em}}count$ 记录窗口内前 $k - 1 $ 位元素的翻转次数。
- 遍历数组 $nums$，对于第 $i$ 位元素：
  - 如果 $i - k >= 0$，并且 $nums[i - k] == 2$，需要缩小窗口，将翻转次数减一。（此时窗口范围为 $[i - k + 1, i - 1]$）。
  - 如果 $(flip\underline{\hspace{0.5em}}count + nums[i]) \mod 2 == 0$，则说明 $nums[i]$ 还需要再翻转一次，将 $nums[i]$ 标记为 $2$，同时更新窗口内翻转次数 $flip\underline{\hspace{0.5em}}count$ 和答案最小翻转次数 $ans$。
- 遍历完之后，返回 $res$。

### 思路 1：代码

```python
class Solution:
    def minKBitFlips(self, nums: List[int], k: int) -> int:
        ans = 0
        flip_count = 0
        for i in range(len(nums)):
            if i - k >= 0 and nums[i - k] == 2:
                flip_count -= 1
            if (flip_count + nums[i]) % 2 == 0:
                if i + k > len(nums):
                    return -1
                nums[i] = 2
                flip_count += 1
                ans += 1
        return ans
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(n)$，其中 $n$ 为数组 $nums$ 的长度。
- **空间复杂度**：$O(1)$。

# [0999. 可以被一步捕获的棋子数](https://leetcode.cn/problems/available-captures-for-rook/)

- 标签：数组、矩阵、模拟
- 难度：简单

## 题目链接

- [0999. 可以被一步捕获的棋子数 - 力扣](https://leetcode.cn/problems/available-captures-for-rook/)

## 题目大意

**描述**：在一个 $8 \times 8$ 的棋盘上，有一个白色的车（Rook），用字符 `'R'` 表示。棋盘上还可能存在空方块，白色的象（Bishop）以及黑色的卒（pawn），分别用字符 `'.'`，`'B'` 和 `'p'` 表示。不难看出，大写字符表示的是白棋，小写字符表示的是黑棋。

**要求**：你现在可以控制车移动一次，请你统计有多少敌方的卒处于你的捕获范围内（即，可以被一步捕获的棋子数）。

**说明**：

- 车按国际象棋中的规则移动。东，西，南，北四个基本方向任选其一，然后一直向选定的方向移动，直到满足下列四个条件之一：
  - 棋手选择主动停下来。
  - 棋子因到达棋盘的边缘而停下。
  - 棋子移动到某一方格来捕获位于该方格上敌方（黑色）的卒，停在该方格内。
  - 车不能进入/越过已经放有其他友方棋子（白色的象）的方格，停在友方棋子前。

- $board.length == board[i].length == 8$
- $board[i][j]$ 可以是 `'R'`，`'.'`，`'B'` 或 `'p'`。
- 只有一个格子上存在 $board[i][j] == 'R'$。

**示例**：

- 示例 1：

![](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2019/02/23/1253_example_1_improved.PNG)

```python
输入：[[".",".",".",".",".",".",".","."],[".",".",".","p",".",".",".","."],[".",".",".","R",".",".",".","p"],[".",".",".",".",".",".",".","."],[".",".",".",".",".",".",".","."],[".",".",".","p",".",".",".","."],[".",".",".",".",".",".",".","."],[".",".",".",".",".",".",".","."]]
输出：3
解释：在本例中，车能够捕获所有的卒。
```

- 示例 2：

![img](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2019/02/23/1253_example_2_improved.PNG) 

```python
输入：[[".",".",".",".",".",".",".","."],[".","p","p","p","p","p",".","."],[".","p","p","B","p","p",".","."],[".","p","B","R","B","p",".","."],[".","p","p","B","p","p",".","."],[".","p","p","p","p","p",".","."],[".",".",".",".",".",".",".","."],[".",".",".",".",".",".",".","."]]
输出：0
解释：象阻止了车捕获任何卒。
```

## 解题思路

### 思路 1：模拟

1. 双重循环遍历确定白色车的位置 $(pos\underline{\hspace{0.5em}}i,poss\underline{\hspace{0.5em}}j)$。
2. 让车向上、下、左、右四个方向进行移动，直到超出边界 / 碰到白色象 / 碰到卒为止。使用计数器 $cnt$ 记录捕获的卒的数量。
3. 返回答案 $cnt$。

### 思路 1：代码

```Python
class Solution:
    def numRookCaptures(self, board: List[List[str]]) -> int:
        directions = {(1, 0), (-1, 0), (0, 1), (0, -1)}
        pos_i, pos_j = -1, -1
        for i in range(len(board)):
            if pos_i != -1 and pos_j != -1:
                break
            for j in range(len(board[i])):
                if board[i][j] == 'R':
                    pos_i, pos_j = i, j
                    break

        cnt = 0
        for direction in directions:
            setp = 0
            while True:
                new_i = pos_i + setp * direction[0]
                new_j = pos_j + setp * direction[1]
                if new_i < 0 or new_i >= 8 or new_j < 0 or new_j >= 8 or board[new_i][new_j] == 'B':
                    break
                if board[new_i][new_j] == 'p':
                    cnt += 1
                    break
                setp += 1
        
        return cnt
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(n^2)$，其中 $n$ 为棋盘的边长。
- **空间复杂度**：$O(1)$。

# [1000. 合并石头的最低成本](https://leetcode.cn/problems/minimum-cost-to-merge-stones/)

- 标签：数组、动态规划、前缀和
- 难度：困难

## 题目链接

- [1000. 合并石头的最低成本 - 力扣](https://leetcode.cn/problems/minimum-cost-to-merge-stones/)

## 题目大意

**描述**：给定一个代表 $n$ 堆石头的整数数组 $stones$，其中 $stones[i]$ 代表第 $i$ 堆中的石头个数。再给定一个整数 $k$， 每次移动需要将连续的 $k$ 堆石头合并为一堆，而这次移动的成本为这 $k$ 堆中石头的总数。

**要求**：返回把所有石头合并成一堆的最低成本。如果无法合并成一堆，则返回 $-1$。

**说明**：

- $n == stones.length$。
- $1 \le n \le 30$。
- $1 \le stones[i] \le 100$。
- $2 \le k \le 30$。

**示例**：

- 示例 1：

```python
输入：stones = [3,2,4,1], K = 2
输出：20
解释：
从 [3, 2, 4, 1] 开始。
合并 [3, 2]，成本为 5，剩下 [5, 4, 1]。
合并 [4, 1]，成本为 5，剩下 [5, 5]。
合并 [5, 5]，成本为 10，剩下 [10]。
总成本 20，这是可能的最小值。
```

- 示例 2：

```python
输入：stones = [3,5,1,2,6], K = 3
输出：25
解释：
从 [3, 5, 1, 2, 6] 开始。
合并 [5, 1, 2]，成本为 8，剩下 [3, 8, 6]。
合并 [3, 8, 6]，成本为 17，剩下 [17]。
总成本 25，这是可能的最小值。
```

## 解题思路

### 思路 1：动态规划 + 前缀和

每次将 $k$ 堆连续的石头合并成 $1$ 堆，石头堆数就会减少 $k - 1$ 堆。总共有 $n$ 堆石子，则：

1. 当 $(n - 1) \mod (k - 1) == 0$ 时，一定可以经过 $\frac{n - 1}{k - 1}$ 次合并，将 $n$ 堆石头合并为 $1$ 堆。
2. 当 $(n - 1) \mod (k - 1) \ne 0$ 时，则无法将所有的石头合并成一堆。

根据以上情况，我们可以先将无法将所有的石头合并成一堆的情况排除出去，接下来只考虑合法情况。

由于每次合并石头的成本为合并的 $k$ 堆的石子总数，即数组 $stones$ 中长度为 $k$ 的连续子数组和，因此为了快速计算数组 $stones$ 的连续子数组和，我们可以使用「前缀和」的方式，预先计算出「前 $i$ 堆的石子总数」，从而可以在 $O(1)$ 的时间复杂度内得到数组 $stones$ 的连续子数组和。

$k$ 堆石头合并为 $1$ 堆石头的过程，可以看做是长度为 $k$ 的连续子数组合并为长度为 $1$ 的子数组的过程，也可以看做是将长度为 $k$ 的区间合并为长度为 $1$ 的区间。

接下来我们就可以按照「区间 DP 问题」的基本思路来做。

###### 1. 划分阶段

按照区间长度进行阶段划分。

###### 2. 定义状态

定义状态 $dp[i][j][m]$ 表示为：将区间 $[i, j]$ 的石堆合并成 $m$ 堆的最低成本，其中 $m$ 的取值为 $[1,k]$。

###### 3. 状态转移方程

我们将区间 $[i, j]$ 的石堆合并成 $m$ 堆，可以枚举 $i \le n \le j$，将区间 $[i, j]$ 拆分为两个区间 $[i, n]$ 和 $[n + 1, j]$。然后将 $[i, n]$ 中的石头合并为 $1$ 堆，将 $[n + 1, j]$ 中的石头合并成 $m - 1$ 堆。最后将 $1$ 堆石头和 $m - 1$ 堆石头合并成 $1$ 堆，这样就可以将 $[i, j]$ 的石堆合并成 $k$ 堆。则状态转移方程为：$dp[i][j][m] = min_{i \le n < j} \lbrace dp[i][n][1] + dp[n + 1][j][m - 1] \rbrace$。

我们再将区间 $[i, j]$ 的 $k$ 堆石头合并成 $1$ 堆，其成本为 区间 $[i, j]$ 的石堆合并成 $k$ 堆的成本，加上将这 $k$ 堆石头合并成 $1$ 堆的成本，即状态转移方程为：$dp[i][j][1] = dp[i][j][k] + \sum_{t = i}^{t = j} stones[t]$。

###### 4. 初始条件

- 长度为 $1$ 的区间 $[i, i]$ 合并为 $1$ 堆成本为 $0$，即：$dp[i][i][1] = 0$。

###### 5. 最终结果

根据我们之前定义的状态，$dp[i][j][m]$ 表示为：将区间 $[i, j]$ 的石堆合并成 $m$ 堆的最低成本，其中 $m$ 的取值为 $[1,k]$。 所以最终结果为 $dp[1][size][1]$，其中 $size$ 为数组 $stones$ 的长度。

### 思路 1：代码

```python
class Solution:
    def mergeStones(self, stones: List[int], k: int) -> int:
        size = len(stones)
        if (size - 1) % (k - 1) != 0:
            return -1

        prefix = [0 for _ in range(size + 1)]
        for i in range(1, size + 1):
            prefix[i] = prefix[i - 1] + stones[i - 1]

        dp = [[[float('inf') for _ in range(k + 1)] for _ in range(size)] for _ in range(size)]

        for i in range(size):
            dp[i][i][1] = 0

        for l in range(2, size + 1):
            for i in range(size):
                j = i + l - 1
                if j >= size:
                    break
                for m in range(2, k + 1):
                    for n in range(i, j, k - 1):
                        dp[i][j][m] = min(dp[i][j][m], dp[i][n][1] + dp[n + 1][j][m - 1])
                dp[i][j][1] = dp[i][j][k] + prefix[j + 1] - prefix[i]

        return dp[0][size - 1][1]
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(n^3 \times k)$，其中 $n$ 是数组 $stones$ 的长度。
- **空间复杂度**：$O(n^2 \times k)$。

### 思路 2：动态规划 + 状态优化

在思路 1 中，我们使用定义状态 $dp[i][j][m]$ 表示为：将区间 $[i, j]$ 的石堆合并成 $m$ 堆的最低成本，其中 $m$ 的取值为 $[1,k]$。

事实上，对于固定区间 $[i, j]$，初始时堆数为 $j - i + 1$，每次合并都会减少 $k - 1$ 堆，合并到无法合并时的堆数固定为 $(j - i) \mod (k - 1) + 1$。

所以，我们可以直接定义状态 $dp[i][j]$ 表示为：将区间 $[i, j]$ 的石堆合并到无法合并时的最低成本。

具体步骤如下：

###### 1. 划分阶段

按照区间长度进行阶段划分。

###### 2. 定义状态

定义状态 $dp[i][j]$ 表示为：将区间 $[i, j]$ 的石堆合并到无法合并时的最低成本。

###### 3. 状态转移方程

枚举 $i \le n \le j$，将区间 $[i, j]$ 拆分为两个区间 $[i, n]$ 和 $[n + 1, j]$。然后将区间 $[i, n]$ 合并成 $1$ 堆，$[n + 1, j]$ 合并成 $m$ 堆。

$dp[i][j] = min_{i \le n < j} \lbrace dp[i][n] + dp[n + 1][j] \rbrace$。

如果 $(j - i) \mod (k - 1) == 0$，则说明区间 $[i, j]$ 能狗合并为 1 堆，则加上区间子数组和，即 $dp[i][j] += prefix[j + 1] - prefix[i]$。

###### 4. 初始条件

- 长度为 $1$ 的区间 $[i, i]$ 合并到无法合并时的最低成本为 $0$，即：$dp[i][i] = 0$。

###### 5. 最终结果

根据我们之前定义的状态，$dp[i][j]$ 表示为：将区间 $[i, j]$ 的石堆合并到无法合并时的最低成本。所以最终结果为 $dp[0][size - 1]$，其中 $size$ 为数组 $stones$ 的长度。

### 思路 2：代码

```python
class Solution:
    def mergeStones(self, stones: List[int], k: int) -> int:
        size = len(stones)
        if (size - 1) % (k - 1) != 0:
            return -1

        prefix = [0 for _ in range(size + 1)]
        for i in range(1, size + 1):
            prefix[i] = prefix[i - 1] + stones[i - 1]

        dp = [[float('inf') for _ in range(size)] for _ in range(size)]

        for i in range(size):
            dp[i][i] = 0

        for l in range(2, size + 1):
            for i in range(size):
                j = i + l - 1
                if j >= size:
                    break
                # 遍历每一个可以组成 k 堆石子的分割点 n，每次递增 k - 1 个
                for n in range(i, j, k - 1):
                    # 判断 [i, n] 到 [n + 1, j] 是否比之前花费小
                    dp[i][j] = min(dp[i][j], dp[i][n] + dp[n + 1][j])
                # 如果 [i, j] 能狗合并为 1 堆，则加上区间子数组和
                if (l - 1) % (k - 1) == 0:
                    dp[i][j] += prefix[j + 1] - prefix[i]

        return dp[0][size - 1]
```

### 思路 2：复杂度分析

- **时间复杂度**：$O(n^3)$，其中 $n$ 是数组 $stones$ 的长度。
- **空间复杂度**：$O(n^2)$。

## 参考资料

- 【题解】[一题一解：动态规划（区间 DP）+ 前缀和（清晰题解） - 合并石头的最低成本](https://leetcode.cn/problems/minimum-cost-to-merge-stones/solution/python3javacgo-yi-ti-yi-jie-dong-tai-gui-lr9q/)
# [1002. 查找共用字符](https://leetcode.cn/problems/find-common-characters/)

- 标签：数组、哈希表、字符串
- 难度：简单

## 题目链接

- [1002. 查找共用字符 - 力扣](https://leetcode.cn/problems/find-common-characters/)

## 题目大意

**描述**：给定一个字符串数组 $words$。

**要求**：找出所有在 $words$ 的每个字符串中都出现的公用字符（包括重复字符），并以数组形式返回。可以按照任意顺序返回答案。

**说明**：

- $1 \le words.length \le 100$。
- $1 \le words[i].length \le 100$。
- $words[i]$ 由小写英文字母组成。

**示例**：

- 示例 1：

```python
输入：words = ["bella","label","roller"]
输出：["e","l","l"]
```

- 示例 2：

```python
输入：words = ["cool","lock","cook"]
输出：["c","o"]
```

## 解题思路

### 思路 1：哈希表

如果某个字符 $ch$ 在所有字符串中都出现了 $k$ 次以上，则最终答案中需要包含 $k$ 个 $ch$。因此，我们可以使用哈希表 $minfreq[ch]$ 记录字符 $ch$ 在所有字符串中出现的最小次数。具体步骤如下：

1. 定义长度为 $26$ 的哈希表 $minfreq$，初始化所有字符出现次数为无穷大，$minfreq[ch] = float('inf')$。
2. 遍历字符串数组中的所有字符串 $word$，对于字符串 $word$：
   1. 记录 $word$ 中所有字符串的出现次数 $freq[ch]$。
   2. 取 $freq[ch]$ 与 $minfreq[ch]$ 中的较小值更新 $minfreq[ch]$。
3. 遍历完之后，再次遍历 $26$ 个字符，将所有最小出现次数大于零的字符按照出现次数存入答案数组中。
4. 最后将答案数组返回。

### 思路 1：代码

```python
class Solution:
    def commonChars(self, words: List[str]) -> List[str]:
        minfreq = [float('inf') for _ in range(26)]
        for word in words:
            freq = [0 for _ in range(26)]
            for ch in word:
                freq[ord(ch) - ord('a')] += 1
            for i in range(26):
                minfreq[i] = min(minfreq[i], freq[i])

        res = []
        for i in range(26):
            while minfreq[i]:
                res.append(chr(i + ord('a')))
                minfreq[i] -= 1
        
        return res
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(n \times (|\sum| + m))$，其中 $n$ 为字符串数组 $words$ 的长度，$m$ 为每个字符串的平均长度，$|\sum|$ 为字符集。
- **空间复杂度**：$O(|\sum|)$。

# [1004. 最大连续1的个数 III](https://leetcode.cn/problems/max-consecutive-ones-iii/)

- 标签：数组、二分查找、前缀和、滑动窗口
- 难度：中等

## 题目链接

- [1004. 最大连续1的个数 III - 力扣](https://leetcode.cn/problems/max-consecutive-ones-iii/)

## 题目大意

**描述**：给定一个由 $0$、$1$ 组成的数组 $nums$，再给定一个整数 $k$。最多可以将 $k$ 个值从 $0$ 变到 $1$。

**要求**：返回仅包含 $1$ 的最长连续子数组的长度。

**说明**：

- $1 \le nums.length \le 10^5$。
- $nums[i]$ 不是 $0$ 就是 $1$。
- $0 \le k \le nums.length$。

**示例**：

- 示例 1：

```python
输入：nums = [1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0], K = 2
输出：6
解释：[1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1]
将 nums[5]、nums[10] 从 0 翻转到 1，最长的子数组长度为 6。
```

- 示例 2：

```python
输入：nums = [0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1], K = 3
输出：10
解释：[0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1]
将 nums[4]、nums[5]、nums[9] 从 0 翻转到 1，最长的子数组长度为 10。
```

## 解题思路

### 思路 1：滑动窗口（不定长度）

1. 使用两个指针 $left$、$right$ 指向数组开始位置。使用 $max\underline{\hspace{0.5em}}count$ 来维护仅包含 $1$ 的最长连续子数组的长度。
2. 不断右移 $right$ 指针，扩大滑动窗口范围，并统计窗口内 $0$ 元素的个数。
3. 直到 $0$ 元素的个数超过 $k$ 时将 $left$ 右移，缩小滑动窗口范围，并减小 $0$ 元素的个数，同时维护 $max\underline{\hspace{0.5em}}count$。
4. 最后输出最长连续子数组的长度 $max\underline{\hspace{0.5em}}count$。

### 思路 1：代码

```python
class Solution:
    def longestOnes(self, nums: List[int], k: int) -> int:
        max_count = 0
        zero_count = 0
        left, right = 0, 0
        while right < len(nums):
            if nums[right] == 0:
                zero_count += 1
            right += 1
            if zero_count > k:
                if nums[left] == 0:
                    zero_count -= 1
                left += 1
            max_count = max(max_count, right - left)
        return max_count
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(n)$。
- **空间复杂度**：$O(1)$。

# [1005. K 次取反后最大化的数组和](https://leetcode.cn/problems/maximize-sum-of-array-after-k-negations/)

- 标签：贪心、数组、排序
- 难度：简单

## 题目链接

- [1005. K 次取反后最大化的数组和 - 力扣](https://leetcode.cn/problems/maximize-sum-of-array-after-k-negations/)

## 题目大意

给定一个整数数组 nums 和一个整数 k。只能用下面的方法修改数组：

- 将数组上第 i 个位置上的值取相反数，即将 `nums[i]` 变为 `-nums[i]`。

用这种方式进行 K 次修改（可以多次修改同一个位置 i） 后，返回数组可能的最大和。

## 解题思路

- 先将数组按绝对值大小进行排序
- 从绝对值大的数开始遍历数组，如果 nums[i] < 0，并且 k > 0：
  - 则对 nums[i] 取相反数，并将 k 值 -1。
- 如果最后 k 还有余值，则判断奇偶性：
  - 若 k 为奇数，则将数组绝对值最小的数进行取反。
  - 若 k 为偶数，则说明可将某一位数进行偶数次取反，和原数值一致，则不需要进行操作。
- 最后返回数组和。

## 代码

```python
class Solution:
    def largestSumAfterKNegations(self, nums: List[int], k: int) -> int:
        nums.sort(key=lambda x: abs(x), reverse = True)
        for i in range(len(nums)):
            if nums[i] < 0 and k > 0:
                nums[i] *= -1
                k -= 1
        if k % 2 == 1:
            nums[-1] *= -1
        return sum(nums)
```

# [1008. 前序遍历构造二叉搜索树](https://leetcode.cn/problems/construct-binary-search-tree-from-preorder-traversal/)

- 标签：栈、树、二叉搜索树、数组、二叉树、单调栈
- 难度：中等

## 题目链接

- [1008. 前序遍历构造二叉搜索树 - 力扣](https://leetcode.cn/problems/construct-binary-search-tree-from-preorder-traversal/)

## 题目大意

给定一棵二叉搜索树的前序遍历结果 `preorder`。

要求：返回与给定前序遍历 `preorder` 相匹配的二叉搜索树的根节点。题目保证，对于给定的测试用例，总能找到满足要求的二叉搜索树。

## 解题思路

二叉搜索树的中序遍历是升序序列。而题目又给了我们二叉搜索树的前序遍历，那么通过对前序遍历结果的排序，我们也可以得到二叉搜索树的中序遍历结果。这样就能根据二叉树的前序、中序遍历序列构造二叉树了。就变成了了「[0105. 从前序与中序遍历序列构造二叉树](https://leetcode.cn/problems/construct-binary-tree-from-preorder-and-inorder-traversal/)」题。

此外，我们还有另一种方法求解。前序遍历的顺序是：根 -> 左 -> 右。并且在二叉搜索树中，左子树的值小于根节点，右子树的值大于根节点。

根据以上性质，我们可以递归地构造二叉搜索树。

首先，以前序遍历的开始位置元素构造为根节点。从开始位置的下一个位置开始，找到序列中第一个大于等于根节点值的位置 `mid`。该位置左侧的值都小于根节点，右侧的值都大于等于根节点。以此位置为中心，递归的构造左子树和右子树。

最后再将根节点进行返回。

## 代码

```python
class Solution:
    def buildTree(self, preorder, start, end):
        if start == end:
            return None
        root = preorder[start]
        mid = start + 1
        while mid < end and preorder[mid] < root:
            mid += 1
        node = TreeNode(root)
        node.left = self.buildTree(preorder, start + 1, mid)
        node.right = self.buildTree(preorder, mid, end)
        return node

    def bstFromPreorder(self, preorder: List[int]) -> TreeNode:
        return self.buildTree(preorder, 0, len(preorder))
```

# [1009. 十进制整数的反码](https://leetcode.cn/problems/complement-of-base-10-integer/)

- 标签：位运算
- 难度：简单

## 题目链接

- [1009. 十进制整数的反码 - 力扣](https://leetcode.cn/problems/complement-of-base-10-integer/)

## 题目大意

**描述**：给定一个十进制数 $n$。

**要求**：返回其二进制表示的反码对应的十进制整数。

**说明**：

- $0 \le N < 10^9$。

**示例**：

- 示例 1：

```python
输入：5
输出：2
解释：5 的二进制表示为 "101"，其二进制反码为 "010"，也就是十进制中的 2 。
```

- 示例 2：

```python
输入：7
输出：0
解释：7 的二进制表示为 "111"，其二进制反码为 "000"，也就是十进制中的 0 。
```

## 解题思路

### 思路 1：模拟

1. 将十进制数 $n$ 转为二进制 $binary$。
2. 遍历二进制 $binary$ 的每一个数位 $digit$。
   1. 如果 $digit$ 为 $0$，则将其转为 $1$，存入答案 $res$ 中。
   2. 如果 $digit$ 为 $1$，则将其转为 $0$，存入答案 $res$ 中。
3. 返回答案 $res$。

### 思路 1：代码

```python
class Solution:
    def bitwiseComplement(self, n: int) -> int:
        binary = ""
        while n:
            binary += str(n % 2)
            n //= 2
        if binary == "":
            binary = "0"
        else:
            binary = binary[::-1]
        res = 0
        for digit in binary:
            if digit == '0':
                res = res * 2 + 1
            else:
                res = res * 2
        
        return res
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(len(n))$，其中 $len(n)$ 为 $n$ 对应二进制的长度。
- **空间复杂度**：$O(1)$。
# [1011. 在 D 天内送达包裹的能力](https://leetcode.cn/problems/capacity-to-ship-packages-within-d-days/)

- 标签：数组、二分查找
- 难度：中等

## 题目链接

- [1011. 在 D 天内送达包裹的能力 - 力扣](https://leetcode.cn/problems/capacity-to-ship-packages-within-d-days/)

## 题目大意

**描述**：传送带上的包裹必须在 $D$ 天内从一个港口运送到另一个港口。给定所有包裹的重量数组 $weights$，货物必须按照给定的顺序装运。且每天船上装载的重量不会超过船的最大运载重量。

**要求**：求能在 $D$ 天内将所有包裹送达的船的最低运载量。

**说明**：

- $1 \le days \le weights.length \le 5 * 10^4$。
- $1 \le weights[i] \le 500$。

**示例**：

- 示例 1：

```python
输入：weights = [1,2,3,4,5,6,7,8,9,10], days = 5
输出：15
解释：
船舶最低载重 15 就能够在 5 天内送达所有包裹，如下所示：
第 1 天：1, 2, 3, 4, 5
第 2 天：6, 7
第 3 天：8
第 4 天：9
第 5 天：10
请注意，货物必须按照给定的顺序装运，因此使用载重能力为 14 的船舶并将包装分成 (2, 3, 4, 5), (1, 6, 7), (8), (9), (10) 是不允许的。 
```

- 示例 2：

```python
输入：weights = [3,2,2,4,1,4], days = 3
输出：6
解释：
船舶最低载重 6 就能够在 3 天内送达所有包裹，如下所示：
第 1 天：3, 2
第 2 天：2, 4
第 3 天：1, 4
```

## 解题思路

### 思路 1：二分查找

船最小的运载能力，最少也要等于或大于最重的那件包裹，即 $max(weights)$。最多的话，可以一次性将所有包裹运完，即 $sum(weights)$。船的运载能力介于 $[max(weights), sum(weights)]$ 之间。

我们现在要做的就是从这个区间内，找到满足可以在 $D$ 天内运送完所有包裹的最小载重量。

可以通过二分查找的方式，找到满足要求的最小载重量。

### 思路 1：代码

```python
class Solution:
    def shipWithinDays(self, weights: List[int], D: int) -> int:
        left = max(weights)
        right = sum(weights)

        while left < right:
            mid = (left + right) >> 1
            days = 1
            cur = 0
            for weight in weights:
                if cur + weight > mid:
                    days += 1
                    cur = 0
                cur += weight

            if days <= D:
                right = mid
            else:
                left = mid + 1
        return left
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(\log n)$。二分查找算法的时间复杂度为 $O(\log n)$。
- **空间复杂度**：$O(1)$。只用到了常数空间存放若干变量。

# [1012. 至少有 1 位重复的数字](https://leetcode.cn/problems/numbers-with-repeated-digits/)

- 标签：数学、动态规划
- 难度：困难

## 题目链接

- [1012. 至少有 1 位重复的数字 - 力扣](https://leetcode.cn/problems/numbers-with-repeated-digits/)

## 题目大意

**描述**：给定一个正整数 $n$。

**要求**：返回在 $[1, n]$ 范围内具有至少 $1$ 位重复数字的正整数的个数。

**说明**：

- $1 \le n \le 10^9$。

**示例**：

- 示例 1：

```python
输入：n = 20
输出：1
解释：具有至少 1 位重复数字的正数（<= 20）只有 11。
```

- 示例 2：

```python
输入：n = 100
输出：10
解释：具有至少 1 位重复数字的正数（<= 100）有 11，22，33，44，55，66，77，88，99 和 100。
```

## 解题思路

### 思路 1：动态规划 + 数位 DP

正向求解在 $[1, n]$ 范围内具有至少 $1$ 位重复数字的正整数的个数不太容易，我们可以反向思考，先求解出在 $[1, n]$ 范围内各位数字都不重复的正整数的个数 $ans$，然后 $n - ans$ 就是题目答案。

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
      2. `isLimit and d == maxX` 表示 $pos + 1$ 位受到之前位限制和 $pos$ 位限制。
      3. $isNum == True$ 表示 $pos$ 位选择了数字。
6. 最后的方案数为 `n - dfs(0, 0, True, False)`，将其返回即可。

### 思路 1：代码

```python
class Solution:
    def numDupDigitsAtMostN(self, n: int) -> int:
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
                # 如果 isNumb 为 False，则可以跳过当前数位
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
    
        return n - dfs(0, 0, True, False)
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(\log n \times 10 \times 2^{10})$。
- **空间复杂度**：$O(\log n \times 2^{10})$。

# [1014. 最佳观光组合](https://leetcode.cn/problems/best-sightseeing-pair/)

- 标签：数组、动态规划
- 难度：中等

## 题目链接

- [1014. 最佳观光组合 - 力扣](https://leetcode.cn/problems/best-sightseeing-pair/)

## 题目大意

给你一个正整数数组 `values`，其中 `values[i]` 表示第 `i` 个观光景点的评分，并且两个景点 `i` 和 `j` 之间的距离 为 `j - i`。一对景点（`i < j`）组成的观光组合的得分为 `values[i] + values[j] + i - j`，也就是景点的评分之和减去它们两者之间的距离。

要求：返回一对观光景点能取得的最高分。

## 解题思路

求解的是 `ans = max(values[i] + values[j] + i - j)`。对于当前第 `j` 个位置上的元素来说，`values[j] - j` 的值是固定的，求解 `ans` 就是在求解 `values[i] + i` 的最大值。我们使用一个变量 `max_score` 来存储当前第 `j` 个位置元素之前 `values[i] + i` 的最大值。然后遍历数组，求出每一个元素位置之前 `values[i] + i` 的最大值，并找出其中最大的 `ans`。

## 代码

```python
class Solution:
    def maxScoreSightseeingPair(self, values: List[int]) -> int:
        ans = 0
        max_score = values[0]
        for i in range(1, len(values)):
            ans = max(ans, max_score + values[i] - i)
            max_score = max(max_score, values[i] + i)
        return ans
```
# [1020. 飞地的数量](https://leetcode.cn/problems/number-of-enclaves/)

- 标签：深度优先搜索、广度优先搜索、并查集、数组、矩阵
- 难度：中等

## 题目链接

- [1020. 飞地的数量 - 力扣](https://leetcode.cn/problems/number-of-enclaves/)

## 题目大意

**描述**：给定一个二维数组 `grid`，每个单元格为 `0`（代表海）或 `1`（代表陆地）。我们可以从一个陆地走到另一个陆地上（朝四个方向之一），然后从边界上的陆地离开网络的边界。

**要求**：返回网格中无法在任意次数的移动中离开网格边界的陆地单元格的数量。

**说明**：

- $m == grid.length$。
- $n == grid[i].length$。
- $1 \le m, n \le 500$。
- $grid[i][j]$ 的值为 $0$ 或 $1$。

**示例**：

- 示例 1：

![](https://assets.leetcode.com/uploads/2021/02/18/enclaves1.jpg)

```python
输入：grid = [[0,0,0,0],[1,0,1,0],[0,1,1,0],[0,0,0,0]]
输出：3
解释：有三个 1 被 0 包围。一个 1 没有被包围，因为它在边界上。
```

- 示例 2：

![](https://assets.leetcode.com/uploads/2021/02/18/enclaves2.jpg)

```python
输入：grid = [[0,1,1,0],[0,0,1,0],[0,0,1,0],[0,0,0,0]]
输出：0
解释：所有 1 都在边界上或可以到达边界。
```

## 解题思路

### 思路 1：深度优先搜索

与四条边界相连的陆地单元是肯定能离开网络边界的。

我们可以先通过深度优先搜索将与四条边界相关的陆地全部变为海（赋值为 `0`）。

然后统计网格中 `1` 的数量，即为答案。

### 思路 1：代码

```python
class Solution:
    directs = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    def dfs(self, grid, i, j):
        rows = len(grid)
        cols = len(grid[0])
        if i < 0 or i >= rows or j < 0 or j >= cols or grid[i][j] == 0:
            return
        grid[i][j] = 0

        for direct in self.directs:
            new_i = i + direct[0]
            new_j = j + direct[1]
            self.dfs(grid, new_i, new_j)

    def numEnclaves(self, grid: List[List[int]]) -> int:
        rows = len(grid)
        cols = len(grid[0])
        for i in range(rows):
            if grid[i][0] == 1:
                self.dfs(grid, i, 0)
            if grid[i][cols - 1] == 1:
                self.dfs(grid, i, cols - 1)

        for j in range(cols):
            if grid[0][j] == 1:
                self.dfs(grid, 0, j)
            if grid[rows - 1][j] == 1:
                self.dfs(grid, rows - 1, j)

        ans = 0
        for i in range(rows):
            for j in range(cols):
                if grid[i][j] == 1:
                    ans += 1
        return ans
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(m \times n)$。其中 $m$ 和 $n$ 分别为行数和列数。
- **空间复杂度**：$O(m \times n)$。# [1021. 删除最外层的括号](https://leetcode.cn/problems/remove-outermost-parentheses/)

- 标签：栈、字符串
- 难度：简单

## 题目链接

- [1021. 删除最外层的括号 - 力扣](https://leetcode.cn/problems/remove-outermost-parentheses/)

## 题目大意

**描述**：有效括号字符串为空 `""`、`"("` + $A$ + `")"` 或 $A + B$ ，其中 $A$ 和 $B$ 都是有效的括号字符串，$+$ 代表字符串的连接。

- 例如，`""`，`"()"`，`"(())()"` 和 `"(()(()))"` 都是有效的括号字符串。

如果有效字符串 $s$ 非空，且不存在将其拆分为 $s = A + B$ 的方法，我们称其为原语（primitive），其中 $A$ 和 $B$ 都是非空有效括号字符串。

给定一个非空有效字符串 $s$，考虑将其进行原语化分解，使得：$s = P_1 + P_2 + ... + P_k$，其中 $P_i$ 是有效括号字符串原语。

**要求**：对 $s$ 进行原语化分解，删除分解中每个原语字符串的最外层括号，返回 $s$。

**说明**：

- $1 \le s.length \le 10^5$。
- $s[i]$ 为 `'('` 或 `')'`。
- $s$ 是一个有效括号字符串。

**示例**：

- 示例 1：

```python
输入：s = "(()())(())"
输出："()()()"
解释：
输入字符串为 "(()())(())"，原语化分解得到 "(()())" + "(())"，
删除每个部分中的最外层括号后得到 "()()" + "()" = "()()()"。
```

- 示例 2：

```python
输入：s = "(()())(())(()(()))"
输出："()()()()(())"
解释：
输入字符串为 "(()())(())(()(()))"，原语化分解得到 "(()())" + "(())" + "(()(()))"，
删除每个部分中的最外层括号后得到 "()()" + "()" + "()(())" = "()()()()(())"。
```

## 解题思路

### 思路 1：计数遍历

题目要求我们对 $s$ 进行原语化分解，并且删除分解中每个原语字符串的最外层括号。

通过观察可以发现，每个原语其实就是一组有效的括号对（左右括号匹配时），此时我们需要删除这组有效括号对的最外层括号。

我们可以使用一个计数器 $cnt$ 来进行原语化分解，并删除每个原语的最外层括号。

当计数器遇到左括号时，令计数器 $cnt$ 加 $1$，当计数器遇到右括号时，令计数器 $cnt$ 减 $1$。这样当计数器为 $0$ 时表示当前左右括号匹配。

为了删除每个原语的最外层括号，当遇到每个原语最外侧的左括号时（此时 $cnt$ 必然等于 $0$，因为之前字符串为空或者为上一个原语字符串），因为我们不需要最外层的左括号，所以此时我们不需要将其存入答案字符串中。只有当 $cnt > 0$ 时，才将其存入答案字符串中。

同理，当遇到每个原语最外侧的右括号时（此时 $cnt$ 必然等于 $1$，因为之前字符串差一个右括号匹配），因为我们不需要最外层的右括号，所以此时我们不需要将其存入答案字符串中。只有当 $cnt > 1$ 时，才将其存入答案字符串中。

具体步骤如下：

1. 遍历字符串 $s$。
2. 如果遇到 `'('`，判断当前计数器是否大于 $0$：
   1. 如果 $cnt > 0$，则将 `'('` 存入答案字符串中，并令计数器加 $1$，即：`cnt += 1`。
   2. 如果 $cnt == 0$，则令计数器加 $1$，即：`cnt += 1`。
3. 如果遇到 `')'`，判断当前计数器是否大于 $1$：
   1. 如果 $cnt > 1$，则将 `')'` 存入答案字符串中，并令计数器减 $1$，即：`cnt -= 1`。
   2. 如果 $cnt == 1$，则令计数器减 $1$，即：`cnt -= 1`。
4. 遍历完返回答案字符串 $ans$。

### 思路 1：代码

```Python
class Solution:
    def removeOuterParentheses(self, s: str) -> str:
        cnt, ans = 0, ""
        
        for ch in s:
            if ch == '(':
                if cnt > 0:
                    ans += ch
                cnt += 1
            else:
                if cnt > 1:
                    ans += ch
                cnt -= 1
            
        return ans
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(n)$，其中 $n$ 为字符串 $s$ 的长度。
- **空间复杂度**：$O(n)$。

# [1023. 驼峰式匹配](https://leetcode.cn/problems/camelcase-matching/)

- 标签：字典树、双指针、字符串、字符串匹配
- 难度：中等

## 题目链接

- [1023. 驼峰式匹配 - 力扣](https://leetcode.cn/problems/camelcase-matching/)

## 题目大意

**描述**：给定待查询列表 `queries`，和模式串 `pattern`。如果我们可以将小写字母（0 个或多个）插入模式串 `pattern` 中间（任意位置）得到待查询项 `queries[i]`，那么待查询项与给定模式串匹配。如果匹配，则对应答案为 `True`，否则为 `False`。

**要求**：将匹配结果存入由布尔值组成的答案列表中，并返回。

**说明**：

- $1 \le queries.length \le 100$。
- $1 \le queries[i].length \le 100$。
- $1 \le pattern.length \le 100$。
- 所有字符串都仅由大写和小写英文字母组成。

**示例**：

- 示例 1：

```python
输入：queries = ["FooBar","FooBarTest","FootBall","FrameBuffer","ForceFeedBack"], pattern = "FB"
输出：[true,false,true,true,false]
示例：
"FooBar" 可以这样生成："F" + "oo" + "B" + "ar"。
"FootBall" 可以这样生成："F" + "oot" + "B" + "all".
"FrameBuffer" 可以这样生成："F" + "rame" + "B" + "uffer".
```

- 示例 2：

```python
输入：queries = ["FooBar","FooBarTest","FootBall","FrameBuffer","ForceFeedBack"], pattern = "FoBa"
输出：[true,false,true,false,false]
解释：
"FooBar" 可以这样生成："Fo" + "o" + "Ba" + "r".
"FootBall" 可以这样生成："Fo" + "ot" + "Ba" + "ll".
```

## 解题思路

### 思路 1：字典树

构建一棵字典树，将 `pattern` 存入字典树中。

1. 对于 `queries[i]` 中的每个字符串。逐个字符与 `pattern` 进行匹配。
   1. 如果遇见小写字母，直接跳过。
   2. 如果遇见大写字母，但是不能匹配，返回 `False`。
   3. 如果遇见大写字母，且可以匹配，继续查找。
   4. 如果到达末尾仍然匹配，则返回 `True`。
2. 最后将所有结果存入答案数组中返回。

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
            if ord(ch) > 96:
                if ch not in cur.children:
                    continue
            else:
                if ch not in cur.children:
                    return False
            cur = cur.children[ch]

        return cur is not None and cur.isEnd

class Solution:
    def camelMatch(self, queries: List[str], pattern: str) -> List[bool]:
        trie_tree = Trie()
        trie_tree.insert(pattern)
        res = []
        for query in queries:
            res.append(trie_tree.search(query))
        return res
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(n \times |T| + |pattern|)$。其中 $n$ 是待查询项的数目，$|T|$  是最长的待查询项的字符串长度，$|pattern|$ 是字符串 `pattern` 的长度。
- **空间复杂度**：$O(|pattern|)$。

# [1025. 除数博弈](https://leetcode.cn/problems/divisor-game/)

- 标签：脑筋急转弯、数学、动态规划、博弈
- 难度：简单

## 题目链接

- [1025. 除数博弈 - 力扣](https://leetcode.cn/problems/divisor-game/)

## 题目大意

爱丽丝和鲍勃一起玩游戏，他们轮流行动。爱丽丝先手开局。最初，黑板上有一个数字 `n`。在每个玩家的回合，玩家需要执行以下操作：

- 选出任一 `x`，满足 `0 < x < n` 且 `n % x == 0`。
- 用 `n - x` 替换黑板上的数字 `n` 。
- 如果玩家无法执行这些操作，就会输掉游戏。

只有在爱丽丝在游戏中取得胜利时才返回 `True`，否则返回 `False`。假设两个玩家都以最佳状态参与游戏。

## 解题思路

- 如果 `n` 为奇数，则 `n` 的约数必然都是奇数；如果 `n` 为偶数，则 `n` 的约数可能为奇数也可能为偶数。
- 无论 `n` 为奇数还是偶数，都可以选择 `1` 作为约数。
- 无论 `n` 初始为多大的数，游戏到最终只能到 `n == 2` 结束，只要谁先到 `n == 2`，谁就赢得胜利。
- 当初始 `n` 为偶数时，爱丽丝只要一直选 `1`，那么鲍勃必然会一直面临 `n` 为奇数的情况，这样最后爱丽丝肯定能先到 `n == 2`，稳赢。
- 当初始 `n` 为奇数时，因为奇数的约数只能是奇数，奇数 - 奇数 必然是偶数，所以给鲍勃的数一定是偶数，鲍勃只需一直选 `1` 就会稳赢，此时爱丽丝稳输。

所以，当 `n` 为偶数时，爱丽丝稳赢。当 `n` 为奇数时，爱丽丝稳输。

## 代码

```python
class Solution:
    def divisorGame(self, n: int) -> bool:
        return n & 1 == 0
```

# [1028. 从先序遍历还原二叉树](https://leetcode.cn/problems/recover-a-tree-from-preorder-traversal/)

- 标签：树、深度优先搜索、字符串、二叉树
- 难度：困难

## 题目链接

- [1028. 从先序遍历还原二叉树 - 力扣](https://leetcode.cn/problems/recover-a-tree-from-preorder-traversal/)

## 题目大意

对一棵二叉树进行深度优先搜索。在遍历的过程中，遇到节点，先输出与该节点深度相同数量的短线，再输出该节点的值。如果节点深度为 `D`，则子节点深度为 `D + 1`。根节点的深度为 `0`。如果节点只有一个子节点，则该子节点一定为左子节点。

现在给定深度优先搜索输出的字符串 `traversal`。

要求：还原二叉树，并返回其根节点 `root`。

## 解题思路

用栈存储需要构建子树的节点。并记录下上一节点深度和当前节点深度。

然后遍历深度优先搜索的输出字符串。

- 先将开始部分的数字作为根节点值，构建一个根节点 `root`，并将根节点插入到栈中。
- 如果遇到 `-`，则更新当前节点深度。

- 然后如果遇到数字，则将数字逐位转为整数。并且在最后进行判断。
  - 如果当前节点深度 > 前一节点深度：
    - 将栈顶节点出栈。
    - 构建一个新节点，值为当前整数。将新节点插入到栈顶节点的左子树上。
    - 将当前节点和新节点插入到栈中。
  - 如果当前节点深度 <= 前一节点深度：
    - 将当前节点深度个数的节点从栈中弹出。
    - 构建一个新节点，值为当前整数。并将新节点插入到最后弹出节点的右子树上。
    - 将当前节点和新节点插入到栈中。
- 最后输出根节点 `root`。

## 代码

```python
class Solution:
    def recoverFromPreorder(self, traversal: str) -> Optional[TreeNode]:
        stack = []

        index, num = 0, 0
        pre_level, cur_level = 0, 0

        size = len(traversal)
        while index < size and traversal[index] != '-':
            num = num * 10 + ord(traversal[index]) - ord('0')
            index += 1

        root = TreeNode(num)
        stack.append(root)

        while index < size:
            if traversal[index] == '-':
                cur_level += 1
                index += 1
            else:
                num = 0
                while index < size and traversal[index] != '-':
                    num = num * 10 + ord(traversal[index]) - ord('0')
                    index += 1

                if cur_level > pre_level:
                    node = stack.pop()
                    node.left = TreeNode(num)
                    stack.append(node)
                    stack.append(node.left)
                    pre_level = cur_level
                    cur_level = 0
                else:
                    while len(stack) > cur_level:
                        stack.pop()
                    node = stack.pop()
                    node.right = TreeNode(num)
                    stack.append(node)
                    stack.append(node.right)
                    pre_level = cur_level
                    cur_level = 0
        return root
```

# [1029. 两地调度](https://leetcode.cn/problems/two-city-scheduling/)

- 标签：贪心、数组、排序
- 难度：中等

## 题目链接

- [1029. 两地调度 - 力扣](https://leetcode.cn/problems/two-city-scheduling/)

## 题目大意

**描述**：公司计划面试 `2 * n` 人。给你一个数组 `costs`，其中 `costs[i] = [aCosti, bCosti]`，表示第 `i` 人飞往 `a` 市的费用为 `aCosti` ，飞往 `b` 市的费用为 `bCosti`。

**要求**：返回将每个人都飞到 `a`、`b` 中某座城市的最低费用，要求每个城市都有 `n` 人抵达。

**说明**：

- $2 * n == costs.length$。
- $2 \le costs.length \le 100$。
- $costs.length$ 为偶数。
- $1 \le aCosti, bCosti \le 1000$。

**示例**：

- 示例 1：

```python
输入：costs = [[10,20],[30,200],[400,50],[30,20]]
输出：110
解释：
第一个人去 a 市，费用为 10。
第二个人去 a 市，费用为 30。
第三个人去 b 市，费用为 50。
第四个人去 b 市，费用为 20。

最低总费用为 10 + 30 + 50 + 20 = 110，每个城市都有一半的人在面试。
```

## 解题思路

### 思路 1：贪心算法

我们先假设所有人都去了城市 `a`。然后令一半的人再去城市 `b`。现在的问题就变成了，让一半的人改变城市去向，从原本的 `a` 城市改成 `b` 城市的最低费用为多少。

已知第 `i` 个人更换去向的费用为「去城市 `b` 的费用 - 去城市 `a` 的费用」。所以我们可以根据「去城市 `b` 的费用 - 去城市 `a` 的费用」对数组 `costs` 进行排序，让前 `n` 个改变方向去城市 `b`，后 `n` 个人去城市 `a`。

最后统计所有人员的费用，将其返回即可。

### 思路 1：贪心算法代码

```python
class Solution:
    def twoCitySchedCost(self, costs: List[List[int]]) -> int:
        costs.sort(key=lambda x:x[1] - x[0])
        cost = 0
        size = len(costs) // 2
        for i in range(size):
            cost += costs[i][ 1]
            cost += costs[i + size][0]

        return cost
```
# [1034. 边界着色](https://leetcode.cn/problems/coloring-a-border/)

- 标签：深度优先搜索、广度优先搜索、数组、矩阵
- 难度：中等

## 题目链接

- [1034. 边界着色 - 力扣](https://leetcode.cn/problems/coloring-a-border/)

## 题目大意

给定一个二维整数矩阵 `grid`，其中 `grid[i][j]` 表示矩阵第 `i` 行、第 `j` 列上网格块的颜色值。再给定一个起始位置 `(row, col)`，以及一个目标颜色 `color`。

要求：对起始位置 `(row, col)` 所在的连通分量边界填充颜色为 `color`。并返回最终的二维整数矩阵 `grid`。

- 连通分量：当两个相邻（上下左右四个方向上）网格块的颜色值相同时，它们属于同一连通分量。
- 连通分量边界：当前连通分量最外圈的所有网格块，这些网格块与连通分量的颜色相同，与其他周围网格块颜色不同。边界上的网格块也是连通分量边界。

## 解题思路

深度优先搜索。使用二维数组 `visited` 标记访问过的节点。遍历上、下、左、右四个方向上的点。如果下一个点位置越界，或者当前位置与下一个点位置颜色不一样，则对该节点进行染色。

在遍历的过程中注意使用 `visited` 标记访问过的节点，以免重复遍历。

## 代码

```python
class Solution:
    directs = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    def dfs(self, grid, i, j, origin_color, color, visited):
        rows, cols = len(grid), len(grid[0])

        for direct in self.directs:
            new_i = i + direct[0]
            new_j = j + direct[1]

            # 下一个位置越界，则当前点在边界，对其进行着色
            if new_i < 0 or new_i >= rows or new_j < 0 or new_j >= cols:
                grid[i][j] = color
                continue

            # 如果访问过，则跳过
            if visited[new_i][new_j]:
                continue

            # 如果下一个位置颜色与当前颜色相同，则继续搜索
            if grid[new_i][new_j] == origin_color:
                visited[new_i][new_j] = True
                self.dfs(grid, new_i, new_j, origin_color, color, visited)
            # 下一个位置颜色与当前颜色不同，则当前位置为连通区域边界，对其进行着色
            else:
                grid[i][j] = color


    def colorBorder(self, grid: List[List[int]], row: int, col: int, color: int) -> List[List[int]]:
        if not grid:
            return grid

        rows, cols = len(grid), len(grid[0])
        visited = [[False for _ in range(cols)] for _ in range(rows)]
        visited[row][col] = True

        self.dfs(grid, row, col, grid[row][col], color, visited)

        return grid
```

# [1035. 不相交的线](https://leetcode.cn/problems/uncrossed-lines/)

- 标签：数组、动态规划
- 难度：中等

## 题目链接

- [1035. 不相交的线 - 力扣](https://leetcode.cn/problems/uncrossed-lines/)

## 题目大意

有两条独立平行的水平线，按照给定的顺序写下 `nums1` 和 `nums2` 的整数。

现在，我们可以绘制一些直线，只要满足以下要求：

- `nums1[i] == nums2[j]`。
- 绘制的直线不与其他任何直线相交。

例如：![](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2019/04/28/142.png)

现在要求：计算出能绘制的最大直线数目。

## 解题思路

动态规划求解。

定义状态 `dp[i][j]` 表示：`nums1` 中前 `i` 个数与 `nums2` 中前 `j` 个数的最大连接数，则：

状态转移方程为：

- 如果 `nums1[i] == nums[j]`，则 `nums1[i]` 与 `nums2[j]` 可连线，此时 `dp[i][j] = dp[i - 1][j - 1] + 1`。
- 如果 `nums1[i] != nums[j]`，则 `nums1[i]` 与 `nums2[j]` 不可连线，此时最大连线数取决于 `dp[i - 1][j]` 和 `dp[i][j - 1]` 的较大值，即：`dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])`。

最后输出 `dp[size1][size2]` 即可。

## 代码

```python
class Solution:
    def maxUncrossedLines(self, nums1: List[int], nums2: List[int]) -> int:
        size1 = len(nums1)
        size2 = len(nums2)
        dp = [[0 for _ in range(size2 + 1)] for _ in range(size1 + 1)]
        for i in range(1, size1 + 1):
            for j in range(1, size2 + 1):
                if nums1[i - 1] == nums2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        return dp[size1][size2]
```

# [1037. 有效的回旋镖](https://leetcode.cn/problems/valid-boomerang/)

- 标签：几何、数组、数学
- 难度：简单

## 题目链接

- [1037. 有效的回旋镖 - 力扣](https://leetcode.cn/problems/valid-boomerang/)

## 题目大意

**描述**：给定一个数组 $points$，其中 $points[i] = [xi, yi]$ 表示平面上的一个点。

**要求**：如果这些点构成一个回旋镖，则返回 `True`，否则，则返回 `False`。

**说明**：

- **回旋镖**：定义为一组三个点，这些点各不相同且不在一条直线上。
- $points.length == 3$。
- $points[i].length == 2$。
- $0 \le xi, yi \le 100$。

**示例**：

- 示例 1：

```python
输入：points = [[1,1],[2,3],[3,2]]
输出：True
```

- 示例 2：

```python
输入：points = [[1,1],[2,2],[3,3]]
输出：False
```

## 解题思路

### 思路 1：

设三点坐标为 $A = (x1, y1)$，$B = (x2, y2)$，$C = (x3, y3)$，则向量 $\overrightarrow{AB} = (x2 - x1, y2 - y1)$，$\overrightarrow{BC} = (x3 - x2, y3 - y2)$。

如果三点共线，则应满足：$\overrightarrow{AB} \times \overrightarrow{BC} = (x2 − x1) \times (y3 − y2) - (x3 − x2) \times (y2 − y1) = 0$。

如果三点不共线，则应满足：$\overrightarrow{AB} \times \overrightarrow{BC} = (x2 − x1) \times (y3 − y2) - (x3 − x2) \times (y2 − y1) \ne 0$。

### 思路 1：代码

```python
class Solution:
    def isBoomerang(self, points: List[List[int]]) -> bool:
        x1, y1 = points[0]
        x2, y2 = points[1]
        x3, y3 = points[2]
        cross1 = (x2 - x1) * (y3 - y2)
        cross2 = (x3 - x2) * (y2 - y1)
        return cross1 - cross2 != 0
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(1)$。
- **空间复杂度**：$O(1)$。
# [1038. 从二叉搜索树到更大和树](https://leetcode.cn/problems/binary-search-tree-to-greater-sum-tree/)

- 标签：树、深度优先搜索、二叉搜索树、二叉树
- 难度：中等

## 题目链接

- [1038. 从二叉搜索树到更大和树 - 力扣](https://leetcode.cn/problems/binary-search-tree-to-greater-sum-tree/)

## 题目大意

给定一棵二叉搜索树（BST）的根节点，且二叉搜索树的节点值各不相同。

要求：将它的每个节点的值替换成树中大于或者等于该节点值的所有节点值之和。

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

    def bstToGst(self, root: TreeNode) -> TreeNode:
        self.pre = 0
        self.createBinaryTree(root)
        return root
```

# [1039. 多边形三角剖分的最低得分](https://leetcode.cn/problems/minimum-score-triangulation-of-polygon/)

- 标签：数组、动态规划
- 难度：中等

## 题目链接

- [1039. 多边形三角剖分的最低得分 - 力扣](https://leetcode.cn/problems/minimum-score-triangulation-of-polygon/)

## 题目大意

**描述**：有一个凸的 $n$ 边形，其每个顶点都有一个整数值。给定一个整数数组 $values$，其中 $values[i]$ 是第 $i$ 个顶点的值（即顺时针顺序）。

现在要将 $n$ 边形剖分为 $n - 2$ 个三角形，对于每个三角形，该三角形的值是顶点标记的乘积，$n$ 边形三角剖分的分数是进行三角剖分后所有 $n - 2$ 个三角形的值之和。

**要求**：返回多边形进行三角剖分可以得到的最低分。

**说明**：

- $n == values.length$。
- $3 \le n \le 50$。
- $1 \le values[i] \le 100$。

**示例**：

- 示例 1：

![](https://assets.leetcode.com/uploads/2021/02/25/shape1.jpg)

```python
输入：values = [1,2,3]
输出：6
解释：多边形已经三角化，唯一三角形的分数为 6。
```

- 示例 2：

![](https://assets.leetcode.com/uploads/2021/02/25/shape2.jpg)

```python
输入：values = [3,7,4,5]
输出：144
解释：有两种三角剖分，可能得分分别为：3*7*5 + 4*5*7 = 245，或 3*4*5 + 3*4*7 = 144。最低分数为 144。
```

## 解题思路

### 思路 1：动态规划

对于 $0 \sim n - 1$ 个顶点组成的凸多边形进行三角剖分，我们可以在 $[0, n - 1]$ 中任选 $1$ 个点 $k$，从而将凸多边形划分为：

1. 顶点 $0 \sim k$ 组成的凸多边形。
2. 顶点 $0$、$k$、$n - 1$ 组成的三角形。
3. 顶点 $k \sim n - 1$  组成的凸多边形。

对于顶点 $0$、$k$、$n - 1$ 组成的三角形，我们可以直接计算对应的三角剖分分数为 $values[0] \times values[k] \times values[n - 1]$。

而对于顶点 $0 \sim k$ 组成的凸多边形和顶点 $k \sim n - 1$  组成的凸多边形，我们可以利用递归或者动态规划的思想，定义一个 $dp[i][j]$ 用于计算顶点 $i$ 到顶点 $j$ 组成的多边形三角剖分的最小分数。

具体做法如下：

###### 1. 划分阶段

按照区间长度进行阶段划分。

###### 2. 定义状态

定义状态 $dp[i][j]$ 表示为：区间 $[i, j]$ 内三角剖分后的最小分数。

###### 3. 状态转移方程

对于区间 $[i, j]$，枚举分割点 $k$，最小分数为 $min(dp[i][k] + dp[k][j] + values[i] \times values[k] \times values[j])$，即：$dp[i][j] = min(dp[i][k] + dp[k][j] + values[i] \times values[k] \times values[j])$。

###### 4. 初始条件

- 默认情况下，所有区间 $[i, j]$ 的最小分数为无穷大。
- 当区间 $[i, j]$ 长度小于 $3$ 时，无法进行三角剖分，其最小分数为 $0$。
- 当区间 $[i, j]$ 长度等于 $3$ 时，其三角剖分的最小分数为 $values[i] * values[i + 1] * values[i + 2]$。

###### 5. 最终结果

根据我们之前定义的状态，$dp[i][j]$ 表示为：区间 $[i, j]$ 内三角剖分后的最小分数。。 所以最终结果为 $dp[0][size - 1]$。

### 思路 1：代码

```python
class Solution:
    def minScoreTriangulation(self, values: List[int]) -> int:
        size = len(values)
        dp = [[float('inf') for _ in range(size)] for _ in range(size)]
        for l in range(1, size + 1):
            for i in range(size):
                j = i + l - 1
                if j >= size:
                    break
                if l < 3:
                    dp[i][j] = 0
                elif l == 3:
                    dp[i][j] = values[i] * values[i + 1] * values[i + 2]
                else:
                    for k in range(i + 1, j):
                        dp[i][j] = min(dp[i][j], dp[i][k] + dp[k][j] + values[i] * values[j] * values[k])

        return dp[0][size - 1]
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(n^3)$，其中 $n$ 为顶点个数。
- **空间复杂度**：$O(n^2)$。

# [1041. 困于环中的机器人](https://leetcode.cn/problems/robot-bounded-in-circle/)

- 标签：数学、字符串、模拟
- 难度：中等

## 题目链接

- [1041. 困于环中的机器人 - 力扣](https://leetcode.cn/problems/robot-bounded-in-circle/)

## 题目大意

**描述**：在无限的平面上，机器人最初位于 $(0, 0)$ 处，面朝北方。注意:

- 北方向 是 $y$ 轴的正方向。
- 南方向 是 $y$ 轴的负方向。
- 东方向 是 $x$ 轴的正方向。
- 西方向 是 $x$ 轴的负方向。

机器人可以接受下列三条指令之一：

- `"G"`：直走 $1$ 个单位
- `"L"`：左转 $90$ 度
- `"R"`：右转 $90$ 度

给定一个字符串 $instructions$，机器人按顺序执行指令 $instructions$，并一直重复它们。

**要求**：只有在平面中存在环使得机器人永远无法离开时，返回 $True$。否则，返回 $False$。

**说明**：

- $1 \le instructions.length \le 100$。
- $instructions[i]$ 仅包含 `'G'`，`'L'`，`'R'`。

**示例**：

- 示例 1：

```python
输入：instructions = "GGLLGG"
输出：True
解释：机器人最初在(0,0)处，面向北方。
“G”:移动一步。位置:(0,1)方向:北。
“G”:移动一步。位置:(0,2).方向:北。
“L”:逆时针旋转90度。位置:(0,2).方向:西。
“L”:逆时针旋转90度。位置:(0,2)方向:南。
“G”:移动一步。位置:(0,1)方向:南。
“G”:移动一步。位置:(0,0)方向:南。
重复指令，机器人进入循环:(0,0)——>(0,1)——>(0,2)——>(0,1)——>(0,0)。
在此基础上，我们返回 True。
```

- 示例 2：

```python
输入：instructions = "GG"
输出：False
解释：机器人最初在(0,0)处，面向北方。
“G”:移动一步。位置:(0,1)方向:北。
“G”:移动一步。位置:(0,2).方向:北。
重复这些指示，继续朝北前进，不会进入循环。
在此基础上，返回 False。
```

## 解题思路

### 思路 1：模拟

设定初始位置为 $(0, 0)$，初始方向 $direction = 0$，假设按照给定字符串 $instructions$ 执行一遍之后，位于 $(x, y)$ 处，且方向为 $direction$，则可能出现的所有情况为：

1. 方向不变（$direction == 0$），且 $(x, y) == (0, 0)$，则会一直在原点，无法走出去。
2. 方向不变（$direction == 0$），且 $(x, y) \ne (0, 0)$，则可以走出去。
3. 方向相反（$direction == 2$），无论是否产生位移，则再执行 $1$ 遍将会回到原点。
4. 方向逆时针 / 顺时针改变 $90°$（$direction == 1 \text{ or } 3$），无论是否产生位移，则再执行 $3$ 遍将会回到原点。

综上所述，最多模拟 $4$ 次即可知道能否回到原点。

从上面也可以等出结论：如果不产生位移，则一定会回到原点。如果改变方向，同样一定会回到原点。

我们只需要根据以上结论，按照 $instructions$ 执行一遍之后，通过判断是否产生位移和改变方向，即可判断是否一定会回到原点。

### 思路 1：代码

```Python
class Solution:
    def isRobotBounded(self, instructions: str) -> bool:
        # 分别代表北、东、南、西
        directions = [(0, 1), (-1, 0), (0, -1), (1, 0)]
        x, y = 0, 0
        # 初始方向为北
        direction = 0
        for step in instructions:
            if step == 'G':
                x += directions[direction][0]
                y += directions[direction][1]
            elif step == 'L':
                direction = (direction + 1) % 4
            else:
                direction = (direction + 3) % 4
        
        return (x == 0 and y == 0) or direction != 0
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(n)$，其中 $n$ 为字符串 $instructions$ 的长度。
- **空间复杂度**：$O(1)$。
# [1047. 删除字符串中的所有相邻重复项](https://leetcode.cn/problems/remove-all-adjacent-duplicates-in-string/)

- 标签：栈、字符串
- 难度：简单

## 题目链接

- [1047. 删除字符串中的所有相邻重复项 - 力扣](https://leetcode.cn/problems/remove-all-adjacent-duplicates-in-string/)

## 题目大意

给定一个全部由小写字母组成的字符串 S，重复的删除相邻且相同的字母，直到相邻字母不再有相同的。

比如 "abbaca"。先删除相邻且相同的字母 "bb"，变为 "aaca"，再删除相邻且相同的字母 "aa"，变为 "ca"，无相邻且相同的字母，即 "ca" 为最终结果。

## 解题思路

跟括号匹配有点类似。我们可以利用「栈」来做这道题。遍历字符串，如果当前字符与栈顶字符相同，则将栈顶所有相同字符删除，否则就将当前字符入栈。

## 代码

```python
class Solution:
    def removeDuplicates(self, S: str) -> str:
        stack = []
        for ch in S:
            if stack and stack[-1] == ch:
                stack.pop()
            else:
                stack.append(ch)
        return "".join(stack)
```

# [1049. 最后一块石头的重量 II](https://leetcode.cn/problems/last-stone-weight-ii/)

- 标签：数组、动态规划
- 难度：中等

## 题目链接

- [1049. 最后一块石头的重量 II - 力扣](https://leetcode.cn/problems/last-stone-weight-ii/)

## 题目大意

**描述**：有一堆石头，用整数数组 $stones$ 表示，其中 $stones[i]$ 表示第 $i$​ 块石头的重量。每一回合，从石头中选出任意两块石头，将这两块石头一起粉碎。假设石头的重量分别为 $x$ 和 $y$。且 $x \le y$，则结果如下：

- 如果 $x = y$，则两块石头都会被完全粉碎；
- 如果 $x < y$，则重量为 $x$ 的石头被完全粉碎，而重量为 $y$ 的石头新重量为 $y - x$。

**要求**：最后，最多只会剩下一块石头，返回此石头的最小可能重量。如果没有石头剩下，则返回 $0$。

**说明**：

- $1 \le stones.length \le 30$。
- $1 \le stones[i] \le 100$。

**示例**：

- 示例 1：

```python
输入：stones = [2,7,4,1,8,1]
输出：1
解释：
组合 2 和 4，得到 2，所以数组转化为 [2,7,1,8,1]，
组合 7 和 8，得到 1，所以数组转化为 [2,1,1,1]，
组合 2 和 1，得到 1，所以数组转化为 [1,1,1]，
组合 1 和 1，得到 0，所以数组转化为 [1]，这就是最优值。
```

- 示例 2：

```python
输入：stones = [31,26,33,21,40]
输出：5
```

## 解题思路

### 思路 1：动态规划

选取两块石头，重新放回去的重量是两块石头的差值绝对值。重新放回去的石头还会进行选取，然后进行粉碎，直到最后只剩一块或者不剩石头。

这个问题其实可以转化为：把一堆石头尽量平均的分成两对，求两堆石头重量差的最小值。

这就和「[0416. 分割等和子集](https://leetcode.cn/problems/partition-equal-subset-sum/)」有点相似。两堆石头的重量要尽可能的接近数组总数量和的一半。

进一步可以变为：「0-1 背包问题」。

1. 假设石头总重量和为 $sum$，将一堆石头放进载重上限为 $sum / 2$ 的背包中，获得的最大价值为 $max\underline{\hspace{0.5em}}weight$（即其中一堆石子的重量）。另一堆石子的重量为 $sum - max\underline{\hspace{0.5em}}weight$。
2. 则两者的差值为 $sum - 2 \times max\underline{\hspace{0.5em}}weight$，即为答案。

###### 1. 划分阶段

按照石头的序号进行阶段划分。

###### 2. 定义状态

定义状态 $dp[w]$ 表示为：将石头放入载重上限为 $w$ 的背包中可以获得的最大价值。

###### 3. 状态转移方程

$dp[w] = max \lbrace dp[w], dp[w - stones[i - 1]] + stones[i - 1] \rbrace$。

###### 4. 初始条件

- 无论背包载重上限为多少，只要不选择石头，可以获得的最大价值一定是 $0$，即 $dp[w] = 0, 0 \le w \le W$。

###### 5. 最终结果

根据我们之前定义的状态，$dp[w]$ 表示为：将石头放入载重上限为 $w$ 的背包中可以获得的最大价值，即第一堆石头的价值为 $dp[size]$，第二堆石头的价值为 $sum - dp[size]$，最终答案为两者的差值，即 $sum - dp[size] \times 2$。

### 思路 1：代码

```python
class Solution:
    def lastStoneWeightII(self, stones: List[int]) -> int:
        W = 1500
        size = len(stones)
        dp = [0 for _ in range(W + 1)]
        target = sum(stones) // 2
        for i in range(1, size + 1):
            for w in range(target, stones[i - 1] - 1, -1):
                dp[w] = max(dp[w], dp[w - stones[i - 1]] + stones[i - 1])

        return sum(stones) - dp[target] * 2
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(n \times W)$，其中 $n$ 为数组 $stones$ 的元素个数，$W$ 为数组 $stones$ 中元素和的一半。
- **空间复杂度**：$O(W)$。
# [1051. 高度检查器](https://leetcode.cn/problems/height-checker/)

- 标签：数组、计数排序、排序
- 难度：简单

## 题目链接

- [1051. 高度检查器 - 力扣](https://leetcode.cn/problems/height-checker/)

## 题目大意

**描述**：学校打算为全体学生拍一张年度纪念照。根据要求，学生需要按照 非递减 的高度顺序排成一行。

排序后的高度情况用整数数组 $expected$ 表示，其中 $expected[i]$ 是预计排在这一行中第 $i$ 位的学生的高度（下标从 $0$ 开始）。

给定一个整数数组 $heights$ ，表示当前学生站位的高度情况。$heights[i]$ 是这一行中第 $i$ 位学生的高度（下标从 $0$ 开始）。

**要求**：返回满足 $heights[i] \ne expected[i]$ 的下标数量 。

**说明**：

- $1 \le heights.length \le 100$。
- $1 \le heights[i] \le 100$。

**示例**：

- 示例 1：

```python
输入：heights = [1,1,4,2,1,3]
输出：3 
解释：
高度：[1,1,4,2,1,3]
预期：[1,1,1,2,3,4]
下标 2 、4 、5 处的学生高度不匹配。
```

- 示例 2：

```python
输入：heights = [5,1,2,3,4]
输出：5
解释：
高度：[5,1,2,3,4]
预期：[1,2,3,4,5]
所有下标的对应学生高度都不匹配。
```

## 解题思路

### 思路 1：排序算法

1. 将数组 $heights$ 复制一份，记为 $expected$。
2. 对数组 $expected$ 进行排序。
3. 排序之后，对比并统计 $heights[i] \ne expected[i]$ 的下标数量，记为 $ans$。
4. 返回 $ans$。

### 思路 1：代码

```Python
class Solution:
    def heightChecker(self, heights: List[int]) -> int:
        expected = sorted(heights)

        ans = 0
        for i in range(len(heights)):
            if expected[i] != heights[i]:
                ans += 1
        return ans
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(n \times \log n)$，其中 $n$ 为数组 $heights$ 的长度。
- **空间复杂度**：$O(n)$。

### 思路 2：计数排序

题目中 $heights[i]$ 的数据范围为 $[1, 100]$，所以我们可以使用计数排序。

### 思路 2：代码

```python
class Solution:
    def heightChecker(self, heights: List[int]) -> int:
        # 待排序数组中最大值元素 heights_max = 100 和最小值元素 heights_min = 1
        heights_min, heights_max = 1, 100
        # 定义计数数组 counts，大小为 最大值元素 - 最小值元素 + 1
        size = heights_max - heights_min + 1
        counts = [0 for _ in range(size)]
		
        # 统计值为 height 的元素出现的次数
        for height in heights:
            counts[height - heights_min] += 1

        ans = 0
        idx = 0
        # 从小到大遍历 counts 的元素值范围
        for height in range(heights_min, heights_max + 1):
            while counts[height - heights_min]:
                # 对于每个元素值，判断是否与对应位置上的 heights[idx] 相等
                if heights[idx] != height:
                    ans += 1
                idx += 1
                counts[height - heights_min] -= 1
        
        return ans
```

### 思路 2：复杂度分析

- **时间复杂度**：$O(n + k)$，其中 $n$ 为数组 $heights$ 的长度，$k$ 为数组 $heights$ 的值域范围。
- **空间复杂度**：$O(k)$。

# [1052. 爱生气的书店老板](https://leetcode.cn/problems/grumpy-bookstore-owner/)

- 标签：数组、滑动窗口
- 难度：中等

## 题目链接

- [1052. 爱生气的书店老板 - 力扣](https://leetcode.cn/problems/grumpy-bookstore-owner/)

## 题目大意

**描述**：书店老板有一家店打算试营业 $len(customers)$ 分钟。每一分钟都有一些顾客 $customers[i]$ 会进入书店，这些顾客会在这一分钟结束后离开。

在某些时候，书店老板会生气。如果书店老板在第 $i$ 分钟生气，则 `grumpy[i] = 1`，如果第 $i$ 分钟不生气，则 `grumpy[i] = 0`。当书店老板生气时，这一分钟的顾客会不满意。当书店老板不生气时，这一分钟的顾客是满意的。

假设老板知道一个秘密技巧，能保证自己连续 $minutes$ 分钟不生气，但只能使用一次。

现在给定代表每分钟进入书店的顾客数量的数组 $customes$，和代表老板生气状态的数组 $grumpy$，以及老板保证连续不生气的分钟数 $minutes$。

**要求**：计算出试营业下来，最多有多少客户能够感到满意。

**说明**：

- $n == customers.length == grumpy.length$。
- $1 \le minutes \le n \le 2 \times 10^4$。
- $0 \le customers[i] \le 1000$。
- $grumpy[i] == 0 \text{ or } 1$。

**示例**：

- 示例 1：

```python
输入：customers = [1,0,1,2,1,1,7,5], grumpy = [0,1,0,1,0,1,0,1], minutes = 3
输出：16
解释：书店老板在最后 3 分钟保持冷静。
感到满意的最大客户数量 = 1 + 1 + 1 + 1 + 7 + 5 = 16.
```

- 示例 2：

```python
输入：customers = [1], grumpy = [0], minutes = 1
输出：1
```

## 解题思路

### 思路 1：滑动窗口

固定长度的滑动窗口题目。我们可以维护一个窗口大小为 $minutes$ 的滑动窗口。使用 $window_count$ 记录当前窗口内生气的顾客人数。然后滑动求出窗口中最大顾客数，然后累加上老板未生气时的顾客数，就是答案。具体做法如下：

1. $ans$ 用来维护答案数目。$window\underline{\hspace{0.5em}}count$ 用来维护窗口中生气的顾客人数。
2. $left$ 、$right$ 都指向序列的第一个元素，即：`left = 0`，`right = 0`。
3. 如果书店老板生气，则将这一分钟的顾客数量加入到 $window\underline{\hspace{0.5em}}count$ 中，然后向右移动 $right$。
4. 当窗口元素个数大于 $minutes$ 时，即：$right - left + 1 > count$ 时，如果最左侧边界老板处于生气状态，则向右移动 $left$，从而缩小窗口长度，即 `left += 1`，使得窗口大小始终保持为小于 $minutes$。
5. 重复 $3 \sim 4$ 步，直到 $right$ 到达数组末尾。
6. 然后累加上老板未生气时的顾客数，最后输出答案。

### 思路 1：代码

```python
class Solution:
    def maxSatisfied(self, customers: List[int], grumpy: List[int], minutes: int) -> int:
        left = 0
        right = 0
        window_count = 0
        ans = 0

        while right < len(customers):
            if grumpy[right] == 1:
                window_count += customers[right]

            if right - left + 1 > minutes:
                if grumpy[left] == 1:
                    window_count -= customers[left]
                left += 1

            right += 1
            ans = max(ans, window_count)

        for i in range(len(customers)):
            if grumpy[i] == 0:
                ans += customers[i]
        return ans
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(n)$，其中 $n$ 为数组 $coustomer$、$grumpy$ 的长度。
- **空间复杂度**：$O(1)$。

# [1065. 字符串的索引对](https://leetcode.cn/problems/index-pairs-of-a-string/)

- 标签：字典树、数组、字符串、排序
- 难度：简单

## 题目链接

- [1065. 字符串的索引对 - 力扣](https://leetcode.cn/problems/index-pairs-of-a-string/)

## 题目大意

给定字符串 `text` 和单词列表 `words`。

要求：在 `text` 中找出所有属于单词列表 `words` 中的单词，并返回该单词在 `text` 中的索引对位置 `[i, j]`。将所有索引对存入列表中返回，并且返回的索引对可以交叉。

## 解题思路

构建字典树，将所有单词存入字典树中。

然后一重循环遍历 `text`，表示从第 `i` 位置开始的字符串 `text[i:]`。然后在字符串前缀中搜索对应的单词，将所有符合要求的单词末尾位置存入列表中，返回所有位置列表。对于列表中每个单词末尾位置 `index` 和 `text` 来说，每个 `[i, i + index]` 都构成了单词在 `text` 中的索引对位置，将其存入答案数组并返回即可。

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
                res.append(i)

        return res

class Solution:
    def indexPairs(self, text: str, words: List[str]) -> List[List[int]]:
        trie_tree = Trie()
        for word in words:
            trie_tree.insert(word)

        res = []
        for i in range(len(text)):
            for index in trie_tree.search(text[i:]):
                res.append([i, i + index])
        return res
```

# [1079. 活字印刷](https://leetcode.cn/problems/letter-tile-possibilities/)

- 标签：哈希表、字符串、回溯、计数
- 难度：中等

## 题目链接

- [1079. 活字印刷 - 力扣](https://leetcode.cn/problems/letter-tile-possibilities/)

## 题目大意

**描述**：给定一个代表活字字模的字符串 $tiles$，其中 $tiles[i]$ 表示第 $i$ 个字模上刻的字母。

**要求**：返回你可以印出的非空字母序列的数目。

**说明**：

- 本题中，每个活字字模只能使用一次。
- $1 <= tiles.length <= 7$。
- $tiles$ 由大写英文字母组成。

**示例**：

- 示例 1：

```python
输入："AAB"
输出：8
解释：可能的序列为 "A", "B", "AA", "AB", "BA", "AAB", "ABA", "BAA"。
```

- 示例 2：

```python
输入："AAABBC"
输出：188
```

## 解题思路

### 思路 1：哈希表 + 回溯算法

1. 使用哈希表存储每个字符的个数。
2. 然后依次从哈希表中取出对应字符，统计排列个数，并进行回溯。
3. 如果当前字符个数为 $0$，则不再进行回溯。
4. 回溯之后将状态回退。

### 思路 1：代码

```python
class Solution:
    ans = 0
    def backtrack(self, tile_map):
        for key, value in tile_map.items():
            if value == 0:
                continue
            self.ans += 1
            tile_map[key] -= 1
            self.backtrack(tile_map)
            tile_map[key] += 1

    def numTilePossibilities(self, tiles: str) -> int:
        tile_map = dict()
        for tile in tiles:
            if tile not in tile_map:
                tile_map[tile] = 1
            else:
                tile_map[tile] += 1

        self.backtrack(tile_map)

        return self.ans
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(n \times n!)$，其中 $n$ 表示 $tiles$  的长度最小值。
- **空间复杂度**：$O(n)$。

# [1081. 不同字符的最小子序列](https://leetcode.cn/problems/smallest-subsequence-of-distinct-characters/)

- 标签：栈、贪心、字符串、单调栈
- 难度：中等

## 题目链接

- [1081. 不同字符的最小子序列 - 力扣](https://leetcode.cn/problems/smallest-subsequence-of-distinct-characters/)

## 题目大意

**描述**：给定一个字符串 `s`。

**要求**：去除字符串中重复的字母，使得每个字母只出现一次。需要保证 **「返回结果的字典序最小（要求不能打乱其他字符的相对位置）」**。

**说明**：

- $1 \le s.length \le 10^4$。
- `s` 由小写英文字母组成。

**示例**：

- 示例 1：

```python
输入：s = "bcabc"
输出："abc"
```

- 示例 2：

```python
输入：s = "cbacdcbc"
输出："acdb"
```

## 解题思路

### 思路 1：哈希表 + 单调栈

针对题目的三个要求：去重、不能打乱其他字符顺序、字典序最小。我们来一一分析。

1. **去重**：可以通过 **「使用哈希表存储字母出现次数」** 的方式，将每个字母出现的次数统计起来，再遍历一遍，去除重复的字母。
2. **不能打乱其他字符顺序**：按顺序遍历，将非重复的字母存储到答案数组或者栈中，最后再拼接起来，就能保证不打乱其他字符顺序。
3. **字典序最小**：意味着字典序小的字母应该尽可能放在前面。
   1. 对于第 `i` 个字符 `s[i]` 而言，如果第 `0` ~ `i - 1` 之间的某个字符 `s[j]` 在 `s[i]` 之后不再出现了，那么 `s[j]` 必须放到 `s[i]` 之前。
   2. 而如果 `s[j]` 在之后还会出现，并且 `s[j]` 的字典序大于 `s[i]`，我们则可以先舍弃 `s[j]`，把 `s[i]` 尽可能的放到前面。后边再考虑使用 `s[j]` 所对应的字符。


要满足第 3 条需求，我们可以使用 **「单调栈」** 来解决。我们使用单调栈存储 `s[i]` 之前出现的非重复、并且字典序最小的字符序列。整个算法步骤如下：

1. 先遍历一遍字符串，用哈希表 `letter_counts` 统计出每个字母出现的次数。
2. 然后使用单调递减栈保存当前字符之前出现的非重复、并且字典序最小的字符序列。
3. 当遍历到 `s[i]` 时，如果 `s[i]` 没有在栈中出现过：
   1. 比较 `s[i]` 和栈顶元素 `stack[-1]` 的字典序。如果 `s[i]` 的字典序小于栈顶元素 `stack[-1]`，并且栈顶元素之后的出现次数大于 `0`，则将栈顶元素弹出。
   2. 然后继续判断 `s[i]` 和栈顶元素 `stack[-1]`，并且知道栈顶元素出现次数为 `0` 时停止弹出。此时将 `s[i]` 添加到单调栈中。
4. 从哈希表 `letter_counts` 中减去 `s[i]` 出现的次数，继续遍历。
5. 最后将单调栈中的字符依次拼接为答案字符串，并返回。

### 思路 1：代码

```python
class Solution:
    def removeDuplicateLetters(self, s: str) -> str:
        stack = []
        letter_counts = dict()
        for ch in s:
            if ch in letter_counts:
                letter_counts[ch] += 1
            else:
                letter_counts[ch] = 1

        for ch in s:
            if ch not in stack:
                while stack and ch < stack[-1] and stack[-1] in letter_counts and letter_counts[stack[-1]] > 0:
                    stack.pop()
                stack.append(ch)
            letter_counts[ch] -= 1

        return ''.join(stack)
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(n)$。
- **空间复杂度**：$O(|\sum|)$，其中 $\sum$ 为字符集合，$|\sum|$ 为字符种类个数。由于栈中字符不能重复，因此栈中最多有 $|\sum|$ 个字符。

## 参考资料

- 【题解】[去除重复数组 - 去除重复字母 - 力扣（LeetCode）](https://leetcode.cn/problems/remove-duplicate-letters/solution/qu-chu-zhong-fu-shu-zu-by-lu-shi-zhe-sokp/)# [1089. 复写零](https://leetcode.cn/problems/duplicate-zeros/)

- 标签：数组、双指针
- 难度：简单

## 题目链接

- [1089. 复写零 - 力扣](https://leetcode.cn/problems/duplicate-zeros/)

## 题目大意

**描述**：给定搞一个长度固定的整数数组 $arr$。

**要求**：键改改数组中出现的每一个 $0$ 都复写一遍，并将其余的元素向右平移。

**说明**：

- 注意：不要在超过该数组长度的位置写上元素。请对输入的数组就地进行上述修改，不要从函数返回任何东西。
- $1 \le arr.length \le 10^4$。
- $0 \le arr[i] \le 9$。

**示例**：

- 示例 1：

```python
输入：arr = [1,0,2,3,0,4,5,0]
输出：[1,0,0,2,3,0,0,4]
解释：调用函数后，输入的数组将被修改为：[1,0,0,2,3,0,0,4]
```

- 示例 2：

```python
输入：arr = [1,2,3]
输出：[1,2,3]
解释：调用函数后，输入的数组将被修改为：[1,2,3]
```

## 解题思路

### 思路 1：两次遍历 + 快慢指针

因为数组中出现的 $0$ 需要复写为 $00$，占用空间从一个单位变成两个单位空间，那么右侧必定会有一部分元素丢失。我们可以先遍历一遍数组，找出复写后需要保留的有效数字部分与需要丢失部分的分界点。则从分界点开始，分界点右侧的元素都可以丢失。

我们再次逆序遍历数组，

1. 使用两个指针 $slow$、$fast$，$slow$ 表示当前有效字符位置，$fast$ 表示当前遍历字符位置。一开始 $slow$ 和 $fast$ 都指向数组开始位置。
2. 正序扫描数组：
   1. 如果遇到 $arr[slow] == 0$，则让 $fast$ 指针多走一步。
   2. 然后 $fast$、$slow$ 各自向右移动 $1$ 位，直到 $fast$ 指针移动到数组末尾。此时 $slow$ 左侧数字 $arr[0]... arr[slow - 1]$ 为需要保留的有效数字部分， $arr[slow]...arr[fast - 1]$ 为需要丢失部分。
3. 令 $slow$、$fast$ 分别左移 $1$ 位，此时 $slow$ 指向最后一个有效数字，$fast$ 指向丢失部分的最后一个数字。此时 $fast$ 可能等于 $size - 1$，也可能等于 $size$（比如输入 $[0, 0, 0]$）。
4. 逆序遍历数组：
   1. 将 $slow$ 位置元素移动到 $fast$ 位置。
   2. 如果遇到 $arr[slow] == 0$，则令 $fast$ 减 $1$，然后再复制 $1$ 个 $0$ 到 $fast$ 位置。
   3. 令 $slow$、$fast$ 分别左移 $1$ 位。

### 思路 1：代码

```python
class Solution:
    def duplicateZeros(self, arr: List[int]) -> None:
        """
        Do not return anything, modify arr in-place instead.
        """
        size = len(arr)
        slow, fast = 0, 0
        while fast < size:
            if arr[slow] == 0:
                fast += 1
            slow += 1
            fast += 1
        
        slow -= 1 # slow 指向最后一个有效数字
        fast -= 1 # fast 指向丢失部分的最后一个数字（可能在减 1 之后为 size，比如输入 [0, 0, 0]）

        while slow >= 0:
            if fast < size: # 防止 fast 越界
                arr[fast] = arr[slow] # 将 slow 位置元素移动到 fast 位置
            if arr[slow] == 0 and fast >= 0: # 遇见 0 则复制 0 到 fast - 1 位置
                fast -= 1
                arr[fast] = arr[slow]
            fast -= 1
            slow -= 1
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(n)$，其中 $n$ 为数组 $arr$ 中的元素个数。
- **空间复杂度**：$O(1)$。
# [1091. 二进制矩阵中的最短路径](https://leetcode.cn/problems/shortest-path-in-binary-matrix/)

- 标签：广度优先搜索、数组、矩阵
- 难度：中等

## 题目链接

- [1091. 二进制矩阵中的最短路径 - 力扣](https://leetcode.cn/problems/shortest-path-in-binary-matrix/)

## 题目大意

给定一个 `n * n` 的二进制矩阵 `grid`。 `grid` 中只含有 `0` 或者 `1`。`grid` 中的畅通路径是一条从左上角 `(0, 0)` 位置上到右下角 `(n - 1, n - 1)`位置上的路径。该路径同时满足以下要求：

- 路径途径的所有单元格的值都是 `0`。
- 路径中所有相邻的单元格应该在 `8` 个方向之一上连通（即相邻两单元格之间彼此不同且共享一条边或者一个角）。
- 畅通路径的长度是该路径途径的单元格总数。

要求：计算出矩阵中最短畅通路径的长度。如果不存在这样的路径，返回 `-1`。

## 解题思路

使用广度优先搜索查找最短路径。具体做法如下：

1. 使用队列 `queue` 存放当前节点位置，使用 set 集合 `visited` 存放遍历过的节点位置。使用 `count` 记录最短路径。将起始位置 `(0, 0)` 加入到 `queue` 中，并标记为访问过。
2. 如果队列不为空，则令 `count += 1`，并将队列中的节点位置依次取出。对于每一个节点位置：
   - 先判断是否为右下角节点，即 `(n - 1, n - 1)`。如果是则返回当前最短路径长度 `count`。
   - 如果不是，则继续遍历 `8` 个方向上、没有访问过、并且值为 `0` 的相邻单元格。
   - 将其加入到队列 `queue` 中，并标记为访问过。
3. 重复进行第 2 步骤，直到队列为空时，返回 `-1`。

## 代码

```python
class Solution:
    def shortestPathBinaryMatrix(self, grid: List[List[int]]) -> int:
        if grid[0][0] == 1:
            return -1
        size = len(grid)
        directions = {(1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1)}
        visited = set((0, 0))
        queue = [(0, 0)]
        count = 0
        while queue:
            count += 1
            for _ in range(len(queue)):
                row, col = queue.pop(0)

                if row == size - 1 and col == size - 1:
                    return count
                for direction in directions:
                    new_row = row + direction[0]
                    new_col = col + direction[1]
                    if 0 <= new_row < size and 0 <= new_col < size and grid[new_row][new_col] == 0 and (new_row, new_col) not in visited:
                        queue.append((new_row, new_col))
                        visited.add((new_row, new_col))
        return -1
```

# [1095. 山脉数组中查找目标值](https://leetcode.cn/problems/find-in-mountain-array/)

- 标签：数组、二分查找、交互
- 难度：困难

## 题目链接

- [1095. 山脉数组中查找目标值 - 力扣](https://leetcode.cn/problems/find-in-mountain-array/)

## 题目大意

**描述**：给定一个山脉数组 $mountainArr$。

**要求**：返回能够使得 `mountainArr.get(index)` 等于 $target$ 最小的下标 $index$ 值。如果不存在这样的下标 $index$，就请返回 $-1$。

**说明**：

- 山脉数组：满足以下属性的数组：

  - $len(arr) \ge 3$；
  - 存在 $i$（$0 < i < len(arr) - 1$），使得：
    - $arr[0] < arr[1] < ... arr[i-1] < arr[i]$;
    - $arr[i] > arr[i+1] > ... > arr[len(arr) - 1]$。
- 不能直接访问该山脉数组，必须通过 `MountainArray` 接口来获取数据：

  - `MountainArray.get(index)`：会返回数组中索引为 $k$ 的元素（下标从 $0$ 开始）。

  - `MountainArray.length()`：会返回该数组的长度。
- 对 `MountainArray.get` 发起超过 $100$ 次调用的提交将被视为错误答案。
- $3 \le mountain_arr.length() \le 10000$。
- $0 \le target \le 10^9$。
- $0 \le mountain_arr.get(index) \le 10^9$。

**示例**：

- 示例 1：

```python
输入：array = [1,2,3,4,5,3,1], target = 3
输出：2
解释：3 在数组中出现了两次，下标分别为 2 和 5，我们返回最小的下标 2。
```

- 示例 2：

```python
输入：array = [0,1,2,4,2,1], target = 3
输出：-1
解释：3 在数组中没有出现，返回 -1。
```

## 解题思路

### 思路 1：二分查找

因为题目要求不能对 `MountainArray.get` 发起超过 $100$ 次调用。所以遍历数组进行查找是不可行的。

根据山脉数组的性质，我们可以把山脉数组分为两部分：「前半部分的升序数组」和「后半部分的降序数组」。在有序数组中查找目标值可以使用二分查找来减少查找次数。

而山脉的峰顶元素索引也可以通过二分查找来做。所以这道题我们可以分为三步：

1. 通过二分查找找到山脉数组的峰顶元素索引。
2. 通过二分查找在前半部分的升序数组中查找目标元素。
3. 通过二分查找在后半部分的降序数组中查找目标元素。

最后，通过对查找结果的判断来输出最终答案。

### 思路 1：代码

```python
#class MountainArray:
#    def get(self, index: int) -> int:
#    def length(self) -> int:

class Solution:
    def binarySearchPeak(self, mountain_arr) -> int:
        left, right = 0, mountain_arr.length() - 1
        while left < right:
            mid = left + (right - left) // 2
            if mountain_arr.get(mid) < mountain_arr.get(mid + 1):
                left = mid + 1
            else:
                right = mid
        return left

    def binarySearchAscending(self, mountain_arr, left, right, target):
        while left < right:
            mid = left + (right - left) // 2
            if mountain_arr.get(mid) < target:
                left = mid + 1
            else:
                right = mid
        return left if mountain_arr.get(left) == target else -1

    def binarySearchDescending(self, mountain_arr, left, right, target):
        while left < right:
            mid = left + (right - left) // 2
            if mountain_arr.get(mid) > target:
                left = mid + 1
            else:
                right = mid
        return left if mountain_arr.get(left) == target else -1

    def findInMountainArray(self, target: int, mountain_arr: 'MountainArray') -> int:
        size = mountain_arr.length()
        peek_i = self.binarySearchPeak(mountain_arr)

        res_left = self.binarySearchAscending(mountain_arr, 0, peek_i, target)
        res_right = self.binarySearchDescending(mountain_arr, peek_i + 1, size - 1, target)
        
        return res_left if res_left != -1 else res_right
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(\log n)$。
- **空间复杂度**：$O(1)$。
# [1099. 小于 K 的两数之和](https://leetcode.cn/problems/two-sum-less-than-k/)

- 标签：数组、双指针、二分查找、排序
- 难度：简单

## 题目链接

- [1099. 小于 K 的两数之和 - 力扣](https://leetcode.cn/problems/two-sum-less-than-k/)

## 题目大意

**描述**：给定一个整数数组 $nums$ 和整数 $k$。 

**要求**：返回最大和 $sum$，满足存在 $i < j$ 使得 $nums[i] + nums[j] = sum$ 且 $sum < k$。如果没有满足此等式的 $i$, $j$ 存在，则返回 $-1$。

**说明**：

- $1 \le nums.length \le 100$。
- $1 \le nums[i] \le 1000$。
- $1 \le k \le 2000$。

**示例**：

- 示例 1：

```python
输入：nums = [34,23,1,24,75,33,54,8], k = 60
输出：58
解释：34 和 24 相加得到 58，58 小于 60，满足题意。
```

- 示例 2：

```python
输入：nums = [10,20,30], k = 15
输出：-1
解释：我们无法找到和小于 15 的两个元素。
```

## 解题思路

### 思路 1：对撞指针

常规暴力枚举时间复杂度为 $O(n^2)$。可以通过双指针降低时间复杂度。具体做法如下：

- 先对数组进行排序（时间复杂度为 $O(n \log n$），使用 $res$ 记录答案，初始赋值为最小值 `float('-inf')`。
- 使用两个指针 $left$、$right$。$left$ 指向第 $0$ 个元素位置，$right$ 指向数组的最后一个元素位置。 
- 计算 $nums[left] + nums[right]$，与 $k$ 进行比较。
  - 如果 $nums[left] + nums[right] \ge k$，则将 $right$ 左移，继续查找。
  - 如果 $nums[left] + nums[rigth] < k$，则将 $left$ 右移，并更新答案值。
- 当 $left == right$ 时，区间搜索完毕，判断 $res$ 是否等于 `float('-inf')`，如果等于，则返回 $-1$，否则返回 $res$。

### 思路 1：代码

```python
class Solution:
    def twoSumLessThanK(self, nums: List[int], k: int) -> int:

        nums.sort()
        res = float('-inf')
        left, right = 0, len(nums) - 1
        while left < right:
            total = nums[left] + nums[right]
            if total >= k:
                right -= 1
            else:
                res = max(res, total)
                left += 1

        return res if res != float('-inf') else -1
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(n^2)$，其中 $n$ 为数组中元素的个数。
- **空间复杂度**：$O(\log n)$，排序需要 $\log n$ 的栈空间。

# [1100. 长度为 K 的无重复字符子串](https://leetcode.cn/problems/find-k-length-substrings-with-no-repeated-characters/)

- 标签：哈希表、字符串、滑动窗口
- 难度：中等

## 题目链接

- [1100. 长度为 K 的无重复字符子串 - 力扣](https://leetcode.cn/problems/find-k-length-substrings-with-no-repeated-characters/)

## 题目大意

**描述**：给定一个字符串 `s`。

**要求**：找出所有长度为 `k` 且不含重复字符的子串，返回全部满足要求的子串的数目。

**说明**：

- $1 \le s.length \le 10^4$。
- $s$ 中的所有字符均为小写英文字母。
- $1 <= k <= 10^4$。

**示例**：

- 示例 1：

```python
输入：s = "havefunonleetcode", k = 5
输出：6
解释：
这里有 6 个满足题意的子串，分别是：'havef','avefu','vefun','efuno','etcod','tcode'。
```

- 示例 2：

```python
输入：s = "home", K = 5
输出：0
解释：
注意：k 可能会大于 s 的长度。在这种情况下，就无法找到任何长度为 k 的子串。
```

## 解题思路

### 思路 1：滑动窗口

固定长度滑动窗口的题目。维护一个长度为 `k` 的滑动窗口。用 `window_count` 来表示窗口内所有字符个数。可以用字典、数组来实现，也可以直接用 `collections.Counter()` 实现。然后不断向右滑动，然后进行比较。如果窗口内字符无重复，则答案数目 + 1。然后继续滑动。直到末尾时。整个解题步骤具体如下：

1. `window_count` 用来维护窗口中 `2` 对应子串的各个字符数量。
2. `left` 、`right` 都指向序列的第一个元素，即：`left = 0`，`right = 0`。
3. 向右移动 `right`，先将 `k` 个元素填入窗口中。
4. 当窗口元素个数为 `k` 时，即：`right - left + 1 >= k` 时，判断窗口内各个字符数量 `window_count` 是否等于 `k`。
   1. 如果等于，则答案 + 1。
   2. 如果不等于，则向右移动 `left`，从而缩小窗口长度，即 `left += 1`，使得窗口大小始终保持为 `k`。
5. 重复 3 ~ 4 步，直到 `right` 到达数组末尾。返回答案。

### 思路 1：代码

```python
import collections

class Solution:
    def numKLenSubstrNoRepeats(self, s: str, k: int) -> int:
        left, right = 0, 0
        window_count = collections.Counter()
        ans = 0

        while right < len(s):
            window_count[s[right]] += 1

            if right - left + 1 >= k:
                if len(window_count) == k:
                    ans += 1
                window_count[s[left]] -= 1
                if window_count[s[left]] == 0:
                    del window_count[s[left]]
                left += 1

            right += 1

        return ans
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(n)$，其中 $n$ 为字符串 $s$ 的长度。
- **空间复杂度**：$O(|\sum|)$，其中 $\sum$ 是字符集。

# [1103. 分糖果 II](https://leetcode.cn/problems/distribute-candies-to-people/)

- 标签：数学、模拟
- 难度：简单

## 题目链接

- [1103. 分糖果 II - 力扣](https://leetcode.cn/problems/distribute-candies-to-people/)

## 题目大意

**描述**：给定一个整数 $candies$，代表糖果的数量。再给定一个整数 $num\underline{\hspace{0.5em}}people$，代表小朋友的数量。

现在开始分糖果，给第 $1$ 个小朋友分 $1$ 颗糖果，第 $2$ 个小朋友分 $2$ 颗糖果，以此类推，直到最后一个小朋友分 $n$ 颗糖果。

然后回到第 $1$ 个小朋友，给第 $1$ 个小朋友分 $n + 1$ 颗糖果，第 $2$ 个小朋友分 $n + 2$ 颗糖果，一次类推，直到最后一个小朋友分 $n + n$ 颗糖果。

重复上述过程（每次都比上一次多给出 $1$ 颗糖果，当分完第 $n$ 个小朋友时回到第 $1$ 个小朋友），直到我们分完所有的糖果。

> 注意：如果我们手中剩下的糖果数不够（小于等于前一次发的糖果数），则将剩下的糖果全部发给当前的小朋友。

**要求**：返回一个长度为 $num\underline{\hspace{0.5em}}people$、元素之和为 $candies$ 的数组，以表示糖果的最终分发情况（即 $ans[i]$ 表示第 $i$ 个小朋友分到的糖果数）。

**说明**：

- $1 \le candies \le 10^9$。
- $1 \le num\underline{\hspace{0.5em}}people \le 1000$。

**示例**：

- 示例 1：

```python
输入：candies = 7, num_people = 4
输出：[1,2,3,1]
解释：
第一次，ans[0] += 1，数组变为 [1,0,0,0]。
第二次，ans[1] += 2，数组变为 [1,2,0,0]。
第三次，ans[2] += 3，数组变为 [1,2,3,0]。
第四次，ans[3] += 1（因为此时只剩下 1 颗糖果），最终数组变为 [1,2,3,1]。
```

- 示例 2：

```python
输入：candies = 10, num_people = 3
输出：[5,2,3]
解释：
第一次，ans[0] += 1，数组变为 [1,0,0]。
第二次，ans[1] += 2，数组变为 [1,2,0]。
第三次，ans[2] += 3，数组变为 [1,2,3]。
第四次，ans[0] += 4，最终数组变为 [5,2,3]。
```

## 解题思路

### 思路 1：暴力模拟

不断遍历数组，将对应糖果数分给当前小朋友，直到糖果数为 $0$ 时停止。

### 思路 1：代码

```python
class Solution:
    def distributeCandies(self, candies: int, num_people: int) -> List[int]:
        ans = [0 for _ in range(num_people)]
        idx = 0
        while candies:
            ans[idx % num_people] += min(idx + 1, candies)
            candies -= min(idx + 1, candies)
            idx += 1
        
        return ans
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(max(\sqrt{m}, n))$，其中 $m$ 为糖果数量，$n$ 为小朋友数量。
- **空间复杂度**：$O(1)$。

# [1108. IP 地址无效化](https://leetcode.cn/problems/defanging-an-ip-address/)

- 标签：字符串
- 难度：简单

## 题目链接

- [1108. IP 地址无效化 - 力扣](https://leetcode.cn/problems/defanging-an-ip-address/)

## 题目大意

**描述**：给定一个有效的 IPv4 的地址 `address`。。

**要求**：返回这个 IP 地址的无效化版本。

**说明**：

- **无效化 IP 地址**：其实就是用 `"[.]"` 代替了每个 `"."`。

**示例**：

- 示例 1：

```python
输入：address = "255.100.50.0"
输出："255[.]100[.]50[.]0"
```

## 解题思路

### 思路 1：字符串替换

依次将字符串 `address` 中的 `"."` 替换为 `"[.]"`。这里为了方便，直接调用了 `replace` 方法。

### 思路 1：字符串替换代码

```python
class Solution:
    def defangIPaddr(self, address: str) -> str:
        return address.replace('.', '[.]')
```
# [1109. 航班预订统计](https://leetcode.cn/problems/corporate-flight-bookings/)

- 标签：数组、前缀和
- 难度：中等

## 题目链接

- [1109. 航班预订统计 - 力扣](https://leetcode.cn/problems/corporate-flight-bookings/)

## 题目大意

**描述**：给定整数 `n`，代表 `n` 个航班。再给定一个包含三元组的数组 `bookings`，代表航班预订表。表中第 `i` 条预订记录 $bookings[i] = [first_i, last_i, seats_i]$ 意味着在从 $first_i$ 到 $last_i$ （包含 $first_i$ 和 $last_i$）的 每个航班上预订了 $seats_i$ 个座位。

**要求**：返回一个长度为 `n` 的数组 `answer`，里面元素是每个航班预定的座位总数。

**说明**：

- $1 \le n \le 2 * 10^4$。
- $1 \le bookings.length \le 2 * 10^4$。
- $bookings[i].length == 3$。
- $1 \le first_i \le last_i \le n$。
- $1 \le seats_i \le 10^4$

**示例**：

- 示例 1：

```python
给定 n = 5。初始 answer = [0, 0, 0, 0, 0]

航班编号        1   2   3   4   5
预订记录 1 ：   10  10
预订记录 2 ：       20  20
预订记录 3 ：       25  25  25  25
总座位数：      10  55  45  25  25

最终 answer = [10, 55, 45, 25, 25]
```

## 解题思路

### 思路 1：线段树

- 初始化一个长度为 `n`，值全为 `0` 的 `nums` 数组。
- 然后根据 `nums` 数组构建一棵线段树。每个线段树的节点类存储当前区间的左右边界和该区间的和。并且线段树使用延迟标记。
- 然后遍历三元组操作，进行区间累加运算。
- 最后从线段树中查询数组所有元素，返回该数组即可。

这样构建线段树的时间复杂度为 $O(\log n)$，单次区间更新的时间复杂度为 $O(\log n)$，单次区间查询的时间复杂度为 $O(\log n)$。总体时间复杂度为 $O(\log n)$。

### 思路 1 线段树代码：

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
    def corpFlightBookings(self, bookings: List[List[int]], n: int) -> List[int]:
        nums = [0 for _ in range(n)]
        self.STree = SegmentTree(nums, lambda x, y: x + y)
        for booking in bookings:
            self.STree.update_interval(booking[0] - 1, booking[1] - 1, booking[2])

        return self.STree.get_nums()
```

# [1110. 删点成林](https://leetcode.cn/problems/delete-nodes-and-return-forest/)

- 标签：树、深度优先搜索、数组、哈希表、二叉树
- 难度：中等

## 题目链接

- [1110. 删点成林 - 力扣](https://leetcode.cn/problems/delete-nodes-and-return-forest/)

## 题目大意

**描述**：给定二叉树的根节点 $root$，树上每个节点都有一个不同的值。

如果节点值在 $to\underline{\hspace{0.5em}}delete$ 中出现，我们就把该节点从树上删去，最后得到一个森林（一些不相交的树构成的集合）。

**要求**：返回森林中的每棵树。你可以按任意顺序组织答案。

**说明**：

- 树中的节点数最大为 $1000$。
- 每个节点都有一个介于 $1$ 到 $1000$ 之间的值，且各不相同。
- $to\underline{\hspace{0.5em}}delete.length \le 1000$。
- $to\underline{\hspace{0.5em}}delete$ 包含一些从 $1$ 到 $1000$、各不相同的值。

**示例**：

- 示例 1：

![](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2019/07/05/screen-shot-2019-07-01-at-53836-pm.png)

```python
输入：root = [1,2,3,4,5,6,7], to_delete = [3,5]
输出：[[1,2,null,4],[6],[7]]
```

- 示例 2：

```python
输入：root = [1,2,4,null,3], to_delete = [3]
输出：[[1,2,4]]
```

## 解题思路

### 思路 1：深度优先搜索

将待删除节点数组 $to\underline{\hspace{0.5em}}delete$ 转为集合 $deletes$，则每次能以  $O(1)$ 的时间复杂度判断节点值是否在待删除节点数组中。

如果当前节点值在待删除节点数组中，则删除当前节点后，我们还需要判断其左右子节点是否也在待删除节点数组中。

以此类推，还需要判断左右子节点的左右子节点。。。

因此，我们应该递归遍历处理完所有的左右子树，再判断当前节点的左右子节点是否在待删除节点数组中。如果在，则将其加入到答案数组中。

为此我们可以写一个深度优先搜索算法，具体步骤如下：

1. 如果当前根节点为空，则返回 `None`。
2. 递归遍历处理完当前根节点的左右子树，更新当前节点的左右子树（子节点被删除的情况下需要更新当前根节点的左右子树）。
3. 如果当前根节点值在待删除节点数组中：
   1. 如果当前根节点的左子树没有在被删除节点数组中，将左子树节点加入到答案数组中。
   2. 如果当前根节点的右子树没有在被删除节点数组中，将右子树节点加入到答案数组中。
   3. 返回 `None`，表示当前节点被删除。
4. 如果当前根节点值不在待删除节点数组中：
   1. 返回根节点，表示当前节点没有被删除。

### 思路 1：代码

```Python
class Solution:
    def delNodes(self, root: Optional[TreeNode], to_delete: List[int]) -> List[TreeNode]:
        forest = []
        deletes = set(to_delete)
        def dfs(root):
            if not root:
                return None
            root.left = dfs(root.left)
            root.right = dfs(root.right)

            if root.val in deletes:
                if root.left:
                    forest.append(root.left)
                if root.right:
                    forest.append(root.right)
                return None
            else:
                return root


        if dfs(root):
            forest.append(root)
        return forest
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(n)$，其中 $n$ 为二叉树中节点个数。
- **空间复杂度**：$O(n)$。

# [1122. 数组的相对排序](https://leetcode.cn/problems/relative-sort-array/)

- 标签：数组、哈希表、计数排序、排序
- 难度：简单

## 题目链接

- [1122. 数组的相对排序 - 力扣](https://leetcode.cn/problems/relative-sort-array/)

## 题目大意

**描述**：给定两个数组，$arr1$ 和 $arr2$，其中 $arr2$ 中的元素各不相同，$arr2$ 中的每个元素都出现在 $arr1$ 中。

**要求**：对 $arr1$ 中的元素进行排序，使 $arr1$ 中项的相对顺序和 $arr2$ 中的相对顺序相同。未在 $arr2$ 中出现过的元素需要按照升序放在 $arr1$ 的末尾。

**说明**：

- $1 \le arr1.length, arr2.length \le 1000$。
- $0 \le arr1[i], arr2[i] \le 1000$。

**示例**：

- 示例 1：

```python
输入：arr1 = [2,3,1,3,2,4,6,7,9,2,19], arr2 = [2,1,4,3,9,6]
输出：[2,2,2,1,4,3,3,9,6,7,19]
```

- 示例 2：

```python
输入：arr1 = [28,6,22,8,44,17], arr2 = [22,28,8,6]
输出：[22,28,8,6,17,44]
```

## 解题思路

### 思路 1：计数排序

因为元素值范围在 $[0, 1000]$，所以可以使用计数排序的思路来解题。

1. 使用数组 $count$ 统计 $arr1$ 各个元素个数。
2. 遍历 $arr2$ 数组，将对应元素$num2$ 按照个数 $count[num2]$ 添加到答案数组 $ans$ 中，同时在 $count$ 数组中减去对应个数。
3. 然后在处理 $count$ 中剩余元素，将 $count$ 中大于 $0$ 的元素下标依次添加到答案数组 $ans$ 中。
4. 最后返回答案数组 $ans$。

### 思路 1：代码

```python
class Solution:
    def relativeSortArray(self, arr1: List[int], arr2: List[int]) -> List[int]:
        # 计算待排序序列中最大值元素 arr_max 和最小值元素 arr_min
        arr1_min, arr1_max = min(arr1), max(arr1)
        # 定义计数数组 counts，大小为 最大值元素 - 最小值元素 + 1
        size = arr1_max - arr1_min + 1
        counts = [0 for _ in range(size)]

        # 统计值为 num 的元素出现的次数
        for num in arr1:
            counts[num - arr1_min] += 1

        res = []
        for num in arr2:
            while counts[num - arr1_min] > 0:
                res.append(num)
                counts[num - arr1_min] -= 1

        for i in range(size):
            while counts[i] > 0:
                num = i + arr1_min
                res.append(num)
                counts[i] -= 1
        
        return res
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(m + n + max(arr_1))$。其中 $m$ 是数组 $arr_1$ 的长度，$n$ 是数组 $arr_2$ 的长度，$max(arr_1)$ 是数组 $arr_1$ 的最大值。
- **空间复杂度**：$O(max(arr_1))$。



# [1136. 并行课程](https://leetcode.cn/problems/parallel-courses/)

- 标签：图、拓扑排序
- 难度：中等

## 题目链接

- [1136. 并行课程 - 力扣](https://leetcode.cn/problems/parallel-courses/)

## 题目大意

有 N 门课程，分别以 1 到 N 进行编号。现在给定一份课程关系表 `relations[i] = [X, Y]`，用以表示课程 `X` 和课程 `Y` 之间的先修关系：课程 `X` 必须在课程 `Y` 之前修完。假设在一个学期里，你可以学习任何数量的课程，但前提是你已经学习了将要学习的这些课程的所有先修课程。

要求：返回学完全部课程所需的最少学期数。如果没有办法做到学完全部这些课程的话，就返回 `-1`。

## 解题思路

拓扑排序。具体解法如下：

1. 使用列表 `edges` 存放课程关系图，并统计每门课程节点的入度，存入入度列表 `indegrees`。使用 `ans` 表示学期数。
2. 借助队列 `queue`，将所有入度为 `0` 的节点入队。
3. 将队列中所有节点依次取出，学期数 +1。对于取出的每个节点：
   1. 对应课程数 -1。
   2. 将该顶点以及该顶点为出发点的所有边的另一个节点入度 -1。如果入度 -1 后的节点入度不为 0，则将其加入队列 `queue`。
4. 重复 3~4 的步骤，直到队列中没有节点。
5. 最后判断剩余课程数是否为 0，如果为 0，则返回 `ans`，否则，返回 `-1`。

## 代码

```python
import collections

class Solution:
    def minimumSemesters(self, n: int, relations: List[List[int]]) -> int:
        indegrees = [0 for _ in range(n + 1)]
        edges = collections.defaultdict(list)
        for x, y in relations:
            edges[x].append(y)
            indegrees[y] += 1
        queue = collections.deque([])
        for i in range(1, n + 1):
            if not indegrees[i]:
                queue.append(i)
        ans = 0

        while queue:
            size = len(queue)
            for i in range(size):
                x = queue.popleft()
                n -= 1
                for y in edges[x]:
                    indegrees[y] -= 1
                    if not indegrees[y]:
                        queue.append(y)
            ans += 1

        return ans if n == 0 else -1
```

# [1137. 第 N 个泰波那契数](https://leetcode.cn/problems/n-th-tribonacci-number/)

- 标签：记忆化搜索、数学、动态规划
- 难度：简单

## 题目链接

- [1137. 第 N 个泰波那契数 - 力扣](https://leetcode.cn/problems/n-th-tribonacci-number/)

## 题目大意

**描述**：给定一个整数 $n$。

**要求**：返回第 $n$ 个泰波那契数。

**说明**：

- **泰波那契数**：$T_0 = 0, T_1 = 1, T_2 = 1$，且在 $n >= 0$ 的条件下，$T_{n + 3} = T_{n} + T_{n+1} + T_{n+2}$。
- $0 \le n \le 37$。
- 答案保证是一个 32 位整数，即 $answer \le 2^{31} - 1$。

**示例**：

- 示例 1：

```python
输入：n = 4
输出：4
解释：
T_3 = 0 + 1 + 1 = 2
T_4 = 1 + 1 + 2 = 4
```

- 示例 2：

```python
输入：n = 25
输出：1389537
```

## 解题思路

### 思路 1：记忆化搜索

1. 问题的状态定义为：第 $n$ 个泰波那契数。其状态转移方程为：$T_0 = 0, T_1 = 1, T_2 = 1$，且在 $n >= 0$ 的条件下，$T_{n + 3} = T_{n} + T_{n+1} + T_{n+2}$。
2. 定义一个长度为 $n + 1$ 数组 `memo` 用于保存一斤个计算过的泰波那契数。
3. 定义递归函数 `my_tribonacci(n, memo)`。
   1. 当 $n = 0$ 或者 $n = 1$，或者 $n = 2$ 时直接返回结果。
   2. 当 $n > 2$ 时，首先检查是否计算过 $T(n)$，即判断 $memo[n]$ 是否等于 $0$。
      1. 如果 $memo[n] \ne 0$，说明已经计算过 $T(n)$，直接返回 $memo[n]$。
      2. 如果 $memo[n] = 0$，说明没有计算过 $T(n)$，则递归调用 `my_tribonacci(n - 3, memo)`、`my_tribonacci(n - 2, memo)`、`my_tribonacci(n - 1, memo)`，并将计算结果存入 $memo[n]$ 中，并返回 $memo[n]$。

### 思路 1：代码

```python
class Solution:
    def tribonacci(self, n: int) -> int:
        # 使用数组保存已经求解过的 T(k) 的结果
        memo = [0 for _ in range(n + 1)]
        return self.my_tribonacci(n, memo)
    
    def my_tribonacci(self, n: int, memo: List[int]) -> int:
        if n == 0:
            return 0
        if n == 1 or n == 2:
            return 1
        
        if memo[n] != 0:
            return memo[n]
        memo[n] = self.my_tribonacci(n - 3, memo) + self.my_tribonacci(n - 2, memo) + self.my_tribonacci(n - 1, memo)
        return memo[n]
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(n)$。
- **空间复杂度**：$O(n)$。

### 思路 2：动态规划

###### 1. 划分阶段

我们可以按照整数顺序进行阶段划分，将其划分为整数 $0 \sim n$。

###### 2. 定义状态

定义状态 `dp[i]` 为：第 `i` 个泰波那契数。

###### 3. 状态转移方程

根据题目中所给的泰波那契数的定义：$T_0 = 0, T_1 = 1, T_2 = 1$，且在 $n >= 0$ 的条件下，$T_{n + 3} = T_{n} + T_{n+1} + T_{n+2}$。，则直接得出状态转移方程为 $dp[i] = dp[i - 3] + dp[i - 2] + dp[i - 1]$（当 $i > 2$ 时）。

###### 4. 初始条件

根据题目中所给的初始条件 $T_0 = 0, T_1 = 1, T_2 = 1$ 确定动态规划的初始条件，即 `dp[0] = 0, dp[1] = 1, dp[2] = 1`。

###### 5. 最终结果

根据状态定义，最终结果为 `dp[n]`，即第 `n` 个泰波那契数为 `dp[n]`。

### 思路 2：代码

```python
class Solution:
    def tribonacci(self, n: int) -> int:
        if n == 0:
            return 0
        if n == 1 or n == 2:
            return 1
        dp = [0 for _ in range(n + 1)]
        dp[1] = dp[2] = 1
        for i in range(3, n + 1):
            dp[i] = dp[i - 3] + dp[i - 2] + dp[i - 1]
        return dp[n]
```

### 思路 2：复杂度分析

- **时间复杂度**：$O(n)$。
- **空间复杂度**：$O(n)$。

# [1143. 最长公共子序列](https://leetcode.cn/problems/longest-common-subsequence/)

- 标签：字符串、动态规划
- 难度：中等

## 题目链接

- [1143. 最长公共子序列 - 力扣](https://leetcode.cn/problems/longest-common-subsequence/)

## 题目大意

**描述**：给定两个字符串 $text1$ 和 $text2$。

**要求**：返回两个字符串的最长公共子序列的长度。如果不存在公共子序列，则返回 $0$。

**说明**：

- **子序列**：原字符串在不改变字符的相对顺序的情况下删除某些字符（也可以不删除任何字符）后组成的新字符串。
- **公共子序列**：两个字符串所共同拥有的子序列。
- $1 \le text1.length, text2.length \le 1000$。
- $text1$ 和 $text2$ 仅由小写英文字符组成。

**示例**：

- 示例 1：

```python
输入：text1 = "abcde", text2 = "ace" 
输出：3  
解释：最长公共子序列是 "ace"，它的长度为 3。
```

- 示例 2：

```python
输入：text1 = "abc", text2 = "abc"
输出：3
解释：最长公共子序列是 "abc"，它的长度为 3。
```

## 解题思路

### 思路 1：动态规划

###### 1. 划分阶段

按照两个字符串的结尾位置进行阶段划分。

###### 2. 定义状态

定义状态 $dp[i][j]$ 表示为：「以 $text1$ 中前 $i$ 个元素组成的子字符串 $str1$ 」与「以 $text2$ 中前 $j$ 个元素组成的子字符串 $str2$」的最长公共子序列长度为 $dp[i][j]$。

###### 3. 状态转移方程

双重循环遍历字符串 $text1$ 和 $text2$，则状态转移方程为：

1. 如果 $text1[i - 1] = text2[j - 1]$，说明两个子字符串的最后一位是相同的，所以最长公共子序列长度加 $1$。即：$dp[i][j] = dp[i - 1][j - 1] + 1$。
2. 如果 $text1[i - 1] \ne text2[j - 1]$，说明两个子字符串的最后一位是不同的，则 $dp[i][j]$ 需要考虑以下两种情况，取两种情况中最大的那种：$dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])$。
	1. 「以 $text1$ 中前 $i - 1$ 个元素组成的子字符串 $str1$ 」与「以 $text2$ 中前 $j$ 个元素组成的子字符串 $str2$」的最长公共子序列长度，即 $dp[i - 1][j]$。
	2. 「以 $text1$ 中前 $i$ 个元素组成的子字符串 $str1$ 」与「以 $text2$ 中前 $j - 1$ 个元素组成的子字符串 $str2$」的最长公共子序列长度，即 $dp[i][j - 1]$。

###### 4. 初始条件

1. 当 $i = 0$ 时，$str1$ 表示的是空串，空串与 $str2$ 的最长公共子序列长度为 $0$，即 $dp[0][j] = 0$。
2. 当 $j = 0$ 时，$str2$ 表示的是空串，$str1$ 与 空串的最长公共子序列长度为 $0$，即 $dp[i][0] = 0$。

###### 5. 最终结果

根据状态定义，最后输出 $dp[sise1][size2]$（即 $text1$ 与 $text2$ 的最长公共子序列长度）即可，其中 $size1$、$size2$ 分别为 $text1$、$text2$ 的字符串长度。

### 思路 1：代码

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

### 思路 1：复杂度分析

- **时间复杂度**：$O(n \times m)$，其中 $n$、$m$ 分别是字符串 $text1$、$text2$ 的长度。两重循环遍历的时间复杂度是 $O(n \times m)$，所以总的时间复杂度为 $O(n \times m)$。
- **空间复杂度**：$O(n \times m)$。用到了二维数组保存状态，所以总体空间复杂度为 $O(n \times m)$。

# [1151. 最少交换次数来组合所有的 1](https://leetcode.cn/problems/minimum-swaps-to-group-all-1s-together/)

- 标签：数组、滑动窗口
- 难度：中等

## 题目链接

- [1151. 最少交换次数来组合所有的 1 - 力扣](https://leetcode.cn/problems/minimum-swaps-to-group-all-1s-together/)

## 题目大意

**描述**：给定一个二进制数组 $data$。

**要求**：通过交换位置，将数组中任何位置上的 $1$ 组合到一起，并返回所有可能中所需的最少交换次数。c 

**说明**：

- $1 \le data.length \le 10^5$。
- $data[i] == 0 \text{ or } 1$。

**示例**：

- 示例 1：

```python
输入: data = [1,0,1,0,1]
输出: 1
解释: 
有三种可能的方法可以把所有的 1 组合在一起：
[1,1,1,0,0]，交换 1 次；
[0,1,1,1,0]，交换 2 次；
[0,0,1,1,1]，交换 1 次。
所以最少的交换次数为 1。
```

- 示例 2：

```python
输入：data = [0,0,0,1,0]
输出：0
解释： 
由于数组中只有一个 1，所以不需要交换。
```

## 解题思路

### 思路 1：滑动窗口

将数组中任何位置上的 $1$ 组合到一起，并要求最少的交换次数。也就是说交换之后，某个连续子数组中全是 $1$，数组其他位置全是 $0$。为此，我们可以维护一个固定长度为 $1$ 的个数的滑动窗口，找到滑动窗口中 $0$ 最少的个数，这样最终交换出去的 $0$ 最少，交换次数也最少。

求最少交换次数，也就是求滑动窗口中最少的 $0$ 的个数。具体做法如下：

1. 统计 $1$ 的个数，并设置为窗口长度 $window\underline{\hspace{0.5em}}size$。使用 $window\underline{\hspace{0.5em}}count$ 维护窗口中 $0$ 的个数。使用 $ans$ 维护窗口中最少的 $0$ 的个数，也可以叫做最少交换次数。
2. 如果 $window\underline{\hspace{0.5em}}size$ 为 $0$，则说明不用交换，直接返回 $0$。
3. 使用两个指针 $left$、$right$。$left$、$right$ 都指向数组的第一个元素，即：`left = 0`，`right = 0`。
4. 如果 $data[right] == 0$，则更新窗口中 $0$ 的个数，即 `window_count += 1`。然后向右移动 $right$。
5. 当窗口元素个数为 $window\underline{\hspace{0.5em}}size$ 时，即：$right - left + 1 \ge window\underline{\hspace{0.5em}}size$ 时，更新窗口中最少的 $0$ 的个数。
6. 然后如果左侧 $data[left] == 0$，则更新窗口中 $0$ 的个数，即 `window_count -= 1`。然后向右移动 $left$，从而缩小窗口长度，即 `left += 1`，使得窗口大小始终保持为 $window\underline{\hspace{0.5em}}size$。
7. 重复 4 ~ 6 步，直到 $right$ 到达数组末尾。返回答案 $ans$。

### 思路 1：代码

```python
class Solution:
    def minSwaps(self, data: List[int]) -> int:
        window_size = 0
        for item in data:
            if item == 1:
                window_size += 1
        if window_size == 0:
            return 0

        left, right = 0, 0
        window_count = 0
        ans = float('inf')
        while right < len(data):
            if data[right] == 0:
                window_count += 1

            if right - left + 1 >= window_size:
                ans = min(ans, window_count)
                if data[left] == 0:
                    window_count -= 1
                left += 1
            right += 1
        return ans if ans != float('inf') else 0
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(n)$，其中 $n$ 为数组 $data$ 的长度。
- **空间复杂度**：$O(1)$。

# [1155. 掷骰子等于目标和的方法数](https://leetcode.cn/problems/number-of-dice-rolls-with-target-sum/)

- 标签：动态规划
- 难度：中等

## 题目链接

- [1155. 掷骰子等于目标和的方法数 - 力扣](https://leetcode.cn/problems/number-of-dice-rolls-with-target-sum/)

## 题目大意

**描述**：有 $n$ 个一样的骰子，每个骰子上都有 $k$ 个面，分别标号为 $1 \sim k$。现在给定三个整数 $n$、$k$ 和 $target$，滚动 $n$ 个骰子。

**要求**：计算出使所有骰子正面朝上的数字和等于 $target$ 的方案数量。

**说明**：

- $1 \le n, k \le 30$。
- $1 \le target \le 1000$。

**示例**：

- 示例 1：

```python
输入：n = 1, k = 6, target = 3
输出：1
解释：你扔一个有 6 个面的骰子。
得到 3 的和只有一种方法。
```

- 示例 2：

```python
输入：n = 2, k = 6, target = 7
输出：6
解释：你扔两个骰子，每个骰子有 6 个面。
得到 7 的和有 6 种方法 1+6 2+5 3+4 4+3 5+2 6+1。
```

## 解题思路

### 思路 1：动态规划

我们可以将这道题转换为「分组背包问题」中求方案总数的问题。将每个骰子看做是一组物品，骰子每一个面上的数值当做是每组物品中的一个物品。这样问题就转换为：用 $n$ 个骰子（$n$ 组物品）进行投掷，投掷出总和（总价值）为 $target$ 的方案数。

###### 1. 划分阶段

按照总价值 $target$ 进行阶段划分。

###### 2. 定义状态

定义状态 $dp[w]$ 表示为：用 $n$ 个骰子（$n$ 组物品）进行投掷，投掷出总和（总价值）为 $w$ 的方案数。

###### 3. 状态转移方程

用 $n$ 个骰子（$n$ 组物品）进行投掷，投掷出总和（总价值）为 $w$ 的方案数，等于用 $n$ 个骰子（$n$ 组物品）进行投掷，投掷出总和（总价值）为 $w - d$ 的方案数累积值，其中 $d$ 为当前骰子掷出的价值，即：$dp[w] = dp[w] + dp[w - d]$。

###### 4. 初始条件

- 用 $n$ 个骰子（$n$ 组物品）进行投掷，投掷出总和（总价值）为 $0$ 的方案数为 $1$。

###### 5. 最终结果

根据我们之前定义的状态， $dp[w]$ 表示为：用 $n$ 个骰子（$n$ 组物品）进行投掷，投掷出总和（总价值）为 $w$ 的方案数。则最终结果为 $dp[target]$。

### 思路 1：代码

```python
class Solution:
    def numRollsToTarget(self, n: int, k: int, target: int) -> int:
        dp = [0 for _ in range(target + 1)]
        dp[0] = 1
        MOD = 10 ** 9 + 7

        # 枚举前 i 组物品
        for i in range(1, n + 1):
            # 逆序枚举背包装载重量
            for w in range(target, -1, -1):
                dp[w] = 0
                # 枚举第 i - 1 组物品能取个数
                for d in range(1, k + 1):
                    if w >= d:
                        dp[w] = (dp[w] + dp[w - d]) % MOD
                        
        return dp[target] % MOD
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(n \times m \times target)$。
- **空间复杂度**：$O(target)$。

# [1161. 最大层内元素和](https://leetcode.cn/problems/maximum-level-sum-of-a-binary-tree/)

- 标签：树、深度优先搜索、广度优先搜索、二叉树
- 难度：中等

## 题目链接

- [1161. 最大层内元素和 - 力扣](https://leetcode.cn/problems/maximum-level-sum-of-a-binary-tree/)

## 题目大意

**描述**：给你一个二叉树的根节点 $root$。设根节点位于二叉树的第 $1$ 层，而根节点的子节点位于第 $2$ 层，依此类推。

**要求**：返回层内元素之和最大的那几层（可能只有一层）的层号，并返回其中层号最小的那个。

**说明**：

- 树中的节点数在 $[1, 10^4]$ 范围内。
- $-10^5 \le Node.val \le 10^5$。

**示例**：

- 示例 1：

![](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2019/08/17/capture.jpeg)

```python
输入：root = [1,7,0,7,-8,null,null]
输出：2
解释：
第 1 层各元素之和为 1，
第 2 层各元素之和为 7 + 0 = 7，
第 3 层各元素之和为 7 + -8 = -1，
所以我们返回第 2 层的层号，它的层内元素之和最大。
```

- 示例 2：

```python
输入：root = [989,null,10250,98693,-89388,null,null,null,-32127]
输出：2
```

## 解题思路

### 思路 1：二叉树的层序遍历

1. 利用广度优先搜索，在二叉树的层序遍历的基础上，统计每一层节点和，并存入数组 $levels$ 中。
2. 遍历 $levels$ 数组，从 $levels$ 数组中找到最大层和 $max\underline{\hspace{0.5em}}sum$。
3. 再次遍历 $levels$ 数组，找出等于最大层和 $max\underline{\hspace{0.5em}}sum$ 的那一层，并返回该层序号。

### 思路 1：代码

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def levelOrder(self, root: TreeNode) -> List[List[int]]:
        if not root:
            return []
        queue = [root]
        levels = []
        while queue:
            level = 0
            size = len(queue)
            for _ in range(size):
                curr = queue.pop(0)
                level += curr.val
                if curr.left:
                    queue.append(curr.left)
                if curr.right:
                    queue.append(curr.right)
            levels.append(level)
        return levels

    def maxLevelSum(self, root: Optional[TreeNode]) -> int:
        levels = self.levelOrder(root)
        max_sum = max(levels)
        for i in range(len(levels)):
            if levels[i] == max_sum:
                return i + 1
```

### 思路 1：复杂度分析

- **时间复杂度**：$O(n)$。其中 $n$ 是二叉树的节点数目。
- **空间复杂度**：$O(n)$。
