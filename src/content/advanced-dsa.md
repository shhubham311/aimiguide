# Advanced Data Structures: Trie, Fenwick Tree & Segment Tree

## Why These Matter

These advanced structures appear in coding interviews and real systems:

- **Trie**: Autocomplete, spell check, IP routing, prefix matching in search engines
- **Fenwick Tree (BIT)**: Frequency counting, prefix sums in O(log n)
- **Segment Tree**: Range queries (min, max, sum) with point updates in O(log n)

## 1. Trie (Prefix Tree)

A tree where each node represents a character. Words share common prefixes.

```
Insert: "cat", "car", "card", "care", "bat"

        root
       /    \
      c      b
      |      |
      a      a
     /|\     |
    r t t    t
   /|  |    |
  d e  |    (end)
  |    (end)
 (end) (end)
  |
  (end)
```

### Implementation

```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False
        self.count = 0

class Trie:
    def __init__(self):
        self.root = TrieNode()
    
    def insert(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end = True
        node.count += 1
    
    def search(self, word):
        node = self._traverse(word)
        return node is not None and node.is_end
    
    def starts_with(self, prefix):
        return self._traverse(prefix) is not None
    
    def _traverse(self, prefix):
        node = self.root
        for char in prefix:
            if char not in node.children:
                return None
            node = node.children[char]
        return node
    
    def autocomplete(self, prefix):
        node = self._traverse(prefix)
        if not node:
            return []
        results = []
        self._dfs(node, prefix, results)
        return results
    
    def _dfs(self, node, prefix, results):
        if node.is_end:
            results.append((prefix, node.count))
        for char, child in sorted(node.children.items()):
            self._dfs(child, prefix + char, results)

# Usage
trie = Trie()
words = ["cat", "car", "card", "care", "careful", "bat", "bar"]
for w in words:
    trie.insert(w)

print(trie.search("cat"))           # True
print(trie.search("cab"))           # False
print(trie.starts_with("car"))     # True
print(trie.autocomplete("car"))
# [('car', 1), ('card', 1), ('care', 1), ('careful', 1)]
```

### ML Application: Autocomplete for Search

```python
class SearchAutocomplete:
    def __init__(self):
        self.trie = Trie()
    
    def add_search(self, query):
        self.trie.insert(query.lower())
    
    def suggest(self, prefix, max_results=5):
        suggestions = self.trie.autocomplete(prefix.lower())
        return suggestions[:max_results]

engine = SearchAutocomplete()
queries = ["linear regression", "logistic regression", "linear algebra",
           "logistic function", "linear programming", "line chart"]
for q in queries:
    engine.add_search(q)

print(engine.suggest("lin"))
# [('linear algebra', 1), ('linear programming', 1), ('linear regression', 1), ('line chart', 1)]
print(engine.suggest("log"))
# [('logistic function', 1), ('logistic regression', 1)]
```

### Complexity

```
Operation       Time        Space
─────────────────────────────────────
Insert          O(m)        O(m)
Search          O(m)        O(1)
Starts With     O(m)        O(1)
Autocomplete    O(m + k)    O(k)
```

Where m = length of word/prefix, k = number of results.

## 2. Fenwick Tree (Binary Indexed Tree)

Efficiently computes **prefix sums** and supports **point updates** in O(log n).

```
Original array: [3, 1, 4, 1, 5, 9, 2, 6]

BIT structure:
  Index:  1  2  3  4  5  6  7  8
  Value:  3  4  4  9  5  14  2  31

Prefix sum queries:
  sum(1..4) = 9  (3+1+4+1)
  sum(1..6) = 14 (3+1+4+1+5+9)
  
Point update:
  update index 3 to value 10:
  Recalculate: O(log n) operations
```

### Implementation

```python
class FenwickTree:
    def __init__(self, size):
        self.n = size
        self.bit = [0] * (size + 1)
    
    def build(self, arr):
        for i, val in enumerate(arr):
            self.update(i + 1, val)
    
    def update(self, index, delta):
        """Add delta to element at index (1-based)"""
        while index <= self.n:
            self.bit[index] += delta
            index += index & (-index)  # Go to next responsible node
    
    def query(self, index):
        """Get prefix sum from 1 to index"""
        result = 0
        while index > 0:
            result += self.bit[index]
            index -= index & (-index)  # Go to parent
        return result
    
    def range_query(self, left, right):
        """Get sum from left to right (1-based)"""
        return self.query(right) - self.query(left - 1)

# Usage
arr = [3, 1, 4, 1, 5, 9, 2, 6]
ft = FenwickTree(len(arr))
ft.build(arr)

print(f"sum(1..4) = {ft.range_query(1, 4)}")   # 9
print(f"sum(3..6) = {ft.range_query(3, 6)}")   # 19
print(f"sum(1..8) = {ft.range_query(1, 8)}")   # 31

ft.update(3, 6)  # arr[2] was 4, now 4+6=10
print(f"After update: sum(1..4) = {ft.range_query(1, 4)}")  # 15
```

### ML Application: Running Mean/Std

```python
class RunningStats:
    """Track running mean and variance using Fenwick tree."""
    def __init__(self, max_size):
        self.ft_sum = FenwickTree(max_size)
        self.ft_sq = FenwickTree(max_size)
        self.n = 0
        self.max_size = max_size
    
    def add(self, value):
        self.n += 1
        idx = self.n
        self.ft_sum.update(idx, value)
        self.ft_sq.update(idx, value * value)
    
    def mean(self):
        if self.n == 0: return 0
        return self.ft_sum.range_query(1, self.n) / self.n
    
    def variance(self):
        if self.n == 0: return 0
        mean = self.mean()
        mean_sq = self.ft_sq.range_query(1, self.n) / self.n
        return mean_sq - mean * mean

rs = RunningStats(1000)
import random
random.seed(42)
for _ in range(100):
    rs.add(random.gauss(50, 10))

print(f"Running mean: {rs.mean():.2f}")
print(f"Running variance: {rs.variance():.2f}")
```

## 3. Segment Tree

Supports **range queries** (min, max, sum) and **point updates** in O(log n). More flexible than Fenwick Tree.

```
Array: [3, 1, 4, 1, 5, 9, 2, 6]

Segment Tree (Sum):
               [31]
            /        \
         [9]         [22]
        /   \       /    \
      [4]   [5]  [14]    [8]
      / \   / \  / \    / \
    [3][1][4][1][5][9][2][6]

Range query: sum(2..6) = ?
  Root [31] → go left [9] → partial, right [22] → partial
  Total: elements 2..6 = 1+4+1+5+9 = 20 (visits O(log n) nodes)
```

### Implementation

```python
class SegmentTree:
    def __init__(self, data, operation='sum'):
        self.n = len(data)
        self.operation = operation
        self.size = 4 * self.n
        self.tree = [0] * self.size
        self._build(data, 1, 0, self.n - 1)
    
    def _build(self, data, node, start, end):
        if start == end:
            self.tree[node] = data[start]
        else:
            mid = (start + end) // 2
            self._build(data, 2*node, start, mid)
            self._build(data, 2*node+1, mid+1, end)
            self.tree[node] = self._combine(self.tree[2*node], self.tree[2*node+1])
    
    def _combine(self, left, right):
        if self.operation == 'sum':
            return left + right
        elif self.operation == 'min':
            return min(left, right)
        elif self.operation == 'max':
            return max(left, right)
    
    def query(self, left, right):
        return self._query(1, 0, self.n - 1, left, right)
    
    def _query(self, node, start, end, left, right):
        if right < start or left > end:
            return 0 if self.operation == 'sum' else float('inf') if self.operation == 'min' else float('-inf')
        if left <= start and end <= right:
            return self.tree[node]
        mid = (start + end) // 2
        return self._combine(
            self._query(2*node, start, mid, left, right),
            self._query(2*node+1, mid+1, end, left, right)
        )
    
    def update(self, index, value):
        self._update(1, 0, self.n - 1, index, value)
    
    def _update(self, node, start, end, index, value):
        if start == end:
            self.tree[node] = value
        else:
            mid = (start + end) // 2
            if index <= mid:
                self._update(2*node, start, mid, index, value)
            else:
                self._update(2*node+1, mid+1, end, index, value)
            self.tree[node] = self._combine(self.tree[2*node], self.tree[2*node+1])

# Usage: Range Sum
arr = [3, 1, 4, 1, 5, 9, 2, 6]
st = SegmentTree(arr, 'sum')
print(f"sum(2,6) = {st.query(2, 6)}")   # 20
print(f"sum(0,3) = {st.query(0, 3)}")   # 9

st.update(3, 10)
print(f"After update: sum(2,6) = {st.query(2, 6)}")  # 29

# Usage: Range Minimum
st_min = SegmentTree(arr, 'min')
print(f"min(1,5) = {st_min.query(1, 5)}")  # 1
print(f"min(0,7) = {st_min.query(0, 7)}")  # 1

# Usage: Range Maximum
st_max = SegmentTree(arr, 'max')
print(f"max(0,7) = {st_max.query(0, 7)}")  # 9
```

### ML Application: Sliding Window Statistics

```python
class SlidingWindowStats:
    """O(log n) range queries over a sliding window of data."""
    def __init__(self, max_size):
        self.data = [0] * max_size
        self.idx = 0
        self.filled = 0
        self.st_sum = SegmentTree([0] * max_size, 'sum')
        self.st_min = SegmentTree([0] * max_size, 'min')
        self.st_max = SegmentTree([0] * max_size, 'max')
        self.max_size = max_size
    
    def add(self, value):
        pos = self.idx % self.max_size
        self.data[pos] = value
        self.st_sum.update(pos, value)
        self.st_min.update(pos, value)
        self.st_max.update(pos, value)
        self.idx += 1
        self.filled = min(self.filled + 1, self.max_size)
    
    def window_sum(self):
        n = min(self.filled, self.max_size)
        return self.st_sum.query(0, n - 1)
    
    def window_min(self):
        n = min(self.filled, self.max_size)
        return self.st_min.query(0, n - 1)
    
    def window_max(self):
        n = min(self.filled, self.max_size)
        return self.st_max.query(0, n - 1)

sws = SlidingWindowStats(10)
import random
random.seed(42)
for _ in range(15):
    val = random.randint(1, 100)
    sws.add(val)

print(f"Last 10 values:")
print(f"  Sum: {sws.window_sum()}")
print(f"  Min: {sws.window_min()}")
print(f"  Max: {sws.window_max()}")
```

## Comparison Table

```
┌──────────────┬────────────────────┬────────────────────┬──────────────────┐
│              │ TRIE               │ FENWICK TREE       │ SEGMENT TREE     │
├──────────────┼────────────────────┼────────────────────┼──────────────────┤
│ Use Case     │ Prefix matching    │ Prefix sums        │ Range queries    │
│              │ Autocomplete       │ Frequency counts   │ Min/Max/Sum      │
├──────────────┼────────────────────┼────────────────────┼──────────────────┤
│ Build        │ O(m) per word      │ O(n log n)         │ O(n)             │
│ Query        │ O(m)               │ O(log n)           │ O(log n)         │
│ Update       │ O(m)               │ O(log n)           │ O(log n)         │
│ Space        │ O(ALPHABET × m)    │ O(n)               │ O(4n)            │
├──────────────┼────────────────────┼────────────────────┼──────────────────┤
│ Flexible     │ Prefix ops only    │ Sum only           │ Any operation    │
│              │                    │                    │ (sum,min,max)    │
└──────────────┴────────────────────┴────────────────────┴──────────────────┘
```

## 🧠 Exercises

### Exercise 1: Implement a Trie that supports wildcard search ('.' matches any character).

<details>
<summary>Solution</summary>

```python
class WordDictionary:
    def __init__(self):
        self.root = {}
    
    def add_word(self, word):
        node = self.root
        for c in word:
            if c not in node:
                node[c] = {}
            node = node[c]
        node['#'] = True
    
    def search(self, word):
        def dfs(node, i):
            if i == len(word):
                return '#' in node
            if word[i] == '.':
                for child in node.values():
                    if isinstance(child, dict) and dfs(child, i + 1):
                        return True
                return False
            if word[i] not in node:
                return False
            return dfs(node[word[i]], i + 1)
        return dfs(self.root, 0)
```

</details>

### Exercise 2: Use Fenwick Tree to count inversions in an array.

<details>
<summary>Solution</summary>

```python
def count_inversions(arr):
    # Coordinate compression
    sorted_unique = sorted(set(arr))
    rank = {v: i+1 for i, v in enumerate(sorted_unique)}
    
    ft = FenwickTree(len(rank))
    inversions = 0
    for num in reversed(arr):
        r = rank[num]
        inversions += ft.query(r - 1)
        ft.update(r, 1)
    return inversions

print(count_inversions([3, 1, 4, 1, 5, 9, 2, 6]))  # 10
```

</details>

### Exercise 3: Use Segment Tree to find the minimum in any range and the index of that minimum.

<details>
<summary>Solution</summary>

```python
class MinSegmentTree:
    def __init__(self, data):
        self.n = len(data)
        self.size = 4 * self.n
        self.tree = [(float('inf'), -1)] * self.size
        self._build(data, 1, 0, self.n - 1)
    
    def _build(self, data, node, start, end):
        if start == end:
            self.tree[node] = (data[start], start)
        else:
            mid = (start + end) // 2
            self._build(data, 2*node, start, mid)
            self._build(data, 2*node+1, mid+1, end)
            self.tree[node] = min(self.tree[2*node], self.tree[2*node+1])
    
    def query(self, left, right):
        return self._query(1, 0, self.n-1, left, right)
    
    def _query(self, node, start, end, left, right):
        if right < start or left > end:
            return (float('inf'), -1)
        if left <= start and end <= right:
            return self.tree[node]
        mid = (start + end) // 2
        return min(self._query(2*node, start, mid, left, right),
                   self._query(2*node+1, mid+1, end, left, right))

st = MinSegmentTree([5, 3, 8, 1, 9, 2, 7])
print(st.query(1, 5))  # (1, 3) → min is 1 at index 3
```

</details>
