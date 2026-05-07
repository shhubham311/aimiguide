# DSA Problems Practice

## Curated Problem Sets by Topic

This file contains essential practice problems organized by data structure. Each problem includes the approach, complexity analysis, and Python solution.

## 1. Array Problems

### Two Sum
**Problem**: Given an array and target, find two numbers that add up to target.
```python
def two_sum(nums, target):
    seen = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    return []

# O(n) time, O(n) space
print(two_sum([2, 7, 11, 15], 9))  # [0, 1]
```

### Maximum Subarray (Kadane's Algorithm)
**Problem**: Find contiguous subarray with largest sum.
```python
def max_subarray(nums):
    max_sum = curr_sum = nums[0]
    for num in nums[1:]:
        curr_sum = max(num, curr_sum + num)
        max_sum = max(max_sum, curr_sum)
    return max_sum

# O(n) time, O(1) space
print(max_subarray([-2,1,-3,4,-1,2,1,-5,4]))  # 6 → [4,-1,2,1]
```

### Product of Array Except Self
```python
def product_except_self(nums):
    n = len(nums)
    result = [1] * n
    left = 1
    for i in range(n):
        result[i] = left
        left *= nums[i]
    right = 1
    for i in range(n-1, -1, -1):
        result[i] *= right
        right *= nums[i]
    return result

print(product_except_self([1, 2, 3, 4]))  # [24, 12, 8, 6]
```

## 2. Linked List Problems

### Reverse Linked List
```python
def reverse_list(head):
    prev = None
    curr = head
    while curr:
        nxt = curr.next
        curr.next = prev
        prev = curr
        curr = nxt
    return prev
```

### Merge Two Sorted Lists
```python
def merge_two_lists(l1, l2):
    dummy = ListNode(0)
    curr = dummy
    while l1 and l2:
        if l1.val <= l2.val:
            curr.next = l1; l1 = l1.next
        else:
            curr.next = l2; l2 = l2.next
        curr = curr.next
    curr.next = l1 or l2
    return dummy.next
```

### Detect Cycle (Floyd's Algorithm)
```python
def has_cycle(head):
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            return True
    return False
```

## 3. Binary Search

### Binary Search Template
```python
def binary_search(nums, target):
    lo, hi = 0, len(nums) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if nums[mid] == target:
            return mid
        elif nums[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1

print(binary_search([1,3,5,7,9,11], 7))  # 3
```

### Search in Rotated Sorted Array
```python
def search_rotated(nums, target):
    lo, hi = 0, len(nums) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if nums[mid] == target:
            return mid
        if nums[lo] <= nums[mid]:
            if nums[lo] <= target < nums[mid]:
                hi = mid - 1
            else:
                lo = mid + 1
        else:
            if nums[mid] < target <= nums[hi]:
                lo = mid + 1
            else:
                hi = mid - 1
    return -1
```

## 4. Greedy Algorithms

### Jump Game
```python
def can_jump(nums):
    max_reach = 0
    for i in range(len(nums)):
        if i > max_reach:
            return False
        max_reach = max(max_reach, i + nums[i])
    return True

print(can_jump([2,3,1,1,4]))  # True
print(can_jump([3,2,1,0,4]))  # False
```

### Activity Selection (Interval Scheduling)
```python
def activity_selection(intervals):
    intervals.sort(key=lambda x: x[1])
    count = 0
    end = float('-inf')
    for start, finish in intervals:
        if start >= end:
            count += 1
            end = finish
    return count

print(activity_selection([(1,2),(3,4),(0,6),(5,7),(8,9)]))  # 4
```

## 5. Recursion & Backtracking

### Generate Parentheses
```python
def generate_parens(n):
    result = []
    def backtrack(s, open_count, close_count):
        if len(s) == 2 * n:
            result.append(s)
            return
        if open_count < n:
            backtrack(s + '(', open_count + 1, close_count)
        if close_count < open_count:
            backtrack(s + ')', open_count, close_count + 1)
    backtrack('', 0, 0)
    return result

print(generate_parens(3))
# ['((()))', '(()())', '(())()', '()(())', '()()()']
```

### Subsets
```python
def subsets(nums):
    result = [[]]
    for num in nums:
        result += [curr + [num] for curr in result]
    return result

print(subsets([1, 2, 3]))
# [[], [1], [2], [1,2], [3], [1,3], [2,3], [1,2,3]]
```

## 6. Heaps

### Top K Elements
```python
import heapq

def top_k(nums, k):
    return heapq.nlargest(k, nums)

def top_k_freq(nums, k):
    from collections import Counter
    counts = Counter(nums)
    return heapq.nlargest(k, counts.keys(), key=counts.get)

print(top_k_freq([1,1,1,2,2,3], 2))  # [1, 2]
```

### Median from Data Stream
```python
import heapq

class MedianFinder:
    def __init__(self):
        self.lo = []  # max heap (negate for min heap)
        self.hi = []  # min heap
    
    def add_num(self, num):
        heapq.heappush(self.lo, -num)
        heapq.heappush(self.hi, -heapq.heappop(self.lo))
        if len(self.lo) < len(self.hi):
            heapq.heappush(self.lo, -heapq.heappop(self.hi))
    
    def find_median(self):
        if len(self.lo) > len(self.hi):
            return -self.lo[0]
        return (-self.lo[0] + self.hi[0]) / 2
```

## 7. Dynamic Programming

### Climbing Stairs
```python
def climb_stairs(n):
    if n <= 2:
        return n
    a, b = 1, 2
    for _ in range(3, n + 1):
        a, b = b, a + b
    return b

print(climb_stairs(5))  # 8
```

### Longest Increasing Subsequence
```python
def length_of_lis(nums):
    tails = []
    for num in nums:
        lo, hi = 0, len(tails)
        while lo < hi:
            mid = (lo + hi) // 2
            if tails[mid] < num:
                lo = mid + 1
            else:
                hi = mid
        if lo == len(tails):
            tails.append(num)
        else:
            tails[lo] = num
    return len(tails)

print(length_of_lis([10,9,2,5,3,7,101,18]))  # 4
```

### Coin Change
```python
def coin_change(coins, amount):
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0
    for coin in coins:
        for i in range(coin, amount + 1):
            dp[i] = min(dp[i], dp[i - coin] + 1)
    return dp[amount] if dp[amount] != float('inf') else -1

print(coin_change([1, 2, 5], 11))  # 3 (5+5+1)
```

### 0/1 Knapsack
```python
def knapsack(weights, values, capacity):
    n = len(weights)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for w in range(capacity + 1):
            dp[i][w] = dp[i-1][w]
            if weights[i-1] <= w:
                dp[i][w] = max(dp[i][w], dp[i-1][w-weights[i-1]] + values[i-1])
    return dp[n][capacity]

print(knapsack([2,3,4,5], [3,4,5,6], 5))  # 7
```

## 8. Graph Problems

### Number of Islands
```python
def num_islands(grid):
    if not grid: return 0
    count = 0
    rows, cols = len(grid), len(grid[0])
    
    def dfs(r, c):
        if r < 0 or r >= rows or c < 0 or c >= cols or grid[r][c] != '1':
            return
        grid[r][c] = '0'
        dfs(r+1, c); dfs(r-1, c); dfs(r, c+1); dfs(r, c-1)
    
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == '1':
                count += 1
                dfs(r, c)
    return count

print(num_islands([["1","1","0"],["0","1","0"],["0","0","1"]]))  # 3
```

### Course Schedule (Topological Sort)
```python
from collections import deque

def can_finish(num_courses, prerequisites):
    graph = [[] for _ in range(num_courses)]
    in_degree = [0] * num_courses
    for course, prereq in prerequisites:
        graph[prereq].append(course)
        in_degree[course] += 1
    
    queue = deque([i for i in range(num_courses) if in_degree[i] == 0])
    completed = 0
    while queue:
        node = queue.popleft()
        completed += 1
        for neighbor in graph[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
    return completed == num_courses

print(can_finish(2, [[1,0]]))  # True
print(can_finish(2, [[1,0],[0,1]]))  # False (cycle!)
```

## Complexity Summary

| Problem | Time | Space | Pattern |
|---------|------|-------|---------|
| Two Sum | O(n) | O(n) | Hash Map |
| Max Subarray | O(n) | O(1) | Kadane's |
| Binary Search | O(log n) | O(1) | Divide & Conquer |
| Merge Lists | O(n+m) | O(1) | Two Pointers |
| Generate Parens | O(4^n/√n) | O(n) | Backtracking |
| Top K | O(n log k) | O(k) | Heap |
| LIS | O(n log n) | O(n) | DP + Binary Search |
| Coin Change | O(n×amount) | O(amount) | DP |
| Num Islands | O(mn) | O(mn) | DFS |
| Course Schedule | O(V+E) | O(V+E) | Topological Sort |

## 🧠 Practice Strategy

1. Start with Array, Linked List, Binary Search (foundational)
2. Move to Recursion, Backtracking (pattern building)
3. Then Trees, Graphs, Heaps (advanced structures)
4. Finally DP (hardest, requires most practice)
5. For each problem: understand WHY, not just HOW
