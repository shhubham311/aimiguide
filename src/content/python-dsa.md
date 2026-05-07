# Python Data Structures

## Why Data Structures Matter in AI/ML

Data structures are the **backbone of every algorithm** you'll implement in ML. Understanding them deeply means:

- Writing efficient data pipelines (processing millions of rows)
- Passing coding interviews at top AI companies
- Understanding how ML libraries store and manipulate data internally
- Building scalable systems

## Arrays (Python Lists)

The most fundamental structure. In Python, lists are **dynamic arrays** under the hood.

```
Operation          Time Complexity    Notes
─────────────────────────────────────────────────────
Access by index    O(1)               Direct memory offset
Append (end)       O(1) amortized     Fast!
Insert (middle)    O(n)               Must shift elements
Delete (middle)    O(n)               Must shift elements
Search (unsorted)  O(n)               Linear scan
Search (sorted)    O(log n)           Binary search
```

```python
# Dynamic array behavior
arr = [1, 2, 3]
arr.append(4)           # O(1) amortized
arr.insert(1, 99)       # O(n) — shifts everything right
arr.pop()               # O(1) — removes from end
arr.pop(0)              # O(n) — shifts everything left
arr[2]                  # O(1) — random access

# ML Application: Feature vector
feature_vector = [0.5, 1.2, 0.8, 3.1, 2.7]  # 5 features
label = 1  # classification label
```

## Linked Lists

Each node contains data + pointer to next. Unlike arrays, no contiguous memory needed.

```
Array:    [1] [2] [3] [4] [5]     ← Contiguous memory
          ↑   ↑   ↑   ↑   ↑       ← O(1) access

Linked:   [1|→]→[2|→]→[3|→]→[4|→]→None
          ↑                      ← Only O(1) at head
                                  ← O(n) to reach end
```

```python
class Node:
    def __init__(self, val, next=None):
        self.val = val
        self.next = next

class LinkedList:
    def __init__(self):
        self.head = None
        self.size = 0
    
    def append(self, val):
        new_node = Node(val)
        if not self.head:
            self.head = new_node
        else:
            curr = self.head
            while curr.next:
                curr = curr.next
            curr.next = new_node
        self.size += 1
    
    def to_list(self):
        result = []
        curr = self.head
        while curr:
            result.append(curr.val)
            curr = curr.next
        return result
    
    def reverse(self):
        prev, curr = None, self.head
        while curr:
            nxt = curr.next
            curr.next = prev
            prev = curr
            curr = nxt
        self.head = prev

ll = LinkedList()
for x in [10, 20, 30, 40]:
    ll.append(x)
ll.reverse()
print(ll.to_list())  # [40, 30, 20, 10]
```

## Stacks (LIFO — Last In, First Out)

```
Push:  [1] → [1,2] → [1,2,3]
Pop:   [1,2,3] → [1,2] → [1]

ML Use: DFS tree traversal, expression evaluation, undo operations
```

```python
class Stack:
    def __init__(self):
        self.items = []
    
    def push(self, val): self.items.append(val)
    def pop(self): return self.items.pop()
    def peek(self): return self.items[-1]
    def is_empty(self): return len(self.items) == 0
    def __len__(self): return len(self.items)

# Application: Check balanced parentheses
def is_balanced(expr):
    stack = Stack()
    pairs = {')': '(', '}': '{', ']': '['}
    for char in expr:
        if char in '({[':
            stack.push(char)
        elif char in ')}]':
            if stack.is_empty() or stack.pop() != pairs[char]:
                return False
    return stack.is_empty()

print(is_balanced("{[()]}"))    # True
print(is_balanced("{[(])}"))    # False
```

## Queues (FIFO — First In, First Out)

```
Enqueue:  [] → [1] → [1,2] → [1,2,3]
Dequeue:  [1,2,3] → [2,3] → [3]

ML Use: BFS graph traversal, level-order traversal, batch processing
```

```python
from collections import deque

class Queue:
    def __init__(self):
        self.items = deque()
    
    def enqueue(self, val): self.items.append(val)
    def dequeue(self): return self.items.popleft()
    def front(self): return self.items[0]
    def is_empty(self): return len(self.items) == 0

# Application: BFS on a graph
from collections import deque

def bfs(graph, start):
    visited = set()
    queue = deque([start])
    order = []
    
    while queue:
        node = queue.popleft()
        if node in visited:
            continue
        visited.add(node)
        order.append(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                queue.append(neighbor)
    return order

graph = {
    'A': ['B', 'C'], 'B': ['A', 'D', 'E'],
    'C': ['A', 'F'], 'D': ['B'], 'E': ['B', 'F'], 'F': ['C', 'E']
}
print(bfs(graph, 'A'))  # ['A', 'B', 'C', 'D', 'E', 'F']
```

## Hash Maps / Dictionaries

```
Average Case    Worst Case
────────────────────────────
O(1) lookup     O(n) collision
O(1) insert     O(n) collision
O(1) delete     O(n) collision
```

```python
# Hash map from scratch using open addressing
class HashMap:
    def __init__(self, size=100):
        self.size = size
        self.buckets = [None] * size
    
    def _hash(self, key):
        return hash(key) % self.size
    
    def put(self, key, value):
        idx = self._hash(key)
        while self.buckets[idx] is not None:
            if self.buckets[idx][0] == key:
                self.buckets[idx] = (key, value)
                return
            idx = (idx + 1) % self.size
        self.buckets[idx] = (key, value)
    
    def get(self, key):
        idx = self._hash(key)
        while self.buckets[idx] is not None:
            if self.buckets[idx][0] == key:
                return self.buckets[idx][1]
            idx = (idx + 1) % self.size
        return None

hm = HashMap()
hm.put("model", "resnet")
hm.put("accuracy", 0.95)
print(hm.get("model"))  # resnet
```

## Binary Trees

```
        10
       /  \
      5    15
     / \   / \
    3   7 12  20

Inorder:   3, 5, 7, 10, 12, 15, 20
Preorder:  10, 5, 3, 7, 15, 12, 20
Postorder: 3, 7, 5, 12, 20, 15, 10
Level:     10, 5, 15, 3, 7, 12, 20
```

```python
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None

class BST:
    def __init__(self):
        self.root = None
    
    def insert(self, val):
        if not self.root:
            self.root = TreeNode(val)
            return
        curr = self.root
        while True:
            if val < curr.val:
                if not curr.left:
                    curr.left = TreeNode(val)
                    return
                curr = curr.left
            else:
                if not curr.right:
                    curr.right = TreeNode(val)
                    return
                curr = curr.right
    
    def inorder(self, node=None, result=None):
        if result is None:
            result = []
            node = self.root
        if node:
            self.inorder(node.left, result)
            result.append(node.val)
            self.inorder(node.right, result)
        return result
    
    def search(self, val):
        curr = self.root
        while curr:
            if val == curr.val:
                return True
            curr = curr.left if val < curr.val else curr.right
        return False

bst = BST()
for x in [10, 5, 15, 3, 7, 12, 20]:
    bst.insert(x)
print(bst.inorder())  # [3, 5, 7, 10, 12, 15, 20]
print(bst.search(7))   # True
print(bst.search(99))  # False
```

## Graphs

```python
from collections import defaultdict, deque

class Graph:
    def __init__(self, directed=False):
        self.adj = defaultdict(list)
        self.directed = directed
    
    def add_edge(self, u, v):
        self.adj[u].append(v)
        if not self.directed:
            self.adj[v].append(u)
    
    def dfs(self, start):
        visited, order = set(), []
        def _dfs(node):
            visited.add(node)
            order.append(node)
            for neighbor in self.adj[node]:
                if neighbor not in visited:
                    _dfs(neighbor)
        _dfs(start)
        return order
    
    def shortest_path(self, start, end):
        queue = deque([(start, [start])])
        visited = {start}
        while queue:
            node, path = queue.popleft()
            if node == end:
                return path
            for neighbor in self.adj[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        return None

g = Graph()
g.add_edge('A', 'B')
g.add_edge('A', 'C')
g.add_edge('B', 'D')
g.add_edge('C', 'D')
g.add_edge('D', 'E')
print(g.dfs('A'))              # ['A', 'B', 'D', 'C', 'E']
print(g.shortest_path('A', 'E'))  # ['A', 'B', 'D', 'E'] or ['A', 'C', 'D', 'E']
```

## Complexity Cheat Sheet

```
┌─────────────────┬───────────┬───────────┬───────────┐
│ Structure       │ Access    │ Search    │ Insert    │
├─────────────────┼───────────┼───────────┼───────────┤
│ Array/List      │ O(1)      │ O(n)      │ O(n)      │
│ Linked List     │ O(n)      │ O(n)      │ O(1)      │
│ Hash Map        │ O(1) avg  │ O(1) avg  │ O(1) avg  │
│ BST (balanced)  │ O(log n)  │ O(log n)  │ O(log n)  │
│ Stack           │ O(n)      │ O(n)      │ O(1)      │
│ Queue           │ O(n)      │ O(n)      │ O(1)      │
│ Heap            │ O(n)      │ O(log n)  │ O(log n)  │
└─────────────────┴───────────┴───────────┴───────────┘
```

## 🧠 Exercises

### Exercise 1: Implement a MinStack that supports push, pop, and getMin in O(1).

<details>
<summary>Solution</summary>

```python
class MinStack:
    def __init__(self):
        self.stack = []
        self.min_stack = []
    
    def push(self, val):
        self.stack.append(val)
        if not self.min_stack or val <= self.min_stack[-1]:
            self.min_stack.append(val)
    
    def pop(self):
        val = self.stack.pop()
        if val == self.min_stack[-1]:
            self.min_stack.pop()
        return val
    
    def getMin(self):
        return self.min_stack[-1]

s = MinStack()
s.push(5); s.push(3); s.push(7); s.push(2)
print(s.getMin())  # 2
s.pop()
print(s.getMin())  # 3
```

</details>

### Exercise 2: Detect a cycle in a linked list.

<details>
<summary>Solution</summary>

```python
def has_cycle(head):
    slow, fast = head, head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            return True
    return False
```

</details>

### Exercise 3: Find the lowest common ancestor of two nodes in a BST.

<details>
<summary>Solution</summary>

```python
def lca(root, p, q):
    if not root:
        return None
    if p < root.val and q < root.val:
        return lca(root.left, p, q)
    if p > root.val and q > root.val:
        return lca(root.right, p, q)
    return root.val
```

</details>
