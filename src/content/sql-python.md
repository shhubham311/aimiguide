# SQL & Usage in Python

## Table of Contents
1. [Introduction to SQL](#introduction-to-sql)
2. [Relational Database Fundamentals](#relational-database-fundamentals)
3. [SQL Query Structure](#sql-query-structure)
4. [Filtering & Sorting](#filtering--sorting)
5. [Aggregation & Grouping](#aggregation--grouping)
6. [Joins](#joins)
7. [Subqueries & CTEs](#subqueries--ctes)
8. [Window Functions](#window-functions)
9. [Indexing & Optimization](#indexing--optimization)
10. [SQL in Python: sqlite3](#sql-in-python-sqlite3)
11. [SQLAlchemy ORM](#sqlalchemy-orm)
12. [Pandas & SQL Integration](#pandas--sql-integration)
13. [Advanced SQL Patterns](#advanced-sql-patterns)
14. [Exercises with Solutions](#exercises-with-solutions)
15. [ML Connections](#ml-connections)

---

## Introduction to SQL

SQL (Structured Query Language) is the standard language for managing and querying relational databases. In data science and ML engineering, SQL is essential for data extraction, transformation, and feature engineering.

### Why SQL Matters for Data Scientists

```
┌─────────────────────────────────────────────────────┐
│                  Data Pipeline                       │
│                                                       │
│   ┌─────────┐    SQL    ┌──────────┐   Python   ┌───┐│
│   │ Database│ ────────> │ DataFrame│ ─────────> │ ML││
│   │ (RDBMS) │           │ (Pandas) │            │   ││
│   └─────────┘           └──────────┘            └───┘│
│       ▲                                            │  │
│       │         SQL is the bridge                  │  │
│       └────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘
```

### SQL vs NoSQL

| Feature        | SQL (Relational)        | NoSQL                  |
|----------------|------------------------|------------------------|
| Data Model     | Tables with relations  | Key-value, Document,   |
|                |                        | Graph, Column-family   |
| Schema         | Fixed (strict)         | Flexible (dynamic)     |
| Scalability    | Vertical               | Horizontal             |
| ACID           | Yes                    | Varies                 |
| Examples       | PostgreSQL, MySQL,     | MongoDB, Redis,        |
|                | SQLite                 | Cassandra, Neo4j       |

---

## Relational Database Fundamentals

### Tables, Rows, and Columns

A database table is a collection of structured data organized in rows and columns.

```
┌─────────────────────────────────────────────────┐
│                 employees                        │
├──────┬──────────┬────────┬────────┬─────────────┤
│  id  │   name   │ dept   │ salary │ hire_date   │
├──────┼──────────┼────────┼────────┼─────────────┤
│  1   │  Alice   │  Eng   │ 95000  │ 2020-03-15  │
│  2   │  Bob     │  Eng   │ 88000  │ 2021-07-01  │
│  3   │  Carol   │  Sales │ 72000  │ 2019-11-20  │
│  4   │  Dave    │  Eng   │ 102000 │ 2018-05-10  │
│  5   │  Eve     │  HR    │ 68000  │ 2022-01-05  │
└──────┴──────────┴────────┴────────┴─────────────┘

Primary Key: id (uniquely identifies each row)
Foreign Key: dept references departments table
```

### Entity-Relationship Diagram

```
    ┌──────────┐          ┌──────────────┐          ┌──────────┐
    │ departments│ 1    N │   employees  │ N    1  │ projects │
    ├──────────┤<─────────┤──────────────┤─────────>├──────────┤
    │ dept_id  │          │ emp_id       │          │ proj_id  │
    │ name     │          │ name         │          │ title    │
    │ budget   │          │ dept_id (FK) │          │ budget   │
    └──────────┘          │ salary       │          │ start    │
                          └──────────────┘          └──────────┘
                                │ N
                                │
                                ▼ N
                          ┌──────────────┐
                          │  assignments │
                          ├──────────────┤
                          │ emp_id (FK)  │
                          │ proj_id (FK) │
                          │ role         │
                          │ hours        │
                          └──────────────┘
```

### Data Types

| SQL Type      | Python Equivalent  | Description                |
|---------------|-------------------|----------------------------|
| INTEGER       | int               | Whole numbers              |
| REAL/FLOAT    | float             | Decimal numbers            |
| TEXT/VARCHAR  | str               | Variable-length text       |
| BOOLEAN       | bool              | True/False                 |
| DATE          | datetime.date     | Calendar date              |
| TIMESTAMP     | datetime.datetime | Date + time                |
| BLOB          | bytes             | Binary large objects       |
| NUMERIC(p,s)  | Decimal           | Fixed precision decimal    |

---

## SQL Query Structure

### Basic SELECT

```sql
-- Select all columns
SELECT * FROM employees;

-- Select specific columns
SELECT name, salary FROM employees;

-- Select with alias
SELECT name AS employee_name, salary * 12 AS annual_salary
FROM employees;
```

### CREATE TABLE & INSERT

```sql
CREATE TABLE students (
    student_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name       TEXT NOT NULL,
    gpa        REAL CHECK (gpa >= 0.0 AND gpa <= 4.0),
    major      TEXT DEFAULT 'Undeclared',
    grad_year  INTEGER
);

INSERT INTO students (name, gpa, major, grad_year)
VALUES
    ('Alice', 3.8, 'Computer Science', 2024),
    ('Bob', 3.5, 'Mathematics', 2025),
    ('Carol', 3.9, 'Physics', 2024);
```

### SQL Execution Order

Understanding the logical execution order is crucial for writing complex queries:

```
Written Order:          Logical Execution Order:
─────────────           ─────────────────────────
SELECT        (5)       FROM            (1)
FROM          (1)       WHERE           (2)
WHERE         (2)       GROUP BY        (3)
GROUP BY      (3)       HAVING          (4)
HAVING        (4)       SELECT          (5)
ORDER BY      (6)       ORDER BY        (6)
LIMIT         (7)       LIMIT           (7)
```

---

## Filtering & Sorting

### WHERE Clause Operators

```sql
-- Comparison operators
SELECT * FROM employees WHERE salary > 80000;
SELECT * FROM employees WHERE dept = 'Eng';
SELECT * FROM employees WHERE hire_date >= '2021-01-01';

-- Logical operators
SELECT * FROM employees WHERE dept = 'Eng' AND salary > 90000;
SELECT * FROM employees WHERE dept = 'Eng' OR dept = 'Sales';
SELECT * FROM employees WHERE dept NOT IN ('HR');

-- Pattern matching with LIKE
SELECT * FROM employees WHERE name LIKE 'A%';     -- starts with A
SELECT * FROM employees WHERE name LIKE '%ar%';   -- contains 'ar'
SELECT * FROM employees WHERE name LIKE '___e';   -- 4 chars ending in e

-- NULL handling
SELECT * FROM employees WHERE manager IS NULL;
SELECT * FROM employees WHERE manager IS NOT NULL;

-- BETWEEN and IN
SELECT * FROM employees WHERE salary BETWEEN 70000 AND 100000;
SELECT * FROM employees WHERE dept IN ('Eng', 'Sales', 'HR');
```

### ORDER BY & LIMIT

```sql
-- Sort by one column
SELECT * FROM employees ORDER BY salary DESC;

-- Sort by multiple columns
SELECT * FROM employees
ORDER BY dept ASC, salary DESC;

-- Pagination with LIMIT/OFFSET
SELECT * FROM employees
ORDER BY salary DESC
LIMIT 10 OFFSET 20;  -- Page 3, 10 per page

-- Top N per group (SQLite approach)
SELECT * FROM employees e
WHERE (
    SELECT COUNT(*) FROM employees e2
    WHERE e2.dept = e.dept AND e2.salary >= e.salary
) <= 3
ORDER BY dept, salary DESC;
```

---

## Aggregation & Grouping

### Aggregate Functions

```
  salary values: [72, 88, 95, 102, 68] (in thousands)

  ┌──────────────────────────────────────────┐
  │           Aggregation Functions          │
  ├──────────────────────────────────────────┤
  │  COUNT(*)  = 5      (number of rows)     │
  │  SUM       = 425    (total)              │
  │  AVG       = 85.0   (mean)               │
  │  MIN       = 68     (minimum)            │
  │  MAX       = 102    (maximum)            │
  │                                          │
  │  Mathematical formulas:                  │
  │  AVG = (1/n) * Σ x_i                    │
  │  VAR = (1/n) * Σ (x_i - μ)²            │
  └──────────────────────────────────────────┘
```

```sql
-- Basic aggregation
SELECT
    COUNT(*)      AS total_employees,
    SUM(salary)   AS total_payroll,
    AVG(salary)   AS avg_salary,
    MIN(salary)   AS min_salary,
    MAX(salary)   AS max_salary,
    ROUND(AVG(salary), 2) AS avg_salary_rounded
FROM employees;

-- Group by department
SELECT
    dept,
    COUNT(*)    AS count,
    AVG(salary) AS avg_salary
FROM employees
GROUP BY dept
ORDER BY avg_salary DESC;
```

### GROUP BY with HAVING

```sql
-- HAVING filters after grouping (WHERE filters before)
SELECT
    dept,
    COUNT(*) AS emp_count,
    AVG(salary) AS avg_salary
FROM employees
GROUP BY dept
HAVING COUNT(*) >= 2 AND AVG(salary) > 75000
ORDER BY avg_salary DESC;

-- Difference: WHERE vs HAVING
-- WHERE: filters individual rows before grouping
-- HAVING: filters groups after aggregation

SELECT
    dept,
    COUNT(*) AS count
FROM employees
WHERE salary > 70000       -- filter rows first
GROUP BY dept
HAVING COUNT(*) >= 2;       -- then filter groups
```

### CASE Expressions

```sql
-- Conditional logic in SQL
SELECT
    name,
    salary,
    CASE
        WHEN salary >= 100000 THEN 'Senior'
        WHEN salary >= 80000  THEN 'Mid-Level'
        WHEN salary >= 60000  THEN 'Junior'
        ELSE 'Entry'
    END AS salary_band,
    CASE
        WHEN salary >= 100000 THEN salary * 0.10
        WHEN salary >= 80000  THEN salary * 0.08
        ELSE salary * 0.05
    END AS bonus
FROM employees;

-- Pivot-like transformation using CASE
SELECT
    dept,
    SUM(CASE WHEN salary >= 90000 THEN 1 ELSE 0 END) AS high_earners,
    SUM(CASE WHEN salary >= 70000 AND salary < 90000 THEN 1 ELSE 0 END) AS mid_earners,
    SUM(CASE WHEN salary < 70000 THEN 1 ELSE 0 END) AS low_earners
FROM employees
GROUP BY dept;
```

---

## Joins

### Types of Joins

```
Table A (employees)          Table B (departments)
┌──────┬──────────┐          ┌──────────┬──────────┐
│  id  │ dept_id  │          │ dept_id  │   name   │
├──────┼──────────┤          ├──────────┼──────────┤
│  1   │  Eng     │          │  Eng     │ Engineering│
│  2   │  Sales   │          │  Sales   │ Sales     │
│  3   │  HR      │          │  HR      │ Human Res.│
│  4   │  NULL    │          │  Finance │ Finance   │
└──────┴──────────┘          └──────────┴──────────┘

INNER JOIN          LEFT JOIN            RIGHT JOIN        FULL OUTER JOIN
(A ∩ B)             (A ∪ matching)       (B ∪ matching)    (A ∪ B)
┌──────┬──────┐    ┌──────┬──────┐     ┌──────┬──────┐   ┌──────┬──────┐
│  id  │ name │    │  id  │ name │     │  id  │ name │   │  id  │ name │
├──────┼──────┤    ├──────┼──────┤     ├──────┼──────┤   ├──────┼──────┤
│  1   │ Eng  │    │  1   │ Eng  │     │  1   │ Eng  │   │  1   │ Eng  │
│  2   │ Sales│    │  2   │ Sales│     │  2   │ Sales│   │  2   │ Sales│
│  3   │ HR   │    │  3   │ HR   │     │  3   │ HR   │   │  3   │ HR   │
└──────┴──────┘    │  4   │ NULL │     │ NULL │ Fin  │   │  4   │ NULL │
                   └──────┴──────┘     └──────┴──────┘   │ NULL │ Fin  │
                                                        └──────┴──────┘
```

### Join Examples

```sql
-- INNER JOIN: only matching rows
SELECT e.name, e.salary, d.name AS dept_name
FROM employees e
INNER JOIN departments d ON e.dept_id = d.dept_id;

-- LEFT JOIN: all rows from left table
SELECT e.name, d.name AS dept_name
FROM employees e
LEFT JOIN departments d ON e.dept_id = d.dept_id;

-- Self-join: join table with itself
SELECT e1.name AS employee, e2.name AS manager
FROM employees e1
LEFT JOIN employees e2 ON e1.manager_id = e2.emp_id;

-- Multi-table join
SELECT e.name, d.name AS dept, p.title AS project, a.role
FROM employees e
JOIN departments d ON e.dept_id = d.dept_id
JOIN assignments a ON e.emp_id = a.emp_id
JOIN projects p ON a.proj_id = p.proj_id;

-- CROSS JOIN (Cartesian product): every combination
-- Useful for generating date series
SELECT
    d.date_val,
    p.product_id
FROM dates d
CROSS JOIN products p;
```

### Join with Aggregation

```sql
-- Department statistics with join
SELECT
    d.name AS department,
    COUNT(e.emp_id) AS num_employees,
    AVG(e.salary) AS avg_salary,
    MAX(e.salary) AS max_salary
FROM departments d
LEFT JOIN employees e ON d.dept_id = e.dept_id
GROUP BY d.dept_id, d.name
ORDER BY avg_salary DESC;
```

---

## Subqueries & CTEs

### Subqueries

```sql
-- Scalar subquery (returns single value)
SELECT name, salary
FROM employees
WHERE salary > (SELECT AVG(salary) FROM employees);

-- IN subquery
SELECT name FROM employees
WHERE dept_id IN (
    SELECT dept_id FROM departments
    WHERE budget > 500000
);

-- EXISTS subquery
SELECT d.name, d.budget
FROM departments d
WHERE EXISTS (
    SELECT 1 FROM employees e
    WHERE e.dept_id = d.dept_id AND e.salary > 100000
);

-- Correlated subquery (executes for each row)
SELECT e.name, e.salary, e.dept_id,
    (SELECT AVG(salary) FROM employees e2
     WHERE e2.dept_id = e.dept_id) AS dept_avg_salary
FROM employees e;
```

### Common Table Expressions (CTEs)

```sql
-- Simple CTE
WITH dept_stats AS (
    SELECT
        dept_id,
        COUNT(*) AS emp_count,
        AVG(salary) AS avg_salary
    FROM employees
    GROUP BY dept_id
)
SELECT d.name, ds.emp_count, ds.avg_salary
FROM departments d
JOIN dept_stats ds ON d.dept_id = ds.dept_id;

-- Multiple CTEs
WITH
high_earners AS (
    SELECT * FROM employees WHERE salary > 90000
),
dept_counts AS (
    SELECT dept_id, COUNT(*) AS cnt FROM employees GROUP BY dept_id
)
SELECT he.name, he.salary, dc.cnt AS dept_size
FROM high_earners he
JOIN dept_counts dc ON he.dept_id = dc.dept_id;

-- Recursive CTE (hierarchical data)
WITH RECURSIVE hierarchy AS (
    -- Base case: top-level employees (no manager)
    SELECT emp_id, name, manager_id, 1 AS level
    FROM employees
    WHERE manager_id IS NULL

    UNION ALL

    -- Recursive case: employees under a manager
    SELECT e.emp_id, e.name, e.manager_id, h.level + 1
    FROM employees e
    JOIN hierarchy h ON e.manager_id = h.emp_id
)
SELECT * FROM hierarchy ORDER BY level, name;
```

### CTE vs Subquery: Performance Considerations

```
┌──────────────────────────────────────────────────┐
│         CTE vs Subquery Decision Matrix           │
├──────────────────────────────────────────────────┤
│                                                    │
│  Use CTE when:                                     │
│  ├── Logic is complex and needs readability       │
│  ├── Same subquery referenced multiple times      │
│  ├── Building recursive queries                   │
│  └── Debugging (can test CTE independently)       │
│                                                    │
│  Use Subquery when:                                │
│  ├── Simple one-time filter                       │
│  ├── Scalar subquery in SELECT/WHERE              │
│  └── Minimal overhead needed                      │
│                                                    │
│  Note: In PostgreSQL, CTEs were "optimization     │
│  fences" before v12 — they always materialized.   │
│  Now, non-recursive CTEs can be inlined.          │
└──────────────────────────────────────────────────┘
```

---

## Window Functions

Window functions perform calculations across a set of rows related to the current row, without collapsing the result set.

### Syntax

```sql
FUNCTION_NAME(args) OVER (
    [PARTITION BY partition_col]
    [ORDER BY sort_col [ASC|DESC]]
    [frame_clause]
)
```

### Common Window Functions

```
Data: employees sorted by dept, salary

┌──────────┬──────┬─────────┬─────────────┬──────────┬──────────┐
│   name   │ dept │ salary  │ ROW_NUMBER  │ RANK     │ DENSE_RK │
├──────────┼──────┼─────────┼─────────────┼──────────┼──────────┤
│  Alice   │ Eng  │  95000  │      1      │    1     │    1     │
│  Dave    │ Eng  │ 102000  │      2      │    1     │    1     │  ← same rank
│  Bob     │ Eng  │  88000  │      3      │    3     │    2     │  ← same rank
│  Carol   │ Sales│  72000  │      1      │    1     │    1     │  (partition reset)
│  Frank   │ Sales│  72000  │      2      │    1     │    1     │
│  Eve     │ HR   │  68000  │      1      │    1     │    1     │
└──────────┴──────┴─────────┴─────────────┴──────────┴──────────┘

ROW_NUMBER: Unique sequential number within partition
RANK:       Same rank for ties, gaps after ties
DENSE_RANK: Same rank for ties, no gaps
```

### Practical Window Function Examples

```sql
-- Running total
SELECT
    order_date,
    amount,
    SUM(amount) OVER (
        ORDER BY order_date
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS running_total
FROM orders;

-- Moving average (7-day window)
SELECT
    date,
    value,
    AVG(value) OVER (
        ORDER BY date
        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    ) AS moving_avg_7d
FROM daily_metrics;

-- Percentile within department
SELECT
    name,
    dept,
    salary,
    NTILE(4) OVER (PARTITION BY dept ORDER BY salary DESC) AS salary_quartile,
    PERCENT_RANK() OVER (PARTITION BY dept ORDER BY salary) AS pct_rank
FROM employees;

-- Lead and Lag
SELECT
    date,
    revenue,
    LAG(revenue, 1) OVER (ORDER BY date) AS prev_day_revenue,
    LEAD(revenue, 1) OVER (ORDER BY date) AS next_day_revenue,
    revenue - LAG(revenue, 1) OVER (ORDER BY date) AS day_over_day_change
FROM daily_revenue;

-- First and last value in group
SELECT
    user_id,
    session_start,
    FIRST_VALUE(page) OVER (
        PARTITION BY user_id ORDER BY session_start
        ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
    ) AS landing_page,
    LAST_VALUE(page) OVER (
        PARTITION BY user_id ORDER BY session_start
        ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
    ) AS exit_page
FROM user_sessions;
```

### Frame Clauses Explained

```
Frame: ROWS BETWEEN start AND end

  ┌─────────────────────────────────────────┐
  │           Window Frame Options          │
  ├─────────────────────────────────────────┤
  │                                          │
  │  Start bounds:                           │
  │  • UNBOUNDED PRECEDING  (beginning)     │
  │  • N PRECEDING           (N rows back)  │
  │  • CURRENT ROW           (current)      │
  │                                          │
  │  End bounds:                             │
  │  • CURRENT ROW           (current)      │
  │  • N FOLLOWING           (N rows ahead) │
  │  • UNBOUNDED FOLLOWING   (end)          │
  │                                          │
  │  Shortcuts:                              │
  │  • RANGE UNBOUNDED PRECEDING = all rows  │
  │    from start to current                 │
  │  • ROWS BETWEEN UNBOUNDED PRECEDING AND  │
  │    UNBOUNDED FOLLOWING = entire partition│
  └─────────────────────────────────────────┘
```

---

## Indexing & Optimization

### What is an Index?

```
Without Index (Full Table Scan):
┌───────┐     ┌───────────────────────────┐
│ Query │ ──> │ Row 1? Row 2? Row 3? ...  │ O(n)
└───────┘     │ Scan ALL rows to find     │
              │ matching records           │
              └───────────────────────────┘

With Index (B-Tree):
┌───────┐     ┌───────────────────────────┐
│ Query │ ──> │ Binary search tree lookup │ O(log n)
└───────┘     │ Skip most rows directly   │
              └───────────────────────────┘
```

### Creating Indexes

```sql
-- Single column index
CREATE INDEX idx_emp_dept ON employees(dept);

-- Composite index (order matters!)
CREATE INDEX idx_emp_dept_salary ON employees(dept, salary);

-- Unique index
CREATE UNIQUE INDEX idx_emp_email ON employees(email);

-- Covering index (includes extra columns)
CREATE INDEX idx_emp_dept_covering ON employees(dept) INCLUDE (name, salary);

-- Check existing indexes
SELECT * FROM sqlite_master WHERE type = 'index';

-- Explain query plan
EXPLAIN QUERY PLAN
SELECT * FROM employees WHERE dept = 'Eng';
```

### Index Selection Rules

```
When to create an index:
  ✓ Columns frequently in WHERE clauses
  ✓ Columns used in JOIN conditions
  ✓ Columns in ORDER BY (avoids sort)
  ✓ High cardinality columns (many unique values)

When NOT to create an index:
  ✗ Small tables (< 1000 rows)
  ✗ Columns with many NULL values
  ✗ Tables with frequent INSERT/UPDATE/DELETE
  ✗ Low cardinality columns (e.g., boolean)

Leftmost prefix rule for composite indexes:
  Index: (A, B, C)
  ✓ WHERE A = ...              (uses index)
  ✓ WHERE A = ... AND B = ...  (uses index)
  ✗ WHERE B = ...              (cannot use index)
  ✗ WHERE C = ...              (cannot use index)
```

### Query Optimization Tips

```sql
-- BAD: Function on indexed column prevents index usage
SELECT * FROM employees WHERE LOWER(name) = 'alice';

-- GOOD: Use sargable (searchable) expressions
SELECT * FROM employees WHERE name = 'Alice';
-- or store normalized value in separate column

-- BAD: Leading wildcard in LIKE
SELECT * FROM employees WHERE name LIKE '%son';

-- GOOD: Trailing wildcard can use index
SELECT * FROM employees WHERE name LIKE 'John%';

-- Avoid SELECT * — specify columns
SELECT emp_id, name, salary FROM employees WHERE dept = 'Eng';

-- Use UNION ALL instead of UNION when duplicates OK
-- UNION performs expensive deduplication
SELECT city FROM customers
UNION ALL  -- faster
SELECT city FROM suppliers;
```

---

## SQL in Python: sqlite3

### Connecting to SQLite

```python
import sqlite3
from pathlib import Path

# Create/connect to database
db_path = "company.db"
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Enable foreign keys (off by default in SQLite)
cursor.execute("PRAGMA foreign_keys = ON;")

print(f"SQLite version: {sqlite3.sqlite_version}")
```

### Creating Tables and Inserting Data

```python
def setup_database(conn):
    cursor = conn.cursor()

    # Create departments table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS departments (
            dept_id TEXT PRIMARY KEY,
            name    TEXT NOT NULL,
            budget  REAL
        )
    """)

    # Create employees table with foreign key
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS employees (
            emp_id    INTEGER PRIMARY KEY AUTOINCREMENT,
            name      TEXT NOT NULL,
            dept_id   TEXT,
            salary    REAL,
            hire_date TEXT,
            manager_id INTEGER,
            FOREIGN KEY (dept_id) REFERENCES departments(dept_id),
            FOREIGN KEY (manager_id) REFERENCES employees(emp_id)
        )
    """)

    # Insert sample data
    departments = [
        ('ENG', 'Engineering', 2000000),
        ('SALES', 'Sales', 1000000),
        ('HR', 'Human Resources', 500000),
    ]
    cursor.executemany(
        "INSERT OR IGNORE INTO departments VALUES (?, ?, ?)",
        departments
    )

    employees = [
        ('Alice', 'ENG', 95000, '2020-03-15', None),
        ('Bob', 'ENG', 88000, '2021-07-01', 1),
        ('Carol', 'SALES', 72000, '2019-11-20', None),
        ('Dave', 'ENG', 102000, '2018-05-10', None),
        ('Eve', 'HR', 68000, '2022-01-05', None),
        ('Frank', 'SALES', 78000, '2020-08-15', 3),
        ('Grace', 'ENG', 115000, '2017-02-28', 4),
    ]
    cursor.executemany(
        "INSERT INTO employees (name, dept_id, salary, hire_date, manager_id) "
        "VALUES (?, ?, ?, ?, ?)",
        employees
    )

    conn.commit()

setup_database(conn)
```

### Querying with sqlite3

```python
def query_examples(conn):
    cursor = conn.cursor()

    # Fetch all results
    cursor.execute("SELECT name, salary FROM employees ORDER BY salary DESC")
    rows = cursor.fetchall()
    print("All employees by salary:")
    for row in rows:
        print(f"  {row[0]}: ${row[1]:,}")

    # Fetch one result
    cursor.execute("SELECT AVG(salary) FROM employees")
    avg_salary = cursor.fetchone()[0]
    print(f"\nAverage salary: ${avg_salary:,.2f}")

    # Parameterized queries (prevents SQL injection!)
    cursor.execute(
        "SELECT name, salary FROM employees WHERE dept_id = ? AND salary > ?",
        ('ENG', 90000)
    )
    print("\nHigh-earning engineers:")
    for row in cursor.fetchall():
        print(f"  {row[0]}: ${row[1]:,}")

    # Using Row factory for named columns
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("""
        SELECT name, salary, dept_id FROM employees
        WHERE salary > (SELECT AVG(salary) FROM employees)
    """)
    print("\nAbove-average earners:")
    for row in cursor.fetchall():
        print(f"  {row['name']} ({row['dept_id']}): ${row['salary']:,}")

query_examples(conn)
```

### Transaction Management

```python
def transaction_example(conn):
    """Demonstrates ACID transactions"""
    cursor = conn.cursor()

    try:
        # Start implicit transaction
        cursor.execute("BEGIN TRANSACTION")

        # Transfer employee between departments
        emp_name = 'Bob'
        old_dept = 'ENG'
        new_dept = 'SALES'

        # Update employee department
        cursor.execute(
            "UPDATE employees SET dept_id = ? WHERE name = ?",
            (new_dept, emp_name)
        )

        # Update department budgets
        cursor.execute(
            "UPDATE departments SET budget = budget - ? WHERE dept_id = ?",
            (88000 * 0.1, old_dept)
        )
        cursor.execute(
            "UPDATE departments SET budget = budget + ? WHERE dept_id = ?",
            (88000 * 0.1, new_dept)
        )

        # Verify constraints
        if cursor.rowcount == 0:
            raise ValueError("No rows affected — rollback needed")

        conn.commit()
        print("Transaction committed successfully")

    except Exception as e:
        conn.rollback()
        print(f"Transaction rolled back: {e}")

    # Verify result
    cursor.execute("SELECT name, dept_id FROM employees WHERE name = 'Bob'")
    print(f"Bob's new department: {cursor.fetchone()[1]}")
```

---

## SQLAlchemy ORM

### Setting Up SQLAlchemy

```python
from sqlalchemy import (
    create_engine, Column, Integer, String, Float,
    ForeignKey, DateTime, CheckConstraint, text
)
from sqlalchemy.orm import (
    declarative_base, relationship, sessionmaker, Session
)
from datetime import datetime

# Create engine
engine = create_engine('sqlite:///company_orm.db', echo=False)
Base = declarative_base()


# Define Models
class Department(Base):
    __tablename__ = 'departments'

    dept_id = Column(String, primary_key=True)
    name = Column(String(100), nullable=False)
    budget = Column(Float, default=0.0)

    # Relationship: one department -> many employees
    employees = relationship('Employee', back_populates='department')

    def __repr__(self):
        return f"<Department {self.dept_id}: {self.name}>"


class Employee(Base):
    __tablename__ = 'employees'

    emp_id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100), nullable=False)
    dept_id = Column(String, ForeignKey('departments.dept_id'))
    salary = Column(Float)
    hire_date = Column(String)
    manager_id = Column(Integer, ForeignKey('employees.emp_id'))

    # Relationships
    department = relationship('Department', back_populates='employees')
    manager = relationship('Employee', remote_side=[emp_id])

    __table_args__ = (
        CheckConstraint('salary >= 0', name='check_salary_positive'),
    )

    def __repr__(self):
        return f"<Employee {self.name}: ${self.salary:,}>"


# Create tables
Base.metadata.create_all(engine)

# Create session
SessionLocal = sessionmaker(bind=engine)
session = SessionLocal()
```

### CRUD Operations with ORM

```python
def orm_operations():
    session = SessionLocal()

    # CREATE
    eng_dept = Department(dept_id='ENG', name='Engineering', budget=2000000)
    sales_dept = Department(dept_id='SALES', name='Sales', budget=1000000)
    session.add_all([eng_dept, sales_dept])

    alice = Employee(name='Alice', dept_id='ENG', salary=95000, hire_date='2020-03-15')
    bob = Employee(name='Bob', dept_id='ENG', salary=88000, hire_date='2021-07-01')
    carol = Employee(name='Carol', dept_id='SALES', salary=72000, hire_date='2019-11-20')
    session.add_all([alice, bob, carol])
    session.commit()

    # READ
    # Get by primary key
    emp = session.get(Employee, 1)
    print(f"Employee 1: {emp}")

    # Filter with conditions
    engineers = session.query(Employee).filter(
        Employee.dept_id == 'ENG'
    ).order_by(Employee.salary.desc()).all()
    print(f"\nEngineers: {engineers}")

    # Complex queries
    high_earners = session.query(Employee).filter(
        Employee.salary > session.query(
            session.query(Employee.salary).order_by(Employee.salary.desc()).limit(10)
                .correlate(None).scalar_subquery()
        )
    ).all()

    # Joins
    results = session.query(Employee.name, Department.name).join(
        Department, Employee.dept_id == Department.dept_id
    ).all()
    print(f"\nEmployee-Department joins: {results}")

    # Aggregation
    from sqlalchemy import func
    dept_stats = session.query(
        Department.name,
        func.count(Employee.emp_id),
        func.avg(Employee.salary)
    ).join(Employee).group_by(Department.dept_id).all()

    print("\nDepartment statistics:")
    for name, count, avg_sal in dept_stats:
        print(f"  {name}: {count} employees, avg ${avg_sal:,.0f}")

    # UPDATE
    session.query(Employee).filter(Employee.name == 'Alice').update(
        {'salary': Employee.salary * 1.10}
    )
    session.commit()

    # DELETE
    session.query(Employee).filter(Employee.name == 'Bob').delete()
    session.commit()

    session.close()
```

---

## Pandas & SQL Integration

### Reading SQL into DataFrames

```python
import pandas as pd
import sqlite3

# Connect to database
conn = sqlite3.connect('company.db')

# Read entire table
df_employees = pd.read_sql("SELECT * FROM employees", conn)
print(df_employees.head())

# Read with parameters
df_eng = pd.read_sql(
    "SELECT * FROM employees WHERE dept_id = ?",
    conn,
    params=('ENG',)
)

# Read complex query
df_stats = pd.read_sql("""
    SELECT
        d.name AS department,
        COUNT(e.emp_id) AS num_employees,
        AVG(e.salary) AS avg_salary,
        MIN(e.salary) AS min_salary,
        MAX(e.salary) AS max_salary,
        AVG(CASE WHEN e.hire_date >= '2021-01-01' THEN 1 ELSE 0 END) AS new_hire_rate
    FROM departments d
    LEFT JOIN employees e ON d.dept_id = e.dept_id
    GROUP BY d.dept_id, d.name
    ORDER BY avg_salary DESC
""", conn)
print(df_stats)
```

### Writing DataFrames to SQL

```python
import numpy as np

# Create sample DataFrame
np.random.seed(42)
n = 1000
df_sales = pd.DataFrame({
    'sale_id': range(1, n + 1),
    'product': np.random.choice(['Widget', 'Gadget', 'Gizmo', 'Doohickey'], n),
    'quantity': np.random.randint(1, 20, n),
    'price': np.round(np.random.uniform(10, 500, n), 2),
    'sale_date': pd.date_range('2023-01-01', periods=n, freq='D'),
    'region': np.random.choice(['North', 'South', 'East', 'West'], n),
})

# Calculate derived columns
df_sales['total'] = df_sales['quantity'] * df_sales['price']

# Write to SQL
df_sales.to_sql('sales', conn, if_exists='replace', index=False)

# Verify
result = pd.read_sql("""
    SELECT
        product,
        COUNT(*) AS num_sales,
        SUM(total) AS revenue,
        AVG(total) AS avg_sale
    FROM sales
    GROUP BY product
    ORDER BY revenue DESC
""", conn)
print("Sales by product:")
print(result)

conn.close()
```

### Chunked Processing for Large Datasets

```python
def process_large_dataset(conn, chunk_size=10000):
    """Process large SQL results in chunks to manage memory"""
    query = "SELECT * FROM large_table ORDER BY id"

    # Read in chunks
    total_processed = 0
    for chunk in pd.read_sql(query, conn, chunksize=chunk_size):
        # Process each chunk
        processed = chunk.dropna(subset=['important_column'])
        total_processed += len(processed)

        # Could write back chunk by chunk
        # processed.to_sql('processed', conn, if_exists='append', index=False)

    print(f"Processed {total_processed:,} rows")

    # Alternatively, use server-side cursors for raw SQL
    cursor = conn.cursor()
    cursor.execute(query)
    while True:
        batch = cursor.fetchmany(chunk_size)
        if not batch:
            break
        # Process batch
        total_processed += len(batch)
```

---

## Advanced SQL Patterns

### Gaps and Islands

```sql
-- Find consecutive date ranges (sessions, streaks)
WITH numbered AS (
    SELECT
        user_id,
        login_date,
        -- Subtract row number from date to get island ID
        DATE(login_date, '-' || ROW_NUMBER() OVER (
            PARTITION BY user_id ORDER BY login_date
        ) || ' days') AS island_id
    FROM user_logins
)
SELECT
    user_id,
    MIN(login_date) AS streak_start,
    MAX(login_date) AS streak_end,
    COUNT(*) AS consecutive_days
FROM numbered
GROUP BY user_id, island_id
HAVING COUNT(*) >= 3  -- at least 3-day streak
ORDER BY consecutive_days DESC;
```

### Median Calculation

```sql
-- Median salary using window functions
WITH ranked AS (
    SELECT
        salary,
        ROW_NUMBER() OVER (ORDER BY salary) AS rn,
        COUNT(*) OVER () AS total
    FROM employees
)
SELECT AVG(salary) AS median_salary
FROM ranked
WHERE rn IN (total / 2, total / 2 + 1);

-- Or simpler for PostgreSQL:
-- SELECT percentile_cont(0.5) WITHIN GROUP (ORDER BY salary) AS median
-- FROM employees;
```

### Time Series Analysis

```sql
-- Year-over-year comparison
SELECT
    EXTRACT(YEAR FROM sale_date) AS year,
    EXTRACT(MONTH FROM sale_date) AS month,
    SUM(amount) AS revenue,
    LAG(SUM(amount), 12) OVER (
        PARTITION BY EXTRACT(MONTH FROM sale_date)
        ORDER BY EXTRACT(YEAR FROM sale_date)
    ) AS prev_year_revenue,
    ROUND(
        (SUM(amount) - LAG(SUM(amount), 12) OVER (
            PARTITION BY EXTRACT(MONTH FROM sale_date)
            ORDER BY EXTRACT(YEAR FROM sale_date)
        )) * 100.0 / LAG(SUM(amount), 12) OVER (
            PARTITION BY EXTRACT(MONTH FROM sale_date)
            ORDER BY EXTRACT(YEAR FROM sale_date)
        ), 2
    ) AS yoy_growth_pct
FROM sales
GROUP BY EXTRACT(YEAR FROM sale_date), EXTRACT(MONTH FROM sale_date)
ORDER BY year, month;
```

### Pivot Table Query

```sql
-- Convert rows to columns (pivot)
SELECT
    dept_id,
    SUM(CASE WHEN salary_band = 'Junior' THEN count ELSE 0 END) AS junior,
    SUM(CASE WHEN salary_band = 'Mid' THEN count ELSE 0 END) AS mid,
    SUM(CASE WHEN salary_band = 'Senior' THEN count ELSE 0 END) AS senior
FROM (
    SELECT
        dept_id,
        CASE
            WHEN salary < 70000 THEN 'Junior'
            WHEN salary < 100000 THEN 'Mid'
            ELSE 'Senior'
        END AS salary_band,
        COUNT(*) AS count
    FROM employees
    GROUP BY dept_id, salary_band
) sub
GROUP BY dept_id;
```

---

## Exercises with Solutions

### Exercise 1: Student Database

```sql
-- Problem: Given a students table with (id, name, major, gpa, grad_year),
-- write a query to find the top 3 students by GPA in each major.

-- Solution:
WITH ranked AS (
    SELECT
        name, major, gpa, grad_year,
        DENSE_RANK() OVER (PARTITION BY major ORDER BY gpa DESC) AS gpa_rank
    FROM students
)
SELECT name, major, gpa, grad_year, gpa_rank
FROM ranked
WHERE gpa_rank <= 3
ORDER BY major, gpa_rank;
```

### Exercise 2: E-commerce Revenue Analysis

```sql
-- Problem: Calculate running monthly revenue and month-over-month growth.

-- Solution:
WITH monthly_rev AS (
    SELECT
        DATE_TRUNC('month', order_date) AS month,
        SUM(amount) AS revenue
    FROM orders
    GROUP BY DATE_TRUNC('month', order_date)
)
SELECT
    month,
    revenue,
    SUM(revenue) OVER (ORDER BY month) AS cumulative_revenue,
    LAG(revenue) OVER (ORDER BY month) AS prev_month,
    ROUND(
        (revenue - LAG(revenue) OVER (ORDER BY month)) * 100.0 /
        LAG(revenue) OVER (ORDER BY month), 2
    ) AS mom_growth_pct
FROM monthly_rev
ORDER BY month;
```

### Exercise 3: User Retention (Python + SQL)

```python
"""
Problem: Calculate day-1, day-7, and day-30 retention rates
for users who signed up in January 2024.
"""
import sqlite3
import pandas as pd

conn = sqlite3.connect(':memory:')

# Create sample data
conn.executescript("""
    CREATE TABLE users (
        user_id INTEGER PRIMARY KEY,
        signup_date DATE
    );
    CREATE TABLE events (
        user_id INTEGER,
        event_date DATE,
        event_type TEXT
    );
""")

# Insert sample data
import random
from datetime import date, timedelta

random.seed(42)
users = []
events = []
start = date(2024, 1, 1)

for i in range(1, 101):
    signup = start + timedelta(days=random.randint(0, 30))
    users.append((i, signup.isoformat()))

    # Generate activity events
    for d in range(0, 45):
        if random.random() < 0.3:  # 30% daily activity
            events.append((i, (signup + timedelta(days=d)).isoformat(), 'pageview'))

conn.executemany("INSERT INTO users VALUES (?, ?)", users)
conn.executemany("INSERT INTO events VALUES (?, ?, ?)", events)
conn.commit()

# Solution: Retention query
retention = pd.read_sql("""
    WITH signup_cohort AS (
        SELECT user_id, signup_date
        FROM users
        WHERE signup_date >= '2024-01-01' AND signup_date < '2024-02-01'
    ),
    active_days AS (
        SELECT DISTINCT
            s.user_id,
            s.signup_date,
            CAST(julianday(e.event_date) - julianday(s.signup_date) AS INTEGER) AS days_since_signup
        FROM signup_cohort s
        JOIN events e ON s.user_id = e.user_id
    ),
    retention_matrix AS (
        SELECT
            user_id,
            MAX(CASE WHEN days_since_signup = 1 THEN 1 ELSE 0 END) AS day_1,
            MAX(CASE WHEN days_since_signup = 7 THEN 1 ELSE 0 END) AS day_7,
            MAX(CASE WHEN days_since_signup = 30 THEN 1 ELSE 0 END) AS day_30
        FROM active_days
        GROUP BY user_id
    )
    SELECT
        ROUND(AVG(day_1) * 100, 2) AS day_1_retention_pct,
        ROUND(AVG(day_7) * 100, 2) AS day_7_retention_pct,
        ROUND(AVG(day_30) * 100, 2) AS day_30_retention_pct
    FROM retention_matrix
""", conn)

print("Retention Rates:")
print(retention)
conn.close()
```

### Exercise 4: Nth Highest Salary

```sql
-- Problem: Find the Nth highest distinct salary using CTE.
-- N is a parameter (e.g., N=3 means third highest salary).

-- Solution using DENSE_RANK:
WITH ranked_salaries AS (
    SELECT DISTINCT salary, DENSE_RANK() OVER (ORDER BY salary DESC) AS rnk
    FROM employees
)
SELECT salary AS nth_highest_salary
FROM ranked_salaries
WHERE rnk = 3;  -- Replace 3 with desired N

-- Alternative using LIMIT/OFFSET:
SELECT DISTINCT salary
FROM employees
ORDER BY salary DESC
LIMIT 1 OFFSET 2;  -- OFFSET = N - 1
```

### Exercise 5: Python ETL Pipeline

```python
"""
Problem: Build an ETL pipeline that:
1. Reads data from multiple CSV files
2. Cleans and transforms data
3. Loads into a normalized SQLite database
4. Validates data integrity
"""
import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path

def etl_pipeline(csv_dir: str, db_path: str):
    """Complete ETL pipeline with data validation"""

    # ---- EXTRACT ----
    print("=" * 50)
    print("EXTRACT PHASE")
    print("=" * 50)

    csv_path = Path(csv_dir)
    dfs = {}
    for csv_file in csv_path.glob("*.csv"):
        table_name = csv_file.stem
        dfs[table_name] = pd.read_csv(csv_file)
        print(f"  Loaded {table_name}: {len(dfs[table_name])} rows")

    # ---- TRANSFORM ----
    print("\n" + "=" * 50)
    print("TRANSFORM PHASE")
    print("=" * 50)

    for name, df in dfs.items():
        # Strip whitespace from string columns
        str_cols = df.select_dtypes(include='object').columns
        df[str_cols] = df[str_cols].apply(lambda x: x.str.strip() if x.dtype == 'object' else x)

        # Standardize column names
        df.columns = [c.lower().replace(' ', '_') for c in df.columns]

        # Handle missing values
        numeric_cols = df.select_dtypes(include=np.number).columns
        df[numeric_cols] = df[numeric_cols].fillna(0)

        print(f"  Transformed {name}: {len(df)} rows, {len(df.columns)} columns")

    # ---- LOAD ----
    print("\n" + "=" * 50)
    print("LOAD PHASE")
    print("=" * 50)

    conn = sqlite3.connect(db_path)

    for name, df in dfs.items():
        df.to_sql(name, conn, if_exists='replace', index=False)
        count = pd.read_sql(f"SELECT COUNT(*) FROM {name}", conn).iloc[0, 0]
        print(f"  Loaded {name}: {count} rows into database")

    # ---- VALIDATE ----
    print("\n" + "=" * 50)
    print("VALIDATION PHASE")
    print("=" * 50)

    cursor = conn.cursor()
    for name in dfs:
        cursor.execute(f"SELECT COUNT(*) FROM {name}")
        count = cursor.fetchone()[0]
        expected = len(dfs[name])
        status = "PASS" if count == expected else "FAIL"
        print(f"  [{status}] {name}: expected {expected}, got {count}")

    conn.close()
    print("\nETL pipeline complete!")

# Usage:
# etl_pipeline("./data", "analytics.db")
```

---

## ML Connections

### SQL in Machine Learning Workflows

```
┌──────────────────────────────────────────────────────────────┐
│                   ML Pipeline with SQL                        │
│                                                               │
│  ┌─────────┐   SQL     ┌──────────┐  Pandas   ┌───────────┐ │
│  │ Data    │ ────────> │ Feature  │ ────────> │  Model    │ │
│  │ Warehouse│          │ Store    │           │  Training │ │
│  │(SQL DB) │          │(SQL DB)  │           │ (sklearn) │ │
│  └─────────┘          └──────────┘           └───────────┘ │
│       ▲                      ▲                        │      │
│       │                      │                        ▼      │
│       │                      │                  ┌───────────┐ │
│       │                Feature engineering     │ Predictions│ │
│       │                with SQL window         │ Store     │ │
│       │                functions               │ (SQL DB)  │ │
│       │                      │                  └───────────┘ │
│       └──────────────────────┘                               │
│                                                               │
│  Key SQL skills for ML:                                       │
│  ├── Feature extraction from relational data                  │
│  ├── Time-series feature engineering                          │
│  ├── Aggregation for user/item profiles                       │
│  ├── Data quality checks                                      │
│  └── A/B test result queries                                  │
└──────────────────────────────────────────────────────────────┘
```

### Feature Engineering with SQL

```sql
-- User behavioral features for a recommendation system
WITH user_features AS (
    SELECT
        user_id,
        -- Engagement features
        COUNT(DISTINCT session_id) AS num_sessions,
        COUNT(*) AS total_events,
        COUNT(DISTINCT product_id) AS unique_products_viewed,

        -- Time-based features
        MIN(event_timestamp) AS first_activity,
        MAX(event_timestamp) AS last_activity,
        JULIANDAY(MAX(event_timestamp)) - JULIANDAY(MIN(event_timestamp)) AS active_days,

        -- Category preferences
        MAX(CASE WHEN category = 'electronics' THEN event_count ELSE 0 END)
            AS electronics_views,
        MAX(CASE WHEN category = 'clothing' THEN event_count ELSE 0 END)
            AS clothing_views,

        -- Purchase behavior
        SUM(CASE WHEN event_type = 'purchase' THEN amount ELSE 0 END) AS total_spend,
        AVG(CASE WHEN event_type = 'purchase' THEN amount END) AS avg_order_value,

        -- Recency features (days since last activity)
        CAST(JULIANDAY('now') - JULIANDAY(MAX(event_timestamp)) AS INTEGER)
            AS days_since_last_activity,

        -- Frequency (events per active day)
        ROUND(COUNT(*) * 1.0 / NULLIF(
            CAST(JULIANDAY(MAX(event_timestamp)) - JULIANDAY(MIN(event_timestamp)) AS INTEGER),
            0
        ), 2) AS daily_event_rate
    FROM user_events
    GROUP BY user_id
)
SELECT * FROM user_features
WHERE total_events > 5  -- Filter cold-start users
ORDER BY total_spend DESC;
```

### Mathematical Connections

SQL aggregation functions map directly to statistical formulas used in ML:

| SQL Function | Mathematical Formula | ML Application |
|-------------|---------------------|----------------|
| `AVG(x)` | μ = (1/n) Σ xᵢ | Mean imputation |
| `VAR(x)` | σ² = (1/n) Σ(xᵢ - μ)² | Feature scaling |
| `STDDEV(x)` | σ = √(σ²) | Z-score normalization |
| `CORR(x, y)` | ρ = Cov(x,y) / (σₓσᵧ) | Feature selection |
| `COALESCE` | x ≠ null ? x : default | Missing value handling |
| `NTILE(4)` | Quartile bucketing | Binning features |

### SQL for Model Evaluation

```sql
-- Calculate model evaluation metrics from prediction tables
SELECT
    -- Classification metrics
    COUNT(*) AS total_samples,
    SUM(CASE WHEN predicted = actual THEN 1 ELSE 0 END) AS correct,
    ROUND(SUM(CASE WHEN predicted = actual THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2)
        AS accuracy_pct,

    -- Per-class metrics
    SUM(CASE WHEN actual = 1 AND predicted = 1 THEN 1 ELSE 0 END) AS true_positives,
    SUM(CASE WHEN actual = 0 AND predicted = 1 THEN 1 ELSE 0 END) AS false_positives,
    SUM(CASE WHEN actual = 1 AND predicted = 0 THEN 1 ELSE 0 END) AS false_negatives,

    ROUND(
        SUM(CASE WHEN actual = 1 AND predicted = 1 THEN 1.0 ELSE 0 END) /
        NULLIF(
            SUM(CASE WHEN actual = 1 AND predicted = 1 THEN 1 ELSE 0 END) +
            SUM(CASE WHEN actual = 0 AND predicted = 1 THEN 1 ELSE 0 END),
            0
        ), 4
    ) AS precision,

    ROUND(
        SUM(CASE WHEN actual = 1 AND predicted = 1 THEN 1.0 ELSE 0 END) /
        NULLIF(
            SUM(CASE WHEN actual = 1 AND predicted = 1 THEN 1 ELSE 0 END) +
            SUM(CASE WHEN actual = 1 AND predicted = 0 THEN 1 ELSE 0 END),
            0
        ), 4
    ) AS recall
FROM model_predictions;
```

### Quick Reference: Essential SQL for Data Scientists

```sql
-- Top-N with ties
SELECT * FROM (
    SELECT *, RANK() OVER (ORDER BY score DESC) AS rnk FROM results
) WHERE rnk <= 10;

-- Percentage of total
SELECT
    category,
    COUNT(*) AS cnt,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) AS pct
FROM items
GROUP BY category;

-- Cohort analysis (first purchase month)
SELECT
    DATE_TRUNC('month', first_order_date) AS cohort,
    COUNT(DISTINCT user_id) AS cohort_size,
    COUNT(DISTINCT CASE
        WHEN DATE_TRUNC('month', order_date) =
             DATE_TRUNC('month', first_order_date, '+1 month')
        THEN user_id END
    ) / COUNT(DISTINCT user_id)::FLOAT AS m1_retention
FROM user_orders
GROUP BY DATE_TRUNC('month', first_order_date);
```

---

## Summary

This guide covered SQL fundamentals through advanced patterns and their integration with Python. Key takeaways:

1. **SQL execution order** matters — understand it to write correct queries
2. **Window functions** are powerful for time-series and ranking analysis
3. **Indexing strategy** significantly impacts query performance
4. **Python integration** via sqlite3, SQLAlchemy, and Pandas enables end-to-end workflows
5. **Feature engineering** with SQL is critical for ML pipeline efficiency
