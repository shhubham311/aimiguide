# Relational vs NoSQL Databases (MongoDB)

## Why Databases Matter for ML/AI

Machine learning doesn't start with modeling—it starts with **data storage and retrieval**. Your training data, user interactions, model predictions, feature stores, experiment logs, and deployment metadata all live in databases. Choosing the right database determines query performance, scalability, and data consistency.

In AI/ML specifically:
- **Feature stores** need fast reads for real-time inference
- **Experiment tracking** (MLflow, Weights & Biases) uses databases to log metrics
- **User interaction data** (for recommendation systems) requires flexible schemas
- **Model metadata** (version, metrics, deployment status) needs structured storage

This lesson covers relational (SQL) vs. NoSQL databases with practical Python implementations.

---

## Relational Databases (SQL)

### Core Concepts

```
┌─────────────────────────────────────────────────────┐
│              Relational Database                      │
│                                                      │
│  ┌──────────────┐    ┌──────────────────┐           │
│  │   users      │    │   predictions     │           │
│  ├──────────────┤    ├──────────────────┤           │
│  │ id (PK)     │◄───│ user_id (FK)     │           │
│  │ name        │    │ model_id (FK)    │──► models  │
│  │ email       │    │ prediction       │           │
│  │ created_at  │    │ confidence       │           │
│  └──────────────┘    │ timestamp        │           │
│                      └──────────────────┘           │
│                                                      │
│  PK = Primary Key (unique identifier)                │
│  FK = Foreign Key (reference to another table)       │
│  Relations enforced by schema                        │
└─────────────────────────────────────────────────────┘
```

### SQL Basics

```sql
-- Create tables
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE model_predictions (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    model_name VARCHAR(50),
    input_text TEXT,
    prediction TEXT,
    confidence FLOAT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Insert data
INSERT INTO users (name, email) VALUES
    ('Alice', 'alice@example.com'),
    ('Bob', 'bob@example.com');

-- Query data
SELECT u.name, m.prediction, m.confidence
FROM model_predictions m
JOIN users u ON m.user_id = u.id
WHERE m.confidence > 0.8
ORDER BY m.created_at DESC
LIMIT 10;

-- Aggregation
SELECT model_name,
       COUNT(*) as total_predictions,
       AVG(confidence) as avg_confidence
FROM model_predictions
GROUP BY model_name;

-- Update
UPDATE model_predictions
SET confidence = 0.95
WHERE id = 42;
```

### ACID Properties

| Property | Meaning | Example |
|----------|---------|---------|
| **Atomicity** | Transaction is all-or-nothing | Either debit AND credit happen, or neither |
| **Consistency** | Database moves between valid states | Balance never goes negative |
| **Isolation** | Concurrent transactions don't interfere | Two users can't book the same seat |
| **Durability** | Committed data survives crashes | Data persists after power outage |

---

## NoSQL Databases

### Types of NoSQL

```
┌─────────────────────────────────────────────────────────┐
│                  NoSQL Database Types                     │
│                                                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐      │
│  │ Document    │  │ Key-Value   │  │ Column      │      │
│  │ (MongoDB,   │  │ (Redis,     │  │ (Cassandra, │      │
│  │  CouchDB)   │  │  DynamoDB)  │  │  HBase)     │      │
│  │             │  │             │  │             │      │
│  │ JSON-like   │  │ Simple      │  │ Wide rows   │      │
│  │ documents   │  │ key→value   │  │ for analytics│      │
│  └─────────────┘  └─────────────┘  └─────────────┘      │
│                                                           │
│  ┌─────────────┐                                         │
│  │ Graph       │                                         │
│  │ (Neo4j)     │                                         │
│  │             │                                         │
│  │ Social      │                                         │
│  │ connections │                                         │
│  └─────────────┘                                         │
└─────────────────────────────────────────────────────────┘
```

### MongoDB: Document Database

MongoDB stores data as **JSON-like documents** in flexible, schema-less collections:

```json
// A user document (users collection)
{
    "_id": ObjectId("507f1f77bcf86cd799439011"),
    "name": "Alice Chen",
    "email": "alice@example.com",
    "preferences": {
        "language": "en",
        "theme": "dark",
        "notifications": true
    },
    "ml_models_used": ["sentiment-v2", "ner-v3", "classifier-v1"],
    "interactions": [
        {"text": "Great product!", "prediction": "positive", "ts": "2024-01-15"},
        {"text": "Terrible service", "prediction": "negative", "ts": "2024-01-16"}
    ],
    "created_at": ISODate("2024-01-01T00:00:00Z")
}
```

### Python with MongoDB

```python
from pymongo import MongoClient
from datetime import datetime
import json

# Connect
client = MongoClient("mongodb://localhost:27017/")
db = client["ml_platform"]
users = db["users"]
predictions = db["predictions"]

# Insert a document
user_doc = {
    "name": "Bob Smith",
    "email": "bob@example.com",
    "preferences": {"language": "en", "theme": "light"},
    "created_at": datetime.utcnow()
}
result = users.insert_one(user_doc)
print(f"Inserted ID: {result.inserted_id}")

# Insert many
users.insert_many([
    {"name": "Charlie", "email": "charlie@example.com"},
    {"name": "Diana", "email": "diana@example.com"}
])

# Query (find)
user = users.find_one({"email": "bob@example.com"})
all_users = list(users.find({"name": {"$regex": "^[A-D]"}}))

# Operators: $eq, $ne, $gt, $gte, $lt, $lte, $in, $regex, $exists, $and, $or
high_confidence = list(predictions.find({
    "confidence": {"$gte": 0.9},
    "model_name": {"$in": ["bert-base", "roberta-large"]},
    "$or": [
        {"prediction": "positive"},
        {"prediction": "negative"}
    ]
}).sort("created_at", -1).limit(10))

# Update
predictions.update_one(
    {"_id": some_id},
    {"$set": {"reviewed": True, "reviewed_by": "admin"}},
    upsert=False
)

# Aggregation pipeline
pipeline = [
    {"$match": {"confidence": {"$gte": 0.8}}},
    {"$group": {
        "_id": "$model_name",
        "count": {"$sum": 1},
        "avg_confidence": {"$avg": "$confidence"}
    }},
    {"$sort": {"count": -1}}
]
results = list(predictions.aggregate(pipeline))
```

---

## SQL vs NoSQL: When to Use What

```
┌────────────────────────────────────────────────────────────┐
│              Decision Framework                             │
│                                                             │
│  Use SQL (PostgreSQL/MySQL) when:                           │
│  ✓ Data has clear structure and relationships              │
│  ✓ Complex queries with JOINs                              │
│  ✓ ACID compliance is critical (finance, healthcare)       │
│  ✓ Schema is stable and won't change frequently            │
│                                                             │
│  Use NoSQL (MongoDB) when:                                  │
│  ✓ Schema is flexible or evolving (ML experiments)         │
│  ✓ Nested/hierarchical data (JSON features, configs)       │
│  ✓ Horizontal scaling is needed                            │
│  ✓ Rapid prototyping with changing requirements            │
│                                                             │
│  Use Redis when:                                           │
│  ✓ Caching (feature store, session store)                  │
│  ✓ Real-time leaderboards                                   │
│  ✓ Pub/Sub messaging                                       │
└────────────────────────────────────────────────────────────┘
```

| Feature | SQL | NoSQL (MongoDB) |
|---------|-----|-------------------|
| Schema | Fixed, enforced | Flexible, dynamic |
| Scaling | Vertical (primarily) | Horizontal (sharding) |
| Relationships | JOINs, foreign keys | Embedded documents, references |
| Query Language | SQL | MQL (MongoDB Query Language) |
| ACID | Full ACID | Single-document ACID |
| Best For | Structured data, reporting | Flexible data, rapid iteration |

---

## Database Design for ML Systems

### Feature Store Schema (PostgreSQL)

```sql
-- Feature store: structured, relational
CREATE TABLE features (
    id SERIAL PRIMARY KEY,
    entity_id VARCHAR(100) NOT NULL,
    feature_name VARCHAR(100) NOT NULL,
    feature_value FLOAT,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    UNIQUE (entity_id, feature_name)
);

CREATE INDEX idx_features_entity ON features(entity_id);
CREATE INDEX idx_features_name ON features(feature_name);

-- Query for inference: fetch all features for an entity
SELECT feature_name, feature_value
FROM features
WHERE entity_id = 'user_12345';
```

### Experiment Tracking Schema (MongoDB)

```python
# Flexible experiment logs
experiment = {
    "experiment_id": "exp_2024_001",
    "model": "bert-base-uncased",
    "hyperparameters": {
        "learning_rate": 2e-5,
        "batch_size": 32,
        "epochs": 3,
        "warmup_steps": 500,
        "weight_decay": 0.01
    },
    "metrics": {
        "train_loss": [0.65, 0.32, 0.18],
        "val_accuracy": 0.85,
        "f1_score": 0.87,
        "perplexity": 1.23
    },
    "dataset": {
        "name": "imdb_reviews",
        "train_size": 25000,
        "val_size": 5000
    },
    "git_commit": "a1b2c3d",
    "created_at": datetime.utcnow()
}
db.experiments.insert_one(experiment)
```

---

## ORM: SQLAlchemy for Python

```python
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime

Base = declarative_base()
engine = create_engine("postgresql://user:pass@localhost/ml_db")
Session = sessionmaker(bind=engine)

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    email = Column(String(255), unique=True, nullable=False)

    predictions = relationship("Prediction", back_populates="user")

class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    model_name = Column(String(50))
    prediction = Column(String(100))
    confidence = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="predictions")

Base.metadata.create_all(engine)

# CRUD operations
session = Session()
user = User(name="Alice", email="alice@example.com")
session.add(user)
session.commit()

alice_predictions = session.query(Prediction).join(User).filter(
    User.name == "Alice"
).all()
```

---

## Exercises

### Exercise 1: SQL Query
Write a SQL query to find the top 3 models with highest average confidence in predictions made in the last 7 days.

**Solution:**
```sql
SELECT model_name,
       COUNT(*) as num_predictions,
       AVG(confidence) as avg_confidence
FROM model_predictions
WHERE created_at >= NOW() - INTERVAL '7 days'
GROUP BY model_name
ORDER BY avg_confidence DESC
LIMIT 3;
```

### Exercise 2: MongoDB Aggregation
Write a MongoDB aggregation pipeline to find users who had more than 5 negative predictions and show their average confidence.

**Solution:**
```python
pipeline = [
    {"$unwind": "$interactions"},
    {"$match": {"interactions.prediction": "negative"}},
    {"$group": {
        "_id": "$_id",
        "name": {"$first": "$name"},
        "neg_count": {"$sum": 1},
        "avg_conf": {"$avg": "$interactions.confidence"}
    }},
    {"$match": {"neg_count": {"$gt": 5}}},
    {"$sort": {"avg_conf": -1}}
]
results = list(db.users.aggregate(pipeline))
```

### Exercise 3: Design a Schema
Design a database schema for an ML experiment tracking system that stores experiments, runs, metrics, and artifacts.

**Solution (PostgreSQL):**
```sql
CREATE TABLE experiments (
    id SERIAL PRIMARY KEY,
    name VARCHAR(200) NOT NULL,
    description TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE runs (
    id SERIAL PRIMARY KEY,
    experiment_id INTEGER REFERENCES experiments(id),
    status VARCHAR(20) DEFAULT 'running',
    start_time TIMESTAMP,
    end_time TIMESTAMP,
    git_commit VARCHAR(40),
    config JSONB  -- Flexible hyperparameter storage
);

CREATE TABLE metrics (
    id SERIAL PRIMARY KEY,
    run_id INTEGER REFERENCES runs(id),
    metric_name VARCHAR(100),
    metric_value FLOAT,
    step INTEGER,
    timestamp TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_runs_experiment ON runs(experiment_id);
CREATE INDEX idx_metrics_run ON metrics(run_id);
```

---

## Key Takeaways

1. **SQL databases** enforce structure with schemas, ACID, and relationships—ideal for structured ML metadata
2. **NoSQL databases** (MongoDB) offer flexible schemas—ideal for experiment logs, feature configs, evolving data
3. **MongoDB** stores JSON-like documents with rich query operators and aggregation pipelines
4. **Redis** excels at caching for real-time feature serving and session management
5. **Choose based on** data structure, consistency needs, scaling requirements, and team expertise
