# Interview Patterns of Top AI Companies

## Overview

Top AI companies (Google, Meta, OpenAI, DeepMind, Netflix, etc.) follow distinct interview patterns. Understanding these patterns helps you prepare strategically.

## 1. Interview Structure

```
Typical AI Engineer Interview (4-5 rounds):
┌────────────────────────────────────────────┐
│ Round 1: Technical Screen (45-60 min)      │
│   - Coding + ML fundamentals               │
│   - LeetCode Easy-Medium + conceptual Qs   │
├────────────────────────────────────────────┤
│ Round 2: ML System Design (45-60 min)      │
│   - Design an end-to-end ML system         │
│   - Recommendations, search, ranking        │
│   - Scalability, tradeoffs, evaluation     │
├────────────────────────────────────────────┤
│ Round 3: Domain Deep Dive (45-60 min)      │
│   - Specific to the role (NLP, CV, RecSys) │
│   - Paper discussions, architecture         │
├────────────────────────────────────────────┤
│ Round 4: Behavioral + Culture Fit (30 min) │
│   - Past projects, challenges, impact      │
│   - How you handle ambiguity, failure      │
├────────────────────────────────────────────┤
│ Round 5: Onsite/Virtual Panel (optional)   │
│   - Whiteboard system design              │
│   - Multi-interviewer panel                │
└────────────────────────────────────────────┘
```

## 2. Question Categories by Company

```
┌──────────────┬──────────────────────────────────────────────┐
│ Google       │ ML fundamentals, system design, coding,      │
│              │ scalability, TensorFlow/PyTorch knowledge     │
├──────────────┼──────────────────────────────────────────────┤
│ Meta         │ Recommendation systems, ranking, A/B testing, │
│              │ distributed systems, feature engineering     │
├──────────────┼──────────────────────────────────────────────┤
│ OpenAI       │ Deep learning fundamentals, transformer      │
│              │ architecture, training infrastructure,        │
│              │ evaluation metrics, research thinking         │
├──────────────┼──────────────────────────────────────────────┤
│ Netflix      │ Recommendation algorithms, A/B testing,       │
│              │ data pipelines, causal inference              │
├──────────────┼──────────────────────────────────────────────┤
│ Amazon       │ ML system design, scalability, cost          │
│              │ optimization, deployment, edge cases          │
├──────────────┼──────────────────────────────────────────────┤
│ DeepMind     │ Research depth, math, RL, multi-agent,       │
│              │ latest papers, originality of thinking        │
└──────────────┴──────────────────────────────────────────────┘
```

## 3. High-Frequency Question Types

### ML Fundamentals
- Explain bias-variance tradeoff with examples
- How does regularization prevent overfitting?
- Why does gradient descent work? When does it fail?
- Difference between generative and discriminative models
- What is the expectation-maximization algorithm?

### System Design
- Design a recommendation system for Netflix
- Design a search ranking system
- Design a spam detection pipeline
- Design an A/B testing framework
- How would you deploy a model serving 1M QPS?

### Coding + ML
- Implement KNN from scratch
- Build a decision tree
- Implement gradient descent for logistic regression
- Design a Bloom filter for duplicate detection
- Implement a basic neural network forward pass

### Mathematics
- Derive the gradient of the cross-entropy loss
- Explain PCA from scratch
- Derive backpropagation for a 2-layer network
- What is the Jacobian and why does it matter?
- Explain the kernel trick geometrically

## 4. The ML System Design Framework

Use this 5-step framework for any ML design question:

```
Step 1: PROBLEM FORMULATION
  - What are we predicting?
  - Supervised/unsupervised/reinforcement?
  - Input/output format?
  - Success metrics?

Step 2: DATA PIPELINE
  - Where does data come from?
  - How to collect, store, process?
  - Feature engineering strategy
  - Handling data drift

Step 3: MODEL SELECTION
  - Baseline approach first
  - Compare multiple algorithms
  - Complexity vs performance tradeoff
  - Online vs batch learning?

Step 4: TRAINING & EVALUATION
  - Train/validation/test split
  - Offline metrics (precision, recall, AUC)
  - Online metrics (CTR, engagement, revenue)
  - A/B testing plan

Step 5: DEPLOYMENT & MONITORING
  - Model serving architecture
  - Latency and throughput requirements
  - Monitoring: data drift, model degradation
  - Retraining strategy
```

## 5. Example: Design a Recommendation System

```
INTERVIEWER: Design YouTube's video recommendation system.

YOUR RESPONSE (using the framework):

1. PROBLEM: Recommend personalized videos to users
   - Input: User history, context, video features
   - Output: Ranked list of videos
   - Metric: Watch time, engagement, satisfaction

2. DATA: 
   - User watch history, likes, shares
   - Video metadata (title, tags, duration, category)
   - Context (time of day, device, location)
   - Implicit signals (scroll, hover, completion rate)

3. MODEL:
   - Candidate Generation: Collaborative filtering + content-based
   - Ranking: Deep neural network (two-tower model)
   - Re-ranking: Business rules, diversity, freshness

4. TRAINING:
   - Loss: Pairwise ranking (cross-entropy)
   - Features: User embeddings, video embeddings, cross features
   - Negative sampling for efficiency

5. DEPLOYMENT:
   - Online serving: Pre-computed embeddings + approximate NN search
   - Latency budget: < 200ms
   - A/B test new model against production
   - Monitor: prediction distribution, feature coverage
```

## 6. How to Prepare

```
Week 1-2: ML Fundamentals
  - Review all math (linear algebra, probability, calculus)
  - Re-derive key algorithms from scratch
  - Practice explaining concepts simply

Week 3-4: Coding Practice
  - LeetCode: 2-3 problems daily
  - Focus on: arrays, trees, graphs, DP, heaps
  - Implement ML algorithms from scratch

Week 5-6: System Design
  - Practice ML system design problems
  - Read engineering blogs (Netflix, Uber, Airbnb)
  - Design end-to-end systems

Week 7-8: Mock Interviews
  - Practice with peers or platforms
  - Time yourself
  - Get feedback and iterate

Resources:
  - "Machine Learning Interviews" by Ali Aminian
  - "Designing ML Systems" by Chip Huyen
  - LeetCode, AlgoExpert, Educative
  - Company engineering blogs
```

## 🧠 Exercise

Design a real-time fraud detection system for a payment platform. Cover: problem formulation, data pipeline, model selection, evaluation metrics, deployment architecture, and monitoring strategy.

<details>
<summary>Key points to cover</summary>

```
Problem: Binary classification (fraud / not fraud) in < 100ms
Data: Transaction history, user behavior, device fingerprint, geolocation
Model: Gradient boosted trees (fast inference) + neural network for complex patterns
Features: Transaction amount, velocity, merchant category, time patterns, device anomalies
Evaluation: Precision (minimize false positives!), recall, F2 score, latency
Deployment: Stream processing (Kafka), online model serving, cascade of models
Monitoring: Real-time dashboards, model drift detection, human review queue for edge cases
```

</details>
