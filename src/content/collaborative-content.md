# Content-Based & Collaborative Filtering

## Why Recommendation Systems Matter

Recommendation systems are everywhere—Netflix suggests movies, Amazon suggests products, Spotify suggests songs, YouTube suggests videos. They drive **35% of Amazon's revenue**, **80% of Netflix watched content**, and are critical to user engagement across all platforms. Understanding recommendation fundamentals is essential because they combine collaborative filtering, content analysis, and increasingly, deep learning.

---

## The Recommendation Problem

```
┌─────────────────────────────────────────────────────────┐
│              The Recommendation Problem                  │
│                                                          │
│  Given:                                                  │
│  • User u (with past interactions)                       │
│  • Item inventory I = {i₁, i₂, ..., iₙ}                 │
│  • Context c (time, device, location)                    │
│                                                          │
│  Find:                                                    │
│  • Ranking of items: score(i, u, c) for each item i     │
│  • Top-K items: argmax_k score(i, u, c)                 │
│                                                          │
│  Core challenge: The interaction matrix is VERY sparse   │
│  (99%+ of user-item pairs have no interaction)           │
└─────────────────────────────────────────────────────────┘
```

### The User-Item Matrix

```
              Item1  Item2  Item3  Item4  Item5  Item6
User1         5.0    ?      3.0    ?      4.0    ?
User2         ?      4.0    ?      2.0    ?      5.0
User3         3.0    5.0    4.0    ?      ?      ?
User4         ?      ?      ?      5.0    2.0    ?
User5         4.0    ?      5.0    ?      3.0    4.0
User6         ?      3.0    ?      4.0    ?      ?

? = Missing (unobserved) — need to predict these!
Sparsity: 9 missing / 30 total = 70% sparse (real systems: 95-99.9% sparse)
```

---

## Content-Based Filtering

### Core Idea: Recommend items **similar to what the user liked before**

```
User Profile ←── "liked" ─── Item Features
     │
     ▼
  Compute similarity
     │
     ▼
  Recommend similar items

Example:
  User watched: "The Matrix" (Action, Sci-Fi, Keanu Reeves)
  → Profile: high weights for Action, Sci-Fi
  → Recommend: "Inception", "Blade Runner", "Interstellar"
```

### Content Features

```
┌──────────────────────────────────────────────────┐
│  Item Feature Types                               │
│                                                    │
│  Text:     Title, description, tags               │
│  Category: Genre, topic, type                     │
│  Metadata: Director, author, year, duration       │
│  Visual:   Cover image, thumbnail (CNN features)  │
│  Audio:    Audio features (for music rec)          │
│  Graph:    Co-occurrence, entity relationships    │
└──────────────────────────────────────────────────┘
```

### TF-IDF Content-Based Recommendation

```python
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Movie descriptions
movies = {
    "The Matrix": "A computer hacker discovers reality is a simulation",
    "Inception": "A thief steals corporate secrets through dream sharing",
    "Toy Story": "Toys come to life when humans are not around",
    "Interstellar": "Astronauts travel through a wormhole to save humanity",
    "Finding Nemo": "A clownfish searches for his missing son",
    "Blade Runner": "A cop hunts rogue AI replicants in dystopian LA",
}

# Build TF-IDF matrix
titles = list(movies.keys())
descriptions = list(movies.values())
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(descriptions)

# Compute similarity between all movies
similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Recommend similar movies to "The Matrix"
def recommend(movie_title, similarity_matrix, titles, top_k=3):
    idx = titles.index(movie_title)
    scores = similarity_matrix[idx]
    # Sort by similarity (exclude the movie itself)
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    recommendations = [(titles[i], scores[i]) for i in ranked[1:top_k+1]]
    return recommendations

print(recommend("The Matrix", similarity_matrix, titles))
# [('Interstellar', 0.15), ('Blade Runner', 0.12), ('Inception', 0.08)]
```

---

## Collaborative Filtering

### Core Idea: Recommend items that **similar users liked**

```
┌───────────────────────────────────────────────────┐
│  User-based CF:                                   │
│  "Users like you also liked..."                    │
│                                                    │
│  Alice: liked A, B, C                             │
│  Bob:   liked A, B, D                             │
│  → Alice and Bob are similar (both liked A, B)    │
│  → Recommend D to Alice (Bob liked it)            │
│  → Recommend C to Bob (Alice liked it)            │
└───────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────┐
│  Item-based CF:                                   │
│  "Because you liked A, you might like B"          │
│                                                    │
│  Users who liked A also liked B → A and B similar │
│  If user liked A → recommend B                    │
└───────────────────────────────────────────────────┘
```

### User-Based Collaborative Filtering

```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# User-item rating matrix (5 users × 6 movies)
R = np.array([
    [5, 0, 3, 0, 4, 0],   # User1
    [0, 4, 0, 2, 0, 5],   # User2
    [3, 5, 4, 0, 0, 0],   # User3
    [0, 0, 0, 5, 2, 0],   # User4
    [4, 0, 5, 0, 3, 4],   # User5
])

# Compute user similarity (cosine similarity between user vectors)
user_similarity = cosine_similarity(R)

def predict_rating(user_idx, item_idx, R, user_sim):
    """Predict rating using user-based CF."""
    # Find users who rated this item
    rated_by = np.where(R[:, item_idx] > 0)[0]

    if len(rated_by) == 0:
        return 0.0

    # Weighted average of ratings from similar users
    similarities = user_sim[user_idx, rated_by]
    ratings = R[rated_by, item_idx]

    # Only consider positive similarity
    mask = similarities > 0
    if mask.sum() == 0:
        return 0.0

    prediction = np.sum(similarities[mask] * ratings[mask]) / np.sum(similarities[mask])
    return prediction

# Predict what User1 would rate Item2
pred = predict_rating(0, 1, R, user_similarity)
print(f"Predicted rating for User1 → Item2: {pred:.2f}")
```

### Item-Based Collaborative Filtering

```python
# Compute item similarity (cosine similarity between item columns)
item_similarity = cosine_similarity(R.T)

def recommend_items(user_idx, R, item_sim, top_k=3):
    """Recommend items to a user based on item-based CF."""
    # Items the user hasn't rated
    unrated = np.where(R[user_idx] == 0)[0]

    # Score each unrated item
    scores = []
    for item_idx in unrated:
        # Items user has rated
        rated_items = np.where(R[user_idx] > 0)[0]
        # Similarity between unrated item and rated items
        sim = item_sim[item_idx, rated_items]
        # Weighted sum of user's ratings
        pred = np.sum(sim * R[user_idx, rated_items]) / (np.sum(np.abs(sim)) + 1e-8)
        scores.append((item_idx, pred))

    # Sort by predicted score
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]

print("Recommendations for User1:", recommend_items(0, R, item_similarity))
```

---

## Matrix Factorization: The Mathematical Foundation

### Core Idea: Decompose the sparse rating matrix into dense user and item embeddings.

```
┌─────────────────────────────────────────────────────────┐
│  Matrix Factorization                                    │
│                                                          │
│  R (m × n)     ≈     P (m × k)   ×   Q (k × n)         │
│                                                          │
│  User-Item         User         Item                    │
│  Ratings           Embeddings    Embeddings              │
│  (sparse)          (dense)       (dense)                 │
│                                                          │
│  R ≈ P × Q^T                                            │
│  r̂(u,i) = p_u · q_i (dot product of embeddings)        │
│                                                          │
│  ┌─────┐     ┌─────┐   ┌─────┐                         │
│  │5  ? 3│     │0.9  │   │0.8  0.2 0.7│                 │
│  │? 4 ?│  ≈  │0.3  │ × │0.1  0.9 0.3│                 │
│  │3 5 4│     │0.7  │   │0.6  0.5 0.8│                 │
│  └─────┘     └─────┘   └─────────────┘                 │
│                                                          │
│  k = number of latent factors (typically 20-200)        │
└─────────────────────────────────────────────────────────┘
```

### The Objective Function

```
min_{P,Q}  Σ_{(u,i) ∈ observed} (r_{ui} - p_u^T q_i)² + λ(||p_u||² + ||q_i||²)

Where:
  r_{ui}     = Observed rating of user u for item i
  p_u        = User u's embedding vector (k-dim)
  q_i        = Item i's embedding vector (k-dim)
  λ          = Regularization parameter (prevents overfitting)
  p_u^T q_i  = Predicted rating (dot product)
```

### SGD Optimization

```python
import numpy as np

class MatrixFactorization:
    def __init__(self, n_users, n_items, n_factors=20, lr=0.01, reg=0.02):
        self.n_factors = n_factors
        self.lr = lr
        self.reg = reg

        # Initialize embeddings randomly (small values)
        np.random.seed(42)
        self.P = np.random.normal(0, 0.1, (n_users, n_factors))  # User embeddings
        self.Q = np.random.normal(0, 0.1, (n_items, n_factors))  # Item embeddings

    def fit(self, R, n_epochs=100):
        """Train using SGD on observed ratings."""
        users, items = np.where(R > 0)  # Get observed ratings

        for epoch in range(n_epochs):
            total_loss = 0

            for idx in range(len(users)):
                u, i = users[idx], items[idx]
                r_ui = R[u, i]

                # Predict
                pred = np.dot(self.P[u], self.Q[i])

                # Compute error
                error = r_ui - pred
                total_loss += error ** 2

                # Update embeddings (SGD with L2 regularization)
                self.P[u] += self.lr * (error * self.Q[i] - self.reg * self.P[u])
                self.Q[i] += self.lr * (error * self.P[u] - self.reg * self.Q[i])

            if (epoch + 1) % 20 == 0:
                rmse = np.sqrt(total_loss / len(users))
                print(f"Epoch {epoch+1}/{n_epochs}, RMSE: {rmse:.4f}")

    def predict(self, u, i):
        """Predict rating for user u, item i."""
        return np.dot(self.P[u], self.Q[i])

    def recommend(self, user_idx, n_items, top_k=3):
        """Top-K recommendations for a user."""
        scores = self.P[user_idx] @ self.Q.T  # (n_items,)
        # Exclude already rated items
        rated = np.where(R[user_idx] > 0)[0]
        scores[rated] = -np.inf
        top_k_idx = np.argsort(scores)[-top_k:][::-1]
        return list(zip(top_k_idx, scores[top_k_idx]))


# Usage
R = np.array([
    [5, 0, 3, 0, 4, 0],
    [0, 4, 0, 2, 0, 5],
    [3, 5, 4, 0, 0, 0],
    [0, 0, 0, 5, 2, 0],
    [4, 0, 5, 0, 3, 4],
])

model = MatrixFactorization(n_users=5, n_items=6, n_factors=10)
model.fit(R, n_epochs=200)
print("Recommendations for User0:", model.recommend(0, 6, top_k=2))
```

---

## Content-Based vs Collaborative: Comparison

```
┌──────────────────┬──────────────────┬──────────────────┐
│ Aspect           │ Content-Based    │ Collaborative    │
├──────────────────┼──────────────────┼──────────────────┤
│ Needs item       │ Yes (features)   │ No               │
│ features?        │                  │                  │
│ Cold start       │ Handles new      │ Struggles with   │
│ (new items)      │ items well       │ new items        │
│ Cold start       │ Struggles with   │ Handles new      │
│ (new users)      │ new users        │ users (slowly)   │
│ Diversity        │ Low (similar     │ Higher (exposes   │
│                  │ items)           │ to new things)   │
│ Serendipity      │ Low              │ Higher           │
│ Data sparsity    │ Handles well     │ Struggles        │
│ Popularity bias  │ None             │ Strong           │
└──────────────────┴──────────────────┴──────────────────┘
```

**Best practice: Hybrid approaches** combine both to get the strengths of each.

---

## Exercises

### Exercise 1: Compute User Similarity
Given user vectors u₁ = [5, 0, 3, 0, 4] and u₂ = [0, 4, 0, 2, 0], compute cosine similarity.

**Solution:**
```
cosine(u₁, u₂) = (u₁ · u₂) / (||u₁|| × ||u₂||)
u₁ · u₂ = 5×0 + 0×4 + 3×0 + 0×2 + 4×0 = 0
||u₁|| = √(25+0+9+0+16) = √50 = 7.07
||u₂|| = √(0+16+0+4+0) = √20 = 4.47

cosine = 0 / (7.07 × 4.47) = 0.0

No overlap in rated items → zero similarity!
This is a limitation of cosine similarity on sparse vectors.
```

### Exercise 2: Matrix Factorization Prediction
Given P[user1] = [0.3, 0.8] and Q[item2] = [0.5, 0.6], predict the rating.

**Solution:**
```
r̂ = P[user1] · Q[item2] = 0.3×0.5 + 0.8×0.6 = 0.15 + 0.48 = 0.63

This is a raw score; typically scaled to [1,5] range via:
r̂_scaled = min_rating + r̂ × (max_rating - min_rating)
```

### Exercise 3: Design a Hybrid System
How would you combine content-based and collaborative filtering for a movie platform?

**Solution:**
```
Hybrid Score = α × CF_score(u, i) + (1-α) × CB_score(u, i)

Where:
  CF_score = Matrix factorization predicted rating
  CB_score = Cosine similarity between user profile and item features
  α = Tunable weight (e.g., 0.7 for CF, 0.3 for CB)

Implementation:
  1. Build user embedding from MF (captures collaborative patterns)
  2. Build item features from movie metadata (genre, director, etc.)
  3. Compute CF score from dot product of user/item embeddings
  4. Compute CB score from TF-IDF or embedding similarity
  5. Combine with weighted sum
  6. Learn α via offline A/B testing
```

---

## Key Takeaways

1. **Content-based filtering** uses item features to recommend similar items—works for cold start on items
2. **Collaborative filtering** uses user behavior patterns—captures serendipity but struggles with cold start
3. **Matrix factorization** decomposes the rating matrix into user/item embeddings—compact and powerful
4. **Cosine similarity** is the standard similarity metric for both user-based and item-based CF
5. **Hybrid approaches** combine both methods for the best of both worlds
