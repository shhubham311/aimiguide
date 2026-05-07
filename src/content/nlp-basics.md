# NLP Basics: Text Processing, Feature Extraction, and Classification

## 1. Why This Topic Matters

Natural Language Processing (NLP) enables machines to understand, interpret, and generate human language. It powers search engines, sentiment analysis, chatbots, machine translation, spam detection, document summarization, and large language models like GPT and BERT. NLP is one of the fastest-growing fields in AI, and mastering the fundamentals — tokenization, vectorization, and text classification — is essential for any ML practitioner working with text data.

---

## 2. The NLP Pipeline

Every NLP system follows a general pipeline from raw text to predictions:

```
Raw Text → Preprocessing → Feature Extraction → Model → Prediction
   │              │                │              │
   │         ┌────┴────┐     ┌────┴────┐    ┌────┴────┐
   │         │ Clean   │     │ BoW/TF-IDF│  │ Naive Bayes│
   │         │ Tokenize│     │ Embeddings │  │ SVM/LogReg│
   │         │ Normalize│    │ Word2Vec  │  │ Neural Nets│
   │         │ Stopwords│     │ n-grams   │  │ Transformers│
   └─────────┴─────────┘     └───────────┘  └────────────┘
```

---

## 3. Text Preprocessing

### 3.1 Tokenization

Splitting text into individual units (tokens):

```
Original:  "I can't believe it's not butter!"

Tokens:    ["I", "can", "'t", "believe", "it", "'s", "not", "butter", "!"]

Word tokens:  ["I", "can't", "believe", "it's", "not", "butter", "!"]
Sentence tokens: ["I can't believe it's not butter!"]
```

### 3.2 Normalization

```
Lowercasing:     "The Quick BROWN Fox" → "the quick brown fox"
Stopword removal: "the cat sat on the mat" → ["cat", "sat", "mat"]
Stemming:        "running" → "run", "better" → "better"
Lemmatization:   "running" → "run", "better" → "good"
```

**Stemming vs Lemmatization:**

| Word | Stemming (Porter) | Lemmatization |
|------|-------------------|---------------|
| running | runn | run |
| am | am | be |
| better | better | good |
| corpora | corpora | corpus |
| geese | gees | goose |

Stemming is faster but cruder. Lemmatization uses dictionaries and produces valid words.

### 3.3 Stopword Removal

Common words that carry little meaning:

```
English stopwords: a, an, the, is, are, was, were, in, on, at, to, for,
                   of, and, or, but, it, he, she, they, this, that, ...

"After removing stopwords:"
Original: "The movie was not good but the acting was great"
Filtered: ["movie", "good", "acting", "great"]
```

**Caveat:** In sentiment analysis, "not" is critical! Removing it changes meaning. Always consider the task before removing stopwords.

---

## 4. Feature Extraction: Representing Text as Numbers

ML models require numerical input. NLP converts text to vectors through several methods:

### 4.1 Bag of Words (BoW)

Count the frequency of each word in a document:

```
Document 1: "cat sat mat"
Document 2: "dog sat mat"
Document 3: "cat dog mat"

Vocabulary: {cat:0, sat:1, mat:2, dog:3}

BoW Matrix:
         cat  sat  mat  dog
  Doc 1: [  1,   1,   1,   0]
  Doc 2: [  0,   1,   1,   1]
  Doc 3: [  1,   0,   1,   1]
```

**Limitations:**
- Ignores word order ("dog bit man" = "man bit dog")
- High-dimensional (vocabulary can be 50,000+ words)
- Sparse (most entries are zero)
- Favors long documents

### 4.2 TF-IDF (Term Frequency-Inverse Document Frequency)

Weight words by their importance:

```
TF(t,d) = (count of t in d) / (total words in d)

IDF(t, D) = log(N / (1 + n_t))

where:
  N    = total number of documents
  n_t  = number of documents containing term t

TF-IDF(t, d, D) = TF(t,d) × IDF(t,D)
```

**Intuition:** Common words across all documents ("the", "is") get low TF-IDF. Rare, discriminative words get high TF-IDF.

```
Example:
  3 documents, word "cat" appears in 2 documents

  TF("cat", Doc1) = 1/3 = 0.33
  IDF("cat") = log(3 / (1+2)) = log(1) = 0
  TF-IDF("cat", Doc1) = 0.33 × 0 = 0  ← common word, low importance

  TF("sat", Doc1) = 1/3 = 0.33
  IDF("sat") = log(3 / (1+2)) = 0
  TF-IDF("sat", Doc1) = 0

  TF("cat", Doc2) = 0/3 = 0
  TF("dog", Doc2) = 1/3 = 0.33
  IDF("dog") = log(3 / (1+2)) = 0  ← also common

  If word "quantum" appears in only 1 of 1000 docs:
  IDF("quantum") = log(1000 / 2) ≈ 6.2  ← very high importance!
```

### 4.3 N-grams

Capture word sequences:

```
Unigrams (n=1): ["I", "love", "NLP"]
Bigrams  (n=2): ["I love", "love NLP"]
Trigrams (n=3): ["I love NLP"]

"not good" as bigram preserves negation:
  BoW: "not" → low weight; "good" → medium weight
  Bigram: "not good" → captures the negative sentiment
```

### 4.4 Word Embeddings (Word2Vec, GloVe)

Dense vector representations that capture semantic meaning:

```
Word2Vec intuition: "You shall know a word by the company it keeps"

  king  - man  + woman  ≈  queen
  paris - france + italy ≈  rome

Each word → vector of 100-300 real numbers

Word vector example (simplified 3D):
  king:    [0.50,  0.68,  0.11]   (royalty, male, ...)
  queen:   [0.49,  0.71,  0.09]   (royalty, female, ...)
  man:     [0.01,  0.65,  0.12]   (common, male, ...)
  woman:   [0.02,  0.69,  0.10]   (common, female, ...)
```

**Word2Vec Training (Skip-gram):**

```
Objective: maximize probability of context words given center word

L = Σ log P(w_context | w_center)

P(w_c | w_t) = exp(v_c · v_t) / Σ_w exp(v_w · v_t)

where:
  v_c = vector for context word
  v_t = vector for target (center) word
  ·   = dot product
```

---

## 5. Text Classification with ML

### 5.1 Naive Bayes Classifier

The go-to baseline for text classification. Based on Bayes' theorem with the "naive" assumption that features (words) are conditionally independent:

```
P(class | document) ∝ P(class) × Π P(word_i | class)

For each class c:
  P(c | d) = P(c) × Π P(w_i | c) / P(d)

Prediction: ĉ = argmax_c P(c) × Π P(w_i | c)

With TF-IDF features, use Multinomial Naive Bayes:
  P(w_i | c) = (count of w_i in class c + α) / (total words in class c + α × V)

where α = Laplace smoothing parameter (typically α = 1)
```

**Why Naive Bayes works well for text:**
- Extremely fast to train and predict (O(n × V))
- Handles high-dimensional sparse features naturally
- Works surprisingly well despite the independence assumption
- Requires very little data

### 5.2 Logistic Regression / SVM for Text

```
Logistic Regression:
  P(y=1 | x) = 1 / (1 + exp(-w^T · x + b))

  where x = TF-IDF vector, w = learned weights, b = bias

SVM (Linear):
  Find w, b that maximizes margin: y_i(w^T · x_i + b) ≥ 1

  For text, linear SVM with L2 regularization is very effective
```

---

## 6. Python Implementation

### 6.1 Text Preprocessing Pipeline

```python
import re
from sklearn.datasets import fetch_20newsgroups

# Load a real text dataset (subset for speed)
categories = ['alt.atheism', 'sci.space', 'comp.graphics', 'sci.med']
newsgroups = fetch_20newsgroups(subset='train', categories=categories, random_state=42)
documents = newsgroups.data
labels = newsgroups.target

print(f"Dataset: {len(documents)} documents, {len(categories)} categories")
print(f"Categories: {newsgroups.target_names}")
print(f"\nSample document (first 200 chars):")
print(documents[0][:200])

# Preprocessing function
import nltk
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess(text):
    """Complete preprocessing pipeline."""
    # Lowercase
    text = text.lower()
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    # Tokenize (split on whitespace)
    tokens = text.split()
    # Remove stopwords
    tokens = [t for t in tokens if t not in stop_words]
    # Stem
    tokens = [stemmer.stem(t) for t in tokens]
    # Remove very short tokens
    tokens = [t for t in tokens if len(t) > 2]
    return ' '.join(tokens)

# Preprocess all documents
processed_docs = [preprocess(doc) for doc in documents]
print(f"\nOriginal:  {documents[0][:100]}...")
print(f"Processed: {processed_docs[0][:100]}...")
```

### 6.2 BoW and TF-IDF Feature Extraction

```python
from sklearn.feature_extraction.text import (
    CountVectorizer, TfidfVectorizer
)

# ---- Bag of Words ----
bow_vectorizer = CountVectorizer(
    max_features=10000,      # Limit vocabulary size
    min_df=3,                # Ignore rare words (in < 3 docs)
    max_df=0.95,             # Ignore very common words (in > 95% docs)
    ngram_range=(1, 2)       # Unigrams + bigrams
)
X_bow = bow_vectorizer.fit_transform(processed_docs)
print(f"BoW shape: {X_bow.shape}")
print(f"Non-zero elements: {X_bow.nnz} / {X_bow.shape[0]*X_bow.shape[1]}")
print(f"Sparsity: {1 - X_bow.nnz / (X_bow.shape[0]*X_bow.shape[1]):.4f}")

# ---- TF-IDF ----
tfidf_vectorizer = TfidfVectorizer(
    max_features=10000,
    min_df=3,
    max_df=0.95,
    ngram_range=(1, 2),
    sublinear_tf=True        # Apply log(1+tf) scaling
)
X_tfidf = tfidf_vectorizer.fit_transform(processed_docs)
print(f"\nTF-IDF shape: {X_tfidf.shape}")

# Show top words by TF-IDF weight
feature_names = tfidf_vectorizer.get_feature_names_out()
for category_idx in range(len(categories)):
    category_docs = X_tfidf[labels == category_idx]
    mean_tfidf = category_docs.mean(axis=0).A1
    top_indices = mean_tfidf.argsort()[-10:][::-1]
    print(f"\n  {newsgroups.target_names[category_idx]} top words:")
    print(f"    {[feature_names[i] for i in top_indices[:5]]}")
```

### 6.3 Text Classification: Naive Bayes

```python
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix
)

X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, labels, test_size=0.2, random_state=42, stratify=labels
)

# Multinomial Naive Bayes
nb = MultinomialNB(alpha=0.1)  # Laplace smoothing
nb.fit(X_train, y_train)
y_pred_nb = nb.predict(X_test)

print(f"Naive Bayes Accuracy: {accuracy_score(y_test, y_pred_nb):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_nb,
      target_names=newsgroups.target_names))
```

### 6.4 Text Classification: Logistic Regression and SVM

```python
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

# Logistic Regression
lr = LogisticRegression(C=5.0, max_iter=1000, random_state=42)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
print(f"Logistic Regression Accuracy: {accuracy_score(y_test, y_pred_lr):.4f}")

# Linear SVM
svm = LinearSVC(C=1.0, max_iter=5000)
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)
print(f"Linear SVM Accuracy: {accuracy_score(y_test, y_pred_svm):.4f}")

# ---- Model Comparison ----
print("\n=== Model Comparison ===")
models = {
    'Naive Bayes': accuracy_score(y_test, y_pred_nb),
    'Logistic Reg': accuracy_score(y_test, y_pred_lr),
    'Linear SVM': accuracy_score(y_test, y_pred_svm)
}
for name, acc in sorted(models.items(), key=lambda x: -x[1]):
    print(f"  {name:15s}: {acc:.4f}")
# Logistic Regression and SVM typically achieve 90%+ on 4-class 20NG
```

### 6.5 Feature Importance: Most Discriminative Words

```python
# For Logistic Regression: weights indicate word importance
import numpy as np

for category_idx, category_name in enumerate(newsgroups.target_names):
    weight = lr.coef_[category_idx]
    top_pos = weight.argsort()[-10:][::-1]
    top_neg = weight.argsort()[:10]
    print(f"\n{category_name}:")
    print(f"  Top positive: {[feature_names[i] for i in top_pos[:5]]}")
    print(f"  Top negative: {[feature_names[i] for i in top_neg[:5]]}")
```

### 6.6 Word2Vec Embeddings

```python
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt', quiet=True)

# Train Word2Vec on the documents
tokenized_docs = [word_tokenize(doc.lower()) for doc in documents[:500]]

w2v_model = Word2Vec(
    sentences=tokenized_docs,
    vector_size=100,     # Dimensionality
    window=5,            # Context window size
    min_count=5,         # Ignore words with < 5 occurrences
    sg=1,                # Skip-gram (1) vs CBOW (0)
    epochs=20,
    workers=4
)

# Word similarity
if 'space' in w2v_model.wv:
    similar = w2v_model.wv.most_similar('space', topn=5)
    print("Words similar to 'space':")
    for word, score in similar:
        print(f"  {word:20s} {score:.4f}")

# Word analogy
try:
    result = w2v_model.wv.most_similar(
        positive=['space', 'earth'], negative=['planet'], topn=3
    )
    print(f"\nAnalogy: space - planet + earth ≈ {result[0][0]}")
except KeyError as e:
    print(f"Word not in vocabulary: {e}")

# Document vector: average of word vectors
def document_vector(doc_tokens, model):
    valid = [t for t in doc_tokens if t in model.wv]
    if not valid:
        return np.zeros(model.vector_size)
    return np.mean([model.wv[t] for t in valid], axis=0)

X_w2v = np.array([document_vector(doc, w2v_model) for doc in tokenized_docs])
print(f"\nWord2Vec document matrix shape: {X_w2v.shape}")
```

### 6.7 Sentiment Analysis on Movie Reviews

```python
from sklearn.datasets import load_files

# Alternative: use sklearn's built-in text datasets
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

# Create a synthetic sentiment dataset for demonstration
sentences = [
    "This movie was absolutely wonderful and amazing",
    "I loved every moment of this fantastic film",
    "Terrible movie, worst I have ever seen",
    "Awful acting, boring plot, waste of time",
    "Great performances and compelling storyline",
    "Horrible direction and poor cinematography",
    "A masterpiece of modern cinema",
    "Disgusting, offensive, and unwatchable",
    "Brilliant script and outstanding direction",
    "Complete garbage, do not waste your money"
]
sentiment_labels = [1, 1, 0, 0, 1, 0, 1, 0, 1, 0]  # 1=positive, 0=negative

# Pipeline: TF-IDF → Logistic Regression
sentiment_pipe = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1, 2), max_features=5000)),
    ('clf', LogisticRegression(C=1.0, random_state=42))
])

# Cross-validate
cv_scores = cross_val_score(sentiment_pipe, sentences, sentiment_labels, cv=3)
print(f"Sentiment Analysis CV Accuracy: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

# Train on all data and show predictions
sentiment_pipe.fit(sentences, sentiment_labels)
test_sentences = [
    "This was a great experience",
    "I absolutely hated this",
    "Quite enjoyable overall"
]
predictions = sentiment_pipe.predict(test_sentences)
probs = sentiment_pipe.predict_proba(test_sentences)
for sent, pred, prob in zip(test_sentences, predictions, probs):
    label = "Positive" if pred == 1 else "Negative"
    print(f"  '{sent}' → {label} (confidence: {max(prob):.3f})")
```

---

## 7. Advanced Topics Overview

### 7.1 Handling Imbalanced Text Data

```
Techniques:
1. Class weights: LogisticRegression(class_weight='balanced')
2. Oversampling: SMOTE on TF-IDF vectors
3. Undersampling: Random under-sampling of majority class
4. Threshold tuning: Adjust decision threshold for precision/recall
```

### 7.2 Topic Modeling (LDA)

Latent Dirichlet Allocation discovers topics in documents:

```
Each document → mixture of topics
Each topic    → distribution over words

Topic 1 (politics):  {president: 0.05, government: 0.04, ...}
Topic 2 (sports):    {game: 0.06, team: 0.05, player: 0.04, ...}
Topic 3 (tech):      {software: 0.05, data: 0.04, algorithm: 0.03, ...}

Document: "The president announced new data regulations"
→ 60% Topic 1, 30% Topic 3, 10% Topic 2
```

### 7.3 Modern NLP: Transformers (Brief)

```
Traditional ML Pipeline:
  Text → TF-IDF → LogisticRegression → Prediction

Modern Pipeline:
  Text → Tokenizer → BERT → Fine-tuned Head → Prediction

Key differences:
  - Contextual embeddings: "bank" in "river bank" vs "bank account"
  - Pre-trained on massive corpora
  - Transfer learning: fine-tune on small labeled dataset
  - Handles long-range dependencies
```

---

## 8. Real-World ML Connections

- **Spam Detection:** Naive Bayes on TF-IDF features is the classic spam filter — used in Gmail's early systems and still effective.
- **Sentiment Analysis:** Companies like Twitter, Amazon, and Yelp analyze customer reviews at scale using text classifiers.
- **Search Engines:** TF-IDF underpins the original PageRank and remains part of search relevance scoring (BM25 is an improved variant).
- **Document Classification:** Legal, medical, and news organizations auto-categorize millions of documents using text classification pipelines.
- **Chatbots:** Intent classification (mapping user messages to actions) uses the same text classification pipeline.
- **Content Moderation:** Social media platforms classify toxic, hateful, or spam content using text classifiers trained on millions of labeled examples.

---

## 9. Exercises

### Exercise 1: Stopword Impact
Train a Naive Bayes classifier on 20 newsgroups with and without stopword removal. Compare accuracy. When might removing stopwords hurt performance?

### Exercise 2: N-gram Analysis
Compare unigrams, bigrams, and trigrams for text classification. Which n-gram range gives the best accuracy on the 20 newsgroups subset?

### Exercise 3: TF-IDF Hyperparameters
Experiment with different `max_features` values (1000, 5000, 10000, 50000) and `min_df` values (1, 3, 5, 10) using Logistic Regression. What is the optimal combination?

### Exercise 4: Word2Vec Document Classification
Train Word2Vec on 20 newsgroups, create document vectors (mean of word vectors), and classify using Logistic Regression. Compare with TF-IDF + Logistic Regression.

### Solutions

**Solution 1:**
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

cats = ['alt.atheism', 'sci.space', 'comp.graphics', 'sci.med']
data = fetch_20newsgroups(subset='all', categories=cats, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42
)

# Without stopword removal
tfidf_no_stop = TfidfVectorizer(max_features=10000)
X_train_ns = tfidf_no_stop.fit_transform(X_train)
X_test_ns = tfidf_no_stop.transform(X_test)
nb_ns = MultinomialNB().fit(X_train_ns, y_train)
print(f"Without stopwords: {accuracy_score(y_test, nb_ns.predict(X_test_ns)):.4f}")

# With stopword removal
tfidf_stop = TfidfVectorizer(max_features=10000, stop_words='english')
X_train_s = tfidf_stop.fit_transform(X_train)
X_test_s = tfidf_stop.transform(X_test)
nb_s = MultinomialNB().fit(X_train_s, y_train)
print(f"With stopwords:    {accuracy_score(y_test, nb_s.predict(X_test_s)):.4f}")
# Stopwords removal usually improves accuracy slightly for topic classification
```

**Solution 2:**
```python
for ngram_range in [(1,1), (1,2), (2,2), (1,3)]:
    tfidf = TfidfVectorizer(max_features=10000, ngram_range=ngram_range)
    X_tr = tfidf.fit_transform(X_train)
    X_te = tfidf.transform(X_test)
    lr = LogisticRegression(max_iter=1000, random_state=42).fit(X_tr, y_train)
    acc = accuracy_score(y_test, lr.predict(X_te))
    print(f"n-grams={ngram_range}  Accuracy={acc:.4f}  Features={X_tr.shape[1]}")
# (1,2) typically best: captures both single words and important phrases
```

**Solution 3:**
```python
from itertools import product
best_acc, best_params = 0, None

for mf in [1000, 5000, 10000, 50000]:
    for mdf in [1, 3, 5, 10]:
        tfidf = TfidfVectorizer(max_features=mf, min_df=mdf)
        X_tr = tfidf.fit_transform(X_train)
        X_te = tfidf.transform(X_test)
        lr = LogisticRegression(max_iter=1000, random_state=42).fit(X_tr, y_train)
        acc = accuracy_score(y_test, lr.predict(X_te))
        if acc > best_acc:
            best_acc, best_params = acc, (mf, mdf)

print(f"Best: max_features={best_params[0]}, min_df={best_params[1]}, acc={best_acc:.4f}")
# Typically: max_features=10000-50000, min_df=1-3
```

**Solution 4:**
```python
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import numpy as np

tokenized = [word_tokenize(doc.lower()) for doc in data.data]
w2v = Word2Vec(tokenized, vector_size=100, window=5, min_count=5, epochs=20)

def doc_vector(tokens, model):
    vecs = [model.wv[t] for t in tokens if t in model.wv]
    return np.mean(vecs, axis=0) if vecs else np.zeros(model.vector_size)

X_w2v = np.array([doc_vector(t, w2v) for t in tokenized])
X_tr, X_te, y_tr, y_te = train_test_split(X_w2v, data.target, test_size=0.2, random_state=42)

lr_w2v = LogisticRegression(max_iter=1000, random_state=42).fit(X_tr, y_tr)
print(f"Word2Vec + LR Accuracy: {accuracy_score(y_te, lr_w2v.predict(X_te)):.4f}")
# TF-IDF + LR usually outperforms Word2Vec on small datasets
# Word2Vec benefits more with very large unlabeled corpora
```

---

## 10. Summary

| Concept | Key Takeaway |
|---------|-------------|
| Tokenization | Split text into words/subwords; first step in every NLP pipeline |
| Stopword Removal | Removes common words; improves efficiency but can hurt for sentiment tasks |
| Stemming/Lemmatization | Reduces words to base form; lemmatization is more accurate |
| Bag of Words | Simple word frequency representation; ignores order, high-dimensional |
| TF-IDF | Weighs words by importance; excellent baseline for classification |
| N-grams | Capture word sequences; bigrams often improve over unigrams |
| Word2Vec | Dense semantic embeddings; captures word similarity and analogies |
| Naive Bayes | Fast, effective text classifier; great baseline |
| Logistic Regression / SVM | Strong text classifiers with TF-IDF features |
| Modern NLP | Transformers (BERT, GPT) use contextual embeddings for state-of-the-art results |
