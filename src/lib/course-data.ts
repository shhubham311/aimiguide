export interface Topic {
  id: string;
  title: string;
  description: string;
  depth: "beginner" | "intermediate" | "advanced";
}

export interface Module {
  id: string;
  number: number;
  title: string;
  icon: string;
  color: string;
  glowColor: string;
  bgAccent: string;
  description: string;
  topics: Topic[];
}

export const courseModules: Module[] = [
  {
    id: "python-programming",
    number: 1,
    title: "Python Programming",
    icon: "Code2",
    color: "text-emerald-400",
    glowColor: "shadow-emerald-500/20",
    bgAccent: "bg-emerald-500/10",
    description: "Master Python fundamentals, data structures, and essential libraries for data science",
    topics: [
      { id: "python-basics", title: "Programming in Python & Basic Data Structures", description: "Variables, control flow, lists, tuples, sets, dictionaries, list comprehensions, and core Python syntax", depth: "beginner" },
      { id: "oops-python", title: "OOPS in Python & Examples", description: "Classes, objects, inheritance, polymorphism, encapsulation, abstraction with real-world examples", depth: "beginner" },
      { id: "file-io", title: "File I/O", description: "Reading, writing, and manipulating files — CSV, JSON, text, context managers, and file operations", depth: "beginner" },
      { id: "numpy", title: "NumPy & Implementing Basic Algorithms", description: "Array operations, broadcasting, vectorization, linear algebra ops, and implementing algorithms from scratch", depth: "intermediate" },
      { id: "pandas-eda", title: "Pandas & EDA on Datasets", description: "DataFrames, Series, data cleaning, merging, groupby, pivot tables, and exploratory data analysis on real datasets", depth: "intermediate" },
      { id: "sql-python", title: "SQL & Usage in Python", description: "SQL fundamentals, queries, joins, subqueries, window functions, and connecting Python to databases", depth: "intermediate" },
      { id: "data-visualization", title: "Data Visualization using Matplotlib & Seaborn", description: "Plots, charts, heatmaps, pair plots, customization, storytelling with data on real-world datasets", depth: "intermediate" },
    ],
  },
  {
    id: "calculus-optimization",
    number: 2,
    title: "Calculus & Optimization",
    icon: "Sigma",
    color: "text-amber-400",
    glowColor: "shadow-amber-500/20",
    bgAccent: "bg-amber-500/10",
    description: "Build the mathematical foundation for understanding ML algorithms at their core",
    topics: [
      { id: "limit-continuity", title: "Limit, Continuity & Differentiability", description: "Epsilon-delta definition, left/right limits, continuity conditions, differentiability, and their ML significance", depth: "beginner" },
      { id: "taylor-series", title: "Taylor Series", description: "Maclaurin series, approximation of functions, convergence, and how Taylor expansion connects to optimization", depth: "intermediate" },
      { id: "maxima-minima", title: "Maxima & Minima", description: "First/second derivative tests, saddle points, convexity, concavity, and their role in loss functions", depth: "intermediate" },
      { id: "single-variable-optimization", title: "Optimization Involving a Single Variable", description: "Gradient-based methods, Newton's method, learning rates, and connecting calculus to gradient descent", depth: "advanced" },
    ],
  },
  {
    id: "linear-algebra",
    number: 3,
    title: "Linear Algebra",
    icon: "Grid3x3",
    color: "text-cyan-400",
    glowColor: "shadow-cyan-500/20",
    bgAccent: "bg-cyan-500/10",
    description: "The language of data — vectors, matrices, transformations that power every ML model",
    topics: [
      { id: "vector-spaces", title: "Vector Space, Subspaces, Linear Dependence & Independence", description: "Span, basis, dimension, linear combinations, and why they matter for feature spaces in ML", depth: "beginner" },
      { id: "matrices", title: "Matrices: Types & Properties", description: "Matrix operations, special matrices (symmetric, orthogonal, diagonal), transpose, and trace properties", depth: "beginner" },
      { id: "linear-equations", title: "System of Linear Equations & Solutions", description: "Gaussian elimination, LU factorization, existence/uniqueness, and applications in solving ML systems", depth: "intermediate" },
      { id: "eigen", title: "Eigenvalues & Eigenvectors", description: "Characteristic polynomial, diagonalization, spectral theorem, and applications in PCA and graph algorithms", depth: "intermediate" },
      { id: "determinant-rank", title: "Determinant, Rank & Projections", description: "Rank-nullity theorem, orthogonal projections, and their geometric interpretations", depth: "intermediate" },
      { id: "decompositions", title: "SVD & LU Decomposition", description: "Singular Value Decomposition, LU factorization, and their critical roles in dimensionality reduction and recommender systems", depth: "advanced" },
    ],
  },
  {
    id: "probability-statistics",
    number: 4,
    title: "Probability & Statistics",
    icon: "BarChart3",
    color: "text-rose-400",
    glowColor: "shadow-rose-500/20",
    bgAccent: "bg-rose-500/10",
    description: "The backbone of every ML model — from Bayesian inference to hypothesis testing",
    topics: [
      { id: "permutation-combination", title: "Permutation & Combinations", description: "Counting principles, factorial, nCr, nPr, and their applications in probability foundations", depth: "beginner" },
      { id: "probability-basics", title: "Probability Basics: Sample Space, Events, Marginal", description: "Experiments, outcomes, axioms of probability, mutually exclusive events, and independence", depth: "beginner" },
      { id: "conditional-bayes", title: "Conditional & Joint Probability, Bayes Theorem", description: "Conditional probability, chain rule, joint distributions, Bayes theorem, and applications in Naive Bayes classifier", depth: "intermediate" },
      { id: "descriptive-stats", title: "Mean, Median, Mode, Variance & Std. Deviations", description: "Measures of central tendency, dispersion, skewness, kurtosis, and statistical moments", depth: "beginner" },
      { id: "correlation-covariance", title: "Correlation & Covariance", description: "Pearson correlation, Spearman rank, covariance matrix, and detecting multicollinearity in datasets", depth: "intermediate" },
      { id: "discrete-rv", title: "Discrete Random Variables: PMF, Uniform, Bernoulli, Binomial", description: "Probability mass functions, expected value, variance, and distributions with real-world ML examples", depth: "intermediate" },
      { id: "continuous-rv", title: "Continuous Random Variables: PDF, CDF, Normal, t, Chi-squared", description: "Probability density functions, cumulative distribution, exponential, Poisson, normal, standard normal, t-distribution, chi-squared", depth: "advanced" },
      { id: "clt", title: "Central Limit Theorem", description: "CLT proof intuition, sampling distributions, law of large numbers, and why CLT is the foundation of statistical ML", depth: "advanced" },
      { id: "hypothesis-testing", title: "Confidence Intervals & Hypothesis Testing (Z, t, Chi-squared), A/B Testing", description: "Type I/II errors, p-values, confidence intervals, Z-test, t-test, chi-squared test, and A/B testing for ML experiments", depth: "advanced" },
    ],
  },
  {
    id: "machine-learning",
    number: 5,
    title: "Machine Learning",
    icon: "BrainCircuit",
    color: "text-violet-400",
    glowColor: "shadow-violet-500/20",
    bgAccent: "bg-violet-500/10",
    description: "The heart of the course — algorithms that learn from data and make intelligent predictions",
    topics: [
      { id: "ml-intro", title: "Introduction: Supervised & Unsupervised Learning", description: "ML paradigms, use cases, the ML pipeline, and when to use which approach with real-world examples", depth: "beginner" },
      { id: "linear-regression", title: "Linear Regression & Ridge Regression", description: "Ordinary least squares, gradient descent, regularization, L2 penalty, bias-variance tradeoff on real datasets", depth: "intermediate" },
      { id: "evaluation-metrics", title: "Evaluation Metrics for Classification & Regression", description: "Accuracy, precision, recall, F1, ROC-AUC, MSE, MAE, R-squared, and when to use each", depth: "intermediate" },
      { id: "logistic-regression", title: "Logistic Regression & Gradient Descent Algorithm", description: "Sigmoid function, maximum likelihood estimation, batch/stochastic/mini-batch gradient descent, convergence analysis", depth: "intermediate" },
      { id: "knn-naive-svm", title: "KNN, Naive Bayes & SVM", description: "Distance metrics, k-selection, Bayes theorem in action, kernel trick, margin maximization, and real-world implementations", depth: "advanced" },
      { id: "decision-trees-ensemble", title: "Decision Trees & Ensemble Learning (Bagging & Boosting)", description: "Information gain, Gini impurity, Random Forest, AdaBoost, Gradient Boosting, XGBoost on real datasets", depth: "advanced" },
      { id: "clustering", title: "Clustering: Hierarchical, DBSCAN, K-Means, K-Means++", description: "Agglomerative clustering, density-based clustering, elbow method, silhouette score, and customer segmentation", depth: "advanced" },
      { id: "pca", title: "Dimensionality Reduction & PCA", description: "Eigenvalue-based PCA, explained variance, t-SNE, UMAP, and visualizing high-dimensional data", depth: "advanced" },
      { id: "time-series", title: "Time Series Analysis", description: "Stationarity, ARIMA, SARIMA, exponential smoothing, trend/seasonality decomposition, and forecasting", depth: "advanced" },
      { id: "nlp-basics", title: "NLP Basics", description: "Tokenization, stemming, lemmatization, TF-IDF, bag of words, text preprocessing, and sentiment analysis", depth: "intermediate" },
    ],
  },
  {
    id: "deep-learning",
    number: 6,
    title: "Deep Learning",
    icon: "Cpu",
    color: "text-orange-400",
    glowColor: "shadow-orange-500/20",
    bgAccent: "bg-orange-500/10",
    description: "Neural networks that power modern AI — from perceptrons to transformers",
    topics: [
      { id: "ai-ml-dl", title: "AI vs ML vs DL", description: "Understanding the hierarchy, history of AI, deep learning revolution, and where each paradigm fits", depth: "beginner" },
      { id: "mlp-gd", title: "Multi-Layer Perceptron & Gradient Descent Algorithms", description: "Forward propagation, backpropagation, chain rule, vanishing/exploding gradients, and implementing MLP from scratch", depth: "intermediate" },
      { id: "activation-functions", title: "Activation Functions (Sigmoid, Tanh, Softmax, ReLU, Leaky ReLU)", description: "Mathematical formulation, derivatives, pros/cons, gradient flow analysis, and choosing the right activation", depth: "intermediate" },
      { id: "optimization-algorithms", title: "Optimization Algorithms & Types", description: "SGD with momentum, Adam, RMSprop, AdaGrad, learning rate scheduling, and their convergence properties", depth: "advanced" },
      { id: "pytorch-basics", title: "Basics of PyTorch & Tensors", description: "Tensor operations, autograd, computational graphs, building neural networks, training loops, and GPU acceleration", depth: "intermediate" },
      { id: "advanced-nlp-embeddings", title: "Advanced NLP & Embedding Concepts (Word2Vec)", description: "Skip-gram, CBOW, GloVe, FastText, embedding spaces, semantic relationships, and analogies", depth: "advanced" },
      { id: "cnn", title: "CNN & Important Architectures (VGG16, ResNet, Inception)", description: "Convolution, pooling, batch normalization, residual connections, inception modules, and transfer learning", depth: "advanced" },
      { id: "rnn-gru-lstm", title: "RNN, GRU, LSTM & Attention Basics", description: "Sequence modeling, vanishing gradients in RNNs, gated architectures, attention mechanism, and seq2seq models", depth: "advanced" },
    ],
  },
  {
    id: "generative-ai",
    number: 7,
    title: "Generative AI",
    icon: "Sparkles",
    color: "text-pink-400",
    glowColor: "shadow-pink-500/20",
    bgAccent: "bg-pink-500/10",
    description: "Create AI that generates — images, text, code, and beyond",
    topics: [
      { id: "transformers-bert", title: "Transformers & BERT", description: "Self-attention mechanism, multi-head attention, positional encoding, BERT architecture, and masked language modeling", depth: "advanced" },
      { id: "llms", title: "Large Language Models (GPT, Llama, Mistral)", description: "Autoregressive generation, scaling laws, RLHF, instruction tuning, prompt engineering, and model architectures", depth: "advanced" },
      { id: "finetuning-llms", title: "Fine-tuning LLMs & Evaluations", description: "LoRA, QLoRA, PEFT, evaluation metrics (perplexity, BLEU, ROUGE), and practical fine-tuning workflows", depth: "advanced" },
      { id: "gans", title: "GANs & StyleGAN", description: "Generator-discriminator dynamics, training instability, DCGAN, StyleGAN architecture, and image generation", depth: "advanced" },
      { id: "diffusion-models", title: "Diffusion Models (DALL-E, Stable Diffusion)", description: "Forward/reverse diffusion, noise scheduling, U-Net architecture, classifier-free guidance, and image generation", depth: "advanced" },
      { id: "vision-language", title: "BLIP, CLIP, ViT etc.", description: "Vision-Language models, contrastive learning, vision transformers, multimodal understanding, and zero-shot classification", depth: "advanced" },
    ],
  },
  {
    id: "ml-dev-tools",
    number: 8,
    title: "ML Development Tools",
    icon: "Wrench",
    color: "text-teal-400",
    glowColor: "shadow-teal-500/20",
    bgAccent: "bg-teal-500/10",
    description: "Professional tools and workflows that production ML engineers use every day",
    topics: [
      { id: "git-github", title: "Version Control Systems — Git & GitHub", description: "Branching strategies, pull requests, code reviews, CI/CD basics, and collaborative ML development workflows", depth: "beginner" },
      { id: "databases", title: "Relational vs NoSQL Databases (MongoDB)", description: "SQL vs NoSQL trade-offs, MongoDB CRUD operations, schema design, and choosing the right database for ML workloads", depth: "intermediate" },
      { id: "fastapi", title: "API Development using FastAPI", description: "RESTful APIs, request/response models, async endpoints, authentication, and serving ML models via API", depth: "intermediate" },
      { id: "testing-models", title: "Testing Model Performance using API Locally", description: "Load testing, integration testing, model monitoring, A/B testing infrastructure, and performance benchmarking", depth: "intermediate" },
    ],
  },
  {
    id: "model-deployment",
    number: 9,
    title: "Model Deployment & Inference",
    icon: "Cloud",
    color: "text-sky-400",
    glowColor: "shadow-sky-500/20",
    bgAccent: "bg-sky-500/10",
    description: "Take your models from notebook to production — serving, scaling, and monitoring",
    topics: [
      { id: "etl-aws", title: "ETL Data Pipeline on AWS", description: "Extract, Transform, Load pipelines using AWS services (S3, Glue, Lambda), data orchestration, and pipeline automation", depth: "advanced" },
      { id: "deployment-types", title: "Deployment Types: On-Device & On-Server", description: "Edge deployment (mobile, IoT), model quantization, ONNX, TensorFlow Lite, server deployment on AWS/GCP", depth: "advanced" },
      { id: "aws-deployment", title: "Model Deployment on AWS: Sagemaker & EC2", description: "Sagemaker endpoints, autoscaling, model registry, CI/CD pipelines, EC2 deployment, Docker containers", depth: "advanced" },
    ],
  },
  {
    id: "recommendation-engine",
    number: 10,
    title: "Recommendation Engine",
    icon: "ThumbsUp",
    color: "text-yellow-400",
    glowColor: "shadow-yellow-500/20",
    bgAccent: "bg-yellow-500/10",
    description: "Build intelligent recommendation systems — the technology behind Netflix, Amazon, and Spotify",
    topics: [
      { id: "collaborative-content", title: "Content-Based & Collaborative Filtering", description: "User-item interactions, cosine similarity, matrix factorization, hybrid approaches, and cold-start solutions", depth: "advanced" },
      { id: "neural-recommendation", title: "Neural Network Based (2-Tower Approach)", description: "Dual encoder architecture, embedding layers, candidate generation, and neural collaborative filtering", depth: "advanced" },
      { id: "retrieval-candidates", title: "Retrieval & Candidate Set Generation", description: "Approximate nearest neighbors (ANN), FAISS, ScaNN, vector search, and efficient retrieval strategies", depth: "advanced" },
      { id: "learning-to-rank", title: "Learning to Rank & Loss Functions", description: "Pointwise (Cross-entropy), Pairwise (RankNET), Listwise (LambdaMART, LambdaRank), and ranking architecture", depth: "advanced" },
      { id: "rec-evaluation", title: "Evaluation Metrics: Offline (MRR@K, MAP@K, NDCG@K) & Online (CTR)", description: "Mean Reciprocal Rank, Mean Average Precision, NDCG, Click-Through Rate, session metrics, and real-world evaluation", depth: "advanced" },
    ],
  },
  {
    id: "ml-interview",
    number: 11,
    title: "ML Design Interview Questions",
    icon: "MessageSquareCode",
    color: "text-lime-400",
    glowColor: "shadow-lime-500/20",
    bgAccent: "bg-lime-500/10",
    description: "Prepare for interviews at top AI companies with design patterns and system thinking",
    topics: [
      { id: "interview-patterns", title: "Interview Patterns of Top AI Companies", description: "FAANG interview structure, ML design framework, expectation management, and common question categories", depth: "advanced" },
      { id: "important-concepts", title: "Important Concepts to Go Through", description: "High-frequency interview topics, mathematical foundations, algorithmic complexity, and conceptual depth", depth: "advanced" },
      { id: "multimodal-embeddings", title: "Multi-modal Representations using Embeddings", description: "Joint embedding spaces, contrastive learning, CLIP-style models, and cross-modal retrieval", depth: "advanced" },
      { id: "search-retrieval-rec", title: "Search, Retrieval & Recommendations", description: "Search ranking, query understanding, relevance scoring, recommendation architecture design, and scalability", depth: "advanced" },
      { id: "spam-outlier", title: "Spam/Outlier Detection & Classification", description: "Anomaly detection techniques, imbalanced learning, precision-recall tradeoffs, and production monitoring", depth: "advanced" },
      { id: "multi-class-multi-modal", title: "Multi-class & Multi-modal Classifications", description: "Hierarchical classification, label embeddings, multi-modal fusion strategies, and real-world challenges", depth: "advanced" },
    ],
  },
  {
    id: "dsa",
    number: 12,
    title: "Data Structures & Algorithms",
    icon: "GitBranch",
    color: "text-fuchsia-400",
    glowColor: "shadow-fuchsia-500/20",
    bgAccent: "bg-fuchsia-500/10",
    description: "Sharpen your problem-solving skills — the foundation of every technical interview",
    topics: [
      { id: "python-dsa", title: "Python Data Structures", description: "Arrays, linked lists, stacks, queues, hash maps, trees, graphs — implementations and complexity analysis", depth: "beginner" },
      { id: "dsa-problems", title: "Problems Practice: Array, Linked List, Greedy, Recursion, Binary Search, Heaps, Stack, Queue, String, Binary Tree, Graph, DP", description: "Curated problem sets for each data structure and algorithm with solutions and complexity analysis", depth: "intermediate" },
      { id: "advanced-dsa", title: "Advanced Topics: Trie, Fenwick Tree, Segment Tree", description: "Prefix trees, Binary Indexed Trees, Range Queries, and their applications in competitive programming and interviews", depth: "advanced" },
    ],
  },
];

export function generateCommand(moduleId: string, topicId: string): string {
  return `/learn ${moduleId}.${topicId}`;
}

export function generateTeachingPrompt(moduleId: string, topicId: string, topicTitle: string, moduleTitle: string, topicDescription: string): string {
  return `Teach me "${topicTitle}" (from Module: ${moduleTitle}).

${topicDescription}

Instructions for teaching:
- I am a beginner to this specific topic. Start from absolute basics and build up progressively.
- Cover the WHY first (why this concept matters in real-world AI/ML), then the WHAT, then the HOW.
- Include mathematical foundations with clear formulas and explain each variable/symbol.
- Provide practical Python code implementations using real-world datasets (e.g., from sklearn, Kaggle, UCI repository).
- Use visuals and diagrams where possible — describe them in text or use ASCII/text-based representations.
- After each major concept, give me a quick exercise or thought question to test understanding.
- Connect this topic to the bigger picture — how it relates to other AI/ML concepts.
- Make it interactive: ask me questions, guide me through thinking, don't just dump information.
- Be thorough and exhaustive — share every piece of knowledge you have on this topic.
- Teaching style: conversational, patient, encouraging, like a mentor sitting next to me.`;
}

export function getAllTopicIds(): string[] {
  const ids: string[] = [];
  courseModules.forEach((mod) => {
    mod.topics.forEach((topic) => {
      ids.push(`${mod.id}.${topic.id}`);
    });
  });
  return ids;
}
