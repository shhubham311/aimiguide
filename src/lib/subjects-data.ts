// ── Learning Subject Tracker for Data Scientist / GenAI Engineer / ML Engineer / AI Engineer ──
// All links redirect to GeeksforGeeks tutorials and references (verified working)

export interface LearningSubject {
  id: string;
  title: string;
  url: string;
  difficulty: "beginner" | "intermediate" | "advanced";
  tags: string[];
}

export interface SubjectCategory {
  id: string;
  name: string;
  icon: string;
  color: string;
  glowColor: string;
  bgAccent: string;
  description: string;
  roleRelevance: string[];
}

export const subjectCategories: SubjectCategory[] = [
  {
    id: "python-fundamentals",
    name: "Python Fundamentals",
    icon: "Code2",
    color: "text-yellow-400",
    glowColor: "shadow-yellow-500/20",
    bgAccent: "bg-yellow-500/10",
    description: "Core Python skills required for all ML/AI/Data Science roles",
    roleRelevance: ["Data Scientist", "ML Engineer", "AI Engineer", "GenAI Engineer"],
  },
  {
    id: "math-foundation",
    name: "Mathematics Foundation",
    icon: "Sigma",
    color: "text-emerald-400",
    glowColor: "shadow-emerald-500/20",
    bgAccent: "bg-emerald-500/10",
    description: "Linear algebra, calculus, probability & statistics for ML theory",
    roleRelevance: ["Data Scientist", "ML Engineer", "AI Engineer", "GenAI Engineer"],
  },
  {
    id: "data-wrangling",
    name: "Data Wrangling & Analysis",
    icon: "Database",
    color: "text-blue-400",
    glowColor: "shadow-blue-500/20",
    bgAccent: "bg-blue-500/10",
    description: "Pandas, NumPy, data cleaning, EDA, and SQL for data manipulation",
    roleRelevance: ["Data Scientist", "ML Engineer", "AI Engineer"],
  },
  {
    id: "data-visualization",
    name: "Data Visualization",
    icon: "BarChart3",
    color: "text-pink-400",
    glowColor: "shadow-pink-500/20",
    bgAccent: "bg-pink-500/10",
    description: "Matplotlib, Seaborn, Plotly for communicating data insights",
    roleRelevance: ["Data Scientist", "ML Engineer", "AI Engineer"],
  },
  {
    id: "ml-algorithms",
    name: "Machine Learning Algorithms",
    icon: "BrainCircuit",
    color: "text-violet-400",
    glowColor: "shadow-violet-500/20",
    bgAccent: "bg-violet-500/10",
    description: "Supervised, unsupervised, and ensemble learning algorithms",
    roleRelevance: ["Data Scientist", "ML Engineer", "AI Engineer", "GenAI Engineer"],
  },
  {
    id: "ml-practice",
    name: "ML Workflow & Best Practices",
    icon: "Settings2",
    color: "text-teal-400",
    glowColor: "shadow-teal-500/20",
    bgAccent: "bg-teal-500/10",
    description: "Feature engineering, cross-validation, hyperparameter tuning, metrics",
    roleRelevance: ["Data Scientist", "ML Engineer", "AI Engineer", "GenAI Engineer"],
  },
  {
    id: "deep-learning",
    name: "Deep Learning",
    icon: "Cpu",
    color: "text-orange-400",
    glowColor: "shadow-orange-500/20",
    bgAccent: "bg-orange-500/10",
    description: "Neural networks, CNNs, RNNs, and frameworks (TensorFlow, PyTorch)",
    roleRelevance: ["ML Engineer", "AI Engineer", "GenAI Engineer"],
  },
  {
    id: "nlp",
    name: "Natural Language Processing",
    icon: "MessageSquare",
    color: "text-cyan-400",
    glowColor: "shadow-cyan-500/20",
    bgAccent: "bg-cyan-500/10",
    description: "Text processing, embeddings, sequence models, and language understanding",
    roleRelevance: ["Data Scientist", "ML Engineer", "AI Engineer", "GenAI Engineer"],
  },
  {
    id: "computer-vision",
    name: "Computer Vision",
    icon: "Eye",
    color: "text-indigo-400",
    glowColor: "shadow-indigo-500/20",
    bgAccent: "bg-indigo-500/10",
    description: "Image classification, object detection, segmentation, and vision models",
    roleRelevance: ["ML Engineer", "AI Engineer", "GenAI Engineer"],
  },
  {
    id: "transformers",
    name: "Transformer Architecture",
    icon: "Sparkles",
    color: "text-fuchsia-400",
    glowColor: "shadow-fuchsia-500/20",
    bgAccent: "bg-fuchsia-500/10",
    description: "Attention mechanism, BERT, GPT, and modern transformer variants",
    roleRelevance: ["ML Engineer", "AI Engineer", "GenAI Engineer"],
  },
  {
    id: "genai",
    name: "Generative AI",
    icon: "Wand2",
    color: "text-purple-400",
    glowColor: "shadow-purple-500/20",
    bgAccent: "bg-purple-500/10",
    description: "LLMs, prompt engineering, RAG, fine-tuning, and AI agents",
    roleRelevance: ["AI Engineer", "GenAI Engineer"],
  },
  {
    id: "mlops",
    name: "MLOps & Deployment",
    icon: "Container",
    color: "text-lime-400",
    glowColor: "shadow-lime-500/20",
    bgAccent: "bg-lime-500/10",
    description: "Docker, model serving, CI/CD, monitoring, and production ML systems",
    roleRelevance: ["ML Engineer", "AI Engineer", "GenAI Engineer"],
  },
  {
    id: "data-engineering",
    name: "Data Engineering",
    icon: "Server",
    color: "text-sky-400",
    glowColor: "shadow-sky-500/20",
    bgAccent: "bg-sky-500/10",
    description: "ETL pipelines, data lakes, streaming, and big data tools",
    roleRelevance: ["Data Scientist", "ML Engineer", "AI Engineer"],
  },
  {
    id: "dsa",
    name: "DSA for ML Interviews",
    icon: "Binary",
    color: "text-red-400",
    glowColor: "shadow-red-500/20",
    bgAccent: "bg-red-500/10",
    description: "Data structures and algorithms essential for coding interviews",
    roleRelevance: ["Data Scientist", "ML Engineer", "AI Engineer", "GenAI Engineer"],
  },
];

export const learningSubjects: LearningSubject[] = [
  // ── Python Fundamentals ──
  { id: "python-basics", title: "Python Basics", url: "https://www.geeksforgeeks.org/python-programming-language/", difficulty: "beginner", tags: ["python", "syntax", "variables"] },
  { id: "python-oops", title: "Python OOPs Concepts", url: "https://www.geeksforgeeks.org/python-classes-and-objects/", difficulty: "beginner", tags: ["python", "oop", "classes"] },
  { id: "python-data-structures", title: "Python Data Structures (List, Tuple, Set, Dict)", url: "https://www.geeksforgeeks.org/python-data-structures/", difficulty: "beginner", tags: ["python", "data-structures"] },
  { id: "python-file-handling", title: "File Handling in Python", url: "https://www.geeksforgeeks.org/file-handling-python/", difficulty: "beginner", tags: ["python", "io"] },
  { id: "python-iterators-generators", title: "Iterators & Generators", url: "https://www.geeksforgeeks.org/generators-in-python/", difficulty: "intermediate", tags: ["python", "generators"] },
  { id: "python-decorators", title: "Decorators in Python", url: "https://www.geeksforgeeks.org/python/decorators-in-python/", difficulty: "intermediate", tags: ["python", "decorators"] },
  { id: "python-exception-handling", title: "Exception Handling", url: "https://www.geeksforgeeks.org/python-exception-handling/", difficulty: "beginner", tags: ["python", "errors"] },
  { id: "python-regex", title: "Regular Expressions in Python", url: "https://www.geeksforgeeks.org/python/regular-expression-python-examples/", difficulty: "intermediate", tags: ["python", "regex"] },

  // ── Mathematics Foundation ──
  { id: "linear-algebra-intro", title: "Linear Algebra for ML", url: "https://www.geeksforgeeks.org/machine-learning/ml-linear-algebra-operations/", difficulty: "beginner", tags: ["math", "linear-algebra"] },
  { id: "matrices-vectors", title: "Matrices & Vectors", url: "https://www.geeksforgeeks.org/data-science/matrices-and-matrix-arithmetic-for-machine-learning/", difficulty: "beginner", tags: ["math", "matrices"] },
  { id: "eigenvalues-svd", title: "Eigenvalues & SVD", url: "https://www.geeksforgeeks.org/artificial-intelligence/what-is-the-difference-between-eigenvalues-and-eigenvectors/", difficulty: "advanced", tags: ["math", "svd", "pca"] },
  { id: "calculus-for-ml", title: "Calculus for Machine Learning", url: "https://www.geeksforgeeks.org/machine-learning/calculus-for-machine-learning-key-concepts-and-applications/", difficulty: "intermediate", tags: ["math", "calculus", "derivatives"] },
  { id: "gradient-descent-math", title: "Gradient Descent (Mathematical Foundation)", url: "https://www.geeksforgeeks.org/machine-learning/gradient-descent-algorithm-and-its-variants/", difficulty: "intermediate", tags: ["math", "optimization"] },
  { id: "probability-intro", title: "Probability for Data Science", url: "https://www.geeksforgeeks.org/machine-learning/probability-in-machine-learning/", difficulty: "beginner", tags: ["math", "probability"] },
  { id: "distributions", title: "Statistical Distributions", url: "https://www.geeksforgeeks.org/data-science/probability-distributions-in-data-science/", difficulty: "intermediate", tags: ["math", "statistics", "distributions"] },
  { id: "bayesian-theorem", title: "Bayes' Theorem & Bayesian Inference", url: "https://www.geeksforgeeks.org/machine-learning/bayes-theorem-in-machine-learning/", difficulty: "intermediate", tags: ["math", "bayesian"] },
  { id: "hypothesis-testing", title: "Hypothesis Testing & P-values", url: "https://www.geeksforgeeks.org/data-science/understanding-hypothesis-testing/", difficulty: "intermediate", tags: ["statistics", "hypothesis-testing"] },

  // ── Data Wrangling & Analysis ──
  { id: "numpy-intro", title: "NumPy Tutorial", url: "https://www.geeksforgeeks.org/python-numpy/", difficulty: "beginner", tags: ["numpy", "arrays"] },
  { id: "numpy-operations", title: "NumPy Array Operations", url: "https://www.geeksforgeeks.org/python/python-operations-on-numpy-arrays/", difficulty: "beginner", tags: ["numpy", "broadcasting"] },
  { id: "pandas-intro", title: "Pandas Tutorial", url: "https://www.geeksforgeeks.org/pandas-tutorial/", difficulty: "beginner", tags: ["pandas", "dataframe"] },
  { id: "pandas-data-cleaning", title: "Data Cleaning with Pandas", url: "https://www.geeksforgeeks.org/data-analysis/data-cleaning-introduction/", difficulty: "intermediate", tags: ["pandas", "cleaning"] },
  { id: "pandas-groupby", title: "GroupBy & Aggregation", url: "https://www.geeksforgeeks.org/pandas-groupby/", difficulty: "intermediate", tags: ["pandas", "aggregation"] },
  { id: "pandas-merge-join", title: "Merge, Join & Concatenate", url: "https://www.geeksforgeeks.org/pandas/python-merge-join-and-concatenate-dataframes-using-panda/", difficulty: "intermediate", tags: ["pandas", "join"] },
  { id: "eda-python", title: "Exploratory Data Analysis (EDA)", url: "https://www.geeksforgeeks.org/exploratory-data-analysis-in-python/", difficulty: "intermediate", tags: ["eda", "analysis"] },
  { id: "sql-basics", title: "SQL for Data Science", url: "https://www.geeksforgeeks.org/sql-tutorial/", difficulty: "beginner", tags: ["sql", "queries"] },
  { id: "sql-advanced", title: "Advanced SQL (Joins, Subqueries, Window Functions)", url: "https://www.geeksforgeeks.org/sql/window-functions-in-sql/", difficulty: "intermediate", tags: ["sql", "joins", "window-functions"] },
  { id: "handling-missing-data", title: "Handling Missing Data", url: "https://www.geeksforgeeks.org/machine-learning/handling-missing-data/", difficulty: "intermediate", tags: ["pandas", "missing-data"] },

  // ── Data Visualization ──
  { id: "matplotlib-intro", title: "Matplotlib Tutorial", url: "https://www.geeksforgeeks.org/matplotlib-tutorial/", difficulty: "beginner", tags: ["matplotlib", "plotting"] },
  { id: "seaborn-intro", title: "Seaborn Tutorial", url: "https://www.geeksforgeeks.org/python-seaborn-tutorial/", difficulty: "beginner", tags: ["seaborn", "visualization"] },
  { id: "plotly-dashboards", title: "Interactive Plots with Plotly", url: "https://www.geeksforgeeks.org/python-plotly-tutorial/", difficulty: "intermediate", tags: ["plotly", "interactive"] },
  { id: "distribution-plots", title: "Visualizing Distributions (Box, Violin, Histogram)", url: "https://www.geeksforgeeks.org/data-visualization/how-is-violinplot-different-from-boxplot/", difficulty: "intermediate", tags: ["visualization", "statistics"] },
  { id: "correlation-heatmap", title: "Correlation & Heatmaps", url: "https://www.geeksforgeeks.org/python/seaborn-heatmap-a-comprehensive-guide/", difficulty: "beginner", tags: ["visualization", "correlation"] },

  // ── Machine Learning Algorithms ──
  { id: "ml-introduction", title: "Introduction to Machine Learning", url: "https://www.geeksforgeeks.org/machine-learning/introduction-machine-learning/", difficulty: "beginner", tags: ["ml", "introduction"] },
  { id: "supervised-learning", title: "Supervised Learning", url: "https://www.geeksforgeeks.org/machine-learning/supervised-machine-learning/", difficulty: "beginner", tags: ["ml", "supervised"] },
  { id: "unsupervised-learning", title: "Unsupervised Learning", url: "https://www.geeksforgeeks.org/machine-learning/unsupervised-learning/", difficulty: "beginner", tags: ["ml", "unsupervised"] },
  { id: "linear-regression", title: "Linear Regression", url: "https://www.geeksforgeeks.org/machine-learning/ml-linear-regression/", difficulty: "beginner", tags: ["ml", "regression"] },
  { id: "logistic-regression", title: "Logistic Regression", url: "https://www.geeksforgeeks.org/machine-learning/understanding-logistic-regression/", difficulty: "beginner", tags: ["ml", "classification"] },
  { id: "decision-trees", title: "Decision Trees", url: "https://www.geeksforgeeks.org/machine-learning/decision-tree-introduction-example/", difficulty: "beginner", tags: ["ml", "trees"] },
  { id: "random-forest", title: "Random Forest", url: "https://www.geeksforgeeks.org/machine-learning/random-forest-algorithm-in-machine-learning/", difficulty: "intermediate", tags: ["ml", "ensemble"] },
  { id: "gradient-boosting", title: "Gradient Boosting (XGBoost, LightGBM)", url: "https://www.geeksforgeeks.org/machine-learning/a-comprehensive-guide-to-ensemble-learning/", difficulty: "advanced", tags: ["ml", "ensemble", "xgboost"] },
  { id: "svm-algorithm", title: "Support Vector Machines (SVM)", url: "https://www.geeksforgeeks.org/machine-learning/support-vector-machine-algorithm/", difficulty: "intermediate", tags: ["ml", "svm", "classification"] },
  { id: "knn-algorithm", title: "K-Nearest Neighbors (KNN)", url: "https://www.geeksforgeeks.org/machine-learning/k-nearest-neighbours/", difficulty: "beginner", tags: ["ml", "knn"] },
  { id: "naive-bayes", title: "Naive Bayes Classifier", url: "https://www.geeksforgeeks.org/machine-learning/naive-bayes-classifiers/", difficulty: "beginner", tags: ["ml", "naive-bayes"] },
  { id: "kmeans-clustering", title: "K-Means Clustering", url: "https://www.geeksforgeeks.org/machine-learning/k-means-clustering-introduction/", difficulty: "beginner", tags: ["ml", "clustering"] },
  { id: "hierarchical-clustering", title: "Hierarchical Clustering", url: "https://www.geeksforgeeks.org/machine-learning/hierarchical-clustering/", difficulty: "intermediate", tags: ["ml", "clustering"] },
  { id: "dbscan-clustering", title: "DBSCAN Clustering", url: "https://www.geeksforgeeks.org/machine-learning/dbscan-clustering-in-ml-density-based-clustering/", difficulty: "intermediate", tags: ["ml", "clustering", "density"] },
  { id: "pca-dimensionality", title: "PCA & Dimensionality Reduction", url: "https://www.geeksforgeeks.org/machine-learning/dimensionality-reduction/", difficulty: "intermediate", tags: ["ml", "pca", "dimensionality-reduction"] },
  { id: "reinforcement-learning", title: "Reinforcement Learning", url: "https://www.geeksforgeeks.org/machine-learning/what-is-reinforcement-learning/", difficulty: "advanced", tags: ["ml", "rl"] },
  { id: "apriori-algorithm", title: "Apriori Algorithm (Association Rules)", url: "https://www.geeksforgeeks.org/machine-learning/apriori-algorithm/", difficulty: "intermediate", tags: ["ml", "association-rules"] },

  // ── ML Workflow & Best Practices ──
  { id: "data-preprocessing", title: "Data Preprocessing for ML", url: "https://www.geeksforgeeks.org/data-analysis/data-preprocessing-machine-learning-python/", difficulty: "beginner", tags: ["ml", "preprocessing"] },
  { id: "feature-engineering", title: "Feature Engineering", url: "https://www.geeksforgeeks.org/machine-learning/what-is-feature-engineering/", difficulty: "intermediate", tags: ["ml", "features"] },
  { id: "feature-selection", title: "Feature Selection Techniques", url: "https://www.geeksforgeeks.org/machine-learning/feature-selection-techniques-in-machine-learning/", difficulty: "intermediate", tags: ["ml", "feature-selection"] },
  { id: "cross-validation", title: "Cross-Validation", url: "https://www.geeksforgeeks.org/machine-learning/cross-validation-machine-learning/", difficulty: "intermediate", tags: ["ml", "validation"] },
  { id: "hyperparameter-tuning", title: "Hyperparameter Tuning (Grid, Random, Bayesian)", url: "https://www.geeksforgeeks.org/machine-learning/hyperparameter-tuning/", difficulty: "intermediate", tags: ["ml", "tuning"] },
  { id: "ml-metrics", title: "ML Evaluation Metrics", url: "https://www.geeksforgeeks.org/machine-learning/metrics-for-machine-learning-model/", difficulty: "intermediate", tags: ["ml", "metrics", "accuracy"] },
  { id: "bias-variance", title: "Bias-Variance Tradeoff", url: "https://www.geeksforgeeks.org/machine-learning/bias-vs-variance-in-machine-learning/", difficulty: "intermediate", tags: ["ml", "bias-variance"] },
  { id: "overfitting-underfitting", title: "Overfitting & Underfitting", url: "https://www.geeksforgeeks.org/machine-learning/underfitting-and-overfitting-in-machine-learning/", difficulty: "beginner", tags: ["ml", "regularization"] },
  { id: "regularization", title: "Regularization (L1, L2, ElasticNet)", url: "https://www.geeksforgeeks.org/machine-learning/regularization-in-machine-learning/", difficulty: "intermediate", tags: ["ml", "regularization"] },
  { id: "ml-pipeline", title: "Building ML Pipelines", url: "https://www.geeksforgeeks.org/blogs/machine-learning-pipeline/", difficulty: "advanced", tags: ["ml", "pipeline"] },

  // ── Deep Learning ──
  { id: "dl-introduction", title: "Introduction to Deep Learning", url: "https://www.geeksforgeeks.org/deep-learning/deep-learning-tutorial/", difficulty: "beginner", tags: ["dl", "introduction"] },
  { id: "neural-networks", title: "Artificial Neural Networks (ANN)", url: "https://www.geeksforgeeks.org/deep-learning/artificial-neural-networks-and-its-applications/", difficulty: "intermediate", tags: ["dl", "ann"] },
  { id: "activation-functions", title: "Activation Functions", url: "https://www.geeksforgeeks.org/machine-learning/activation-functions-neural-networks/", difficulty: "intermediate", tags: ["dl", "activation"] },
  { id: "backpropagation", title: "Backpropagation Algorithm", url: "https://www.geeksforgeeks.org/machine-learning/backpropagation-in-neural-network/", difficulty: "intermediate", tags: ["dl", "backpropagation"] },
  { id: "cnn-intro", title: "Convolutional Neural Networks (CNN)", url: "https://www.geeksforgeeks.org/machine-learning/introduction-convolution-neural-network/", difficulty: "intermediate", tags: ["dl", "cnn"] },
  { id: "rnn-intro", title: "Recurrent Neural Networks (RNN)", url: "https://www.geeksforgeeks.org/machine-learning/introduction-to-recurrent-neural-network/", difficulty: "intermediate", tags: ["dl", "rnn"] },
  { id: "lstm-gru", title: "LSTM & GRU Networks", url: "https://www.geeksforgeeks.org/deep-learning/deep-learning-introduction-to-long-short-term-memory/", difficulty: "advanced", tags: ["dl", "lstm", "gru"] },
  { id: "tensorflow-tutorial", title: "TensorFlow Tutorial", url: "https://www.geeksforgeeks.org/deep-learning/tensorflow/", difficulty: "beginner", tags: ["tensorflow", "framework"] },
  { id: "keras-tutorial", title: "Keras Tutorial", url: "https://www.geeksforgeeks.org/python/keras-tutorial/", difficulty: "beginner", tags: ["keras", "framework"] },
  { id: "pytorch-tutorial", title: "PyTorch Tutorial", url: "https://www.geeksforgeeks.org/python/pytorch-tutorial/", difficulty: "beginner", tags: ["pytorch", "framework"] },
  { id: "transfer-learning", title: "Transfer Learning", url: "https://www.geeksforgeeks.org/machine-learning/ml-introduction-to-transfer-learning/", difficulty: "advanced", tags: ["dl", "transfer-learning"] },
  { id: "autoencoders", title: "Autoencoders", url: "https://www.geeksforgeeks.org/machine-learning/auto-encoders/", difficulty: "advanced", tags: ["dl", "autoencoder"] },
  { id: "gan-intro", title: "Generative Adversarial Networks (GAN)", url: "https://www.geeksforgeeks.org/generative-adversarial-network-gan/", difficulty: "advanced", tags: ["dl", "gan"] },

  // ── Natural Language Processing ──
  { id: "nlp-introduction", title: "Introduction to NLP", url: "https://www.geeksforgeeks.org/nlp/natural-language-processing-nlp-tutorial/", difficulty: "beginner", tags: ["nlp", "introduction"] },
  { id: "text-preprocessing", title: "Text Preprocessing (Tokenization, Stemming, Lemmatization)", url: "https://www.geeksforgeeks.org/machine-learning/text-preprocessing-in-python-set-1/", difficulty: "beginner", tags: ["nlp", "preprocessing"] },
  { id: "tfidf-bow", title: "TF-IDF & Bag of Words", url: "https://www.geeksforgeeks.org/machine-learning/understanding-tf-idf-term-frequency-inverse-document-frequency/", difficulty: "beginner", tags: ["nlp", "tfidf", "bow"] },
  { id: "word-embeddings", title: "Word Embeddings (Word2Vec, GloVe)", url: "https://www.geeksforgeeks.org/nlp/word-embeddings-in-nlp/", difficulty: "intermediate", tags: ["nlp", "embeddings"] },
  { id: "text-classification", title: "Text Classification Techniques", url: "https://www.geeksforgeeks.org/nlp/what-is-text-classification/", difficulty: "intermediate", tags: ["nlp", "classification"] },
  { id: "named-entity-recognition", title: "Named Entity Recognition (NER)", url: "https://www.geeksforgeeks.org/nlp/named-entity-recognition/", difficulty: "advanced", tags: ["nlp", "ner"] },
  { id: "pos-tagging", title: "Part-of-Speech Tagging", url: "https://www.geeksforgeeks.org/nlp/nlp-part-of-speech-default-tagging/", difficulty: "intermediate", tags: ["nlp", "pos"] },
  { id: "topic-modeling", title: "Topic Modeling (LDA)", url: "https://www.geeksforgeeks.org/nlp/topic-modeling-using-latent-dirichlet-allocation-lda/", difficulty: "advanced", tags: ["nlp", "lda"] },
  { id: "spacy-tutorial", title: "spaCy NLP Library", url: "https://www.geeksforgeeks.org/nlp/spacy-for-natural-language-processing/", difficulty: "intermediate", tags: ["nlp", "spacy"] },
  { id: "nltk-tutorial", title: "NLTK Tutorial", url: "https://www.geeksforgeeks.org/nlp/nltk-tutorial/", difficulty: "beginner", tags: ["nlp", "nltk"] },
  { id: "text-similarity", title: "Text Similarity & Cosine Similarity", url: "https://www.geeksforgeeks.org/dbms/cosine-similarity/", difficulty: "intermediate", tags: ["nlp", "similarity"] },

  // ── Computer Vision ──
  { id: "cv-introduction", title: "Introduction to Computer Vision", url: "https://www.geeksforgeeks.org/computer-vision/computer-vision/", difficulty: "beginner", tags: ["cv", "introduction"] },
  { id: "opencv-tutorial", title: "OpenCV Tutorial", url: "https://www.geeksforgeeks.org/opencv-python-tutorial/", difficulty: "beginner", tags: ["opencv", "image-processing"] },
  { id: "image-preprocessing", title: "Image Preprocessing Techniques", url: "https://www.geeksforgeeks.org/python/image-processing-in-python/", difficulty: "intermediate", tags: ["cv", "preprocessing"] },
  { id: "edge-detection", title: "Edge Detection (Canny, Sobel)", url: "https://www.geeksforgeeks.org/python/real-time-edge-detection-using-opencv-python/", difficulty: "intermediate", tags: ["cv", "edge-detection"] },
  { id: "object-detection", title: "Object Detection (YOLO, SSD, Faster R-CNN)", url: "https://www.geeksforgeeks.org/computer-vision/what-is-object-detection-in-computer-vision/", difficulty: "advanced", tags: ["cv", "object-detection", "yolo"] },
  { id: "image-segmentation", title: "Image Segmentation", url: "https://www.geeksforgeeks.org/computer-vision/image-segmentation-techniques-and-applications/", difficulty: "advanced", tags: ["cv", "segmentation"] },
  { id: "face-detection-recognition", title: "Face Detection & Recognition", url: "https://www.geeksforgeeks.org/python/opencv-python-program-face-detection/", difficulty: "intermediate", tags: ["cv", "face"] },
  { id: "yolo-object-detection", title: "YOLO for Real-Time Object Detection", url: "https://www.geeksforgeeks.org/machine-learning/yolo-you-only-look-once-real-time-object-detection/", difficulty: "advanced", tags: ["cv", "yolo", "real-time"] },
  { id: "optical-character-recognition", title: "OCR with Tesseract & OpenCV", url: "https://www.geeksforgeeks.org/python/text-detection-and-extraction-using-opencv-and-ocr/", difficulty: "intermediate", tags: ["cv", "ocr"] },

  // ── Transformer Architecture ──
  { id: "attention-mechanism", title: "Attention Mechanism", url: "https://www.geeksforgeeks.org/artificial-intelligence/ml-attention-mechanism/", difficulty: "advanced", tags: ["transformer", "attention"] },
  { id: "transformer-intro", title: "Transformer Architecture", url: "https://www.geeksforgeeks.org/machine-learning/getting-started-with-transformers/", difficulty: "advanced", tags: ["transformer", "architecture"] },
  { id: "bert-intro", title: "BERT (Bidirectional Encoder Representations)", url: "https://www.geeksforgeeks.org/nlp/explanation-of-bert-model-nlp/", difficulty: "advanced", tags: ["transformer", "bert"] },
  { id: "gpt-intro", title: "GPT Architecture (Generative Pre-trained)", url: "https://www.geeksforgeeks.org/artificial-intelligence/introduction-to-generative-pre-trained-transformer-gpt/", difficulty: "advanced", tags: ["transformer", "gpt"] },
  { id: "t5-model", title: "T5 (Text-to-Text Transfer Transformer)", url: "https://www.geeksforgeeks.org/nlp/t5-text-to-text-transfer-transformer/", difficulty: "advanced", tags: ["transformer", "t5"] },
  { id: "vision-transformer", title: "Vision Transformer (ViT)", url: "https://www.geeksforgeeks.org/deep-learning/vision-transformer-vit-architecture/", difficulty: "advanced", tags: ["transformer", "vit", "cv"] },
  { id: "encoder-decoder", title: "Encoder-Decoder Architecture", url: "https://www.geeksforgeeks.org/nlp/encoder-decoder-models/", difficulty: "advanced", tags: ["transformer", "seq2seq"] },

  // ── Generative AI ──
  { id: "genai-introduction", title: "Introduction to Generative AI", url: "https://www.geeksforgeeks.org/artificial-intelligence/generative-ai-tutorial/", difficulty: "beginner", tags: ["genai", "introduction"] },
  { id: "llm-intro", title: "Large Language Models (LLMs)", url: "https://www.geeksforgeeks.org/artificial-intelligence/large-language-model-llm/", difficulty: "intermediate", tags: ["genai", "llm"] },
  { id: "prompt-engineering", title: "Prompt Engineering", url: "https://www.geeksforgeeks.org/blogs/what-is-prompt-engineering-the-ai-revolution/", difficulty: "beginner", tags: ["genai", "prompting"] },
  { id: "prompt-engineering-advanced", title: "Advanced Prompting (Chain-of-Thought, Few-Shot)", url: "https://www.geeksforgeeks.org/blogs/prompt-engineering-best-practices/", difficulty: "intermediate", tags: ["genai", "prompting"] },
  { id: "rag-intro", title: "Retrieval-Augmented Generation (RAG)", url: "https://www.geeksforgeeks.org/nlp/what-is-retrieval-augmented-generation-rag/", difficulty: "intermediate", tags: ["genai", "rag"] },
  { id: "fine-tuning-llm", title: "Fine-tuning LLMs", url: "https://www.geeksforgeeks.org/deep-learning/fine-tuning-large-language-model-llm/", difficulty: "advanced", tags: ["genai", "fine-tuning"] },
  { id: "langchain-intro", title: "LangChain Tutorial", url: "https://www.geeksforgeeks.org/langchain-tutorial/", difficulty: "intermediate", tags: ["genai", "langchain"] },
  { id: "llamaindex-intro", title: "LlamaIndex Tutorial", url: "https://www.geeksforgeeks.org/artificial-intelligence/building-ai-application-with-llamaindex/", difficulty: "intermediate", tags: ["genai", "llamaindex"] },
  { id: "vector-databases", title: "Vector Databases (Pinecone, ChromaDB)", url: "https://www.geeksforgeeks.org/data-science/what-is-a-vector-database/", difficulty: "intermediate", tags: ["genai", "vectors", "embeddings"] },
  { id: "ai-agents-intro", title: "AI Agents & Tool Use", url: "https://www.geeksforgeeks.org/artificial-intelligence/agents-artificial-intelligence/", difficulty: "advanced", tags: ["genai", "agents"] },
  { id: "diffusion-models", title: "Diffusion Models for Image Generation", url: "https://www.geeksforgeeks.org/artificial-intelligence/what-are-diffusion-models/", difficulty: "advanced", tags: ["genai", "diffusion", "image-generation"] },
  { id: "text-to-image", title: "Text-to-Image Generation (Stable Diffusion, DALL-E)", url: "https://www.geeksforgeeks.org/artificial-intelligence/text-to-image-using-stable-diffusion-huggingface-model/", difficulty: "advanced", tags: ["genai", "stable-diffusion"] },
  { id: "openai-api", title: "OpenAI API (GPT-4, DALL-E, Whisper)", url: "https://www.geeksforgeeks.org/artificial-intelligence/openai-python-api/", difficulty: "intermediate", tags: ["genai", "openai", "api"] },
  { id: "huggingface-intro", title: "Hugging Face Tutorial", url: "https://www.geeksforgeeks.org/hugging-face-tutorial/", difficulty: "beginner", tags: ["genai", "huggingface"] },
  { id: "genai-evaluation", title: "Evaluating GenAI Models", url: "https://www.geeksforgeeks.org/artificial-intelligence/how-to-evaluate-llms/", difficulty: "advanced", tags: ["genai", "evaluation"] },

  // ── MLOps & Deployment ──
  { id: "docker-intro", title: "Docker for ML Engineers", url: "https://www.geeksforgeeks.org/docker-tutorial/", difficulty: "intermediate", tags: ["mlops", "docker", "containers"] },
  { id: "model-deployment", title: "Model Deployment (Flask, FastAPI)", url: "https://www.geeksforgeeks.org/deploy-machine-learning-model-using-flask/", difficulty: "intermediate", tags: ["mlops", "deployment"] },
  { id: "fastapi-tutorial", title: "FastAPI for ML Model Serving", url: "https://www.geeksforgeeks.org/python/fastapi-tutorial/", difficulty: "intermediate", tags: ["mlops", "fastapi", "api"] },
  { id: "mlflow-tutorial", title: "MLflow for Experiment Tracking", url: "https://www.geeksforgeeks.org/machine-learning/machine-learning-with-mlflow/", difficulty: "intermediate", tags: ["mlops", "mlflow", "tracking"] },
  { id: "kubernetes-ml", title: "Kubernetes for ML", url: "https://www.geeksforgeeks.org/kubernetes-tutorial/", difficulty: "advanced", tags: ["mlops", "kubernetes"] },
  { id: "model-monitoring", title: "ML Model Monitoring", url: "https://www.geeksforgeeks.org/machine-learning/model-management-in-mlops/", difficulty: "advanced", tags: ["mlops", "monitoring"] },
  { id: "ci-cd-ml", title: "CI/CD for Machine Learning", url: "https://www.geeksforgeeks.org/machine-learning/continuous-integration-and-continuous-deployment-ci-cd-in-mlops/", difficulty: "advanced", tags: ["mlops", "cicd"] },
  { id: "aws-ml-services", title: "AWS ML Services (SageMaker, Lambda)", url: "https://www.geeksforgeeks.org/machine-learning/what-is-sagemaker-in-aws/", difficulty: "intermediate", tags: ["mlops", "aws", "cloud"] },

  // ── Data Engineering ──
  { id: "data-engineering-intro", title: "Introduction to Data Engineering", url: "https://www.geeksforgeeks.org/data-engineering/data-engineering-101/", difficulty: "beginner", tags: ["data-eng", "introduction"] },
  { id: "etl-pipelines", title: "ETL Pipelines", url: "https://www.geeksforgeeks.org/data-analysis/what-is-etl-extract-transform-load/", difficulty: "intermediate", tags: ["data-eng", "etl"] },
  { id: "apache-spark", title: "Apache Spark for Big Data", url: "https://www.geeksforgeeks.org/big-data/introduction-pyspark-distributed-computing-apache-spark/", difficulty: "intermediate", tags: ["data-eng", "spark", "big-data"] },
  { id: "data-lakes", title: "Data Lakes & Warehouses", url: "https://www.geeksforgeeks.org/dbms/difference-between-data-lake-and-data-warehouse/", difficulty: "intermediate", tags: ["data-eng", "data-lake"] },
  { id: "streaming-data", title: "Real-Time Data Streaming (Kafka)", url: "https://www.geeksforgeeks.org/apache-kafka/apache-kafka/", difficulty: "advanced", tags: ["data-eng", "kafka", "streaming"] },
  { id: "airflow-tutorial", title: "Apache Airflow for Orchestration", url: "https://www.geeksforgeeks.org/devops/what-is-apache-airflow/", difficulty: "intermediate", tags: ["data-eng", "airflow", "orchestration"] },
  { id: "data-quality", title: "Data Quality & Validation", url: "https://www.geeksforgeeks.org/software-testing/data-quality-testing-in-etl-testing/", difficulty: "intermediate", tags: ["data-eng", "quality"] },

  // ── DSA for ML Interviews ──
  { id: "arrays-strings", title: "Arrays & Strings", url: "https://www.geeksforgeeks.org/array-data-structure/", difficulty: "beginner", tags: ["dsa", "arrays"] },
  { id: "linked-lists", title: "Linked Lists", url: "https://www.geeksforgeeks.org/data-structures/linked-list/", difficulty: "beginner", tags: ["dsa", "linked-list"] },
  { id: "trees-graphs", title: "Trees & Graphs", url: "https://www.geeksforgeeks.org/tree-data-structure/", difficulty: "intermediate", tags: ["dsa", "trees", "graphs"] },
  { id: "sorting-algorithms", title: "Sorting Algorithms", url: "https://www.geeksforgeeks.org/sorting-algorithms/", difficulty: "beginner", tags: ["dsa", "sorting"] },
  { id: "dynamic-programming", title: "Dynamic Programming", url: "https://www.geeksforgeeks.org/dynamic-programming/", difficulty: "advanced", tags: ["dsa", "dp"] },
  { id: "hashing", title: "Hashing & Hash Maps", url: "https://www.geeksforgeeks.org/hashing-data-structure/", difficulty: "beginner", tags: ["dsa", "hashing"] },
  { id: "binary-search", title: "Binary Search", url: "https://www.geeksforgeeks.org/binary-search/", difficulty: "beginner", tags: ["dsa", "searching"] },
  { id: "stacks-queues", title: "Stacks & Queues", url: "https://www.geeksforgeeks.org/stack-data-structure/", difficulty: "beginner", tags: ["dsa", "stack", "queue"] },
  { id: "ml-interview-prep", title: "ML Interview Questions", url: "https://www.geeksforgeeks.org/machine-learning/machine-learning-interview-questions/", difficulty: "advanced", tags: ["dsa", "interview"] },
  { id: "system-design-ml", title: "System Design for ML Engineers", url: "https://www.geeksforgeeks.org/system-design/top-10-system-design-interview-questions-and-answers/", difficulty: "advanced", tags: ["dsa", "system-design"] },
];

export function getSubjectsByCategorySimple(categoryId: string): LearningSubject[] {
  const prefixMap: Record<string, string[]> = {
    "python-fundamentals": ["python-basics", "python-oops", "python-data", "python-file", "python-iterators", "python-decorators", "python-exception", "python-regex"],
    "math-foundation": ["linear-algebra", "matrices", "eigenvalues", "calculus", "gradient-descent", "probability", "distributions", "bayesian", "hypothesis"],
    "data-wrangling": ["numpy", "pandas", "eda-python", "sql-basics", "sql-advanced", "handling-missing"],
    "data-visualization": ["matplotlib", "seaborn", "plotly", "distribution", "correlation"],
    "ml-algorithms": ["ml-introduction", "supervised", "unsupervised", "linear-regression", "logistic-regression", "decision-trees", "random-forest", "gradient-boosting", "svm", "knn", "naive-bayes", "kmeans", "hierarchical", "dbscan", "pca", "reinforcement", "apriori"],
    "ml-practice": ["data-preprocessing", "feature-engineering", "feature-selection", "cross-validation", "hyperparameter", "ml-metrics", "bias-variance", "overfitting", "regularization", "ml-pipeline"],
    "deep-learning": ["dl-introduction", "neural-networks", "activation", "backpropagation", "cnn-intro", "rnn-intro", "lstm", "tensorflow", "keras", "pytorch", "transfer", "autoencoders", "gan"],
    "nlp": ["nlp-introduction", "text-preprocessing", "tfidf", "word-embeddings", "text-classification", "named-entity", "pos-tagging", "topic-modeling", "spacy", "nltk", "text-similarity"],
    "computer-vision": ["cv-introduction", "opencv-tutorial", "image-preprocessing", "edge-detection", "object-detection", "image-segmentation", "face-detection", "yolo", "optical-character"],
    "transformers": ["attention-mechanism", "transformer-intro", "bert-intro", "gpt-intro", "t5-model", "vision-transformer", "encoder-decoder"],
    "genai": ["genai-introduction", "llm-intro", "prompt-engineering", "rag", "fine-tuning-llm", "langchain", "llamaindex", "vector-databases", "ai-agents", "diffusion", "text-to-image", "openai-api", "huggingface", "genai-evaluation"],
    "mlops": ["docker-intro", "model-deployment", "fastapi", "mlflow", "kubernetes-ml", "model-monitoring", "ci-cd", "aws-ml"],
    "data-engineering": ["data-engineering", "etl-pipelines", "apache-spark", "data-lakes", "streaming", "airflow", "data-quality"],
    "dsa": ["arrays-strings", "linked-lists", "trees-graphs", "sorting-algorithms", "dynamic-programming", "hashing", "binary-search", "stacks-queues", "ml-interview", "system-design"],
  };

  const prefixes = prefixMap[categoryId] || [];
  return learningSubjects.filter((s) => prefixes.some((p) => s.id.startsWith(p)));
}

export function searchSubjects(query: string): LearningSubject[] {
  const q = query.toLowerCase();
  return learningSubjects.filter(
    (s) =>
      s.title.toLowerCase().includes(q) ||
      s.tags.some((t) => t.includes(q))
  );
}
