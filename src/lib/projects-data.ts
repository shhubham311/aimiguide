export interface MLProject {
  id: string;
  title: string;
  url: string;
  difficulty: "beginner" | "intermediate" | "advanced";
  category: string;
  tags: string[];
}

export interface ProjectCategory {
  id: string;
  name: string;
  icon: string;
  color: string;
  glowColor: string;
  bgAccent: string;
  description: string;
}

export const projectCategories: ProjectCategory[] = [
  {
    id: "price-stock",
    name: "Price & Stock Prediction",
    icon: "TrendingUp",
    color: "text-emerald-400",
    glowColor: "shadow-emerald-500/20",
    bgAccent: "bg-emerald-500/10",
    description: "Predict prices, stocks, and financial metrics using ML models",
  },
  {
    id: "nlp-sentiment",
    name: "NLP & Sentiment Analysis",
    icon: "MessageSquare",
    color: "text-violet-400",
    glowColor: "shadow-violet-500/20",
    bgAccent: "bg-violet-500/10",
    description: "Text processing, sentiment analysis, spam detection, and BERT fine-tuning",
  },
  {
    id: "image-vision",
    name: "Image & Computer Vision",
    icon: "Eye",
    color: "text-cyan-400",
    glowColor: "shadow-cyan-500/20",
    bgAccent: "bg-cyan-500/10",
    description: "Classification, detection, segmentation, and image generation with CNNs",
  },
  {
    id: "health-medical",
    name: "Health & Medical",
    icon: "Heart",
    color: "text-rose-400",
    glowColor: "shadow-rose-500/20",
    bgAccent: "bg-rose-500/10",
    description: "Disease prediction, cancer detection, and medical diagnosis with ML/DL",
  },
  {
    id: "recommendation",
    name: "Recommendation Systems",
    icon: "ThumbsUp",
    color: "text-amber-400",
    glowColor: "shadow-amber-500/20",
    bgAccent: "bg-amber-500/10",
    description: "Movie, music, and content recommendation engines",
  },
  {
    id: "classification-ml",
    name: "Classification & Prediction",
    icon: "BrainCircuit",
    color: "text-blue-400",
    glowColor: "shadow-blue-500/20",
    bgAccent: "bg-blue-500/10",
    description: "Fraud detection, customer analysis, loan approval, and classification tasks",
  },
  {
    id: "deep-learning",
    name: "Deep Learning Projects",
    icon: "Cpu",
    color: "text-orange-400",
    glowColor: "shadow-orange-500/20",
    bgAccent: "bg-orange-500/10",
    description: "CNNs, RNNs, transfer learning, and neural network based projects",
  },
  {
    id: "detection-security",
    name: "Detection & Security",
    icon: "Shield",
    color: "text-red-400",
    glowColor: "shadow-red-500/20",
    bgAccent: "bg-red-500/10",
    description: "Intrusion detection, face mask, license plate, and security systems",
  },
];

export const mlProjects: MLProject[] = [
  // ── Price & Stock Prediction ──
  { id: "stock-price-prediction", title: "Stock Price Prediction", url: "https://www.geeksforgeeks.org/machine-learning/stock-price-prediction-using-machine-learning-in-python/", difficulty: "intermediate", category: "price-stock", tags: ["finance", "regression", "time-series"] },
  { id: "stock-price-tensorflow", title: "Stock Price Prediction using TensorFlow", url: "https://www.geeksforgeeks.org/nlp/stock-price-prediction-project-using-tensorflow/", difficulty: "advanced", category: "price-stock", tags: ["deep-learning", "lstm"] },
  { id: "microsoft-stock-prediction", title: "Microsoft Stock Price Prediction", url: "https://www.geeksforgeeks.org/machine-learning/microsoft-stock-price-prediction-with-machine-learning/", difficulty: "intermediate", category: "price-stock", tags: ["finance", "regression"] },
  { id: "stock-svm-prediction", title: "Stock Price Direction using SVM", url: "https://www.geeksforgeeks.org/machine-learning/predicting-stock-price-direction-using-support-vector-machines/", difficulty: "advanced", category: "price-stock", tags: ["svm", "classification"] },
  { id: "share-price-prophet", title: "Share Price Forecasting with Prophet", url: "https://www.geeksforgeeks.org/machine-learning/share-price-forecasting-using-facebook-prophet/", difficulty: "advanced", category: "price-stock", tags: ["prophet", "time-series"] },
  { id: "bitcoin-prediction", title: "Bitcoin Price Prediction", url: "https://www.geeksforgeeks.org/machine-learning/bitcoin-price-prediction-using-machine-learning-in-python/", difficulty: "intermediate", category: "price-stock", tags: ["crypto", "regression"] },
  { id: "dogecoin-prediction", title: "Dogecoin Price Prediction", url: "https://www.geeksforgeeks.org/machine-learning/dogecoin-price-prediction-with-machine-learning/", difficulty: "intermediate", category: "price-stock", tags: ["crypto", "regression"] },
  { id: "house-price-prediction", title: "House Price Prediction", url: "https://www.geeksforgeeks.org/machine-learning/house-price-prediction-using-machine-learning-in-python/", difficulty: "beginner", category: "price-stock", tags: ["regression", "housing"] },
  { id: "zillow-prediction", title: "Zillow Home Value (Zestimate) Prediction", url: "https://www.geeksforgeeks.org/machine-learning/zillow-home-value-zestimate-prediction-in-ml/", difficulty: "advanced", category: "price-stock", tags: ["regression", "housing"] },
  { id: "sales-forecast", title: "Sales Forecast Prediction", url: "https://www.geeksforgeeks.org/python/sales-forecast-prediction-python/", difficulty: "intermediate", category: "price-stock", tags: ["forecasting", "business"] },
  { id: "box-office-prediction", title: "Box Office Revenue Prediction", url: "https://www.geeksforgeeks.org/machine-learning/box-office-revenue-prediction-using-linear-regression-in-ml/", difficulty: "intermediate", category: "price-stock", tags: ["regression", "entertainment"] },
  { id: "medical-insurance-prediction", title: "Medical Insurance Price Prediction", url: "https://www.geeksforgeeks.org/machine-learning/medical-insurance-price-prediction-using-machine-learning-python/", difficulty: "beginner", category: "price-stock", tags: ["regression", "insurance"] },
  { id: "used-cars-analysis", title: "Analyzing Selling Price of Used Cars", url: "https://www.geeksforgeeks.org/python/analyzing-selling-price-of-used-cars-using-python/", difficulty: "beginner", category: "price-stock", tags: ["regression", "eda"] },

  // ── NLP & Sentiment Analysis ──
  { id: "twitter-sentiment", title: "Twitter Sentiment Analysis", url: "https://www.geeksforgeeks.org/python/twitter-sentiment-analysis-using-python/", difficulty: "beginner", category: "nlp-sentiment", tags: ["nlp", "sentiment"] },
  { id: "facebook-sentiment", title: "Facebook Sentiment Analysis", url: "https://www.geeksforgeeks.org/nlp/facebook-sentiment-analysis-using-python/", difficulty: "beginner", category: "nlp-sentiment", tags: ["nlp", "sentiment"] },
  { id: "flipkart-sentiment", title: "Flipkart Reviews Sentiment Analysis", url: "https://www.geeksforgeeks.org/nlp/flipkart-reviews-sentiment-analysis-using-python/", difficulty: "beginner", category: "nlp-sentiment", tags: ["nlp", "reviews"] },
  { id: "restaurant-reviews", title: "Analysis of Restaurant Reviews", url: "https://www.geeksforgeeks.org/nlp/python-nlp-analysis-of-restaurant-reviews/", difficulty: "beginner", category: "nlp-sentiment", tags: ["nlp", "reviews"] },
  { id: "restaurant-sqlite", title: "Restaurant Review Analysis (NLP + SQLite)", url: "https://www.geeksforgeeks.org/nlp/restaurant-review-analysis-using-nlp-and-sqlite/", difficulty: "intermediate", category: "nlp-sentiment", tags: ["nlp", "sql"] },
  { id: "sms-spam-detection", title: "SMS Spam Detection", url: "https://www.geeksforgeeks.org/deep-learning/sms-spam-detection-using-tensorflow-in-python/", difficulty: "intermediate", category: "nlp-sentiment", tags: ["nlp", "classification", "tensorflow"] },
  { id: "spam-emails-tensorflow", title: "Detecting Spam Emails with TensorFlow", url: "https://www.geeksforgeeks.org/nlp/detecting-spam-emails-using-tensorflow-in-python/", difficulty: "intermediate", category: "nlp-sentiment", tags: ["nlp", "classification", "tensorflow"] },
  { id: "text-classification-nb", title: "Classification of Text Documents (Naive Bayes)", url: "https://www.geeksforgeeks.org/machine-learning/classification-of-text-documents-using-the-approach-of-naive-bayes/", difficulty: "beginner", category: "nlp-sentiment", tags: ["nlp", "naive-bayes"] },
  { id: "fake-news-detection", title: "Fake News Detection", url: "https://www.geeksforgeeks.org/machine-learning/fake-news-detection-using-machine-learning/", difficulty: "intermediate", category: "nlp-sentiment", tags: ["nlp", "classification"] },
  { id: "fake-news-tensorflow", title: "Fake News Detection with TensorFlow", url: "https://www.geeksforgeeks.org/nlp/fake-news-detection-model-using-tensorflow-in-python/", difficulty: "advanced", category: "nlp-sentiment", tags: ["nlp", "tensorflow", "deep-learning"] },
  { id: "hate-speech-detection", title: "Hate Speech Detection", url: "https://www.geeksforgeeks.org/deep-learning/hate-speech-detection-using-deep-learning/", difficulty: "advanced", category: "nlp-sentiment", tags: ["nlp", "deep-learning", "classification"] },
  { id: "bert-sentiment", title: "Fine-tuning BERT for Sentiment Analysis", url: "https://www.geeksforgeeks.org/nlp/fine-tuning-bert-model-for-sentiment-analysis/", difficulty: "advanced", category: "nlp-sentiment", tags: ["bert", "transformers"] },
  { id: "bert-sentiment-classification", title: "Sentiment Classification Using BERT", url: "https://www.geeksforgeeks.org/nlp/sentiment-classification-using-bert/", difficulty: "advanced", category: "nlp-sentiment", tags: ["bert", "transformers"] },
  { id: "next-sentence-bert", title: "Next Sentence Prediction using BERT", url: "https://www.geeksforgeeks.org/machine-learning/next-sentence-prediction-using-bert/", difficulty: "advanced", category: "nlp-sentiment", tags: ["bert", "nlp"] },
  { id: "sentiment-rnn", title: "Sentiment Analysis with RNN", url: "https://www.geeksforgeeks.org/python/sentiment-analysis-with-an-recurrent-neural-networks-rnn/", difficulty: "advanced", category: "nlp-sentiment", tags: ["rnn", "deep-learning"] },
  { id: "speech-recognition", title: "Speech Recognition", url: "https://www.geeksforgeeks.org/python/speech-recognition-in-python-using-google-speech-api/", difficulty: "intermediate", category: "nlp-sentiment", tags: ["speech", "audio"] },

  // ── Image & Computer Vision ──
  { id: "cifar-10-classification", title: "CIFAR-10 Image Classification", url: "https://www.geeksforgeeks.org/deep-learning/cifar-10-image-classification-in-tensorflow/", difficulty: "intermediate", category: "image-vision", tags: ["cnn", "tensorflow", "classification"] },
  { id: "cat-dog-cnn", title: "Cat & Dog Classification using CNN", url: "https://www.geeksforgeeks.org/deep-learning/cat-dog-classification-using-convolutional-neural-network-in-python/", difficulty: "intermediate", category: "image-vision", tags: ["cnn", "classification"] },
  { id: "dog-breed-classification", title: "Dog Breed Classification (Transfer Learning)", url: "https://www.geeksforgeeks.org/deep-learning/dog-breed-classification-using-transfer-learning/", difficulty: "advanced", category: "image-vision", tags: ["transfer-learning", "classification"] },
  { id: "multiclass-image", title: "Multiclass Image Classification", url: "https://www.geeksforgeeks.org/deep-learning/multiclass-image-classification-using-transfer-learning/", difficulty: "advanced", category: "image-vision", tags: ["transfer-learning", "classification"] },
  { id: "flower-recognition", title: "Flower Recognition with CNN", url: "https://www.geeksforgeeks.org/deep-learning/flower-recognition-using-convolutional-neural-network/", difficulty: "intermediate", category: "image-vision", tags: ["cnn", "classification"] },
  { id: "handwritten-digits-tf", title: "Classify Handwritten Digits (TensorFlow)", url: "https://www.geeksforgeeks.org/python/python-classifying-handwritten-digits-with-tensorflow/", difficulty: "beginner", category: "image-vision", tags: ["tensorflow", "mnist"] },
  { id: "handwritten-digits-sklearn", title: "Recognizing Handwritten Digits (scikit-learn)", url: "https://www.geeksforgeeks.org/machine-learning/recognizing-handwritten-digits-in-scikit-learn/", difficulty: "beginner", category: "image-vision", tags: ["sklearn", "mnist"] },
  { id: "handwritten-digits-nn", title: "Handwritten Digit Recognition (Neural Network)", url: "https://www.geeksforgeeks.org/machine-learning/handwritten-digit-recognition-using-neural-network/", difficulty: "intermediate", category: "image-vision", tags: ["neural-network", "mnist"] },
  { id: "handwritten-logistic-pytorch", title: "Handwritten Digits with Logistic Regression (PyTorch)", url: "https://www.geeksforgeeks.org/machine-learning/identifying-handwritten-digits-using-logistic-regression-pytorch/", difficulty: "intermediate", category: "image-vision", tags: ["pytorch", "logistic-regression"] },
  { id: "ocr-handwritten", title: "OCR of Handwritten Digits (OpenCV)", url: "https://www.geeksforgeeks.org/machine-learning/ocr-of-handwritten-digits-opencv/", difficulty: "intermediate", category: "image-vision", tags: ["opencv", "ocr"] },
  { id: "cartooning-image", title: "Cartooning an Image (OpenCV)", url: "https://www.geeksforgeeks.org/blogs/cartooning-an-image-using-opencv-python/", difficulty: "beginner", category: "image-vision", tags: ["opencv", "image-processing"] },
  { id: "face-counting", title: "Count Number of Faces", url: "https://www.geeksforgeeks.org/python/count-number-of-faces-using-python-opencv/", difficulty: "beginner", category: "image-vision", tags: ["opencv", "face-detection"] },
  { id: "text-detection-ocr", title: "Text Detection and Extraction (OCR)", url: "https://www.geeksforgeeks.org/python/text-detection-and-extraction-using-opencv-and-ocr/", difficulty: "intermediate", category: "image-vision", tags: ["opencv", "ocr"] },
  { id: "bw-colorization", title: "Black & White Image Colorization", url: "https://www.geeksforgeeks.org/computer-vision/black-and-white-image-colorization-with-opencv-and-deep-learning/", difficulty: "advanced", category: "image-vision", tags: ["deep-learning", "opencv"] },
  { id: "image-caption-generator", title: "Image Caption Generator", url: "https://www.geeksforgeeks.org/deep-learning/image-caption-generator-using-deep-learning-on-flickr8k-dataset/", difficulty: "advanced", category: "image-vision", tags: ["deep-learning", "nlp", "caption"] },
  { id: "image-segmentation", title: "Image Segmentation (TensorFlow)", url: "https://www.geeksforgeeks.org/deep-learning/image-segmentation-using-tensorflow/", difficulty: "advanced", category: "image-vision", tags: ["tensorflow", "segmentation"] },
  { id: "traffic-signs-cnn", title: "Traffic Signs Recognition (CNN + Keras)", url: "https://www.geeksforgeeks.org/deep-learning/traffic-signs-recognition-using-cnn-and-keras-in-python/", difficulty: "advanced", category: "image-vision", tags: ["cnn", "keras", "classification"] },
  { id: "fuel-efficiency", title: "Predict Fuel Efficiency (TensorFlow)", url: "https://www.geeksforgeeks.org/deep-learning/predict-fuel-efficiency-using-tensorflow-in-python/", difficulty: "intermediate", category: "image-vision", tags: ["tensorflow", "regression"] },

  // ── Health & Medical ──
  { id: "disease-prediction", title: "Disease Prediction", url: "https://www.geeksforgeeks.org/machine-learning/disease-prediction-using-machine-learning/", difficulty: "intermediate", category: "health-medical", tags: ["classification", "healthcare"] },
  { id: "heart-disease-lr", title: "Heart Disease Prediction (Logistic Regression)", url: "https://www.geeksforgeeks.org/machine-learning/ml-heart-disease-prediction-using-logistic-regression/", difficulty: "intermediate", category: "health-medical", tags: ["logistic-regression", "healthcare"] },
  { id: "heart-disease-ann", title: "Heart Disease Prediction (ANN)", url: "https://www.geeksforgeeks.org/deep-learning/heart-disease-prediction-using-ann/", difficulty: "advanced", category: "health-medical", tags: ["ann", "deep-learning", "healthcare"] },
  { id: "parkinson-prediction", title: "Parkinson's Disease Prediction", url: "https://www.geeksforgeeks.org/machine-learning/parkinson-disease-prediction-using-machine-learning-python/", difficulty: "intermediate", category: "health-medical", tags: ["classification", "healthcare"] },
  { id: "autism-prediction", title: "Autism Prediction", url: "https://www.geeksforgeeks.org/machine-learning/autism-prediction-using-machine-learning/", difficulty: "intermediate", category: "health-medical", tags: ["classification", "healthcare"] },
  { id: "cancer-cell-classification", title: "Cancer Cell Classification", url: "https://www.geeksforgeeks.org/machine-learning/ml-cancer-cell-classification-using-scikit-learn/", difficulty: "intermediate", category: "health-medical", tags: ["sklearn", "classification", "healthcare"] },
  { id: "breast-cancer-knn", title: "Breast Cancer Diagnosis (KNN)", url: "https://www.geeksforgeeks.org/machine-learning/ml-kaggle-breast-cancer-wisconsin-diagnosis-using-knn/", difficulty: "intermediate", category: "health-medical", tags: ["knn", "healthcare"] },
  { id: "breast-cancer-lr", title: "Breast Cancer Diagnosis (Logistic Regression)", url: "https://www.geeksforgeeks.org/machine-learning/ml-kaggle-breast-cancer-wisconsin-diagnosis-using-logistic-regression/", difficulty: "intermediate", category: "health-medical", tags: ["logistic-regression", "healthcare"] },
  { id: "skin-cancer-detection", title: "Skin Cancer Detection (TensorFlow)", url: "https://www.geeksforgeeks.org/deep-learning/skin-cancer-detection-using-tensorflow/", difficulty: "advanced", category: "health-medical", tags: ["tensorflow", "cnn", "healthcare"] },
  { id: "lung-cancer-cnn", title: "Lung Cancer Detection (CNN)", url: "https://www.geeksforgeeks.org/deep-learning/lung-cancer-detection-using-convolutional-neural-network-cnn/", difficulty: "advanced", category: "health-medical", tags: ["cnn", "healthcare"] },
  { id: "lung-cancer-transfer", title: "Lung Cancer Detection (Transfer Learning)", url: "https://www.geeksforgeeks.org/deep-learning/lung-cancer-detection-using-transfer-learning/", difficulty: "advanced", category: "health-medical", tags: ["transfer-learning", "healthcare"] },
  { id: "pneumonia-detection", title: "Pneumonia Detection (Deep Learning)", url: "https://www.geeksforgeeks.org/deep-learning/pneumonia-detection-using-deep-learning/", difficulty: "advanced", category: "health-medical", tags: ["deep-learning", "healthcare"] },
  { id: "covid-chest-xray", title: "Detecting COVID-19 with Chest X-ray", url: "https://www.geeksforgeeks.org/machine-learning/detecting-covid-19-with-chest-x-ray/", difficulty: "advanced", category: "health-medical", tags: ["classification", "healthcare"] },
  { id: "covid-cnn", title: "Detecting COVID-19 (CNN + X-Ray)", url: "https://www.geeksforgeeks.org/deep-learning/detecting-covid-19-from-chest-x-ray-images-using-cnn/", difficulty: "advanced", category: "health-medical", tags: ["cnn", "healthcare"] },
  { id: "calories-burnt", title: "Calories Burnt Prediction", url: "https://www.geeksforgeeks.org/machine-learning/calories-burnt-prediction-using-machine-learning/", difficulty: "intermediate", category: "health-medical", tags: ["regression", "fitness"] },
  { id: "air-quality-prediction", title: "Predicting Air Quality Index", url: "https://www.geeksforgeeks.org/python/predicting-air-quality-index-using-python/", difficulty: "intermediate", category: "health-medical", tags: ["regression", "environment"] },
  { id: "air-quality-nn", title: "Air Quality Prediction (Neural Networks)", url: "https://www.geeksforgeeks.org/deep-learning/predicting-air-quality-with-neural-networks/", difficulty: "advanced", category: "health-medical", tags: ["neural-network", "environment"] },

  // ── Recommendation Systems ──
  { id: "movie-recommender", title: "Movie Recommender System", url: "https://www.geeksforgeeks.org/machine-learning/python-implementation-of-movie-recommender-system/", difficulty: "intermediate", category: "recommendation", tags: ["collaborative-filtering"] },
  { id: "movie-emotion-recommender", title: "Movie Recommendation based on Emotion", url: "https://www.geeksforgeeks.org/python/movie-recommendation-based-emotion-python/", difficulty: "intermediate", category: "recommendation", tags: ["emotion", "nlp"] },
  { id: "music-recommender", title: "Music Recommendation System", url: "https://www.geeksforgeeks.org/machine-learning/music-recommendation-system-using-machine-learning/", difficulty: "intermediate", category: "recommendation", tags: ["collaborative-filtering"] },
  { id: "ted-talks-recommender", title: "TED Talks Recommendation System", url: "https://www.geeksforgeeks.org/machine-learning/ted-talks-recommendation-system-with-machine-learning/", difficulty: "intermediate", category: "recommendation", tags: ["collaborative-filtering"] },

  // ── Classification & Prediction ──
  { id: "credit-card-fraud", title: "Credit Card Fraud Detection", url: "https://www.geeksforgeeks.org/machine-learning/ml-credit-card-fraud-detection/", difficulty: "intermediate", category: "classification-ml", tags: ["classification", "fraud"] },
  { id: "online-payment-fraud", title: "Online Payment Fraud Detection", url: "https://www.geeksforgeeks.org/machine-learning/online-payment-fraud-detection-using-machine-learning-in-python/", difficulty: "intermediate", category: "classification-ml", tags: ["classification", "fraud"] },
  { id: "customer-churn", title: "Customer Churn Analysis & Prediction", url: "https://www.geeksforgeeks.org/machine-learning/python-customer-churn-analysis-prediction/", difficulty: "intermediate", category: "classification-ml", tags: ["classification", "business"] },
  { id: "customer-segmentation", title: "Customer Segmentation (Unsupervised ML)", url: "https://www.geeksforgeeks.org/machine-learning/customer-segmentation-using-unsupervised-machine-learning-in-python/", difficulty: "intermediate", category: "classification-ml", tags: ["clustering", "business"] },
  { id: "click-through-rate", title: "Click-Through Rate Prediction", url: "https://www.geeksforgeeks.org/machine-learning/click-through-rate-prediction/", difficulty: "advanced", category: "classification-ml", tags: ["classification", "advertising"] },
  { id: "loan-approval", title: "Loan Approval Prediction (Multiple Models)", url: "https://www.geeksforgeeks.org/machine-learning/loan-approval-prediction-using-machine-learning/", difficulty: "intermediate", category: "classification-ml", tags: ["classification", "finance"] },
  { id: "loan-eligibility-svm", title: "Loan Eligibility Prediction (SVM)", url: "https://www.geeksforgeeks.org/machine-learning/loan-eligibility-prediction-using-machine-learning-models-in-python/", difficulty: "intermediate", category: "classification-ml", tags: ["svm", "finance"] },
  { id: "titanic-survival", title: "Titanic Survival Prediction", url: "https://www.geeksforgeeks.org/machine-learning/titanic-survival-prediction-using-ml/", difficulty: "beginner", category: "classification-ml", tags: ["classification", "classic"] },
  { id: "spaceship-titanic", title: "Spaceship Titanic Project", url: "https://www.geeksforgeeks.org/machine-learning/spaceship-titanic-project-using-machine-learning-python/", difficulty: "beginner", category: "classification-ml", tags: ["classification", "kaggle"] },
  { id: "wine-quality", title: "Wine Quality Prediction", url: "https://www.geeksforgeeks.org/machine-learning/wine-quality-prediction-machine-learning/", difficulty: "beginner", category: "classification-ml", tags: ["classification", "regression"] },
  { id: "wine-type-dl", title: "Wine Type Prediction (Deep Learning)", url: "https://www.geeksforgeeks.org/deep-learning/prediction-of-wine-type-using-deep-learning/", difficulty: "intermediate", category: "classification-ml", tags: ["deep-learning", "classification"] },
  { id: "demand-forecasting", title: "Inventory Demand Forecasting", url: "https://www.geeksforgeeks.org/machine-learning/inventory-demand-forecasting-using-machine-learning-python/", difficulty: "intermediate", category: "classification-ml", tags: ["forecasting", "business"] },
  { id: "ola-bike-forecast", title: "Ola Bike Ride Request Forecast", url: "https://www.geeksforgeeks.org/machine-learning/ola-bike-ride-request-forecast-using-ml/", difficulty: "intermediate", category: "classification-ml", tags: ["forecasting", "business"] },
  { id: "vehicle-count", title: "Vehicle Count Prediction (Sensor Data)", url: "https://www.geeksforgeeks.org/dsa/vehicle-count-prediction-from-sensor-data/", difficulty: "intermediate", category: "classification-ml", tags: ["iot", "regression"] },
  { id: "waiters-tip", title: "Waiter's Tip Prediction", url: "https://www.geeksforgeeks.org/machine-learning/waiters-tip-prediction-using-machine-learning/", difficulty: "beginner", category: "classification-ml", tags: ["regression"] },
  { id: "rainfall-prediction", title: "Rainfall Prediction", url: "https://www.geeksforgeeks.org/machine-learning/rainfall-prediction-using-machine-learning-python/", difficulty: "intermediate", category: "classification-ml", tags: ["classification", "weather"] },

  // ── Deep Learning ──
  { id: "ipl-score-dl", title: "IPL Score Prediction (Deep Learning)", url: "https://www.geeksforgeeks.org/deep-learning/ipl-score-prediction-using-deep-learning/", difficulty: "advanced", category: "deep-learning", tags: ["deep-learning", "sports"] },
  { id: "sequential-model", title: "Sequential Model (Abalone Dataset)", url: "https://www.geeksforgeeks.org/deep-learning/how-can-tensorflow-be-used-with-abalone-dataset-to-build-a-sequential-model/", difficulty: "intermediate", category: "deep-learning", tags: ["tensorflow", "sequential"] },
  { id: "face-landmarks", title: "Face and Hand Landmarks Detection", url: "https://www.geeksforgeeks.org/machine-learning/face-and-hand-landmarks-detection-using-python-mediapipe-opencv/", difficulty: "advanced", category: "deep-learning", tags: ["mediapipe", "opencv"] },
  { id: "boston-housing", title: "Boston Housing Kaggle (Linear Regression)", url: "https://www.geeksforgeeks.org/machine-learning/ml-boston-housing-kaggle-challenge-with-linear-regression/", difficulty: "beginner", category: "deep-learning", tags: ["regression", "kaggle"] },
  { id: "resnet-deep-learning", title: "Residual Networks (ResNet)", url: "https://www.geeksforgeeks.org/deep-learning/residual-networks-resnet-deep-learning/", difficulty: "advanced", category: "deep-learning", tags: ["cnn", "resnet"] },
  { id: "emotion-recommender", title: "Book Recommendation based on Emotion", url: "https://www.geeksforgeeks.org/python/emotion-based-book-recommendation-system/", difficulty: "intermediate", category: "deep-learning", tags: ["nlp", "emotion", "recommendation"] },
  { id: "gender-age-detection", title: "Gender & Age Detection (OpenCV)", url: "https://www.geeksforgeeks.org/python/gender-and-age-detection-using-opencv/", difficulty: "intermediate", category: "deep-learning", tags: ["opencv", "detection"] },
  { id: "color-detection", title: "Color Detection (OpenCV)", url: "https://www.geeksforgeeks.org/python/color-detection-using-opencv-python/", difficulty: "beginner", category: "deep-learning", tags: ["opencv", "image-processing"] },
  { id: "object-count", title: "Count Number of Objects (OpenCV)", url: "https://www.geeksforgeeks.org/computer-vision/count-number-of-object-using-python-opencv/", difficulty: "beginner", category: "deep-learning", tags: ["opencv", "detection"] },

  // ── Detection & Security ──
  { id: "facemask-detection", title: "FaceMask Detection (TensorFlow)", url: "https://www.geeksforgeeks.org/deep-learning/facemask-detection-using-tensorflow-in-python/", difficulty: "intermediate", category: "detection-security", tags: ["cnn", "detection"] },
  { id: "age-detection", title: "Age Detection (Deep Learning + OpenCV)", url: "https://www.geeksforgeeks.org/computer-vision/age-detection-using-deep-learning-in-opencv/", difficulty: "advanced", category: "detection-security", tags: ["deep-learning", "opencv"] },
  { id: "license-plate-recognition", title: "License Plate Recognition", url: "https://www.geeksforgeeks.org/machine-learning/license-plate-recognition-with-opencv-and-tesseract-ocr/", difficulty: "advanced", category: "detection-security", tags: ["opencv", "ocr"] },
  { id: "license-plate-video", title: "Car License Plate Detection (Real-time Video)", url: "https://www.geeksforgeeks.org/python/detect-and-recognize-car-license-plate-from-a-video-in-real-time/", difficulty: "advanced", category: "detection-security", tags: ["opencv", "video", "real-time"] },
  { id: "intrusion-detection", title: "Intrusion Detection System", url: "https://www.geeksforgeeks.org/machine-learning/intrusion-detection-system-using-machine-learning-algorithms/", difficulty: "advanced", category: "detection-security", tags: ["security", "classification"] },
  { id: "human-activity", title: "Human Activity Recognition (Deep Learning)", url: "https://www.geeksforgeeks.org/deep-learning/human-activity-recognition-using-deep-learning-model/", difficulty: "advanced", category: "detection-security", tags: ["deep-learning", "sensors"] },
  { id: "autocorrector-nlp", title: "Autocorrector using NLP", url: "https://www.geeksforgeeks.org/nlp/autocorrector-feature-using-nlp-in-python/", difficulty: "intermediate", category: "detection-security", tags: ["nlp", "text-processing"] },
  { id: "drowsiness-detection", title: "Driver Drowsiness Detection", url: "https://www.geeksforgeeks.org/python/driver-drowsiness-detection-using-opencv/", difficulty: "intermediate", category: "detection-security", tags: ["opencv", "detection", "real-time"] },
];

export function getProjectsByCategory(categoryId: string): MLProject[] {
  return mlProjects.filter((p) => p.category === categoryId);
}

export function getProjectById(projectId: string): MLProject | undefined {
  return mlProjects.find((p) => p.id === projectId);
}

export function searchProjects(query: string): MLProject[] {
  const q = query.toLowerCase();
  return mlProjects.filter(
    (p) =>
      p.title.toLowerCase().includes(q) ||
      p.tags.some((t) => t.includes(q)) ||
      p.category.toLowerCase().includes(q)
  );
}
