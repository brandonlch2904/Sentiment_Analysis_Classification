# Sentiment Analysis Project

## Overview

This project is a sentiment analysis implementation using machine learning models to classify movie reviews into positive or negative sentiments. It processes and trains on the [Large Movie Review Dataset v1.0](http://ai.stanford.edu/~amaas/data/sentiment/), a well-known dataset for benchmarking sentiment classification.

## Features

- **Dataset Handling**: Utilizes the `.feat` files (LIBSVM format) provided by the Large Movie Review Dataset.
- **Preprocessing Pipeline**:
  - Binary label mapping (positive/negative sentiment).
  - Feature alignment to ensure consistent input dimensions.
  - TF-IDF transformation for better representation of textual data.
  - Feature selection using Chi-Square to reduce dimensionality.
- **Machine Learning Models**:
  - Linear Support Vector Machine (via `SGDClassifier`).
  - K-Nearest Neighbors (KNN).
  - Random Forest Classifier.
- **Performance Evaluation**:
  - Metrics: Accuracy, Precision, Recall, F1 Score.
  - Confusion Matrix visualization for detailed model evaluation.
- **Visualization**:
  - Comparative bar charts for model performance metrics.
- **Model Saving**:
  - Pretrained models, TF-IDF transformer, and feature selector are saved for reuse.

## Prerequisites

- Python 3.8+
- Required Libraries: `numpy`, `scikit-learn`, `scipy`, `matplotlib`, `joblib`, `pandas

# Dataset Details
The project uses the Large Movie Review Dataset v1.0, which contains:

50,000 labeled reviews: 25,000 for training and 25,000 for testing, balanced evenly between positive and negative sentiments.
Additional unlabeled data for unsupervised learning (not used in this project).
The dataset’s unique properties ensure robust benchmarking:
However, the dataset is excluded in this repository and the data have been preprocessed to 2 pkt file, *processed_training_data.pkl* and *processed_testing_data.pkl*. 

No overlap between movies in training and test sets.
Binary sentiment labels: Negative (≤4 stars), Positive (≥7 stars).
For more details, refer to the original dataset's documentation or cite:
```
Maas, Andrew L., Daly, Raymond E., Pham, Peter T., Huang, Dan, Ng, Andrew Y., & Potts, Christopher. (2011). 
Learning Word Vectors for Sentiment Analysis. Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies, 142–150. 
Download Link: https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
```

# Workflow
## Step 1: 
- Load and Preprocess Data
- Loads .feat files for training and testing.
- Maps labels to binary sentiment classes.
- Aligns feature spaces and applies TF-IDF transformation.
- Selects the top 10,000 features using the Chi-Square method.
## Step 2: Train Models
- **Models trained:**
    - SGDClassifier
    - K-Nearest Neighbors
    - Random Forest
- Each model is trained on the processed features and evaluated.
## Step 3: Evaluate Models
- Metrics: Accuracy, Precision, Recall, F1 Score.
- Saves model performance to a CSV file and visualizes metrics.
## Step 4: Save Trained Models
- All trained models, transformers, and feature selectors are saved for deployment or further analysis.
## Results
- Performance metrics are saved in models_performance_comparison.csv and visualized as bar charts. Confusion matrices for each model are plotted to evaluate misclassification patterns.
