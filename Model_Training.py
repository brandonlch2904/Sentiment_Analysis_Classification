import time
import numpy as np
from sklearn.datasets import load_svmlight_file
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_selection import SelectKBest, chi2
from scipy.sparse import hstack
import matplotlib.pyplot as plt
import joblib
import pandas as pd  # For creating a performance comparison table

# Step 1: Load .feat Data
def load_feat_file(file_path):
    """
    Load a .feat file in LIBSVM format.
    Args:
        file_path (str): Path to the .feat file.
    Returns:
        X (sparse matrix): Feature matrix.
        y (array): Labels.
    """
    print(f"Loading data from {file_path}...")
    X, y = load_svmlight_file(file_path)
    print(f"Data loaded: {X.shape[0]} samples, {X.shape[1]} features.")
    return X, y

# Load training and testing data
X_train, y_train = load_feat_file('train/labeledBow.feat')
X_test, y_test = load_feat_file('test/labeledBow.feat')

# Map the labels to binary sentiment labels
def map_labels(y):
    """
    Map ratings to binary sentiment labels.
    Args:
        y (array): Original ratings.
    Returns:
        y_mapped (array): Mapped binary labels.
    """
    y_mapped = np.where(y <= 4, 0, 1)  # Negative: 0, Positive: 1
    return y_mapped

y_train_mapped = map_labels(y_train)
y_test_mapped = map_labels(y_test)

# Verify the mapped labels
unique_train, counts_train = np.unique(y_train_mapped, return_counts=True)
print(f"Mapped training label distribution: {dict(zip(unique_train, counts_train))}")

unique_test, counts_test = np.unique(y_test_mapped, return_counts=True)
print(f"Mapped testing label distribution: {dict(zip(unique_test, counts_test))}")

# Find the maximum feature index
max_features = max(X_train.shape[1], X_test.shape[1])

# Align the feature spaces
# Pad X_train if necessary
if X_train.shape[1] < max_features:
    from scipy.sparse import csr_matrix
    padding = csr_matrix((X_train.shape[0], max_features - X_train.shape[1]))
    X_train = hstack([X_train, padding])

# Pad X_test if necessary
if X_test.shape[1] < max_features:
    from scipy.sparse import csr_matrix
    padding = csr_matrix((X_test.shape[0], max_features - X_test.shape[1]))
    X_test = hstack([X_test, padding])

# Verify the new shapes
print(f"Aligned training data shape: {X_train.shape}")
print(f"Aligned testing data shape: {X_test.shape}")
print("---------------------------------------------------------------------------")

# Step 2: Apply TF-IDF Transformation
print("Applying TF-IDF transformation...")
tfidf_transformer = TfidfTransformer()

# Fit on training data and transform
X_train_tfidf = tfidf_transformer.fit_transform(X_train)

# Transform testing data
X_test_tfidf = tfidf_transformer.transform(X_test)

print(f"TF-IDF transformed training data shape: {X_train_tfidf.shape}")
print(f"TF-IDF transformed testing data shape: {X_test_tfidf.shape}")
print("---------------------------------------------------------------------------")

# Step 3: Select Top 10,000 Features
print("Selecting top 10,000 features using SelectKBest with chi2...")
k = 10000  # Number of top features to select
selector = SelectKBest(score_func=chi2, k=k)

# Fit selector on the training data and transform
X_train_tfidf_selected = selector.fit_transform(X_train_tfidf, y_train_mapped)

# Transform testing data using the same selector
X_test_tfidf_selected = selector.transform(X_test_tfidf)

print(f"Reduced training data shape: {X_train_tfidf_selected.shape}")
print(f"Reduced testing data shape: {X_test_tfidf_selected.shape}")
print("---------------------------------------------------------------------------")

# Save preprocessed testing data and labels
joblib.dump((X_test_tfidf_selected, y_test_mapped), 'processed_testing_data.pkl')
# Save preprocessed training data and labels
joblib.dump((X_train_tfidf_selected, y_train_mapped), 'processed_training_data.pkl')

# Step 4: Train Classifiers
# Initialize a dictionary to store models and their names
models = {
    'SGDClassifier': SGDClassifier(
        loss='hinge',               # For linear SVM
        penalty='l2',               # Regularization type
        alpha=0.0001,               # Regularization strength (inverse of C)
        learning_rate='optimal',    # Learning rate schedule
        max_iter=1000,              # Maximum iterations
        tol=1e-4,                   # Tolerance for stopping criteria
        n_iter_no_change=5,         # Early stopping after no improvement
        random_state=42,
        verbose=1
    ),
    'KNN': KNeighborsClassifier(n_neighbors=8, n_jobs=-1),
    'RandomForest': RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
}

# Dictionary to store performance metrics
performance_metrics = {
    'Model': [],
    'Accuracy': [],
    'Precision': [],
    'Recall': [],
    'F1 Score': []
}

# Train and evaluate each model
for model_name, model in models.items():
    print(f"Training {model_name}...")
    start_time = time.time()
    model.fit(X_train_tfidf_selected, y_train_mapped)
    end_time = time.time()
    training_time = end_time - start_time
    print(f"{model_name} trained in {training_time:.2f} seconds.")
    
    print(f"Predicting on the test set with {model_name}...")
    y_pred = model.predict(X_test_tfidf_selected)
    
    # Compute performance metrics
    accuracy = accuracy_score(y_test_mapped, y_pred)
    precision = precision_score(y_test_mapped, y_pred)
    recall = recall_score(y_test_mapped, y_pred)
    f1 = f1_score(y_test_mapped, y_pred)
    
    # Store the metrics
    performance_metrics['Model'].append(model_name)
    performance_metrics['Accuracy'].append(accuracy)
    performance_metrics['Precision'].append(precision)
    performance_metrics['Recall'].append(recall)
    performance_metrics['F1 Score'].append(f1)
    
    print(f"{model_name} Performance Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}\n")
    
    # Optionally, display confusion matrix for each model
    conf_matrix = confusion_matrix(y_test_mapped, y_pred)
    print(f"{model_name} Confusion Matrix:")
    print(conf_matrix)
    
    # Visualize Confusion Matrix
    disp = ConfusionMatrixDisplay(
        confusion_matrix=conf_matrix, display_labels=['Negative', 'Positive']
    )
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.show()
    print("---------------------------------------------------------------------------")

# Step 5: Compare Performance Metrics
# Create a DataFrame for better visualization
performance_df = pd.DataFrame(performance_metrics)
print("Performance Comparison of Models:")
print(performance_df)

# Save the performance metrics to a CSV file
performance_df.to_csv('models_performance_comparison.csv', index=False)
print("Performance metrics saved to 'models_performance_comparison.csv'.")

# Visualize the performance metrics using bar charts
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
plt.figure(figsize=(12, 8))
for i, metric in enumerate(metrics, 1):
    plt.subplot(2, 2, i)
    plt.bar(performance_df['Model'], performance_df[metric], color=['blue', 'green', 'red'])
    plt.ylim(0, 1)
    plt.title(metric)
    plt.ylabel(metric)
    for index, value in enumerate(performance_df[metric]):
        plt.text(index, value + 0.01, f"{value:.2f}", ha='center')
plt.tight_layout()
plt.show()

# Step 6: Save the Trained Models and Transformers
# Save all trained models
for model_name, model in models.items():
    joblib.dump(model, f'sentiment_{model_name.lower()}_model.pkl')
    print(f"{model_name} saved successfully as 'sentiment_{model_name.lower()}_model.pkl'.")

# Save the TF-IDF transformer and selector for future use
joblib.dump(tfidf_transformer, 'tfidf_transformer.pkl')
joblib.dump(selector, 'feature_selector.pkl')
print("Transformers saved successfully.")
