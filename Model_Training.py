import numpy as np
from sklearn.datasets import load_svmlight_file
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import cross_validate, StratifiedKFold
import matplotlib.pyplot as plt
import pandas as pd

# Step 1: Load .feat Data
def load_feat_file(file_path):
    print(f"Loading data from {file_path}...")
    X, y = load_svmlight_file(file_path)
    print(f"Data loaded: {X.shape[0]} samples, {X.shape[1]} features.")
    return X, y

# Load training data
X_train, y_train = load_feat_file('train/labeledBow.feat')

# Map the labels to binary sentiment labels
def map_labels(y):
    y_mapped = np.where(y <= 4, 0, 1)  # Negative: 0, Positive: 1
    return y_mapped

y_train_mapped = map_labels(y_train)

# Verify the mapped labels
unique_train, counts_train = np.unique(y_train_mapped, return_counts=True)
print(f"Mapped training label distribution: {dict(zip(unique_train, counts_train))}")

# Step 2: Align the feature spaces
max_features = X_train.shape[1]

# Apply TF-IDF Transformation
print("Applying TF-IDF transformation...")
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train)

print(f"TF-IDF transformed training data shape: {X_train_tfidf.shape}")
print("---------------------------------------------------------------------------")

# Step 3: Select Top 10,000 Features
print("Selecting top 10,000 features using SelectKBest with chi2...")
k = 10000  # Number of top features to select
selector = SelectKBest(score_func=chi2, k=k)
X_train_tfidf_selected = selector.fit_transform(X_train_tfidf, y_train_mapped)
print(f"Reduced training data shape: {X_train_tfidf_selected.shape}")
print("---------------------------------------------------------------------------")

# Step 4: Train Classifiers with Cross-Validation
# Initialize a dictionary to store models and their names
models = {
    'SGDClassifier': SGDClassifier(
        loss='hinge',
        penalty='l2',
        alpha=0.0001,
        learning_rate='optimal',
        max_iter=1000,
        tol=1e-4,
        n_iter_no_change=5,
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

# Dictionary to store cross-validation metrics
cv_metrics = {
    'Model': [],
    'CV Accuracy': [],
    'CV Precision': [],
    'CV Recall': [],
    'CV F1 Score': []
}

# Dictionary to store average confusion matrices
confusion_matrices = {}

# Perform 5-fold cross-validation for each model
n_folds = 5
for model_name, model in models.items():
    print(f"Performing {n_folds}-fold cross-validation for {model_name}...")

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    conf_matrix_sum = np.zeros((2, 2))

    # Cross-validation loop
    for train_index, test_index in skf.split(X_train_tfidf_selected, y_train_mapped):
        X_train_cv, X_test_cv = X_train_tfidf_selected[train_index], X_train_tfidf_selected[test_index]
        y_train_cv, y_test_cv = y_train_mapped[train_index], y_train_mapped[test_index]

        model.fit(X_train_cv, y_train_cv)
        y_pred_cv = model.predict(X_test_cv)

        # Compute confusion matrix for this fold
        conf_matrix = confusion_matrix(y_test_cv, y_pred_cv)
        conf_matrix_sum += conf_matrix

    # Average confusion matrix
    avg_conf_matrix = conf_matrix_sum / n_folds
    avg_conf_matrix = np.round(avg_conf_matrix).astype(int)  # Round and convert to integer
    confusion_matrices[model_name] = avg_conf_matrix

    print(f"Average Confusion Matrix for {model_name}:")
    print(avg_conf_matrix)

    # Plot the averaged confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=avg_conf_matrix, display_labels=['Negative', 'Positive'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f'Average Confusion Matrix - {model_name}')
    plt.show()

    # Use cross_validate to compute multiple metrics
    cv_results = cross_validate(
        model,
        X_train_tfidf_selected,
        y_train_mapped,
        cv=n_folds,
        scoring=['accuracy', 'precision', 'recall', 'f1'],
        n_jobs=-1
    )

    # Compute average metrics across folds
    mean_accuracy = np.mean(cv_results['test_accuracy'])
    mean_precision = np.mean(cv_results['test_precision'])
    mean_recall = np.mean(cv_results['test_recall'])
    mean_f1 = np.mean(cv_results['test_f1'])

    # Store the metrics
    cv_metrics['Model'].append(model_name)
    cv_metrics['CV Accuracy'].append(mean_accuracy)
    cv_metrics['CV Precision'].append(mean_precision)
    cv_metrics['CV Recall'].append(mean_recall)
    cv_metrics['CV F1 Score'].append(mean_f1)

    print(f"{model_name} Cross-Validation Metrics (Averaged over {n_folds} folds):")
    print(f"Average Accuracy: {mean_accuracy:.4f}")
    print(f"Average Precision: {mean_precision:.4f}")
    print(f"Average Recall: {mean_recall:.4f}")
    print(f"Average F1 Score: {mean_f1:.4f}\n")
    print("---------------------------------------------------------------------------")

# Step 5: Compare Cross-Validation Metrics
# Create a DataFrame for better visualization
cv_performance_df = pd.DataFrame(cv_metrics)
print("Cross-Validation Performance Comparison of Models:")
print(cv_performance_df)

# Save the cross-validation performance metrics to a CSV file
cv_performance_df.to_csv('models_cv_performance_comparison.csv', index=False)
print("Cross-validation performance metrics saved to 'models_cv_performance_comparison.csv'.")

# Visualize the cross-validation performance metrics using bar charts
metrics = ['CV Accuracy', 'CV Precision', 'CV Recall', 'CV F1 Score']
plt.figure(figsize=(12, 8))
for i, metric in enumerate(metrics, 1):
    plt.subplot(2, 2, i)
    plt.bar(cv_performance_df['Model'], cv_performance_df[metric], color=['blue', 'green', 'red'])
    plt.ylim(0, 1)
    plt.title(metric)
    plt.ylabel(metric)
    for index, value in enumerate(cv_performance_df[metric]):
        plt.text(index, value + 0.01, f"{value:.2f}", ha='center')
plt.tight_layout()
plt.show()
