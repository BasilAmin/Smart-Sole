import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

LABELS = {
     'Standing': 0,
     'Sitting': 1,
     'Walking': 2,
     'limping': 3,
     'heel_avoidance_stationary': 4,
     'heel_avoidance_dynamic': 5,
     'LateralArch_avoidance_stationary': 6,
     'LateralArch_avoidance_dynamic': 7,
}
reverse = {v: k for k, v in LABELS.items()}

def verify_data(X_test):
    """Verify data format and ranges"""
    print("\nData Verification:")
    print(f"Number of features: {X_test.shape[1]}")  # Should be 60
    print(f"Value ranges:")
    print(f"Min: {X_test.values.min()}")
    print(f"Max: {X_test.values.max()}")
    
    if X_test.shape[1] != 60:
        print("WARNING: Number of features doesn't match model input shape (60)!")
    if X_test.values.max() > 4095 or X_test.values.min() < 0:
        print("WARNING: Feature values outside expected range (0-4095)!")
    
    return X_test.shape[1] == 60 and 0 <= X_test.values.min() <= X_test.values.max() <= 4095

def evaluate_model(model_path, test_data_path):
    """
    Evaluates model performance with detailed metrics and visualizations
    """
    # Load model and data
    print("Loading model and data...")
    model = tf.keras.models.load_model(model_path)
    test_data = pd.read_csv(test_data_path)
    
    # Separate features and labels
    X_test = test_data.iloc[:, :-1].astype(np.float32)
    y_test = test_data.iloc[:, -1].astype(int)
    
    # Verify data
    if not verify_data(X_test):
        print("WARNING: Data verification failed! Results may be unreliable.")
    
    # Apply normalization
    print("\nApplying batch normalization...")
    features_mean = np.mean(X_test, axis=0)
    features_std = np.std(X_test, axis=0)
    X_test_normalized = (X_test - features_mean) / (features_std + 1e-7)
    
    # Get predictions using normalized data
    print("Making predictions...")
    y_pred_prob = model.predict(X_test_normalized)
    y_pred = np.argmax(y_pred_prob, axis=1)
    
    # Calculate metrics
    accuracy = np.mean(y_pred == y_test)
    report = classification_report(y_test, y_pred,
                                 target_names=LABELS.keys(),
                                 output_dict=True)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # Print results
    print("\nResults:")
    print(f"Overall Accuracy: {accuracy:.2%}")
    print("\nDetailed Classification Report:")
    print(pd.DataFrame(report).transpose())
    
    # Visualize confusion matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(conf_matrix, annot=True, fmt='d',
                xticklabels=LABELS.keys(),
                yticklabels=LABELS.keys())
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    # Analyze misclassifications
    print("\nAnalyzing misclassifications...")
    misclassified_indices = np.where(y_pred != y_test)[0]
    if len(misclassified_indices) > 0:
        print("\nMisclassification Analysis:")
        for idx in misclassified_indices:
            print(f"\nInstance {idx+1}:")
            print(f"Actual: {reverse[y_test[idx]]}")
            print(f"Predicted: {reverse[y_pred[idx]]}")
            print(f"Confidence: {np.max(y_pred_prob[idx]):.2%}")
            
        # Print confusion patterns
        print("\nCommon confusion patterns:")
        misclassified_pairs = list(zip(y_test[misclassified_indices], 
                                     y_pred[misclassified_indices]))
        from collections import Counter
        patterns = Counter([(reverse[actual], reverse[pred]) 
                          for actual, pred in misclassified_pairs])
        for (actual, pred), count in patterns.most_common(5):
            print(f"{actual} misclassified as {pred}: {count} times")
    else:
        print("No misclassifications found!")

# Usage
evaluate_model(
    "C:/Users/basil/Documents/PROJECTS/SMART__SOLE/Smart Sole Software/Behavioural detection model/Behavioural_classification_model_R2.keras",
    "C:/Users/basil/Documents/PROJECTS/SMART__SOLE/Smart Sole Software/DATA/Training_setR2.csv"
)