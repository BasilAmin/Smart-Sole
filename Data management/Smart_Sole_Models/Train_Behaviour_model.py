import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import numpy as np
import pandas as pd

# Behavior labels
LABELS = {
    'Standing/sitting': 0,
    'Walking': 1,
    'limping': 2,
    'heel_avoidance_stationary': 3,
    'heel_avoidance_dynamic': 4,
    'LateralArch_pressure_stationary': 5,
    'LateralArch_pressure_dynamic': 6,
    'Running': 7,
}

# Load CSV data
df = pd.read_csv("TRAINING SET.csv", header=None)  # Replace with your CSV file path

# Reshape data to (70, 3, 10)
data = df.to_numpy().reshape(-1, 3, 10)

# Define labels (ensure correct alignment)
labels = np.array([1] * 16 + [0] * 19 + [2] * 6 + [5] * 7 + [6] * 9 + [3] * 10 + [4] * 9)
assert len(data) == len(labels), "Mismatch between X and Y lengths."

# Split into training and testing sets
split_index = int(0.8 * len(data))  # 80% for training, 20% for testing
X_train, X_test = data[:split_index], data[split_index:]
Y_train, Y_test = labels[:split_index], labels[split_index:]

# Define the model
Behavioural_classification_model = Sequential([
    layers.Flatten(input_shape=(3, 10)),
    layers.Dense(64, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(len(LABELS), activation='softmax')
])

Behavioural_classification_model.compile(optimizer='adam',
                                         loss='sparse_categorical_crossentropy',
                                         metrics=['accuracy'])

# Train the model
def train_model(X_train, Y_train, X_test, Y_test):
    Behavioural_classification_model.fit(X_train, Y_train, epochs=10)
    test_loss, test_acc = Behavioural_classification_model.evaluate(X_test, Y_test, verbose=2)
    print('\nTest accuracy:', test_acc)
    Behavioural_classification_model.save("Behavioural_classification_model.h5")

train_model(X_train, Y_train, X_test, Y_test)
