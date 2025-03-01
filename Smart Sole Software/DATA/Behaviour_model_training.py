import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

# Load the data
Training_data = pd.read_csv("Training_setR2.csv")
Testing_data = pd.read_csv("Testing_R2.csv")




X_train = Training_data.iloc[:, :-1].to_numpy()
Y_train = Training_data.iloc[:, -1].to_numpy()

X_test = Testing_data.iloc[:, :-1].to_numpy()
Y_test = Testing_data.iloc[:, -1].to_numpy()

print("Training data loaded")

from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler
scalar =  StandardScaler()
X_train = scalar.fit_transform(X_train)
X_test = scalar.transform(X_test)

class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(Y_train),y = Y_train)
class_weights_dict = dict(enumerate(class_weights))


LABELS = {
     'Standing': 0,
     'Sitting': 1,
     'Walking': 2,
     'limping': 3,
     'heel_avoidance_stationary' : 4,
     'heel_avoidance_dynamic' : 5,
     'LateralArch_avoidance_stationary' : 6,
     'LateralArch_avoidance_dynamic' : 7,
}


Behavioural_classification_model = Sequential([
    layers.Input(shape=(60,)),
    layers.BatchNormalization(),
    layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    layers.Dropout(0.3),
    layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    layers.Flatten(),
    layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    layers.Dropout(0.2),
    layers.Dense(len(LABELS), activation='softmax')
])


Behavioural_classification_model.compile(optimizer= tf.keras.optimizers.Adam(learning_rate=0.001),
 loss='sparse_categorical_crossentropy', metrics=['accuracy'])

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

def train_model(X_train, Y_train, X_test, Y_test):
    model_history = Behavioural_classification_model.fit(X_train, Y_train, epochs=50, batch_size = 16, validation_split = 0.2, class_weight=class_weights_dict, callbacks=[early_stopping])
    testloss, test_accuracy = Behavioural_classification_model.evaluate(X_test, Y_test, verbose=2)
    print(f"Test loss: {testloss}")
    print(f"Test accuracy: {test_accuracy}")
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(model_history.history['accuracy'], label='accuracy')
    plt.plot(model_history.history['val_accuracy'], label='val_accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy') 
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(model_history.history['loss'], label='loss')
    plt.plot(model_history.history['val_loss'], label='val_loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    Behavioural_classification_model.save("Behavioural_classification_model_R2.keras")
    return model_history
train_model(X_train, Y_train, X_test, Y_test)