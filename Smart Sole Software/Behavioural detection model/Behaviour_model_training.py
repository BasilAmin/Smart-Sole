import asyncio as asy
from bleak import BleakClient
from multiprocessing import Queue
import numpy as np
import tensorflow as tf
from sklearn.ensemble import IsolationForest
import pandas as pd
import matplotlib.py.matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import threading
from sklearn.preprocessing import StandardScaler
import queue

# BLE connectivity
MAC_Address = "24:EC:4A:00:04:89"
Service_UUID = "d90562d9-6939-4431-b382-48e783366d67"
Characteristic_UUID = "0a002f5b-d64e-4667-9cc3-c0c5c857715a"

# Behavior model labels
Labels = {
    0: 'Standing',
    1: 'Sitting',
    2: 'Walking',
    3: 'Limping',
    4: 'heel_avoidance_stationary',
    5: 'heel_avoidance_dynamic',
    6: 'LateralArch_avoidance_stationary',
    7: 'LateralArch_avoidance_dynamic'
}

# Load the behavioral model
bh_model = tf.keras.models.load_model("C:/Users/basil/Documents/PROJECTS/SMART__SOLE/Smart Sole Software/Smart_Sole_Models/Behavioural_classification_model.keras")
bh_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Isolation Forest for anomaly detection
Iso = IsolationForest()

# Read the pre-trained datasets
Walking_data_reg = pd.read_csv("Iso_walking.csv")
Standing_data_reg = pd.read_csv("Iso_standing.csv")
Sitting_data_reg = pd.read_csv("Iso_sitting.csv")
Limping_data_reg = pd.read_csv("Iso_limping.csv")
Heel_av_st_data_reg = pd.read_csv("Iso_Heel_Av_st.csv")
Heel_av_dy_data_reg = pd.read_csv("Iso_Heel_Av_dy.csv")
Lat_pre_st_data_reg = pd.read_csv("Iso_Lat_pre_st.csv")
Lat_pre_dy_data_reg = pd.read_csv("Iso_Lat_pre_dy.csv")

# Data structure init
data_queue = Queue()
preprocessed__data_queue = Queue()
matrix_buffer = []  # Ensure matrix_buffer is a list

# Notifications (to receive BLE data)
async def notifications(sender, data):
    global matrix_buffer
    try:
        decoded_data = data.decode("utf-8")
        heel = int(decoded_data.split("HEEL ")[1].split(",")[0])
        lat_arch = int(decoded_data.split("LATERAL ARCH ")[1].split(",")[0])
        medial = int(decoded_data.split("MEDIAL METATARSAL ")[1].split(",")[0])
        data_queue.put((heel, lat_arch, medial))
        matrix_buffer.append((heel, lat_arch, medial))  # Append data to matrix_buffer
        print(f"Parsed Data: {heel}, {lat_arch}, {medial}")
    except Exception as e:
        print(f"Error in data: {e}")

async def start_ble():
    async with BleakClient(MAC_Address) as client:
        await client.start_notify(Characteristic_UUID, notifications)
        await asy.Event().wait()

# Load training set for scaling
scalar_training_set = pd.read_csv("Training_setR2.csv")

# Preprocess data and run model predictions
async def preprocess_data_model():
    global matrix_buffer
    scalar = StandardScaler()
    if len(matrix_buffer) == 10:  # Assuming each sample contains 10 sets of measurements
        dataset = np.array(matrix_buffer)
        # Reshape the dataset to fit the model requirements: (10, 3) -> (10*3,)
        dataset = dataset.reshape(-1, 3).flatten()
        matrix_buffer = []  # Reset buffer after using the data
        print(f"Dataset: {dataset}")
        
        try:
            scalar.fit(scalar_training_set)
            scaled_dataset = scalar.transform(dataset)
            
            # Reshape scaled dataset to fit the model input shape: (30,)
            scaled_dataset = scaled_dataset.reshape(-1, 30)
            
            # Run behavioral classification model
            prediction = bh_model.predict(scaled_dataset)
            predicted_label = np.argmax(prediction, axis=1)
            print(f"Predicted label: {predicted_label}")
            
            # Choose the appropriate data for the Isolation Forest
            if predicted_label == 0:
                regular_data = Standing_data_reg
            elif predicted_label == 1:
                regular_data = Sitting_data_reg
            elif predicted_label == 2:
                regular_data = Walking_data_reg
            elif predicted_label == 3:
                regular_data = Limping_data_reg
            elif predicted_label == 4:
                regular_data = Heel_av_st_data_reg
            elif predicted_label == 5:
                regular_data = Heel_av_dy_data_reg
            elif predicted_label == 6:
                regular_data = Lat_pre_st_data_reg
            else:
                regular_data = Lat_pre_dy_data_reg
                
            # Make sure regular_data and dataset are both 2D
            Iso_set = np.vstack([regular_data.values, dataset])  # Ensure both arrays are 2D
            if Iso_set.shape[0] > 0:
                Iso.fit(Iso_set)
                predict = Iso.predict(Iso_set)
                scores = Iso.score_samples(Iso_set)

                # Check for anomalies in the last data point
                added_index = Iso_set.shape[0] - 1
                is_anomaly = predict[added_index] == -1
                print(f"Is anomaly: {is_anomaly}")
                return predicted_label, is_anomaly
            else:
                print("No data points for anomaly detection.")
                return None, None
        except Exception as e:
            print(f"An error occurred during prediction: {e}")
            # Default behavior: Assume label 0
            predicted_label = 0
            print(f"Using default label: {predicted_label}")
            return predicted_label, None
    else:
        print("Not enough data points yet.")
        return None, None

# Main plotting and data update
x_data = np.arange(100)
y1_data = [0] * 100
y2_data = [0] * 100
y3_data = [0] * 100

def update_graph(frame):
    global y1_data, y2_data, y3_data

    try:
        while not data_queue.empty():
            heel, lat_arch, medial = data_queue.get_nowait()
            y1_data = y1_data[1:] + [heel]
            y2_data = y2_data[1:] + [lat_arch]
            y3_data = y3_data[1:] + [medial]

    except queue.Empty:
        pass

    # Update the graph with the new data
    line1.set_ydata(y1_data)
    line2.set_ydata(y2_data)
    line3.set_ydata(y3_data)

    # Process the model (this is where we call the model function)
    asyncio.run(run_preprocess_data_model())

    return line1, line2, line3

async def run_preprocess_data_model():
    # Call preprocess_data_model within the event loop and handle its output
    result = await preprocess_data_model()
    if result is not None:
        predicted_label, is_anomaly = result
        if predicted_label is None:
            print("Received None for predicted label; using default label 0.")
            predicted_label = 0
        print(f"Behavior: {Labels[predicted_label]}, Anomaly Detected: {is_anomaly}")
    else:
        print("No results available; skipping.")

fig, ax = plt.subplots()
ax.set_xlim(0, 100)
ax.set_ylim(0, 4095)

line1, = ax.plot(x_data, y1_data, label="Heel", color="r")
line2, = ax.plot(x_data, y2_data, label="Lateral Arch", color="g")
line3, = ax.plot(x_data, y3_data, label="Medial Metatarsal", color="b")
ax.legend()

ani = FuncAnimation(fig, update_graph, interval=1000)

def startPlotting():
    plt.show()

if __name__ == "__main__":
    # Start BLE thread
    ble_thread = threading.Thread(target=asy.run, args=(start_ble(),))
    ble_thread.start()
    startPlotting()