import asyncio
from bleak import BleakClient
from multiprocessing import Queue
import numpy as np
import tensorflow as tf
from sklearn.ensemble import IsolationForest
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import threading
from sklearn.preprocessing import StandardScaler
import time

MAC_Address = "24:EC:4A:00:04:89"
Service_UUID = "d90562d9-6939-4431-b382-48e783366d67"
Characteristic_UUID = "0a002f5b-d64e-4667-9cc3-c0c5c857715a"

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


print("loading model . . .")
bh_model = tf.keras.models.load_model("C:/Users/basil/Documents/PROJECTS/SMART__SOLE/Smart Sole Software/Smart_Sole_Models/Behavioural_classification_model.keras")
Iso = IsolationForest()
scalar = StandardScaler()

scalar_training_set = pd.read_csv("Testing_R2_2.csv")

datasets = {
0 : pd.read_csv("Iso_walking.csv"),
1 : pd.read_csv("Iso_standing.csv"),
2 : pd.read_csv("Iso_sitting.csv"),
3 : pd.read_csv("Iso_limping.csv"),
4 : pd.read_csv("Iso_Heel_Av_st.csv"),
5 : pd.read_csv("Iso_Heel_Av_dy.csv"),
6 : pd.read_csv("Iso_Lat_pre_st.csv"),
7 : pd.read_csv("Iso_Lat_pre_dy.csv"),
}
#data structure init
data_queue = Queue()
current_behavior = "Waiting for data"
current_anomaly = False
matrix_buffer = []
last_prediction_time = 0
prediction_Interval = 20

fig, (ax1, ax2) = plt.subplots(2, 1, height_ratios=[3, 1], figsize=(10, 8))
plt.subplots_adjust(hspace=0.3)

ax1.set_title('Smart Sole metrics')
ax1.set_ylim(0, 4095)
x_data = np.arange(100)
y1_data = [0] * 100
y2_data = [0] * 100
y3_data = [0] * 100

line1, = ax1.plot(x_data, y1_data, label="Heel", color="r")
line2, = ax1.plot(x_data, y2_data, label="Lateral Arch", color="g")
line3, = ax1.plot(x_data, y3_data, label="Medial Metatarsal", color="b")
ax1.legend()

behavior_text = ax2.text(0.5, 0.7, f"Behavior: {current_behavior}", 
                        horizontalalignment='center', fontsize=12)
anomaly_text = ax2.text(0.5, 0.3, f"Anomaly Detected: {current_anomaly}", 
                        horizontalalignment='center', fontsize=12)
ax2.axis('off')


def preprocess_buffer_data(buffer):
    heel_data = [x[0] for x in buffer]
    lat_arch_data = [x[1] for x in buffer]
    medial_data = [x[2] for x in buffer]

    processed_data = []
    processed_data.extend(heel_data[:10])      
    processed_data.extend(lat_arch_data[:10])  
    processed_data.extend(medial_data[:10])    
    
    return np.array(processed_data)

async def notifications(sender, data):
    global matrix_buffer
    try:
        decoded_data = data.decode("utf-8")
        heel = int(decoded_data.split("HEEL ")[1].split(",")[0])
        lat_arch = int(decoded_data.split("LATERAL ARCH ")[1].split(",")[0])
        medial = int(decoded_data.split("MEDIAL METATARSAL ")[1].split(",")[0])
        
        data_queue.put((heel, lat_arch, medial))
        matrix_buffer.append([heel, lat_arch, medial])
        if len(matrix_buffer) > 10: 
            matrix_buffer.pop(0)
            
        print(f"Buffer size: {len(matrix_buffer)}")
    except Exception as e:
        print(f"Error in data parsing: {e}")

async def start_ble():
    while True:
        try:
            print("Attempting to connect to BLE device...")
            async with BleakClient(MAC_Address) as client:
                print("Connected to BLE device")
                await client.start_notify(Characteristic_UUID, notifications)
                await asyncio.Event().wait()
        except Exception as e:
            print(f"BLE Error: {e}")
            print("Retrying connection in 5 seconds...")
            await asyncio.sleep(5)

def process_model_data():
    global current_behavior, current_anomaly, last_prediction_time
    
    current_time = time.time()
    
    if current_time - last_prediction_time < prediction_Interval:
        return
    if len(matrix_buffer) >= 10:
        try:
            model_input = preprocess_buffer_data(matrix_buffer)
            model_input = model_input.reshape(1, 30)
            print(f"Model input shape: {model_input.shape}")
            scaled_data = scalar.fit_transform(model_input)
            prediction = bh_model.predict(scaled_data)
            label = np.argmax(prediction[0])
            current_behavior = Labels[label]
            iso_input = model_input.reshape(1, -1) 
            regular_data = datasets[label]
            regular_data_values = regular_data.values.reshape(-1, iso_input.shape[1])
            Iso_set = np.vstack([regular_data_values, iso_input])
            Iso.fit(Iso_set)
            predict = Iso.predict(Iso_set)
            current_anomaly = predict[-1] == -1
            
            print(f"Time: {time.strftime('%H:%M:%S')}")
            print(f"Prediction: {current_behavior}, Anomaly: {current_anomaly}")
            last_prediction_time = current_time
        except Exception as e:
            print(f"Model Error: {e}")
            import traceback
            print(traceback.format_exc())
def update_graph(frame):
    global y1_data, y2_data, y3_data, current_behavior, current_anomaly
    try:
        while not data_queue.empty():
            heel, lat_arch, medial = data_queue.get_nowait()
            y1_data = y1_data[1:] + [heel]
            y2_data = y2_data[1:] + [lat_arch]
            y3_data = y3_data[1:] + [medial]
        
        line1.set_ydata(y1_data)
        line2.set_ydata(y2_data)
        line3.set_ydata(y3_data)
        process_model_data()
        current_time = time.strftime('%H:%M:%S')
        behavior_text.set_text(f"Behavior: {current_behavior}\nLast updated: {current_time}")
        anomaly_text.set_text(f"Anomaly Detected: {current_anomaly}")
        return line1, line2, line3, behavior_text, anomaly_text
    except Exception as e:
        print(f"Update Error: {e}")
        return line1, line2, line3, behavior_text, anomaly_text
if __name__ == "__main__":
    print("Starting Smart Sole Analysis...")
    ble_thread = threading.Thread(target=asyncio.run, args=(start_ble(),))
    ble_thread.daemon = True 
    ble_thread.start()
    ani = FuncAnimation(fig, update_graph, interval=100)
    plt.show()
