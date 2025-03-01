import numpy as np
import tensorflow as tf
import pandas as pd
from bleak import BleakClient
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from multiprocessing import Queue
import asyncio as asy
from collections import deque

class Behavior_detection:
 def __init__(self, model_path, scalar_training_path):
    self.MAC_address = "24:EC:4A:00:04:89"
    self.Service_UUID = "d90562d9-6939-4431-b382-48e783366d67"
    self.Characteristic_UUID = "0a002f5b-d64e-4667-9cc3-c0c5c857715a"


    self.Labels = {
    0: 'Standing',
    1: 'Sitting',
    2: 'Walking',
    3: 'Limping',
    4: 'heel_avoidance_stationary',
    5: 'heel_avoidance_dynamic',
    6: 'LateralArch_avoidance_stationary',
    7: 'LateralArch_avoidance_dynamic' 
 }
    self.data_buffer = deque(maxlen=20)

    self.Behavior_model = tf.keras.models.load_model(model_path)
    self.scalar = StandardScaler()

    scalar_training_data = pd.read_csv(scalar_training_path)
    self.scalar.fit(scalar_training_data)

 async def notifications(self, sender, data):
    try:
        decoded_data = data.decode("utf-8")
        heel = int(decoded_data.split("HEEL ")[1].split(",")[0])
        lat_arch = int(decoded_data.split("LATERAL ARCH ")[1].split(",")[0])
        medial = int(decoded_data.split("MEDIAL METATARSAL ")[1].split(",")[0])

        self.data_buffer.append([heel, lat_arch, medial])
        print(f"update: recieved heel {heel}, lateral arch {lat_arch} and medial metatarsal {medial}")

        if len(self.data_buffer) == self.data_buffer.maxlen:
           await self.process_data()

    except Exception as e:
        print(f"Error in data {e}")



 async def process_data(self):
    try:
       data_input = np.array(self.data_buffer).flatten()
       data_reshaped = data_input.reshape(1, -1)
       data_scaled = self.scalar.transform(data_reshaped)
       prediction = self.Behavior_model.predict(data_scaled)
       prediction_label = np.argmax(prediction[0])
       behavior = self.Labels[prediction_label]

       print(f"Behavior detected: {behavior}  (confidence level: {prediction[0][prediction_label]:.2f})")
       return behavior
    
    except Exception as e:
       print(f"error making prediction {e}")


 async def run(self):
       try:
          async with BleakClient(self.MAC_address) as client:
             print(f"Device connected to {self.MAC_address}")
             await client.start_notify(self.Characteristic_UUID, self.notifications)

             while True:
                await asy.sleep(1)

       except Exception as e:
          print(f"{e}")

    
async def main():
       detect = Behavior_detection(model_path="C:/Users/basil/Documents/PROJECTS/SMART__SOLE/Smart Sole Software/Behavioural detection model/Behavioural_classification_model_R2.keras",
                                   scalar_training_path="C:/Users/basil/Documents/PROJECTS/SMART__SOLE/Smart Sole Software/Smart_Sole_Models/Testing_R2_2.csv")
       await detect.run()

if __name__ =="__main__":
 asy.run(main())





#path for model: "C:/Users/basil/Documents/PROJECTS/SMART__SOLE/Smart Sole Software/Smart_Sole_Models/Behavioural_classification_model.keras"
#path for scalar_training: "C:/Users/basil/Documents/PROJECTS/SMART__SOLE/Smart Sole Software/Smart_Sole_Models/Testing_R2_2.csv"