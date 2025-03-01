from bleak import BleakClient
import asyncio as asy
from multiprocessing import Pipe, Queue, Process
import numpy as np
from sklearn.ensemble import IsolationForest as iso
import matplotlib.pyplot as plt
import pandas as pd


#BLE CONNECTIVITY SETTINGS
MAC_Address = "24:EC:4A:00:04:89"
Service_UUID ="d90562d9-6939-4431-b382-48e783366d67"
Characteristic_UUID = "0a002f5b-d64e-4667-9cc3-c0c5c857715a"

async def read_characteristic(client):
        try:
            value = await client.read_gatt_char(Characteristic_UUID)
            return value
        except Exception as e:
            print(f"error getting ch: {e}")

async def notifications(client):
        async def handler(sender, data):
            print(f"{data}")
        try:
            await client.start_notify(Characteristic_UUID, handler)
            await asy.sleep(float("inf"))
        except Exception as e:
            print(f"Error with notifications: {e}")
        finally:
            await client.stop_notify(Characteristic_UUID)



def parse_data(Input_Data):
     matrix_buffer = np.array([])
     Input_data = Input_Data.decode("utf-8")

     heel_Input_value = Input_data.split("HEEL ")[1].split(",")[0]
     lat_Input_value = Input_data.split("LATERAL ARCH ")[1].split(",")[0]
     medial_Input_value = Input_data.split("MEDIAL METATARSAL ")[1].split(",")[0]
     time = Input_data.split("TIME ")[1]

     for i in range(10):
         Input_data = np.array([[int(heel_Input_value)],
                                [int(lat_Input_value)],
                                [int(medial_Input_value)],
                                [int(time)]])
         matrix_buffer = np.append(matrix_buffer, Input_data)
         print( "INPUT MATRIX" + matrix_buffer)
         return matrix_buffer


    
    #Isolation Forest
def isolation_forest(InputDataset):
        clf = iso(contamination = 0.1)
        clf.fit(InputDataset)
        return clf

def display_Anomalies(InputDataset, clf):
     Anomaly_score = clf.score_samples(InputDataset)



def dataset_injection(InputDataset, Behaviour_Label):
     if Behaviour_Label == 0:
         #Standing
         standing_dataset_injection = np.array([[ [0, 0, 0, 0],
                                                  [0, 0, 0, 0],
                                                  [0, 0, 0, 0],]])
         InputDataset = np.append(InputDataset, standing_dataset_injection)
         return InputDataset
     
     elif Behaviour_Label == 1:
            #Walking
            walking_dataset_injection = np.array([[ [1, 1, 1, 1],
                                                    [1, 1, 1, 1],
                                                    [1, 1, 1, 1], ]])
            InputDataset = np.append(InputDataset, walking_dataset_injection)
            return InputDataset

     elif Behaviour_Label == 2:
            #Limping
            limping_dataset_injection = np.array([[ [2, 2, 2, 2],
                                                    [2, 2, 2, 2],
                                                    [2, 2, 2, 2], ]])
            InputDataset = np.append(InputDataset, limping_dataset_injection)
            return InputDataset
     
     elif Behaviour_Label == 3:
          #Heel avoidance stationary
            heel_avoidance_stationary_dataset_injection = np.array([[ [3, 3, 3, 3],
                                                                      [3, 3, 3, 3],
                                                                      [3, 3, 3, 3], ]])
            InputDataset = np.append(InputDataset, heel_avoidance_stationary_dataset_injection)
            return InputDataset
     
     elif Behaviour_Label == 4:
          #jHeel avoidance dynamic
            heel_avoidance_dynamic_dataset_injection = np.array([[ [4, 4, 4, 4],
                                                                  [4, 4, 4, 4],
                                                                  [4, 4, 4, 4], ]])
            InputDataset = np.append(InputDataset, heel_avoidance_dynamic_dataset_injection)
            return InputDataset
     
     elif Behaviour_Label == 5:  
            #Lateral Arch avoidance stationary
            LateralArch_avoidance_stationary_dataset_injection = np.array([[ [5, 5, 5, 5],
                                                                            [5, 5, 5, 5],
                                                                            [5, 5, 5, 5], ]])
            InputDataset = np.append(InputDataset, LateralArch_avoidance_stationary_dataset_injection)
            return InputDataset
     
     elif Behaviour_Label == 6:
            #Lateral Arch avoidance dynamic
            LateralArch_avoidance_dynamic_dataset_injection = np.array([[ [6, 6, 6, 6],
                                                                        [6, 6, 6, 6],
                                                                        [6, 6, 6, 6], ]])
            InputDataset = np.append(InputDataset, LateralArch_avoidance_dynamic_dataset_injection)
            return InputDataset
            
     elif Behaviour_Label == 7:
          #Running
            Running_dataset_injection = np.array([[ [7, 7, 7, 7],
                                                  [7, 7, 7, 7],
                                                  [7, 7, 7, 7], ]])
            InputDataset = np.append(InputDataset, Running_dataset_injection)
            return InputDataset
     
     elif Behaviour_Label == 8:
            #Other
            Other_dataset_injection = np.array([[ [8, 8, 8, 8],
                                                [8, 8, 8, 8],
                                                [8, 8, 8, 8], ]])
            InputDataset = np.append(InputDataset, Other_dataset_injection)
            return InputDataset





async def main():
    if __name__ == "__main__":
        try:
            async with BleakClient(MAC_Address) as client:
                await read_characteristic(client)
                await notifications(client)
                print("Connected")
        except Exception as e:
            print(f"Connection error: {e}")

asy.run(main())