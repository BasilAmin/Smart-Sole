import asyncio as asy
from bleak import BleakClient
from multiprocessing import Queue
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import threading
import queue

# BLE connectivity
MAC_Address = "24:EC:4A:00:04:89"
Service_UUID = "d90562d9-6939-4431-b382-48e783366d67"
Characteristic_UUID = "0a002f5b-d64e-4667-9cc3-c0c5c857715a"

data_queue = Queue()
matrix_buffer = np.array([])

async def notifications(sender, data):
    try:
        decoded_data = data.decode("utf-8")
        heel = int(decoded_data.split("HEEL ")[1].split(",")[0])
        lat_arch = int(decoded_data.split("LATERAL ARCH ")[1].split(",")[0])
        medial = int(decoded_data.split("MEDIAL METATARSAL ")[1].split(",")[0])

        data_queue.put((heel, lat_arch, medial))
        print(f"Parsed Data {heel}, {lat_arch}, {medial}")
    except Exception as e:
        print(f"Error in data {e}")

async def start_ble():
    async with BleakClient(MAC_Address) as client:
        await client.start_notify(Characteristic_UUID, notifications)
        print("Notification started")
        await asy.Event().wait()

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

    line1.set_ydata(y1_data)
    line2.set_ydata(y2_data)
    line3.set_ydata(y3_data)

    return line1, line2, line3

fig, ax = plt.subplots()
ax.set_xlim(0, 100)
ax.set_ylim(0, 4095)

line1, = ax.plot(x_data, y1_data, label="Heel", color="r")
line2, = ax.plot(x_data, y2_data, label="Lateral Arch", color="g")
line3, = ax.plot(x_data, y3_data, label="Medial Metatarsal", color="b")
ax.legend()

ani = FuncAnimation(fig, update_graph, interval=1000)

def start_plotting():
    plt.show()

if __name__ == "__main__":
    ble_thread = threading.Thread(target=asy.run, args=(start_ble(),))
    ble_thread.start()
    start_plotting()
