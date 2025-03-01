from bleak import BleakClient

MAC_Address = "24:EC:4A:00:04:89"
Service_UUID ="d90562d9-6939-4431-b382-48e783366d67"
Characteristic_UUID = "0a002f5b-d64e-4667-9cc3-c0c5c857715a"

def BLE(client_stream):
     with BleakClient(MAC_Address) as client:
        print("Smart Sole software is connected to Smart Sole device")
        
        def notifications(sender, data):
            try:
                if isinstance(data, bytes):
                       decoded_Input = data.decode("utf-8")
                else:  
                      decoded_Input = data
                heel = int(decoded_Input.split("HEEL ")[1].split(",")[0])
                lateral_arch = int(decoded_Input.split("LATERAL ARCH")[1].split(",")[0])
                medial_metatarsal = int(decoded_Input.split("MEDIAL METATARSAL")[1].split(",")[0])
                print(heel, lateral_arch, medial_metatarsal)



            except Exception as e:
                 print(e)
