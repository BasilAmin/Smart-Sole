#include <Arduino.h>

#include <BLE2902.h>
#include <BLEDevice.h>
#include <BLEUtils.h>
#include <BLEServer.h>

#define FSR_HEELPIN A0
#define FSR_LATPIN A1
#define FSR_MEDPIN A2

#define Service_UUID "d90562d9-6939-4431-b382-48e783366d67"
#define Characteristic_UUID "0a002f5b-d64e-4667-9cc3-c0c5c857715a"

BLECharacteristic *pCharacteristic;
bool Connection_status = false;

class MyServerCallbacks: public BLEServerCallbacks
{
  void onConnect(BLEServer* pServer )
  {
    Connection_status = true;
    Serial.println("Connected!");
  }

  void onDisconnect(BLEServer* pServer)
  {
    Connection_status = false;
    Serial.println("Disconnected");
    BLEDevice::startAdvertising();
    

  }  
};
 

void setup()
{
    Serial.begin(9600);
    Serial.println("Smart Sole connection testing ...");

    pinMode(FSR_HEELPIN, INPUT);
    pinMode(FSR_LATPIN, INPUT);
    pinMode(FSR_MEDPIN, INPUT);


    BLEDevice::init("Smart Sole");
    BLEServer *pServer = BLEDevice::createServer();
    pServer->setCallbacks(new MyServerCallbacks());

    BLEService *pService = pServer->createService(Service_UUID);

    pCharacteristic = pService->createCharacteristic(
    Characteristic_UUID, 
    BLECharacteristic::PROPERTY_NOTIFY |
     BLECharacteristic::PROPERTY_READ);
    

    pCharacteristic->addDescriptor(new BLE2902());


    pService->start();

    BLEAdvertising *pAdvertising = BLEDevice::getAdvertising();
    pAdvertising->addServiceUUID(Service_UUID);
    pAdvertising->setScanResponse(true);
    pAdvertising->setMinPreferred(0x06);
    pAdvertising->setMinPreferred(0x12);
    BLEDevice::startAdvertising();
    
    Serial.println("Started advertising connectivity");



}

void loop()
{
    if (Connection_status){
    int heel_val = analogRead(FSR_HEELPIN);
    int lat_val = analogRead(FSR_LATPIN);
    int med_val = analogRead(FSR_MEDPIN);

  int  time = millis() / 1000;

    String InputData_stream =
            "HEEL " + String(heel_val) +
        ", LATERAL ARCH " + String(lat_val) +
        ", MEDIAL METATARSAL " + String(med_val) + ", TIME " + String(time);
  

    pCharacteristic->setValue(InputData_stream.c_str());
    pCharacteristic->notify();

    Serial.println("Data: " + InputData_stream);




delay(1000);
    }
}