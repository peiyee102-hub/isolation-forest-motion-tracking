#include <WiFi.h>
#include <esp_now.h>
#include <esp_wifi.h>
#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLE2902.h>
#include "esp_bt.h"
#include <Wire.h>
#include "SparkFun_BNO080_Arduino_Library.h"
#include <Preferences.h>
#include <esp_sleep.h>

// ================= CONFIGURATION =================
#define SDA_1 6
#define SCL_1 7
#define BNO_ADDR 0x4B
#define SWITCH_PIN 3
#define LED_PIN 20
#define BATT_PIN 4

// ================= BATTERY CONFIGURATION =================
#define ADC_MAX 4095.0
#define VREF 2.95          // Adjust to match your board (e.g., 2.25 if needed)
#define DIVIDER_RATIO 2.0  // Adjust based on your voltage divider resistors

#define BATTERY_MAX 4.2
#define BATTERY_MIN 3.0

// ================= MATH STRUCTS =================
struct Quat { float x, y, z, w; };

// ================= OBJECTS =================
BNO080 bno1; 
Preferences prefs; 

// Alignment Offsets & State Tracking
Quat offset1 = {0, 0, 0, 1};
bool bno1_ok = false;
bool magEnabled = false; 

// Watchdog Vars
Quat prevQ1 = {0,0,0,0};
int staleCount1 = 0;

// ================= MATH HELPERS =================
Quat qMul(Quat a, Quat b) {
  return {
    a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
    a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
    a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
    a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z
  };
}
Quat qInv(Quat q) { return {-q.x, -q.y, -q.z, q.w}; }
Quat qNorm(Quat q) {
  float len = sqrt(q.x*q.x + q.y*q.y + q.z*q.z + q.w*q.w);
  if (len == 0) return {0,0,0,1};
  return {q.x/len, q.y/len, q.z/len, q.w/len};
}
bool isSame(Quat a, Quat b) {
  return (a.x == b.x && a.y == b.y && a.z == b.z && a.w == b.w);
}

// Extracts the Yaw (Z-axis rotation) in radians from a quaternion
float getYaw(Quat q) {
  // LaTeX representation: \psi = \text{atan2}(2(w \cdot z + x \cdot y), 1 - 2(y^2 + z^2))
  float siny_cosp = 2.0f * (q.w * q.z + q.x * q.y);
  float cosy_cosp = 1.0f - 2.0f * (q.y * q.y + q.z * q.z);
  return atan2(siny_cosp, cosy_cosp);
}

// Creates a quaternion representing ONLY a Yaw rotation
Quat quatFromYaw(float yaw) {
  return {
    0.0f, 
    0.0f, 
    sin(yaw / 2.0f), 
    cos(yaw / 2.0f)
  };
}

// --- UUIDs for BLE ---
#define SERVICE_UUID        "4fafc201-1fb5-459e-8fcc-c5c9c331914b"
#define CHARACTERISTIC_UUID "beb5483e-36e1-4688-b7f5-ea07361b26a8"

// --- System States & Constants ---
enum DeviceState { STATE_INIT_LISTEN, STATE_MASTER, STATE_SLAVE };
DeviceState currentState = STATE_INIT_LISTEN;

// --- Calibration Flags ---
volatile bool pendingFullTare = false;
volatile bool pendingHeadingTare = false;

unsigned long listenStartTime = 0;
const unsigned long LISTEN_TIMEOUT = 30000; // 30 seconds

// --- Network Globals ---
uint8_t broadcastMac[] = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF};
uint8_t masterMac[6];
bool isSynced = false;
uint8_t mySlaveId = 0; // Master is 0. Slaves will be 1 and 2.
uint32_t timeOffset = 0;

// BLE Variables
BLEServer* pServer = NULL;
BLECharacteristic* pCharacteristic = NULL;
bool deviceConnected = false;

// --- FreeRTOS Buffers (Queues) ---
QueueHandle_t bleQueue;
QueueHandle_t espnowQueue;

// --- 60Hz Sampling Variables ---
unsigned long lastSensorRead = 0;
const unsigned long SENSOR_INTERVAL = 16; // 1000ms / 60 ≈ 16.6ms
unsigned long lastMasterBleSend = 0;
const unsigned long BLE_INTERVAL = 50; // 20Hz

// --- Message Structures ---
enum MsgType { MSG_ANNOUNCE, MSG_SYNC, MSG_DATA_CHUNK, MSG_CALIBRATE};

// A single raw data point (17 bytes: id + 4 floats)
typedef struct __attribute__((packed)) {
  uint8_t id;
  uint32_t timestamp; 
  uint8_t battery;
  float x;
  float y;
  float z;
  float w; // Added w for full quaternion support!
} SensorSample;

typedef struct __attribute__((packed)) {
  uint8_t type;       // Will be MSG_CALIBRATE
  uint8_t calibType;  // 0 for Full Tare, 1 for Heading Only
} CalibrateMsg;

bool readMySensor(SensorSample &outSample, uint8_t myId); 

// ESP-NOW Payload
#define SAMPLES_PER_CHUNK 1 
typedef struct __attribute__((packed)) {
  uint8_t type;       // MSG_DATA_CHUNK
  uint8_t count;      // How many samples are in this packet
  SensorSample samples[SAMPLES_PER_CHUNK];
} DataChunkMsg;

typedef struct __attribute__((packed)) {
  uint8_t type; 
  uint8_t mac[6]; 
} AnnounceMsg;

typedef struct __attribute__((packed)) {
  uint8_t type; 
  uint32_t masterTime;
  uint8_t assignedId; 
} SyncMsg;

// --- Master's Tracking Variables ---
#define MAX_SLAVES 2
struct SlaveData {
  uint8_t mac[6];
  uint8_t id;
  bool active;
};
SlaveData slaves[MAX_SLAVES];
uint8_t nextSlaveId = 1; // 1 and 2 reserved for slaves

// --- Timers for Loop Tasks ---
unsigned long lastSlaveBroadcast = 0;
unsigned long lastBlinkTime = 0;   
bool currentLedState = HIGH;

// --- BLE Callbacks ---
class MyServerCallbacks: public BLEServerCallbacks {
    void onConnect(BLEServer* pServer) { deviceConnected = true; };
    void onDisconnect(BLEServer* pServer) {
      deviceConnected = false;
      pServer->startAdvertising(); 
    }
};

// --- BLE Write Callbacks (Handles PC Commands) ---
class MyCharacteristicCallbacks: public BLECharacteristicCallbacks {
    void onWrite(BLECharacteristic *pChar) {
      String rxValue = pChar->getValue();
      
      if (rxValue.length() > 0) {
        char c = rxValue[0]; // Look at the first character sent
        
        if (c == 'c') {
          pendingFullTare = true;
          Serial.println("BLE Command Received: Queueing Full Tare...");
        } 
        else if (c == 'h') {
          pendingHeadingTare = true;
          Serial.println("BLE Command Received: Queueing Heading Tare...");
        }
      }
    }
};

String macToStr(const uint8_t* mac) {
  char buf[18];
  snprintf(buf, sizeof(buf), "%02X:%02X:%02X:%02X:%02X:%02X", mac[0], mac[1], mac[2], mac[3], mac[4], mac[5]);
  return String(buf);
}



// ================= BATTERY READING =================
float readBatteryVoltage() {
  int adcValue = 0;
  
  // Read 10 times and average for a stable reading
  for (int i = 0; i < 10; i++) {
    adcValue += analogRead(BATT_PIN); 
  }
  adcValue /= 10;

  float voltage = (adcValue / ADC_MAX) * VREF;
  return voltage * DIVIDER_RATIO;
}

uint8_t getBatteryPercentage() {
  float v = readBatteryVoltage();
  float p = (v - BATTERY_MIN) / (BATTERY_MAX - BATTERY_MIN) * 100.0;

  // Clamp the percentage between 0 and 100
  if (p > 100.0) p = 100.0;
  if (p < 0.0) p = 0.0;

  return (uint8_t)p;
}

// ================= RESET & CALIBRATION =================
void resetSensor(int id) {
  Serial.print(">>> WATCHDOG: Full Reset Sensor "); Serial.println(id);
  Wire.end(); 
  Wire.begin(SDA_1, SCL_1); 
  Wire.setClock(400000);
  delay(100); 

  if (bno1.begin(BNO_ADDR, Wire)) {
    Serial.println(">>> Sensor Recovered!");
    if(magEnabled) bno1.enableRotationVector(50);    
    else bno1.enableGameRotationVector(50);          
    bno1_ok = true;
    staleCount1 = 0;
  } else {
    Serial.println(">>> Sensor Recovery Failed.");
  }
}

// Option A: Zeros Pitch, Roll, and Yaw
void calibrateFullTare() {
  Serial.println(">>> FULL TARE: Zeroing all axes <<<");
  if (bno1_ok) {
    Quat raw = {bno1.getQuatI(), bno1.getQuatJ(), bno1.getQuatK(), bno1.getQuatReal()};
    
    // Invert the entire rotation
    offset1 = qInv(qNorm(raw)); 
    
    prefs.begin("mount", false);
    prefs.putBytes("o1", &offset1, sizeof(offset1));
    prefs.end();
    Serial.println(">>> System Fully Tared.");
  }
}

// Option B: Zeros Yaw (Heading) only, keeps gravity/tilt accurate
void calibrateHeadingOnly() {
  Serial.println(">>> HEADING CORRECTION: Zeroing Yaw only <<<");
  if (bno1_ok) {
    Quat raw = {bno1.getQuatI(), bno1.getQuatJ(), bno1.getQuatK(), bno1.getQuatReal()};
    
    // 1. Find out what our current Yaw heading is
    float currentYaw = getYaw(raw);
    
    // 2. Create a quaternion that represents ONLY that Yaw rotation
    Quat yawOnlyQuat = quatFromYaw(currentYaw);
    
    // 3. Invert that yaw-only quaternion to create our offset
    offset1 = qInv(yawOnlyQuat); 
    
    prefs.begin("mount", false);
    prefs.putBytes("o1", &offset1, sizeof(offset1));
    prefs.end();
    Serial.println(">>> Heading Corrected. Forward is now Zero.");
  }
}

void loadMountingCalibration() {
  prefs.begin("mount", true);
  if (prefs.isKey("o1")) prefs.getBytes("o1", &offset1, sizeof(offset1));
  prefs.end();
}

// ================= SENSOR READ HELPER =================
// This cleanly reads the sensor and packages it into a struct.
bool readMySensor(SensorSample &outSample, uint8_t myId) {
  if (!bno1_ok) return false;
  if (bno1.dataAvailable()) {
    float qi = bno1.getQuatI();
    float qj = bno1.getQuatJ();
    float qk = bno1.getQuatK();
    float qr = bno1.getQuatReal();
    
    if (qi != 0.0 || qj != 0.0 || qk != 0.0 || qr != 0.0) {
      Quat raw = {qi, qj, qk, qr};
      
      if (isSame(raw, prevQ1)) staleCount1++;
      else { staleCount1 = 0; prevQ1 = raw; }

      if (staleCount1 >= 300) { staleCount1 = 0; return false; }

      Quat q = qNorm(qMul(offset1, raw));
      
      outSample.id = myId;
      outSample.timestamp = millis() + timeOffset;
      outSample.battery = getBatteryPercentage(); // Replace this with bat per
      outSample.x = q.x;
      outSample.y = q.y;
      outSample.z = q.z;
      outSample.w = q.w;
      return true;
    }
  }
  return false;
}

// ================= ESP-NOW RECEIVE =================
void OnDataRecv(const esp_now_recv_info_t *esp_now_info, const uint8_t *incomingData, int len) {
  const uint8_t *mac_addr = esp_now_info->src_addr;
  uint8_t msgType = incomingData[0];

  if (currentState == STATE_INIT_LISTEN && msgType == MSG_ANNOUNCE) {
    Serial.println("Received Slave Broadcast. I am becoming MASTER.");
    currentState = STATE_MASTER;
    mySlaveId = 0; // Master is always ID 0
    setupBLE(); 
  }

  if (currentState == STATE_MASTER) {
    if (msgType == MSG_ANNOUNCE) {
      AnnounceMsg msg;
      memcpy(&msg, incomingData, sizeof(msg));
      
      int slaveIdx = -1;
      for (int i = 0; i < MAX_SLAVES; i++) {
        if (slaves[i].active && memcmp(slaves[i].mac, msg.mac, 6) == 0) {
          slaveIdx = i; break;
        }
      }

      if (slaveIdx == -1 && nextSlaveId <= MAX_SLAVES) { 
        for (int i = 0; i < MAX_SLAVES; i++) {
          if (!slaves[i].active) {
            slaveIdx = i;
            memcpy(slaves[i].mac, msg.mac, 6);
            slaves[i].id = nextSlaveId++;
            slaves[i].active = true;
            break;
          }
        }
        
        if (!esp_now_is_peer_exist(msg.mac)) {
          esp_now_peer_info_t peerInfo;
          memset(&peerInfo, 0, sizeof(peerInfo));
          memcpy(peerInfo.peer_addr, msg.mac, 6);
          esp_now_add_peer(&peerInfo);
        }
      }

      if (slaveIdx != -1) {
        SyncMsg syncMsg;
        syncMsg.type = MSG_SYNC;
        syncMsg.masterTime = millis();
        syncMsg.assignedId = slaves[slaveIdx].id;
        esp_now_send(msg.mac, (uint8_t *) &syncMsg, sizeof(syncMsg));
        Serial.printf("Synced Slave ID %d\n", slaves[slaveIdx].id);
      }
    }
    
    else if (msgType == MSG_DATA_CHUNK) {
      DataChunkMsg chunk;
      memcpy(&chunk, incomingData, sizeof(chunk));
      
      // Master drops incoming Slave data directly into the BLE queue!
      for (int i = 0; i < chunk.count; i++) {
        xQueueSend(bleQueue, &chunk.samples[i], 0); 
      }
    }
  }

  // ==========================================================
  // --- SLAVE RECEIVE LOGIC ---
  // ==========================================================
  if (currentState == STATE_SLAVE) {
    
    // 1. Handle Initial Sync & Time Offset
    if (msgType == MSG_SYNC) {
      SyncMsg msg;
      memcpy(&msg, incomingData, sizeof(msg));
      
      memcpy(masterMac, mac_addr, 6); 
      mySlaveId = msg.assignedId;
      isSynced = true;

      // ---> HERE IS THE TIME OFFSET CALCULATION <---
      timeOffset = msg.masterTime - millis(); 

      if (!esp_now_is_peer_exist(masterMac)) {
        esp_now_peer_info_t peerInfo;
        memset(&peerInfo, 0, sizeof(peerInfo));
        memcpy(peerInfo.peer_addr, masterMac, 6);
        esp_now_add_peer(&peerInfo);
      }
      Serial.printf("Synced with Master! My ID is %d. Time Offset: %d ms\n", mySlaveId, timeOffset);
    }
    
    // 2. Handle Calibration Commands from Master
    else if (msgType == MSG_CALIBRATE) {
      CalibrateMsg msg;
      memcpy(&msg, incomingData, sizeof(msg));
      
      // Just set the flags, don't execute the heavy functions yet!
      if (msg.calibType == 0) {
        pendingFullTare = true;
      } else if (msg.calibType == 1) {
        pendingHeadingTare = true;
      }
      Serial.println("<<< Received Calibrate Command from Master! Queued for processing.");
    }
  }
} 


// ================= BLE SETUP =================
void setupBLE() {
  BLEDevice::init("Physio_A"); // Changed to match your Flutter scan filter
  pServer = BLEDevice::createServer();
  pServer->setCallbacks(new MyServerCallbacks());
  
  BLEService *pService = pServer->createService(SERVICE_UUID);
  pCharacteristic = pService->createCharacteristic(
                        CHARACTERISTIC_UUID,
                        BLECharacteristic::PROPERTY_READ   |
                        BLECharacteristic::PROPERTY_NOTIFY |
                        BLECharacteristic::PROPERTY_WRITE
                      );
  pCharacteristic->addDescriptor(new BLE2902());
  pService->start();
  
  BLEAdvertising *pAdvertising = BLEDevice::getAdvertising();
  pAdvertising->addServiceUUID(SERVICE_UUID);
  
  // Clean advertising for better Android compatibility
  BLEAdvertisementData advData;
  advData.setFlags(0x06);
  advData.setName("Physio_A");
  pAdvertising->setAdvertisementData(advData);

  pAdvertising->start();
  Serial.println("BLE Started as 'Physio_A'. Waiting for Flutter...");
}

// ================= SETUP =================
void setup() {
  Serial.begin(115200);
  analogReadResolution(12);
  analogSetPinAttenuation(BATT_PIN, ADC_11db);
  pinMode(SDA_1, INPUT_PULLUP);
  pinMode(SCL_1, INPUT_PULLUP);
  delay(100);
  Wire.begin(SDA_1, SCL_1);
  Wire.setClock(100000);
  Wire.setTimeOut(50);
  pinMode(BATT_PIN, INPUT); 
  pinMode(SWITCH_PIN, INPUT);   // external resistor used
  pinMode(LED_PIN, OUTPUT);
  int state = digitalRead(SWITCH_PIN); 
  if (state == HIGH) {
    // Switch ON → stay awake
    digitalWrite(LED_PIN, HIGH);
    Serial.println("Switch ON → Awake, LED ON");
  } else {
    // Switch OFF → go to sleep
    Serial.println("Switch OFF → Sleeping...");

    esp_deep_sleep_enable_gpio_wakeup(
      (1ULL << SWITCH_PIN),
      ESP_GPIO_WAKEUP_GPIO_HIGH   // 🔥 wake when HIGH
    );

    delay(100);
    esp_deep_sleep_start();
  }

// Add this to setup()
  Serial.println("Attempting to initialize Master Sensor...");
  if (bno1.begin(BNO_ADDR, Wire)) {
    Serial.println("BNO1: SUCCESS (Master sensor found)");
    bno1.enableGameRotationVector(50); 
    bno1_ok = true;
  } else { 
    Serial.println("BNO1: FAILED! Check pins 6(SDA) and 7(SCL) and Power.");
    // Try one more time with a reset if it failed
    delay(500);
    if (bno1.begin(BNO_ADDR, Wire)) {
       Serial.println("BNO1: Success on second attempt");
       bno1_ok = true;
    }
  }
  
  loadMountingCalibration();
  // Help BLE and ESP-NOW play nice
  esp_bt_controller_mem_release(ESP_BT_MODE_CLASSIC_BT); 

  // Slow down I2C to 100kHz for better stability against radio noise
  Wire.setClock(100000);
  WiFi.mode(WIFI_STA);
  esp_wifi_set_promiscuous(true);
  esp_wifi_set_channel(1, WIFI_SECOND_CHAN_NONE);
  esp_wifi_set_promiscuous(false);
  if (esp_now_init() != ESP_OK) return;
  esp_now_register_recv_cb(OnDataRecv);

  esp_now_peer_info_t peerInfo;
  memset(&peerInfo, 0, sizeof(peerInfo));
  memcpy(peerInfo.peer_addr, broadcastMac, 6);
  esp_now_add_peer(&peerInfo);

  bleQueue = xQueueCreate(300, sizeof(SensorSample));
  espnowQueue = xQueueCreate(300, sizeof(SensorSample));

  listenStartTime = millis();
  Serial.println("Listening for 30 seconds to determine role...");
}

// ================= LOOP =================
void loop() {
  //Switch state
  if (digitalRead(SWITCH_PIN) == LOW) {
    Serial.println("Switch turned OFF → Sleeping...");

    digitalWrite(LED_PIN, LOW);

    esp_deep_sleep_enable_gpio_wakeup(
      (1ULL << SWITCH_PIN),
      ESP_GPIO_WAKEUP_GPIO_HIGH   // wake when switch ON again
    );

    delay(100);
    esp_deep_sleep_start();
  }
  // --- 1. INITIALIZATION PHASE (FORCED MASTER) ---
  if (currentState == STATE_INIT_LISTEN) {
      currentState = STATE_MASTER;
      mySlaveId = 0; 
      setupBLE(); // Turn on Bluetooth immediately
      Serial.println(">>> INSTANT MASTER MODE ENABLED <<<");
  }

// --- 2. MASTER PHASE (FORCE DATA OUT) ---
  else if (currentState == STATE_MASTER) {
    static SensorSample masterLatest;
    static SensorSample slaveLatest;
    static bool freshMaster = false;
    static bool freshSlave = false;

    // 1. Read Master Sensor
    SensorSample tempRead;
    if (readMySensor(tempRead, 0)) {
      masterLatest = tempRead;
      freshMaster = true;
    }

    // 2. Try to get Slave data
    SensorSample tempSlave;
    if (xQueueReceive(bleQueue, &tempSlave, 0) == pdTRUE) {
      slaveLatest = tempSlave;
      freshSlave = true;
    }

    // 3. FORCE OUTPUT (Even if Slave is missing)
    if (millis() - lastMasterBleSend > BLE_INTERVAL) {
      lastMasterBleSend = millis();

      char payload[120];
      // If Slave is stuck, we use 0,0,0,1 as a placeholder so the code doesn't stop
      snprintf(payload, sizeof(payload),
               "%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f",
               masterLatest.x, masterLatest.y, masterLatest.z, masterLatest.w,
               freshSlave ? slaveLatest.x : 0.0, 
               freshSlave ? slaveLatest.y : 0.0, 
               freshSlave ? slaveLatest.z : 0.0, 
               freshSlave ? slaveLatest.w : 1.0);

      if (deviceConnected) {
          pCharacteristic->setValue(payload);
          pCharacteristic->notify();
      }

      // THIS WILL FINALLY PRINT DATA TO YOUR SCREEN
      Serial.println(payload);

      freshMaster = false;
      freshSlave = false;
    }
  }


  // --- UART Calibration Checks ---
  if (Serial.available()) {
    char c = Serial.read();
    
    if (c == 'c' || c == 'h') {
      // 1. Calibrate this specific board's sensor
      if (c == 'c') calibrateFullTare();
      if (c == 'h') calibrateHeadingOnly();

      // 2. If this is the Master, tell the Slaves to do it too
      if (currentState == STATE_MASTER) {
        CalibrateMsg calibMsg;
        calibMsg.type = MSG_CALIBRATE;
        calibMsg.calibType = (c == 'c') ? 0 : 1; 

        esp_now_send(broadcastMac, (uint8_t *) &calibMsg, sizeof(calibMsg));
        Serial.println(">>> Broadcasted Calibration Command to all Slaves!");
      }
    }
  }
}