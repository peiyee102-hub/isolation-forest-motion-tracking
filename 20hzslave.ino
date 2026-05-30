#include <WiFi.h>
#include <esp_now.h>
#include <esp_wifi.h>
#include <Wire.h>
#include "SparkFun_BNO080_Arduino_Library.h"
#include <Preferences.h>

// ================= CONFIGURATION =================
#define SDA_1 6
#define SCL_1 7
#define BNO_ADDR 0x4B
#define SAMPLES_PER_CHUNK 1 

uint8_t masterAddress[] = {0x58, 0x8C, 0x81, 0xAD, 0x02, 0x50};

struct Quat { float x, y, z, w; };
Quat offset1 = {0, 0, 0, 1};
BNO080 bno1;
bool bno1_ok = false;
Preferences prefs;

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

// ================= STRUCTS =================
typedef struct __attribute__((packed)) {
  uint8_t id;
  uint32_t timestamp;
  uint8_t battery;
  float x, y, z, w;
} SensorSample;

typedef struct __attribute__((packed)) {
  uint8_t type; 
  uint8_t count;
  SensorSample samples[SAMPLES_PER_CHUNK];
} DataChunkMsg;

// ================= CALIBRATION FUNCTION =================
void calibrateFullTare() {
    if (bno1_ok) {
        float rawI = bno1.getQuatI();
        float rawJ = bno1.getQuatJ();
        float rawK = bno1.getQuatK();
        float rawReal = bno1.getQuatReal();
        
        Quat raw = {rawI, rawJ, rawK, rawReal};
        offset1 = qNorm(raw);  // normalize before saving

        prefs.begin("mount", false);
        prefs.putFloat("ox", offset1.x);
        prefs.putFloat("oy", offset1.y);
        prefs.putFloat("oz", offset1.z);
        prefs.putFloat("ow", offset1.w);
        prefs.end();
        Serial.println(">>> SLAVE: Full Tare Complete.");
    }
}

// ================= CALLBACKS =================
void OnDataRecv(const esp_now_recv_info_t *info, const uint8_t *data, int len) {
    if (data[0] == 3) { 
        uint8_t calibType = data[1];
        if (calibType == 0) calibrateFullTare();
    }
}

// ================= SETUP =================
void setup() {
  Serial.begin(115200);

  WiFi.mode(WIFI_STA);
  esp_wifi_set_promiscuous(true);
  esp_wifi_set_channel(1, WIFI_SECOND_CHAN_NONE);
  esp_wifi_set_promiscuous(false);

  pinMode(SDA_1, INPUT_PULLUP);
  pinMode(SCL_1, INPUT_PULLUP);
  delay(100);

  Wire.begin(SDA_1, SCL_1);
  Wire.setClock(10000); 

  Serial.println("Starting BNO080...");
  if (bno1.begin(BNO_ADDR, Wire)) {
    bno1.enableGameRotationVector(50);
    bno1_ok = true;
    Serial.println("BNO1: SUCCESS");
  } else {
    Serial.println("BNO1: FAILED.");
  }

  if (esp_now_init() != ESP_OK) return;
  esp_now_register_recv_cb(OnDataRecv);

  esp_now_peer_info_t peerInfo = {};
  memcpy(peerInfo.peer_addr, masterAddress, 6);
  peerInfo.channel = 1;  
  peerInfo.encrypt = false;
  esp_now_add_peer(&peerInfo);
  
  prefs.begin("mount", true);
  if (prefs.isKey("ox")) {
    offset1.x = prefs.getFloat("ox");
    offset1.y = prefs.getFloat("oy");
    offset1.z = prefs.getFloat("oz");
    offset1.w = prefs.getFloat("ow");
    Serial.println(">>> Loaded saved calibration offset.");
  }
  prefs.end();
} // Fixed the missing bracket here!

// ================= LOOP =================
void loop() {
  if (bno1_ok && bno1.dataAvailable()) {
    DataChunkMsg chunk;
    chunk.type = 2; 
    chunk.count = 1;

    float rawI = bno1.getQuatI();
    float rawJ = bno1.getQuatJ();
    float rawK = bno1.getQuatK();
    float rawReal = bno1.getQuatReal();

    // APPLY MATH
    Quat currentRaw = {rawI, rawJ, rawK, rawReal};
    Quat calibrated = qNorm(qMul(qInv(offset1), currentRaw));

    chunk.samples[0].id = 1; 
    chunk.samples[0].x = calibrated.x;
    chunk.samples[0].y = calibrated.y;
    chunk.samples[0].z = calibrated.z;
    chunk.samples[0].w = calibrated.w;
   
    esp_now_send(masterAddress, (uint8_t *) &chunk, sizeof(chunk));
    delay(20); 
  }
}