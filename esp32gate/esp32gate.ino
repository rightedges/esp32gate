#include "model_data.h"
#include <EloquentTinyML.h>
#include <WiFi.h>
#include <eloquent_tinyml/tensorflow.h>
#include <esp_system.h>

// Define constants before including headers that depend on them
#define NUMBER_OF_INPUTS 4096 // 64 * 64
#define NUMBER_OF_OUTPUTS 2
#define TENSOR_ARENA_SIZE 256 * 1024 // 256KB for safety, using PSRAM

// ROI Settings (from roi.json)
const int ROI_X = 230;
const int ROI_Y = 174;
const int ROI_X1 = 643;
const int ROI_Y1 = 324;
const int ROI_W = ROI_X1 - ROI_X; // 413
const int ROI_H = ROI_Y1 - ROI_Y; // 150

// Global buffers removed - managed by DEV_DoorsWindows.h now
// Use global static allocation - more stable for DRAM tracking
// Eloquent::TinyML::TensorFlow::TensorFlow<NUMBER_OF_INPUTS, NUMBER_OF_OUTPUTS,
//                                          TENSOR_ARENA_SIZE>
//     ml;
Eloquent::TinyML::TensorFlow::TensorFlow<NUMBER_OF_INPUTS, NUMBER_OF_OUTPUTS,
                                         TENSOR_ARENA_SIZE> *ml;

// Shared Globals
unsigned long CHECKINTERVAL = 10000;
unsigned long GATECLOSETIME = 15000;
unsigned long lastTime = 0;
unsigned long remoteButtonDelay = 1500;
String gateStatus = "unknown";

// The HomeSpan header must be included after the ML types are defined
#include "DEV_DoorsWindows.h"
#include "HomeSpan.h"

// === WiFi Configuration ===
const char *ssid = "cloud1";
const char *password = "pccw1234";

// === Camera Configuration ===
const char *camera_url = "http://192.168.50.82/ISAPI/ContentMgmt/"
                         "StreamingProxy/channels/801/picture?cmd=refresh";
const char *camera_user = "admin";
const char *camera_pass = "pccw1234";

// === Server Capture API ===
// REPLACE WITH YOUR SERVER IP!
const char *server_capture_url = "http://192.168.50.231:5001/capture";

void setup() {
  // Serial must be absolutely first
  Serial.begin(115200);
  delay(2000); // Longer delay to catch Serial
  Serial.println("\n\n--- BREADCRUMB: setup() started ---");

  // Check PSRAM
  if (psramFound()) {
    Serial.printf("PSRAM Found! Size: %d bytes\n", ESP.getPsramSize());
    Serial.printf("Free PSRAM: %d bytes\n", ESP.getFreePsram());
  } else {
    Serial.println("WARNING: PSRAM NOT FOUND! Optimization will fail.");
  }

  // Allocate ML in PSRAM
  // Allocate ML in PSRAM - Explicitly use persistent allocation with placement
  // new if library doesn't support it directly, but here we know this object is
  // huge so we want it on heap (PSRAM hopefully) The library allocates arena
  // internally. We need to make sure the *object* is in PSRAM if the arena is
  // static inside it?
  // Actually, EloquentTinyML usually allocates arena as a member array.
  // If we `new` the object, the member array is on the heap where `new`
  // allocated it. So `ps_malloc` logic is correct.

  void *ml_mem = ps_malloc(
      sizeof(Eloquent::TinyML::TensorFlow::TensorFlow<
             NUMBER_OF_INPUTS, NUMBER_OF_OUTPUTS, TENSOR_ARENA_SIZE>));
  if (!ml_mem) {
    Serial.println("CRITICAL ERROR: Failed to allocate ML memory in PSRAM!");
    while (1)
      delay(1000);
  }
  ml = new (ml_mem) Eloquent::TinyML::TensorFlow::TensorFlow<
      NUMBER_OF_INPUTS, NUMBER_OF_OUTPUTS, TENSOR_ARENA_SIZE>();

  if (!ml) {
    Serial.println("CRITICAL ERROR: Failed to allocate ML object!");
    while (1)
      delay(1000);
  }

  // 1. Connect to WiFi
  Serial.print("Connecting to WiFi...");
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nWiFi connected");

  // 2. Initialize ML
  Serial.print("Initializing TinyML... ");
  if (!ml->begin(gate_detector_model)) {
    Serial.print("Model initialization failed! Error: ");
    Serial.println(ml->getErrorMessage());
  } else {
    Serial.println("TinyML Model Initialized Successfully!");
  }

  Serial.print("Free heap: ");
  Serial.println(ESP.getFreeHeap());

  // 3. Initialize HomeSpan
  Serial.println("Starting HomeSpan...");
  homeSpan.enableWebLog(10, "time4.google.com", "UTC+8:00", "log");
  homeSpan.begin(Category::Bridges, "HomeSpan Gate Bridge");

  new SpanAccessory();
  new Service::AccessoryInformation();
  new Characteristic::Identify();
  new Characteristic::Manufacturer("William's Labs");
  new Characteristic::Model("GateController-S3");

  new SpanAccessory();
  new Service::AccessoryInformation();
  new Characteristic::Identify();
  new Characteristic::Name("Main Gate");
  new DEV_GarageDoor(2); // Relay on Pin 2

  Serial.println("System Ready!");
}

void loop() { homeSpan.poll(); }
