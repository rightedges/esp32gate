#ifndef DEV_DOORSWINDOWS_H
#define DEV_DOORSWINDOWS_H

#include "HomeSpan.h"
#include "model_data.h"
#include <Arduino.h>
#include <EloquentTinyML.h>
#include <HTTPClient.h>
#include <TJpg_Decoder.h>
#include <WiFi.h>
#include <base64.h>
#include <eloquent_tinyml/tensorflow.h>

// Constants already defined in .ino
extern Eloquent::TinyML::TensorFlow::TensorFlow<
    NUMBER_OF_INPUTS, NUMBER_OF_OUTPUTS, TENSOR_ARENA_SIZE> *ml;
extern float *model_input;
extern uint8_t *roi_buffer;

// ROI Settings (Confirmed working OLD ROI)
#define ROI_X 230
#define ROI_Y 174
#define ROI_X1 643
#define ROI_Y1 324
#define ROI_W (ROI_X1 - ROI_X)
#define ROI_H (ROI_Y1 - ROI_Y)

// Camera Configuration
extern const char *camera_url;
extern const char *camera_user;
extern const char *camera_pass;

// Timing and Status
extern unsigned long CHECKINTERVAL;
extern unsigned long GATECLOSETIME;
extern unsigned long lastTime;
extern unsigned long remoteButtonDelay;
extern String gateStatus;

// Accumulators for Mean Pooling (Allocated in Constructor in PSRAM)
static uint16_t *pixel_sums = nullptr;
static uint8_t *pixel_counts = nullptr;
static uint8_t *jpg_buffer = nullptr;
static float *inference_input = nullptr;

// Diagnostic: Whole Frame preview (16x16)
static uint32_t world_sums[16 * 16];
static uint8_t world_counts[16 * 16];
static uint16_t global_w = 0, global_h = 0;

#define MAX_JPG_SIZE (256 * 1024) // 256KB buffer for JPG

// JPEG Decoder Callback
inline bool tjpg_callback(int16_t x, int16_t y, uint16_t w, uint16_t h,
                          uint16_t *bitmap) {
  if (global_w == 0 || global_h == 0)
    return true;

  for (int16_t j = 0; j < h; j++) {
    for (int16_t i = 0; i < w; i++) {
      int16_t abs_x = x + i;
      int16_t abs_y = y + j;

      uint16_t pixel = bitmap[j * w + i];

      // RGB565 to Grayscale
      uint8_t r = (pixel >> 11) & 0x1F;
      uint8_t g = (pixel >> 5) & 0x3F;
      uint8_t b = pixel & 0x1F;
      uint8_t gray = (uint8_t)((r * 255 / 31) * 0.299 + (g * 255 / 63) * 0.587 +
                               (b * 255 / 31) * 0.114);

      // 1. Process for AI (ROI only)
      if (abs_x >= ROI_X && abs_x < ROI_X1 && abs_y >= ROI_Y &&
          abs_y < ROI_Y1) {
        // Map to 64x64 grid
        int target_x = (abs_x - ROI_X) * 64 / ROI_W;
        int target_y = (abs_y - ROI_Y) * 64 / ROI_H;

        if (target_x >= 0 && target_x < 64 && target_y >= 0 && target_y < 64) {
          int idx = target_y * 64 + target_x;
          if (pixel_sums && pixel_counts) {
            pixel_sums[idx] += gray;
            pixel_counts[idx]++;
          }
        }
      }

      // 2. Process for Diagnostic (Whole frame 16x16)
      int wx = (int)((uint32_t)abs_x * 16 / global_w);
      int wy = (int)((uint32_t)abs_y * 16 / global_h);

      if (wx >= 0 && wx < 16 && wy >= 0 && wy < 16) {
        int widx = wy * 16 + wx;
        world_sums[widx] += gray;
        world_counts[widx]++;
      }
    }
  }
  return true;
}

struct DEV_GarageDoor : Service::GarageDoorOpener {

  SpanCharacteristic *current;
  SpanCharacteristic *target;
  SpanCharacteristic *obstruction;
  int relayInPin;
  unsigned long numDiffState = 0;
  const unsigned long NUMCHECK = 2;

  DEV_GarageDoor(int relayInPin) : Service::GarageDoorOpener() {
    current = new Characteristic::CurrentDoorState(1); // closed
    target = new Characteristic::TargetDoorState(1);   // closed
    obstruction = new Characteristic::ObstructionDetected(false);
    this->relayInPin = relayInPin;
    pinMode(relayInPin, OUTPUT);
    digitalWrite(relayInPin, HIGH);

    // Allocate persistent buffers in PSRAM
    if (psramFound()) {
      jpg_buffer = (uint8_t *)ps_malloc(MAX_JPG_SIZE);
      pixel_sums = (uint16_t *)ps_malloc(64 * 64 * sizeof(uint16_t));
      pixel_counts = (uint8_t *)ps_malloc(64 * 64 * sizeof(uint8_t));
      inference_input = (float *)ps_malloc(NUMBER_OF_INPUTS * sizeof(float));
    } else {
      Serial.println("WARNING: PSRAM NOT FOUND! Attempting malloc...");
      // Fallback or halt? For now fallback but it will likely fail for 8MB
      // needs
      jpg_buffer = (uint8_t *)malloc(MAX_JPG_SIZE);
      pixel_sums = (uint16_t *)malloc(64 * 64 * sizeof(uint16_t));
      pixel_counts = (uint8_t *)malloc(64 * 64 * sizeof(uint8_t));
      inference_input = (float *)malloc(NUMBER_OF_INPUTS * sizeof(float));
    }

    if (!jpg_buffer || !pixel_sums || !pixel_counts || !inference_input) {
      Serial.println("CRITICAL: Failed to allocate buffers!");
    } else {
      Serial.println("Buffers successfully allocated (PSRAM preferred).");
    }

    TJpgDec.setCallback(tjpg_callback);
    Serial.print(
        F("Configuring HomeSpan Garage Door (v1.7: Deep Introspection)\n"));
  }

  boolean update() {
    if (target->getNewVal() == current->getVal())
      return (true);

    if (target->getNewVal() == 0) { // Opening
      WEBLOG("HomeKit: Opening Gate");
      current->setVal(2); // opening
      digitalWrite(relayInPin, LOW);
      delay(remoteButtonDelay);
      digitalWrite(relayInPin, HIGH);
    } else { // Closing
      WEBLOG("HomeKit: Closing Gate");
      current->setVal(3); // closing
      digitalWrite(relayInPin, LOW);
      delay(remoteButtonDelay);
      digitalWrite(relayInPin, HIGH);
    }
    return (true);
  }

  void loop() {
    if ((millis() - lastTime) > CHECKINTERVAL) {
      lastTime = millis();

      if (WiFi.status() == WL_CONNECTED) {
        performInference();
        updateHomeKitStatus();
      }
    }

    if (current->getVal() != target->getVal()) {
      if (current->getVal() == 4)
        return; // Stopped

      if (target->timeVal() > GATECLOSETIME) {
        current->setVal(target->getVal());
        WEBLOG("Gate transition timed out. Setting to target state.");
      }
    }
  }

  void performInference() {
    HTTPClient http;
    http.begin(camera_url);
    String auth = String(camera_user) + ":" + String(camera_pass);
    String authEncoded = base64::encode(auth);
    http.addHeader("Authorization", "Basic " + authEncoded);

    int httpCode = http.GET();
    if (httpCode == HTTP_CODE_OK) {
      int len = http.getSize();
      if (len > 0 && len < MAX_JPG_SIZE) {
        if (jpg_buffer) {
          WiFiClient *stream = http.getStreamPtr();
          int totalRead = 0;
          unsigned long start = millis();

          while (totalRead < len && (millis() - start < 10000)) {
            if (stream->available()) {
              int read = stream->read(jpg_buffer + totalRead, len - totalRead);
              if (read > 0)
                totalRead += read;
            }
          }

          if (totalRead == len) {
            // SEQUENTIAL ALLOCATION: Alloc Sums/Counts - ALREADY ALLOCATED
            if (pixel_sums && pixel_counts) {
              memset(pixel_sums, 0, 64 * 64 * sizeof(uint16_t));
              memset(pixel_counts, 0, 64 * 64 * sizeof(uint8_t));
              memset(world_sums, 0, sizeof(world_sums));
              memset(world_counts, 0, sizeof(world_counts));

              if (TJpgDec.getJpgSize(&global_w, &global_h, jpg_buffer, len) ==
                  JDR_OK) {
                if (TJpgDec.drawJpg(0, 0, jpg_buffer, len) == JDR_OK) {

                  // JPG Buffer is reusable, no free needed

                  // ALLOCATE MODEL INPUT (Float) - ALREADY ALLOCATED
                  if (inference_input) {
                    float totalSum = 0;
                    int totalPixels = 0;
                    for (int i = 0; i < 64 * 64; i++) {
                      if (pixel_counts[i] > 0) {
                        uint8_t avg_gray =
                            (uint8_t)(pixel_sums[i] / pixel_counts[i]);
                        inference_input[i] = (float)avg_gray / 255.0f;
                        totalSum += avg_gray;
                        totalPixels++;
                      } else {
                        inference_input[i] = 0.0f;
                      }
                    }

                    // NO FREE NEEDED

                    if (totalPixels > 0) {
                      float output[NUMBER_OF_OUTPUTS] = {0};
                      ml->predict(inference_input, output);
                      float prob_closed = output[0];
                      float prob_open = output[1];

                      gateStatus =
                          (prob_open > prob_closed) ? "open" : "closed";

                      WEBLOG("TinyML: %dx%d Gate is %s (Scores: %.2f, %.2f) "
                             "Avg: %ld",
                             global_w, global_h, gateStatus.c_str(),
                             prob_closed, prob_open, totalSum / totalPixels);
                    }
                  } else {
                    WEBLOG("TinyML: Inference Input Buf Missing");
                  }
                } else {
                  WEBLOG("TinyML: Draw Failed");
                }
              } else {
                WEBLOG("TinyML: Size Failed");
              }
            } else {
              WEBLOG("TinyML: Pixel Buffers Missing");
            }
          } else {
            WEBLOG("TinyML: Stream incomplete (%d/%d)", totalRead, len);
          }
        } else {
          WEBLOG("TinyML: JPG Buf Missing");
        }
      } else {
        if (len >= MAX_JPG_SIZE) {
          WEBLOG("TinyML: Image too large (%d > %d)", len, MAX_JPG_SIZE);
        }
      }
    } else {
      WEBLOG("TinyML: GET Fail (%d). WiFi RSSI: %ld, IP: %s", httpCode,
             WiFi.RSSI(), WiFi.localIP().toString().c_str());
    }
    http.end();
  }

  void updateHomeKitStatus() {
    if (current->getVal() == 0 && gateStatus == "closed") {
      numDiffState++;
      if (numDiffState >= NUMCHECK) {
        target->setVal(1);
        current->setVal(1);
        WEBLOG("HomeKit: Syncing -> CLOSED");
        numDiffState = 0;
      }
    } else if (current->getVal() == 1 && gateStatus == "open") {
      numDiffState++;
      if (numDiffState >= NUMCHECK) {
        target->setVal(0);
        current->setVal(0);
        WEBLOG("HomeKit: Syncing -> OPEN");
        numDiffState = 0;
      }
    } else {
      numDiffState = 0;
    }
  }
};

#endif
