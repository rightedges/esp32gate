#include "model_data.h"
#include <EloquentTinyML.h>
#include <HTTPClient.h>
#include <TJpg_Decoder.h>
#include <WiFi.h>
#include <base64.h>
#include <eloquent_tinyml/tensorflow.h>

// --- Configuration ---
const char *ssid = "cloud1";
const char *password = "pccw1234";
const char *camera_url = "http://192.168.50.82/ISAPI/ContentMgmt/"
                         "StreamingProxy/channels/801/picture?cmd=refresh";
const char *camera_user = "admin";
const char *camera_pass = "pccw1234";

// --- TinyML Settings ---
#define NUMBER_OF_INPUTS 4096 // 64 * 64
#define NUMBER_OF_OUTPUTS 2
#define TENSOR_ARENA_SIZE 120 * 1024

Eloquent::TinyML::TensorFlow::TensorFlow<NUMBER_OF_INPUTS, NUMBER_OF_OUTPUTS,
                                         TENSOR_ARENA_SIZE>
    ml;
float *model_input = nullptr;

// --- ROI Settings (from roi.json) ---
#define ROI_X 230
#define ROI_Y 174
#define ROI_X1 643
#define ROI_Y1 324
#define ROI_W (ROI_X1 - ROI_X)
#define ROI_H (ROI_Y1 - ROI_Y)

// Buffers for resizing (Moving to heap to save DRAM)
uint16_t *pixel_sums = nullptr;
uint8_t *pixel_counts = nullptr;
uint16_t global_w = 0, global_h = 0;

// JPEG Decoder Callback: Just like the Python logic
bool tjpg_callback(int16_t x, int16_t y, uint16_t w, uint16_t h,
                   uint16_t *bitmap) {
  if (global_w == 0 || global_h == 0)
    return true;

  for (int16_t j = 0; j < h; j++) {
    for (int16_t i = 0; i < w; i++) {
      int16_t abs_x = x + i;
      int16_t abs_y = y + j;

      // Only process pixels within ROI
      if (abs_x >= ROI_X && abs_x < ROI_X1 && abs_y >= ROI_Y &&
          abs_y < ROI_Y1) {
        uint16_t pixel = bitmap[j * w + i];
        // RGB565 to Grayscale
        uint8_t r = (pixel >> 11) & 0x1F;
        uint8_t g = (pixel >> 5) & 0x3F;
        uint8_t b = pixel & 0x1F;
        uint8_t gray =
            (uint8_t)((r * 255 / 31) * 0.299 + (g * 255 / 63) * 0.587 +
                      (b * 255 / 31) * 0.114);

        // Map to 64x64 grid
        int target_x = (abs_x - ROI_X) * 64 / ROI_W;
        int target_y = (abs_y - ROI_Y) * 64 / ROI_H;

        if (target_x >= 0 && target_x < 64 && target_y >= 0 && target_y < 64) {
          int idx = target_y * 64 + target_x;
          pixel_sums[idx] += gray;
          pixel_counts[idx]++;
        }
      }
    }
  }
  return true;
}

#define MAX_JPG_SIZE 80 * 1024
uint8_t *jpg_buffer = nullptr;

// ... (ROI remains the same)

void performInference() {
  Serial.printf("\nFree Heap: %d bytes\n", ESP.getFreeHeap());
  Serial.println("--- Starting Fetch ---");
  HTTPClient http;
  http.begin(camera_url);
  String auth = String(camera_user) + ":" + String(camera_pass);
  http.addHeader("Authorization", "Basic " + base64::encode(auth));

  int httpCode = http.GET();
  Serial.printf("HTTP Code: %d\n", httpCode);

  if (httpCode == HTTP_CODE_OK) {
    int len = http.getSize();
    Serial.printf("Content Length: %d bytes\n", len);

    // 1. Allocate JPG Buffer
    jpg_buffer = (uint8_t *)malloc(len);
    if (!jpg_buffer) {
      Serial.printf("JPG Buffer Malloc Failed! Free Heap: %d\n",
                    ESP.getFreeHeap());
      http.end();
      return;
    }

    if (!jpg_buffer) {
      Serial.printf("JPG Buffer Malloc Failed! Free Heap: %d, Free PSRAM: %d\n",
                    ESP.getFreeHeap(), ESP.getFreePsram());
      http.end();
      return;
    }

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
      // 2. Allocate Temp Sums/Counts (Reduced size for 64x64)
      pixel_sums = (uint16_t *)malloc(64 * 64 * sizeof(uint16_t));
      pixel_counts = (uint8_t *)malloc(64 * 64 * sizeof(uint8_t));

      if (pixel_sums && pixel_counts) {
        memset(pixel_sums, 0, 64 * 64 * sizeof(uint16_t));
        memset(pixel_counts, 0, 64 * 64 * sizeof(uint8_t));

        // 3. Decode
        if (TJpgDec.getJpgSize(&global_w, &global_h, jpg_buffer, len) ==
            JDR_OK) {
          Serial.printf("Image Res: %dx%d\n", global_w, global_h);
          if (TJpgDec.drawJpg(0, 0, jpg_buffer, len) == JDR_OK) {
            // 4. FREE JPG Buffer immediately
            free(jpg_buffer);
            jpg_buffer = NULL;

            // 5. Allocate Model Input (Float) - Library handles quantization
            model_input = (float *)malloc(NUMBER_OF_INPUTS * sizeof(float));

            if (model_input) {
              float total_gray = 0;
              int filled_cells = 0;
              for (int i = 0; i < 64 * 64; i++) {
                if (pixel_counts[i] > 0) {
                  float avg = (float)pixel_sums[i] / pixel_counts[i];
                  // Normalize to 0.0 - 1.0
                  model_input[i] = avg / 255.0f;
                  total_gray += avg;
                  filled_cells++;
                } else {
                  model_input[i] = 0.0f;
                }
              }

              // 6. Free Sums/Counts
              free(pixel_sums);
              pixel_sums = NULL;
              free(pixel_counts);
              pixel_counts = NULL;

              if (filled_cells > 0) {
                Serial.printf("ROI Avg Brightness: %.2f\n",
                              total_gray / filled_cells);

                // Visualizer (Sample 16x32)
                Serial.println("--- 64x64 ROI Preview ---");
                for (int y = 0; y < 64; y += 4) {
                  for (int x = 0; x < 64; x += 2) {
                    float val = model_input[y * 64 + x];
                    if (val < 0.25)
                      Serial.print(" ");
                    else if (val < 0.5)
                      Serial.print(".");
                    else if (val < 0.75)
                      Serial.print("x");
                    else
                      Serial.print("M");
                  }
                  Serial.println();
                }
                Serial.println("-------------------------");
              } else {
                Serial.println("WARNING: No pixels were processed in ROI!");
              }

              // 7. Predict (Standard Float Interface)
              Serial.print("Input Sample (0-9): ");
              for (int k = 0; k < 10; k++) {
                Serial.print(model_input[k]);
                Serial.print(", ");
              }
              Serial.println();

              float output[NUMBER_OF_OUTPUTS] = {0};
              ml.predict(model_input, output);

              Serial.print("Raw Output: ");
              Serial.print(output[0], 6);
              Serial.print(", ");
              Serial.print(output[1], 6);
              Serial.println();

              float prob_closed = output[0];
              float prob_open = output[1];

              Serial.printf(">>> Prediction: %s <<<\n",
                            (prob_open > prob_closed ? "OPEN" : "CLOSED"));
              Serial.printf(
                  ">>> Confidence: %.2f%% [Closed: %.4f, Open: %.4f] <<<\n",
                  (prob_open > prob_closed ? prob_open : prob_closed) * 100.0,
                  prob_closed, prob_open);

              // 8. Free Model Input
              free(model_input);
              model_input = NULL;

            } else {
              Serial.println("Model Input Malloc Failed");
              // Clean up previously allocated buffers if model_input fails
              free(pixel_sums);
              pixel_sums = NULL;
              free(pixel_counts);
              pixel_counts = NULL;
            }
          } else {
            Serial.println("Jpg Draw Failed!");
            // Clean up jpg_buffer if draw fails
            free(jpg_buffer);
            jpg_buffer = NULL;
          }
        } else {
          Serial.println("Jpg Size Failed!");
          // Clean up jpg_buffer if getJpgSize fails
          free(jpg_buffer);
          jpg_buffer = NULL;
        }
      } else {
        Serial.println("Pixel Buffers Malloc Failed");
        if (pixel_sums)
          free(pixel_sums);
        if (pixel_counts)
          free(pixel_counts);
        if (jpg_buffer)
          free(jpg_buffer); // Also free jpg_buffer if pixel buffers fail
      }
    } else {
      Serial.printf("Read Timeout/Partial: %d/%d\n", totalRead, len);
      // Clean up jpg_buffer if read fails
      free(jpg_buffer);
      jpg_buffer = NULL;
    }
  }
  http.end();
}

void setup() {
  Serial.begin(115200);
  delay(2000);
  Serial.println("\n--- NEW SIMPLIFIED GATE TESTER ---");

  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nWiFi Connected");

  if (!ml.begin(gate_detector_model)) {
    Serial.print("TinyML Begin Failed: ");
    Serial.println(ml.getErrorMessage());
  } else {
    Serial.println("TinyML Initialized");
  }

  TJpgDec.setCallback(tjpg_callback);
}

void loop() {
  performInference();
  delay(15000); // Check every 15 seconds
}
