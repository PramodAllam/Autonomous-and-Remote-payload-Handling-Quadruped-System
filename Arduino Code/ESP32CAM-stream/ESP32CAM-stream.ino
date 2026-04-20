#include "esp_camera.h"
#include <WiFi.h>
#include "esp_http_server.h"

// ==== WiFi Credentials ====
const char* ssid = "******";
const char* password = "******";

// ==== AI Thinker Camera Pin Map ====
#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27
#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5
#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22

#define LED_GPIO_NUM       4
#define PWM_FREQ           5000
#define PWM_RESOLUTION     8  // 0–255

// Hysteresis thresholds
const int on_threshold = 100;
const int off_threshold = 160;
bool led_on = false;

httpd_handle_t stream_httpd = NULL;

// ========== MJPEG Stream ==========
static esp_err_t stream_handler(httpd_req_t *req) {
  camera_fb_t *fb = NULL;
  esp_err_t res = ESP_OK;
  char buf[64];

  res = httpd_resp_set_type(req, "multipart/x-mixed-replace; boundary=frame");
  if (res != ESP_OK) return res;

  while (true) {
    fb = esp_camera_fb_get();
    if (!fb) continue;

    // Calculate brightness from grayscale approximation
    long sum = 0;
    int count = 0;
    const int sample_step = 40;

    for (int i = 0; i < fb->len; i += sample_step) {
      sum += fb->buf[i];
      count++;
    }
    int avg_brightness = sum / count;
    //Serial.printf("Brightness Avg= %d\n",avg_brightness);

    // LED control with hysteresis
    if (!led_on && avg_brightness < on_threshold) {
      ledcWrite(LED_GPIO_NUM, 10);  // 50% brightness
      led_on = true;
      //Serial.println("LED ON (dark)");
    } else if (led_on && avg_brightness > off_threshold) {
      ledcWrite(LED_GPIO_NUM, 0);    // OFF
      led_on = false;
      //Serial.println("LED OFF (bright)");
    }

    // Stream frame
    res = httpd_resp_send_chunk(req, "--frame\r\n", strlen("--frame\r\n"));
    snprintf(buf, sizeof(buf), "Content-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n", fb->len);
    res = httpd_resp_send_chunk(req, buf, strlen(buf));
    res = httpd_resp_send_chunk(req, (const char *)fb->buf, fb->len);
    res = httpd_resp_send_chunk(req, "\r\n", 2);

    esp_camera_fb_return(fb);
    if (res != ESP_OK) break;

    delay(30); // ~33 FPS
  }

  return res;
}

// ========== Web Interface ==========
static esp_err_t index_handler(httpd_req_t *req) {
  const char* html = "<!DOCTYPE html><html><body><h2>ESP32-CAM Auto Flash</h2><img src='/stream'></body></html>";
  httpd_resp_set_type(req, "text/html");
  return httpd_resp_send(req, html, strlen(html));
}

void startCameraServer() {
  httpd_config_t config = HTTPD_DEFAULT_CONFIG();
  config.server_port = 80;

  httpd_uri_t stream_uri = {
    .uri       = "/stream",
    .method    = HTTP_GET,
    .handler   = stream_handler,
    .user_ctx  = NULL
  };

  httpd_uri_t index_uri = {
    .uri       = "/",
    .method    = HTTP_GET,
    .handler   = index_handler,
    .user_ctx  = NULL
  };

  if (httpd_start(&stream_httpd, &config) == ESP_OK) {
    httpd_register_uri_handler(stream_httpd, &stream_uri);
    httpd_register_uri_handler(stream_httpd, &index_uri);
  }
}

void setup() {
  delay(2000);
  Serial.begin(115200);

  // Attach PWM to LED
  ledcAttach(LED_GPIO_NUM, PWM_FREQ, PWM_RESOLUTION);
  ledcWrite(LED_GPIO_NUM, 0); // LED off at start

  // Camera config
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sscb_sda = SIOD_GPIO_NUM;
  config.pin_sscb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;
  config.frame_size = FRAMESIZE_QVGA;
  config.jpeg_quality = 12;
  config.fb_count = psramFound() ? 2 : 1;
  config.fb_location = psramFound() ? CAMERA_FB_IN_PSRAM : CAMERA_FB_IN_DRAM;
  config.grab_mode = CAMERA_GRAB_LATEST;

  // Camera init
  if (esp_camera_init(&config) != ESP_OK) {
    Serial.println("Camera init failed");
    return;
  }

  // Connect to WiFi
  WiFi.begin(ssid, password);
  WiFi.setSleep(false);
  Serial.print("Connecting to WiFi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  Serial.println("\nWiFi connected!");
  Serial.print("Stream ready at: http://");
  Serial.println(WiFi.localIP());

  startCameraServer();
}

void loop() {
  delay(10000);  // nothing to do here
}
