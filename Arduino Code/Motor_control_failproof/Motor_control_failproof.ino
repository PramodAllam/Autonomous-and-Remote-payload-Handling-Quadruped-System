#include <WiFi.h>
#include <WebServer.h>
#include <EEPROM.h>

// ================== Wi-Fi ==================
const char* ssid = "*****";
const char* password = "******";

// ================== Module 1 (Gripper) Pins ==================
const int pwm1 = 12;
const int in1a = 21;
const int in1b = 22;
const int enc1A = 32;   // input-only
const int enc1B = 33;
const int sw1A = 14;
const int sw1B = 27;


// ================== Module 2 Pins ==================
const int pwm2 = 25;       // ENA
const int in2a = 26;       // IN1
const int in2b = 23;       // IN2 (CHANGED from 27)
const int enc2A = 36;
const int enc2B = 39;
const int sw2A = 18;
const int sw2B = 19;

// ================== Globals ==================
volatile long encCount1 = 0;
volatile long encCount2 = 0;
long OPEN_COUNT1 = 1525;
long OPEN_COUNT2 = 1500;
const int motorPWM = 240;
const unsigned long maxDuration = 8000;

WebServer server(80);

// EEPROM config
#define EEPROM_SIZE 64
#define ADDR_COUNT1 0
#define ADDR_COUNT2 8

// ================== Encoder ISRs ==================
void IRAM_ATTR updateEnc1() {
  int b = digitalRead(enc1B);
  encCount1 += (b == HIGH) ? 1 : -1;
}
void IRAM_ATTR updateEnc2() {
  int b = digitalRead(enc2B);
  encCount2 += (b == HIGH) ? 1 : -1;
}

// ================== Motor Helpers ==================
void motorStop(int pwmPin) { analogWrite(pwmPin, 0); }

void motorDir(int inA, int inB, bool forward) {
  digitalWrite(inA, !forward);
  digitalWrite(inB, forward);
}

// ================== Switch Check ==================
bool bothPressed(int s1, int s2) {
  return (digitalRead(s1) == HIGH && digitalRead(s2) == HIGH);
}

// ================== Homing ==================
void goHome(int pwmPin, int inA, int inB, int s1, int s2, volatile long &encCount) {
  Serial.println("[Home] Starting homing...");

  if (bothPressed(s1, s2)) {
    noInterrupts();
    encCount = 0;
    interrupts();
    Serial.println("[Home] Already at home (encoder=0)");
    return;
  }

  motorDir(inA, inB, true);
  analogWrite(pwmPin, motorPWM);

  unsigned long start = millis();
  while (!bothPressed(s1, s2) && (millis() - start < maxDuration)) {
    delay(2);
  }

  motorStop(pwmPin);
  if (bothPressed(s1, s2)) {
    noInterrupts();
    encCount = 0;
    interrupts();
    Serial.println("[Home] Homed successfully");
  } else {
    Serial.println("[ERROR] Homing timeout");
  }
}

// ================== Open / Close ==================
void openMotor(int pwmPin, int inA, int inB, volatile long &encCount, long openCount) {
  Serial.println("[Motor] Opening...");
  encCount = 0;
  motorDir(inA, inB, false);
  analogWrite(pwmPin, motorPWM);

  unsigned long start = millis();
  while (abs(encCount) < openCount && (millis() - start < maxDuration)) delay(1);

  motorStop(pwmPin);
  Serial.println("[Motor] Open complete");
}

void closeMotor(int pwmPin, int inA, int inB,
                int s1, int s2,
                volatile long &encCount) {

  Serial.println("[Motor] Closing...");

  motorDir(inA, inB, true);
  analogWrite(pwmPin, motorPWM);

  unsigned long start = millis();

  while (millis() - start < maxDuration) {

    bool sw1 = digitalRead(s1) == HIGH;
    bool sw2 = digitalRead(s2) == HIGH;
    bool encoderAtHome = (encCount >= 0);
    bool anySwitch = (sw1 || sw2);

    if ((sw1 && sw2) || (encoderAtHome && anySwitch)) {
        Serial.println("[Motor] Close condition satisfied");
        break;
    }

    delay(2);
}

  motorStop(pwmPin);

  if (digitalRead(s1) == HIGH && digitalRead(s2) == HIGH) {
    noInterrupts();
    encCount = 0;
    interrupts();
    Serial.println("[Motor] Closed and re-synced");
  }
}


// ================== HTTP Handlers ==================

// Module 1
void handleOpen1() { openMotor(pwm1, in1a, in1b, encCount1, OPEN_COUNT1); server.send(200, "text/plain", "Module1 opened"); }
void handleClose1() { closeMotor(pwm1, in1a, in1b, sw1A, sw1B, encCount1); server.send(200, "text/plain", "Module1 closed"); }
void handleStatus1() {
  String msg = "M1 Enc=" + String(encCount1) +
               " SW1A=" + String(digitalRead(sw1A)) +
               " SW1B=" + String(digitalRead(sw1B)) +
               " OPEN_COUNT1=" + String(OPEN_COUNT1);
  server.send(200, "text/plain", msg);
}
void handleSetCount1() {
  if (server.hasArg("value")) {
    OPEN_COUNT1 = server.arg("value").toInt();
    EEPROM.put(ADDR_COUNT1, OPEN_COUNT1);
    EEPROM.commit();
    String msg = "M1 OPEN_COUNT updated to " + String(OPEN_COUNT1);
    Serial.println(msg);
    server.send(200, "text/plain", msg);
  } else server.send(400, "text/plain", "Missing value param");
}

// Module 2
void handleOpen2() { openMotor(pwm2, in2a, in2b, encCount2, OPEN_COUNT2); server.send(200, "text/plain", "Module2 opened"); }
void handleClose2() { closeMotor(pwm2, in2a, in2b, sw2A, sw2B, encCount2); server.send(200, "text/plain", "Module2 closed"); }
void handleStatus2() {
  String msg = "M2 Enc=" + String(encCount2) +
               " SW2A=" + String(digitalRead(sw2A)) +
               " SW2B=" + String(digitalRead(sw2B)) +
               " OPEN_COUNT2=" + String(OPEN_COUNT2);
  server.send(200, "text/plain", msg);
}
void handleSetCount2() {
  if (server.hasArg("value")) {
    OPEN_COUNT2 = server.arg("value").toInt();
    EEPROM.put(ADDR_COUNT2, OPEN_COUNT2);
    EEPROM.commit();
    String msg = "M2 OPEN_COUNT updated to " + String(OPEN_COUNT2);
    Serial.println(msg);
    server.send(200, "text/plain", msg);
  } else server.send(400, "text/plain", "Missing value param");
}

// ================== Setup ==================
void setup() {
  Serial.begin(115200);

  // Module 1 pins
  pinMode(in1a, OUTPUT); pinMode(in1b, OUTPUT); pinMode(pwm1, OUTPUT);
  pinMode(enc1A, INPUT_PULLUP); pinMode(enc1B, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(enc1A), updateEnc1, RISING);
  pinMode(sw1A, INPUT_PULLUP); pinMode(sw1B, INPUT_PULLUP);

  // Module 2 pins
  pinMode(in2a, OUTPUT); pinMode(in2b, OUTPUT); pinMode(pwm2, OUTPUT);
  pinMode(enc2A, INPUT); pinMode(enc2B, INPUT);
  attachInterrupt(digitalPinToInterrupt(enc2A), updateEnc2, RISING);
  pinMode(sw2A, INPUT_PULLUP); pinMode(sw2B, INPUT_PULLUP);

  EEPROM.begin(EEPROM_SIZE);
  EEPROM.get(ADDR_COUNT1, OPEN_COUNT1);
  EEPROM.get(ADDR_COUNT2, OPEN_COUNT2);
  if (OPEN_COUNT1 <= 0 || OPEN_COUNT1 > 10000) OPEN_COUNT1 = 1525;
  if (OPEN_COUNT2 <= 0 || OPEN_COUNT2 > 10000) OPEN_COUNT2 = 1500;
  Serial.printf("[EEPROM] M1=%ld M2=%ld\n", OPEN_COUNT1, OPEN_COUNT2);

  WiFi.begin(ssid,password);
  while (WiFi.status() != WL_CONNECTED) { delay(500); Serial.print("."); }
  Serial.println("\nWiFi connected. IP: "); Serial.println(WiFi.localIP());

  // Homing both modules
  goHome(pwm1, in1a, in1b, sw1A, sw1B, encCount1);
  goHome(pwm2, in2a, in2b, sw2A, sw2B, encCount2);

  // HTTP routes
  server.on("/open1", handleOpen1);
  server.on("/close1", handleClose1);
  server.on("/status1", handleStatus1);
  server.on("/setcount1", handleSetCount1);

  server.on("/open2", handleOpen2);
  server.on("/close2", handleClose2);
  server.on("/status2", handleStatus2);
  server.on("/setcount2", handleSetCount2);

  server.begin();
}

// ================== Loop ==================
void loop() {
  server.handleClient();
  if (encCount1 == 0 && !bothPressed(sw1A, sw1B)) {
    Serial.println("[Safety M1] Reclosing...");
    closeMotor(pwm1, in1a, in1b, sw1A, sw1B, encCount1);
  }
  if (encCount2 == 0 && !bothPressed(sw2A, sw2B)) {
    Serial.println("[Safety M2] Reclosing...");
    closeMotor(pwm2, in2a, in2b, sw2A, sw2B, encCount2);
  }
}
