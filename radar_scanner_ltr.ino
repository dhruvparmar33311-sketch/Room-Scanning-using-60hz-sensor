/*
 * ESP32 3D Room Scanner  (28BYJ-48 pan + servo tilt + Acconeer XM125 radar)
 * LEFT-TO-RIGHT-ONLY version
 *
 * Scan pattern (9 rows: servo 130, 135, ... 170):
 *   1. Stepper goes to the leftmost point.
 *   2. Servo moves to the row angle (only ever done at the leftmost point).
 *   3. Stepper sweeps LEFT -> RIGHT, 3 steps at a time, one radar reading per stop.
 *   4. At the rightmost point the stepper returns to the left WITHOUT taking readings.
 *   5. Repeat with servo +5 deg, until 170 deg is done.
 *   6. At the end: servo back to SERVO_HOME, stepper back to centre.
 *
 * SERIAL OUTPUT (115200 baud, USB / COM10):
 *   Data lines  :  <stepper_step>,<servo_angle>,<distance_mm>      e.g.  -512,130,2471.1
 *   Status lines:  always start with '#'  (#SCAN_START, #ROW,130, #SCAN_END ...)
 *                  -> the Python viewer ignores them
 *
 * START A SCAN: ground GPIO14  OR  send the character 'S' from Python.
 */

#include <Arduino.h>
#include <ESP32Servo.h>
#include <Stepper.h>

// ---------------- PINS ----------------
constexpr uint8_t IN1 = 19, IN2 = 18, IN3 = 5, IN4 = 17;
constexpr uint8_t SERVO_PIN  = 16;
constexpr uint8_t START_PIN  = 14;     // INPUT_PULLUP, LOW = start
constexpr uint8_t SENSOR_RX  = 25;     // ESP32 RX  <- radar TX
constexpr uint8_t SENSOR_TX  = 26;     // ESP32 TX  -> radar RX
constexpr uint32_t SENSOR_BAUD = 921600;

// ---------------- SCAN SETTINGS ----------------
constexpr int  STEPS_PER_REV   = 2048;
constexpr int  STEPS_90        = 512;
constexpr int  STEPPER_RPM     = 12;   // scan speed. 28BYJ-48 is reliable up to ~15 rpm
constexpr int  RETURN_RPM      = 15;   // speed of the no-reading return to the left
constexpr int  SAMPLE_EVERY    = 3;    // stepper moves 3 steps (0.53 deg) between readings

// Sweep goes from -90 deg to the last point that is an exact multiple of SAMPLE_EVERY,
// so every reading is spaced exactly 3 steps apart (-512, -509, ... 508, 511).
constexpr int  SWEEP_START = -STEPS_90;
constexpr int  SWEEP_END   = SWEEP_START + SAMPLE_EVERY * ((2 * STEPS_90) / SAMPLE_EVERY);

constexpr int  SERVO_START = 130;
constexpr int  SERVO_END   = 170;
constexpr int  SERVO_STEP  = 5;
constexpr int  SERVO_HOME  = 130;      // where the servo rests after a scan

constexpr int  BACKLASH_TAKEUP = 20;   // extra steps left-then-right before each row so the gear
                                       // slack is always taken up in the scan direction. 0 = off

constexpr bool     DETACH_SERVO_DURING_SWEEP = true;  // false = keep holding position (no sag, tiny hum)
constexpr uint32_t SERVO_SETTLE_MS   = 400;  // time for servo to arrive before detaching
constexpr uint32_t STEPPER_SETTLE_MS = 15;   // let mechanical vibration die before measuring
constexpr uint32_t RADAR_TIMEOUT_MS  = 250;  // max wait for one fresh radar line
constexpr int      RADAR_RETRIES     = 2;    // extra attempts per point
constexpr float    MIN_VALID_MM      = 10.0f;
constexpr float    MAX_VALID_MM      = 20000.0f;

// ---------------- GLOBALS ----------------
Servo tiltServo;
Stepper myStepper(STEPS_PER_REV, IN1, IN3, IN2, IN4);  // correct order for 28BYJ-48 + ULN2003
int  stepperPos = 0;           // dead-reckoned position, 0 = centre
uint32_t okCount = 0, missCount = 0;

// ---------------- STEPPER ----------------
void releaseStepper() {        // de-energize coils: no heat, no electrical noise on the servo
  digitalWrite(IN1, LOW); digitalWrite(IN2, LOW);
  digitalWrite(IN3, LOW); digitalWrite(IN4, LOW);
}

inline void stepOnce(int dir) {
  myStepper.step(dir);
  stepperPos += dir;
}

// Move to a position in chunks (no readings). Used for the return sweep and homing.
void moveStepperTo(int target, int rpm) {
  myStepper.setSpeed(rpm);
  while (stepperPos != target) {
    int diff  = target - stepperPos;
    int chunk = (abs(diff) > 64) ? (diff > 0 ? 64 : -64) : diff;
    myStepper.step(chunk);
    stepperPos += chunk;
    yield();
  }
  myStepper.setSpeed(STEPPER_RPM);
}

// ---------------- SERVO ----------------
void servoGoTo(int angle) {
  releaseStepper();                       // quiet supply while the servo moves
  tiltServo.setPeriodHertz(50);
  tiltServo.attach(SERVO_PIN, 500, 2400);
  tiltServo.write(angle);
  delay(SERVO_SETTLE_MS);
  if (DETACH_SERVO_DURING_SWEEP) {
    tiltServo.detach();
    pinMode(SERVO_PIN, OUTPUT);
    digitalWrite(SERVO_PIN, LOW);         // keep the signal line quiet (no PWM = no jitter)
  }
}

// ---------------- RADAR ----------------
// Parses e.g. "1 detected distances: 2.471099 m" -> 2471.099 mm (first/nearest peak)
bool parseRadarLine(const char *line, float &mm) {
  const char *colon = strchr(line, ':');
  if (!colon) return false;
  char *end;
  float meters = strtof(colon + 1, &end);
  if (end == colon + 1) return false;     // no number (e.g. "0 detected distances:")
  float v = meters * 1000.0f;
  if (v < MIN_VALID_MM || v > MAX_VALID_MM) return false;
  mm = v;
  return true;
}

// Returns ONE fresh measurement taken AFTER this call (stale buffered data is thrown away).
bool readRadarOnce(float &mm) {
  while (Serial2.available()) Serial2.read();      // drop old lines
  static char buf[96];
  size_t len = 0;
  bool synced = false, overflow = false;           // first partial line is discarded
  uint32_t t0 = millis();

  while (millis() - t0 < RADAR_TIMEOUT_MS) {
    while (Serial2.available()) {
      char c = Serial2.read();
      if (!synced) { if (c == '\n') synced = true; continue; }
      if (c == '\r') continue;
      if (c == '\n') {
        buf[len] = 0;
        if (overflow) { len = 0; overflow = false; continue; }
        return parseRadarLine(buf, mm);            // complete line: valid or "no detection"
      }
      if (len < sizeof(buf) - 1) buf[len++] = c; else overflow = true;
    }
    delayMicroseconds(100);
  }
  return false;                                    // timeout
}

bool readRadar(float &mm) {
  for (int i = 0; i <= RADAR_RETRIES; i++) if (readRadarOnce(mm)) return true;
  return false;
}

// ---------------- OUTPUT ----------------
void captureAndSend(int servoAngle) {
  float mm;
  if (readRadar(mm)) {
    char out[40];
    snprintf(out, sizeof(out), "%d,%d,%.1f\n", stepperPos, servoAngle, mm);
    Serial.print(out);                             // one atomic line
    okCount++;
  } else {
    missCount++;                                   // nothing printed -> Python never sees bad data
  }
}

// ---------------- SCAN ----------------
// One row: LEFT -> RIGHT only, SAMPLE_EVERY steps between readings.
void sweepRowLeftToRight(int servoAngle) {
  delay(STEPPER_SETTLE_MS);
  captureAndSend(servoAngle);                      // reading at the left edge
  while (stepperPos < SWEEP_END) {
    for (int i = 0; i < SAMPLE_EVERY && stepperPos < SWEEP_END; i++) stepOnce(1);
    delay(STEPPER_SETTLE_MS);
    captureAndSend(servoAngle);
  }
}

// Take up gear slack in the scan direction so every row starts the same way.
void backlashTakeup() {
  if (BACKLASH_TAKEUP <= 0) return;
  moveStepperTo(SWEEP_START - BACKLASH_TAKEUP, STEPPER_RPM);
  moveStepperTo(SWEEP_START, STEPPER_RPM);
}

void returnHome() {
  Serial.println("#HOMING");
  servoGoTo(SERVO_HOME);                           // still at the leftmost point
  moveStepperTo(0, STEPPER_RPM);
  releaseStepper();
}

void runScan() {
  okCount = missCount = 0;
  Serial.println("#SCAN_START");
  Serial.println("#FORMAT,step,servo_deg,dist_mm");

  moveStepperTo(SWEEP_START, STEPPER_RPM);         // go to the leftmost point first

  for (int angle = SERVO_START; angle <= SERVO_END; angle += SERVO_STEP) {
    servoGoTo(angle);                              // servo ONLY changes at the leftmost point
    Serial.printf("#ROW,%d\n", angle);
    backlashTakeup();
    sweepRowLeftToRight(angle);                    // readings: left -> right
    moveStepperTo(SWEEP_START, RETURN_RPM);        // right -> left: NO readings
  }

  returnHome();
  Serial.printf("#SCAN_END,points=%lu,missed=%lu\n", (unsigned long)okCount, (unsigned long)missCount);
}

// ---------------- SETUP / LOOP ----------------
bool startPressed() {                              // debounced LOW
  if (digitalRead(START_PIN) != LOW) return false;
  delay(30);
  return digitalRead(START_PIN) == LOW;
}

void waitForRelease() {
  uint32_t highSince = 0;
  while (true) {
    if (digitalRead(START_PIN) == HIGH) {
      if (!highSince) highSince = millis();
      if (millis() - highSince > 50) break;
    } else highSince = 0;
    delay(5);
  }
}

void setup() {
  Serial.begin(115200);
  Serial2.setRxBufferSize(2048);                   // must be BEFORE begin()
  Serial2.begin(SENSOR_BAUD, SERIAL_8N1, SENSOR_RX, SENSOR_TX);

  pinMode(START_PIN, INPUT_PULLUP);
  myStepper.setSpeed(STEPPER_RPM);
  releaseStepper();

  ESP32PWM::allocateTimer(0);
  servoGoTo(SERVO_HOME);

  Serial.println("#READY  (ground GPIO14 or send 'S' to scan)");
}

void loop() {
  bool serialStart = false;
  while (Serial.available()) if (Serial.read() == 'S') serialStart = true;

  if (startPressed() || serialStart) {
    runScan();
    waitForRelease();                              // no re-trigger while the button is still held
    while (Serial.available()) Serial.read();
    Serial.println("#READY");
  }
  delay(20);
}
