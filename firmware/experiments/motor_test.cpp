// Single motor test: slowly spins one motor forward, then backward, through its
// BTS7960 driver, then stops. It runs once each time the ESP32 starts.
//
// Keep the wheel off the ground and the kill switch within reach.
//
// Run it with: pio run -e motor_test -t upload -t monitor

#include <Arduino.h>

// Pins, from docs/tasks/hardware/wiring/esp32_pin_map.md (rear-left motor).
const int PIN_ENABLE_R = 6;  // Connects to the driver's R_EN.
const int PIN_ENABLE_L = 3;  // Connects to the driver's L_EN.
const int PIN_FORWARD = 5;  // Connects to the driver's RPWM.
const int PIN_REVERSE = 8;  // Connects to the driver's LPWM.

const uint32_t PWM_FREQUENCY_HZ = 20000;  // Above hearing range, so the motor doesn't whine.
const uint8_t PWM_BITS = 8;               // Speed runs from 0 (stopped) to 255 (full).

const int TEST_SPEED = 75;  // About 30% of full speed.
const int RUN_TIME_MS = 3000;
const int RAMP_STEP_DELAY_MS = 20;
const int PAUSE_BETWEEN_DIRECTIONS_MS = 1000;  // Lets the motor stop before reversing.

// Speeds up on one PWM pin, holds, then slows back to a stop. Only one pin is
// ever above 0, because the other is left at 0 the whole time.
void spinMotor(int pin) {
  for (int speed = 0; speed <= TEST_SPEED; speed++) {
    ledcWrite(pin, speed);
    delay(RAMP_STEP_DELAY_MS);
  }

  delay(RUN_TIME_MS);

  for (int speed = TEST_SPEED; speed >= 0; speed--) {
    ledcWrite(pin, speed);
    delay(RAMP_STEP_DELAY_MS);
  }
}

void setup() {
  // Keep the driver off first, so the motor can't move during setup.
  pinMode(PIN_ENABLE_R, OUTPUT);
  pinMode(PIN_ENABLE_L, OUTPUT);
  digitalWrite(PIN_ENABLE_R, LOW);
  digitalWrite(PIN_ENABLE_L, LOW);

  Serial.begin(115200);
  delay(1000);

  ledcAttach(PIN_FORWARD, PWM_FREQUENCY_HZ, PWM_BITS);
  ledcAttach(PIN_REVERSE, PWM_FREQUENCY_HZ, PWM_BITS);
  ledcWrite(PIN_FORWARD, 0);
  ledcWrite(PIN_REVERSE, 0);

  Serial.println("Motor test: starting in 3 seconds...");
  delay(3000);

  // The motor only spins with both sides of the driver on.
  digitalWrite(PIN_ENABLE_R, HIGH);
  digitalWrite(PIN_ENABLE_L, HIGH);

  Serial.println("Motor spinning forward");
  spinMotor(PIN_FORWARD);

  delay(PAUSE_BETWEEN_DIRECTIONS_MS);

  Serial.println("Motor going backward");
  spinMotor(PIN_REVERSE);

  digitalWrite(PIN_ENABLE_R, LOW);
  digitalWrite(PIN_ENABLE_L, LOW);
  Serial.println("Done. Motor is off. Press RST to run it again.");
}

void loop() {
  // Nothing to do: the test only runs once.
}
