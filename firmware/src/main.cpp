// IMU test: checks that the BNO085 answers over SPI, then prints its heading,
// how many readings arrive per second, and the longest gap between readings.
// It does not move any motors.
//
// Passes when (docs/tasks/hardware/sensors.md):
//   - readings stay at 200 per second or more, with no gaps
//   - heading returns to within a few degrees after turning 90 degrees and back

#include <Arduino.h>
#include <SPI.h>
#include <Wire.h>
#include <Adafruit_BNO08x.h>

// IMU pins, from docs/tasks/hardware/wiring/esp32_pin_map.md.
const int PIN_SCK = 9
;
const int PIN_SI = 11;  // Data into the IMU.
const int PIN_SO = 10;  // Data out of the IMU.
const int PIN_CS = 12;
const int PIN_INT = 13;
const int PIN_RST = 15;  // Lets the ESP32 restart the IMU each time it starts.

const uint32_t REPORT_INTERVAL_US = 5000;  // One reading every 5 ms = 200 per second.

Adafruit_BNO08x imu(PIN_RST);
sh2_SensorValue_t reading;

float headingDegrees = 0;
uint32_t readingCount = 0;
uint32_t longestGapMs = 0;
uint32_t lastReadingMs = 0;
uint32_t lastPrintMs = 0;

// Asks the IMU to send game rotation vector readings, which skip the magnetometer.
void startReadings() {
  if (!imu.enableReport(SH2_GAME_ROTATION_VECTOR, REPORT_INTERVAL_US)) {
    Serial.println("FAILED: the IMU refused the game rotation vector request.");
  }
}

// Converts the IMU's orientation (a quaternion) into a compass-style angle.
float headingFromReading() {
  float w = reading.un.gameRotationVector.real;
  float x = reading.un.gameRotationVector.i;
  float y = reading.un.gameRotationVector.j;
  float z = reading.un.gameRotationVector.k;
  return atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)) * RAD_TO_DEG;
}

// PS0 and PS1 can't be read directly, so this looks for the IMU over I2C instead.
// An answer means PS0 and PS1 are low. Both data wires are tried, which also
// catches them being swapped.
void checkForI2cMode() {
  SPI.end();
  imu.hardwareReset();
  delay(200);

  const int dataPins[] = {PIN_SO, PIN_SI};
  const uint8_t addresses[] = {0x4A, 0x4B};
  bool found = false;

  for (int dataPin : dataPins) {
    Wire.begin(dataPin, PIN_SCK);
    for (uint8_t address : addresses) {
      Wire.beginTransmission(address);
      if (Wire.endTransmission() == 0) {
        Serial.printf("I2C check: IMU answered at 0x%02X with data on GPIO %d. It is in I2C mode, not SPI.\n",
                      address, dataPin);
        found = true;
      }
    }
    Wire.end();
  }

  if (!found) {
    Serial.println("I2C check: no answer, so it is probably not in I2C mode.");
  }
}

// After a failed start, restarts the IMU and prints what it does, to help find
// which wire is at fault. INT should go LOW once the IMU has a message waiting.
void printFailureDetails() {
  imu.hardwareReset();

  uint32_t start = millis();
  while (digitalRead(PIN_INT) == HIGH && millis() - start < 1000) {
    delay(1);
  }
  bool intIsLow = digitalRead(PIN_INT) == LOW;
  Serial.printf("INT after reset: %s\n", intIsLow ? "LOW" : "HIGH");

  // Read the first 4 bytes of the IMU's message directly.
  Serial.print("First bytes from IMU:");
  SPI.beginTransaction(SPISettings(1000000, MSBFIRST, SPI_MODE3));
  digitalWrite(PIN_CS, LOW);
  for (int i = 0; i < 4; i++) {
    Serial.printf(" %02X", SPI.transfer(0x00));
  }
  digitalWrite(PIN_CS, HIGH);
  SPI.endTransaction();
  Serial.println();

  checkForI2cMode();
}

void setup() {
  Serial.begin(115200);
  delay(1000);
  Serial.println("Starting BNO085 test...");

  SPI.begin(PIN_SCK, PIN_SO, PIN_SI, PIN_CS);

  if (!imu.begin_SPI(PIN_CS, PIN_INT)) {
    Serial.println("FAILED: no answer from the BNO085. Check its wiring and SPI mode pins.");
    printFailureDetails();
    while (true) {
      delay(1000);  // Stop here. Fix the wiring, then press RST.
    }
  }

  Serial.println("PASSED: the BNO085 answered over SPI.");
  Serial.printf("Firmware version: %d.%d.%d\n",
                imu.prodIds.entry[0].swVersionMajor,
                imu.prodIds.entry[0].swVersionMinor,
                imu.prodIds.entry[0].swVersionPatch);

  startReadings();
  lastReadingMs = millis();
  lastPrintMs = millis();
}

void loop() {
  // The IMU forgets its settings if it resets, so ask for readings again.
  if (imu.wasReset()) {
    Serial.println("WARNING: the IMU reset itself.");
    startReadings();
  }

  if (imu.getSensorEvent(&reading) && reading.sensorId == SH2_GAME_ROTATION_VECTOR) {
    uint32_t now = millis();
    longestGapMs = max(longestGapMs, now - lastReadingMs);
    lastReadingMs = now;
    readingCount++;
    headingDegrees = headingFromReading();
  }

  // Print a summary once per second, since printing every reading would be too slow.
  if (millis() - lastPrintMs >= 1000) {
    Serial.printf("Heading: %7.1f deg | %3lu readings/s | longest gap: %lu ms\n",
                  headingDegrees, readingCount, longestGapMs);
    readingCount = 0;
    longestGapMs = 0;
    lastPrintMs = millis();
  }
}
