#include <Arduino.h>

#include "landing_frame_parser.h"

const int PI_LINK_RX_PIN = 18;
const int PI_LINK_TX_PIN = 17;
const uint32_t PI_LINK_BAUD = 921600;
const size_t PI_LINK_RX_BUFFER_BYTES = 1024;
const uint32_t COUNTER_REPORT_INTERVAL_MS = 1000;

HardwareSerial piLink(1);
LandingFrameParser landingParser;
uint32_t lastCounterReportMs = 0;
uint32_t framesAtLastReport = 0;

void onLanding(int16_t xMm, int16_t yMm, bool final) {
  Serial.printf("landing x %d mm, y %d mm, %s\n", xMm, yMm, final ? "final" : "in flight");
}

void reportCounters() {
  const PiLinkCounters &counters = landingParser.counters;
  Serial.printf("pi link: %lu frames, %lu lost, %lu crc errors, %lu bad lengths, %lu other types\n",
                static_cast<unsigned long>(counters.framesReceived), static_cast<unsigned long>(counters.framesLost),
                static_cast<unsigned long>(counters.crcErrors), static_cast<unsigned long>(counters.badLengths),
                static_cast<unsigned long>(counters.otherTypes));
}

void setup() {
  Serial.begin(115200);
  piLink.setRxBufferSize(PI_LINK_RX_BUFFER_BYTES);
  piLink.begin(PI_LINK_BAUD, SERIAL_8N1, PI_LINK_RX_PIN, PI_LINK_TX_PIN);
  Serial.println("waiting for landings from the Pi");
}

void loop() {
  LandingTarget landing;
  while (piLink.available() > 0) {
    if (landingParser.push(static_cast<uint8_t>(piLink.read()), landing)) {
      onLanding(landing.xMm, landing.yMm, landing.final);
    }
  }
  bool newFramesSinceReport = landingParser.counters.framesReceived != framesAtLastReport;
  if (newFramesSinceReport && millis() - lastCounterReportMs >= COUNTER_REPORT_INTERVAL_MS) {
    reportCounters();
    framesAtLastReport = landingParser.counters.framesReceived;
    lastCounterReportMs = millis();
  }
}
