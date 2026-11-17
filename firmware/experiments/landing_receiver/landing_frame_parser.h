#pragma once

#include <stddef.h>
#include <stdint.h>

const uint8_t SYNC_FIRST_BYTE = 0xAA;
const uint8_t SYNC_SECOND_BYTE = 0x55;
const uint8_t LANDING_TARGET_TYPE = 0x01;
const uint8_t LANDING_PAYLOAD_LENGTH = 5;
const uint8_t FINAL_LANDING_FLAG = 0x01;
const uint8_t LARGEST_PAYLOAD_LENGTH = 32;
const size_t BODY_HEADER_LENGTH = 3;

inline uint16_t crc16Ccitt(const uint8_t *data, size_t length) {
  uint16_t crc = 0xFFFF;
  for (size_t i = 0; i < length; i++) {
    crc ^= static_cast<uint16_t>(data[i]) << 8;
    for (int bit = 0; bit < 8; bit++) {
      crc = (crc & 0x8000) ? static_cast<uint16_t>((crc << 1) ^ 0x1021) : static_cast<uint16_t>(crc << 1);
    }
  }
  return crc;
}

inline int16_t int16FromLittleEndian(const uint8_t *bytes) {
  return static_cast<int16_t>(static_cast<uint16_t>(bytes[0]) | (static_cast<uint16_t>(bytes[1]) << 8));
}

struct LandingTarget {
  int16_t xMm;
  int16_t yMm;
  bool final;
  uint8_t seq;
};

struct PiLinkCounters {
  uint32_t framesReceived = 0;
  uint32_t framesLost = 0;
  uint32_t crcErrors = 0;
  uint32_t badLengths = 0;
  uint32_t otherTypes = 0;
};

class LandingFrameParser {
 public:
  PiLinkCounters counters;

  bool push(uint8_t byte, LandingTarget &landing) {
    switch (step) {
      case Step::WaitingForSyncFirst:
        if (byte == SYNC_FIRST_BYTE) {
          step = Step::WaitingForSyncSecond;
        }
        return false;
      case Step::WaitingForSyncSecond:
        if (byte == SYNC_SECOND_BYTE) {
          bodyLength = 0;
          step = Step::ReadingHeader;
        } else if (byte != SYNC_FIRST_BYTE) {
          step = Step::WaitingForSyncFirst;
        }
        return false;
      case Step::ReadingHeader:
        body[bodyLength++] = byte;
        if (bodyLength == 2 && byte > LARGEST_PAYLOAD_LENGTH) {
          counters.badLengths++;
          step = Step::WaitingForSyncFirst;
        } else if (bodyLength == BODY_HEADER_LENGTH) {
          step = payloadLength() == 0 ? Step::ReadingCrcLow : Step::ReadingPayload;
        }
        return false;
      case Step::ReadingPayload:
        body[bodyLength++] = byte;
        if (bodyLength == BODY_HEADER_LENGTH + payloadLength()) {
          step = Step::ReadingCrcLow;
        }
        return false;
      case Step::ReadingCrcLow:
        receivedCrcLow = byte;
        step = Step::ReadingCrcHigh;
        return false;
      case Step::ReadingCrcHigh:
        step = Step::WaitingForSyncFirst;
        return finishFrame(static_cast<uint16_t>(receivedCrcLow | (static_cast<uint16_t>(byte) << 8)), landing);
    }
    return false;
  }

 private:
  enum class Step { WaitingForSyncFirst, WaitingForSyncSecond, ReadingHeader, ReadingPayload, ReadingCrcLow, ReadingCrcHigh };

  Step step = Step::WaitingForSyncFirst;
  uint8_t body[BODY_HEADER_LENGTH + LARGEST_PAYLOAD_LENGTH];
  size_t bodyLength = 0;
  uint8_t receivedCrcLow = 0;
  bool anyFrameSeen = false;
  uint8_t previousSeq = 0;

  uint8_t messageType() const { return body[0]; }
  uint8_t payloadLength() const { return body[1]; }
  uint8_t seq() const { return body[2]; }
  const uint8_t *payload() const { return body + BODY_HEADER_LENGTH; }

  void countLostFrames() {
    if (anyFrameSeen) {
      counters.framesLost += static_cast<uint8_t>(seq() - previousSeq - 1);
    }
    anyFrameSeen = true;
    previousSeq = seq();
  }

  bool finishFrame(uint16_t receivedCrc, LandingTarget &landing) {
    if (receivedCrc != crc16Ccitt(body, bodyLength)) {
      counters.crcErrors++;
      return false;
    }
    counters.framesReceived++;
    countLostFrames();
    if (messageType() != LANDING_TARGET_TYPE || payloadLength() != LANDING_PAYLOAD_LENGTH) {
      counters.otherTypes++;
      return false;
    }
    landing.xMm = int16FromLittleEndian(payload());
    landing.yMm = int16FromLittleEndian(payload() + 2);
    landing.final = (payload()[4] & FINAL_LANDING_FLAG) != 0;
    landing.seq = seq();
    return true;
  }
};
