import struct

SYNC = b"\xaa\x55"
LANDING_TARGET = 0x01
FINAL_ESTIMATE = 0x01
DEFAULT_PORT = "/dev/serial0"
DEFAULT_BAUD = 921600
HEADER_BYTES = 5
CRC_BYTES = 2
LANDING_PAYLOAD = struct.Struct("<hhB")


def crc16_ccitt(data):
    crc = 0xFFFF
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) if crc & 0x8000 else (crc << 1)
            crc &= 0xFFFF
    return crc


def frame(message_type, sequence, payload):
    body = bytes([message_type, len(payload), sequence & 0xFF]) + payload
    return SYNC + body + struct.pack("<H", crc16_ccitt(body))


def landing_payload(x_cm, y_cm, final):
    return LANDING_PAYLOAD.pack(round(x_cm * 10), round(y_cm * 10), FINAL_ESTIMATE if final else 0)


def decode_frames(data):
    messages, index = [], 0
    while index + HEADER_BYTES + CRC_BYTES <= len(data):
        if data[index:index + 2] != SYNC:
            index += 1
            continue
        length = data[index + 3]
        end = index + HEADER_BYTES + length
        if end + CRC_BYTES > len(data):
            break
        body = data[index + 2:end]
        if struct.unpack("<H", data[end:end + CRC_BYTES])[0] == crc16_ccitt(body):
            messages.append((body[0], body[2], body[3:]))
            index = end + CRC_BYTES
        else:
            index += 1
    return messages


def decode_landing(payload):
    x_mm, y_mm, flags = LANDING_PAYLOAD.unpack(payload)
    return x_mm / 10, y_mm / 10, bool(flags & FINAL_ESTIMATE)


class Esp32Link:
    def __init__(self, port=DEFAULT_PORT, baud=DEFAULT_BAUD, connection=None):
        if connection is None:
            import serial
            connection = serial.Serial(port, baud, timeout=0)
        self.connection = connection
        self.sequence = 0

    def send_landing(self, x_cm, y_cm, final):
        message = frame(LANDING_TARGET, self.sequence, landing_payload(x_cm, y_cm, final))
        self.connection.write(message)
        self.sequence = (self.sequence + 1) & 0xFF
        return message

    def close(self):
        self.connection.close()
