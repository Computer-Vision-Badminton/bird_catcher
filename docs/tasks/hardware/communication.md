### Tasks

- **Pi–ESP32 link:** Connect the Pi and ESP32 over UART1 on GPIO17/18 at 921600 baud, with a shared ground.
  - **Done when:** messages pass both ways using the `[0xAA 0x55][type][len][seq][payload][CRC16]` format, with no CRC errors or skipped sequence numbers over a 10-minute run at full rate.
  - **Why CRC and seq:** the CRC catches scrambled messages caused by motor noise, and the sequence number shows when a message went missing.
