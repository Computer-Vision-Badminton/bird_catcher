### Task
Connect the ESP32 to the motor PWM, encoders, IMU, the Pi’s UART and the battery sense.

### Suggested Layout
| GPIO | Job |
|---|---|
| 1 | Battery voltage, through the 100k/22k divider |
| 2 | Enable for all 4 drivers: every board's R_EN and L_EN tie to this pin |
| 10 / 11 | Front-left motor PWM, forward / reverse |
| 12 / 13 | Front-right motor PWM |
| 14 / 21 | Rear-left motor PWM |
| 41 / 42 | Rear-right motor PWM |
| 4 / 5 | Front-left encoder A / B |
| 6 / 7 | Front-right encoder A / B |
| 8 / 9 | Rear-left encoder A / B |
| 15 / 16 | Rear-right encoder A / B |
| 17 / 18 | UART1 to the Pi, TX / RX |
| 39 / 40 / 47 | IMU SPI: SCK / SI / SO |
| 19 / 20 | IMU CS / INT |
| 48 or 38 | v2 tilt servo: use whichever pin the board's RGB LED is not on |

- Avoid GPIO 0, 3, 45 and 46 (boot strapping) and 26–37 (flash and PSRAM)
- All 8 encoder lines go through 10k/20k dividers to bring their 5 V signals down to about 3.3 V
- IMU reset is not wired; reset it in software
- GPIO 19/20 are the native USB pins, so flash and watch logs through the UART USB-C port
- GPIO 39–42 are the JTAG pins, so JTAG debugging is off
- Confirm the clone board's pinout before wiring