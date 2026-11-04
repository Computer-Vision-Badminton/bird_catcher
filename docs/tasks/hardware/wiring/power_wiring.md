### Task
Connect the battery, fuse, kill switch and 5 V regulator to power the motor drivers, Pi, ESP32 and encoders.


### Suggested Layout:
```
3S LiPo ─ XT60 ─ 30 A fuse ─ kill switch ─ power bus (lever nuts)
                                              ├─ 4× BTS7960 motor power, 14 AWG
                                              ├─ D36V50F5 5 V ─┬─ Pi 5 via pins 2/4 + GND
                                              │                ├─ ESP32 5V pin
                                              │                ├─ BTS7960 VCC ×4
                                              │                └─ encoder VCC ×4
                                              └─ 100k/22k divider → ESP32 GPIO1
```

- IMU 3.3 V from the ESP32's 3V3 pin
- All grounds joined at the power bus, including the Pi–ESP32 UART ground
- Plug nothing until test with a multimeter, use a multimeter to confirm the regulator puts out 5.0 V, and that flipping the kill switch drops everything to 0.
