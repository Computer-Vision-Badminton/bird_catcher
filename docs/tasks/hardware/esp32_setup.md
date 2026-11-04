### Task
- FreeRTOS skeleton: Set up the four ESP32 tasks (control, IMU, comms and safety) running 
at their target rates.

### ESP32 FreeRTOS task rates

| Task | Target rate | Core | Why |
|---|---|---|---|
| Control | 1,000 Hz | 1 | Motor speed control needs fast, steady updates to stay smooth |
| IMU | 200–400 Hz | — | Fast enough to track quick turns and stops |
| Comms | 100–200 Hz | 0 | Keeps up with new targets from the Pi at 60–150 fps |
| Safety | 50 Hz | — | Checks battery voltage and the Pi link watchdog; catches a lost link well within 100 ms |

- Control runs alone on core 1 and comms runs on core 0, so talking to the Pi never slows motor control