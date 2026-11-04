### Tasks

- **Safety stops:** Make the motors stop on their own when something goes wrong.
  - **Done when:**
    - Unplugging the Pi link stops the motors within 100 ms.
    - The battery reading reaching 10.5 V stops the motors.
    - The kill switch cuts all power.
    - The motors stay off while the ESP32 boots or resets.
  - **Why:** the robot moves near players, so it must stop by itself if the Pi crashes, the link breaks or the battery runs low. Add a pull-down resistor on GPIO2 so the drivers stay disabled until the firmware turns them on.