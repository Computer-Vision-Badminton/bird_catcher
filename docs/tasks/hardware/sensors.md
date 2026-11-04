### Tasks

- **Battery voltage:** Read the battery voltage through the 100k/22k divider on GPIO1.
  - **Done when:** the reading matches a multimeter within 0.1 V.
  - **Why 0.1 V:** the usable 3S range is only about 2 V (12.6 V full, 10.5 V cutoff), so an accurate reading keeps the low-battery stop at the right point. Calibrate against the multimeter in code.

- **Encoders:** Connect the 4 encoders through 10k/20k dividers and count with PCNT.
  - **Done when:** one full wheel turn by hand gives about 1,200 counts, going up forward and down backward.
  - **Why 1,200:** 64 counts per motor turn × the 18.75:1 gearbox. About 300 or 600 means you're counting only one wire or part of each pulse; counting the wrong way means A and B are swapped.

- **IMU:** Connect the BNO085 over SPI using game rotation vector mode.
  - **Done when:** the heading streams at 200 Hz or more with no gaps, and returns to the same value within a few degrees after rotating the robot 90° and back.
  - **Why game rotation vector:** it skips the magnetometer, which motors and gym steel throw off.