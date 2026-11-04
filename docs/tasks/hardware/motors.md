### Tasks

- **Single motor test:** Drive one motor through its BTS7960 using PWM on its two pins, with the enable line on GPIO2.
  - **Done when:** it spins forward and backward at different speeds, and setting GPIO2 low stops it instantly.
  - **Why one first:** a wiring mistake only risks one motor and driver, and it proves the PWM and enable wiring before copying it 3 more times.

- **All motors test:** Drive all 4 motors with the robot on blocks so the wheels are off the ground.
  - **Done when:** each wheel spins the commanded direction, and its encoder counts up for forward.
  - **Why on blocks:** a backwards motor is a common, easy fix, but on the floor it makes the robot lurch. Fix direction in code or by swapping the motor's two wires.