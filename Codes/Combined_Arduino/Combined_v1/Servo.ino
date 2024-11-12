#include <Servo.h>

Servo servo;  // servo controller (multiple can exist)

void servo_init() {

  servo.attach(servo_pin);  // start servo control
  delay(10);
  drive_servo(&init_servo_pos);
  Serial.println("Servo Initialized");
}

void drive_servo(float* angle) {

  servo.write(*angle);
}