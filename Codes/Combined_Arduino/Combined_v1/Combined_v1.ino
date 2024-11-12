#include </Users/yasas/Desktop/Combined_v1/Combined_v1/PID.h>

PIDController servo_angle(2.0, 0.05, 0.01, 0, 45);
const int servo_pin = 3;
float init_servo_pos = 90;

double x, y, z;
float init_x_angle;

void setup() {

  Serial.begin(9600);

  imu_init();
  servo_init();

  delay(2000);
  imu_getangle(&x, &y, &z);
  init_x_angle = x;
  servo_angle.setSetpoint(init_x_angle);
  Serial.print("Servo Setpoint = ");
  Serial.println(init_x_angle);

  time_int_init();
}

void loop() {

  imu_getangle(&x, &y, &z);
}

void servo_control_loop() {

  float servo_out;
  float servo_input;

  if (x <= init_x_angle) {

    servo_input = x;
  } else {

    servo_input = init_x_angle - (x - init_x_angle);
  }

  servo_out = servo_angle.compute(servo_input);

  if (x <= init_x_angle) {

    servo_out = init_servo_pos + servo_out;
  } else {

    servo_out = init_servo_pos - servo_out;
  }

  drive_servo(&servo_out);
  //Serial.println(servo_out);
}