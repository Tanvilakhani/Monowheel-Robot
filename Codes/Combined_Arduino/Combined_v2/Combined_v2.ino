#include <PID.h>

PIDController servo_angle(0.7, 0.0003, 0.001, 0, 20);
const int servo_pin = 3;
float init_servo_pos = 95;

PIDController drive_wheel(0.005, 0.0001, 0, -155, 155);
const int A_IA = 7;
const int A_IB = 6;
const int A_EN = 5;

double x, y;
float init_x_angle;
float init_y_angle;

void setup() {

  Serial.begin(115200);

  imu_init();
  servo_init();

  short i = 0;
  while (i <= 200) {

    imu_getangle(&x, &y);
    i++;
    delay(10);
  }
  init_x_angle = x;
  servo_angle.setSetpoint(init_x_angle);
  Serial.print("Servo Setpoint = ");
  Serial.println(init_x_angle);

  init_y_angle = y;
  drive_wheel.setSetpoint(init_y_angle);
  Serial.print("Drive Motor Setpoint = ");
  Serial.println(init_y_angle);

  time_int_init();
}

void loop() {

  imu_getangle(&x, &y);

  Serial.print(x);
  Serial.print(",");
  Serial.println(y);
  if (abs(y - init_y_angle) < 1) drive_motor_stop();
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

void drive_motor_control_loop() {

  float drive_out;
  float drive_input = y;

  drive_out = drive_wheel.compute(drive_input);

  if (drive_out == 0 || abs(y - init_y_angle) < 1) {

    drive_motor_stop();
  } else if (drive_out > 0) {

    drive_motor_drive(0, &drive_out);
  } else {

    drive_motor_drive(1, &drive_out);
  }
  // Serial.println(drive_out);
}