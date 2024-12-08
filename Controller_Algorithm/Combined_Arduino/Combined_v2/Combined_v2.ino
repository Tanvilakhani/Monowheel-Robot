#include <PID.h>

PIDController servo_angle(2, 0.0, 0, 0, 40);
const int servo_pin = 32;
int init_servo_pos = 95;

PIDController drive_wheel(2, 0.0, 0, -155, 155);
const int A_IA = 33;
const int A_IB = 27;
const int A_EN = 13;

double x, y;
float init_x_angle;
float init_y_angle;

void imuTask(void* parameter) {
  for (;;) {
    double new_x, new_y;

    // Read IMU angles
    imu_getangle(&new_x, &new_y);

    // Safely update global variables
    noInterrupts();
    x = new_x;
    y = new_y;
    interrupts();

    // Delay to control task frequency
    // Adjust based on your requirements
    vTaskDelay(pdMS_TO_TICKS(5));
  }
}

void setup() {

  Serial.begin(115200);

  imu_init();
  servo_init();
  drive_motor_init();

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

  xTaskCreate(
    imuTask,     // Task function
    "IMU_Task",  // Task name
    4096,        // Stack size (adjust as needed)
    NULL,        // Task parameters
    2,           // Priority
    NULL         // Task handle
  );
}

void loop() {

  //imu_getangle(&x, &y);

  // Serial.print(x);
  // Serial.print(",");
  // Serial.println(y);
  if (abs(y - init_y_angle) < 1) drive_motor_stop();
  //delay(5);
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

  int angle = int(servo_out);
  drive_servo(&angle);
  //Serial.println(angle);
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
  Serial.println(drive_out);
}