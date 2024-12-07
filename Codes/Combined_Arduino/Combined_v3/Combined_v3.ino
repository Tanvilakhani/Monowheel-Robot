#include <PID.h>

PIDController servo_angle(0.5, 0.02, 0.002, 0, 35);
const int servo_pin = 32;
int init_servo_pos = 75;

PIDController drive_wheel(3, 0.01, 0, -155, 155);
const int A_IA = 33;
const int A_IB = 27;
const int A_EN = 13;

double x, y;
float init_x_angle;
float init_y_angle;
unsigned long ptim;

bool drive;
unsigned long pdrtim;
unsigned long pdrtim2;
float spd = 1;

const int led_pin = 2;

void setup() {

  Serial.begin(115200);

  led_init();
  imu_init();
  servo_init();
  drive_motor_init();
  in_init();

  short i = 0;
  while (i <= 40) {

    imu_getangle(&x, &y);
    i++;
    led_on();
    delay(50);
    led_off();
    delay(50);
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

  pdrtim = millis();
  delay(1000);
  led_on();
}

void loop() {

  if (millis() - ptim >= 10) {

    imu_getangle(&x, &y);
    ptim = millis();
  }

  // Serial.print(x);
  // Serial.print(",");
  // Serial.println(y);

  bool a, b = in_stat();

  if (abs(y - init_y_angle) < 5) {

    if (!drive) drive_motor_stop();

    if (!drive && ((millis() - pdrtim) >= 1000) && !(!a && !b)) drive = true;

    if (drive) {
      
      if (millis() - pdrtim2 >= 10) {

        spd++;

        if ((a && !b) || (!a && b)) spd = constrain(spd, 0, 50);
        else if (a && b) spd = constrain(spd, 0, 100);

        pdrtim2 = millis();
      }

      if (!a && !b) drive = false;
      else if ((a && !b) || (a && b)) drive_motor_drive(1, &spd);
      else if (a && b) drive_motor_drive(0, &spd);
      led_on();
    }
  } else {

    drive = false;
    pdrtim = millis();
    led_off();
    spd = 1;
  }
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

    servo_out = init_servo_pos - servo_out;
  } else {

    servo_out = init_servo_pos + servo_out;
  }

  int angle = int(servo_out);
  drive_servo(&angle);
  //Serial.println(angle);
}

void drive_motor_control_loop() {

  float drive_out;
  float drive_input = y;

  drive_out = drive_wheel.compute(drive_input);

  if ((drive_out == 0 || abs(y - init_y_angle) < 5) && !drive) {

    drive_motor_stop();
  } else if ((drive_out > 0) && !drive) {

    drive_motor_drive(0, &drive_out);
  } else if (!drive) {

    drive_motor_drive(1, &drive_out);
  }
  //Serial.println(drive_out);
}