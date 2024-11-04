#include <Wire.h>
#include <MPU6050.h>

MPU6050 mpu; // MPU6050 object

// PID parameters
float kp = 1.0;
float ki = 0.1;
float kd = 0.5;

float setPoint;
float input;
float output;
float lastInput = 0;
float integral = 0;

// Motor pins
const int motorPin1 = 5;  // Input A for Motor 1 (forward)
const int motorPin2 = 6;  // Input B for Motor 1 (backward)

// Angle threshold for movement detection
const float angleThreshold = 2.0;

// Maximum motor speed limit
const int maxMotorSpeed = 100; // Max PWM value (0-255)

// Function prototypes
float getAngle();
void calibrate();

void setup() {
  Serial.begin(115200);
  Wire.begin();
  mpu.initialize();

  pinMode(motorPin1, OUTPUT);
  pinMode(motorPin2, OUTPUT);

  calibrate();
}

void loop() {
  input = getAngle();

  if (abs(input - setPoint) > angleThreshold) {
    float error = setPoint - input;
    integral += error;
    float derivative = input - lastInput;

    output = kp * error + ki * integral - kd * derivative;
    lastInput = input;

    // Constrain the output to be between -maxMotorSpeed and maxMotorSpeed
    int motorSpeed = constrain(output, -maxMotorSpeed, maxMotorSpeed);
    
    // Determine direction and control motor speed
    if (motorSpeed > 0) { // Move forward
      analogWrite(motorPin1, motorSpeed);  // PWM signal to forward pin
      analogWrite(motorPin2, 0);           // No signal to backward pin
    } else { // Move backward
      analogWrite(motorPin1, 0);           // No signal to forward pin
      analogWrite(motorPin2, -motorSpeed); // PWM signal to backward pin
    }
  } else {
    // Stop the motors if stable
    analogWrite(motorPin1, 0);
    analogWrite(motorPin2, 0);
  }

  delay(100);
}

float getAngle() {
  int16_t ax, ay, az, gx, gy, gz;
  mpu.getMotion6(&ax, &ay, &az, &gx, &gy, &gz);
  return atan2(ay, az) * 180 / PI;
}

void calibrate() {
  Serial.println("Calibrating... Please place the robot upright.");
  delay(5000);
  setPoint = getAngle();
  Serial.print("Initial setPoint: ");
  Serial.println(setPoint);
}
