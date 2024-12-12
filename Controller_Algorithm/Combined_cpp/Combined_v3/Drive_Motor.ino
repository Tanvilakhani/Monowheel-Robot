void drive_motor_init() {
  pinMode(A_IA, OUTPUT);
  pinMode(A_IB, OUTPUT);
  pinMode(A_EN, OUTPUT);

  Serial.println("Drive Motor Initialized");
}

void drive_motor_drive(bool dir, float* speed) {

  if (dir) {
    digitalWrite(A_IA, HIGH);
    digitalWrite(A_IB, LOW);
  } else {
    digitalWrite(A_IA, LOW);
    digitalWrite(A_IB, HIGH);
  }

  // Map and write the speed to the PWM channel
  int spd = 60 + abs(*speed); // Adjust speed range as needed
  spd = constrain(spd, 60, 255);
  set_motor_speed(&spd);
}

void drive_motor_stop() {
  
  digitalWrite(A_IA, HIGH);
  digitalWrite(A_IB, HIGH);
  int spd = 0;
  set_motor_speed(&spd);
}