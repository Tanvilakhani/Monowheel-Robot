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

  // int spd = map(abs(*speed), 0, 255, 110, 255);
  int spd = 90 + (abs(*speed));
  analogWrite(A_EN, spd);
}

void drive_motor_stop() {

  digitalWrite(A_IA, HIGH);
  digitalWrite(A_IB, HIGH);
  digitalWrite(A_EN, LOW);
}