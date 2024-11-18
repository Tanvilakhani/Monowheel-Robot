void drive_motor_init() {

  pinMode(A_IA, OUTPUT);
  pinMode(A_IB, OUTPUT);
  pinMode(A_EN, OUTPUT);
  Serial.println("Drive Motor Initialized");
}

void drive_motor_drive(bool dir, float* speed) {

  if (dir) {

    analogWrite(A_IA, HIGH);
    digitalWrite(A_IB, LOW);
  } else {

    analogWrite(A_IA, LOW);
    digitalWrite(A_IB, HIGH);
  }

  digitalWrite(A_EN, abs(*speed));
}

void drive_motor_stop() {

  digitalWrite(A_IA, HIGH);
  digitalWrite(A_IB, HIGH);
}