void drive_motor_init() {

  pinMode(A_IA, OUTPUT);
  pinMode(A_IB, OUTPUT);
  Serial.println("Drive Motor Initialized");
}

void drive_motor_drive(bool dir, float* speed) {
Serial.println(abs(*speed));
  if (dir) {

    analogWrite(A_IA, abs(*speed));
    digitalWrite(A_IB, LOW);
  } else {

    analogWrite(A_IA, abs(*speed));
    digitalWrite(A_IB, HIGH);
  }
}

void drive_motor_stop() {

  digitalWrite(A_IA, HIGH);
  digitalWrite(A_IB, HIGH);
}