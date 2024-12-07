const int O1 = 42;
const int O2 = 41;

void out_init() {

  pinMode(O1, OUTPUT);
  pinMode(O2, OUTPUT);
}

void drive() {

  digitalWrite(O1, HIGH);
}

void drive_stop() {

  digitalWrite(O1, LOW);
}

void object_detected() {

  digitalWrite(O2, HIGH);
}

void object_not_detected() {

  digitalWrite(O2, LOW);
}