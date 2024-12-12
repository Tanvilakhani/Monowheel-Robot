const int O1 = 41;
const int O2 = 42;

void out_init() {

  pinMode(O1, OUTPUT);
  pinMode(O2, OUTPUT);
  Serial.println("Output connections initialized");
}

void out_send(bool a, bool b) {

  if (a) digitalWrite(O1, HIGH);
  else digitalWrite(O1, LOW);
  
  if (b) digitalWrite(O2, HIGH);
  else digitalWrite(O2, LOW);
}