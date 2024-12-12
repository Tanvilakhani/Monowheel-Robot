const int I1 = 19;
const int I2 = 23;

void in_init() {

  pinMode(I1, INPUT_PULLDOWN);
  pinMode(I2, INPUT_PULLDOWN);
  Serial.println("Input connections initialized");
}

void in_stat(bool& a, bool& b) {

  a = digitalRead(I1);
  b = digitalRead(I2);
}