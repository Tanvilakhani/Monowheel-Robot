const int I1 = 19;
const int I2 = 23;

void in_init() {

  pinMode(I1, INPUT_PULLDOWN);
  pinMode(I2, INPUT_PULLDOWN);
  Serial.println("Input connections initialized");
}
bool in_stat() {

  bool a = digitalRead(I1);
  bool b = digitalRead(I2);

  return a, b;
}