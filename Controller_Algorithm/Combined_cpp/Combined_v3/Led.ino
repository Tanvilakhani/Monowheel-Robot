void led_init() {

  pinMode(led_pin, OUTPUT);
  Serial.println("LED Initilized.");
}

void led_on() {

  digitalWrite(led_pin, HIGH);
}

void led_off() {

  digitalWrite(led_pin, LOW);
}