/***
  Adapted for Seeed Studio Xiao ESP32-S3
***/

// Assign GPIO pins for TRIG and ECHO
const int trigPin = D5;
const int echoPin = D6;

// Define sound velocity in cm/uS
#define SOUND_VELOCITY 0.034
#define CM_TO_INCH 0.393701

long duration;
float distanceCm;
float distanceInch;

unsigned long ptim;

void usnic_init() {

  pinMode(trigPin, OUTPUT);  // Set the trigPin as an OUTPUT
  pinMode(echoPin, INPUT);   // Set the echoPin as an INPUT
}

void get_distance() {

  if (millis() - ptim >= 1000) {

    // Clear the trigPin
    digitalWrite(trigPin, LOW);
    delayMicroseconds(2);

    // Set the trigPin HIGH for 10 microseconds
    digitalWrite(trigPin, HIGH);
    delayMicroseconds(10);
    digitalWrite(trigPin, LOW);

    // Read the echoPin and measure the duration of the pulse
    duration = pulseIn(echoPin, HIGH);

    // Calculate the distance in cm and inches
    distanceCm = duration * SOUND_VELOCITY / 2;
    distanceInch = distanceCm * CM_TO_INCH;

    // Print the results to the Serial Monitor
    Serial.print("Distance (cm): ");
    Serial.println(distanceCm);
    Serial.print("Distance (inch): ");
    Serial.println(distanceInch);
    ptim = millis();
  }
}