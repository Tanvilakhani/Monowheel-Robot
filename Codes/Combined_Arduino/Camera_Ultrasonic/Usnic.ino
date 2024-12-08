// Ultrasonic Sensor Pins
const int trigPin = D5;
const int echoPin = D6;

// Global Variables
#define SOUND_VELOCITY 0.034
#define CM_TO_INCH 0.393701

unsigned long previousMillis = 0;
float dist_thresh = 50;

unsigned long pundtim;

void usnic_init() {

  pinMode(trigPin, OUTPUT);
  pinMode(echoPin, INPUT);
}

bool measureDistance() {

  // Clear the trigPin
  digitalWrite(trigPin, LOW);
  delayMicroseconds(2);

  // Trigger pulse
  digitalWrite(trigPin, HIGH);
  delayMicroseconds(10);
  digitalWrite(trigPin, LOW);

  // Measure pulse duration
  long duration = pulseIn(echoPin, HIGH);

  // Validate measurement
  if (duration > 0) {
    distanceCm = duration * SOUND_VELOCITY / 2;

    // Optional range validation
    if (distanceCm > 400) return false;
    else if (distanceCm < dist_thresh) {

      out_send(0, 0);
      pundtim = millis();
    } else if (distanceCm >= dist_thresh) {

      if ((millis() - pundtim) >= 2000) {

        out_send(1, 0);
      }
    }

    // Log distance
    // Serial.print("Distance (cm): ");
    // Serial.println(distanceCm);

    return true;
  }

  return false;
}