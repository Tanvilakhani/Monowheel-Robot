// Arduino Nano Timer Interrupt Example
// Uses Timer1 for precise timing interrupts

// Pin definitions for Arduino Nano
const int ledPin = 13;    // Built-in LED on Nano (Pin 13)
const int outputPin = 2;  // Additional output pin for testing

// Global variables
volatile unsigned long interruptCount = 0;
volatile bool ledState = false;

// Frequency options (uncomment desired frequency)
//#define FREQ_1HZ    // 1 time per second
//#define FREQ_2HZ    // 2 times per second
#define FREQ_10HZ   // 10 times per second
//#define FREQ_100HZ  // 100 times per second

void setup() {
  // Initialize pins
  pinMode(ledPin, OUTPUT);
  pinMode(outputPin, OUTPUT);
  
  // Start serial communication
  Serial.begin(9600);
  
  // Configure Timer1
  cli();                      // Disable interrupts during setup
  TCCR1A = 0;                // Clear Timer1 control registers
  TCCR1B = 0;
  
  // Calculate compare match value based on desired frequency
  #ifdef FREQ_1HZ
    OCR1A = 15624;  // 1 Hz (1 second)
  #elif defined(FREQ_2HZ)
    OCR1A = 7812;   // 2 Hz (0.5 seconds)
  #elif defined(FREQ_10HZ)
    OCR1A = 1562;   // 10 Hz (0.1 seconds)
  #elif defined(FREQ_100HZ)
    OCR1A = 156;    // 100 Hz (0.01 seconds)
  #endif
  
  // Set CTC mode (Clear Timer on Compare Match)
  TCCR1B |= (1 << WGM12);
  
  // Set prescaler to 1024
  TCCR1B |= (1 << CS12) | (1 << CS10);
  
  // Enable Timer1 compare interrupt
  TIMSK1 |= (1 << OCIE1A);
  
  sei();  // Re-enable interrupts
  
  Serial.println("Timer1 interrupt initialized");
}

// Timer1 Interrupt Service Routine (ISR)
ISR(TIMER1_COMPA_vect) {
  interruptCount++;
  ledState = !ledState;
  
  // Toggle both LED and output pin
  digitalWrite(ledPin, ledState);
  digitalWrite(outputPin, ledState);
}

void loop() {
  // Print interrupt count and frequency every second
  Serial.print("Interrupts: ");
  Serial.print(interruptCount);
  
  #ifdef FREQ_1HZ
    Serial.println(" (1 Hz)");
  #elif defined(FREQ_2HZ)
    Serial.println(" (2 Hz)");
  #elif defined(FREQ_10HZ)
    Serial.println(" (10 Hz)");
  #elif defined(FREQ_100HZ)
    Serial.println(" (100 Hz)");
  #endif
  
  delay(1000);  // Update serial output every second
}