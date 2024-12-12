// Frequency options (uncomment desired frequency)
//#define FREQ_1HZ    // 1 time per second
//#define FREQ_2HZ    // 2 times per second
//#define FREQ_10HZ  // 10 times per second
#define FREQ_100HZ  // 100 times per second

volatile uint8_t timer2_counter = 0;

void time_int_init() {

  cli();  // disable interrupts

  TCCR2A = 0;  // Clear Timer2 control registers
  TCCR2B = 0;

// Calculate compare match value based on desired frequency
#ifdef FREQ_1HZ
  // For 1 Hz: Need to use a counter since 15624 > 255 (Timer2 max)
  OCR2A = 249;  // Create a 62.5Hz base frequency
  volatile static uint8_t counter = 0;
  // Will count up to 62 in the ISR to achieve 1Hz

#elif defined(FREQ_2HZ)
  // For 2 Hz: Need to use a counter since 7812 > 255
  OCR2A = 249;  // Create a 62.5Hz base frequency
  volatile static uint8_t counter = 0;
  // Will count up to 31 in the ISR to achieve 2Hz

#elif defined(FREQ_10HZ)
  // For 10 Hz: Need to use a counter since 1562 > 255
  OCR2A = 249;  // Create a 62.5Hz base frequency
  volatile static uint8_t counter = 0;
  // Will count up to 6 in the ISR to achieve ~10Hz

#elif defined(FREQ_100HZ)
  OCR2A = 155;  // 100 Hz directly achievable
#endif

  // Set CTC mode (Clear Timer on Compare Match)
  TCCR2A |= (1 << WGM21);

  // Set prescaler to 1024
  TCCR2B |= (1 << CS22) | (1 << CS21) | (1 << CS20);

  // Enable Timer2 compare interrupt
  TIMSK2 |= (1 << OCIE2A);

  sei();

  Serial.println("Timer2 interrupt initialized");
}

// Timer2 Interrupt Service Routine (ISR)
ISR(TIMER2_COMPA_vect) {
    #ifdef FREQ_1HZ
        if (++timer2_counter >= 62) {  // 62.5Hz / 62 ≈ 1Hz
            timer2_counter = 0;
            servo_control_loop();
            //drive_motor_control_loop();
        }
    
    #elif defined(FREQ_2HZ)
        if (++timer2_counter >= 31) {  // 62.5Hz / 31 ≈ 2Hz
            timer2_counter = 0;
            servo_control_loop();
            //drive_motor_control_loop();
        }
    
    #elif defined(FREQ_10HZ)
        if (++timer2_counter >= 6) {   // 62.5Hz / 6 ≈ 10Hz
            timer2_counter = 0;
            servo_control_loop();
            //drive_motor_control_loop();
        }
    
    #elif defined(FREQ_100HZ)
        servo_control_loop();  // Direct 100Hz execution
        //drive_motor_control_loop();
    #endif
}