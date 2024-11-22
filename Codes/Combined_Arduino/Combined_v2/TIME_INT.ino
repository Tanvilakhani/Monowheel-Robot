#include "driver/timer.h"

// Modify the ISR to match the expected function signature
void IRAM_ATTR timer_group1_isr(void *args) {
  // Clear the interrupt
  timer_group_clr_intr_status_in_isr(TIMER_GROUP_1, TIMER_0);

  servo_control_loop();
  drive_motor_control_loop();

  // Your 100 Hz interrupt logic here
  // For example:
  // - Update a counter
  // - Toggle a pin
  // - Set a flag for main loop processing
}

void time_int_init() {
  timer_config_t timer_config = {
    .alarm_en = TIMER_ALARM_EN,
    .counter_en = TIMER_PAUSE,
    .intr_type = TIMER_INTR_LEVEL,
    .counter_dir = TIMER_COUNT_UP,
    .auto_reload = TIMER_AUTORELOAD_EN,
    .divider = 80  // 1 MHz clock (80 MHz / 80)
  };

  // Initialize timer in group 1, unit 0
  timer_init(TIMER_GROUP_1, TIMER_0, &timer_config);

  // Calculate alarm value for 100 Hz (1 MHz / 100)
  timer_set_alarm_value(TIMER_GROUP_1, TIMER_0, 10000);

  // Enable timer interrupt
  timer_enable_intr(TIMER_GROUP_1, TIMER_0);
  timer_isr_register(TIMER_GROUP_1, TIMER_0, timer_group1_isr, NULL, 0, NULL);

  // Start the timer
  timer_start(TIMER_GROUP_1, TIMER_0);
}