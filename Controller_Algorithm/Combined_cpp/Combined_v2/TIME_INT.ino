#include "driver/timer.h"

// Modify the ISR to match the expected function signature
void IRAM_ATTR timer_group1_isr(void *args) {
  // Clear the interrupt
  timer_group_clr_intr_status_in_isr(TIMER_GROUP_1, TIMER_0);

  // Enable alarm again for next time
  timer_group_enable_alarm_in_isr(TIMER_GROUP_1, TIMER_0);

  servo_control_loop();
  drive_motor_control_loop();

  // Your 100 Hz interrupt logic here
  // For example:
  // - Update a counter
  // - Toggle a pin
  // - Set a flag for main loop processing
}

void time_int_init() {
  setCpuFrequencyMhz(240);

  timer_config_t timer_config = {
    .alarm_en = TIMER_ALARM_EN,
    .counter_en = TIMER_START,
    .intr_type = TIMER_INTR_LEVEL,
    .counter_dir = TIMER_COUNT_UP,
    .auto_reload = TIMER_AUTORELOAD_EN,
    .divider = 240  // Dividing 240MHz by 240 gives 1MHz base frequency
  };

  // Initialize timer in group 1, unit 0
  timer_init(TIMER_GROUP_1, TIMER_0, &timer_config);

  // For 100Hz interrupt:
  // Base frequency = 240MHz / 240 = 1MHz
  // For 100Hz: 1MHz / 100 = 10000 ticks
  timer_set_alarm_value(TIMER_GROUP_1, TIMER_0, 10000);

  // Enable timer interrupt
  timer_enable_intr(TIMER_GROUP_1, TIMER_0);

  // Register the ISR
  timer_isr_register(TIMER_GROUP_1, TIMER_0, timer_group1_isr, NULL, ESP_INTR_FLAG_IRAM, NULL);

  // Start the timer
  timer_start(TIMER_GROUP_1, TIMER_0);
}