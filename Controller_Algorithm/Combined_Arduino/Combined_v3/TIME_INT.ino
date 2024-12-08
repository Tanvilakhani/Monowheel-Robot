#include "driver/timer.h"

int motor_counter;
int motor_dutyCycle;

// ISR for Timer Group 1 (First timer)
void IRAM_ATTR timer_group1_isr(void *args) {
  timer_group_clr_intr_status_in_isr(TIMER_GROUP_1, TIMER_0);
  timer_group_enable_alarm_in_isr(TIMER_GROUP_1, TIMER_0);

  servo_control_loop();
  drive_motor_control_loop();
}

// ISR for Timer Group 0 (Second timer)
void IRAM_ATTR timer_group0_isr(void *args) {
  timer_group_clr_intr_status_in_isr(TIMER_GROUP_0, TIMER_1);
  timer_group_enable_alarm_in_isr(TIMER_GROUP_0, TIMER_1);

  motor_counter++;
  if (motor_counter >= 255) motor_counter = 0;

  if (motor_counter < motor_dutyCycle) {
    digitalWrite(A_EN, HIGH);
  } else {
    digitalWrite(A_EN, LOW);
  }
}

void set_motor_speed(int *spd) {

  motor_dutyCycle = *spd;
}

void time_int_init() {

  setCpuFrequencyMhz(240);

  // Create tasks pinned to different cores
  xTaskCreatePinnedToCore(
    timer1_init_task,  // Task function for Timer 1
    "Timer1InitTask",  // Task name
    2048,              // Stack size
    NULL,              // Task parameters
    1,                 // Priority
    NULL,              // Task handle
    0                  // Core 0
  );

  xTaskCreatePinnedToCore(
    timer0_init_task,  // Task function for Timer 0
    "Timer0InitTask",  // Task name
    2048,              // Stack size
    NULL,              // Task parameters
    1,                 // Priority
    NULL,              // Task handle
    1                  // Core 1
  );
}

void timer1_init_task(void *pvParameters) {
  // Timer 1 initialization (on Core 0)
  timer_config_t timer_config = {
    .alarm_en = TIMER_ALARM_EN,
    .counter_en = TIMER_START,
    .intr_type = TIMER_INTR_LEVEL,
    .counter_dir = TIMER_COUNT_UP,
    .auto_reload = TIMER_AUTORELOAD_EN,
    .divider = 240  // 1MHz base frequency
  };

  // Initialize Timer 1 (TIMER_GROUP_1)
  timer_init(TIMER_GROUP_1, TIMER_0, &timer_config);
  timer_set_alarm_value(TIMER_GROUP_1, TIMER_0, 10000);
  timer_enable_intr(TIMER_GROUP_1, TIMER_0);
  timer_isr_register(TIMER_GROUP_1, TIMER_0, timer_group1_isr, NULL, ESP_INTR_FLAG_IRAM, NULL);
  timer_start(TIMER_GROUP_1, TIMER_0);

  // Delete this task after initialization
  vTaskDelete(NULL);
}

void timer0_init_task(void *pvParameters) {
  // Timer 0 initialization (on Core 1)
  timer_config_t timer_config = {
    .alarm_en = TIMER_ALARM_EN,
    .counter_en = TIMER_START,
    .intr_type = TIMER_INTR_LEVEL,
    .counter_dir = TIMER_COUNT_UP,
    .auto_reload = TIMER_AUTORELOAD_EN,
    .divider = 240  // 1MHz base frequency
  };

  // Initialize Timer 2 (TIMER_GROUP_0)
  timer_init(TIMER_GROUP_0, TIMER_1, &timer_config);
  timer_set_alarm_value(TIMER_GROUP_0, TIMER_1, 8);
  timer_enable_intr(TIMER_GROUP_0, TIMER_1);
  timer_isr_register(TIMER_GROUP_0, TIMER_1, timer_group0_isr, NULL, ESP_INTR_FLAG_IRAM, NULL);
  timer_start(TIMER_GROUP_0, TIMER_1);

  // Delete this task after initialization
  vTaskDelete(NULL);
}