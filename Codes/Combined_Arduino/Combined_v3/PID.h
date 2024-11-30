class PIDController {
private:
  float kp, ki, kd;            // PID gains
  float setpoint;              // Desired value
  float integral;              // Integral term
  float prevError;             // Previous error for derivative
  unsigned long prevTime;      // Previous time for dt calculation
  float outputMin, outputMax;  // Output limits

public:
  PIDController(float p, float i, float d, float min, float max) {
    kp = p;
    ki = i;
    kd = d;
    outputMin = min;
    outputMax = max;
    integral = 0;
    prevError = 0;
    prevTime = 0;
    setpoint = 0;
  }

  void setSetpoint(float sp) {
    setpoint = sp;
  }

  void resetIntegral() {
    integral = 0;
  }

  float compute(float input) {
    // Calculate time difference
    unsigned long now = millis();
    float dt = (now - prevTime) / 1000.0;  // Convert to seconds

    if (prevTime == 0) {
      prevTime = now;
      return 0;
    }

    // Calculate error
    float error = setpoint - input;

    // Proportional term
    float P = kp * error;

    // Integral term
    integral += error * dt;
    float I = ki * integral;

    // Derivative term
    float derivative = (error - prevError) / dt;
    float D = kd * derivative;

    // Calculate total output
    float output = P + I + D;

    // Apply output limits
    if (output > outputMax) {
      output = outputMax;
    } else if (output < outputMin) {
      output = outputMin;
    }

    // Save values for next iteration
    prevError = error;
    prevTime = now;

    return output;
  }
};