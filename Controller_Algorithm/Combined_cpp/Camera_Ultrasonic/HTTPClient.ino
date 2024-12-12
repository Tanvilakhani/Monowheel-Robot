#include <HTTPClient.h>

const String serverName = "http://172.20.10.3:5001/data";

void send_dat(float* distanceCm) {

  if (WiFi.status() == WL_CONNECTED) {
    HTTPClient http;

    // Your Domain name with URL path or IP address with path
    http.begin(serverName);

    // Specify content-type header
    http.addHeader("Content-Type", "application/json");

    // Prepare JSON data
    String jsonData = "{\"distance\":\"" + String(*distanceCm) + "\"}";

    // Send HTTP POST request
    int httpResponseCode = http.POST(jsonData);

    // Check the returning code
    if (httpResponseCode == 200) {

    } else if (httpResponseCode > 0) {
      // String response = http.getString();

      // Serial.print("HTTP Response code: ");
      // Serial.println(httpResponseCode);
      // Serial.println(response);
    } else {
      Serial.print("Error on sending POST Request: ");
      Serial.println(httpResponseCode);
    }

    // Free resources
    http.end();
  }
}