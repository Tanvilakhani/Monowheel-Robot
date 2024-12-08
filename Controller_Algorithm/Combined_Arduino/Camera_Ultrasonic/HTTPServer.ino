#include <WebServer.h>

WebServer server(90);

void server_init() {

  // Define routes/endpoints
  server.on("/climb", handleClimb);
  server.on("/reverse", handleReverse);

  server.onNotFound(handleNotFound);

  // Start the server
  server.begin();
  Serial.println("HTTP server started");
}

void handleClimb() {

  String message = "Climb OK";
  out_send(1, 1);
  server.send(200, "text/plain", message);
}

void handleReverse() {

  String message = "Reverse OK";
  out_send(0, 1);
  server.send(200, "text/plain", message);
}

// Handler for 404 Not Found errors
void handleNotFound() {
  String message = "File Not Found\n\n";
  message += "URI: " + server.uri() + "\n";
  //message += "Method: " + (server.method() == HTTP_GET ? "GET" : "Unknown") + "\n";
  message += "Arguments: " + String(server.args()) + "\n";

  server.send(404, "text/plain", message);
}

void run_server(void* parameter) {
  for (;;) {
    server.handleClient();
    //vTaskDelay(pdMS_TO_TICKS(1000));
  }
}