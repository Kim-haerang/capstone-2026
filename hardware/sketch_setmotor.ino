String inputString = "";
bool stringComplete = false;

const int pumps[4] = {4, 5, 6, 7};

void setup() {
  Serial.begin(9600);

  for (int i = 0; i < 4; i++) {
    pinMode(pumps[i], OUTPUT);
    digitalWrite(pumps[i], LOW);
  }

  inputString.reserve(100);
  Serial.println("ARDUINO_READY");
}

void loop() {
  if (stringComplete) {
    inputString.trim();

    if (inputString.startsWith("PUMP")) {
      handlePumpCommand(inputString);
    }
    else if (inputString == "STOP_ALL") {
      stopAllPumps();
      Serial.println("DONE_STOP_ALL");
    }
    else {
      Serial.print("UNKNOWN_COMMAND: ");
      Serial.println(inputString);
    }

    inputString = "";
    stringComplete = false;
  }
}

void handlePumpCommand(String cmd) {
  int underscoreIndex = cmd.indexOf('_');

  if (underscoreIndex == -1) {
    Serial.print("BAD_COMMAND: ");
    Serial.println(cmd);
    return;
  }

  String pumpPart = cmd.substring(4, underscoreIndex);
  String msPart = cmd.substring(underscoreIndex + 1);

  int pumpNumber = pumpPart.toInt();
  int durationMs = msPart.toInt();

  if (pumpNumber < 1 || pumpNumber > 4 || durationMs <= 0) {
    Serial.print("BAD_VALUES: ");
    Serial.println(cmd);
    return;
  }

  runPumpWithPulse(pumpNumber - 1, (unsigned long)durationMs);

  Serial.print("DONE_PUMP");
  Serial.print(pumpNumber);
  Serial.print("_");
  Serial.println(durationMs);
}

void runPumpWithPulse(int pumpIndex, unsigned long totalMs) {

  // 🔥 물 / 세제 동일하게 약하게
  int onTime =40;
  int offTime = 250;

  stopAllPumps();

  unsigned long elapsed = 0;

  while (elapsed < totalMs) {
    digitalWrite(pumps[pumpIndex], HIGH);
    delay(onTime);

    digitalWrite(pumps[pumpIndex], LOW);
    delay(offTime);

    elapsed += (unsigned long)(onTime + offTime);
  }

  digitalWrite(pumps[pumpIndex], LOW);
}

void stopAllPumps() {
  for (int i = 0; i < 4; i++) {
    digitalWrite(pumps[i], LOW);
  }
}

void serialEvent() {
  while (Serial.available()) {
    char inChar = (char)Serial.read();

    if (inChar == '\n') {
      stringComplete = true;
    } else if (inChar != '\r') {
      inputString += inChar;
    }
  }
}