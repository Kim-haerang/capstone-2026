String input = "";

// 아두이노: 4,5,6 = 세제 펌프 / 7 = 수중모터
const int pumps[3] = {4, 5, 6};
const int UMOTOR_PIN = 7;

// ===== 펌프 상태 =====
bool isPumpRunning = false;
int currentPump = -1;
unsigned long pumpStartMs = 0;
unsigned long pumpDurationMs = 0;

// 펄스 제어
bool pumpOutputOn = false;
unsigned long lastPumpPulseToggleMs = 0;
unsigned long pulseOnMs = 70;
unsigned long pulseOffMs = 150;

// ===== 수중모터 상태 =====
bool isUMotorRunning = false;
unsigned long uMotorStartMs = 0;
unsigned long uMotorDurationMs = 0;

void allPumpsOff() {
  for (int i = 0; i < 3; i++) {
    digitalWrite(pumps[i], LOW);
  }
  isPumpRunning = false;
  currentPump = -1;
  pumpStartMs = 0;
  pumpDurationMs = 0;
  pumpOutputOn = false;
  lastPumpPulseToggleMs = 0;
}

void uMotorOff() {
  digitalWrite(UMOTOR_PIN, LOW);
  isUMotorRunning = false;
  uMotorStartMs = 0;
  uMotorDurationMs = 0;
}

void allOff() {
  allPumpsOff();
  uMotorOff();
}

void startPump(int idx, unsigned long durationMs) {
  if (idx < 0 || idx >= 3) {
    Serial.println("ERR:BAD_PUMP_INDEX");
    return;
  }

  pulseOnMs = 70;
  pulseOffMs = 150;

  allPumpsOff();

  currentPump = idx;
  isPumpRunning = true;
  pumpStartMs = millis();
  pumpDurationMs = durationMs;

  digitalWrite(pumps[idx], HIGH);
  pumpOutputOn = true;
  lastPumpPulseToggleMs = millis();

  Serial.print("OK:P");
  Serial.print(idx + 1);
  Serial.print(",");
  Serial.println(durationMs);
}

void startUMotor(unsigned long durationMs) {
  if (durationMs == 0) {
    uMotorOff();
    Serial.println("OK:UM_OFF");
    return;
  }

  uMotorOff();

  isUMotorRunning = true;
  uMotorStartMs = millis();
  uMotorDurationMs = durationMs;
  digitalWrite(UMOTOR_PIN, HIGH);

  Serial.print("OK:UM,");
  Serial.println(durationMs);
}

void handleCommand(String cmd) {
  cmd.trim();

  if (cmd.length() == 0) return;

  if (cmd == "PING") {
    Serial.println("PONG");
    return;
  }

  if (cmd == "ALL_OFF") {
    allOff();
    Serial.println("OK:ALL_OFF");
    return;
  }

  if (cmd == "UM_OFF") {
    uMotorOff();
    Serial.println("OK:UM_OFF");
    return;
  }

  if (cmd == "STATUS") {
    if (isPumpRunning) {
      Serial.print("RUNNING,P");
      Serial.print(currentPump + 1);
      Serial.print(",");
      Serial.println(pumpDurationMs);
    } else if (isUMotorRunning) {
      Serial.print("RUNNING,UM,");
      Serial.println(uMotorDurationMs);
    } else {
      Serial.println("IDLE");
    }
    return;
  }

  // P1,3000 형식
  if (cmd.charAt(0) == 'P') {
    int commaIndex = cmd.indexOf(',');
    if (commaIndex <= 1) {
      Serial.println("ERR:BAD_FORMAT");
      return;
    }

    int pumpNum = cmd.substring(1, commaIndex).toInt();
    unsigned long msTime = cmd.substring(commaIndex + 1).toInt();

    if (pumpNum < 1 || pumpNum > 3) {
      Serial.println("ERR:BAD_PUMP_NUM");
      return;
    }

    if (msTime == 0) {
      allPumpsOff();
      Serial.println("OK:ZERO_TIME_OFF");
      return;
    }

    startPump(pumpNum - 1, msTime);
    return;
  }

  // UM,3000 형식
  if (cmd.startsWith("UM,")) {
    int commaIndex = cmd.indexOf(',');
    if (commaIndex < 0) {
      Serial.println("ERR:BAD_UM_FORMAT");
      return;
    }

    unsigned long msTime = cmd.substring(commaIndex + 1).toInt();
    startUMotor(msTime);
    return;
  }

  Serial.println("ERR:UNKNOWN_COMMAND");
}

void setup() {
  Serial.begin(9600);

  for (int i = 0; i < 3; i++) {
    pinMode(pumps[i], OUTPUT);
    digitalWrite(pumps[i], LOW);
  }

  pinMode(UMOTOR_PIN, OUTPUT);
  digitalWrite(UMOTOR_PIN, LOW);

  allOff();
  delay(1000);
  Serial.println("READY");
}

void loop() {
  while (Serial.available() > 0) {
    char c = Serial.read();

    if (c == '\n') {
      handleCommand(input);
      input = "";
    } else if (c != '\r') {
      input += c;
    }
  }

  unsigned long now = millis();

  if (isPumpRunning) {
    if (now - pumpStartMs >= pumpDurationMs) {
      allPumpsOff();
      Serial.println("DONE");
      return;
    }

    if (pumpOutputOn) {
      if (now - lastPumpPulseToggleMs >= pulseOnMs) {
        digitalWrite(pumps[currentPump], LOW);
        pumpOutputOn = false;
        lastPumpPulseToggleMs = now;
      }
    } else {
      if (now - lastPumpPulseToggleMs >= pulseOffMs) {
        digitalWrite(pumps[currentPump], HIGH);
        pumpOutputOn = true;
        lastPumpPulseToggleMs = now;
      }
    }
  }

  if (isUMotorRunning) {
    if (now - uMotorStartMs >= uMotorDurationMs) {
      uMotorOff();
      Serial.println("DONE");
    }
  }
}
