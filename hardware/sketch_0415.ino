String input = "";

// 펌프 제어 핀
const int pumps[4] = {4, 5, 6, 7};

// 상태 변수
bool isRunning = false;
int currentPump = -1;
unsigned long pumpStartMs = 0;
unsigned long pumpDurationMs = 0;

// 펄스 제어
bool pumpOutputOn = false;
unsigned long lastPulseToggleMs = 0;
unsigned long pulseOnMs = 70;
unsigned long pulseOffMs = 150;

void allOff() {
  for (int i = 0; i < 4; i++) {
    digitalWrite(pumps[i], LOW);
  }
  isRunning = false;
  currentPump = -1;
  pumpStartMs = 0;
  pumpDurationMs = 0;
  pumpOutputOn = false;
  lastPulseToggleMs = 0;
}

void startPump(int idx, unsigned long durationMs) {
  if (idx < 0 || idx >= 4) {
    Serial.println("ERR:BAD_PUMP_INDEX");
    return;
  }

  // 🔥 펌프별 세기 조절
  if (idx == 3) {  
    // 물 (4번)
    pulseOnMs = 60;
    pulseOffMs = 140;
  } else {
    // 세제
    pulseOnMs = 70;
    pulseOffMs = 150;
  }

  allOff();

  currentPump = idx;
  isRunning = true;
  pumpStartMs = millis();
  pumpDurationMs = durationMs;

  digitalWrite(pumps[idx], HIGH);
  pumpOutputOn = true;
  lastPulseToggleMs = millis();

  Serial.print("OK:P");
  Serial.print(idx + 1);
  Serial.print(",");
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

  if (cmd == "STATUS") {
    if (isRunning) {
      Serial.print("RUNNING,P");
      Serial.print(currentPump + 1);
      Serial.print(",");
      Serial.println(pumpDurationMs);
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

    if (pumpNum < 1 || pumpNum > 4) {
      Serial.println("ERR:BAD_PUMP_NUM");
      return;
    }

    if (msTime == 0) {
      allOff();
      Serial.println("OK:ZERO_TIME_OFF");
      return;
    }

    startPump(pumpNum - 1, msTime);
    return;
  }

  Serial.println("ERR:UNKNOWN_COMMAND");
}

void setup() {
  Serial.begin(9600);

  for (int i = 0; i < 4; i++) {
    pinMode(pumps[i], OUTPUT);
  }

  allOff();
  delay(1000);
  Serial.println("READY");
}

void loop() {
  // 시리얼 입력
  while (Serial.available() > 0) {
    char c = Serial.read();

    if (c == '\n') {
      handleCommand(input);
      input = "";
    } else if (c != '\r') {
      input += c;
    }
  }

  // 펄스 제어 + 시간 종료
  if (isRunning) {
    unsigned long now = millis();

    // 전체 시간 끝나면 종료
    if (now - pumpStartMs >= pumpDurationMs) {
      allOff();
      Serial.println("DONE");
      return;
    }

    // ON/OFF 반복
    if (pumpOutputOn) {
      if (now - lastPulseToggleMs >= pulseOnMs) {
        digitalWrite(pumps[currentPump], LOW);
        pumpOutputOn = false;
        lastPulseToggleMs = now;
      }
    } else {
      if (now - lastPulseToggleMs >= pulseOffMs) {
        digitalWrite(pumps[currentPump], HIGH);
        pumpOutputOn = true;
        lastPulseToggleMs = now;
      }
    }
  }
}
