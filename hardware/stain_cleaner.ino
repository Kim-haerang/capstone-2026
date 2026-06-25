// ======================================
// Arduino 통합 코드
// - 스텝모터: 라즈베리파이 명령으로 DOWN / UP
// - 펌프/수중모터: 기존 코드 그대로 유지
// ======================================

String input = "";

// ===== 스텝모터 핀 =====
// 주의: 기존 EN_PIN 4는 펌프 1번과 겹쳐서 8번으로 변경
#define STEP_PIN 3
#define DIR_PIN  2
#define EN_PIN   8

// 방향은 실제 움직임 보고 반대로 바꾸면 됨
#define DIR_DOWN HIGH
#define DIR_UP   LOW

unsigned int stepPulseDelayUs = 800;

// ===== 펌프 / 수중모터 핀 =====
// 아두이노: 4,5,6 = 세제 펌프 / 7 = 수중모터
const int pumps[3] = {4, 5, 6};
const int UMOTOR_PIN = 7;

// ===== 펌프 상태 =====
bool isPumpRunning = false;
int currentPump = -1;
unsigned long pumpStartMs = 0;
unsigned long pumpDurationMs = 0;

bool pumpOutputOn = false;
unsigned long lastPumpPulseToggleMs = 0;
unsigned long pulseOnMs = 70;
unsigned long pulseOffMs = 150;

// ===== 수중모터 상태 =====
bool isUMotorRunning = false;
unsigned long uMotorStartMs = 0;
unsigned long uMotorDurationMs = 0;

// ======================================
// 공통 OFF
// ======================================
void stepperOff() {
  digitalWrite(EN_PIN, HIGH);  // DRV8825 비활성화
}

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
  stepperOff();
}

// ======================================
// 스텝모터 동작
// ======================================
void stepMoveForTime(unsigned long moveTimeMs, bool dir) {
  digitalWrite(EN_PIN, LOW);     // 드라이버 활성화
  delay(10);                     // 드라이버 켜질 시간 확보

  digitalWrite(DIR_PIN, dir);    // 방향 설정
  delayMicroseconds(100);        // DIR 안정화 시간 확보

  unsigned long startTime = millis();

  while (millis() - startTime < moveTimeMs) {
    digitalWrite(STEP_PIN, HIGH);
    delayMicroseconds(stepPulseDelayUs);

    digitalWrite(STEP_PIN, LOW);
    delayMicroseconds(stepPulseDelayUs);
  }

  digitalWrite(EN_PIN, HIGH);    // 발열 방지
}

void stepDown(unsigned long msTime) {
  Serial.print("OK:STEP_DOWN,");
  Serial.println(msTime);

  stepMoveForTime(msTime, DIR_DOWN);

  Serial.println("DONE:STEP_DOWN");
}

void stepUp(unsigned long msTime) {
  Serial.print("OK:STEP_UP,");
  Serial.println(msTime);

  stepMoveForTime(msTime, DIR_UP);

  Serial.println("DONE:STEP_UP");
}

// ======================================
// 펌프 / 수중모터 동작
// ======================================
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

// ======================================
// 명령 처리
// ======================================
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

  // ==============================
  // 스텝모터 명령
  // ==============================

  // 기본: 5초 내려가기
  if (cmd == "STEP_DOWN") {
    stepDown(5000);
    return;
  }

  // 기본: 5초 올라가기
  if (cmd == "STEP_UP") {
    stepUp(5000);
    return;
  }

  // 시간 지정: STEP_DOWN,5000
  if (cmd.startsWith("STEP_DOWN,")) {
    int commaIndex = cmd.indexOf(',');
    unsigned long msTime = cmd.substring(commaIndex + 1).toInt();

    if (msTime == 0) {
      Serial.println("ERR:BAD_STEP_TIME");
      return;
    }

    stepDown(msTime);
    return;
  }

  // 시간 지정: STEP_UP,5000
  if (cmd.startsWith("STEP_UP,")) {
    int commaIndex = cmd.indexOf(',');
    unsigned long msTime = cmd.substring(commaIndex + 1).toInt();

    if (msTime == 0) {
      Serial.println("ERR:BAD_STEP_TIME");
      return;
    }

    stepUp(msTime);
    return;
  }

  // ==============================
  // 펌프 명령: P1,3000
  // ==============================
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

  // ==============================
  // 수중모터 명령: UM,3000
  // ==============================
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

// ======================================
// setup
// ======================================
void setup() {
  Serial.begin(9600);

  pinMode(STEP_PIN, OUTPUT);
  pinMode(DIR_PIN, OUTPUT);
  pinMode(EN_PIN, OUTPUT);

  digitalWrite(STEP_PIN, LOW);
  digitalWrite(DIR_PIN, LOW);
  digitalWrite(EN_PIN, HIGH); // 시작 시 스텝 드라이버 OFF

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

// ======================================
// loop
// ======================================
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

  // 펌프 펄스 제어
  if (isPumpRunning) {
    if (now - pumpStartMs >= pumpDurationMs) {
      allPumpsOff();
      Serial.println("DONE:PUMP");
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

  // 수중모터 시간 제어
  if (isUMotorRunning) {
    if (now - uMotorStartMs >= uMotorDurationMs) {
      uMotorOff();
      Serial.println("DONE:UM");
    }
  }
}