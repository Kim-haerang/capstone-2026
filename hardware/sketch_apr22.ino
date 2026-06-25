
#define STEP_PIN 2
#define DIR_PIN  3
#define EN_PIN   4

void stepMoveForTime(unsigned long moveTimeMs, bool dir) {
  // 드라이버 활성화
  digitalWrite(EN_PIN, LOW);

  // 방향 설정
  digitalWrite(DIR_PIN, dir);

  unsigned long startTime = millis();

  // 지정한 시간 동안만 회전
  while (millis() - startTime < moveTimeMs) {
    digitalWrite(STEP_PIN, HIGH);
    delayMicroseconds(800);

    digitalWrite(STEP_PIN, LOW);
    delayMicroseconds(800);
  }

  // 정지 후 드라이버 비활성화 (발열 방지)
  digitalWrite(EN_PIN, HIGH);
}

void setup() {
  pinMode(STEP_PIN, OUTPUT);
  pinMode(DIR_PIN, OUTPUT);
  pinMode(EN_PIN, OUTPUT);

  // 시작 시 비활성화
  digitalWrite(EN_PIN, HIGH);

  delay(1000); // 전원 안정화
}

void loop() {
  // 위로 1초 이동
  stepMoveForTime(3000, HIGH);
  delay(500);

  // 아래로 1초 이동
  stepMoveForTime(2000, LOW); 

  // 완전히 멈춤
  while (true) {
    digitalWrite(EN_PIN, HIGH); // 드라이버 OFF
  }
}

