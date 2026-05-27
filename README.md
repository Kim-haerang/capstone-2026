# 스마트 자동 세척기 (Capstone 2026)

얼룩 이미지를 분류하고(모델 추론), 원단/색상 맥락을 함께 반영해 세제 및 세척 동작을 추천/실행하는 프로젝트입니다.  
웹 UI + FastAPI 백엔드 + TFLite 추론 서버 구조로 구성되어 있습니다.

## 1. 핵심 기능

- 얼룩 분류: MobileNet 기반 CNN(TFLite)으로 오염원 분류
- 보정 로직: 색상군(빨강/갈색/노랑/흰색 계열), 원단 맥락(데님/어두운 배경) 반영
- 세척 계획: 오염원/원단 기반 세제 종류, 펌프 시간, 브러시 강도/횟수 산출
- 세척 후 평가: 1차/2차/세척완료 사진 비교 기반 제거율 피드백
- 학습 피드백 수집: 예측 정오답/실제 라벨 저장(재학습용)

## 2. 기술 스택

- Backend API: `FastAPI`
- Inference: `TensorFlow Lite` (`.tflite`)
- CV/OCR: `OpenCV`, `pytesseract`
- Frontend: 단일 HTML UI (`stain_fused_feedback_app.html`, `stain_fused_feedback_dev.html`)
- Hardware (옵션): `pyserial`, `RPi.GPIO`

## 3. 저장소 파일 구성

- `stain_backend_app.py`: 메인 API 서버, 분석/보정/세척 제어 로직
- `tflite_server.py`: TFLite 추론 API 서버 (`/predict`)
- `stain_fused_feedback_app.html`: 메인 사용자 UI
- `stain_fused_feedback_dev.html`: 개발자 UI
- `stain_classifier_v3.tflite`: 학습된 분류 모델
- `class_names.txt`: 클래스 이름 매핑
- `dev_state_latest.json`: 개발 상태 저장 데이터
- `mobilenet_v1.py`: 모델 관련 코드(학습/구조 참고용)

## 4. 데이터셋/모델 개요

- 모델: MobileNet 기반 CNN
- 클래스: 오염원 12종(예: 커피, 케찹, 고추장, 카레, 머스타드, 기름 등)
- 수집: 클래스당 약 1000장 수준으로 촬영(빛/각도 다양화)
- 증강: Roboflow 기반 증강 데이터 포함
- 총 데이터: 약 36,000장
- 배포: 라즈베리파이 환경 추론을 위해 TFLite 변환 후 사용

## 5. 빠른 실행 방법

### 5-1. Python 환경 준비

```bash
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install fastapi uvicorn numpy opencv-python requests pydantic pytesseract pyserial openai tensorflow
```

`RPi.GPIO`는 라즈베리파이에서만 설치/사용합니다.

### 5-2. 모델 파일 위치 확인

`tflite_server.py`는 기본적으로 아래 경로를 읽습니다.

- `models/stain_classifier_v3.tflite`
- `models/class_names.txt`

현재 저장소 루트에 모델 파일이 있다면, 실행 전에 `models` 폴더를 만들고 옮겨 주세요.

```bash
mkdir models
move stain_classifier_v3.tflite models\stain_classifier_v3.tflite
move class_names.txt models\class_names.txt
```

### 5-3. 서버 실행

터미널 1 (추론 서버):

```bash
uvicorn tflite_server:app --host 0.0.0.0 --port 9000
```

터미널 2 (메인 서버):

```bash
uvicorn stain_backend_app:app --host 0.0.0.0 --port 8000
```

### 5-4. 접속

- 메인 UI: `http://<장비IP>:8000/`
- 개발자 UI: `http://<장비IP>:8000/developer`
- 상태 확인:
  - `http://<장비IP>:8000/api/health`
  - `http://<장비IP>:9000/health`

## 6. 주요 API

- `POST /api/analyze-triplet`: 1차/2차 이미지 분석, 최종 오염원/세제/근거 산출
- `POST /api/execute`: 세척 동작 실행(펌프/브러시)
- `POST /api/cleanup-feedback`: 세척 완료 후 제거율 분석
- `GET /api/video_feed`: 카메라 스트리밍
- `GET /api/capture`: 프레임 캡처
- `POST /api/learning-feedback`: 사용자 피드백 데이터 저장

## 7. 환경 변수 (자주 쓰는 항목)

- `TFLITE_SERVER_URL` (기본: `http://127.0.0.1:9000`)
- `CAMERA_INDEX`
- `OPENAI_API_KEY`, `OPENAI_MODEL`
- `ARDUINO_PORT`, `ARDUINO_BAUD`
- `FEEDBACK_ADJUST_ENABLED`
