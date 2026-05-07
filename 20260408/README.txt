[스마트 얼룩 제거기 - 2회 촬영 버전]

이 버전은 같은 얼룩을 2번 촬영합니다.
1) 물 뿌리기 전 이미지
2) 물 뿌린 후 이미지

그 다음 처리 순서는 아래와 같습니다.
- 두 이미지를 각각 전처리
- TFLite CNN으로 각각 추론
- 두 결과를 가중 평균으로 결합
- 물 전/후 반응 특징(RGB/HSV/Gray diff) 계산
- 종합 신뢰도가 낮으면 OpenAI로 최종판단 및 한국어 설명 5개 생성

중요:
- 현재 TFLite 모델 자체는 '2장 동시 입력'으로 재학습된 모델이 아니라,
  기존 단일 이미지 분류 모델을 2번 호출하여 결합하는 구조입니다.
- 즉, 지금은 '코드 구조를 2회 촬영 방식으로 변경'한 버전입니다.
- 진짜 성능을 더 끌어올리려면 나중에 before/after pair 기반으로 재학습한 모델을 붙이면 됩니다.

==================================================
1. 폴더 구성
==================================================

stain_dual_capture_project/
 ├─ stain_backend_app_dual.py
 ├─ tflite_server_dual.py
 ├─ templates/
 │   └─ stain_controller_ui_dual.html
 ├─ static/
 │   └─ uploads/
 ├─ models/
 │   ├─ class_names.txt
 │   └─ stain_classifier_v2.tflite   ← 네 모델 파일 여기에 넣기
 ├─ requirements_app.txt
 ├─ requirements_tf.txt
 ├─ .env.example
 └─ README.txt

==================================================
2. 네가 해야 하는 것
==================================================

(1) 기존 TFLite 모델 파일을 아래 위치에 복사
    models/stain_classifier_v2.tflite

(2) .env.example 을 복사해서 .env 로 변경
    cp .env.example .env

(3) .env 안에 OPENAI_API_KEY 입력

==================================================
3. 가상환경 예시
==================================================

[TF 서버용]
python -m venv tf_env
source tf_env/bin/activate
pip install -r requirements_tf.txt

[앱 서버용]
python -m venv stain_env
source myenv/bin/activate
pip install -r requirements_app.txt

라즈베리파이에서 source 대신 아래처럼 쓸 수 있습니다.
. tf_env/bin/activate
. myenv/bin/activate

==================================================
4. 실행 방법
==================================================

[터미널 1 - TFLite 서버]
cd stain_dual_capture_project
. tf_env/bin/activate
uvicorn tflite_server_dual:app --host 0.0.0.0 --port 9000

[터미널 2 - 메인 앱]
cd stain_dual_capture_project
. stain_env/bin/activate
uvicorn stain_backend_app_dual:app --host 0.0.0.0 --port 8000

브라우저 접속:
http://127.0.0.1:8000

다른 기기에서 접속:
http://라즈베리파이IP:8000

==================================================
5. 현재 동작 방식 핵심
==================================================

- 1차 촬영 버튼: 물 뿌리기 전 이미지 저장
- 2차 촬영 버튼: 물 뿌린 후 이미지 저장
- 분석 시작:
  1) 두 이미지 업로드
  2) 전처리 이미지 저장
  3) TFLite 서버에 before/after 전송
  4) before/after 확률 가중결합
  5) confidence < threshold 면 OpenAI fallback
  6) 결과 + 처리방법 + 선택이유 5개 표시

==================================================
6. 주의사항
==================================================

- OpenAI fallback 은 OPENAI_API_KEY 없으면 동작하지 않습니다.
- 신뢰도가 임계값보다 높으면 CNN 결과만 사용합니다.
- 선택이유는 항상 5개가 나오도록 되어 있습니다.
- 문장 끝은 모두 '입니다' 체를 유지하도록 OpenAI 프롬프트를 넣었습니다.
- 기존 프로젝트의 정확한 CSS/HTML 원본 파일이 지금 대화에 없어서,
  이번에는 2회 촬영 구조가 바로 실행되도록 독립형 UI로 구성했습니다.

==================================================
7. 다음 단계 추천
==================================================

정확도를 더 높이려면 아래 중 하나를 하면 됩니다.

A. 현재 방식 유지
- 기존 single-image TFLite 모델 그대로 사용
- before/after 결과를 결합
- 가장 빠르게 적용 가능

B. pair 전용 재학습
- 입력을 [before, after, diff] 구조로 만들어 재학습
- 물 반응 정보가 CNN 내부에 직접 반영되므로 성능 향상 가능

