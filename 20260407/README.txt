적용 파일
- stain_controller_ui.html : 사용자용 / 개발자용 버튼 전환 UI
- stain_backend_app_final_v3_reason.py : 기존 안정판 백엔드
- tflite_server.py : 기존 TFLite 서버

적용 방법
1) stain_controller_ui.html 파일만 기존 Gmail 폴더의 동일 이름 파일로 교체
2) 백엔드와 TFLite 서버는 그대로 사용 가능

실행
터미널1
source ~/tf_env/bin/activate
cd ~/project/20260407
uvicorn tflite_server:app --host 0.0.0.0 --port 9000

터미널2
source ~/stain_env/bin/activate
cd ~/project/20260407
export OPENAI_API_KEY=
export TFLITE_SERVER_URL=http://127.0.0.1:9000
uvicorn stain_backend_app:app --host 0.0.0.0 --port 8000
