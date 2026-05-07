Triplet V3 mustard guard version

핵심 변경점
- TFLite 입력 전처리는 건드리지 않았습니다. 기존에 잘 되던 다른 클래스 분포를 최대한 유지합니다.
- 머스타드 vs 갈색 계열 혼동이 의심될 때만 백엔드에서 추가 재검토를 수행합니다.
- 추가 재검토는 OpenAI 최종판단에서만 사용하며, 원본 이미지 + 색상 집중 crop 이미지를 함께 비교합니다.
- CNN이 매우 강하게 같은 갈색 소스로 일치하는 경우에는 불필요한 재검토를 생략합니다.

추가 환경변수(선택)
export YELLOW_BROWN_RECHECK_ENABLED=1
export MUSTARD_BROWN_RECHECK_MAX_CONF=0.96
export MUSTARD_BROWN_MIN_CAND_SCORE=0.12

실행
1) tf_env
uvicorn tflite_server_v3:app --host 127.0.0.1 --port 9000

2) stain_env
export TFLITE_SERVER_URL=http://127.0.0.1:9000

export YELLOW_BROWN_RECHECK_ENABLED=1
python -m uvicorn stain_backend_final_rpi_servo30:app --host 0.0.0.0 --port 8000




