Triplet V3 final decider version

핵심 변경점
- 1차/2차 CNN 결과를 각각 독립적으로 유지합니다.
- 두 결과가 같은 소스를 가리키고 충분히 높으면 CNN 결과를 바로 채택합니다.
- 두 결과가 다르면 OpenAI가 두 이미지와 두 CNN 결과를 함께 보고 최종 선택합니다.
- bbq_sauce는 갈색 바비큐 소스가 아니라 붉은 양념치킨 소스 의미로 프롬프트와 표시명을 수정했습니다.

실행
1) tf_env
uvicorn tflite_server_v3:app --host 127.0.0.1 --port 9000

2) stain_env
export TFLITE_SERVER_URL=http://127.0.0.1:9000

uvicorn stain_backend_app_triplet_v3_final_decider:app --host 0.0.0.0 --port 8000
