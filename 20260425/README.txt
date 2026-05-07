이 세트는 최신 HTML 업로드본을 기준으로 실행하도록 맞춘 파일입니다.

포함:
- stain_backend_final.py
- stain_fused_feedback_fixed.html

특징:
- 개발자 화면 버튼은 메인 HTML 안에 내장된 base64 Blob 방식으로 열립니다.
- 별도 dev html 파일이 없어도 버튼이 열리도록 설계되었습니다.
- 만약 버튼 클릭 시 아무 반응이 없다면 브라우저 팝업 차단을 해제해야 합니다.

실행:
python -m uvicorn stain_backend_final:app --host 0.0.0.0 --port 8000
