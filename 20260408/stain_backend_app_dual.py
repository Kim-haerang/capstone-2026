# -*- coding: utf-8 -*-
import os, sys, time, json, base64, threading
from pathlib import Path
from typing import List, Optional, Literal

import cv2
import numpy as np
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse
from pydantic import BaseModel, Field

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["LANG"] = "C.UTF-8"
os.environ["LC_ALL"] = "C.UTF-8"
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

BASE_DIR = Path(__file__).resolve().parent
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4.1-mini")
JPEG_QUALITY = 85
MAX_PUMP_MS = 5000
MAX_BRUSH_MS = 5000
USE_GPIO = os.environ.get("USE_GPIO", "0") == "1"
CAMERA_INDEX = 0
PREVIEW_FLIP_HORIZONTAL = True
PREVIEW_FLIP_VERTICAL = False
TFLITE_SERVER_URL = os.environ.get("TFLITE_SERVER_URL", "http://127.0.0.1:9000")
TFLITE_CONFIDENCE_THR = float(os.environ.get("TFLITE_CONFIDENCE_THR", "0.82"))

SOURCE_INFO = {
    "coffee": {"type": "water_based", "detergent": "water_based_detergent"},
    "teriyaki_sauce": {"type": "mixed", "detergent": "mixed_detergent"},
    "oriental_dressing": {"type": "oil_based", "detergent": "oil_based_detergent"},
    "soy_sauce": {"type": "water_based", "detergent": "water_based_detergent"},
    "bbq_sauce": {"type": "mixed", "detergent": "mixed_detergent"},
    "gochujang": {"type": "mixed", "detergent": "mixed_detergent"},
    "ketchup": {"type": "water_based", "detergent": "water_based_detergent"},
    "milk": {"type": "water_based", "detergent": "water_based_detergent"},
    "mayonnaise": {"type": "oil_based", "detergent": "oil_based_detergent"},
    "curry": {"type": "mixed", "detergent": "mixed_detergent"},
    "mustard": {"type": "water_based", "detergent": "water_based_detergent"},
    "oil": {"type": "mixed", "detergent": "mixed_detergent"},
}
SOURCE_DISPLAY_KO = {
    "coffee": "커피", "teriyaki_sauce": "데리야끼 소스", "oriental_dressing": "오리엔탈 소스",
    "soy_sauce": "간장", "bbq_sauce": "바비큐 소스", "gochujang": "고추장", "ketchup": "케찹",
    "milk": "우유", "mayonnaise": "마요네즈", "curry": "카레", "mustard": "머스타드", "oil": "기름",
}
DETERGENT_LABEL_KO = {
    "water_based_detergent": "수용성 세제",
    "mixed_detergent": "복합성 세제",
    "oil_based_detergent": "지용성 세제",
}
BRUSH_INTENSITY_MAP = {
    "coffee": "medium", "teriyaki_sauce": "high", "oriental_dressing": "medium", "soy_sauce": "medium",
    "bbq_sauce": "high", "gochujang": "high", "ketchup": "medium", "milk": "low",
    "mayonnaise": "medium", "curry": "high", "mustard": "medium", "oil": "medium",
}
INTENSITY_TO_BRUSH_MS = {"high": 1500, "medium": 1000, "low": 700}
INTENSITY_LABEL_KO = {"high": "강", "medium": "중", "low": "약"}
PUMP_MS_BY_SOURCE = {
    "coffee": 1000, "teriyaki_sauce": 1300, "oriental_dressing": 1200, "soy_sauce": 900,
    "bbq_sauce": 1300, "gochujang": 1400, "ketchup": 1000, "milk": 800,
    "mayonnaise": 1100, "curry": 1400, "mustard": 1000, "oil": 1400,
}
CLASS_REASON_TEMPLATES = {
    "coffee": ["갈색 계열의 액체성 얼룩 패턴이 커피와 유사합니다.", "물 반응 후 색 농도가 비교적 빠르게 옅어지는 편입니다."],
    "teriyaki_sauce": ["갈색과 주황색이 함께 보이는 점성 자국이 데리야끼 소스와 유사합니다.", "물 반응 후에도 점성 자국이 비교적 남는 편입니다."],
    "oriental_dressing": ["옅은 갈색과 기름기 있는 경계가 드레싱 계열과 유사합니다.", "물 반응 후에도 번들거림이 남을 가능성이 큽니다."],
    "soy_sauce": ["짙은 갈색의 얇은 번짐이 간장 얼룩과 유사합니다.", "물 반응에 따라 빠르게 번지며 농도가 낮아질 수 있습니다."],
    "bbq_sauce": ["붉은 갈색의 진한 소스 자국이 바비큐 소스와 유사합니다.", "물 반응 후에도 두꺼운 자국이 남는 편입니다."],
    "gochujang": ["붉은색이 강하고 점도가 높은 얼룩이 고추장 계열과 유사합니다.", "물 반응 뒤에도 색소와 점성 자국이 함께 남기 쉽습니다."],
    "ketchup": ["밝은 붉은색의 균일한 자국이 케찹과 유사합니다.", "물 반응 후 색이 다소 퍼지는 경향이 있습니다."],
    "milk": ["희고 옅은 잔흔이 우유류가 마른 뒤 남는 형태와 유사합니다.", "물 반응 시 경계가 비교적 부드럽게 퍼질 수 있습니다."],
    "mayonnaise": ["흰색 계열의 점성 있는 경계가 마요네즈와 유사합니다.", "물 반응 뒤에도 유분 자국이 남을 수 있습니다."],
    "curry": ["진한 노란색 착색이 카레 계열과 유사합니다.", "물 반응 후에도 색소가 남아 있을 가능성이 높습니다."],
    "mustard": ["밝은 노란색의 소스 자국이 머스타드와 유사합니다.", "물 반응 후 경계가 비교적 부드럽게 번질 수 있습니다."],
    "oil": ["노란 기름기와 번들거림이 기름성 오염과 유사합니다.", "물 반응 뒤에도 투명한 유분 경계가 남기 쉽습니다."],
}

class AnalyzePairRequest(BaseModel):
    before_image: str
    after_image: str

class ExecuteRequest(BaseModel):
    pump_ms: int = Field(ge=0, le=MAX_PUMP_MS)
    brush_ms: int = Field(ge=0, le=MAX_BRUSH_MS)
    brush_intensity: Literal["high", "medium", "low"]

def clamp_int(v: int, low: int, high: int) -> int:
    return max(low, min(high, int(v)))

def apply_preview_flip(frame: np.ndarray) -> np.ndarray:
    if PREVIEW_FLIP_HORIZONTAL and PREVIEW_FLIP_VERTICAL:
        return cv2.flip(frame, -1)
    if PREVIEW_FLIP_HORIZONTAL:
        return cv2.flip(frame, 1)
    if PREVIEW_FLIP_VERTICAL:
        return cv2.flip(frame, 0)
    return frame

def decode_data_url_to_bgr(data_url: str) -> np.ndarray:
    if "," not in data_url:
        raise ValueError("invalid_data_url_format")
    _, encoded = data_url.split(",", 1)
    arr = np.frombuffer(base64.b64decode(encoded), dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("image_decode_failed")
    return img

def image_to_data_url(image_bgr: np.ndarray, quality: int = JPEG_QUALITY) -> str:
    ok, buf = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("image_encode_failed")
    return f"data:image/jpeg;base64,{base64.b64encode(buf).decode()}"

def preprocess_analysis_image(image_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = cv2.equalizeHist(v)
    return cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2BGR)

def compute_reaction_features(before_bgr: np.ndarray, after_bgr: np.ndarray) -> dict:
    b = cv2.resize(before_bgr, (224, 224))
    a = cv2.resize(after_bgr, (224, 224))
    diff = cv2.absdiff(b, a)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    b_hsv = cv2.cvtColor(b, cv2.COLOR_BGR2HSV)
    a_hsv = cv2.cvtColor(a, cv2.COLOR_BGR2HSV)
    return {
        "mean_rgb_diff": float(np.mean(diff)),
        "mean_gray_diff": float(np.mean(diff_gray)),
        "mean_h_diff": float(np.mean(cv2.absdiff(b_hsv[:, :, 0], a_hsv[:, :, 0]))),
        "mean_s_diff": float(np.mean(cv2.absdiff(b_hsv[:, :, 1], a_hsv[:, :, 1]))),
        "mean_v_diff": float(np.mean(cv2.absdiff(b_hsv[:, :, 2], a_hsv[:, :, 2]))),
    }

def call_tflite_server(image_data_url: str) -> dict:
    r = requests.post(f"{TFLITE_SERVER_URL}/predict", json={"image": image_data_url}, timeout=20)
    if not r.ok:
        raise RuntimeError(f"{r.status_code}:{r.text}")
    return r.json()

def combine_predictions(before_pred: dict, after_pred: dict):
    scores, labels_ko = {}, {}
    for item in before_pred.get("top3", []):
        source = item["source"]
        scores[source] = scores.get(source, 0.0) + float(item["confidence"]) * 0.45
        labels_ko[source] = item.get("source_display", SOURCE_DISPLAY_KO.get(source, source))
    for item in after_pred.get("top3", []):
        source = item["source"]
        scores[source] = scores.get(source, 0.0) + float(item["confidence"]) * 0.55
        labels_ko[source] = item.get("source_display", SOURCE_DISPLAY_KO.get(source, source))
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    if not ranked:
        raise RuntimeError("no_prediction_candidates")
    top_candidates = [{"label": s, "label_kr": labels_ko.get(s, SOURCE_DISPLAY_KO.get(s, s)), "score": round(float(sc), 6)} for s, sc in ranked[:5]]
    return top_candidates, float(ranked[0][1]), ranked[0][0]

def build_reason_kr(source: str, reaction: dict, top_candidates: list, decision_source: str) -> List[str]:
    display = SOURCE_DISPLAY_KO.get(source, source)
    hints = CLASS_REASON_TEMPLATES.get(source, [f"얼룩의 색상과 질감이 {display} 특징과 유사합니다.", f"물 반응 후 변화 양상이 {display} 계열과 비슷합니다."])
    cands = ", ".join([f"{x['label_kr']} {round(x['score']*100,1)}%" for x in top_candidates[:3]])
    lines = [
        f"1차 촬영과 2차 촬영을 함께 비교했을 때 {display}일 가능성이 가장 높습니다.",
        hints[0],
        hints[1],
        f"물 반응 전후 평균 변화량은 RGB {reaction['mean_rgb_diff']:.1f}, 명도 {reaction['mean_v_diff']:.1f}로 확인되었습니다.",
        f"최종 판단은 {decision_source} 기준으로 이루어졌으며 주요 후보는 {cands}입니다.",
    ]
    out = []
    for line in lines:
        txt = line.strip()
        if not txt.endswith("입니다."):
            txt = txt.rstrip(".") + "입니다."
        out.append(txt)
    return out[:5]

def build_treatment(source: str) -> dict:
    detergent = SOURCE_INFO.get(source, {"detergent": "mixed_detergent"})["detergent"]
    intensity = BRUSH_INTENSITY_MAP.get(source, "medium")
    return {
        "brush_intensity": intensity,
        "pump_ms": clamp_int(PUMP_MS_BY_SOURCE.get(source, 1000), 0, MAX_PUMP_MS),
        "brush_ms": clamp_int(INTENSITY_TO_BRUSH_MS.get(intensity, 1000), 0, MAX_BRUSH_MS),
        "summary": f"{DETERGENT_LABEL_KO.get(detergent, detergent)}를 사용하고 브러시 강도는 {INTENSITY_LABEL_KO.get(intensity, intensity)}으로 설정합니다.",
    }

def analyze_with_openai(before_img: np.ndarray, after_img: np.ndarray, top_candidates: list, reaction: dict, fallback_source: str) -> dict:
    if OpenAI is None:
        raise RuntimeError("openai_package_missing")
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("openai_api_key_missing")
    client = OpenAI(api_key=api_key)
    candidate_names = ", ".join([c["label"] for c in top_candidates[:5]])
    prompt = f"""
You are the final decision system for a smart stain remover.
Two images show the same stain:
1) before water spray
2) after water spray

Choose one source only from:
{candidate_names}

Reaction features:
{json.dumps(reaction, ensure_ascii=True)}

CNN merged candidates:
{json.dumps(top_candidates, ensure_ascii=False)}

Rules:
- Return JSON only.
- source must be one of the candidate labels above.
- reason_kr must be exactly 5 Korean sentences.
- Every Korean sentence must end with '입니다.'
- If uncertain, prefer the highest CNN candidate.
""".strip()
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": image_to_data_url(before_img), "detail": "low"}},
            {"type": "image_url", "image_url": {"url": image_to_data_url(after_img), "detail": "low"}},
        ]}],
        temperature=0,
        max_tokens=400,
    )
    data = json.loads(response.choices[0].message.content.strip())
    source = data.get("source", fallback_source)
    if source not in [c["label"] for c in top_candidates]:
        source = fallback_source
    conf = max(0.0, min(1.0, float(data.get("confidence", 0.75))))
    reason_kr = data.get("reason_kr", []) if isinstance(data.get("reason_kr", []), list) else []
    cleaned = []
    for item in reason_kr[:5]:
        txt = str(item).strip()
        if not txt.endswith("입니다."):
            txt = txt.rstrip(".") + "입니다."
        cleaned.append(txt)
    while len(cleaned) < 5:
        cleaned.append(f"{SOURCE_DISPLAY_KO.get(source, source)} 후보를 중심으로 전후 반응을 종합한 결과입니다.")
    return {"source": source, "confidence": conf, "reason_kr": cleaned[:5]}

class CameraManager:
    def __init__(self, index: int = CAMERA_INDEX):
        self.index = index
        self.cap: Optional[cv2.VideoCapture] = None
        self.latest_raw_frame: Optional[np.ndarray] = None
        self.lock = threading.Lock()
        self.running = False
        self.thread: Optional[threading.Thread] = None
    def start(self) -> None:
        if self.running:
            return
        self.cap = cv2.VideoCapture(self.index, cv2.CAP_V4L2)
        time.sleep(0.3)
        if not self.cap.isOpened():
            self.cap.release(); self.cap = None
            raise RuntimeError("camera_open_failed")
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
    def _loop(self) -> None:
        assert self.cap is not None
        while self.running and self.cap is not None:
            ret, frame = self.cap.read()
            if ret and frame is not None:
                with self.lock:
                    self.latest_raw_frame = frame.copy()
            time.sleep(0.03)
    def get_raw_frame(self) -> np.ndarray:
        if not self.running:
            self.start()
        timeout = time.time() + 2.0
        while time.time() < timeout:
            with self.lock:
                if self.latest_raw_frame is not None:
                    return self.latest_raw_frame.copy()
            time.sleep(0.05)
        raise RuntimeError("frame_unavailable")
    def get_preview_frame(self) -> np.ndarray:
        return apply_preview_flip(self.get_raw_frame())
    def mjpeg_generator(self):
        if not self.running:
            self.start()
        while True:
            try:
                frame = self.get_preview_frame()
            except Exception:
                time.sleep(0.05); continue
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ok:
                time.sleep(0.03); continue
            yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")
            time.sleep(0.03)

camera_manager = CameraManager()
app = FastAPI(title="Smart Stain Cleaner API - Dual Capture")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

@app.get("/", response_class=HTMLResponse)
def serve_ui():
    ui_path = BASE_DIR / "stain_controller_ui_dual.html"
    if not ui_path.exists():
        raise HTTPException(status_code=500, detail=f"ui_file_missing:{ui_path}")
    return ui_path.read_text(encoding="utf-8")

@app.get("/api/health")
def health_check():
    tflite_ok = False; tflite_detail = None
    try:
        r = requests.get(f"{TFLITE_SERVER_URL}/health", timeout=5)
        tflite_ok = r.ok; tflite_detail = r.json() if r.ok else r.text
    except Exception as e:
        tflite_detail = str(e)
    return {"ok": True, "openai_model": OPENAI_MODEL, "preview_flip_horizontal": PREVIEW_FLIP_HORIZONTAL, "preview_flip_vertical": PREVIEW_FLIP_VERTICAL, "tflite_server_url": TFLITE_SERVER_URL, "tflite_ok": tflite_ok, "tflite_detail": tflite_detail}

@app.get("/api/capture")
def capture():
    try:
        frame = camera_manager.get_preview_frame()
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            raise HTTPException(status_code=500, detail="image_encode_failed")
        return {"image": f"data:image/jpeg;base64,{base64.b64encode(buf).decode()}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"capture_error:{repr(e)}")

@app.get("/api/video_feed")
def video_feed():
    try:
        camera_manager.start()
        return StreamingResponse(camera_manager.mjpeg_generator(), media_type="multipart/x-mixed-replace; boundary=frame")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"video_feed_error:{repr(e)}")

@app.post("/api/analyze-pair")
def analyze_pair(payload: AnalyzePairRequest):
    try:
        before_img = decode_data_url_to_bgr(payload.before_image)
        after_img = decode_data_url_to_bgr(payload.after_image)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"image_parse_error:{repr(e)}")
    before_proc = preprocess_analysis_image(before_img)
    after_proc = preprocess_analysis_image(after_img)
    before_data_url = image_to_data_url(before_proc)
    after_data_url = image_to_data_url(after_proc)
    try:
        before_pred = call_tflite_server(before_data_url)
        after_pred = call_tflite_server(after_data_url)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"tflite_error:{repr(e)}")
    reaction = compute_reaction_features(before_proc, after_proc)
    top_candidates, combined_conf, final_source = combine_predictions(before_pred, after_pred)
    decision_source = "cnn"
    reason_kr = build_reason_kr(final_source, reaction, top_candidates, decision_source)
    if combined_conf < TFLITE_CONFIDENCE_THR:
        try:
            ai_data = analyze_with_openai(before_proc, after_proc, top_candidates, reaction, final_source)
            final_source = ai_data["source"]
            combined_conf = max(combined_conf, ai_data["confidence"])
            reason_kr = ai_data["reason_kr"]
            decision_source = "openai_fallback"
        except Exception:
            decision_source = "cnn_fallback"
    treatment = build_treatment(final_source)
    return {
        "final_label": final_source,
        "final_label_kr": SOURCE_DISPLAY_KO.get(final_source, final_source),
        "source": final_source,
        "source_display": SOURCE_DISPLAY_KO.get(final_source, final_source),
        "confidence": round(float(combined_conf), 6),
        "decision_source": decision_source,
        "before_top3": before_pred.get("top3", []),
        "after_top3": after_pred.get("top3", []),
        "top_candidates": top_candidates,
        "reason_kr": reason_kr,
        "reaction": reaction,
        "group_code": "GROUP_A",
        "group_display": "일반 의류 원단 (기본값)",
        "treatment": treatment,
        "treatment_kr": treatment["summary"],
        "detergent": DETERGENT_LABEL_KO.get(SOURCE_INFO.get(final_source, {"detergent":"mixed_detergent"})["detergent"], "-"),
        "processed_before_url": before_data_url,
        "processed_after_url": after_data_url,
    }

@app.post("/api/execute")
def execute_motor(payload: ExecuteRequest):
    try:
        return {"ok": True, "gpio_used": USE_GPIO, "pump_ms": clamp_int(payload.pump_ms,0,MAX_PUMP_MS), "brush_ms": clamp_int(payload.brush_ms,0,MAX_BRUSH_MS), "brush_intensity": payload.brush_intensity, "message": "모터 실행 요청이 처리되었습니다."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"execute_error:{repr(e)}")
