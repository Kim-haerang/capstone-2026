# -*- coding: utf-8 -*-
import os, re, sys, time, json, base64, threading
from pathlib import Path
from typing import List, Literal, Optional, Tuple

import cv2
import numpy as np
import pytesseract
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse
from pydantic import BaseModel, Field
from openai import OpenAI

os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["LANG"] = "C.UTF-8"
os.environ["LC_ALL"] = "C.UTF-8"
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

BASE_DIR = Path(__file__).resolve().parent
DEBUG_DIR = BASE_DIR / "ocr_debug"

OPENAI_MODEL = "gpt-4.1-mini"
JPEG_QUALITY = 85
MAX_PUMP_MS = 5000
MAX_BRUSH_MS = 5000
USE_GPIO = True
PUMP_PIN = 18
BRUSH_PIN = 23
OCR_CONFIG = "--oem 3 --psm 6 -l kor+eng"
CAMERA_INDEX = 0

PREVIEW_FLIP_HORIZONTAL = True
PREVIEW_FLIP_VERTICAL = False
UNFLIP_TAG_BEFORE_OCR = True
SAVE_OCR_DEBUG = True

TFLITE_SERVER_URL = os.environ.get("TFLITE_SERVER_URL", "http://127.0.0.1:9000")
TFLITE_CONFIDENCE_THR = 0.85

PERCENT_PATTERN = re.compile(
    r'(cotton|poly(?:ester)?|denim|nylon|wool|knit|cashmere|silk|'
    r'면|폴리|폴리에스터|데님|나일론|울|니트|캐시미어|실크)'
    r'\s*(\d{1,3})\s*%?', re.IGNORECASE
)

MATERIAL_KEYWORDS = {
    "GROUP_A": ["cotton", "poly", "polyester", "denim", "면", "폴리", "폴리에스터", "데님"],
    "GROUP_B": ["nylon", "나일론"],
    "GROUP_C": ["wool", "knit", "cashmere", "silk", "울", "니트", "캐시미어", "실크"],
}
SYNONYM_MAP = {
    "poly": "polyester", "폴리": "polyester", "폴리에스터": "polyester",
    "면": "cotton", "데님": "denim", "나일론": "nylon",
    "울": "wool", "니트": "knit", "캐시미어": "cashmere", "실크": "silk",
}

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

CLASS_TO_SOURCE = {
    "Brown_cof": "coffee",
    "Brown_deri": "teriyaki_sauce",
    "Brown_ori": "oriental_dressing",
    "Brown_soy": "soy_sauce",
    "Red_bbq": "bbq_sauce",
    "Red_go": "gochujang",
    "Red_ket": "ketchup",
    "White_Milk": "milk",
    "White_ma": "mayonnaise",
    "Yellow_ca": "curry",
    "Yellow_mus": "mustard",
    "Yellow_oil": "oil",
}

SOURCE_DISPLAY_KO = {
    "coffee": "커피",
    "teriyaki_sauce": "데리야끼 소스",
    "oriental_dressing": "오리엔탈 소스",
    "soy_sauce": "간장",
    "bbq_sauce": "바비큐 소스",
    "gochujang": "고추장",
    "ketchup": "케찹",
    "milk": "우유",
    "mayonnaise": "마요네즈",
    "curry": "카레",
    "mustard": "머스타드",
    "oil": "기름",
}
KO_TO_SOURCE = {v: k for k, v in SOURCE_DISPLAY_KO.items()}

BRUSH_INTENSITY_MAP = {
    "coffee": {"GROUP_A": "medium", "GROUP_B": "low", "GROUP_C": "low"},
    "teriyaki_sauce": {"GROUP_A": "high", "GROUP_B": "medium", "GROUP_C": "low"},
    "oriental_dressing": {"GROUP_A": "medium", "GROUP_B": "medium", "GROUP_C": "low"},
    "soy_sauce": {"GROUP_A": "medium", "GROUP_B": "low", "GROUP_C": "low"},
    "bbq_sauce": {"GROUP_A": "high", "GROUP_B": "medium", "GROUP_C": "low"},
    "gochujang": {"GROUP_A": "high", "GROUP_B": "medium", "GROUP_C": "low"},
    "ketchup": {"GROUP_A": "medium", "GROUP_B": "low", "GROUP_C": "low"},
    "milk": {"GROUP_A": "low", "GROUP_B": "low", "GROUP_C": "low"},
    "mayonnaise": {"GROUP_A": "medium", "GROUP_B": "medium", "GROUP_C": "low"},
    "curry": {"GROUP_A": "high", "GROUP_B": "medium", "GROUP_C": "low"},
    "mustard": {"GROUP_A": "medium", "GROUP_B": "low", "GROUP_C": "low"},
    "seasoned_chicken_sauce": {"GROUP_A": "high", "GROUP_B": "medium", "GROUP_C": "low"},
}

INTENSITY_TO_BRUSH_MS = {"high": 1500, "medium": 1000, "low": 700}
INTENSITY_LABEL_KO = {"high": "강", "medium": "중", "low": "약"}
INTENSITY_DESCRIPTION_KO = {
    "high": "강한 브러싱으로 점성이 큰 얼룩을 분해합니다.",
    "medium": "일반적인 얼룩 제거에 적합한 표준 브러싱입니다.",
    "low": "섬유 손상을 줄이기 위한 약한 브러싱입니다.",
}
PUMP_MS_BY_SOURCE = {
    "coffee": 1000,
    "teriyaki_sauce": 1300,
    "oriental_dressing": 1200,
    "soy_sauce": 900,
    "bbq_sauce": 1300,
    "gochujang": 1400,
    "ketchup": 1000,
    "milk": 800,
    "mayonnaise": 1100,
    "curry": 1400,
    "mustard": 1000,
    "oil": 1400,
}
DETERGENT_LABEL_KO = {
    "water_based_detergent": "수용성 세제",
    "mixed_detergent": "복합성 세제",
    "oil_based_detergent": "지용성 세제",
}

CLASS_REASON_TEMPLATES = {
    "Brown_cof": {
        "reason": "얼룩의 갈색 계열 색상과 액체성 번짐 형태가 커피 얼룩 특징과 잘 맞습니다.",
        "evidence": ["갈색 계열 얼룩", "묽게 번진 액체 자국", "커피류와 유사한 색감"],
    },
    "Brown_deri": {
        "reason": "갈색과 주황색이 섞인 색상과 점성 있는 질감이 데리야끼 소스 얼룩 특징과 유사합니다.",
        "evidence": ["갈색과 주황색 혼합", "소스류의 점성 있는 자국", "데리야끼 소스와 유사한 색감"],
    },
    "Brown_ori": {
        "reason": "옅은 갈색 번짐과 기름기 있는 경계가 오리엔탈 소스 계열의 얼룩 특성과 비슷합니다.",
        "evidence": ["옅은 갈색 번짐", "기름기 있는 경계", "드레싱류와 유사한 얼룩 분포"],
    },
    "Brown_soy": {
        "reason": "짙은 갈색의 비교적 얇은 번짐 패턴이 간장 얼룩의 전형적인 형태와 유사합니다.",
        "evidence": ["짙은 갈색 얼룩", "얇게 퍼진 번짐", "간장류와 유사한 색상"],
    },
    "Red_bbq": {
        "reason": "붉은 갈색 계열의 진한 얼룩과 점성 있는 자국이 바비큐 소스와 유사합니다.",
        "evidence": ["붉은 갈색 계열", "진한 소스 자국", "바비큐 소스와 유사한 농도"],
    },
    "Red_go": {
        "reason": "붉은색이 강하고 점도가 높은 얼룩 형태가 고추장 계열의 특징과 잘 맞습니다.",
        "evidence": ["선명한 붉은색", "점도 높은 얼룩", "고추장류와 유사한 질감"],
    },
    "Red_ket": {
        "reason": "밝은 붉은색과 비교적 균일한 소스 자국이 케찹 얼룩 특성과 유사합니다.",
        "evidence": ["밝은 붉은색", "균일한 소스 자국", "케찹과 유사한 색감"],
    },
    "White_Milk": {
        "reason": "희고 옅게 남은 얼룩 자국이 우유류가 마른 뒤 남는 형태와 비슷합니다.",
        "evidence": ["희거나 옅은 자국", "부드러운 번짐", "우유류와 유사한 잔흔"],
    },
    "White_ma": {
        "reason": "흰색 계열의 점성 있는 얼룩 경계가 마요네즈 같은 유분성 소스와 유사합니다.",
        "evidence": ["흰색 계열 잔흔", "점성 있는 경계", "유분성 소스와 유사한 자국"],
    },
    "Yellow_ca": {
        "reason": "노란색이 진하게 남은 얼룩 색상이 카레 계열의 전형적인 특징과 가깝습니다.",
        "evidence": ["진한 노란색", "색소가 남는 얼룩", "카레와 유사한 착색"],
    },
    "Yellow_mus": {
        "reason": "밝은 노란색의 소스 자국이 머스타드 얼룩과 유사한 색상 특성을 보입니다.",
        "evidence": ["밝은 노란색", "소스 자국 형태", "머스타드와 유사한 색감"],
    },
    "Yellow_oil": {
        "reason": "노란 기름기 얼룩 특성과 비슷합니다.",
        "evidence": ["노란 기름기", "소스성 번짐", "기름 유사한 형태"],
    },
}

class AnalyzeRequest(BaseModel):
    stain_image: str
    tag_image: str

class ExecuteRequest(BaseModel):
    pump_ms: int = Field(ge=0, le=MAX_PUMP_MS)
    brush_ms: int = Field(ge=0, le=MAX_BRUSH_MS)
    brush_intensity: Literal["high", "medium", "low"]

class Treatment(BaseModel):
    brush_intensity: Literal["high", "medium", "low"]
    pump_ms: int
    brush_ms: int
    summary: str

class AnalyzeResponse(BaseModel):
    source: str
    source_display: str
    confidence: float
    cnn_class: str
    cnn_confidence: float
    material_group: Literal["GROUP_A", "GROUP_B", "GROUP_C"]
    material_reason: str
    reason: str
    evidence: List[str]
    detergent: str
    treatment: Treatment
    ocr_text_preview: List[str]
    ocr_debug_files: List[str]
    tflite_top3: List[dict]

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

def restore_tag_orientation(frame: np.ndarray) -> np.ndarray:
    if not UNFLIP_TAG_BEFORE_OCR:
        return frame
    return apply_preview_flip(frame)

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

def ensure_debug_dir() -> None:
    if SAVE_OCR_DEBUG:
        DEBUG_DIR.mkdir(parents=True, exist_ok=True)

def save_debug_image(name: str, image: np.ndarray) -> str:
    ensure_debug_dir()
    ts = time.strftime("%Y%m%d_%H%M%S")
    filename = f"{ts}_{name}.png"
    path = DEBUG_DIR / filename
    cv2.imwrite(str(path), image)
    return filename

def preprocess_tag_image_for_ocr(image: np.ndarray) -> List[Tuple[str, np.ndarray]]:
    h, w = image.shape[:2]
    scale = 1.5 if max(h, w) < 1000 else 1.0
    resized = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8)).apply(gray)
    return [
        ("00_original", image),
        ("01_resized", resized),
        ("02_gray", gray),
        ("03_clahe", clahe),
    ]

def ocr_material_texts(image: np.ndarray) -> Tuple[List[str], List[str]]:
    variants = preprocess_tag_image_for_ocr(image)
    best_texts: List[str] = []
    best_score = -1.0
    saved_files: List[str] = []

    for name, img in variants:
        if SAVE_OCR_DEBUG:
            try:
                saved_files.append(save_debug_image(name, img))
            except Exception:
                pass
        try:
            raw = pytesseract.image_to_data(
                img,
                config=OCR_CONFIG,
                output_type=pytesseract.Output.DICT,
                lang="kor+eng",
            )
        except Exception:
            continue

        valid = []
        for i in range(len(raw["text"])):
            txt = str(raw["text"][i]).strip()
            try:
                conf = int(float(raw["conf"][i]))
            except Exception:
                conf = -1
            if conf >= 20 and txt:
                valid.append((txt.lower(), conf / 100.0))

        score = sum(c for _, c in valid) / len(valid) if valid else 0.0
        if score > best_score:
            best_score = score
            best_texts = [t for t, _ in valid]

    return best_texts, saved_files

def parse_material_group(ocr_texts: List[str]) -> Tuple[str, str]:
    full_text = " ".join(ocr_texts).lower()
    percentages: dict[str, int] = {}

    for match in PERCENT_PATTERN.finditer(full_text):
        kw = SYNONYM_MAP.get(match.group(1).lower(), match.group(1).lower())
        percentages[kw] = percentages.get(kw, 0) + int(match.group(2))

    dominant: Optional[str] = None
    reason = ""

    if percentages:
        dominant = max(percentages, key=percentages.get)
        reason = f"택 OCR에서 {dominant} 재질 비율이 가장 높게 감지되었습니다."
    else:
        for _, keywords in MATERIAL_KEYWORDS.items():
            for kw in keywords:
                if kw.lower() in full_text:
                    dominant = SYNONYM_MAP.get(kw.lower(), kw.lower())
                    reason = f"택 OCR에서 '{dominant}' 재질 키워드를 감지했습니다."
                    break
            if dominant:
                break

    if not dominant:
        return "GROUP_A", "택 OCR에서 재질을 읽지 못해 기본값 GROUP_A로 처리했습니다."

    for group, keywords in MATERIAL_KEYWORDS.items():
        normalized = [SYNONYM_MAP.get(k.lower(), k.lower()) for k in keywords]
        if dominant in normalized or dominant in [k.lower() for k in keywords]:
            return group, reason

    return "GROUP_A", "재질 그룹 매핑에 실패해 기본값 GROUP_A로 처리했습니다."

def normalize_source_name(raw_source: str, raw_class: Optional[str] = None) -> str:
    if raw_source in SOURCE_INFO:
        return raw_source
    if raw_source in KO_TO_SOURCE:
        return KO_TO_SOURCE[raw_source]
    if raw_class and raw_class in CLASS_TO_SOURCE:
        return CLASS_TO_SOURCE[raw_class]
    raise ValueError(f"invalid_source:{raw_source}")

def call_tflite_server(stain_image_data_url: str) -> dict:
    resp = requests.post(
        f"{TFLITE_SERVER_URL}/predict",
        json={"image": stain_image_data_url},
        timeout=20,
    )
    if not resp.ok:
        try:
            detail = resp.json().get("detail", resp.text)
        except Exception:
            detail = resp.text
        raise RuntimeError(f"TFLite 서버 호출 실패: {detail}")

    data = resp.json()
    cnn_class = data.get("cnn_class")
    source = normalize_source_name(data.get("source"), cnn_class)
    confidence = max(0.0, min(1.0, float(data.get("confidence", 0.0))))

    top3_out = []
    for item in data.get("top3", []):
        cname = item.get("class_name")
        s = normalize_source_name(item.get("source", ""), cname)
        top3_out.append({
            "rank": int(item.get("rank", len(top3_out) + 1)),
            "class_name": cname,
            "source": s,
            "source_display": SOURCE_DISPLAY_KO.get(s, s),
            "confidence": float(item.get("confidence", 0.0)),
        })

    return {
        "cnn_class": cnn_class,
        "source": source,
        "source_display": SOURCE_DISPLAY_KO.get(source, source),
        "confidence": confidence,
        "top3": top3_out,
    }

def build_reason_and_evidence_from_cnn(cnn_class: str, source: str, confidence: float) -> Tuple[str, List[str]]:
    template = CLASS_REASON_TEMPLATES.get(cnn_class)
    source_display = SOURCE_DISPLAY_KO.get(source, source)
    conf_pct = round(confidence * 100, 1)

    if template:
        reason = f"{template['reason']} CNN 분류 결과는 {source_display}이며 신뢰도는 {conf_pct}%입니다."
        evidence = list(template["evidence"])
        evidence.append(f"CNN 분류 결과 {source_display}")
        return reason, evidence[:4]

    reason = f"얼룩의 색상과 형태가 {source_display} 특징과 유사하며 CNN 모델 신뢰도는 {conf_pct}%입니다."
    evidence = [
        f"추정 오염원: {source_display}",
        f"CNN 클래스: {cnn_class}",
        f"신뢰도: {conf_pct}%",
    ]
    return reason, evidence

GPIO = None
if USE_GPIO:
    try:
        import RPi.GPIO as GPIO  # type: ignore
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(PUMP_PIN, GPIO.OUT)
        GPIO.setup(BRUSH_PIN, GPIO.OUT)
    except Exception:
        GPIO = None
        USE_GPIO = False

def run_motor(pump_ms: int, brush_ms: int) -> None:
    pump_ms = clamp_int(pump_ms, 0, MAX_PUMP_MS)
    brush_ms = clamp_int(brush_ms, 0, MAX_BRUSH_MS)
    if not USE_GPIO or GPIO is None:
        print(f"[SIM] pump={pump_ms}ms brush={brush_ms}ms")
        return
    GPIO.output(PUMP_PIN, 1)
    time.sleep(pump_ms / 1000.0)
    GPIO.output(PUMP_PIN, 0)
    GPIO.output(BRUSH_PIN, 1)
    time.sleep(brush_ms / 1000.0)
    GPIO.output(BRUSH_PIN, 0)

api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("missing_openai_api_key")
client = OpenAI(api_key=api_key)

def build_treatment(source: str, material_group: str) -> Treatment:
    intensity = BRUSH_INTENSITY_MAP[source][material_group]
    pump_ms = clamp_int(PUMP_MS_BY_SOURCE[source], 0, MAX_PUMP_MS)
    brush_ms = clamp_int(INTENSITY_TO_BRUSH_MS[intensity], 0, MAX_BRUSH_MS)
    detergent_ko = DETERGENT_LABEL_KO.get(SOURCE_INFO[source]["detergent"], SOURCE_INFO[source]["detergent"])
    intensity_ko = INTENSITY_LABEL_KO.get(intensity, intensity)
    summary = (
        f"{detergent_ko} 기준으로 펌프를 {pump_ms}ms 분사하고, "
        f"브러시는 {intensity_ko} 강도로 {brush_ms}ms 작동합니다. "
        f"{INTENSITY_DESCRIPTION_KO[intensity]}"
    )
    return Treatment(
        brush_intensity=intensity,
        pump_ms=pump_ms,
        brush_ms=brush_ms,
        summary=summary,
    )

def extract_json_block(text: str) -> str:
    if not text:
        raise ValueError("empty_model_output")
    cleaned = text.strip()
    if "```" in cleaned:
        parts = cleaned.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith("{") and part.endswith("}"):
                return part
    start = cleaned.find("{")
    if start == -1:
        raise ValueError(f"json_start_not_found:{cleaned[:120]}")
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(cleaned)):
        ch = cleaned[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
        else:
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return cleaned[start:i+1]
    raise ValueError(f"unterminated_json_block:{cleaned[:200]}")

def sanitize_model_text(text: str) -> str:
    return text.replace("\r", " ").strip()

def parse_openai_json(text: str) -> dict:
    block = extract_json_block(sanitize_model_text(text))
    try:
        return json.loads(block)
    except json.JSONDecodeError:
        repaired = re.sub(r'[\x00-\x1f]', ' ', block)
        repaired = repaired.replace("\n", "\\n")
        return json.loads(repaired)

def fallback_parse_openai_text(text: str, default_source: Optional[str] = None) -> dict:
    cleaned = sanitize_model_text(text)
    source = default_source
    for name in SOURCE_INFO.keys():
        if name in cleaned:
            source = name
            break
    if source is None:
        raise ValueError(f"fallback_source_not_found:{cleaned[:200]}")
    conf_match = re.search(r'"confidence"\s*:\s*([0-9.]+)', cleaned)
    confidence = float(conf_match.group(1)) if conf_match else 0.7
    confidence = max(0.0, min(1.0, confidence))
    source_display = SOURCE_DISPLAY_KO.get(source, source)
    return {
        "source": source,
        "confidence": confidence,
        "reason": f"이미지 특징을 종합했을 때 {source_display}일 가능성이 높습니다.",
        "evidence": ["색상 특징", "얼룩 형태", f"CNN 후보 {source_display}"],
    }

def analyze_with_openai(
    stain_img: np.ndarray,
    tag_img: np.ndarray,
    material_group: str,
    material_reason: str,
    cnn_class: Optional[str] = None,
    cnn_source: Optional[str] = None,
    cnn_confidence: float = 0.0,
) -> dict:
    source_names = list(SOURCE_INFO.keys())
    cnn_hint = ""
    if cnn_class:
        cnn_hint = (
            f"\nCNN hint:\n"
            f"- class: {cnn_class}\n"
            f"- source: {cnn_source}\n"
            f"- confidence: {cnn_confidence:.4f}\n"
        )

    prompt = f"""
Analyze two images:
1) clothing stain
2) clothing care tag

Possible sources:
{", ".join(source_names)}

Fabric group: {material_group}
Fabric reason: {material_reason}
{cnn_hint}

IMPORTANT:
- reason must be written in Korean
- evidence must be written in Korean
- source must remain one of the English source names above
- JSON only
- Do not use quotation marks inside Korean strings
- Keep reason as one short sentence
- Keep evidence as 2 or 3 short phrases
- Even if CNN confidence is high, explain stain color or texture in evidence when possible

Return JSON only:
{{
  "source": "coffee",
  "confidence": 0.0,
  "reason": "한국어 짧은 설명 한 문장",
  "evidence": ["한국어 근거1", "한국어 근거2", "한국어 근거3"]
}}
"""
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_to_data_url(stain_img), "detail": "low"}},
                {"type": "image_url", "image_url": {"url": image_to_data_url(tag_img), "detail": "low"}},
            ],
        }],
        max_tokens=180,
        temperature=0,
    )
    raw_text = response.choices[0].message.content.strip()
    try:
        data = parse_openai_json(raw_text)
    except Exception:
        data = fallback_parse_openai_text(raw_text, default_source=cnn_source)

    source = data.get("source")
    if source not in SOURCE_INFO:
        if cnn_source in SOURCE_INFO:
            source = cnn_source
        else:
            raise ValueError(f"invalid_source:{source}")
    data["source"] = source
    data["confidence"] = max(0.0, min(1.0, float(data.get("confidence", 0.0))))
    if not isinstance(data.get("evidence"), list) or not data["evidence"]:
        source_display = SOURCE_DISPLAY_KO.get(source, source)
        data["evidence"] = [f"재질 그룹 제외", f"오염원 후보: {source_display}", "색상 및 형태 특징"]
    if not isinstance(data.get("reason"), str) or not data["reason"].strip():
        source_display = SOURCE_DISPLAY_KO.get(source, source)
        data["reason"] = f"이미지 특징을 종합했을 때 {source_display}일 가능성이 높습니다."
    return data

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
            self.cap.release()
            self.cap = None
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
                time.sleep(0.05)
                continue
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ok:
                time.sleep(0.03)
                continue
            yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")
            time.sleep(0.03)

    def stop(self) -> None:
        self.running = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None

camera_manager = CameraManager()

app = FastAPI(title="Smart Stain Cleaner API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_class=HTMLResponse)
def serve_ui():
    ui_path = BASE_DIR / "stain_controller_ui.html"
    if not ui_path.exists():
        raise HTTPException(status_code=500, detail=f"ui_file_missing:{ui_path}")
    return ui_path.read_text(encoding="utf-8")

@app.get("/api/health")
def health_check():
    tflite_ok = False
    tflite_detail = None
    try:
        r = requests.get(f"{TFLITE_SERVER_URL}/health", timeout=5)
        tflite_ok = r.ok
        tflite_detail = r.json() if r.ok else r.text
    except Exception as e:
        tflite_detail = str(e)

    return {
        "ok": True,
        "gpio": USE_GPIO,
        "openai_model": OPENAI_MODEL,
        "preview_flip_horizontal": PREVIEW_FLIP_HORIZONTAL,
        "preview_flip_vertical": PREVIEW_FLIP_VERTICAL,
        "unflip_tag_before_ocr": UNFLIP_TAG_BEFORE_OCR,
        "save_ocr_debug": SAVE_OCR_DEBUG,
        "tflite_server_url": TFLITE_SERVER_URL,
        "tflite_ok": tflite_ok,
        "tflite_detail": tflite_detail,
    }

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
        return StreamingResponse(
            camera_manager.mjpeg_generator(),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"video_feed_error:{repr(e)}")

@app.post("/api/analyze", response_model=AnalyzeResponse)
def analyze(payload: AnalyzeRequest):
    try:
        stain_img = decode_data_url_to_bgr(payload.stain_image)
        tag_img = decode_data_url_to_bgr(payload.tag_image)
        tag_img = restore_tag_orientation(tag_img)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"image_parse_error:{repr(e)}")

    try:
        tflite_data = call_tflite_server(payload.stain_image)
        cnn_class = tflite_data["cnn_class"]
        cnn_source = tflite_data["source"]
        cnn_source_display = tflite_data["source_display"]
        cnn_confidence = float(tflite_data["confidence"])
        tflite_top3 = tflite_data["top3"]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"tflite_error:{repr(e)}")

    try:
        ocr_texts, debug_files = ocr_material_texts(tag_img)
        material_group, material_reason = parse_material_group(ocr_texts)

        if cnn_confidence >= TFLITE_CONFIDENCE_THR:
            source = cnn_source
            confidence = cnn_confidence
            reason, evidence = build_reason_and_evidence_from_cnn(cnn_class, source, cnn_confidence)
        else:
            ai_data = analyze_with_openai(
                stain_img,
                tag_img,
                material_group,
                material_reason,
                cnn_class=cnn_class,
                cnn_source=cnn_source,
                cnn_confidence=cnn_confidence,
            )
            source = ai_data["source"]
            confidence = ai_data["confidence"]
            reason = ai_data["reason"]
            evidence = ai_data["evidence"]

        treatment = build_treatment(source, material_group)

        return AnalyzeResponse(
            source=source,
            source_display=SOURCE_DISPLAY_KO.get(source, source),
            confidence=confidence,
            cnn_class=cnn_class,
            cnn_confidence=cnn_confidence,
            material_group=material_group,
            material_reason=material_reason,
            reason=reason,
            evidence=evidence,
            detergent=DETERGENT_LABEL_KO.get(SOURCE_INFO[source]["detergent"], SOURCE_INFO[source]["detergent"]),
            treatment=treatment,
            ocr_text_preview=ocr_texts[:12],
            ocr_debug_files=debug_files,
            tflite_top3=tflite_top3,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"analyze_error:{repr(e)}")

@app.post("/api/execute")
def execute(payload: ExecuteRequest):
    try:
        run_motor(payload.pump_ms, payload.brush_ms)
        return {
            "ok": True,
            "pump_ms": payload.pump_ms,
            "brush_ms": payload.brush_ms,
            "brush_intensity": payload.brush_intensity,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"execute_error:{repr(e)}")

@app.on_event("shutdown")
def shutdown_cleanup():
    global GPIO
    camera_manager.stop()
    if USE_GPIO and GPIO is not None:
        GPIO.cleanup()
