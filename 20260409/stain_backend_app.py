# -*- coding: utf-8 -*-
import os
import re
import sys
import time
import json
import base64
import threading
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import cv2
import numpy as np
import pytesseract
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
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
DEBUG_DIR = BASE_DIR / "ocr_debug"
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4.1-mini")
JPEG_QUALITY = 85
MAX_PUMP_MS = 5000
MAX_BRUSH_MS = 5000
OCR_CONFIG = "--oem 3 --psm 6 -l kor+eng"
CAMERA_INDEX = int(os.environ.get("CAMERA_INDEX", "0"))
PREVIEW_FLIP_HORIZONTAL = True
PREVIEW_FLIP_VERTICAL = False
UNFLIP_TAG_BEFORE_OCR = True
SAVE_OCR_DEBUG = True
TFLITE_SERVER_URL = os.environ.get("TFLITE_SERVER_URL", "http://127.0.0.1:9000")
OPENAI_LOW_CONF_THR = float(os.environ.get("OPENAI_LOW_CONF_THR", "0.84"))
PAIR_STRONG_CONF_THR = float(os.environ.get("PAIR_STRONG_CONF_THR", "0.90"))
CNN_DIRECT_AGREE_HIGH = float(os.environ.get("CNN_DIRECT_AGREE_HIGH", "0.75"))
CNN_DIRECT_AGREE_LOW = float(os.environ.get("CNN_DIRECT_AGREE_LOW", "0.68"))

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
GROUP_DISPLAY = {
    "GROUP_A": "일반 의류 원단 (면/폴리/데님)",
    "GROUP_B": "합성 섬유 (나일론)",
    "GROUP_C": "민감 원단 (울/실크/캐시미어)",
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
    "bbq_sauce": "양념치킨 소스",
    "gochujang": "고추장",
    "ketchup": "케찹",
    "milk": "우유",
    "mayonnaise": "마요네즈",
    "curry": "카레",
    "mustard": "머스타드",
    "oil": "기름",
}
KO_TO_SOURCE = {v: k for k, v in SOURCE_DISPLAY_KO.items()}
SOURCE_INFO = {
    "coffee": {"detergent": "water_based_detergent"},
    "teriyaki_sauce": {"detergent": "mixed_detergent"},
    "oriental_dressing": {"detergent": "oil_based_detergent"},
    "soy_sauce": {"detergent": "water_based_detergent"},
    "bbq_sauce": {"detergent": "mixed_detergent"},
    "gochujang": {"detergent": "mixed_detergent"},
    "ketchup": {"detergent": "water_based_detergent"},
    "milk": {"detergent": "water_based_detergent"},
    "mayonnaise": {"detergent": "oil_based_detergent"},
    "curry": {"detergent": "mixed_detergent"},
    "mustard": {"detergent": "water_based_detergent"},
    "oil": {"detergent": "mixed_detergent"},
}
DETERGENT_LABEL_KO = {
    "water_based_detergent": "수용성 세제",
    "mixed_detergent": "복합성 세제",
    "oil_based_detergent": "지용성 세제",
}
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
    "oil": {"GROUP_A": "medium", "GROUP_B": "medium", "GROUP_C": "low"},
}
INTENSITY_TO_BRUSH_MS = {"high": 1500, "medium": 1000, "low": 700}
INTENSITY_LABEL_KO = {"high": "강", "medium": "중", "low": "약"}
INTENSITY_DESCRIPTION_KO = {
    "high": "강한 브러싱으로 점성이 큰 얼룩을 분해합니다.",
    "medium": "일반적인 얼룩 제거에 적합한 표준 브러싱입니다.",
    "low": "섬유 손상을 줄이기 위한 약한 브러싱입니다.",
}
PUMP_MS_BY_SOURCE = {
    "coffee": 1000, "teriyaki_sauce": 1300, "oriental_dressing": 1200, "soy_sauce": 900,
    "bbq_sauce": 1300, "gochujang": 1400, "ketchup": 1000, "milk": 800,
    "mayonnaise": 1100, "curry": 1400, "mustard": 1000, "oil": 1400,
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
        "reason": "붉은색이 강한 점성 소스 자국이 양념치킨의 빨간 양념 소스 특징과 유사합니다.",
        "evidence": ["붉은색이 강한 소스 자국", "점성 있는 양념 형태", "양념치킨 소스와 유사한 색감"],
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

class AnalyzeTripletRequest(BaseModel):
    before_image: str
    after_image: str
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

class AnalyzeTripletResponse(BaseModel):
    final_label: str
    final_label_kr: str
    decision_source: str
    confidence: float
    cnn_before_class: str
    cnn_after_class: str
    cnn_before_confidence: float
    cnn_after_confidence: float
    material_group: Literal["GROUP_A", "GROUP_B", "GROUP_C"]
    material_group_display: str
    material_reason: str
    treatment_kr: str
    detergent_kr: str
    reason_kr: List[str]
    top_candidates: List[Dict[str, Any]]
    ocr_text_preview: List[str]
    ocr_debug_files: List[str]
    reaction_summary: Dict[str, float]


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
    percentages: Dict[str, int] = {}

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
        raise RuntimeError(f"tflite_http_error:{resp.status_code}:{detail}")

    data = resp.json()
    cnn_class = data.get("cnn_class")
    source = data.get("source") or CLASS_TO_SOURCE.get(cnn_class, cnn_class)
    if source not in SOURCE_INFO and source in KO_TO_SOURCE:
        source = KO_TO_SOURCE[source]
    confidence = max(0.0, min(1.0, float(data.get("confidence", 0.0))))

    top3_out = []
    for item in data.get("top3", []):
        cname = item.get("class_name")
        s = item.get("source") or CLASS_TO_SOURCE.get(cname or "", cname or "unknown")
        top3_out.append({
            "rank": int(item.get("rank", len(top3_out) + 1)),
            "class_name": cname,
            "source": s,
            "source_display": SOURCE_DISPLAY_KO.get(s, s),
            "confidence": float(item.get("confidence", item.get("score", 0.0))),
        })

    return {
        "cnn_class": cnn_class,
        "source": source,
        "source_display": SOURCE_DISPLAY_KO.get(source, source),
        "confidence": confidence,
        "top3": top3_out,
    }


def compute_reaction_summary(before_bgr: np.ndarray, after_bgr: np.ndarray) -> Dict[str, float]:
    b = cv2.resize(before_bgr, (224, 224))
    a = cv2.resize(after_bgr, (224, 224))
    diff = cv2.absdiff(b, a)
    hsv_b = cv2.cvtColor(b, cv2.COLOR_BGR2HSV)
    hsv_a = cv2.cvtColor(a, cv2.COLOR_BGR2HSV)
    gray_b = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    gray_a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    return {
        "rgb_diff_mean": float(np.mean(diff)),
        "hue_diff_mean": float(np.mean(cv2.absdiff(hsv_b[:, :, 0], hsv_a[:, :, 0]))),
        "sat_diff_mean": float(np.mean(cv2.absdiff(hsv_b[:, :, 1], hsv_a[:, :, 1]))),
        "val_diff_mean": float(np.mean(cv2.absdiff(hsv_b[:, :, 2], hsv_a[:, :, 2]))),
        "gray_diff_mean": float(np.mean(cv2.absdiff(gray_b, gray_a))),
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


def merge_top_candidates(before_data: Dict[str, Any], after_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    score_map: Dict[str, float] = {}

    def add_items(items: List[Dict[str, Any]], weight: float) -> None:
        for item in items:
            label = item["source"]
            score_map[label] = score_map.get(label, 0.0) + float(item.get("confidence", 0.0)) * weight

    score_map[before_data["source"]] = score_map.get(before_data["source"], 0.0) + float(before_data["confidence"]) * 1.0
    score_map[after_data["source"]] = score_map.get(after_data["source"], 0.0) + float(after_data["confidence"]) * 0.9
    add_items(before_data.get("top3", []), 0.30)
    add_items(after_data.get("top3", []), 0.25)

    ranked = sorted(score_map.items(), key=lambda x: x[1], reverse=True)
    total = sum(max(v, 0.0) for _, v in ranked) or 1.0
    return [
        {
            "label": label,
            "label_kr": SOURCE_DISPLAY_KO.get(label, label),
            "score": min(0.999, max(0.0, score / total)),
        }
        for label, score in ranked[:5]
    ]


def should_use_openai(before_data: Dict[str, Any], after_data: Dict[str, Any]) -> bool:
    before_label = before_data["source"]
    after_label = after_data["source"]
    before_conf = float(before_data["confidence"])
    after_conf = float(after_data["confidence"])

    if before_label == after_label:
        if max(before_conf, after_conf) >= CNN_DIRECT_AGREE_HIGH:
            return False
        if min(before_conf, after_conf) >= CNN_DIRECT_AGREE_LOW:
            return False
        return True

    return True


def choose_cnn_agreement(before_data: Dict[str, Any], after_data: Dict[str, Any], top_candidates: List[Dict[str, Any]]) -> Tuple[str, float, str]:
    before_label = before_data["source"]
    after_label = after_data["source"]
    before_conf = float(before_data["confidence"])
    after_conf = float(after_data["confidence"])

    if before_label == after_label:
        return before_label, max(before_conf, after_conf), "cnn_agreement"

    stronger = before_data if before_conf >= after_conf else after_data
    return stronger["source"], float(stronger["confidence"]), "cnn_fallback"


def build_pair_reasons(final_label: str, before_pred: Dict[str, Any], after_pred: Dict[str, Any], material_group: str, top_candidates: List[Dict[str, Any]], reaction: Dict[str, float], decision_source: str) -> List[str]:
    before_src = SOURCE_DISPLAY_KO.get(before_pred["source"], before_pred["source"])
    after_src = SOURCE_DISPLAY_KO.get(after_pred["source"], after_pred["source"])
    final_src = SOURCE_DISPLAY_KO.get(final_label, final_label)
    reasons = [
        f"1차 촬영 CNN 결과는 {before_src}이며 신뢰도는 {before_pred['confidence'] * 100:.1f}%입니다.",
        f"2차 촬영 CNN 결과는 {after_src}이며 신뢰도는 {after_pred['confidence'] * 100:.1f}%입니다.",
    ]
    if before_pred["source"] == after_pred["source"]:
        reasons.append(f"두 촬영 모두 같은 후보인 {final_src}를 가리켜 CNN 일치 결과로 판단했습니다.")
    else:
        reasons.append(f"두 촬영의 CNN 상위 후보가 서로 달라 최종 판단 단계에서 두 결과를 함께 비교했습니다.")
    reasons.append(f"태그 OCR로 확인한 원단 그룹은 {GROUP_DISPLAY[material_group]}이며 처리 강도 판단에 반영되었습니다.")
    reasons.append(f"물 반응 비교에서 RGB 변화 {reaction['rgb_diff_mean']:.1f}, 채도 변화 {reaction['sat_diff_mean']:.1f}, 명도 변화 {reaction['val_diff_mean']:.1f}가 확인되었습니다.")
    return reasons[:5]


def normalize_openai_output(data: Dict[str, Any], fallback_label: str, fallback_conf: float, fallback_reasons: List[str], treatment_summary: str) -> Tuple[str, float, List[str], str]:
    final_label = str(data.get("final_label", fallback_label)).strip()
    if final_label not in SOURCE_INFO:
        final_label = fallback_label
    conf = max(0.0, min(1.0, float(data.get("confidence", fallback_conf))))
    reasons = data.get("reason_kr")
    if not isinstance(reasons, list) or len(reasons) < 5:
        reasons = fallback_reasons
    reasons = [str(x).strip() for x in reasons[:5]]
    reasons = [r if r.endswith("입니다.") else (r.rstrip(".") + "입니다.") for r in reasons]
    treatment_kr = str(data.get("treatment_kr", treatment_summary)).strip() or treatment_summary
    if not treatment_kr.endswith("입니다."):
        treatment_kr = treatment_kr.rstrip(".") + "입니다."
    return final_label, conf, reasons, treatment_kr


def try_openai_final_decision(
    before_bgr: np.ndarray,
    after_bgr: np.ndarray,
    tag_bgr: np.ndarray,
    tag_texts: List[str],
    material_group: str,
    material_reason: str,
    reaction: Dict[str, float],
    before_pred: Dict[str, Any],
    after_pred: Dict[str, Any],
    top_candidates: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key or OpenAI is None:
        return None
    client = OpenAI(api_key=api_key)
    prompt = f"""
You are the final decision system for a smart stain remover.
The user captured three images of the same clothing item:
1) stain before water
2) stain after water
3) care tag

Important label note:
- bbq_sauce means Korean seasoned fried chicken sauce with a strong red color.
- Do NOT interpret bbq_sauce as brown Western barbecue sauce unless the image really looks brown.

Fabric group: {material_group}
Fabric reason: {material_reason}
OCR tag texts: {json.dumps(tag_texts[:20], ensure_ascii=False)}
Reaction summary: {json.dumps(reaction, ensure_ascii=False)}
Before CNN result: {json.dumps(before_pred, ensure_ascii=False)}
After CNN result: {json.dumps(after_pred, ensure_ascii=False)}
Merged candidate view: {json.dumps(top_candidates, ensure_ascii=False)}

Decision rules:
- If before and after strongly agree on the same source, keep that source.
- If before and after disagree, compare both images and both CNN outputs, then choose the single most likely final source.
- Prefer one of the provided candidate labels.
- Return STRICT JSON only.
- Write exactly 5 Korean reasons.
- treatment_kr must be Korean.
- final_label must stay in English source format.


"""
    try:
        resp = client.responses.create(
            model=OPENAI_MODEL,
            input=[{
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": image_to_data_url(before_bgr, quality=75)},
                    {"type": "input_image", "image_url": image_to_data_url(after_bgr, quality=75)},
                    {"type": "input_image", "image_url": image_to_data_url(tag_bgr, quality=75)},
                ],
            }],
            temperature=0.0,
        )
        text = getattr(resp, "output_text", "") or ""
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end == -1:
            return None
        return json.loads(text[start:end + 1])
    except Exception:
        return None


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


def run_motor(pump_ms: int, brush_ms: int) -> None:
    time.sleep((pump_ms + brush_ms) / 1000.0)


class CameraManager:
    def __init__(self):
        self.cap = None
        self.running = False
        self.thread = None
        self.lock = threading.Lock()
        self.latest_raw_frame = None

    def start(self) -> None:
        if self.running:
            return
        self.cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)
        time.sleep(0.3)
        if not self.cap.isOpened():
            self.cap.release()
            self.cap = cv2.VideoCapture(CAMERA_INDEX)
        if not self.cap or not self.cap.isOpened():
            raise RuntimeError("camera_open_failed")
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self) -> None:
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

camera_manager = CameraManager()
app = FastAPI(title="Smart Stain Cleaner API - Triplet V3 Refined")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])


@app.get("/", response_class=HTMLResponse)
def serve_ui():
    ui_path = BASE_DIR / "stain_controller_ui_triplet.html"
    if not ui_path.exists():
        raise HTTPException(status_code=500, detail=f"ui_file_missing:{ui_path}")
    return ui_path.read_text(encoding="utf-8")


@app.get("/api/health")
def health_check():
    try:
        r = requests.get(f"{TFLITE_SERVER_URL}/health", timeout=5)
        detail = r.json() if r.ok else r.text
        tflite_ok = r.ok
    except Exception as e:
        detail = str(e)
        tflite_ok = False
    return {
        "ok": True,
        "openai_model": OPENAI_MODEL,
        "preview_flip_horizontal": PREVIEW_FLIP_HORIZONTAL,
        "preview_flip_vertical": PREVIEW_FLIP_VERTICAL,
        "unflip_tag_before_ocr": UNFLIP_TAG_BEFORE_OCR,
        "save_ocr_debug": SAVE_OCR_DEBUG,
        "tflite_server_url": TFLITE_SERVER_URL,
        "tflite_ok": tflite_ok,
        "tflite_detail": detail,
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
        return StreamingResponse(camera_manager.mjpeg_generator(), media_type="multipart/x-mixed-replace; boundary=frame")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"video_feed_error:{repr(e)}")


@app.post("/api/analyze-triplet", response_model=AnalyzeTripletResponse)
def analyze_triplet(payload: AnalyzeTripletRequest):
    try:
        before_img_raw = decode_data_url_to_bgr(payload.before_image)
        after_img_raw = decode_data_url_to_bgr(payload.after_image)
        tag_img = restore_tag_orientation(decode_data_url_to_bgr(payload.tag_image))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"image_parse_error:{repr(e)}")

    try:
        # 추론 정확도를 위해 이전에 더 잘 맞았던 방식처럼 원본 캡처 이미지를 그대로 추론에 사용합니다.
        before_pred = call_tflite_server(payload.before_image)
        after_pred = call_tflite_server(payload.after_image)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"tflite_error:{repr(e)}")

    try:
        ocr_texts, debug_files = ocr_material_texts(tag_img)
        material_group, material_reason = parse_material_group(ocr_texts)        
        reaction = compute_reaction_summary(before_img_raw, after_img_raw)

        top_candidates = merge_top_candidates(before_pred, after_pred)
        final_label, pair_conf, decision_source = choose_cnn_agreement(before_pred, after_pred, top_candidates)
        fallback_reasons = build_pair_reasons(final_label, before_pred, after_pred, material_group, top_candidates, reaction, decision_source)

        treatment = build_treatment(final_label, material_group)
        treatment_kr = treatment.summary if treatment.summary.endswith("입니다.") else treatment.summary + "입니다."

        use_openai = should_use_openai(before_pred, after_pred)
        if use_openai:
            openai_data = try_openai_final_decision(
                before_img_raw,
                after_img_raw,
                tag_img,
                ocr_texts,
                material_group,
                material_reason,
                reaction,
                before_pred,
                after_pred,
                top_candidates,
            )
            if openai_data:
                final_label, pair_conf, fallback_reasons, treatment_kr = normalize_openai_output(
                    openai_data, final_label, max(float(before_pred["confidence"]), float(after_pred["confidence"])), fallback_reasons, treatment.summary
                )
                decision_source = "openai_final"
                treatment = build_treatment(final_label, material_group)
            else:
                decision_source = "cnn_fallback_after_openai_skip"

        return AnalyzeTripletResponse(
            final_label=final_label,
            final_label_kr=SOURCE_DISPLAY_KO.get(final_label, final_label),
            decision_source=decision_source,
            confidence=pair_conf,
            cnn_before_class=before_pred["cnn_class"],
            cnn_after_class=after_pred["cnn_class"],
            cnn_before_confidence=float(before_pred["confidence"]),
            cnn_after_confidence=float(after_pred["confidence"]),
            material_group=material_group,
            material_group_display=GROUP_DISPLAY[material_group],
            material_reason=material_reason,
            treatment_kr=treatment_kr,
            detergent_kr=DETERGENT_LABEL_KO.get(SOURCE_INFO[final_label]["detergent"], SOURCE_INFO[final_label]["detergent"]),
            reason_kr=fallback_reasons,
            top_candidates=top_candidates,
            ocr_text_preview=ocr_texts[:12],
            ocr_debug_files=debug_files,
            reaction_summary=reaction,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"analyze_triplet_error:{repr(e)}")


@app.post("/api/execute")
def execute(payload: ExecuteRequest):
    try:
        run_motor(payload.pump_ms, payload.brush_ms)
        return {"ok": True, "pump_ms": payload.pump_ms, "brush_ms": payload.brush_ms, "brush_intensity": payload.brush_intensity}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"execute_error:{repr(e)}")
