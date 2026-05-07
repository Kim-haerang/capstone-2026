# -*- coding: utf-8 -*-
import os
import re
import sys
import time
import json
import base64
import threading
import atexit
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
    import serial
except Exception:
    serial = None

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

try:
    import RPi.GPIO as GPIO
except Exception:
    GPIO = None

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
# 아두이노: 핀 4=수용성, 5=복합성, 6=지용성, 7=수중모터
ARDUINO_PORT = os.environ.get("ARDUINO_PORT", "/dev/ttyACM0")
ARDUINO_BAUD = int(os.environ.get("ARDUINO_BAUD", "9600"))
ARDUINO_TIMEOUT = float(os.environ.get("ARDUINO_TIMEOUT", "2.5"))
ARDUINO_LONG_TIMEOUT = float(os.environ.get("ARDUINO_LONG_TIMEOUT", "15.0"))
ARDUINO_PUMP_MAP = {
    "water_based_detergent": 1,  # 핀 4
    "mixed_detergent":       2,  # 핀 5
    "oil_based_detergent":   3,  # 핀 6
    # 핀 7은 수중모터 — UM 명령으로 별도 제어
}
# 라즈베리파이 PWM 브러시
BRUSH_PWM_PIN  = int(os.environ.get("BRUSH_PWM_PIN",  "18"))
BRUSH_DIR_A    = int(os.environ.get("BRUSH_DIR_A",    "23"))
BRUSH_DIR_B    = int(os.environ.get("BRUSH_DIR_B",    "24"))
BRUSH_PWM_FREQ = int(os.environ.get("BRUSH_PWM_FREQ", "1000"))
BRUSH_DUTY = {
    "high":   int(os.environ.get("BRUSH_DUTY_HIGH",   "86")),
    "medium": int(os.environ.get("BRUSH_DUTY_MEDIUM", "63")),
    "low":    int(os.environ.get("BRUSH_DUTY_LOW",    "39")),
}
BRUSH_COUNT_BY_GROUP = {"GROUP_A": 8, "GROUP_B": 6, "GROUP_C": 4}
# 수중모터
MAX_UMOTOR_MS     = int(os.environ.get("MAX_UMOTOR_MS",    "10000"))
DEFAULT_UMOTOR_MS = int(os.environ.get("DEFAULT_UMOTOR_MS",  "3000"))
OCR_CONFIG = "--oem 3 --psm 6 -l kor+eng"
CAMERA_INDEX = int(os.environ.get("CAMERA_INDEX", "0"))
PREVIEW_FLIP_HORIZONTAL = True
PREVIEW_FLIP_VERTICAL = False
UNFLIP_TAG_BEFORE_OCR = True
SAVE_OCR_DEBUG = True
FEEDBACK_ROI_X1 = float(os.environ.get("FEEDBACK_ROI_X1", "0.22"))
FEEDBACK_ROI_Y1 = float(os.environ.get("FEEDBACK_ROI_Y1", "0.22"))
FEEDBACK_ROI_X2 = float(os.environ.get("FEEDBACK_ROI_X2", "0.78"))
FEEDBACK_ROI_Y2 = float(os.environ.get("FEEDBACK_ROI_Y2", "0.78"))
TFLITE_SERVER_URL = os.environ.get("TFLITE_SERVER_URL", "http://127.0.0.1:9000")
OPENAI_LOW_CONF_THR = float(os.environ.get("OPENAI_LOW_CONF_THR", "0.84"))
PAIR_STRONG_CONF_THR = float(os.environ.get("PAIR_STRONG_CONF_THR", "0.90"))
CNN_DIRECT_AGREE_HIGH = float(os.environ.get("CNN_DIRECT_AGREE_HIGH", "0.75"))
CNN_DIRECT_AGREE_LOW = float(os.environ.get("CNN_DIRECT_AGREE_LOW", "0.68"))

YELLOW_BROWN_RECHECK_ENABLED = os.environ.get("YELLOW_BROWN_RECHECK_ENABLED", "1").strip().lower() not in {"0", "false", "no"}
MUSTARD_BROWN_RECHECK_MAX_CONF = float(os.environ.get("MUSTARD_BROWN_RECHECK_MAX_CONF", "0.96"))
MUSTARD_BROWN_MIN_CAND_SCORE = float(os.environ.get("MUSTARD_BROWN_MIN_CAND_SCORE", "0.12"))
BROWN_SOURCE_SET = {"coffee", "teriyaki_sauce", "oriental_dressing", "soy_sauce"}
YELLOW_SOURCE_SET = {"mustard", "curry", "oil"}

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
GPIO_READY = False
GPIO_ERROR: Optional[str] = None
_BRUSH_PWM = None
_plan_lock = threading.Lock()
LAST_EXECUTION_PLAN: Dict[str, Any] = {}
ARDUINO_SERIAL = None
ARDUINO_ERROR: Optional[str] = None


def setup_brush_gpio() -> None:
    global GPIO_READY, GPIO_ERROR, _BRUSH_PWM
    if GPIO_READY:
        return
    if GPIO is None:
        GPIO_ERROR = "RPi.GPIO import failed."
        raise RuntimeError(GPIO_ERROR)
    try:
        GPIO.setwarnings(False)
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(BRUSH_PWM_PIN, GPIO.OUT)
        GPIO.setup(BRUSH_DIR_A,   GPIO.OUT)
        GPIO.setup(BRUSH_DIR_B,   GPIO.OUT)
        GPIO.output(BRUSH_DIR_A, GPIO.LOW)
        GPIO.output(BRUSH_DIR_B, GPIO.LOW)
        _BRUSH_PWM = GPIO.PWM(BRUSH_PWM_PIN, BRUSH_PWM_FREQ)
        _BRUSH_PWM.start(0)
        GPIO_READY = True
        GPIO_ERROR = None
    except Exception as e:
        GPIO_ERROR = repr(e)
        raise


def cleanup_brush_gpio() -> None:
    global GPIO_READY, _BRUSH_PWM
    try:
        if _BRUSH_PWM is not None:
            _BRUSH_PWM.ChangeDutyCycle(0)
            _BRUSH_PWM.stop()
        if GPIO is not None and GPIO_READY:
            GPIO.output(BRUSH_DIR_A, GPIO.LOW)
            GPIO.output(BRUSH_DIR_B, GPIO.LOW)
            GPIO.cleanup([BRUSH_PWM_PIN, BRUSH_DIR_A, BRUSH_DIR_B])
    except Exception:
        pass
    finally:
        GPIO_READY = False
        _BRUSH_PWM = None


def get_arduino_serial():
    global ARDUINO_SERIAL, ARDUINO_ERROR

    if serial is None:
        ARDUINO_ERROR = "pyserial import failed. Install pyserial first."
        raise RuntimeError(ARDUINO_ERROR)

    try:
        if ARDUINO_SERIAL is None or not ARDUINO_SERIAL.is_open:
            ARDUINO_SERIAL = serial.Serial(
                ARDUINO_PORT,
                ARDUINO_BAUD,
                timeout=ARDUINO_TIMEOUT,
                write_timeout=ARDUINO_TIMEOUT,
            )
            time.sleep(2.0)
            try:
                ARDUINO_SERIAL.reset_input_buffer()
                ARDUINO_SERIAL.reset_output_buffer()
            except Exception:
                pass

        ARDUINO_ERROR = None
        return ARDUINO_SERIAL
    except Exception as e:
        ARDUINO_ERROR = repr(e)
        raise RuntimeError(f"arduino_serial_open_failed:{repr(e)}")


def close_arduino_serial() -> None:
    global ARDUINO_SERIAL
    try:
        if ARDUINO_SERIAL is not None and ARDUINO_SERIAL.is_open:
            ARDUINO_SERIAL.close()
    except Exception:
        pass
    finally:
        ARDUINO_SERIAL = None


def arduino_command(command: str, wait_done: bool = False) -> List[str]:
    ser = get_arduino_serial()

    try:
        ser.reset_input_buffer()
    except Exception:
        pass

    ser.write((command.strip() + "\n").encode("utf-8"))
    ser.flush()

    responses: List[str] = []
    deadline = time.time() + max(ARDUINO_TIMEOUT, 2.0)

    while time.time() < deadline:
        raw = ser.readline()
        if not raw:
            continue

        line = raw.decode("utf-8", errors="ignore").strip()
        if not line:
            continue

        responses.append(line)

        if line == "READY":
            continue

        if not wait_done:
            break

        if line == "DONE":
            break

    return responses


def setup_arduino() -> None:
    responses = arduino_command("PING", wait_done=False)
    if not responses or "PONG" not in responses[-1]:
        raise RuntimeError(f"arduino_ping_failed:{responses}")


def all_pumps_off() -> None:
    responses = arduino_command("ALL_OFF", wait_done=False)
    if not responses:
        raise RuntimeError("arduino_all_off_no_response")
    if not any(line.startswith("OK:ALL_OFF") or line.startswith("OK:ZERO_TIME_OFF") for line in responses):
        raise RuntimeError(f"arduino_all_off_failed:{responses}")


def run_pump(detergent_key: str, pump_ms: int) -> None:
    if detergent_key not in ARDUINO_PUMP_MAP:
        raise RuntimeError(f"unknown_detergent_key:{detergent_key}")

    pump_num = ARDUINO_PUMP_MAP[detergent_key]
    pump_ms = clamp_int(pump_ms, 0, MAX_PUMP_MS)

    if pump_ms <= 0:
        all_pumps_off()
        return

    cmd = f"P{pump_num},{pump_ms}"
    responses = arduino_command(cmd, wait_done=True)

    if not responses:
        raise RuntimeError("arduino_pump_no_response")

    ok_seen = any(line.startswith(f"OK:P{pump_num},") for line in responses)
    done_seen = any(line == "DONE" for line in responses)

    if not ok_seen:
        raise RuntimeError(f"arduino_pump_start_failed:{responses}")

    if not done_seen:
        raise RuntimeError(f"arduino_pump_done_failed:{responses}")


def run_brush_pwm(brush_count: int, brush_intensity: str = "medium") -> None:
    setup_brush_gpio()
    if _BRUSH_PWM is None:
        raise RuntimeError("brush_pwm_not_initialized")
    duty   = max(0, min(100, BRUSH_DUTY.get(brush_intensity, BRUSH_DUTY["medium"])))
    dur_ms = max(200, min(MAX_BRUSH_MS, brush_count * 500))
    try:
        GPIO.output(BRUSH_DIR_A, GPIO.HIGH)
        GPIO.output(BRUSH_DIR_B, GPIO.LOW)
        _BRUSH_PWM.ChangeDutyCycle(duty)
        time.sleep(dur_ms / 1000.0)
    finally:
        _BRUSH_PWM.ChangeDutyCycle(0)
        GPIO.output(BRUSH_DIR_A, GPIO.LOW)
        GPIO.output(BRUSH_DIR_B, GPIO.LOW)


def run_underwater_motor(duration_ms: int) -> None:
    duration_ms = max(0, min(MAX_UMOTOR_MS, int(duration_ms)))
    if duration_ms <= 0:
        arduino_command("UM_OFF", wait_done=False)
        return
    ser = get_arduino_serial()
    cmd = f"UM,{duration_ms}"
    try:
        ser.reset_input_buffer()
    except Exception:
        pass
    ser.write((cmd + "\n").encode("utf-8"))
    ser.flush()
    responses: List[str] = []
    deadline = time.time() + (duration_ms / 1000.0) + ARDUINO_LONG_TIMEOUT
    while time.time() < deadline:
        try:
            raw = ser.readline()
        except Exception:
            break
        if not raw:
            continue
        line = raw.decode("utf-8", errors="ignore").strip()
        if not line:
            continue
        responses.append(line)
        if line == "DONE":
            break
    if not any(l.startswith("OK:UM") for l in responses):
        raise RuntimeError(f"arduino_umotor_start_failed:{responses}")
    if not any(l == "DONE" for l in responses):
        raise RuntimeError(f"arduino_umotor_done_failed:{responses}")


atexit.register(cleanup_brush_gpio)
atexit.register(close_arduino_serial)
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
    tag_image: Optional[str] = None
    manual_material_group: Optional[Literal["GROUP_A", "GROUP_B", "GROUP_C"]] = None

class ExecuteRequest(BaseModel):
    pump_ms: int = Field(ge=0, le=MAX_PUMP_MS)
    brush_intensity: Literal["high", "medium", "low"] = "medium"
    detergent_key: Optional[str] = None

class CleanupFeedbackRequest(BaseModel):
    baseline_image: str
    cleaned_image: str
    stain_label: str
    stain_label_kr: Optional[str] = None
    material_group: Optional[Literal["GROUP_A", "GROUP_B", "GROUP_C"]] = None

class Treatment(BaseModel):
    brush_intensity: Literal["high", "medium", "low"]
    pump_ms: int
    brush_count: int
    detergent_key: str
    pump_pin: int
    summary: str

class AnalyzeTripletResponse(BaseModel):
    final_label: str
    final_label_kr: str
    decision_source: str
    material_input_mode: str
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

class CleanupFeedbackResponse(BaseModel):
    removal_percent: float
    status: Literal["good", "partial", "poor"]
    status_kr: str
    recommendation_kr: str
    comment_kr: str
    before_confidence: float
    cleaned_confidence: float
    confidence_drop: float
    metrics: Dict[str, float]

class UnderwaterMotorRequest(BaseModel):
    duration_ms: int = Field(default=DEFAULT_UMOTOR_MS, ge=0, le=MAX_UMOTOR_MS)

class UnderwaterMotorResponse(BaseModel):
    ok: bool
    duration_ms: int
    message: str


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


def detect_material_group_from_ocr(ocr_texts: List[str]) -> Tuple[Optional[str], str]:
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
        return None, "택 OCR에서 재질을 읽지 못했습니다."

    for group, keywords in MATERIAL_KEYWORDS.items():
        normalized = [SYNONYM_MAP.get(k.lower(), k.lower()) for k in keywords]
        if dominant in normalized or dominant in [k.lower() for k in keywords]:
            return group, reason

    return None, "재질 그룹 매핑에 실패했습니다."


def parse_material_group(ocr_texts: List[str]) -> Tuple[str, str]:
    material_group, reason = detect_material_group_from_ocr(ocr_texts)
    if material_group:
        return material_group, reason
    return "GROUP_A", "택 OCR에서 재질을 읽지 못해 기본값 GROUP_A로 처리했습니다."


def resolve_material_input(tag_image: Optional[str], manual_material_group: Optional[str]) -> Tuple[Optional[np.ndarray], List[str], List[str], str, str, str]:
    if manual_material_group:
        if manual_material_group not in GROUP_DISPLAY:
            raise HTTPException(status_code=400, detail="invalid_manual_material_group")
        return None, [], [], manual_material_group, f"사용자가 화면에서 {GROUP_DISPLAY[manual_material_group]}을(를) 직접 선택했습니다.", "manual_select"

    if not tag_image:
        raise HTTPException(status_code=400, detail="material_input_required:tag_or_manual")

    try:
        tag_img = restore_tag_orientation(decode_data_url_to_bgr(tag_image))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"tag_image_parse_error:{repr(e)}")

    ocr_texts, debug_files = ocr_material_texts(tag_img)
    material_group, material_reason = detect_material_group_from_ocr(ocr_texts)
    if not material_group:
        raise HTTPException(status_code=400, detail="tag_ocr_material_not_detected:manual_selection_required")
    return tag_img, ocr_texts, debug_files, material_group, material_reason, "tag_ocr"


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


def _flatten_candidate_sources(*items: Dict[str, Any]) -> Dict[str, float]:
    score_map: Dict[str, float] = {}
    for data in items:
        if not data:
            continue
        src = str(data.get("source", "")).strip()
        if src:
            score_map[src] = max(score_map.get(src, 0.0), float(data.get("confidence", 0.0)))
        for cand in data.get("top3", []) or []:
            label = str(cand.get("source", "")).strip()
            if label:
                score_map[label] = max(score_map.get(label, 0.0), float(cand.get("confidence", 0.0)))
    return score_map


def should_run_mustard_brown_recheck(before_data: Dict[str, Any], after_data: Dict[str, Any], top_candidates: List[Dict[str, Any]]) -> bool:
    if not YELLOW_BROWN_RECHECK_ENABLED:
        return False

    score_map = _flatten_candidate_sources(before_data, after_data)
    for cand in top_candidates[:5]:
        label = str(cand.get("label", "")).strip()
        if label:
            score_map[label] = max(score_map.get(label, 0.0), float(cand.get("score", 0.0)))

    mustard_score = score_map.get("mustard", 0.0)
    has_brown = any(score_map.get(label, 0.0) >= MUSTARD_BROWN_MIN_CAND_SCORE for label in BROWN_SOURCE_SET)
    has_yellow = any(score_map.get(label, 0.0) >= MUSTARD_BROWN_MIN_CAND_SCORE for label in YELLOW_SOURCE_SET)
    if mustard_score < MUSTARD_BROWN_MIN_CAND_SCORE or not has_brown or not has_yellow:
        return False

    before_label = str(before_data.get("source", "")).strip()
    after_label = str(after_data.get("source", "")).strip()
    max_conf = max(float(before_data.get("confidence", 0.0)), float(after_data.get("confidence", 0.0)))

    if before_label == after_label and max_conf >= MUSTARD_BROWN_RECHECK_MAX_CONF:
        return False

    return True


def extract_mustard_focus_crop(image_bgr: np.ndarray) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    if h == 0 or w == 0:
        return image_bgr

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    yellow_mask = cv2.inRange(hsv, (12, 35, 40), (45, 255, 255))
    brown_mask = cv2.inRange(hsv, (5, 25, 20), (25, 255, 230))
    sat_mask = cv2.inRange(hsv[:, :, 1], 45, 255)
    mask = cv2.bitwise_or(yellow_mask, brown_mask)
    mask = cv2.bitwise_and(mask, sat_mask)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    coords = cv2.findNonZero(mask)
    if coords is None or len(coords) < 40:
        return image_bgr

    x, y, bw, bh = cv2.boundingRect(coords)
    pad_x = max(12, int(bw * 0.22))
    pad_y = max(12, int(bh * 0.22))
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(w, x + bw + pad_x)
    y2 = min(h, y + bh + pad_y)
    crop = image_bgr[y1:y2, x1:x2]
    return crop if crop.size else image_bgr


def build_mustard_focus_image(image_bgr: np.ndarray) -> np.ndarray:
    crop = extract_mustard_focus_crop(image_bgr)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.18 + 6.0, 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.04, 0, 255)
    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    out = cv2.convertScaleAbs(out, alpha=1.03, beta=2)
    return out


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
    reasons.append(f"재질 정보로 확인한 원단 그룹은 {GROUP_DISPLAY[material_group]}이며 처리 강도 판단에 반영되었습니다.")
    reasons.append(f"물 반응 비교에서 RGB 변화 {reaction['rgb_diff_mean']:.1f}, 채도 변화 {reaction['sat_diff_mean']:.1f}, 명도 변화 {reaction['val_diff_mean']:.1f}가 확인되었습니다.")
    if should_run_mustard_brown_recheck(before_pred, after_pred, top_candidates):
        reasons[-1] = "머스타드와 갈색 계열 후보가 함께 보여 보수적인 재검토 조건이 활성화되었습니다."
    return reasons[:5]


def normalize_openai_output(data: Dict[str, Any], fallback_label: str, fallback_conf: float, fallback_reasons: List[str], treatment_summary: str) -> Tuple[str, float, List[str], str]:
    final_label = str(data.get("final_label", fallback_label)).strip()
    if final_label not in SOURCE_INFO:
        final_label = fallback_label
    conf = max(0.0, min(1.0, float(data.get("confidence", fallback_conf))))
    reasons = data.get("reason_kr")
    if not isinstance(reasons, list) or len(reasons) == 0:
        reasons = fallback_reasons
    reasons = [str(x).strip() for x in reasons[:5]]
    reasons = [r if r else (r.rstrip(".") ) for r in reasons]
    treatment_kr = str(data.get("treatment_kr", treatment_summary)).strip() or treatment_summary
    if not treatment_kr:
        treatment_kr = treatment_kr.rstrip(".")
    return final_label, conf, reasons, treatment_kr


def try_openai_final_decision(
    before_bgr: np.ndarray,
    after_bgr: np.ndarray,
    tag_bgr: Optional[np.ndarray],
    tag_texts: List[str],
    material_group: str,
    material_reason: str,
    reaction: Dict[str, float],
    before_pred: Dict[str, Any],
    after_pred: Dict[str, Any],
    top_candidates: List[Dict[str, Any]],
    review_mode: str = "default",
) -> Optional[Dict[str, Any]]:
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key or OpenAI is None:
        return None
    client = OpenAI(api_key=api_key)
    extra_rule = ""
    if review_mode == "mustard_brown_recheck":
        extra_rule = (
            "\nSpecial review mode: mustard-vs-brown confusion was detected.\n"
            "- Additional cropped review images are provided for color-focused inspection.\n"
            "- Override a brown CNN result only when the stain truly shows clear mustard-like bright yellow evidence in the original images and focus crops.\n"
            "- Do NOT force mustard when the stain mainly looks dark brown, soy-like, coffee-like, or teriyaki-like.\n"
            "- If yellow is weak or uncertain, prefer the safer non-mustard candidate already supported by the CNN candidates.\n"
        )

    prompt = f"""
You are the final decision system for a smart stain remover.
The user captured stain images of the same clothing item.
Available inputs:
1) stain before water
2) stain after water
3) optional care tag image or a manually selected fabric group

Important label note:
- bbq_sauce means Korean seasoned fried chicken sauce with a strong red color.
- Do NOT interpret bbq_sauce as brown Western barbecue sauce unless the image really looks brown.
{extra_rule}
Fabric group: {material_group}
Fabric reason: {material_reason}
OCR tag texts: {json.dumps(tag_texts[:20], ensure_ascii=False)}
Reaction summary: {json.dumps(reaction, ensure_ascii=False)}
Before CNN result: {json.dumps(before_pred, ensure_ascii=False)}
After CNN result: {json.dumps(after_pred, ensure_ascii=False)}
Merged candidate view: {json.dumps(top_candidates, ensure_ascii=False)}

Decision rules:
- If before and after strongly agree on the same source, keep that source unless the special review mode explicitly requires a second look.
- If before and after disagree, compare both images and both CNN outputs, then choose the single most likely final source.
- Prefer one of the provided candidate labels.
- Return STRICT JSON only.
- Write exactly 5 Korean reasons
- treatment_kr must be Korean
- final_label must stay in English source format.

JSON schema:
{{
  "final_label": "coffee",
  "confidence": 0.84,
  "reason_kr": ["이유1입니다.", "이유2입니다.", "이유3입니다.", "이유4입니다.", "이유5입니다."],
  "treatment_kr": "수용성 세제로 처리합니다."
}}
"""
    try:
        content = [
            {"type": "input_text", "text": prompt},
            {"type": "input_image", "image_url": image_to_data_url(before_bgr, quality=75)},
            {"type": "input_image", "image_url": image_to_data_url(after_bgr, quality=75)},
        ]
        if tag_bgr is not None:
            content.append({"type": "input_image", "image_url": image_to_data_url(tag_bgr, quality=75)})
        if review_mode == "mustard_brown_recheck":
            content.append({"type": "input_image", "image_url": image_to_data_url(build_mustard_focus_image(before_bgr), quality=85)})
            content.append({"type": "input_image", "image_url": image_to_data_url(build_mustard_focus_image(after_bgr), quality=85)})

        resp = client.responses.create(
            model=OPENAI_MODEL,
            input=[{
                "role": "user",
                "content": content,
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
    pump_ms = PUMP_MS_BY_SOURCE.get(source, 1000)
    brush_count = BRUSH_COUNT_BY_GROUP.get(material_group, 6)
    detergent_key = SOURCE_INFO[source]["detergent"]
    pump_num = ARDUINO_PUMP_MAP[detergent_key]
    detergent_ko = DETERGENT_LABEL_KO.get(detergent_key, detergent_key)
    intensity_ko = INTENSITY_LABEL_KO.get(intensity, intensity)
    summary = (
        f"{detergent_ko} {pump_ms}ms → 브러시 {brush_count}회 왕복 순서로 작동합니다. "
        f"브러시 강도는 {intensity_ko}이며, {INTENSITY_DESCRIPTION_KO[intensity]}"
    )
    return Treatment(
        brush_intensity=intensity,
        pump_ms=pump_ms,
        brush_count=brush_count,
        detergent_key=detergent_key,
        pump_pin=pump_num,
        summary=summary,
    )


def run_motor(detergent_key: str, pump_ms: int, brush_count: int, brush_intensity: str) -> None:
    run_pump(detergent_key, pump_ms)
    run_brush_pwm(brush_count, brush_intensity)


def _cleanup_family_mask(image_bgr: np.ndarray, stain_label: str) -> np.ndarray:
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    label = str(stain_label or "").strip()
    if label in {"coffee", "teriyaki_sauce", "oriental_dressing", "soy_sauce"}:
        mask = cv2.inRange(hsv, (5, 20, 20), (28, 255, 230))
    elif label in {"bbq_sauce", "gochujang", "ketchup"}:
        m1 = cv2.inRange(hsv, (0, 45, 30), (12, 255, 255))
        m2 = cv2.inRange(hsv, (165, 45, 30), (179, 255, 255))
        mask = cv2.bitwise_or(m1, m2)
    elif label in {"curry", "mustard", "oil"}:
        mask = cv2.inRange(hsv, (12, 35, 35), (45, 255, 255))
    elif label in {"milk", "mayonnaise"}:
        mask = cv2.inRange(hsv, (0, 0, 120), (179, 80, 255))
    else:
        sat_mask = cv2.inRange(hsv[:, :, 1], 25, 255)
        val_mask = cv2.inRange(hsv[:, :, 2], 25, 245)
        mask = cv2.bitwise_and(sat_mask, val_mask)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return mask


def _safe_pct(delta: float, baseline: float, scale: float = 1.0) -> float:
    if baseline <= 1e-6:
        return 0.0
    return max(0.0, min(100.0, (delta / baseline) * 100.0 * scale))


def _extract_feedback_roi(image_bgr: np.ndarray) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    if h <= 0 or w <= 0:
        return image_bgr
    x1 = max(0, min(w - 1, int(w * FEEDBACK_ROI_X1)))
    y1 = max(0, min(h - 1, int(h * FEEDBACK_ROI_Y1)))
    x2 = max(x1 + 1, min(w, int(w * FEEDBACK_ROI_X2)))
    y2 = max(y1 + 1, min(h, int(h * FEEDBACK_ROI_Y2)))
    roi = image_bgr[y1:y2, x1:x2]
    return roi if roi.size else image_bgr


def _masked_mean(channel: np.ndarray, mask: np.ndarray) -> float:
    nz = float(cv2.countNonZero(mask))
    if nz <= 0:
        return 0.0
    return float(cv2.mean(channel, mask=mask)[0])


def compute_cleanup_feedback_metrics(baseline_bgr: np.ndarray, cleaned_bgr: np.ndarray, stain_label: str) -> Dict[str, float]:
    roi_b = cv2.resize(_extract_feedback_roi(baseline_bgr), (224, 224))
    roi_c = cv2.resize(_extract_feedback_roi(cleaned_bgr), (224, 224))
    reaction = compute_reaction_summary(roi_b, roi_c)

    baseline_mask = _cleanup_family_mask(roi_b, stain_label)
    cleaned_mask = _cleanup_family_mask(roi_c, stain_label)
    baseline_area = float(cv2.countNonZero(baseline_mask))
    cleaned_area = float(cv2.countNonZero(cleaned_mask))
    area_reduction_pct = _safe_pct(baseline_area - cleaned_area, baseline_area)

    hsv_b = cv2.cvtColor(roi_b, cv2.COLOR_BGR2HSV)
    hsv_c = cv2.cvtColor(roi_c, cv2.COLOR_BGR2HSV)
    sat_mean_baseline = _masked_mean(hsv_b[:, :, 1], baseline_mask)
    sat_mean_cleaned = _masked_mean(hsv_c[:, :, 1], cleaned_mask) if cleaned_area > 0 else 0.0
    val_mean_baseline = _masked_mean(hsv_b[:, :, 2], baseline_mask)
    val_mean_cleaned = _masked_mean(hsv_c[:, :, 2], cleaned_mask) if cleaned_area > 0 else 0.0

    sat_reduction_pct = _safe_pct(sat_mean_baseline - sat_mean_cleaned, max(sat_mean_baseline, 1.0))
    val_shift_pct = min(100.0, abs(val_mean_baseline - val_mean_cleaned) / 80.0 * 100.0)
    hsv_change_pct = min(100.0, ((reaction["sat_diff_mean"] * 0.65) + (reaction["val_diff_mean"] * 0.35)) / 40.0 * 100.0)
    gray_change_pct = min(100.0, reaction["gray_diff_mean"] / 30.0 * 100.0)

    if baseline_area < 60:
        area_reduction_pct = hsv_change_pct * 0.55 + gray_change_pct * 0.45

    removal_percent = (
        area_reduction_pct * 0.50
        + sat_reduction_pct * 0.20
        + hsv_change_pct * 0.20
        + gray_change_pct * 0.10
    )
    removal_percent = max(0.0, min(100.0, removal_percent))

    return {
        "roi_x1_ratio": float(FEEDBACK_ROI_X1),
        "roi_y1_ratio": float(FEEDBACK_ROI_Y1),
        "roi_x2_ratio": float(FEEDBACK_ROI_X2),
        "roi_y2_ratio": float(FEEDBACK_ROI_Y2),
        "rgb_diff_mean": float(reaction["rgb_diff_mean"]),
        "sat_diff_mean": float(reaction["sat_diff_mean"]),
        "val_diff_mean": float(reaction["val_diff_mean"]),
        "gray_diff_mean": float(reaction["gray_diff_mean"]),
        "baseline_mask_area": baseline_area,
        "cleaned_mask_area": cleaned_area,
        "area_reduction_pct": float(area_reduction_pct),
        "sat_mean_baseline": float(sat_mean_baseline),
        "sat_mean_cleaned": float(sat_mean_cleaned),
        "val_mean_baseline": float(val_mean_baseline),
        "val_mean_cleaned": float(val_mean_cleaned),
        "sat_reduction_pct": float(sat_reduction_pct),
        "val_shift_pct": float(val_shift_pct),
        "hsv_change_pct": float(hsv_change_pct),
        "gray_change_pct": float(gray_change_pct),
        "removal_percent": float(removal_percent),
    }


def build_cleanup_feedback_fallback(removal_percent: float) -> Tuple[str, str, str]:
    if removal_percent >= 70.0:
        return (
            "good",
            "대체로 잘 지워졌습니다.",
            "현재 상태면 세척을 여기서 마무리해도 좋습니다.",
        )
    if removal_percent >= 40.0:
        return (
            "partial",
            "일부는 지워졌지만 아직 얼룩이 남아 있습니다.",
            "같은 방식으로 한 번 더 세척하면 더 좋아질 가능성이 큽니다.",
        )
    return (
        "poor",
        "현재 기준으로는 제거 효과가 크지 않습니다.",
        "한 번 더 세척하거나 세척 강도를 다시 점검하는 것이 좋습니다.",
    )


def try_openai_cleanup_feedback(
    baseline_bgr: np.ndarray,
    cleaned_bgr: np.ndarray,
    stain_label: str,
    stain_label_kr: str,
    removal_percent: float,
    metrics: Dict[str, float],
) -> Optional[Dict[str, Any]]:
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key or OpenAI is None:
        return None
    client = OpenAI(api_key=api_key)
    prompt = f"""
You evaluate how well a stain was removed after cleaning.

Input facts:
- stain_label: {stain_label}
- stain_label_kr: {stain_label_kr}
- estimated_removal_percent: {removal_percent:.1f}
- metrics: {json.dumps(metrics, ensure_ascii=False)}

Rules:
- The final answer must be strict JSON only.
- Use Korean.
- Do not mention water temperature, recommended water temperature, detergent type changes, or any hardware details.
- Focus only on whether the stain looks sufficiently removed or whether one more cleaning cycle would help.
- recommendation_kr must be a short user-facing recommendation sentence
- comment_kr must be a short explanation sentence
- status must be one of: good, partial, poor.
- status_kr should be concise.

JSON schema:
{{
  "status": "partial",
  "status_kr": "일부 제거되었습니다.",
  "recommendation_kr": "같은 방식으로 한 번 더 세척하는 것이 좋습니다.",
  "comment_kr": "얼룩이 줄어들었지만 아직 남은 자국이 확인됩니다."
}}
"""
    try:
        resp = client.responses.create(
            model=OPENAI_MODEL,
            input=[{
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": image_to_data_url(baseline_bgr, quality=75)},
                    {"type": "input_image", "image_url": image_to_data_url(cleaned_bgr, quality=75)},
                ],
            }],
            temperature=0.0,
        )
        text = getattr(resp, "output_text", "") or ""
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end == -1:
            return None
        data = json.loads(text[start:end + 1])
        if str(data.get("status", "")).strip() not in {"good", "partial", "poor"}:
            return None
        return data
    except Exception:
        return None


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


@app.on_event("startup")
def on_startup():
    try:
        setup_arduino()
    except Exception as e:
        ARDUINO_ERROR = repr(e)
        print(f"[WARN] Arduino init failed: {e}", flush=True)


@app.on_event("shutdown")
def on_shutdown():
    cleanup_brush_gpio()
    close_arduino_serial()


@app.get("/", response_class=HTMLResponse)
def serve_ui():
    ui_path = BASE_DIR / "stain_fused_feedback_fixed.html"
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
        "brush_pwm_pin": BRUSH_PWM_PIN,
        "brush_dir_a": BRUSH_DIR_A,
        "brush_dir_b": BRUSH_DIR_B,
        "brush_duty": BRUSH_DUTY,
        "pump_map": ARDUINO_PUMP_MAP,
        "max_umotor_ms": MAX_UMOTOR_MS,
        "arduino_port": ARDUINO_PORT,
        "arduino_baud": ARDUINO_BAUD,
        "arduino_error": ARDUINO_ERROR,
        "gpio_ready": GPIO_READY,
        "gpio_error": GPIO_ERROR,
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
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"image_parse_error:{repr(e)}")

    try:
        tag_img, ocr_texts, debug_files, material_group, material_reason, material_input_mode = resolve_material_input(
            payload.tag_image, payload.manual_material_group
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"material_input_error:{repr(e)}")

    try:
        before_pred = call_tflite_server(payload.before_image)
        after_pred = call_tflite_server(payload.after_image)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"tflite_error:{repr(e)}")

    try:
        reaction = compute_reaction_summary(before_img_raw, after_img_raw)

        top_candidates = merge_top_candidates(before_pred, after_pred)
        final_label, pair_conf, decision_source = choose_cnn_agreement(before_pred, after_pred, top_candidates)
        fallback_reasons = build_pair_reasons(final_label, before_pred, after_pred, material_group, top_candidates, reaction, decision_source)

        treatment = build_treatment(final_label, material_group)
        with _plan_lock:
            LAST_EXECUTION_PLAN.clear()
            LAST_EXECUTION_PLAN.update({
                "final_label":    final_label,
                "detergent_key":  treatment.detergent_key,
                "pump_ms":        treatment.pump_ms,
                "brush_count":    treatment.brush_count,
                "brush_intensity": treatment.brush_intensity,
            })
        treatment_kr = treatment.summary if treatment.summary else treatment.summary

        mustard_brown_recheck = should_run_mustard_brown_recheck(before_pred, after_pred, top_candidates)
        use_openai = should_use_openai(before_pred, after_pred) or mustard_brown_recheck
        if use_openai:
            review_mode = "mustard_brown_recheck" if mustard_brown_recheck else "default"
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
                review_mode=review_mode,
            )
            if openai_data:
                final_label, pair_conf, fallback_reasons, treatment_kr = normalize_openai_output(
                    openai_data, final_label, max(float(before_pred["confidence"]), float(after_pred["confidence"])), fallback_reasons, treatment.summary
                )
                decision_source = "openai_final"
                treatment = build_treatment(final_label, material_group)
                with _plan_lock:
                    LAST_EXECUTION_PLAN.clear()
                    LAST_EXECUTION_PLAN.update({
                        "final_label":    final_label,
                        "detergent_key":  treatment.detergent_key,
                        "pump_ms":        treatment.pump_ms,
                        "brush_count":    treatment.brush_count,
                        "brush_intensity": treatment.brush_intensity,
                    })
            else:
                decision_source = "cnn_fallback_after_openai_skip"

        return AnalyzeTripletResponse(
            final_label=final_label,
            final_label_kr=SOURCE_DISPLAY_KO.get(final_label, final_label),
            decision_source=decision_source,
            material_input_mode=material_input_mode,
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
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"analyze_triplet_error:{repr(e)}")


@app.post("/api/execute")
def execute(payload: ExecuteRequest):
    try:
        with _plan_lock:
            detergent_key   = payload.detergent_key or LAST_EXECUTION_PLAN.get("detergent_key")
            pump_ms         = LAST_EXECUTION_PLAN.get("pump_ms", payload.pump_ms)
            brush_count     = LAST_EXECUTION_PLAN.get("brush_count", 6)
            brush_intensity = LAST_EXECUTION_PLAN.get("brush_intensity", payload.brush_intensity)
        if not detergent_key:
            raise HTTPException(status_code=400, detail="execute_error:no_detergent_key_available_run_analyze_first")
        run_motor(detergent_key, pump_ms, brush_count, brush_intensity)
        duty   = BRUSH_DUTY.get(brush_intensity, BRUSH_DUTY["medium"])
        dur_ms = max(200, min(MAX_BRUSH_MS, brush_count * 500))
        return {
            "ok": True,
            "sequence": [
                {"step": "detergent", "detergent_key": detergent_key, "pump_ms": pump_ms,
                 "pump_num": ARDUINO_PUMP_MAP.get(detergent_key)},
                {"step": "brush", "brush_count": brush_count, "brush_intensity": brush_intensity,
                 "brush_duty_pct": duty, "brush_duration_ms": dur_ms, "brush_pwm_pin": BRUSH_PWM_PIN},
            ],
            "pump_ms": pump_ms,
            "brush_count": brush_count,
            "brush_intensity": brush_intensity,
            "brush_duty_pct": duty,
            "detergent_key": detergent_key,
            "pump_num": ARDUINO_PUMP_MAP.get(detergent_key),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"execute_error:{repr(e)}")


@app.post("/api/cleanup-feedback", response_model=CleanupFeedbackResponse)
def cleanup_feedback(payload: CleanupFeedbackRequest):
    try:
        baseline_img = decode_data_url_to_bgr(payload.baseline_image)
        cleaned_img = decode_data_url_to_bgr(payload.cleaned_image)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"cleanup_image_parse_error:{repr(e)}")

    stain_label = str(payload.stain_label or "").strip()
    if stain_label not in SOURCE_INFO:
        raise HTTPException(status_code=400, detail=f"cleanup_invalid_stain_label:{stain_label}")

    try:
        metrics = compute_cleanup_feedback_metrics(
            baseline_img,
            cleaned_img,
            stain_label,
        )
        removal_percent = float(metrics["removal_percent"])
        status, status_kr, recommendation_kr = build_cleanup_feedback_fallback(removal_percent)
        stain_label_kr = payload.stain_label_kr or SOURCE_DISPLAY_KO.get(stain_label, stain_label)
        comment_kr = (
            f"1차 촬영 이미지를 기준으로 중앙 얼룩 영역의 HSV 변화와 색상 마스크 감소량을 비교한 제거율은 {removal_percent:.1f}%이며, 현재 상태는 {status_kr}로 판단됩니다."
        )
        openai_data = try_openai_cleanup_feedback(
            _extract_feedback_roi(baseline_img),
            _extract_feedback_roi(cleaned_img),
            stain_label,
            stain_label_kr,
            removal_percent,
            metrics,
        )
        if openai_data:
            status = str(openai_data.get("status", status)).strip()
            if status not in {"good", "partial", "poor"}:
                status = build_cleanup_feedback_fallback(removal_percent)[0]
            status_kr = str(openai_data.get("status_kr", status_kr)).strip() or status_kr
            recommendation_kr = str(openai_data.get("recommendation_kr", recommendation_kr)).strip() or recommendation_kr
            comment_kr = str(openai_data.get("comment_kr", comment_kr)).strip() or comment_kr
        if not status_kr:
            status_kr = status_kr.rstrip(".")
        if not recommendation_kr:
            recommendation_kr = recommendation_kr.rstrip(".") 
        if not comment_kr:
            comment_kr = comment_kr.rstrip(".") 

        return CleanupFeedbackResponse(
            removal_percent=removal_percent,
            status=status,
            status_kr=status_kr,
            recommendation_kr=recommendation_kr,
            comment_kr=comment_kr,
            before_confidence=0.0,
            cleaned_confidence=0.0,
            confidence_drop=0.0,
            metrics=metrics,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"cleanup_feedback_error:{repr(e)}")


from fastapi import Response

@app.get("/favicon.ico")
def favicon():
    return Response(status_code=204)


@app.get("/developer", response_class=HTMLResponse)
def serve_developer_ui():
    ui_path = BASE_DIR / "stain_fused_feedback_dev.html"
    if not ui_path.exists():
        raise HTTPException(status_code=500, detail=f"ui_file_missing:{ui_path}")
    return ui_path.read_text(encoding="utf-8")
