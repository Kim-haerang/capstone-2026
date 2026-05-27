# -*- coding: utf-8 -*-

import os
import re
import sys
import time
import json
import base64
import threading
import atexit
import uuid
from datetime import datetime
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
MAIN_UI_PATH = BASE_DIR / "stain_fused_feedback_app.html"
FEEDBACK_DATASET_DIR = BASE_DIR / "feedback_dataset"
FEEDBACK_IMAGE_DIR = FEEDBACK_DATASET_DIR / "images"
FEEDBACK_LABELS_PATH = FEEDBACK_DATASET_DIR / "labels.jsonl"
FEEDBACK_LABELS_FALLBACK_PATH = BASE_DIR / "labels.jsonl"
DEV_STATE_PATH = BASE_DIR / "dev_state_latest.json"
_learning_write_lock = threading.Lock()
_dev_state_lock = threading.Lock()
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4.1-mini")
JPEG_QUALITY = 85
MAX_PUMP_MS = 5000
MAX_BRUSH_MS = 5000
ARDUINO_PORT = os.environ.get("ARDUINO_PORT", "/dev/ttyACM0")
ARDUINO_BAUD = int(os.environ.get("ARDUINO_BAUD", "9600"))
ARDUINO_TIMEOUT = float(os.environ.get("ARDUINO_TIMEOUT", "2.5"))
ARDUINO_LONG_TIMEOUT = float(os.environ.get("ARDUINO_LONG_TIMEOUT", "15.0"))
ARDUINO_PUMP_MAP = {
    "water_based_detergent": 1,  # 핀 4
    "mixed_detergent":       2,  # 핀 5
    "oil_based_detergent":   3,  # 핀 6
}
BRUSH_PWM_PIN  = int(os.environ.get("BRUSH_PWM_PIN",  "18"))
BRUSH_DIR_A    = int(os.environ.get("BRUSH_DIR_A",    "23"))
BRUSH_DIR_B    = int(os.environ.get("BRUSH_DIR_B",    "24"))
BRUSH_PWM_FREQ = int(os.environ.get("BRUSH_PWM_FREQ", "50"))

SERVO_MIN_DUTY = float(os.environ.get("SERVO_MIN_DUTY", "2.5"))
SERVO_MAX_DUTY = float(os.environ.get("SERVO_MAX_DUTY", "12.5"))


def angle_to_duty(angle: float) -> float:
    angle = max(0.0, min(180.0, float(angle)))
    return SERVO_MIN_DUTY + (angle / 180.0) * (SERVO_MAX_DUTY - SERVO_MIN_DUTY)


SERVO_CENTER_ANGLE = float(os.environ.get("SERVO_CENTER_ANGLE", "90"))
SERVO_CENTER_DUTY = angle_to_duty(SERVO_CENTER_ANGLE)

SERVO_SWING_ANGLE = {
    "high": (
        float(os.environ.get("SERVO_HIGH_LEFT_ANGLE", "30")),
        float(os.environ.get("SERVO_HIGH_RIGHT_ANGLE", "145")),
    ),
    "medium": (
        float(os.environ.get("SERVO_MEDIUM_LEFT_ANGLE", "45")),
        float(os.environ.get("SERVO_MEDIUM_RIGHT_ANGLE", "135")),
    ),
    "low": (
        float(os.environ.get("SERVO_LOW_LEFT_ANGLE", "55")),
        float(os.environ.get("SERVO_LOW_RIGHT_ANGLE", "125")),
    ),
}

SERVO_SWING_DUTY = {
    key: (angle_to_duty(left), angle_to_duty(right))
    for key, (left, right) in SERVO_SWING_ANGLE.items()
}

BRUSH_DUTY = {
    key: round(max(left_duty, right_duty), 2)
    for key, (left_duty, right_duty) in SERVO_SWING_DUTY.items()
}

SERVO_MOVE_DELAY = {
    "high":   0.3,
    "medium": 0.42,
    "low":    0.65,
}

FIRST_BRUSH_COUNT_BY_GROUP = {"GROUP_A": 3, "GROUP_B": 2, "GROUP_C": 1, "GROUP_D": 1}
SECOND_BRUSH_COUNT_BY_GROUP = {"GROUP_A": 5, "GROUP_B": 4, "GROUP_C": 3, "GROUP_D": 1}
BRUSH_COUNT_BY_GROUP = {
    group: FIRST_BRUSH_COUNT_BY_GROUP[group] + SECOND_BRUSH_COUNT_BY_GROUP[group]
    for group in FIRST_BRUSH_COUNT_BY_GROUP
}
MAX_UMOTOR_MS     = int(os.environ.get("MAX_UMOTOR_MS",    "10000"))
DEFAULT_UMOTOR_MS = int(os.environ.get("DEFAULT_UMOTOR_MS",  "1000"))
RINSE_PHASE_MS    = int(os.environ.get("RINSE_PHASE_MS",     "500"))
DEFAULT_STEPPER_MS = int(os.environ.get("DEFAULT_STEPPER_MS", "5000"))
MAX_STEPPER_MS = int(os.environ.get("MAX_STEPPER_MS", "6000"))
OCR_CONFIG = "--oem 3 --psm 6 -l kor+eng"
CAMERA_INDEX = int(os.environ.get("CAMERA_INDEX", "1"))
PREVIEW_FLIP_HORIZONTAL = True
PREVIEW_FLIP_VERTICAL = True
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

CLEANUP_OPENAI_ENABLED = os.environ.get("CLEANUP_OPENAI_ENABLED", "0").strip().lower() in {"1", "true", "yes", "on"}
FEEDBACK_ADJUST_ENABLED = os.environ.get("FEEDBACK_ADJUST_ENABLED", "1").strip().lower() not in {"0", "false", "no", "off"}
FEEDBACK_ADJUST_MIN_COUNT = int(os.environ.get("FEEDBACK_ADJUST_MIN_COUNT", "3"))
FEEDBACK_ADJUST_MIN_RATIO = float(os.environ.get("FEEDBACK_ADJUST_MIN_RATIO", "0.25"))
FEEDBACK_ADJUST_MARGIN = float(os.environ.get("FEEDBACK_ADJUST_MARGIN", "0.65"))
FEEDBACK_GOOD_THRESHOLD  = float(os.environ.get("FEEDBACK_GOOD_THRESHOLD",  "70.0"))
FEEDBACK_PARTIAL_THRESHOLD = float(os.environ.get("FEEDBACK_PARTIAL_THRESHOLD", "40.0"))
PALE_YELLOW_RECHECK_ENABLED = os.environ.get("PALE_YELLOW_RECHECK_ENABLED", "1").strip().lower() not in {"0", "false", "no", "off"}
PALE_YELLOW_RECHECK_MIN_SIGNAL = float(os.environ.get("PALE_YELLOW_RECHECK_MIN_SIGNAL", "0.34"))
PALE_YELLOW_RECHECK_MARGIN = float(os.environ.get("PALE_YELLOW_RECHECK_MARGIN", "0.04"))
ACHROMATIC_FABRIC_ONLY = os.environ.get("ACHROMATIC_FABRIC_ONLY", "0").strip().lower() not in {"0", "false", "no", "off"}
FAMILY_COMBO_OVERRIDE_MAX_CONF = float(os.environ.get("FAMILY_COMBO_OVERRIDE_MAX_CONF", "0.66"))
FAMILY_COMBO_OVERRIDE_MIN_SCORE = float(os.environ.get("FAMILY_COMBO_OVERRIDE_MIN_SCORE", "0.34"))
FAMILY_COMBO_OVERRIDE_NEAR_RATIO = float(os.environ.get("FAMILY_COMBO_OVERRIDE_NEAR_RATIO", "0.96"))
DENIM_RED_RECHECK_MAX_CONF = float(os.environ.get("DENIM_RED_RECHECK_MAX_CONF", "0.74"))
DENIM_RED_RECHECK_SCORE_RATIO = float(os.environ.get("DENIM_RED_RECHECK_SCORE_RATIO", "0.84"))
DARK_FABRIC_VALUE_MAX = float(os.environ.get("DARK_FABRIC_VALUE_MAX", "105"))
DARK_FABRIC_RATIO_MIN = float(os.environ.get("DARK_FABRIC_RATIO_MIN", "0.34"))
DARK_FABRIC_DENIM_RATIO_MIN = float(os.environ.get("DARK_FABRIC_DENIM_RATIO_MIN", "0.22"))
BROWN_SOURCE_SET = {"coffee", "teriyaki_sauce", "oriental_dressing", "soy_sauce"}
YELLOW_SOURCE_SET = {"mustard", "curry", "oil"}
RED_SOURCE_SET = {"bbq_sauce", "gochujang", "ketchup"}

SOURCE_FAMILY = {
    "coffee": "brown",
    "teriyaki_sauce": "brown",
    "oriental_dressing": "brown",
    "soy_sauce": "brown",
    "bbq_sauce": "red",
    "gochujang": "red",
    "ketchup": "red",
    "curry": "yellow",
    "mustard": "yellow",
    "oil": "yellow",
    "milk": "pale",
    "mayonnaise": "pale",
}
FAMILY_DISPLAY_KO = {
    "red": "빨강",
    "brown": "갈색",
    "yellow": "노랑",
    "pale": "흰색 계열",
    "other": "기타",
    "unknown": "판단 어려움",
}
FAMILY_SOURCE_SETS = {
    "red": RED_SOURCE_SET,
    "brown": BROWN_SOURCE_SET,
    "yellow": YELLOW_SOURCE_SET,
    "pale": {"milk", "mayonnaise"},
}
STAIN_SENSORY_PROFILE = {
    "coffee": {
        "color": "맑은 갈색에서 짙은 갈색으로 번지는 색감",
    },
    "soy_sauce": {
        "color": "진한 갈색이 얇게 물드는 색감",
    },
    "teriyaki_sauce": {
        "color": "갈색에 윤기가 섞인 색감",
    },
    "oriental_dressing": {
        "color": "갈색과 노란 기름기가 섞인 색감",
    },
    "ketchup": {
        "color": "밝고 선명한 빨강 계열 색감",
    },
    "bbq_sauce": {
        "color": "윤기 있는 붉은색과 붉은 갈색이 섞인 색감",
    },
    "gochujang": {
        "color": "진하고 어두운 빨강 계열 색감",
    },
    "curry": {
        "color": "노랑과 주황이 섞인 강한 색감",
    },
    "mustard": {
        "color": "밝은 노랑 계열 색감",
    },
    "oil": {
        "color": "투명하거나 옅은 노랑의 번들거리는 색감",
    },
    "milk": {
        "color": "흰색 계열로 옅게 번지는 색감",
    },
    "mayonnaise": {
        "color": "흰색과 옅은 노랑이 섞인 색감",
    },
}


PERCENT_PATTERN = re.compile(
    r'(cotton|poly(?:ester)?|synthetic|denim|nylon|wool|knit|cashmere|silk|'
    r'면|폴리|폴리에스터|합성|데님|나일론|울|니트|캐시미어|실크)'
    r'\s*(\d{1,3})\s*%?', re.IGNORECASE
)
MATERIAL_KEYWORDS = {
    "GROUP_A": ["cotton", "denim", "면", "데님"],
    "GROUP_B": ["poly", "polyester", "synthetic", "nylon", "폴리", "폴리에스터", "합성", "나일론"],
    "GROUP_C": ["wool", "cashmere", "silk", "울", "캐시미어", "실크"],
    "GROUP_D": ["knit", "니트"],
}
SYNONYM_MAP = {
    "poly": "polyester", "폴리": "polyester", "폴리에스터": "polyester",
    "synthetic": "synthetic", "합성": "synthetic",
    "nylon": "synthetic", "나일론": "synthetic",
    "면": "cotton", "데님": "denim",
    "울": "wool", "니트": "knit", "캐시미어": "cashmere", "실크": "silk",
}
GROUP_DISPLAY = {
    "GROUP_A": "일반 원단 (면/데님)",
    "GROUP_B": "합성 원단 (폴리/나일론)",
    "GROUP_C": "민감 원단 (울/캐시미어/실크)",
    "GROUP_D": "니트 원단",
}
MATERIAL_ABSORB_REASON = {
    "GROUP_A": "면·데님은 흡수성이 높아 수용성 오염이 섬유 속으로 빠르게 침투합니다. 충분한 세제와 브러싱을 적용했습니다.",
    "GROUP_B": "합성 섬유는 흡수성이 낮아 오염이 표면에 머물러 세척이 비교적 쉽습니다.",
    "GROUP_C": "울·캐시미어·실크는 흡수성이 있으나 섬유 손상에 민감해 브러싱 강도를 최소화했습니다.",
    "GROUP_D": "니트는 구조 변형 위험이 있어 민감 원단과 같은 약한 강도로 짧게 나누어 브러싱합니다.",
}

CLASS_TO_SOURCE = {
    "bbq":     "bbq_sauce",
    "coffee":  "coffee",
    "curry":   "curry",
    "deri":    "teriyaki_sauce",
    "go":      "gochujang",
    "ketchup": "ketchup",
    "mayo":    "mayonnaise",
    "milk":    "milk",
    "mustard": "mustard",
    "oil":     "oil",
    "ori":     "oriental_dressing",
    "soy":     "soy_sauce",
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
    "coffee":           {"GROUP_A": "medium", "GROUP_B": "low",    "GROUP_C": "low", "GROUP_D": "low"},
    "teriyaki_sauce":   {"GROUP_A": "high",   "GROUP_B": "medium", "GROUP_C": "low", "GROUP_D": "low"},
    "oriental_dressing":{"GROUP_A": "medium", "GROUP_B": "medium", "GROUP_C": "low", "GROUP_D": "low"},
    "soy_sauce":        {"GROUP_A": "medium", "GROUP_B": "low",    "GROUP_C": "low", "GROUP_D": "low"},
    "bbq_sauce":        {"GROUP_A": "high",   "GROUP_B": "medium", "GROUP_C": "low", "GROUP_D": "low"},
    "gochujang":        {"GROUP_A": "high",   "GROUP_B": "medium", "GROUP_C": "low", "GROUP_D": "low"},
    "ketchup":          {"GROUP_A": "medium", "GROUP_B": "low",    "GROUP_C": "low", "GROUP_D": "low"},
    "milk":             {"GROUP_A": "low",    "GROUP_B": "low",    "GROUP_C": "low", "GROUP_D": "low"},
    "mayonnaise":       {"GROUP_A": "medium", "GROUP_B": "medium", "GROUP_C": "low", "GROUP_D": "low"},
    "curry":            {"GROUP_A": "high",   "GROUP_B": "medium", "GROUP_C": "low", "GROUP_D": "low"},
    "mustard":          {"GROUP_A": "medium", "GROUP_B": "low",    "GROUP_C": "low", "GROUP_D": "low"},
    "oil":              {"GROUP_A": "medium", "GROUP_B": "medium", "GROUP_C": "low", "GROUP_D": "low"},
}
PER_STROKE_MS = {"high": 600, "medium": 450, "low": 300}
INTENSITY_TO_BRUSH_MS = PER_STROKE_MS  # 하위 호환용 별칭
INTENSITY_LABEL_KO = {"high": "강", "medium": "중", "low": "약"}
INTENSITY_DESCRIPTION_KO = {
    "high": "강한 브러싱으로 점성이 큰 얼룩을 분해합니다.",
    "medium": "일반적인 얼룩 제거에 적합한 표준 브러싱입니다.",
    "low": "섬유 손상을 줄이기 위한 약한 브러싱입니다.",
}
PUMP_MS_BY_SOURCE = {
    "coffee": 100, "teriyaki_sauce": 300, "oriental_dressing": 200, "soy_sauce": 100,
    "bbq_sauce": 300, "gochujang": 400, "ketchup": 200, "milk": 100,
    "mayonnaise": 300, "curry": 300, "mustard": 300, "oil": 400,
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
    deadline = time.time() + max(ARDUINO_LONG_TIMEOUT, 2.0)

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

        if line == "DONE" or line.startswith("DONE:"):
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
    done_seen = any(line == "DONE" or line.startswith("DONE:") for line in responses)

    if not ok_seen:
        raise RuntimeError(f"arduino_pump_start_failed:{responses}")

    if not done_seen:
        raise RuntimeError(f"arduino_pump_done_failed:{responses}")


def run_brush_pwm(brush_count: int, brush_intensity: str = "medium") -> None:
    if brush_count <= 0:
        return

    setup_brush_gpio()

    if _BRUSH_PWM is None:
        raise RuntimeError("brush_pwm_not_initialized")

    intensity = brush_intensity if brush_intensity in SERVO_SWING_DUTY else "medium"
    left_duty, right_duty = SERVO_SWING_DUTY[intensity]
    move_delay = SERVO_MOVE_DELAY[intensity]

    try:
        _BRUSH_PWM.ChangeDutyCycle(SERVO_CENTER_DUTY)
        time.sleep(0.4)

        for _ in range(brush_count):
            _BRUSH_PWM.ChangeDutyCycle(left_duty)
            time.sleep(move_delay)

            _BRUSH_PWM.ChangeDutyCycle(right_duty)
            time.sleep(move_delay)

        _BRUSH_PWM.ChangeDutyCycle(SERVO_CENTER_DUTY)
        time.sleep(0.4)

        _BRUSH_PWM.ChangeDutyCycle(0)

    finally:
        try:
            _BRUSH_PWM.ChangeDutyCycle(0)
        except Exception:
            pass
def run_stepper(direction: str, duration_ms: int = DEFAULT_STEPPER_MS) -> List[str]:
    direction = str(direction or "").strip().lower()
    if direction not in {"down", "up"}:
        raise RuntimeError(f"bad_stepper_direction:{direction}")
    duration_ms = max(1, min(MAX_STEPPER_MS, int(duration_ms)))
    cmd = f"STEP_{direction.upper()},{duration_ms}"
    responses = arduino_command(cmd, wait_done=True)
    if not responses:
        raise RuntimeError("arduino_stepper_no_response")
    ok_prefix = f"OK:STEP_{direction.upper()}"
    done_prefix = f"DONE:STEP_{direction.upper()}"
    ok_seen = any(line.startswith(ok_prefix) for line in responses)
    done_seen = any(line == "DONE" or line.startswith(done_prefix) for line in responses)
    if not ok_seen:
        raise RuntimeError(f"arduino_stepper_start_failed:{responses}")
    if not done_seen:
        raise RuntimeError(f"arduino_stepper_done_failed:{responses}")
    return responses


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
        if line == "DONE" or line.startswith("DONE:"):
            break
    if not any(l.startswith("OK:UM") for l in responses):
        raise RuntimeError(f"arduino_umotor_start_failed:{responses}")
    if not any(l == "DONE" or l.startswith("DONE:") for l in responses):
        raise RuntimeError(f"arduino_umotor_done_failed:{responses}")


atexit.register(cleanup_brush_gpio)
atexit.register(close_arduino_serial)

class AnalyzeTripletRequest(BaseModel):
    before_image: str
    after_image: str
    tag_image: Optional[str] = None
    manual_material_group: Optional[Literal["GROUP_A", "GROUP_B", "GROUP_C", "GROUP_D"]] = None
    manual_material_label: Optional[str] = None

class ExecuteRequest(BaseModel):
    pump_ms: int = Field(ge=0, le=MAX_PUMP_MS)
    brush_intensity: Literal["high", "medium", "low"] = "medium"
    detergent_key: Optional[str] = None

class CleanupFeedbackRequest(BaseModel):
    baseline_image: str
    cleaned_image: str
    stain_label: str
    stain_label_kr: Optional[str] = None
    material_group: Optional[Literal["GROUP_A", "GROUP_B", "GROUP_C", "GROUP_D"]] = None


class DevStateRequest(BaseModel):
    stage: Optional[str] = None
    history: List[str] = Field(default_factory=list)
    tagMode: Optional[str] = None
    manualMaterialGroup: Optional[str] = None
    manualMaterialLabel: Optional[str] = None
    hasTagImage: Optional[bool] = None
    hasBeforeImage: Optional[bool] = None
    hasAfterImage: Optional[bool] = None
    hasFeedbackImage: Optional[bool] = None
    tagImage: Optional[str] = None
    beforeImage: Optional[str] = None
    afterImage: Optional[str] = None
    feedbackImage: Optional[str] = None
    latestAnalysis: Optional[Dict[str, Any]] = None
    logs: List[str] = Field(default_factory=list)


class LearningFeedbackRequest(BaseModel):
    verdict: Literal["correct", "incorrect"]
    correct_label: Optional[str] = None
    actual_color: Optional[Literal["red", "brown", "yellow", "pale", "other", "unknown"]] = None
    candidate_fit_score: Optional[int] = None
    before_image: Optional[str] = None
    after_image: Optional[str] = None
    tag_image: Optional[str] = None
    feedback_image: Optional[str] = None
    analysis: Dict[str, Any] = Field(default_factory=dict)

class Treatment(BaseModel):
    brush_intensity: Literal["high", "medium", "low"]
    pump_ms: int
    brush_count: int
    first_brush_count: int
    second_brush_count: int
    rinse_phase_ms: int
    detergent_key: str
    pump_pin: int
    summary: str

class AnalyzeTripletResponse(BaseModel):
    final_label: str
    final_label_kr: str
    decision_source: str
    decision_summary: Optional[str] = None
    material_input_mode: str
    confidence: float
    cnn_before_class: str
    cnn_after_class: str
    cnn_before_confidence: float
    cnn_after_confidence: float
    material_group: Literal["GROUP_A", "GROUP_B", "GROUP_C", "GROUP_D"]
    material_group_display: str
    material_reason: str
    treatment_kr: str
    detergent_kr: str
    reason_kr: List[str]
    top_candidates: List[Dict[str, Any]]
    visual_family: Optional[str] = None
    visual_family_display: Optional[str] = None
    family_summary: List[Dict[str, Any]] = Field(default_factory=list)
    visual_evidence: List[str] = Field(default_factory=list)
    feedback_adjust_reason: Optional[str] = None
    feedback_adjust_meta: Optional[Dict[str, Any]] = None
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

class StepperRequest(BaseModel):
    direction: Literal["down", "up"]
    duration_ms: int = Field(default=DEFAULT_STEPPER_MS, ge=1, le=MAX_STEPPER_MS)

class StepperResponse(BaseModel):
    ok: bool
    direction: Literal["down", "up"]
    duration_ms: int
    responses: List[str]

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


def tag_orientation_candidates(frame: np.ndarray) -> List[Tuple[str, np.ndarray]]:
    candidates = [("display", frame)]
    restored = restore_tag_orientation(frame)
    if not np.array_equal(restored, frame):
        candidates.append(("restored", restored))
    return candidates


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

    if any(keyword.lower() in full_text for keyword in MATERIAL_KEYWORDS["GROUP_D"]):
        return "GROUP_D", "택 OCR에서 니트 구조 키워드를 감지해 약한 브러싱을 짧게 나누는 그룹으로 처리했습니다."

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


def is_denim_context(
    material_group: str,
    material_reason: str,
    ocr_texts: List[str],
    manual_material_label: Optional[str] = None,
) -> bool:
    if material_group != "GROUP_A":
        return False

    label_text = str(manual_material_label or "").strip().lower()
    if any(token in label_text for token in ["데님", "denim", "jean", "jeans", "청"]):
        return True
    if label_text:
        return False

    reason_text = str(material_reason or "").lower()
    if any(token in reason_text for token in ["데님", "denim", "jean", "jeans", "청"]):
        return True

    joined_ocr = " ".join([str(t or "").lower() for t in (ocr_texts or [])])
    if any(token in joined_ocr for token in ["데님", "denim", "jean", "jeans", "청"]):
        return True

    return False


def detect_dark_fabric_context(before_bgr: np.ndarray, after_bgr: np.ndarray) -> bool:
    dark_ratios: List[float] = []
    achro_dark_ratios: List[float] = []
    denim_ratios: List[float] = []
    for image_bgr in [before_bgr, after_bgr]:
        if image_bgr is None or image_bgr.size == 0:
            continue
        img = cv2.resize(_center_crop_for_color(image_bgr, ratio=0.82), (224, 224))
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h = hsv[:, :, 0]
        s = hsv[:, :, 1]
        v = hsv[:, :, 2]

        red_stain = ((h <= 13) | (h >= 165)) & (s >= 55) & (v >= 65)
        yellow_brown_stain = (h >= 8) & (h <= 45) & (s >= 45) & (v >= 70)
        stain_like = red_stain | yellow_brown_stain
        dark_fabric = ((v <= DARK_FABRIC_VALUE_MAX) & (s >= 18) & ~stain_like) | (v <= DARK_FABRIC_VALUE_MAX * 0.72)
        achro_dark = ((v <= DARK_FABRIC_VALUE_MAX * 1.05) & (s <= 48) & ~stain_like) | ((v <= 128) & (s <= 36))
        denim_like = (h >= 84) & (h <= 132) & (s >= 24) & (v >= 22) & (v <= 215) & ~stain_like
        deep_denim_like = (h >= 86) & (h <= 128) & (s >= 40) & (v <= 160) & ~stain_like

        total = float(img.shape[0] * img.shape[1])
        dark_ratios.append(float(np.count_nonzero(dark_fabric)) / total)
        achro_dark_ratios.append(float(np.count_nonzero(achro_dark)) / total)
        denim_ratios.append(float(np.count_nonzero(denim_like | deep_denim_like)) / total)

    if not dark_ratios:
        return False

    dark_peak = max(dark_ratios)
    achro_peak = max(achro_dark_ratios) if achro_dark_ratios else 0.0
    denim_peak = max(denim_ratios) if denim_ratios else 0.0
    blended_peak = max(dark_peak, achro_peak * 0.92) + denim_peak * 0.30

    if ACHROMATIC_FABRIC_ONLY:
        return dark_peak >= DARK_FABRIC_RATIO_MIN or achro_peak >= max(0.22, DARK_FABRIC_RATIO_MIN * 0.86)

    return (
        dark_peak >= DARK_FABRIC_RATIO_MIN
        or achro_peak >= max(0.24, DARK_FABRIC_RATIO_MIN * 0.90)
        or denim_peak >= DARK_FABRIC_DENIM_RATIO_MIN
        or blended_peak >= DARK_FABRIC_RATIO_MIN
    )


def _background_ring_profile_for_analysis(image_bgr: np.ndarray) -> Dict[str, float]:
    if image_bgr is None or image_bgr.size == 0:
        return {
            "mask_ratio": 0.0,
            "mean_v": 0.0,
            "mean_s": 0.0,
            "mean_h": 0.0,
            "mean_lab_b": 0.0,
            "dark_ratio": 0.0,
            "very_dark_ratio": 0.0,
            "denim_ratio": 0.0,
            "warm_ratio": 0.0,
            "dark_hint": 0.0,
        }

    img = cv2.resize(_center_crop_for_color(image_bgr, ratio=0.82), (224, 224))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    lab_b = lab[:, :, 2]

    hh, ww = img.shape[:2]
    yy, xx = np.indices((hh, ww))
    cx = (ww - 1) / 2.0
    cy = (hh - 1) / 2.0
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / float(min(hh, ww))

    # 중앙 얼룩을 피하고 주변 원단 톤만 읽기 위한 링 영역
    ring = (dist >= 0.16) & (dist <= 0.44)
    red_stain = ((h <= 14) | (h >= 164)) & (s >= 45) & (v >= 45)
    warm_stain = (h >= 8) & (h <= 52) & ((s >= 26) | (lab_b >= 145)) & (v >= 42)
    pale_warm_stain = (h >= 10) & (h <= 52) & (s <= 48) & (v >= 145) & (lab_b >= 144)
    stain_like = (red_stain | warm_stain | pale_warm_stain).astype(np.uint8) * 255
    stain_like = cv2.dilate(stain_like, np.ones((13, 13), np.uint8), iterations=1) > 0

    bg_mask = ring & ~stain_like
    if np.count_nonzero(bg_mask) < int(hh * ww * 0.03):
        # 링 영역이 부족하면 중앙을 제외한 외곽 링으로 fallback
        bg_mask = (dist >= 0.24) & (dist <= 0.46)

    bg_count = float(np.count_nonzero(bg_mask))
    total = float(hh * ww)
    if bg_count < 1.0:
        return {
            "mask_ratio": 0.0,
            "mean_v": 0.0,
            "mean_s": 0.0,
            "mean_h": 0.0,
            "mean_lab_b": 0.0,
            "dark_ratio": 0.0,
            "very_dark_ratio": 0.0,
            "denim_ratio": 0.0,
            "warm_ratio": 0.0,
            "dark_hint": 0.0,
        }

    bg_h = h[bg_mask]
    bg_s = s[bg_mask]
    bg_v = v[bg_mask]
    bg_b = lab_b[bg_mask]

    dark_ratio = float(np.mean((bg_v <= DARK_FABRIC_VALUE_MAX) & (bg_s >= 12)))
    very_dark_ratio = float(np.mean(bg_v <= (DARK_FABRIC_VALUE_MAX * 0.82)))
    denim_ratio = float(np.mean((bg_h >= 84) & (bg_h <= 132) & (bg_s >= 24) & (bg_v <= 215)))
    warm_ratio = float(np.mean((bg_h >= 10) & (bg_h <= 52) & ((bg_s >= 24) | (bg_b >= 145))))
    dark_hint = 1.0 if (
        dark_ratio >= max(0.22, DARK_FABRIC_RATIO_MIN * 0.78)
        or very_dark_ratio >= 0.16
        or denim_ratio >= max(0.16, DARK_FABRIC_DENIM_RATIO_MIN * 0.80)
    ) else 0.0

    return {
        "mask_ratio": float(bg_count / total),
        "mean_v": float(np.mean(bg_v)),
        "mean_s": float(np.mean(bg_s)),
        "mean_h": float(np.mean(bg_h)),
        "mean_lab_b": float(np.mean(bg_b)),
        "dark_ratio": dark_ratio,
        "very_dark_ratio": very_dark_ratio,
        "denim_ratio": denim_ratio,
        "warm_ratio": warm_ratio,
        "dark_hint": dark_hint,
    }


def _normalize_analysis_image_with_background(
    image_bgr: np.ndarray,
    bg_profile: Dict[str, float],
    force_dark_mode: bool = False,
) -> np.ndarray:
    if image_bgr is None or image_bgr.size == 0:
        return image_bgr
    if not bg_profile:
        return image_bgr

    dark_mode = force_dark_mode or bool(bg_profile.get("dark_hint", 0.0) >= 0.5)
    if not dark_mode:
        return image_bgr

    bg_v = float(bg_profile.get("mean_v", 0.0))
    bg_s = float(bg_profile.get("mean_s", 0.0))
    if bg_v <= 1.0:
        return image_bgr

    denim_ratio = float(bg_profile.get("denim_ratio", 0.0))
    target_v = 94.0 if denim_ratio >= 0.15 else 102.0
    target_s = 72.0 if denim_ratio >= 0.15 else 64.0
    # 어두운 배경 자동노출을 누르기 위해 밝기(V)를 낮추고 채도(S)를 보정
    val_gain = max(0.68, min(1.02, target_v / bg_v))
    sat_gain = max(1.00, min(1.38, target_s / max(bg_s, 1.0)))

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * sat_gain, 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * val_gain, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)



def resolve_material_input(
    tag_image: Optional[str],
    manual_material_group: Optional[str],
    manual_material_label: Optional[str] = None,
) -> Tuple[Optional[np.ndarray], List[str], List[str], str, str, str]:
    if manual_material_group:
        if manual_material_group not in GROUP_DISPLAY:
            raise HTTPException(status_code=400, detail="invalid_manual_material_group")
        label_text = str(manual_material_label or "").strip()
        if label_text:
            reason = f"사용자가 화면에서 {label_text}({GROUP_DISPLAY[manual_material_group]})을(를) 직접 선택했습니다."
        else:
            reason = f"사용자가 화면에서 {GROUP_DISPLAY[manual_material_group]}을(를) 직접 선택했습니다."
        return None, [], [], manual_material_group, reason, "manual_select"

    if not tag_image:
        raise HTTPException(status_code=400, detail="material_input_required:tag_or_manual")

    try:
        decoded_tag_img = decode_data_url_to_bgr(tag_image)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"tag_image_parse_error:{repr(e)}")

    tag_img = decoded_tag_img
    ocr_texts: List[str] = []
    debug_files: List[str] = []
    raw_group = None
    material_reason = ""
    for orientation_name, candidate_img in tag_orientation_candidates(decoded_tag_img):
        candidate_texts, candidate_debug_files = ocr_material_texts(candidate_img)
        candidate_group, candidate_reason = detect_material_group_from_ocr(candidate_texts)
        if not ocr_texts:
            ocr_texts = candidate_texts
            debug_files = candidate_debug_files
            material_reason = candidate_reason
        if candidate_group is not None:
            tag_img = candidate_img
            ocr_texts = candidate_texts
            debug_files = candidate_debug_files
            raw_group = candidate_group
            material_reason = f"{candidate_reason} ({orientation_name} 방향 기준)"
            break

    if raw_group is None:
        material_group = "GROUP_C"
        material_reason = "택 OCR에서 재질을 읽지 못해 안전한 기본값 GROUP_C(민감 원단)로 처리했습니다."
        input_mode = "tag_ocr_fallback"
    else:
        material_group = raw_group
        input_mode = "tag_ocr"
    return tag_img, ocr_texts, debug_files, material_group, material_reason, input_mode


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


def _center_crop_for_color(image_bgr: np.ndarray, ratio: float = 0.74) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    if h <= 0 or w <= 0:
        return image_bgr
    crop_w = max(1, int(w * ratio))
    crop_h = max(1, int(h * ratio))
    x1 = max(0, (w - crop_w) // 2)
    y1 = max(0, (h - crop_h) // 2)
    return image_bgr[y1:y1 + crop_h, x1:x1 + crop_w]


def _visual_family_scores_for_image(
    image_bgr: np.ndarray,
    is_dark_fabric_context: bool = False,
) -> Dict[str, float]:
    if image_bgr is None or image_bgr.size == 0:
        return {"red": 0.0, "brown": 0.0, "yellow": 0.0, "pale": 0.0}

    img = cv2.resize(_center_crop_for_color(image_bgr), (224, 224))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    lab_b = lab[:, :, 2]
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]

    if is_dark_fabric_context:
        colored = (s >= 34) & (v >= 24)
        red_rg_ratio = 1.06
        red_rb_ratio = 1.04
        red_sat_min = 52
        red_val_min = 72
        brown_core_v_max = 176
        brown_dark_sat_min = 42
        brown_dark_v_max = 102
        yellow_h_min, yellow_h_max = 16, 47
        yellow_sat_min = 34
        yellow_val_min = 68
        yellow_lab_min = 142
    else:
        colored = (s >= 40) & (v >= 30)
        red_rg_ratio = 1.10
        red_rb_ratio = 1.08
        red_sat_min = 58
        red_val_min = 86
        brown_core_v_max = 188
        brown_dark_sat_min = 45
        brown_dark_v_max = 120
        yellow_h_min, yellow_h_max = 18, 45
        yellow_sat_min = 42
        yellow_val_min = 82
        yellow_lab_min = 146

    red_dominant = (r >= g * red_rg_ratio) & (r >= b * red_rb_ratio)
    red_hue = (h <= 13) | (h >= 165)
    red_mask = colored & red_hue & red_dominant & (s >= red_sat_min) & (v >= red_val_min)

    brown_hue = (h >= 6) & (h <= 30)
    brown_core = colored & brown_hue & (s >= 40) & (v >= 32) & (v <= brown_core_v_max)
    brown_dark_red = colored & red_hue & red_dominant & (s >= brown_dark_sat_min) & (v < brown_dark_v_max)
    brown_mask = (brown_core | brown_dark_red) & ~red_mask
    yellow_mask = (
        colored
        & (h >= yellow_h_min)
        & (h <= yellow_h_max)
        & (v >= yellow_val_min)
        & ((s >= yellow_sat_min) | (lab_b >= yellow_lab_min))
        & ~red_mask
        & ~brown_mask
    )

    total_pixels = float(img.shape[0] * img.shape[1])
    colored_count = float(np.count_nonzero(colored))
    colored_base = colored_count if colored_count >= 1.0 else 1.0

    red_ratio = float(np.count_nonzero(red_mask)) / colored_base
    brown_ratio = float(np.count_nonzero(brown_mask)) / colored_base
    yellow_ratio = float(np.count_nonzero(yellow_mask)) / colored_base

    neutral_rgb = (np.abs(r - g) <= 22) & (np.abs(g - b) <= 22) & (np.abs(r - b) <= 22)
    pale_core = (s <= 55) & (v >= 150)
    pale_soft = (s <= 68) & (v >= 175)
    warm_shift = (h >= 12) & (h <= 52) & (lab_b >= (142 if is_dark_fabric_context else 146))
    pale_mask = (pale_core | pale_soft | (neutral_rgb & (v >= 145))) & ~warm_shift
    pale_ratio = float(np.count_nonzero(pale_mask)) / total_pixels
    colored_ratio = colored_count / total_pixels
    chroma_peak = max(red_ratio, brown_ratio, yellow_ratio)
    pale_score = pale_ratio * (1.0 - min(0.65, chroma_peak * 0.90))
    if colored_ratio >= 0.65:
        pale_score *= 0.75
    if is_dark_fabric_context:
        pale_score *= 0.78
        if yellow_ratio >= 0.10:
            pale_score *= 0.82

    return {
        "red": max(0.0, min(1.0, red_ratio)),
        "brown": max(0.0, min(1.0, brown_ratio)),
        "yellow": max(0.0, min(1.0, yellow_ratio)),
        "pale": max(0.0, min(1.0, pale_score)),
    }


def detect_visual_family(
    before_bgr: np.ndarray,
    after_bgr: np.ndarray,
    is_dark_fabric_context: bool = False,
) -> Dict[str, Any]:
    before_scores = _visual_family_scores_for_image(before_bgr, is_dark_fabric_context)
    after_scores = _visual_family_scores_for_image(after_bgr, is_dark_fabric_context)
    scores = {
        family: before_scores.get(family, 0.0) * 0.45 + after_scores.get(family, 0.0) * 0.55
        for family in FAMILY_DISPLAY_KO
        if family != "unknown"
    }
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    best_family, best_score = ranked[0] if ranked else ("unknown", 0.0)
    next_score = ranked[1][1] if len(ranked) > 1 else 0.0
    margin = best_score - next_score
    brown_score = float(scores.get("brown", 0.0))
    yellow_score = float(scores.get("yellow", 0.0))
    pale_score = float(scores.get("pale", 0.0))

    if best_family == "red":
        brown_near_ratio = 0.93 if is_dark_fabric_context else 0.88
        brown_abs_min = 0.20 if is_dark_fabric_context else 0.18
        if brown_score >= best_score * brown_near_ratio and brown_score >= brown_abs_min:
            best_family = "brown"
            best_score = brown_score
            margin = brown_score - yellow_score
        elif yellow_score >= best_score * (0.95 if is_dark_fabric_context else 0.92) and yellow_score >= 0.19:
            best_family = "yellow"
            best_score = yellow_score
            margin = yellow_score - brown_score
    elif best_family == "brown":
        pale_ratio_thr = 1.12 if is_dark_fabric_context else 0.95
        pale_abs_thr = 0.36 if is_dark_fabric_context else 0.30
        if pale_score >= best_score * pale_ratio_thr and pale_score >= pale_abs_thr:
            best_family = "pale"
            best_score = pale_score
            margin = pale_score - max(red_score for red_score in [float(scores.get("red", 0.0)), yellow_score, brown_score])

    if best_family == "pale":
        if best_score < (0.30 if is_dark_fabric_context else 0.24):
            best_family = "unknown"
    else:
        min_best = 0.17 if is_dark_fabric_context else 0.16
        min_margin = 0.045 if is_dark_fabric_context else 0.05
        if best_score < min_best or margin < min_margin:
            best_family = "unknown"

    return {
        "family": best_family,
        "family_kr": FAMILY_DISPLAY_KO.get(best_family, best_family),
        "score": float(best_score),
        "margin_to_next": float(margin),
        "scores": {k: float(v) for k, v in scores.items()},
    }


def _yellow_focus_metrics_for_image(image_bgr: np.ndarray) -> Dict[str, float]:
    if image_bgr is None or image_bgr.size == 0:
        return {
            "focus_area_ratio": 0.0,
            "yellow_focus_ratio": 0.0,
            "pale_focus_ratio": 0.0,
            "lab_yellow_mean": 0.0,
            "yellow_signal": 0.0,
            "mean_yellow_h": 0.0,
            "mean_yellow_s": 0.0,
            "mean_yellow_v": 0.0,
        }

    img = cv2.resize(_center_crop_for_color(image_bgr), (224, 224))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    b_channel = lab[:, :, 2]

    warm_mask = (h >= 8) & (h <= 52) & (v >= 58) & (s >= 10)
    chroma_mask = (s >= 20) & (v >= 40) & (v <= 245)
    shadow_warm = (h >= 8) & (h <= 45) & (v >= 35) & (v < 120) & (s >= 15)
    focus_mask = (warm_mask | chroma_mask | shadow_warm).astype(np.uint8) * 255
    kernel_close = np.ones((5, 5), np.uint8)
    kernel_open = np.ones((3, 3), np.uint8)
    focus_mask = cv2.morphologyEx(focus_mask, cv2.MORPH_CLOSE, kernel_close, iterations=1)
    focus_mask = cv2.morphologyEx(focus_mask, cv2.MORPH_OPEN, kernel_open, iterations=1)

    total_pixels = float(img.shape[0] * img.shape[1])
    focus_area = float(cv2.countNonZero(focus_mask))
    focus_area_ratio = focus_area / total_pixels if total_pixels > 0 else 0.0
    if focus_area < 80:
        return {
            "focus_area_ratio": float(focus_area_ratio),
            "yellow_focus_ratio": 0.0,
            "pale_focus_ratio": 0.0,
            "lab_yellow_mean": 0.0,
            "yellow_signal": 0.0,
            "mean_yellow_h": 0.0,
            "mean_yellow_s": 0.0,
            "mean_yellow_v": 0.0,
        }

    focus_bool = focus_mask > 0
    yellow_focus = focus_bool & (h >= 10) & (h <= 50) & (v >= 60) & ((s >= 16) | (b_channel >= 145))
    pale_focus = focus_bool & (s <= 48) & (v >= 160)

    yellow_count = float(np.count_nonzero(yellow_focus))
    yellow_focus_ratio = yellow_count / max(1.0, focus_area)
    pale_focus_ratio = float(np.count_nonzero(pale_focus)) / max(1.0, focus_area)

    focus_b = b_channel[focus_bool].astype(np.float32)
    lab_yellow_mean = 0.0
    if focus_b.size > 0:
        lab_yellow_mean = float(np.mean(np.clip((focus_b - 128.0) / 127.0, 0.0, 1.0)))
    yellow_signal = yellow_focus_ratio * 0.70 + lab_yellow_mean * 0.30

    mean_yellow_h = 0.0
    mean_yellow_s = 0.0
    mean_yellow_v = 0.0
    if yellow_count >= 1.0:
        mean_yellow_h = float(np.mean(h[yellow_focus]))
        mean_yellow_s = float(np.mean(s[yellow_focus]))
        mean_yellow_v = float(np.mean(v[yellow_focus]))

    return {
        "focus_area_ratio": float(focus_area_ratio),
        "yellow_focus_ratio": float(yellow_focus_ratio),
        "pale_focus_ratio": float(pale_focus_ratio),
        "lab_yellow_mean": float(lab_yellow_mean),
        "yellow_signal": float(yellow_signal),
        "mean_yellow_h": float(mean_yellow_h),
        "mean_yellow_s": float(mean_yellow_s),
        "mean_yellow_v": float(mean_yellow_v),
    }


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


def merge_top_candidates(
    before_data: Dict[str, Any],
    after_data: Dict[str, Any],
    visual_profile: Optional[Dict[str, Any]] = None,
    is_denim_context: bool = False,
    is_dark_fabric_context: bool = False,
) -> List[Dict[str, Any]]:
    score_map: Dict[str, float] = {}

    def add_items(items: List[Dict[str, Any]], weight: float) -> None:
        for item in items:
            label = item["source"]
            score_map[label] = score_map.get(label, 0.0) + float(item.get("confidence", 0.0)) * weight

    score_map[before_data["source"]] = score_map.get(before_data["source"], 0.0) + float(before_data["confidence"]) * 0.85
    score_map[after_data["source"]] = score_map.get(after_data["source"], 0.0) + float(after_data["confidence"]) * 1.25
    add_items(before_data.get("top3", []), 0.20)
    add_items(after_data.get("top3", []), 0.38)

    raw_score_map = dict(score_map)
    before_scores = _flatten_candidate_sources(before_data)
    after_scores = _flatten_candidate_sources(after_data)
    anchor_label = before_data["source"] if float(before_data["confidence"]) > float(after_data["confidence"]) else after_data["source"]

    visual_family = str((visual_profile or {}).get("family", "unknown"))
    visual_strength = float((visual_profile or {}).get("score", 0.0))
    visual_margin = float((visual_profile or {}).get("margin_to_next", 0.0))
    visual_scores = (visual_profile or {}).get("scores", {}) or {}
    visual_pale = float(visual_scores.get("pale", 0.0))
    visual_chroma = float(visual_scores.get("red", 0.0)) + float(visual_scores.get("brown", 0.0)) + float(visual_scores.get("yellow", 0.0))
    if visual_family in FAMILY_SOURCE_SETS and visual_strength >= 0.18 and visual_margin >= 0.05:
        same_family = FAMILY_SOURCE_SETS[visual_family]
        if visual_family in {"brown", "yellow"}:
            family_mul = 1.10
            boost = min(0.16, max(0.05, visual_strength * 0.12))
            other_mul = 0.86
        elif visual_family == "red":
            if is_denim_context or is_dark_fabric_context:
                family_mul = 1.08
                boost = min(0.08, max(0.01, visual_strength * 0.07))
                other_mul = 0.93
            else:
                family_mul = 1.03
                boost = min(0.03, max(0.0, visual_strength * 0.03))
                other_mul = 0.95
        elif visual_family == "pale":
            if is_dark_fabric_context and visual_chroma >= 0.10:
                family_mul = 1.00
                boost = 0.0
                other_mul = 1.00
            elif visual_pale >= 0.62 and visual_chroma <= 0.16:
                family_mul = 1.12
                boost = min(0.14, max(0.04, visual_strength * 0.10))
                other_mul = 0.86
            elif visual_pale >= 0.50 and visual_chroma <= 0.24:
                family_mul = 1.05
                boost = min(0.06, max(0.01, visual_strength * 0.05))
                other_mul = 0.95
            else:
                family_mul = 1.00
                boost = 0.0
                other_mul = 1.00
        else:
            family_mul = 1.06
            boost = min(0.08, max(0.02, visual_strength * 0.07))
            other_mul = 0.92
        for label in same_family:
            score_map[label] = score_map.get(label, 0.0) * family_mul + boost
        for label in list(score_map.keys()):
            family = SOURCE_FAMILY.get(label, "unknown")
            if family != visual_family:
                score_map[label] = score_map[label] * other_mul

    provisional_rank = sorted(score_map.items(), key=lambda x: x[1], reverse=True)
    top3_families = {
        SOURCE_FAMILY.get(label, "unknown")
        for label, _ in provisional_rank[:3]
        if label in SOURCE_FAMILY
    }
    if "brown" in top3_families and "yellow" in top3_families:
        for label in list(score_map.keys()):
            family = SOURCE_FAMILY.get(label, "unknown")
            if family == "brown":
                if is_dark_fabric_context:
                    score_map[label] = score_map[label] * 1.10 + 0.020
                else:
                    score_map[label] = score_map[label] * 1.16 + 0.030
            elif family == "yellow":
                if is_dark_fabric_context:
                    score_map[label] = score_map[label] * 1.12 + 0.024
                else:
                    score_map[label] = score_map[label] * 1.09 + 0.020
            elif family == "red":
                if is_dark_fabric_context:
                    score_map[label] = score_map[label] * 0.84
                else:
                    score_map[label] = score_map[label] * 0.76
            elif family == "pale":
                score_map[label] = score_map[label] * 0.92
    elif "brown" in top3_families and "red" in top3_families:
        for label in list(score_map.keys()):
            family = SOURCE_FAMILY.get(label, "unknown")
            if family == "brown":
                if is_denim_context or is_dark_fabric_context:
                    score_map[label] = score_map[label] * 1.07 + 0.016
                else:
                    score_map[label] = score_map[label] * 1.15 + 0.028
            elif family == "red":
                if is_denim_context or is_dark_fabric_context:
                    score_map[label] = score_map[label] * 1.13 + 0.022
                else:
                    score_map[label] = score_map[label] * 1.07 + 0.014
            elif family == "yellow":
                if is_denim_context or is_dark_fabric_context:
                    score_map[label] = score_map[label] * 0.84
                else:
                    score_map[label] = score_map[label] * 0.79
            elif family == "pale":
                score_map[label] = score_map[label] * 0.92

    if "brown" in top3_families and ("yellow" in top3_families or "red" in top3_families):
        brown_tail_boost = 1.04
        if is_dark_fabric_context:
            brown_tail_boost = 1.02
        if "red" in top3_families and (is_denim_context or is_dark_fabric_context):
            brown_tail_boost = 1.01
        for label in list(score_map.keys()):
            if SOURCE_FAMILY.get(label, "unknown") == "brown":
                score_map[label] = score_map[label] * brown_tail_boost

    def candidate_sort_key(item: Tuple[str, float]) -> Tuple[float, float, float]:
        label, score = item
        family = SOURCE_FAMILY.get(label, "unknown")
        if label == anchor_label:
            return (4.0, 0.0, score)
        if visual_family != "unknown" and family == visual_family:
            return (2.0, 0.0, score)
        return (1.0, 0.0, score)

    ranked = sorted(score_map.items(), key=candidate_sort_key, reverse=True)
    top_scale = max([float(v) for _, v in ranked], default=0.0)
    if top_scale <= 0:
        top_scale = 1.0
    result = []
    for label, score in ranked[:5]:
        b = before_scores.get(label)
        a = after_scores.get(label)
        vals = [v for v in (b, a) if v is not None]
        cnn_avg = float(sum(vals) / len(vals)) if vals else 0.0
        score_norm = float(score) / float(top_scale)
        result.append({
            "label": label,
            "label_kr": SOURCE_DISPLAY_KO.get(label, label),
            "family": SOURCE_FAMILY.get(label, "unknown"),
            "family_kr": FAMILY_DISPLAY_KO.get(SOURCE_FAMILY.get(label, "unknown"), SOURCE_FAMILY.get(label, "unknown")),
            "score": min(0.999, max(0.0, score_norm)),
            "raw_score": max(0.0, raw_score_map.get(label, 0.0)),
            "cnn_score": min(0.999, max(0.0, cnn_avg)),
            "before_score": b,
            "after_score": a,
            "color_adjusted": visual_family in FAMILY_SOURCE_SETS and SOURCE_FAMILY.get(label) == visual_family,
        })
    result.sort(key=lambda x: x["score"], reverse=True)
    return result


def align_top_candidates_with_final(final_label: str, top_candidates: List[Dict[str, Any]], before_data: Dict[str, Any], after_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    before_scores = _flatten_candidate_sources(before_data)
    after_scores = _flatten_candidate_sources(after_data)
    score_map = _flatten_candidate_sources(before_data, after_data)
    for cand in top_candidates or []:
        label = str(cand.get("label", "")).strip()
        if label:
            score_map[label] = max(score_map.get(label, 0.0), float(cand.get("score", 0.0)))
            if cand.get("before_score") is not None:
                before_scores[label] = max(before_scores.get(label, 0.0), float(cand.get("before_score", 0.0)))
            if cand.get("after_score") is not None:
                after_scores[label] = max(after_scores.get(label, 0.0), float(cand.get("after_score", 0.0)))

    final_score = max(score_map.get(final_label, 0.0), float(before_data.get("confidence", 0.0)) if before_data.get("source") == final_label else 0.0, float(after_data.get("confidence", 0.0)) if after_data.get("source") == final_label else 0.0)
    if final_score <= 0:
        final_score = 0.86

    final_family = SOURCE_FAMILY.get(final_label, "unknown")
    final_detergent = SOURCE_INFO.get(final_label, {}).get("detergent")

    evidence_labels = {
        label for label, value in score_map.items()
        if label in SOURCE_INFO and float(value) > 0
    }
    evidence_labels.add(final_label)

    def rank_key(label: str) -> Tuple[float, float, float]:
        if label == final_label:
            return (6.0, 0.0, final_score)
        family_bonus = 1.2 if SOURCE_FAMILY.get(label) == final_family else 0.0
        detergent_bonus = 0.5 if SOURCE_INFO.get(label, {}).get("detergent") == final_detergent else 0.0
        model_score = float(score_map.get(label, 0.0))
        return (2.0 + family_bonus + detergent_bonus, model_score, model_score)

    ordered = sorted(evidence_labels, key=rank_key, reverse=True)

    result: List[Dict[str, Any]] = []
    for label in ordered:
        model_score = max(0.0, float(score_map.get(label, 0.0)))
        if label == final_label:
            score = final_score
        else:
            relation_gain = 0.0
            if SOURCE_FAMILY.get(label) == final_family:
                relation_gain += final_score * 0.12
            if SOURCE_INFO.get(label, {}).get("detergent") == final_detergent:
                relation_gain += final_score * 0.07
            score = max(model_score, min(0.98, model_score + relation_gain))

        result.append({
            "label": label,
            "label_kr": SOURCE_DISPLAY_KO.get(label, label),
            "family": SOURCE_FAMILY.get(label, "unknown"),
            "family_kr": FAMILY_DISPLAY_KO.get(SOURCE_FAMILY.get(label, "unknown"), SOURCE_FAMILY.get(label, "unknown")),
            "score": min(0.999, max(0.0, score)),
            "raw_score": model_score,
            "before_score": before_scores.get(label),
            "after_score": after_scores.get(label),
            "color_adjusted": label != final_label and SOURCE_FAMILY.get(label) == final_family,
        })
        if len(result) >= 3:
            break

    return result


def build_family_summary(top_candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    family_map: Dict[str, Dict[str, Any]] = {}
    for cand in top_candidates:
        family = str(cand.get("family", "unknown"))
        bucket = family_map.setdefault(family, {
            "family": family,
            "family_kr": FAMILY_DISPLAY_KO.get(family, family),
            "score": 0.0,
            "members": [],
        })
        score = float(cand.get("score", 0.0))
        bucket["score"] += score
        bucket["members"].append({
            "label": cand.get("label"),
            "label_kr": cand.get("label_kr"),
            "score": score,
        })

    ranked = sorted(family_map.values(), key=lambda x: x["score"], reverse=True)
    for idx, item in enumerate(ranked):
        next_score = float(ranked[idx + 1]["score"]) if idx + 1 < len(ranked) else 0.0
        item["margin_to_next"] = max(0.0, float(item["score"]) - next_score)
    return ranked


def build_visual_evidence(visual_profile: Dict[str, Any], top_candidates: List[Dict[str, Any]]) -> List[str]:
    family = str((visual_profile or {}).get("family", "unknown"))
    family_kr = FAMILY_DISPLAY_KO.get(family, family)
    score = float((visual_profile or {}).get("score", 0.0))
    margin = float((visual_profile or {}).get("margin_to_next", 0.0))
    evidence: List[str] = []
    family_text = family_kr if str(family_kr).endswith("계열") else f"{family_kr} 계열"
    if family != "unknown":
        if score < 0.38 or margin < 0.06:
            evidence.append(
                f"이미지 색상은 {family_text} 신호가 일부 보였지만 단정하기엔 약해, "
                f"계열 신호 {score * 100:.1f}% / 다음 계열과 {margin * 100:.1f}%p 차이로 기록했습니다."
            )
        else:
            evidence.append(
                f"이미지 색상은 {family_text}로 감지되었습니다. "
                f"계열 신호 {score * 100:.1f}%, 다음 계열과 {margin * 100:.1f}%p 차이입니다."
            )
        same_family_names = [
            str(c.get("label_kr", c.get("label")))
            for c in top_candidates[:3]
            if c.get("family") == family
        ]
        if same_family_names:
            evidence.append(f"TOP 후보 표시에서 같은 색상 계열 후보({', '.join(same_family_names)})를 우선 반영했습니다.")
    else:
        evidence.append("색상 계열 신호가 충분히 뚜렷하지 않아 CNN 후보를 중심으로 표시했습니다.")
    return evidence


def build_decision_summary(final_label: str, decision_source: str, visual_profile: Dict[str, Any]) -> str:
    final_kr = SOURCE_DISPLAY_KO.get(final_label, final_label)
    family = str((visual_profile or {}).get("family", "unknown"))
    family_kr = FAMILY_DISPLAY_KO.get(family, family)
    family_text = family_kr if str(family_kr).endswith("계열") else f"{family_kr} 계열"
    if decision_source == "family_combo_adjusted":
        return f"최종 라벨은 {final_kr}이며, TOP1~3 계열 조합(노랑·갈색 / 빨강·갈색) 기반 보정을 적용했습니다."
    if decision_source == "visual_pale_override":
        return f"최종 라벨은 {final_kr}이며, 이미지의 저채도·고밝기(흰색 계열) 신호가 강해 커피 편향을 방지하는 보수 보정을 적용했습니다."
    if decision_source == "pale_yellow_recheck":
        return f"최종 라벨은 {final_kr}이며, 흰색 계열 판단 구간에서 노랑 신호를 별도 재검토해 노랑/흰색 후보를 다시 비교했습니다."
    if family != "unknown":
        return f"최종 라벨은 {final_kr}이며, TOP 후보 표시는 {family_text} 색상 보정을 함께 반영했습니다."
    return f"최종 라벨은 {final_kr}이며, 색상 계열이 불명확해 CNN 후보 중심으로 표시했습니다."


def _reaction_absorption_text(reaction: Dict[str, float]) -> str:
    rgb = float(reaction.get("rgb_diff_mean", 0.0))
    sat = float(reaction.get("sat_diff_mean", 0.0))
    val = float(reaction.get("val_diff_mean", 0.0))
    if rgb >= 28 or sat >= 34:
        return "1차·2차 이미지의 RGB/HSV 변화량이 커서 물을 뿌린 뒤 색 분포가 많이 달라진 것으로 기록되었습니다"
    if val >= 22:
        return "1차·2차 이미지의 밝기 변화가 있어 물을 뿌린 뒤 얼룩 농도가 일부 달라진 것으로 기록되었습니다"
    return "1차·2차 이미지의 변화량이 작아 물을 뿌린 뒤에도 자국 위치와 경계가 비교적 유지된 것으로 기록되었습니다"


def should_use_openai(before_data: Dict[str, Any], after_data: Dict[str, Any], visual_profile: Optional[Dict[str, Any]] = None) -> bool:
    before_label = before_data["source"]
    after_label = after_data["source"]
    before_conf = float(before_data["confidence"])
    after_conf = float(after_data["confidence"])
    visual_family = str((visual_profile or {}).get("family", "unknown"))
    visual_score = float((visual_profile or {}).get("score", 0.0))
    visual_margin = float((visual_profile or {}).get("margin_to_next", 0.0))

    if before_label == after_label:
        if (
            visual_family != "unknown"
            and SOURCE_FAMILY.get(before_label, "unknown") != visual_family
            and visual_score >= 0.20
            and visual_margin >= 0.05
        ):
            return True
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


def apply_pale_visual_override(
    final_label: str,
    pair_conf: float,
    visual_profile: Optional[Dict[str, Any]],
    before_pred: Dict[str, Any],
    after_pred: Dict[str, Any],
    is_dark_fabric_context: bool = False,
) -> Tuple[str, float, Optional[str]]:
    profile = visual_profile or {}
    family = str(profile.get("family", "unknown"))
    scores = profile.get("scores", {}) or {}
    pale_score = float(scores.get("pale", 0.0))
    chroma_sum = float(scores.get("red", 0.0)) + float(scores.get("brown", 0.0)) + float(scores.get("yellow", 0.0))
    visual_score = float(profile.get("score", 0.0))
    visual_margin = float(profile.get("margin_to_next", 0.0))

    visual_yellow = float(scores.get("yellow", 0.0))
    if is_dark_fabric_context:
        # 핵심: 어두운 배경에서는 노랑 신호가 조금이라도 있으면 pale 강제 전환을 막는다.
        if visual_yellow >= 0.06 and visual_yellow >= pale_score * 0.22:
            return final_label, pair_conf, None
        if pale_score < 0.52:
            return final_label, pair_conf, None
        if chroma_sum > 0.22:
            return final_label, pair_conf, None
        if visual_yellow >= 0.09:
            return final_label, pair_conf, None
    else:
        if pale_score < 0.42:
            return final_label, pair_conf, None
        if chroma_sum > 0.28:
            return final_label, pair_conf, None
    if family not in {"pale", "unknown"}:
        return final_label, pair_conf, None
    if family == "unknown" and visual_score < (0.36 if is_dark_fabric_context else 0.30):
        return final_label, pair_conf, None
    if visual_margin < 0.03:
        return final_label, pair_conf, None

    if SOURCE_FAMILY.get(final_label, "unknown") == "pale":
        return final_label, pair_conf, None

    before_label = _normalize_feedback_label(before_pred.get("source"))
    after_label = _normalize_feedback_label(after_pred.get("source"))
    before_conf = float(before_pred.get("confidence", 0.0))
    after_conf = float(after_pred.get("confidence", 0.0))

    if (
        before_label == after_label
        and SOURCE_FAMILY.get(before_label, "unknown") == "brown"
        and min(before_conf, after_conf) >= 0.93
    ):
        return final_label, pair_conf, None

    pale_candidates = ["milk", "mayonnaise"]
    best_label = max(pale_candidates, key=lambda lbl: _current_prediction_support(lbl, before_pred, after_pred))
    best_support = _current_prediction_support(best_label, before_pred, after_pred)
    if best_support < 0.10 and pale_score < 0.62:
        return final_label, pair_conf, None
    if best_support < 0.10:
        best_label = "milk"

    adjusted_conf = max(0.58, min(pair_conf, 0.78))
    reason = (
        f"이미지에서 저채도·고밝기(흰색 계열) 비중이 높고 색 경계가 약해, "
        f"{SOURCE_DISPLAY_KO.get(final_label, final_label)} 대신 {SOURCE_DISPLAY_KO.get(best_label, best_label)} 계열로 보수 보정했습니다."
    )
    return best_label, adjusted_conf, reason


def apply_pale_yellow_recheck(
    final_label: str,
    pair_conf: float,
    visual_profile: Optional[Dict[str, Any]],
    before_pred: Dict[str, Any],
    after_pred: Dict[str, Any],
    before_img_raw: np.ndarray,
    after_img_raw: np.ndarray,
    is_dark_fabric_context: bool = False,
) -> Tuple[str, float, Optional[str]]:
    if not PALE_YELLOW_RECHECK_ENABLED:
        return final_label, pair_conf, None
    if SOURCE_FAMILY.get(final_label, "unknown") != "pale":
        return final_label, pair_conf, None

    before_label = _normalize_feedback_label(before_pred.get("source"))
    after_label = _normalize_feedback_label(after_pred.get("source"))
    before_conf = float(before_pred.get("confidence", 0.0))
    after_conf = float(after_pred.get("confidence", 0.0))
    if (
        before_label == after_label
        and SOURCE_FAMILY.get(before_label, "unknown") == "brown"
        and min(before_conf, after_conf) >= 0.90
    ):
        return final_label, pair_conf, None

    metrics_before = _yellow_focus_metrics_for_image(before_img_raw)
    metrics_after = _yellow_focus_metrics_for_image(after_img_raw)
    focus_area_ratio = float(metrics_before.get("focus_area_ratio", 0.0)) * 0.40 + float(metrics_after.get("focus_area_ratio", 0.0)) * 0.60
    min_focus_ratio = 0.010 if is_dark_fabric_context else 0.020
    visual_scores = (visual_profile or {}).get("scores", {}) or {}
    visual_yellow = float(visual_scores.get("yellow", 0.0))
    visual_brown = float(visual_scores.get("brown", 0.0))
    visual_pale = float(visual_scores.get("pale", 0.0))
    if focus_area_ratio < min_focus_ratio:
        if not (is_dark_fabric_context and visual_yellow >= 0.07 and visual_yellow >= visual_pale * 0.55):
            return final_label, pair_conf, None

    yellow_signal = float(metrics_before.get("yellow_signal", 0.0)) * 0.35 + float(metrics_after.get("yellow_signal", 0.0)) * 0.65
    pale_signal = float(metrics_before.get("pale_focus_ratio", 0.0)) * 0.40 + float(metrics_after.get("pale_focus_ratio", 0.0)) * 0.60

    blended_yellow = (
        yellow_signal * 0.72
        + max(0.0, visual_yellow - 0.05) * 0.22
        + max(0.0, visual_brown - 0.06) * 0.06
    )
    if is_dark_fabric_context:
        blended_yellow = max(blended_yellow, yellow_signal * 0.86 + max(0.0, visual_yellow) * 0.14)
    yellow_margin = blended_yellow - (pale_signal * (0.58 if is_dark_fabric_context else 1.0))
    min_signal = 0.22 if is_dark_fabric_context else PALE_YELLOW_RECHECK_MIN_SIGNAL
    min_margin = -0.01 if is_dark_fabric_context else PALE_YELLOW_RECHECK_MARGIN
    if blended_yellow < min_signal:
        return final_label, pair_conf, None
    if yellow_margin < min_margin:
        return final_label, pair_conf, None
    if not is_dark_fabric_context and visual_pale >= 0.62 and visual_yellow <= 0.08 and yellow_signal < 0.22:
        return final_label, pair_conf, None

    mean_h = float(metrics_before.get("mean_yellow_h", 0.0)) * 0.35 + float(metrics_after.get("mean_yellow_h", 0.0)) * 0.65
    mean_s = float(metrics_before.get("mean_yellow_s", 0.0)) * 0.35 + float(metrics_after.get("mean_yellow_s", 0.0)) * 0.65
    mean_v = float(metrics_before.get("mean_yellow_v", 0.0)) * 0.35 + float(metrics_after.get("mean_yellow_v", 0.0)) * 0.65

    heuristics = {"curry": 0.0, "mustard": 0.0, "oil": 0.0}
    if mean_s <= 54:
        heuristics["oil"] += 0.16
    if mean_h >= 30 and mean_v >= 115:
        heuristics["mustard"] += 0.14
    if mean_h < 30 and mean_s >= 52:
        heuristics["curry"] += 0.14
    if mean_s >= 72:
        heuristics["curry"] += 0.06
        heuristics["mustard"] += 0.04

    yellow_labels = ["curry", "mustard", "oil"]
    scored_candidates: Dict[str, float] = {}
    for label in yellow_labels:
        model_support = _current_prediction_support(label, before_pred, after_pred)
        scored_candidates[label] = float(model_support) + float(heuristics.get(label, 0.0))
    best_label = max(yellow_labels, key=lambda lbl: scored_candidates.get(lbl, 0.0))
    best_score = float(scored_candidates.get(best_label, 0.0))
    second_score = max([scored_candidates.get(lbl, 0.0) for lbl in yellow_labels if lbl != best_label], default=0.0)
    if best_score < 0.08 and (best_score - second_score) < 0.02 and blended_yellow < 0.48:
        return final_label, pair_conf, None

    adjusted_conf = max(0.56, min(0.84, max(pair_conf, 0.60 + min(0.18, blended_yellow * 0.24))))
    context_prefix = "어두운 원단에서 자동 노출로 흰색 계열처럼 보였던 장면을" if is_dark_fabric_context else "흰색 계열로 보였던 장면을"
    reason = (
        f"{context_prefix} 노랑/흰색 기준으로 재검토한 결과, "
        f"노랑 신호({blended_yellow * 100:.1f}%)가 흰색 신호({pale_signal * 100:.1f}%)보다 높아 "
        f"{SOURCE_DISPLAY_KO.get(best_label, best_label)} 후보를 우선했습니다."
    )
    return best_label, adjusted_conf, reason


def apply_family_combo_override(
    final_label: str,
    pair_conf: float,
    top_candidates: List[Dict[str, Any]],
    visual_profile: Optional[Dict[str, Any]],
    before_pred: Dict[str, Any],
    after_pred: Dict[str, Any],
    is_denim_context: bool = False,
    is_dark_fabric_context: bool = False,
) -> Tuple[str, float, Optional[str]]:
    top3 = (top_candidates or [])[:3]
    family_set = {
        str(c.get("family", "unknown"))
        for c in top3
        if c.get("family")
    }
    def best_by_families(families: set, brown_boost: float = 1.0) -> Tuple[Optional[str], float]:
        best_label = None
        best_score = -1.0
        for cand in top3:
            label = str(cand.get("label", "")).strip()
            if not label:
                continue
            family = SOURCE_FAMILY.get(label, "unknown")
            if family not in families:
                continue
            score = float(cand.get("score", 0.0))
            if family == "brown":
                score *= brown_boost
            if score > best_score:
                best_score = score
                best_label = label
        return best_label, best_score
    def score_of(label_name: str) -> float:
        for cand in top3:
            if str(cand.get("label", "")).strip() == label_name:
                return float(cand.get("score", 0.0))
        return 0.0

    before_label = _normalize_feedback_label(before_pred.get("source"))
    after_label = _normalize_feedback_label(after_pred.get("source"))
    before_conf = float(before_pred.get("confidence", 0.0))
    after_conf = float(after_pred.get("confidence", 0.0))
    max_conf = max(before_conf, after_conf)

    if before_label == after_label == final_label and max_conf >= 0.93:
        return final_label, pair_conf, None

    visual_family = str((visual_profile or {}).get("family", "unknown"))
    visual_scores = (visual_profile or {}).get("scores", {}) or {}
    visual_red = float(visual_scores.get("red", 0.0))
    visual_brown = float(visual_scores.get("brown", 0.0))
    visual_yellow = float(visual_scores.get("yellow", 0.0))
    visual_pale = float(visual_scores.get("pale", 0.0))
    final_family = SOURCE_FAMILY.get(final_label, "unknown")
    allow_denim_red_recheck = bool(
        (is_denim_context or is_dark_fabric_context)
        and final_family == "brown"
        and "brown" in family_set
        and "red" in family_set
    )

    if max_conf >= FAMILY_COMBO_OVERRIDE_MAX_CONF and not allow_denim_red_recheck:
        return final_label, pair_conf, None

    if final_family == "pale":
        brown_label, brown_score = best_by_families({"brown"}, brown_boost=1.10)
        pale_label, pale_score = best_by_families({"pale"}, brown_boost=1.0)
        if brown_label and max_conf >= 0.34 and brown_score >= max(0.30, pale_score * 0.95):
            reason = (
                f"흰색 계열 후보가 있었지만 갈색 후보 근거가 더 강해 "
                f"{SOURCE_DISPLAY_KO.get(final_label, final_label)} 대신 {SOURCE_DISPLAY_KO.get(brown_label, brown_label)}를 우선했습니다."
            )
            return brown_label, max(0.56, min(pair_conf, 0.82)), reason

    if "pale" in family_set:
        pale_label, pale_score = best_by_families({"pale"}, brown_boost=1.0)
        top1_score = float(top3[0].get("score", 0.0)) if top3 else 0.0
        # 노랑 신호가 살아있는 어두운 배경에서는 pale 후보로 덮어쓰지 않는다.
        if is_dark_fabric_context and visual_yellow >= max(0.08, visual_pale * 0.55):
            pale_label = None
        if pale_label and max_conf <= 0.58 and visual_family in {"pale", "unknown"} and pale_score >= top1_score * 0.55:
            reason = (
                f"TOP 후보에 흰색 계열({SOURCE_DISPLAY_KO.get(pale_label, pale_label)})이 포함되고 색 경계가 약해 "
                f"{SOURCE_DISPLAY_KO.get(final_label, final_label)} 대신 {SOURCE_DISPLAY_KO.get(pale_label, pale_label)}를 우선했습니다."
            )
            return pale_label, max(0.54, min(pair_conf, 0.78)), reason

    if "brown" not in family_set:
        return final_label, pair_conf, None

    if "yellow" in family_set and "brown" in family_set:
        orient_score = score_of("oriental_dressing")
        mustard_score = score_of("mustard")
        if (
            orient_score > 0
            and mustard_score > 0
            and final_label == "mustard"
            and orient_score >= mustard_score * max(0.92, FAMILY_COMBO_OVERRIDE_NEAR_RATIO)
            and max_conf <= 0.60
            and visual_family in {"yellow", "brown", "unknown"}
        ):
            reason = "노랑·갈색 혼합 패턴에서 오리엔탈 소스의 물반응(노란 기운) 특성을 반영해 머스타드보다 오리엔탈 소스를 우선했습니다."
            return "oriental_dressing", max(0.56, min(pair_conf, 0.84)), reason

        target_families = {"yellow", "brown"}
        target_label, target_score = best_by_families(target_families, brown_boost=1.10)
        if (
            target_label
            and SOURCE_FAMILY.get(target_label, "unknown") == "brown"
            and final_family not in target_families
            and target_score >= FAMILY_COMBO_OVERRIDE_MIN_SCORE
            and max_conf <= 0.52
            and visual_family in {"yellow", "brown", "unknown"}
        ):
            reason = (
                f"TOP1~3 후보가 노랑·갈색 계열에 집중되어 "
                f"{SOURCE_DISPLAY_KO.get(final_label, final_label)} 대신 {SOURCE_DISPLAY_KO.get(target_label, target_label)}로 보정했습니다."
            )
            return target_label, max(0.56, min(pair_conf, 0.84)), reason
        if final_family == "yellow":
            brown_label, brown_score = best_by_families({"brown"}, brown_boost=1.12)
            yellow_label, yellow_score = best_by_families({"yellow"}, brown_boost=1.0)
            if (
                brown_label
                and yellow_label
                and brown_score >= yellow_score * FAMILY_COMBO_OVERRIDE_NEAR_RATIO
                and max_conf <= 0.62
                and visual_family in {"brown", "unknown"}
            ):
                reason = (
                    f"노랑·갈색 혼합 후보에서 갈색 후보 근거가 근접해 "
                    f"{SOURCE_DISPLAY_KO.get(yellow_label, yellow_label)} 대신 {SOURCE_DISPLAY_KO.get(brown_label, brown_label)}를 우선했습니다."
                )
                return brown_label, max(0.56, min(pair_conf, 0.82)), reason

    if "red" in family_set and "brown" in family_set:
        yellow_label, yellow_score = best_by_families({"yellow"}, brown_boost=1.0)
        red_label, red_score = best_by_families({"red"}, brown_boost=1.0)
        target_families = {"red", "brown"}
        brown_boost_for_red_mix = 1.04 if (is_denim_context or is_dark_fabric_context) else 1.10
        target_label, target_score = best_by_families(target_families, brown_boost=brown_boost_for_red_mix)
        if (
            target_label
            and final_family not in target_families
            and target_score >= FAMILY_COMBO_OVERRIDE_MIN_SCORE
            and max_conf <= 0.52
            and visual_family in {"red", "brown", "unknown"}
        ):
            reason = (
                f"TOP1~3 후보가 빨강·갈색 계열에 집중되어 "
                f"{SOURCE_DISPLAY_KO.get(final_label, final_label)} 대신 {SOURCE_DISPLAY_KO.get(target_label, target_label)}로 보정했습니다."
            )
            return target_label, max(0.56, min(pair_conf, 0.84)), reason
        if final_family == "brown" and (is_denim_context or is_dark_fabric_context):
            top1_label = str(top3[0].get("label", "")).strip() if top3 else ""
            top1_family = SOURCE_FAMILY.get(top1_label, "unknown")
            red_in_top23 = any(
                idx >= 1 and SOURCE_FAMILY.get(str(c.get("label", "")).strip(), "unknown") == "red"
                for idx, c in enumerate(top3[:3])
            )
            if top1_label and top1_family == "brown" and red_label and red_in_top23:
                brown_top_score = score_of(top1_label)
                red_support = _current_prediction_support(red_label, before_pred, after_pred)
                brown_support = _current_prediction_support(top1_label, before_pred, after_pred)
                score_ratio = DENIM_RED_RECHECK_SCORE_RATIO if is_denim_context else min(0.90, DENIM_RED_RECHECK_SCORE_RATIO + 0.05)
                support_ratio = 0.85 if is_denim_context else 0.88
                visual_ratio = 0.66 if is_denim_context else 0.70
                visual_floor = 0.12 if is_denim_context else 0.14
                max_conf_thr = DENIM_RED_RECHECK_MAX_CONF + (0.03 if is_dark_fabric_context and not is_denim_context else 0.0)
                red_close_enough = red_score >= brown_top_score * score_ratio
                support_close = red_support >= brown_support * support_ratio
                visual_support = visual_red >= max(visual_floor, visual_brown * visual_ratio)
                if max_conf <= max_conf_thr and red_close_enough and (support_close or visual_support):
                    if is_denim_context and is_dark_fabric_context:
                        context_name = "데님/어두운 원단"
                    elif is_denim_context:
                        context_name = "데님"
                    else:
                        context_name = "어두운 원단"
                    reason = (
                        f"{context_name} 기준에서 TOP1 갈색 후보와 함께 TOP2/3 빨강 후보가 근접하게 검출되어, "
                        f"{SOURCE_DISPLAY_KO.get(top1_label, top1_label)} 대신 {SOURCE_DISPLAY_KO.get(red_label, red_label)}를 재검토 우선했습니다."
                    )
                    return red_label, max(0.56, min(pair_conf, 0.82)), reason
        if final_family == "red":
            brown_label, brown_score = best_by_families({"brown"}, brown_boost=1.12)
            red_label, red_score = best_by_families({"red"}, brown_boost=1.0)
            brown_near_ratio = FAMILY_COMBO_OVERRIDE_NEAR_RATIO
            if is_denim_context or is_dark_fabric_context:
                brown_near_ratio = max(1.02, brown_near_ratio + 0.06)
            prefer_red_under_dark = (
                (is_denim_context or is_dark_fabric_context)
                and visual_red >= max(0.12, visual_brown * 0.82)
            )
            if (
                brown_label
                and red_label
                and not prefer_red_under_dark
                and brown_score >= red_score * brown_near_ratio
                and max_conf <= 0.62
                and visual_family in {"brown", "unknown"}
            ):
                reason = (
                    f"빨강·갈색 혼합 후보에서 갈색 후보 근거가 근접해 "
                    f"{SOURCE_DISPLAY_KO.get(red_label, red_label)} 대신 {SOURCE_DISPLAY_KO.get(brown_label, brown_label)}를 우선했습니다."
                )
                return brown_label, max(0.56, min(pair_conf, 0.82)), reason

    return final_label, pair_conf, None


def build_pair_reasons(final_label: str, before_pred: Dict[str, Any], after_pred: Dict[str, Any], material_group: str, top_candidates: List[Dict[str, Any]], reaction: Dict[str, float], decision_source: str, visual_profile: Optional[Dict[str, Any]] = None) -> List[str]:
    final_src = SOURCE_DISPLAY_KO.get(final_label, final_label)
    profile = STAIN_SENSORY_PROFILE.get(final_label, {})
    visual_family = str((visual_profile or {}).get("family", "unknown"))
    visual_family_kr = FAMILY_DISPLAY_KO.get(visual_family, "판단 어려움")
    visual_score = float((visual_profile or {}).get("score", 0.0))
    visual_margin = float((visual_profile or {}).get("margin_to_next", 0.0))
    visual_family_text = visual_family_kr if str(visual_family_kr).endswith("계열") else f"{visual_family_kr} 계열"
    color_text = profile.get("color", f"{visual_family_text} 색감")
    reaction_text = _reaction_absorption_text(reaction)
    detergent_key = SOURCE_INFO.get(final_label, {}).get("detergent")
    detergent_text = DETERGENT_LABEL_KO.get(detergent_key, detergent_key or "추천 세제")
    final_family = SOURCE_FAMILY.get(final_label, "unknown")
    same_family_candidate_names = [
        SOURCE_DISPLAY_KO.get(str(c.get("label", "")), str(c.get("label", "")))
        for c in (top_candidates or [])[:3]
        if c.get("label") and SOURCE_FAMILY.get(str(c.get("label", ""))) == final_family
    ]
    candidate_names = [
        SOURCE_DISPLAY_KO.get(str(c.get("label", "")), str(c.get("label", "")))
        for c in (top_candidates or [])[:3]
        if c.get("label")
    ]
    candidate_text = ", ".join(candidate_names) if candidate_names else "상위 후보"
    same_family_text = ", ".join(same_family_candidate_names) if same_family_candidate_names else ""

    reasons = []
    if visual_family != "unknown":
        if visual_score < 0.38 or visual_margin < 0.06:
            reasons.append(
                f"이미지 색상 분포는 {visual_family_text} 신호가 일부 보였지만 단정 구간은 아니어서, "
                f"{final_src}의 일반 색상 특징인 {color_text}과 후보 흐름을 함께 비교했습니다."
            )
        else:
            reasons.append(
                f"이미지 색상 분포는 {visual_family_text}로 감지되어, "
                f"{final_src}의 일반 색상 특징인 {color_text}과 비교했습니다."
            )
    else:
        reasons.append(f"색상 계열이 뚜렷하게 고정되지는 않아, {final_src}의 일반 색상 특징인 {color_text}과 후보 흐름을 함께 비교했습니다.")
    if same_family_text:
        reasons.append(f"{final_src}은 {detergent_text} 기준으로 처리하는 후보이므로, 세제 계열이 다른 후보는 최종 세제 결정 근거에서 분리하고 동일 색상 계열 후보({same_family_text})를 보조 비교로 참고했습니다.")
    else:
        reasons.append(f"{final_src}은 {detergent_text} 기준으로 처리하는 후보이므로, 세제 계열이 다른 후보는 최종 세제 결정 근거에서 분리하고 상위 후보({candidate_text})는 참고 비교로만 사용했습니다.")
    reasons.append(f"{reaction_text}. 이 전후 변화량은 색상 변화와 후보 신뢰도 보조 지표로만 참고했습니다.")
    reasons.append(MATERIAL_ABSORB_REASON.get(
        material_group,
        f"원단({GROUP_DISPLAY.get(material_group, material_group)})의 흡수성 특성을 세척 강도에 반영했습니다.",
    ))
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
    reasons = [r.rstrip(".") for r in reasons if r]
    if any(("CNN" in r or "신뢰도" in r or "API" in r or "score" in r.lower()) for r in reasons):
        reasons = fallback_reasons
    treatment_kr = treatment_summary
    if treatment_kr:
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
) -> Optional[Dict[str, Any]]:
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key or OpenAI is None:
        return None
    client = OpenAI(api_key=api_key)
    prompt = f"""
You are the final decision system for a smart stain remover.
The user captured stain images of the same clothing item.
Available inputs:
1) stain before water
2) stain after water
3) optional care tag image or a manually selected fabric group

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
- Write exactly 5 Korean reasons
- Korean reasons must be user-facing visual explanations, not developer logs.
- The final UI will use server-side reason templates. If you write reason_kr, keep it conservative.
- Do not claim texture, viscosity, transparency, gloss, or water reaction as directly measured unless it is clearly visible in the images.
- Prefer explanations based on visible color family, before/after change, and candidate comparison.
- Do not write CNN, model confidence, API, or internal score details in reason_kr.
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
    first_brush_count = FIRST_BRUSH_COUNT_BY_GROUP.get(material_group, 2)
    second_brush_count = SECOND_BRUSH_COUNT_BY_GROUP.get(material_group, 4)
    brush_count = first_brush_count + second_brush_count
    detergent_key = SOURCE_INFO[source]["detergent"]
    pump_num = ARDUINO_PUMP_MAP[detergent_key]
    detergent_ko = DETERGENT_LABEL_KO.get(detergent_key, detergent_key)
    intensity_ko = INTENSITY_LABEL_KO.get(intensity, intensity)
    if brush_count <= 0:
        summary = f"{detergent_ko} {pump_ms}ms 처리 후 원단 보호를 위해 브러싱은 생략합니다."
    else:
        summary = (
            f"{detergent_ko} {pump_ms}ms → 1차 브러시 {first_brush_count}회 → 물 헹굼 {RINSE_PHASE_MS}ms "
            f"→ 2차 브러시 {second_brush_count}회 → 물 헹굼 {RINSE_PHASE_MS}ms 순서로 작동합니다. "
            f"브러시 강도는 {intensity_ko}이며, {INTENSITY_DESCRIPTION_KO[intensity]}"
        )
    return Treatment(
        brush_intensity=intensity,
        pump_ms=pump_ms,
        brush_count=brush_count,
        first_brush_count=first_brush_count,
        second_brush_count=second_brush_count,
        rinse_phase_ms=RINSE_PHASE_MS,
        detergent_key=detergent_key,
        pump_pin=pump_num,
        summary=summary,
    )


def split_brush_count_for_two_passes(brush_count: int) -> Tuple[int, int]:
    if brush_count <= 0:
        return 0, 0
    for group, total in BRUSH_COUNT_BY_GROUP.items():
        if brush_count == total:
            return FIRST_BRUSH_COUNT_BY_GROUP[group], SECOND_BRUSH_COUNT_BY_GROUP[group]
    first = max(1, brush_count // 3)
    return first, max(0, brush_count - first)


def run_motor(
    detergent_key: str,
    pump_ms: int,
    brush_count: int,
    brush_intensity: str,
    first_brush_count: Optional[int] = None,
    second_brush_count: Optional[int] = None,
    rinse_phase_ms: int = RINSE_PHASE_MS,
) -> None:
    if first_brush_count is None or second_brush_count is None:
        first_brush_count, second_brush_count = split_brush_count_for_two_passes(brush_count)

    run_pump(detergent_key, pump_ms)
    run_brush_pwm(first_brush_count, brush_intensity)
    run_underwater_motor(rinse_phase_ms)
    run_brush_pwm(second_brush_count, brush_intensity)
    run_underwater_motor(rinse_phase_ms)


def _cleanup_family_mask(image_bgr: np.ndarray, stain_label: str) -> np.ndarray:
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    label = str(stain_label or "").strip()
    if label in {"coffee", "teriyaki_sauce", "oriental_dressing", "soy_sauce"}:
        mask = cv2.inRange(hsv, (5, 20, 20), (28, 255, 230))
    elif label in {"bbq_sauce", "gochujang", "ketchup"}:
        m1 = cv2.inRange(hsv, (0, 45, 30), (12, 255, 255))
        m2 = cv2.inRange(hsv, (165, 45, 30), (179, 255, 255))
        mask = cv2.bitwise_or(m1, m2)
    elif label in {"curry", "mustard"}:
        mask = cv2.inRange(hsv, (12, 35, 35), (45, 255, 255))
    elif label == "oil":
        mask = cv2.inRange(hsv, (10, 10, 30), (45, 160, 220))
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


def _background_reference_mask(stain_mask: np.ndarray) -> np.ndarray:
    h, w = stain_mask.shape[:2]
    kernel = np.ones((17, 17), np.uint8)
    expanded = cv2.dilate(stain_mask, kernel, iterations=1)
    border = np.zeros_like(stain_mask)
    margin_x = max(6, int(w * 0.08))
    margin_y = max(6, int(h * 0.08))
    border[:margin_y, :] = 255
    border[-margin_y:, :] = 255
    border[:, :margin_x] = 255
    border[:, -margin_x:] = 255
    bg_mask = cv2.bitwise_or(cv2.bitwise_not(expanded), border)
    if cv2.countNonZero(bg_mask) < h * w * 0.12:
        bg_mask = cv2.bitwise_not(stain_mask)
    return bg_mask


def _wetness_normalized_cleaned_roi(roi_b: np.ndarray, roi_c: np.ndarray, baseline_mask: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
    bg_mask = _background_reference_mask(baseline_mask)
    hsv_b = cv2.cvtColor(roi_b, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv_c = cv2.cvtColor(roi_c, cv2.COLOR_BGR2HSV).astype(np.float32)

    bg_val_b = _masked_mean(hsv_b[:, :, 2], bg_mask)
    bg_val_c = _masked_mean(hsv_c[:, :, 2], bg_mask)
    bg_sat_b = _masked_mean(hsv_b[:, :, 1], bg_mask)
    bg_sat_c = _masked_mean(hsv_c[:, :, 1], bg_mask)

    val_gain = max(0.70, min(1.45, bg_val_b / max(bg_val_c, 1.0)))
    sat_gain = max(0.75, min(1.35, bg_sat_b / max(bg_sat_c, 1.0))) if bg_sat_b > 1.0 and bg_sat_c > 1.0 else 1.0

    hsv_norm = hsv_c.copy()
    hsv_norm[:, :, 1] = np.clip(hsv_norm[:, :, 1] * sat_gain, 0, 255)
    hsv_norm[:, :, 2] = np.clip(hsv_norm[:, :, 2] * val_gain, 0, 255)
    norm_bgr = cv2.cvtColor(hsv_norm.astype(np.uint8), cv2.COLOR_HSV2BGR)
    wetness_darkening_pct = max(0.0, min(100.0, (bg_val_b - bg_val_c) / max(bg_val_b, 1.0) * 100.0))

    return norm_bgr, {
        "background_val_baseline": float(bg_val_b),
        "background_val_cleaned": float(bg_val_c),
        "background_sat_baseline": float(bg_sat_b),
        "background_sat_cleaned": float(bg_sat_c),
        "wetness_val_gain": float(val_gain),
        "wetness_sat_gain": float(sat_gain),
        "wetness_darkening_pct": float(wetness_darkening_pct),
        "background_mask_area": float(cv2.countNonZero(bg_mask)),
    }


def compute_cleanup_feedback_metrics(baseline_bgr: np.ndarray, cleaned_bgr: np.ndarray, stain_label: str) -> Dict[str, float]:
    roi_b = cv2.resize(_extract_feedback_roi(baseline_bgr), (224, 224))
    roi_c = cv2.resize(_extract_feedback_roi(cleaned_bgr), (224, 224))
    reaction = compute_reaction_summary(roi_b, roi_c)

    baseline_mask = _cleanup_family_mask(roi_b, stain_label)
    roi_c_normalized, wetness_metrics = _wetness_normalized_cleaned_roi(roi_b, roi_c, baseline_mask)
    cleaned_mask_raw = _cleanup_family_mask(roi_c, stain_label)
    cleaned_mask = _cleanup_family_mask(roi_c_normalized, stain_label)
    baseline_area = float(cv2.countNonZero(baseline_mask))
    cleaned_area_raw = float(cv2.countNonZero(cleaned_mask_raw))
    cleaned_area = float(cv2.countNonZero(cleaned_mask))
    area_reduction_pct = _safe_pct(baseline_area - cleaned_area, baseline_area)

    hsv_b = cv2.cvtColor(roi_b, cv2.COLOR_BGR2HSV)
    hsv_c = cv2.cvtColor(roi_c_normalized, cv2.COLOR_BGR2HSV)
    sat_mean_baseline = _masked_mean(hsv_b[:, :, 1], baseline_mask)
    sat_mean_cleaned = _masked_mean(hsv_c[:, :, 1], cleaned_mask) if cleaned_area > 0 else 0.0
    val_mean_baseline = _masked_mean(hsv_b[:, :, 2], baseline_mask)
    val_mean_cleaned = _masked_mean(hsv_c[:, :, 2], cleaned_mask) if cleaned_area > 0 else 0.0

    sat_reduction_pct = _safe_pct(sat_mean_baseline - sat_mean_cleaned, max(sat_mean_baseline, 1.0))
    val_shift_pct = min(100.0, abs(val_mean_baseline - val_mean_cleaned) / 80.0 * 100.0)
    corrected_reaction = compute_reaction_summary(roi_b, roi_c_normalized)
    hsv_change_pct = min(100.0, corrected_reaction["sat_diff_mean"] / 40.0 * 100.0)
    gray_change_pct = min(100.0, corrected_reaction["gray_diff_mean"] / 30.0 * 100.0)

    if baseline_area < 60:
        area_reduction_pct = hsv_change_pct * 0.55 + gray_change_pct * 0.45

    removal_percent = (
        area_reduction_pct * 0.60
        + sat_reduction_pct * 0.25
        + hsv_change_pct * 0.10
        + gray_change_pct * 0.05
    )
    if wetness_metrics["wetness_darkening_pct"] >= 12.0:
        removal_percent = removal_percent * 0.85 + max(area_reduction_pct, sat_reduction_pct) * 0.15
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
        "corrected_sat_diff_mean": float(corrected_reaction["sat_diff_mean"]),
        "corrected_val_diff_mean": float(corrected_reaction["val_diff_mean"]),
        "corrected_gray_diff_mean": float(corrected_reaction["gray_diff_mean"]),
        "baseline_mask_area": baseline_area,
        "cleaned_mask_area_raw": cleaned_area_raw,
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
        **wetness_metrics,
        "removal_percent": float(removal_percent),
    }


def build_cleanup_feedback_fallback(removal_percent: float) -> Tuple[str, str, str]:
    if removal_percent >= FEEDBACK_GOOD_THRESHOLD:
        return (
            "good",
            "대체로 잘 지워졌습니다.",
            "현재 상태면 세척을 여기서 마무리해도 좋습니다.",
        )
    if removal_percent >= FEEDBACK_PARTIAL_THRESHOLD:
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

    def stop(self) -> None:
        self.running = False
        if self.thread and self.thread.is_alive() and threading.current_thread() is not self.thread:
            self.thread.join(timeout=1.0)
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
        self.cap = None
        self.thread = None
        with self.lock:
            self.latest_raw_frame = None

    def restart(self) -> None:
        self.stop()
        time.sleep(0.3)
        self.start()

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


def _safe_dataset_name(value: str) -> str:
    value = re.sub(r"[^0-9A-Za-z가-힣_.-]+", "_", value.strip())
    return value[:80] or "unknown"


def _save_learning_image(sample_dir: Path, name: str, data_url: Optional[str]) -> Optional[str]:
    if not data_url:
        return None
    image_bgr = decode_data_url_to_bgr(data_url)
    image_path = sample_dir / f"{name}.jpg"
    ok = cv2.imwrite(str(image_path), image_bgr)
    if not ok:
        raise RuntimeError(f"image_save_failed:{image_path}")
    return str(image_path.relative_to(FEEDBACK_DATASET_DIR)).replace("\\", "/")


def _append_learning_jsonl(record: Dict[str, Any]) -> None:
    FEEDBACK_DATASET_DIR.mkdir(parents=True, exist_ok=True)
    with FEEDBACK_LABELS_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _extract_jsonl_string(line: str, key: str) -> Optional[str]:
    match = re.search(rf'"{re.escape(key)}"\s*:\s*"([^"]*)"', line)
    return match.group(1).strip() if match else None


def _extract_jsonl_number(line: str, key: str) -> Optional[float]:
    match = re.search(rf'"{re.escape(key)}"\s*:\s*(-?\d+(?:\.\d+)?)', line)
    if not match:
        return None
    try:
        return float(match.group(1))
    except Exception:
        return None


def _parse_learning_feedback_line(line: str) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        pass

    record = {
        "sample_id": _extract_jsonl_string(line, "sample_id"),
        "verdict": _extract_jsonl_string(line, "verdict"),
        "predicted_label": _extract_jsonl_string(line, "predicted_label"),
        "correct_label": _extract_jsonl_string(line, "correct_label"),
        "actual_color": _extract_jsonl_string(line, "actual_color") or _extract_jsonl_string(line, "color_family"),
        "candidate_fit_score": _extract_jsonl_number(line, "candidate_fit_score"),
    }
    if not record["verdict"] or not record["predicted_label"]:
        return None
    if not record["correct_label"] and record["verdict"] == "correct":
        record["correct_label"] = record["predicted_label"]
    return record


def _load_learning_feedback_records() -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    paths = [FEEDBACK_LABELS_PATH]
    if FEEDBACK_LABELS_FALLBACK_PATH != FEEDBACK_LABELS_PATH:
        paths.append(FEEDBACK_LABELS_FALLBACK_PATH)
    seen_sample_ids = set()
    for path in paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = _parse_learning_feedback_line(line)
                if not record:
                    continue
                sample_id = record.get("sample_id")
                if sample_id and sample_id in seen_sample_ids:
                    continue
                if sample_id:
                    seen_sample_ids.add(sample_id)
                records.append(record)
    return records


def _normalize_feedback_label(label: Any) -> str:
    value = str(label or "").strip()
    if not value:
        return ""
    if value in SOURCE_INFO:
        return value
    if value in CLASS_TO_SOURCE:
        return CLASS_TO_SOURCE[value]
    if value in KO_TO_SOURCE:
        return KO_TO_SOURCE[value]
    return value


def _feedback_record_weight(record: Dict[str, Any], target_label: str, candidate_labels: set) -> float:
    weight = 1.0
    try:
        fit_score = record.get("candidate_fit_score")
        if fit_score is not None:
            fit = int(fit_score)
            if record.get("verdict") == "correct" and fit >= 4:
                weight += 0.4
            if record.get("verdict") == "incorrect" and fit <= 2:
                weight += 0.4
    except Exception:
        pass
    if target_label in candidate_labels:
        weight += 0.35
    return weight


def _collect_raw_prediction_labels(before_pred: Dict[str, Any], after_pred: Dict[str, Any]) -> set:
    labels = set()
    for data in (before_pred, after_pred):
        if not data:
            continue
        src = _normalize_feedback_label(data.get("source"))
        if src in SOURCE_INFO:
            labels.add(src)
        for cand in data.get("top3", []) or []:
            label = _normalize_feedback_label(cand.get("source") or cand.get("class_name"))
            if label in SOURCE_INFO:
                labels.add(label)
    return labels


def _prediction_label_support(label: str, pred: Dict[str, Any]) -> float:
    if not pred:
        return 0.0
    best = 0.0
    src = _normalize_feedback_label(pred.get("source"))
    if src == label:
        best = max(best, float(pred.get("confidence", 0.0)))
    for cand in pred.get("top3", []) or []:
        cand_label = _normalize_feedback_label(cand.get("source") or cand.get("class_name"))
        if cand_label != label:
            continue
        best = max(best, float(cand.get("confidence", cand.get("score", 0.0))))
    return best


def _current_prediction_support(label: str, before_pred: Dict[str, Any], after_pred: Dict[str, Any]) -> float:
    return max(_prediction_label_support(label, before_pred), _prediction_label_support(label, after_pred))


def apply_learning_feedback_adjustment(
    final_label: str,
    pair_conf: float,
    top_candidates: List[Dict[str, Any]],
    visual_profile: Dict[str, Any],
    before_pred: Dict[str, Any],
    after_pred: Dict[str, Any],
) -> Tuple[str, float, Optional[str], Optional[Dict[str, Any]]]:
    if not FEEDBACK_ADJUST_ENABLED:
        return final_label, pair_conf, None, None

    records = _load_learning_feedback_records()
    if not records:
        return final_label, pair_conf, None, None

    final_label = _normalize_feedback_label(final_label)
    visual_family = str((visual_profile or {}).get("family", "unknown"))
    raw_candidate_labels = _collect_raw_prediction_labels(before_pred, after_pred)
    display_candidate_labels = {
        _normalize_feedback_label(c.get("label"))
        for c in (top_candidates or [])
        if c.get("label")
    }
    candidate_labels = set(raw_candidate_labels) | set(display_candidate_labels)
    candidate_labels.add(final_label)

    scores: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    current_support = 0.0
    correction_total = 0

    for record in records:
        predicted = _normalize_feedback_label(record.get("predicted_label"))
        correct = _normalize_feedback_label(record.get("correct_label"))
        verdict = str(record.get("verdict") or "").strip()
        if not predicted or not correct or predicted != final_label or correct not in SOURCE_INFO:
            continue

        if verdict == "correct" and correct == final_label:
            weight = _feedback_record_weight(record, final_label, candidate_labels)
            current_support += weight
            scores[final_label] = scores.get(final_label, 0.0) + weight
            counts[final_label] = counts.get(final_label, 0) + 1
        elif verdict == "incorrect" and correct != final_label:
            weight = _feedback_record_weight(record, correct, candidate_labels)
            scores[correct] = scores.get(correct, 0.0) + weight
            counts[correct] = counts.get(correct, 0) + 1
            correction_total += 1

    if not scores:
        return final_label, pair_conf, None, None

    best_label = max(scores, key=scores.get)
    if best_label == final_label:
        return final_label, pair_conf, None, None

    best_count = counts.get(best_label, 0)
    best_ratio = best_count / max(1, correction_total)
    best_score = scores.get(best_label, 0.0)
    current_score = max(current_support, scores.get(final_label, 0.0))

    candidate_matches = best_label in raw_candidate_labels
    strong_history = best_count >= max(FEEDBACK_ADJUST_MIN_COUNT + 3, 7) and best_ratio >= 0.65
    repeated_confusion = best_count >= max(FEEDBACK_ADJUST_MIN_COUNT + 1, 5) and best_score >= current_score * 0.80

    before_label = _normalize_feedback_label(before_pred.get("source"))
    after_label = _normalize_feedback_label(after_pred.get("source"))
    before_conf = float(before_pred.get("confidence", 0.0))
    after_conf = float(after_pred.get("confidence", 0.0))
    max_conf = max(before_conf, after_conf)

    direct_agreement = (before_label == after_label == final_label)
    direct_support = (before_label == final_label or after_label == final_label)
    hard_model_lock = (direct_agreement and max_conf >= 0.80) or (direct_support and max_conf >= 0.93)
    best_support_now = _current_prediction_support(best_label, before_pred, after_pred)
    final_support_now = _current_prediction_support(final_label, before_pred, after_pred)

    adjust_meta: Dict[str, Any] = {
        "from_label": final_label,
        "to_label": best_label,
        "from_label_kr": SOURCE_DISPLAY_KO.get(final_label, final_label),
        "to_label_kr": SOURCE_DISPLAY_KO.get(best_label, best_label),
        "best_count": int(best_count),
        "best_ratio": float(best_ratio),
        "best_score": float(best_score),
        "current_score": float(current_score),
        "candidate_match": bool(candidate_matches),
        "raw_candidate_match": bool(candidate_matches),
        "display_candidate_match": bool(best_label in display_candidate_labels),
        "visual_family": visual_family,
        "model_lock": bool(hard_model_lock),
        "before_label": before_label,
        "after_label": after_label,
        "before_conf": before_conf,
        "after_conf": after_conf,
        "best_support_now": float(best_support_now),
        "final_support_now": float(final_support_now),
        "correction_total": int(correction_total),
    }

    if hard_model_lock:
        return final_label, pair_conf, None, None

    if best_count < max(FEEDBACK_ADJUST_MIN_COUNT + 2, 5):
        return final_label, pair_conf, None, None
    if correction_total < max(10, best_count + 2):
        return final_label, pair_conf, None, None
    if best_ratio < max(FEEDBACK_ADJUST_MIN_RATIO, 0.45):
        return final_label, pair_conf, None, None
    if best_score < current_score + max(FEEDBACK_ADJUST_MARGIN, 0.75) and not repeated_confusion:
        return final_label, pair_conf, None, None

    if not candidate_matches:
        return final_label, pair_conf, None, None

    if best_support_now < 0.12:
        return final_label, pair_conf, None, None

    if final_support_now >= 0.88 and best_support_now < final_support_now * 0.70:
        return final_label, pair_conf, None, None

    if not (candidate_matches or strong_history):
        return final_label, pair_conf, None, None

    adjusted_conf = max(pair_conf, min(0.96, 0.72 + min(0.20, best_ratio * 0.20)))
    reason = (
        f"학습 피드백 {best_count}건에서 {SOURCE_DISPLAY_KO.get(final_label, final_label)} 판단이 "
        f"{SOURCE_DISPLAY_KO.get(best_label, best_label)}로 반복 수정되었고 현재 샘플 후보 근거도 확인되어 최종 라벨을 보정했습니다."
    )
    return best_label, adjusted_conf, reason, adjust_meta


def _read_dev_state() -> Dict[str, Any]:
    with _dev_state_lock:
        if not DEV_STATE_PATH.exists():
            return {}
        try:
            return json.loads(DEV_STATE_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}


def _write_dev_state(state: Dict[str, Any]) -> Dict[str, Any]:
    state = dict(state or {})
    if isinstance(state.get("logs"), list):
        state["logs"] = [str(line) for line in state["logs"][:120]]
    if isinstance(state.get("history"), list):
        state["history"] = [str(step) for step in state["history"][:40]]
    state["updated_at"] = datetime.now().isoformat(timespec="seconds")
    with _dev_state_lock:
        DEV_STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    return state


@app.on_event("startup")
def on_startup():
    global ARDUINO_ERROR
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
    if not MAIN_UI_PATH.exists():
        raise HTTPException(status_code=500, detail=f"ui_file_missing:{MAIN_UI_PATH}")
    return MAIN_UI_PATH.read_text(encoding="utf-8")


@app.get("/learning", response_class=HTMLResponse)
def serve_learning_ui():
    return serve_ui()


@app.post("/api/dev-state")
def save_dev_state(payload: DevStateRequest):
    state = _write_dev_state(payload.dict())
    return {"ok": True, "state": state, "updated_at": state.get("updated_at")}


@app.get("/api/dev-state")
def get_dev_state():
    state = _read_dev_state()
    return {"ok": True, "state": state, "updated_at": state.get("updated_at")}


@app.get("/api/learning-labels")
def learning_labels():
    return {
        "labels": [
            {"label": label, "label_kr": SOURCE_DISPLAY_KO.get(label, label)}
            for label in SOURCE_INFO.keys()
        ]
    }


@app.post("/api/learning-feedback")
def save_learning_feedback(payload: LearningFeedbackRequest):
    analysis = payload.analysis or {}
    predicted_label = analysis.get("final_label") or ""
    correct_label = payload.correct_label or (predicted_label if payload.verdict == "correct" else None)
    actual_color = payload.actual_color or (SOURCE_FAMILY.get(correct_label) if correct_label else None)

    if payload.verdict == "incorrect" and not correct_label:
        raise HTTPException(status_code=400, detail="correct_label_required_for_incorrect_feedback")
    if correct_label and correct_label not in SOURCE_INFO:
        raise HTTPException(status_code=400, detail=f"unknown_correct_label:{correct_label}")
    if not actual_color:
        raise HTTPException(status_code=400, detail="actual_color_required")
    if actual_color not in {"red", "brown", "yellow", "pale", "other", "unknown"}:
        raise HTTPException(status_code=400, detail=f"unknown_actual_color:{actual_color}")
    if payload.candidate_fit_score is not None and not 0 <= int(payload.candidate_fit_score) <= 5:
        raise HTTPException(status_code=400, detail="candidate_fit_score_must_be_0_to_5")

    timestamp = datetime.now().isoformat(timespec="seconds")
    sample_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    label_for_dir = _safe_dataset_name(correct_label or predicted_label or "unknown")
    sample_dir = FEEDBACK_IMAGE_DIR / label_for_dir / sample_id

    try:
        with _learning_write_lock:
            sample_dir.mkdir(parents=True, exist_ok=True)
            images = {
                "before": _save_learning_image(sample_dir, "before", payload.before_image),
                "after": _save_learning_image(sample_dir, "after", payload.after_image),
                "tag": _save_learning_image(sample_dir, "tag", payload.tag_image),
                "feedback": _save_learning_image(sample_dir, "feedback", payload.feedback_image),
            }
            images = {key: value for key, value in images.items() if value}

            record = {
                "sample_id": sample_id,
                "timestamp": timestamp,
                "verdict": payload.verdict,
                "predicted_label": predicted_label or None,
                "predicted_label_kr": analysis.get("final_label_kr"),
                "correct_label": correct_label,
                "correct_label_kr": SOURCE_DISPLAY_KO.get(correct_label, correct_label) if correct_label else None,
                "confidence": analysis.get("confidence"),
                "decision_source": analysis.get("decision_source"),
                "top_candidates": analysis.get("top_candidates", []),
                "candidate_fit_score": payload.candidate_fit_score,
                "actual_color": actual_color,
                "actual_color_kr": FAMILY_DISPLAY_KO.get(actual_color, actual_color),
                "cnn_before_class": analysis.get("cnn_before_class"),
                "cnn_before_confidence": analysis.get("cnn_before_confidence"),
                "cnn_after_class": analysis.get("cnn_after_class"),
                "cnn_after_confidence": analysis.get("cnn_after_confidence"),
                "material_group": analysis.get("material_group"),
                "material_group_display": analysis.get("material_group_display"),
                "detergent_kr": analysis.get("detergent_kr"),
                "images": images,
            }
            _append_learning_jsonl(record)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"learning_feedback_save_error:{repr(exc)}")

    return {"ok": True, "sample_id": sample_id, "saved_images": images}


@app.get("/api/learning-feedback/stats")
def learning_feedback_stats():
    records = _load_learning_feedback_records()
    verdict_counts = {"correct": 0, "incorrect": 0}
    label_counts: Dict[str, int] = {}

    for record in records:
        verdict = record.get("verdict")
        if verdict not in verdict_counts:
            continue
        verdict_counts[verdict] += 1
        label = record.get("correct_label") or record.get("predicted_label") or "unknown"
        label_counts[label] = label_counts.get(label, 0) + 1

    return {
        "total": sum(verdict_counts.values()),
        "verdict_counts": verdict_counts,
        "label_counts": label_counts,
        "dataset_dir": str(FEEDBACK_DATASET_DIR),
    }


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
        "brush_count_by_group": BRUSH_COUNT_BY_GROUP,
        "first_brush_count_by_group": FIRST_BRUSH_COUNT_BY_GROUP,
        "second_brush_count_by_group": SECOND_BRUSH_COUNT_BY_GROUP,
        "pump_map": ARDUINO_PUMP_MAP,
        "max_umotor_ms": MAX_UMOTOR_MS,
        "rinse_phase_ms": RINSE_PHASE_MS,
        "default_stepper_ms": DEFAULT_STEPPER_MS,
        "max_stepper_ms": MAX_STEPPER_MS,
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

@app.post("/api/camera/restart")
def camera_restart():
    try:
        camera_manager.restart()
        return {
            "ok": True,
            "running": camera_manager.running,
            "camera_index": CAMERA_INDEX
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"camera_restart_error:{repr(e)}")


@app.post("/api/stepper", response_model=StepperResponse)
def stepper(payload: StepperRequest):
    try:
        responses = run_stepper(payload.direction, payload.duration_ms)
        return {"ok": True, "direction": payload.direction, "duration_ms": payload.duration_ms, "responses": responses}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"stepper_error:{repr(e)}")


@app.post("/api/step-down", response_model=StepperResponse)
def step_down():
    try:
        responses = run_stepper("down", DEFAULT_STEPPER_MS)
        return {"ok": True, "direction": "down", "duration_ms": DEFAULT_STEPPER_MS, "responses": responses}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"stepper_error:{repr(e)}")


@app.post("/api/step-up", response_model=StepperResponse)
def step_up():
    try:
        responses = run_stepper("up", DEFAULT_STEPPER_MS)
        return {"ok": True, "direction": "up", "duration_ms": DEFAULT_STEPPER_MS, "responses": responses}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"stepper_error:{repr(e)}")


@app.post("/api/analyze-triplet", response_model=AnalyzeTripletResponse)
def analyze_triplet(payload: AnalyzeTripletRequest):
    try:
        before_img_raw = decode_data_url_to_bgr(payload.before_image)
        after_img_raw = decode_data_url_to_bgr(payload.after_image)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"image_parse_error:{repr(e)}")

    try:
        tag_img, ocr_texts, debug_files, material_group, material_reason, material_input_mode = resolve_material_input(
            payload.tag_image, payload.manual_material_group, payload.manual_material_label
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
        denim_context = is_denim_context(
            material_group=material_group,
            material_reason=material_reason,
            ocr_texts=ocr_texts,
            manual_material_label=payload.manual_material_label,
        )
        dark_fabric_context_raw = detect_dark_fabric_context(before_img_raw, after_img_raw)
        before_bg_profile = _background_ring_profile_for_analysis(before_img_raw)
        after_bg_profile = _background_ring_profile_for_analysis(after_img_raw)
        dark_fabric_context_bg = bool(
            before_bg_profile.get("dark_hint", 0.0) >= 0.5
            or after_bg_profile.get("dark_hint", 0.0) >= 0.5
        )
        dark_fabric_context = bool(dark_fabric_context_raw or dark_fabric_context_bg)

        # 전역 색상 판단은 원본 기준으로 유지하고,
        # 노랑↔흰색 충돌 재검토 분기에서만 배경 기준 보정을 사용한다.
        before_img_for_color = before_img_raw
        after_img_for_color = after_img_raw
        before_img_for_yellow_recheck = _normalize_analysis_image_with_background(
            before_img_raw,
            before_bg_profile,
            force_dark_mode=dark_fabric_context,
        )
        after_img_for_yellow_recheck = _normalize_analysis_image_with_background(
            after_img_raw,
            after_bg_profile,
            force_dark_mode=dark_fabric_context,
        )
        reaction = compute_reaction_summary(before_img_raw, after_img_raw)

        visual_profile = detect_visual_family(before_img_for_color, after_img_for_color, dark_fabric_context)
        top_candidates = merge_top_candidates(
            before_pred,
            after_pred,
            visual_profile,
            is_denim_context=denim_context,
            is_dark_fabric_context=dark_fabric_context,
        )
        final_label, pair_conf, decision_source = choose_cnn_agreement(before_pred, after_pred, top_candidates)
        visual_override_reason = None
        pale_yellow_recheck_reason = None
        combo_override_reason = None
        override_label, override_conf, override_reason = apply_pale_visual_override(
            final_label,
            pair_conf,
            visual_profile,
            before_pred,
            after_pred,
            dark_fabric_context,
        )
        if override_label != final_label:
            final_label = override_label
            pair_conf = override_conf
            decision_source = "visual_pale_override"
            visual_override_reason = override_reason
        recheck_label, recheck_conf, recheck_reason = apply_pale_yellow_recheck(
            final_label,
            pair_conf,
            visual_profile,
            before_pred,
            after_pred,
            before_img_for_yellow_recheck,
            after_img_for_yellow_recheck,
            dark_fabric_context,
        )
        if recheck_label != final_label:
            final_label = recheck_label
            pair_conf = recheck_conf
            decision_source = "pale_yellow_recheck"
            pale_yellow_recheck_reason = recheck_reason
        combo_label, combo_conf, combo_reason = apply_family_combo_override(
            final_label,
            pair_conf,
            top_candidates,
            visual_profile,
            before_pred,
            after_pred,
            denim_context,
            dark_fabric_context,
        )
        if combo_label != final_label:
            final_label = combo_label
            pair_conf = combo_conf
            decision_source = "family_combo_adjusted"
            combo_override_reason = combo_reason
        fallback_reasons = build_pair_reasons(final_label, before_pred, after_pred, material_group, top_candidates, reaction, decision_source, visual_profile)
        if combo_override_reason:
            fallback_reasons = [combo_override_reason] + fallback_reasons[:4]
        elif pale_yellow_recheck_reason:
            fallback_reasons = [pale_yellow_recheck_reason] + fallback_reasons[:4]
        elif visual_override_reason:
            fallback_reasons = [visual_override_reason] + fallback_reasons[:4]

        treatment = build_treatment(final_label, material_group)
        with _plan_lock:
            LAST_EXECUTION_PLAN.clear()
            LAST_EXECUTION_PLAN.update({
                "final_label":    final_label,
                "detergent_key":  treatment.detergent_key,
                "pump_ms":        treatment.pump_ms,
                "brush_count":    treatment.brush_count,
                "first_brush_count": treatment.first_brush_count,
                "second_brush_count": treatment.second_brush_count,
                "rinse_phase_ms": treatment.rinse_phase_ms,
                "brush_intensity": treatment.brush_intensity,
            })
        treatment_kr = treatment.summary

        use_openai = should_use_openai(before_pred, after_pred, visual_profile)
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
                fallback_reasons = build_pair_reasons(
                    final_label,
                    before_pred,
                    after_pred,
                    material_group,
                    top_candidates,
                    reaction,
                    decision_source,
                    visual_profile,
                )
                recheck_label, recheck_conf, recheck_reason = apply_pale_yellow_recheck(
                    final_label,
                    pair_conf,
                    visual_profile,
                    before_pred,
                    after_pred,
                    before_img_for_yellow_recheck,
                    after_img_for_yellow_recheck,
                    dark_fabric_context,
                )
                if recheck_label != final_label:
                    final_label = recheck_label
                    pair_conf = recheck_conf
                    decision_source = "pale_yellow_recheck"
                    pale_yellow_recheck_reason = recheck_reason
                    fallback_reasons = [pale_yellow_recheck_reason] + fallback_reasons[:4]
                combo_label, combo_conf, combo_reason = apply_family_combo_override(
                    final_label,
                    pair_conf,
                    top_candidates,
                    visual_profile,
                    before_pred,
                    after_pred,
                    denim_context,
                    dark_fabric_context,
                )
                if combo_label != final_label:
                    final_label = combo_label
                    pair_conf = combo_conf
                    decision_source = "family_combo_adjusted"
                    combo_override_reason = combo_reason
                    fallback_reasons = [combo_override_reason] + fallback_reasons[:4]
                treatment = build_treatment(final_label, material_group)
                treatment_kr = treatment.summary
                with _plan_lock:
                    LAST_EXECUTION_PLAN.clear()
                    LAST_EXECUTION_PLAN.update({
                        "final_label":    final_label,
                        "detergent_key":  treatment.detergent_key,
                        "pump_ms":        treatment.pump_ms,
                        "brush_count":    treatment.brush_count,
                        "first_brush_count": treatment.first_brush_count,
                        "second_brush_count": treatment.second_brush_count,
                        "rinse_phase_ms": treatment.rinse_phase_ms,
                        "brush_intensity": treatment.brush_intensity,
                    })
            else:
                if decision_source not in {"visual_pale_override", "pale_yellow_recheck", "family_combo_adjusted"}:
                    decision_source = "cnn_fallback_after_openai_skip"

        feedback_reason = None
        feedback_adjust_meta = None
        adjusted_label, adjusted_conf, feedback_reason, feedback_adjust_meta = apply_learning_feedback_adjustment(
            final_label,
            pair_conf,
            top_candidates,
            visual_profile,
            before_pred,
            after_pred,
        )
        if adjusted_label != final_label:
            final_label = adjusted_label
            pair_conf = adjusted_conf
            decision_source = "feedback_adjusted"
            treatment = build_treatment(final_label, material_group)
            treatment_kr = treatment.summary
            with _plan_lock:
                LAST_EXECUTION_PLAN.clear()
                LAST_EXECUTION_PLAN.update({
                    "final_label":    final_label,
                    "detergent_key":  treatment.detergent_key,
                    "pump_ms":        treatment.pump_ms,
                    "brush_count":    treatment.brush_count,
                    "first_brush_count": treatment.first_brush_count,
                    "second_brush_count": treatment.second_brush_count,
                    "rinse_phase_ms": treatment.rinse_phase_ms,
                    "brush_intensity": treatment.brush_intensity,
                })
        top_candidates = align_top_candidates_with_final(final_label, top_candidates, before_pred, after_pred)
        fallback_reasons = build_pair_reasons(
            final_label,
            before_pred,
            after_pred,
            material_group,
            top_candidates,
            reaction,
            decision_source,
            visual_profile,
        )
        if combo_override_reason and decision_source == "family_combo_adjusted":
            fallback_reasons = [combo_override_reason] + fallback_reasons[:4]
        elif pale_yellow_recheck_reason and decision_source == "pale_yellow_recheck":
            fallback_reasons = [pale_yellow_recheck_reason] + fallback_reasons[:4]
        elif visual_override_reason and decision_source == "visual_pale_override":
            fallback_reasons = [visual_override_reason] + fallback_reasons[:4]
        elif feedback_reason:
            fallback_reasons = [feedback_reason] + fallback_reasons[:4]

        return AnalyzeTripletResponse(
            final_label=final_label,
            final_label_kr=SOURCE_DISPLAY_KO.get(final_label, final_label),
            decision_source=decision_source,
            decision_summary=build_decision_summary(final_label, decision_source, visual_profile),
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
            visual_family=visual_profile.get("family"),
            visual_family_display=visual_profile.get("family_kr"),
            family_summary=build_family_summary(top_candidates),
            visual_evidence=build_visual_evidence(visual_profile, top_candidates),
            feedback_adjust_reason=feedback_reason,
            feedback_adjust_meta=feedback_adjust_meta,
            ocr_text_preview=ocr_texts[:30],
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
            first_brush_count = LAST_EXECUTION_PLAN.get("first_brush_count")
            second_brush_count = LAST_EXECUTION_PLAN.get("second_brush_count")
            rinse_phase_ms = LAST_EXECUTION_PLAN.get("rinse_phase_ms", RINSE_PHASE_MS)
            brush_intensity = LAST_EXECUTION_PLAN.get("brush_intensity", payload.brush_intensity)
        if not detergent_key:
            raise HTTPException(status_code=400, detail="execute_error:no_detergent_key_available_run_analyze_first")
        if first_brush_count is None or second_brush_count is None:
            first_brush_count, second_brush_count = split_brush_count_for_two_passes(brush_count)
        first_brush_count = int(first_brush_count)
        second_brush_count = int(second_brush_count)
        rinse_phase_ms = int(rinse_phase_ms)
        run_motor(detergent_key, pump_ms, brush_count, brush_intensity, first_brush_count, second_brush_count, rinse_phase_ms)
        duty   = BRUSH_DUTY.get(brush_intensity, BRUSH_DUTY["medium"])
        per_ms = PER_STROKE_MS.get(brush_intensity, PER_STROKE_MS["medium"])
        first_dur_ms = 0 if first_brush_count <= 0 else max(300, min(MAX_BRUSH_MS, first_brush_count * per_ms))
        second_dur_ms = 0 if second_brush_count <= 0 else max(300, min(MAX_BRUSH_MS, second_brush_count * per_ms))
        dur_ms = first_dur_ms + second_dur_ms
        return {
            "ok": True,
            "sequence": [
                {"step": "detergent", "detergent_key": detergent_key, "pump_ms": pump_ms,
                 "pump_num": ARDUINO_PUMP_MAP.get(detergent_key)},
                {"step": "brush_1", "brush_count": first_brush_count, "brush_intensity": brush_intensity,
                 "brush_duty_pct": duty, "brush_duration_ms": first_dur_ms, "brush_pwm_pin": BRUSH_PWM_PIN},
                {"step": "rinse_1", "duration_ms": rinse_phase_ms},
                {"step": "brush_2", "brush_count": second_brush_count, "brush_intensity": brush_intensity,
                 "brush_duty_pct": duty, "brush_duration_ms": second_dur_ms, "brush_pwm_pin": BRUSH_PWM_PIN},
                {"step": "rinse_2", "duration_ms": rinse_phase_ms},
            ],
            "pump_ms": pump_ms,
            "brush_count": brush_count,
            "first_brush_count": first_brush_count,
            "second_brush_count": second_brush_count,
            "rinse_phase_ms": rinse_phase_ms,
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
        if status_kr:
            status_kr = status_kr.rstrip(".")
        if recommendation_kr:
            recommendation_kr = recommendation_kr.rstrip(".")
        if comment_kr:
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
