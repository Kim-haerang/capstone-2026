# -*- coding: utf-8 -*-
import base64
from pathlib import Path
from typing import List
import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent
TFLITE_MODEL_PATH = BASE_DIR / 'models' / 'stain_classifier_v3.tflite'
CLASS_NAMES_PATH = BASE_DIR / 'models' / 'class_names.txt'
IMG_SIZE = 224
CLASS_TO_SOURCE = {"Brown_cof":"coffee","Brown_deri":"teriyaki_sauce","Brown_ori":"oriental_dressing","Brown_soy":"soy_sauce","Red_bbq":"bbq_sauce","Red_go":"gochujang","Red_ket":"ketchup","White_Milk":"milk","White_ma":"mayonnaise","Yellow_ca":"curry","Yellow_mus":"mustard","Yellow_oil":"oil"}
SOURCE_DISPLAY_KO = {"coffee":"커피","teriyaki_sauce":"데리야끼 소스","oriental_dressing":"오리엔탈 소스","soy_sauce":"간장","bbq_sauce":"바비큐 소스","gochujang":"고추장","ketchup":"케찹","milk":"우유","mayonnaise":"마요네즈","curry":"카레","mustard":"머스타드","oil":"기름"}
class PredictRequest(BaseModel): image: str
class Top3Item(BaseModel):
    rank: int; class_name: str; source: str; source_display: str; confidence: float
class PredictResponse(BaseModel):
    cnn_class: str; source: str; source_display: str; confidence: float; top3: List[Top3Item]
interpreter = None; input_details = None; output_details = None; class_names = []; model_error = None

def load_model():
    global interpreter, input_details, output_details, class_names, model_error
    try:
        import tensorflow as tf
        if not TFLITE_MODEL_PATH.exists(): raise RuntimeError(f'missing_model:{TFLITE_MODEL_PATH}')
        if not CLASS_NAMES_PATH.exists(): raise RuntimeError(f'missing_class_names:{CLASS_NAMES_PATH}')
        interpreter = tf.lite.Interpreter(model_path=str(TFLITE_MODEL_PATH))
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details(); output_details = interpreter.get_output_details()
        class_names = [line.strip() for line in CLASS_NAMES_PATH.read_text(encoding='utf-8').splitlines() if line.strip()]
        model_error = None
    except Exception as e:
        interpreter = None; input_details = None; output_details = None; model_error = repr(e)

def preprocess_for_v3(image_bgr):
    img = cv2.resize(image_bgr, (IMG_SIZE, IMG_SIZE))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    combined = np.concatenate([img_rgb.astype(np.float32), img_hsv.astype(np.float32)], axis=-1)
    combined = (combined / 127.5) - 1.0
    return np.expand_dims(combined, axis=0).astype(np.float32)

load_model(); app = FastAPI(title='TFLite V3 Inference Server')
@app.get('/health')
def health():
    return {'status': 'ok' if model_error is None else 'error', 'model_path': str(TFLITE_MODEL_PATH), 'model_exists': TFLITE_MODEL_PATH.exists(), 'class_count': len(class_names), 'input_shape': input_details[0]['shape'].tolist() if input_details else None, 'model_error': model_error}
@app.post('/reload')
def reload_model():
    load_model(); return health()
@app.post('/predict', response_model=PredictResponse)
def predict(payload: PredictRequest):
    if interpreter is None: raise HTTPException(status_code=503, detail=f'model_not_loaded:{model_error}')
    try:
        _, encoded = payload.image.split(',', 1)
        arr = np.frombuffer(base64.b64decode(encoded), dtype=np.uint8)
        img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img_bgr is None: raise ValueError('image_decode_failed')
    except Exception as e:
        raise HTTPException(status_code=400, detail=f'image_parse_error:{repr(e)}')
    try:
        x = preprocess_for_v3(img_bgr)
        interpreter.set_tensor(input_details[0]['index'], x)
        interpreter.invoke()
        probs = interpreter.get_tensor(output_details[0]['index'])[0].astype(np.float32)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'inference_error:{repr(e)}')
    top_idx = probs.argsort()[::-1][:3]
    top3 = []
    for rank, idx in enumerate(top_idx, start=1):
        class_name = class_names[idx] if idx < len(class_names) else f'class_{idx}'
        source = CLASS_TO_SOURCE.get(class_name, class_name)
        top3.append(Top3Item(rank=rank, class_name=class_name, source=source, source_display=SOURCE_DISPLAY_KO.get(source, source), confidence=float(probs[idx])))
    best_idx = int(np.argmax(probs)); best_class = class_names[best_idx] if best_idx < len(class_names) else f'class_{best_idx}'; best_source = CLASS_TO_SOURCE.get(best_class, best_class)
    return PredictResponse(cnn_class=best_class, source=best_source, source_display=SOURCE_DISPLAY_KO.get(best_source, best_source), confidence=float(probs[best_idx]), top3=top3)
