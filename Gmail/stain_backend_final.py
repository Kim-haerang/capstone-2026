
# stain_backend_final.py
import cv2
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
import base64

app = FastAPI()

class Req(BaseModel):
    image: str

def decode(img_str):
    _, encoded = img_str.split(",", 1)
    arr = np.frombuffer(base64.b64decode(encoded), np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)

def color_balance(image):
    img = cv2.resize(image, (224,224)).astype(np.float32)
    border = np.concatenate([
        img[:20,:,:], img[-20:,:,:], img[:,:20,:], img[:,-20:,:]
    ], axis=0)
    avg = np.mean(border, axis=0)
    gray = np.mean(avg)
    scale = gray / (avg + 1e-6)
    balanced = np.clip(img * scale, 0, 255)
    return balanced.astype(np.uint8)

def enhance_stain(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h,s,v = cv2.split(hsv)
    s = cv2.normalize(s, None, 0,255, cv2.NORM_MINMAX)
    return cv2.cvtColor(cv2.merge([h,s,v]), cv2.COLOR_HSV2BGR)

def preprocess(image):
    return enhance_stain(color_balance(image))

@app.post("/analyze")
def analyze(req: Req):
    img = decode(req.image)
    img = preprocess(img)
    return {"status":"ok"}
