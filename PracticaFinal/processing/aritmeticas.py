import cv2
import numpy as np

def suma(img1, img2):
    return cv2.add(img1, img2)

def resta(img1, img2):
    return cv2.subtract(img1, img2)

def multiplicacion(img1, img2):
    img1_f = img1.astype(np.float32)
    img2_f = img2.astype(np.float32)
    result = img1_f * img2_f / 255.0
    return np.clip(result, 0, 255).astype(np.uint8)

def division(img1, img2):
    img1_f = img1.astype(np.float32)
    img2_f = img2.astype(np.float32)
    result = img1_f / (img2_f + 1)
    return np.clip(result, 0, 255).astype(np.uint8)
