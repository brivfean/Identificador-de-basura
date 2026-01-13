import cv2
import numpy as np
from ..image_utils import convertir_a_grises_cv

def grises(img):
    return convertir_a_grises_cv(img)

def binarizacion_umbral(img, t):
    gray = convertir_a_grises_cv(img)
    _, bin_img = cv2.threshold(gray, t, 255, cv2.THRESH_BINARY)
    return bin_img

def gaussiano(img, sigma):
    k = int(6 * sigma + 1)
    if k % 2 == 0:
        k += 1
    return cv2.GaussianBlur(img, (k, k), sigma)
