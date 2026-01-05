import numpy as np
import cv2
from PIL import Image

def convertir_pil_a_cv(im_pil):
    arr = np.array(im_pil)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

def convertir_cv_a_pil(img_cv):
    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb)

def separar_rgb_cv(img_cv):
    """
    Recibe imagen BGR (OpenCV)
    Devuelve lista de imágenes PIL: [R, G, B] coloreadas
    """
    b, g, r = cv2.split(img_cv)
    zeros = np.zeros_like(r)

    rgb_r = np.stack([r, zeros, zeros], axis=2)
    rgb_g = np.stack([zeros, g, zeros], axis=2)
    rgb_b = np.stack([zeros, zeros, b], axis=2)

    return [
        Image.fromarray(rgb_r),
        Image.fromarray(rgb_g),
        Image.fromarray(rgb_b)
    ]

def convertir_a_grises_cv(img_cv):
    if img_cv.ndim == 3:
        return cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    return img_cv

