import numpy as np
import cv2

def ruido_sal_pimienta(img_cv, p):
    """
    Ruido combinado sal y pimienta
    p: probabilidad (0 - 0.5)
    """
    out = img_cv.copy()
    rnd = np.random.rand(img_cv.shape[0], img_cv.shape[1])

    salt = rnd < p
    pepper = rnd > (1 - p)

    out[salt] = 255
    out[pepper] = 0

    return out


def ruido_sal(img_cv, p):
    """
    Ruido tipo SAL
    p: probabilidad (0 - 0.5)
    """
    out = img_cv.copy()
    rnd = np.random.rand(img_cv.shape[0], img_cv.shape[1])

    salt = rnd < p
    out[salt] = 255

    return out


def ruido_pimienta(img_cv, p):
    """
    Ruido tipo PIMIENTA
    p: probabilidad (0 - 0.5)
    """
    out = img_cv.copy()
    rnd = np.random.rand(img_cv.shape[0], img_cv.shape[1])

    pepper = rnd < p
    out[pepper] = 0

    return out


def ruido_gaussiano(img_cv, sigma):
    img = img_cv.astype(np.float32)
    ruido = np.random.normal(0, sigma, img.shape).astype(np.float32)
    noisy = img + ruido
    return np.clip(noisy, 0, 255).astype(np.uint8)
