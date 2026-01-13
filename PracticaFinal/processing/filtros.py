import cv2
import numpy as np

# =========================
# Filtros espaciales básicos
# =========================

def filtro_media(img_cv, k):
    return cv2.blur(img_cv, (k, k))

def filtro_mediana(img_cv, k):
    return cv2.medianBlur(img_cv, k)

def filtro_gaussiano(img_cv, k):
    if k % 2 == 0:
        k += 1
    return cv2.GaussianBlur(img_cv, (k, k), 0)

def filtro_laplaciano(img_cv, ksize):
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=ksize)
    lap_abs = cv2.convertScaleAbs(lap)
    return cv2.cvtColor(lap_abs, cv2.COLOR_GRAY2BGR)

# =========================
# Filtros de bordes
# =========================

def filtro_sobel(img_cv):
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    sobel_mag = np.sqrt(sobelx**2 + sobely**2)

    return (
        cv2.convertScaleAbs(sobelx),
        cv2.convertScaleAbs(sobely),
        cv2.convertScaleAbs(sobel_mag)
    )

def filtro_prewitt(img_cv):
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

    kernelx = np.array([[1, 0, -1],
                        [1, 0, -1],
                        [1, 0, -1]])

    kernely = np.array([[1, 1, 1],
                        [0, 0, 0],
                        [-1, -1, -1]])

    prewittx = cv2.filter2D(gray, -1, kernelx)
    prewitty = cv2.filter2D(gray, -1, kernely)

    mag = np.sqrt(prewittx.astype(np.float32)**2 + prewitty.astype(np.float32)**2)

    return (
        prewittx,
        prewitty,
        cv2.convertScaleAbs(mag)
    )

def filtro_roberts(img_cv):
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

    kernelx = np.array([[1, 0],
                        [0, -1]])
    kernely = np.array([[0, 1],
                        [-1, 0]])

    robertsx = cv2.filter2D(gray, -1, kernelx)
    robertsy = cv2.filter2D(gray, -1, kernely)

    mag = np.sqrt(robertsx.astype(np.float32)**2 + robertsy.astype(np.float32)**2)

    return (
        robertsx,
        robertsy,
        cv2.convertScaleAbs(mag)
    )

def filtro_canny(img_cv, t1, t2):
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    return cv2.Canny(gray, t1, t2)
