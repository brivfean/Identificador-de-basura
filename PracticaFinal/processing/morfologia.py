import cv2
import numpy as np

def erosion(img_cv, k=5):
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((k, k), np.uint8)
    result = cv2.erode(gray, kernel, iterations=1)
    return cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

def dilatacion(img_cv, k=5):
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((k, k), np.uint8)
    result = cv2.dilate(gray, kernel, iterations=1)
    return cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

def apertura(img_cv, k=5):
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((k, k), np.uint8)

    tradicional = cv2.dilate(
        cv2.erode(gray, kernel, iterations=1),
        kernel,
        iterations=1
    )

    morph = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)

    return (
        cv2.cvtColor(tradicional, cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(morph, cv2.COLOR_GRAY2BGR)
    )

def cierre(img_cv, k=5):
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((k, k), np.uint8)

    tradicional = cv2.erode(
        cv2.dilate(gray, kernel, iterations=1),
        kernel,
        iterations=1
    )

    morph = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

    return (
        cv2.cvtColor(tradicional, cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(morph, cv2.COLOR_GRAY2BGR)
    )
