import cv2
import numpy as np

# ==================================================
# UTILIDAD INTERNA
# ==================================================
def _aplicar_morfologia_color(img, operacion, kernel):
    """
    Aplica una operación morfológica canal por canal (BGR)
    """
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("La imagen debe ser BGR para morfología en color")

    canales = cv2.split(img)
    procesados = [operacion(c, kernel) for c in canales]
    return cv2.merge(procesados)


# ==================================================
# MORFOLOGÍA BINARIA / ESCALA DE GRISES
# ==================================================
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


# ==================================================
# MORFOLOGÍA EN COLOR (BGR)
# ==================================================
def erosion_color(img_cv, k=5):
    kernel = np.ones((k, k), np.uint8)
    return _aplicar_morfologia_color(
        img_cv,
        lambda c, k: cv2.erode(c, k, iterations=1),
        kernel
    )


def dilatacion_color(img_cv, k=5):
    kernel = np.ones((k, k), np.uint8)
    return _aplicar_morfologia_color(
        img_cv,
        lambda c, k: cv2.dilate(c, k, iterations=1),
        kernel
    )


def gradiente_morfologico_color(img_cv, k=5):
    kernel = np.ones((k, k), np.uint8)

    def gradiente(c, k):
        dil = cv2.dilate(c, k, iterations=1)
        ero = cv2.erode(c, k, iterations=1)
        return cv2.subtract(dil, ero)

    return _aplicar_morfologia_color(img_cv, gradiente, kernel)


def tophat_color(img_cv, k=5):
    kernel = np.ones((k, k), np.uint8)
    return _aplicar_morfologia_color(
        img_cv,
        lambda c, k: cv2.morphologyEx(c, cv2.MORPH_TOPHAT, k),
        kernel
    )


def blackhat_color(img_cv, k=5):
    kernel = np.ones((k, k), np.uint8)
    return _aplicar_morfologia_color(
        img_cv,
        lambda c, k: cv2.morphologyEx(c, cv2.MORPH_BLACKHAT, k),
        kernel
    )
