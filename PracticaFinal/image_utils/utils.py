import cv2
import numpy as np
from typing import Any


def to_gray(img: Any) -> np.ndarray:
    """Convierte una imagen a escala de grises de forma segura.

    Soporta imágenes ya en gris, BGR (3 canales) y BGRA (4 canales).
    Para otros números de canales toma el primer canal.
    """
    if img is None:
        raise ValueError("Imagen None.")

    arr = np.asarray(img)
    if arr.ndim == 2:
        return arr.copy()
    if arr.ndim == 3:
        ch = arr.shape[2]
        if ch == 3:
            return cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
        if ch == 4:
            return cv2.cvtColor(arr, cv2.COLOR_BGRA2GRAY)
        return arr[:, :, 0].copy()
    raise ValueError("Imagen con dimensiones no soportadas.")
