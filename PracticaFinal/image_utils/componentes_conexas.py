import numpy as np
import cv2
from scipy import ndimage
import matplotlib.pyplot as plt


def preparar_binaria(img_cv, umbral=None):
    if img_cv is None:
        raise ValueError("Imagen None.")

    img = np.asarray(img_cv)

    # Convertir a gris solo si es necesario; soportar BGR y BGRA
    if img.ndim == 3:
        ch = img.shape[2]
        if ch == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif ch == 4:
            gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        else:
            # Si tiene un número inesperado de canales, tomar el primer canal
            gray = img[:, :, 0].copy()
    elif img.ndim == 2:
        gray = img.copy()
    else:
        raise ValueError("Imagen con dimensiones no soportadas.")

    if umbral is None:
        _, bin_img = cv2.threshold(
            gray, 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
    else:
        _, bin_img = cv2.threshold(
            gray, int(umbral), 255, cv2.THRESH_BINARY
        )

    return bin_img


def etiquetar_componentes(bin_img, vecindad=8):
    binary01 = (bin_img > 0).astype(int)

    if vecindad == 4:
        structure = np.array(
            [[0, 1, 0],
             [1, 1, 1],
             [0, 1, 0]], dtype=int
        )
    else:
        structure = np.ones((3, 3), dtype=int)

    etiquetas, num_obj = ndimage.label(binary01, structure=structure)
    return etiquetas, num_obj


def etiquetas_a_rgb(etiquetas):
    if etiquetas.max() == 0:
        return np.zeros((*etiquetas.shape, 3), dtype=np.uint8)

    norm = etiquetas.astype(float) / etiquetas.max()
    cmap = plt.cm.get_cmap("nipy_spectral")
    rgb = (cmap(norm)[:, :, :3] * 255).astype(np.uint8)
    return rgb


def calcular_stats_cc(etiquetas):
    stats = []

    for i in range(1, etiquetas.max() + 1):
        mask = (etiquetas == i).astype(np.uint8) * 255
        area = int(np.sum(etiquetas == i))

        conts, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )

        perimetro = sum(cv2.arcLength(c, True) for c in conts) if conts else 0.0
        stats.append((i, area, float(perimetro)))

    return stats
