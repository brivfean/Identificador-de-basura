import numpy as np
import cv2
from math import log2
from scipy.stats import skew

def compute_histogramas_rgb_arrays_from_cv(img_cv):
    if img_cv is None:
        return None

    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    resultados = {}

    for i, canal in enumerate(['Rojo', 'Verde', 'Azul']):
        datos = img_rgb[:, :, i].flatten()
        histograma, _ = np.histogram(datos, bins=256, range=(0, 256))
        resultados[canal] = histograma

    return resultados

def compute_histograma_gris_array_from_cv(img_cv):
    if img_cv is None:
        return None

    img_gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    datos = img_gray.flatten()
    histograma, _ = np.histogram(datos, bins=256, range=(0, 256))
    return histograma

def calcular_stats_rgb_from_cv(img_cv):
    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    resultados = {}

    for i, canal in enumerate(['Rojo', 'Verde', 'Azul']):
        datos = img_rgb[:, :, i].flatten()
        histograma, _ = np.histogram(datos, bins=256, range=(0, 256))

        prob = histograma / histograma.sum() if histograma.sum() > 0 else np.zeros_like(histograma)
        energia = np.sum(prob ** 2)
        entropia = -np.sum([p * log2(p) for p in prob if p > 0])
        asimetria = skew(datos)
        media = np.mean(datos)
        varianza = np.var(datos)

        resultados[canal] = {
            'Energía': energia,
            'Entropía': entropia,
            'Asimetría': asimetria,
            'Media': media,
            'Varianza': varianza
        }

    return resultados
