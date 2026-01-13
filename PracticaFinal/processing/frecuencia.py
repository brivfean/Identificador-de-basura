"""
Módulo de Transformaciones en el Dominio de la Frecuencia (FFT y DCT)
Incluye: Filtros pasa bajas / pasa altas en frecuencia (ideal, gaussiano, Butterworth)
y compresión DCT por bloques 8x8 con cuantización tipo JPEG.
"""

import math
import numpy as np
from typing import Literal, Tuple


# ===================== Utilidades de imagen =====================

def _to_gray(img: np.ndarray) -> np.ndarray:
    """Convierte BGR a GRAY si es necesario; preserva si ya es GRAY."""
    import cv2
    if img is None:
        raise ValueError("Imagen None.")
    arr = img
    if len(arr.shape) == 3:
        ch = arr.shape[2]
        if ch == 3:
            return cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
        if ch == 4:
            return cv2.cvtColor(arr, cv2.COLOR_BGRA2GRAY)
        return arr[:, :, 0].copy()
    return arr.copy()


def normalizar_float(img: np.ndarray) -> np.ndarray:
    """Convierte imagen a float32 en rango [0, 1]."""
    img_f = np.array(img, dtype=np.float32)
    max_val = img_f.max()
    if max_val > 1.5:  # Si viene en 0-255
        img_f /= 255.0
    return img_f


# ===================== FFT y filtros en frecuencia =====================

def fft2_imagen(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Calcula FFT 2D, espectros de magnitud (log) y fase."""
    F = np.fft.fft2(img)
    Fshift = np.fft.fftshift(F)
    magnitud = np.log(1 + np.abs(Fshift))
    fase = np.angle(Fshift)
    return F, Fshift, magnitud, fase


def crear_mascara(img_shape: Tuple[int, int],
                  filtro: Literal["ideal", "gaussiano", "butterworth"] = "ideal",
                  tipo: Literal["lowpass", "highpass"] = "lowpass",
                  cutoff: float = 0.2,
                  orden: int = 2) -> np.ndarray:
    """
    Crea una máscara de filtro en el dominio de la frecuencia.
    
    - filtro: 'ideal', 'gaussiano', 'butterworth'
    - tipo: 'lowpass' o 'highpass'
    - cutoff: radio de corte normalizado (0 a ~0.5)
    - orden: parámetro para Butterworth
    """
    rows, cols = img_shape
    crow, ccol = rows // 2, cols // 2
    Y, X = np.ogrid[:rows, :cols]
    D = np.sqrt((Y - crow) ** 2 + (X - ccol) ** 2)
    Dnorm = D / float(max(crow, ccol, 1))

    if filtro == "ideal":
        H = (Dnorm <= cutoff).astype(np.float32)
    elif filtro == "gaussiano":
        H = np.exp(-(Dnorm ** 2) / (2 * (cutoff ** 2)))
    elif filtro == "butterworth":
        H = 1 / (1 + (Dnorm / (cutoff + 1e-8)) ** (2 * orden))
    else:
        raise ValueError(f"Filtro desconocido: {filtro}")

    if tipo == "lowpass":
        mask = H
    elif tipo == "highpass":
        mask = 1 - H
    else:
        raise ValueError(f"Tipo de filtro debe ser lowpass o highpass, no {tipo}")

    return mask.astype(np.float32)


def aplicar_filtro_fft(img: np.ndarray,
                       filtro: Literal["ideal", "gaussiano", "butterworth"] = "ideal",
                       tipo: Literal["lowpass", "highpass"] = "lowpass",
                       cutoff: float = 0.2,
                       orden: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """
    Aplica un filtro en el dominio de la frecuencia y reconstruye la imagen.
    
    Retorna: (imagen_filtrada, mascara)
    """
    img_norm = normalizar_float(img)
    F = np.fft.fft2(img_norm)
    Fshift = np.fft.fftshift(F)
    mask = crear_mascara(img.shape[:2], filtro=filtro, tipo=tipo, cutoff=cutoff, orden=orden)
    Gshift = Fshift * mask
    G = np.fft.ifftshift(Gshift)
    g = np.fft.ifft2(G)
    g = np.real(g)
    g = np.clip(g, 0, 1)
    
    # Reconvertir a uint8 para compatibilidad con OpenCV
    g_uint8 = (g * 255).astype(np.uint8)
    return g_uint8, mask


# ===================== DCT por bloques 8x8 =====================

def _dct_matrix(N: int = 8) -> np.ndarray:
    """Genera la matriz DCT tipo II de tamaño N."""
    C = np.zeros((N, N), dtype=np.float64)
    for k in range(N):
        alpha = np.sqrt(1 / N) if k == 0 else np.sqrt(2 / N)
        for n in range(N):
            C[k, n] = alpha * math.cos(((2 * n + 1) * k * math.pi) / (2 * N))
    return C


_C8 = _dct_matrix(8)


def _dct_bloque_2d(b: np.ndarray) -> np.ndarray:
    """DCT 2D tipo II."""
    return _C8 @ b @ _C8.T


def _idct_bloque_2d(D: np.ndarray) -> np.ndarray:
    """IDCT 2D inverso ortogonal."""
    return _C8.T @ D @ _C8


_Q_JPEG = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
], dtype=np.float64)


def _pad_a_multiplo(img: np.ndarray, N: int = 8) -> Tuple[np.ndarray, int, int]:
    """Rellena imagen para que ambos ejes sean múltiplos de N."""
    h, w = img.shape[:2]
    nh = ((h + N - 1) // N) * N
    nw = ((w + N - 1) // N) * N
    if len(img.shape) == 2:
        padded = np.zeros((nh, nw), dtype=img.dtype)
    else:
        padded = np.zeros((nh, nw, img.shape[2]), dtype=img.dtype)
    padded[:h, :w] = img
    return padded, h, w


def _calcular_psnr(img_ref: np.ndarray, img_rec: np.ndarray) -> float:
    """Calcula PSNR entre dos imágenes."""
    mse = np.mean((img_ref - img_rec) ** 2)
    if mse == 0:
        return float('inf')
    PIXEL_MAX = 1.0
    return 20 * math.log10(PIXEL_MAX) - 10 * math.log10(mse)


def dct_compresion(img: np.ndarray, q_factor: float = 0.5) -> Tuple[np.ndarray, float]:
    """
    Aplica compresión DCT por bloques 8x8 con cuantización.
    
    Retorna: (imagen_reconstruida, psnr)
    """
    # Convertir a escala de grises si es necesario
    img_gray = _to_gray(img)
    img_p = normalizar_float(img_gray)

    padded, h, w = _pad_a_multiplo(img_p, 8)
    H, W = padded.shape

    Q = _Q_JPEG * q_factor
    recon = np.zeros_like(padded, dtype=np.float64)

    for i in range(0, H, 8):
        for j in range(0, W, 8):
            b = padded[i:i+8, j:j+8]
            b_shift = b - 0.5
            D = _dct_bloque_2d(b_shift)
            Dq = np.round(D / Q)
            Dr = Dq * Q
            br = _idct_bloque_2d(Dr) + 0.5
            recon[i:i+8, j:j+8] = br

    recon = np.clip(recon[:h, :w], 0, 1)
    recon_uint8 = (recon * 255).astype(np.uint8)

    psnr = _calcular_psnr(img_p[:h, :w], recon)
    return recon_uint8, psnr
