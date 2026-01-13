import cv2
import numpy as np
from typing import Literal, Optional, Tuple

# ==================================================
# UTILIDADES INTERNAS
# ==================================================
def _to_gray(img: np.ndarray) -> np.ndarray:
    """Convierte BGR a GRAY si es necesario; preserva si ya es GRAY."""
    if img is None:
        raise ValueError("Imagen None.")
    if len(img.shape) == 3 and img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img.copy()


def _is_binary(img: np.ndarray) -> bool:
    """Heurística: ¿solo 0/255?"""
    u = np.unique(img)
    return u.size <= 2 and set(u.tolist()).issubset({0, 255})


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
# CONVERSIONES Y UTILIDADES
# ==================================================
def to_binary(img: np.ndarray, thresh: int = 127, use_otsu: bool = False, invert: bool = False) -> np.ndarray:
    """
    Convierte la imagen a binaria (0/255) con umbral fijo u Otsu.
    """
    gray = _to_gray(img)
    if use_otsu:
        ttype = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
        _, binimg = cv2.threshold(gray, 0, 255, ttype | cv2.THRESH_OTSU)
    else:
        t = np.clip(int(thresh), 0, 255)
        ttype = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
        _, binimg = cv2.threshold(gray, t, 255, ttype)
    return binimg


def ensure_same_size(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Redimensiona B a A si no coinciden tamaños (interpolación adecuada).
    """
    ha, wa = a.shape[:2]
    hb, wb = b.shape[:2]
    if (ha, wa) == (hb, wb):
        return a, b
    interp = cv2.INTER_NEAREST if (len(b.shape) == 2 or b.shape[2] == 1) else cv2.INTER_AREA
    b2 = cv2.resize(b, (wa, ha), interpolation=interp)
    return a, b2


def make_se(ksize: int = 3, shape: Literal["rect", "ellipse", "cross"] = "rect") -> np.ndarray:
    """
    Crea elemento estructurante (EE) con forma y tamaño dados.
    """
    k = max(1, int(ksize))
    if k % 2 == 0:  # forzar impar
        k += 1
    if shape == "ellipse":
        st = cv2.MORPH_ELLIPSE
    elif shape == "cross":
        st = cv2.MORPH_CROSS
    else:
        st = cv2.MORPH_RECT
    return cv2.getStructuringElement(st, (k, k))


# ==================================================
# OPERACIONES BÁSICAS (binario y gris)
# ==================================================
def erode(img: np.ndarray, ksize: int = 3, shape: str = "rect", iterations: int = 1) -> np.ndarray:
    """Erosión morfológica."""
    return cv2.erode(img, make_se(ksize, shape), iterations=iterations)


def dilate(img: np.ndarray, ksize: int = 3, shape: str = "rect", iterations: int = 1) -> np.ndarray:
    """Dilatación morfológica."""
    return cv2.dilate(img, make_se(ksize, shape), iterations=iterations)


def open_morph(img: np.ndarray, ksize: int = 3, shape: str = "rect", iterations: int = 1) -> np.ndarray:
    """Apertura morfológica (erosión → dilatación)."""
    return cv2.morphologyEx(img, cv2.MORPH_OPEN, make_se(ksize, shape), iterations=iterations)


def close_morph(img: np.ndarray, ksize: int = 3, shape: str = "rect", iterations: int = 1) -> np.ndarray:
    """Cierre morfológico (dilatación → erosión)."""
    return cv2.morphologyEx(img, cv2.MORPH_CLOSE, make_se(ksize, shape), iterations=iterations)


# ==================================================
# GRADIENTES MORFOLÓGICOS
# ==================================================
def gradient(img: np.ndarray, ksize: int = 3, shape: str = "rect",
             mode: Literal["sym", "int", "ext"] = "sym") -> np.ndarray:
    """
    Gradientes:
      - 'sym'  (simétrico): dilate - erode
      - 'int'  (interno):   img - erode
      - 'ext'  (externo):   dilate - img
    Soporta gris/binario.
    """
    se = make_se(ksize, shape)
    if mode == "sym":
        return cv2.morphologyEx(img, cv2.MORPH_GRADIENT, se)
    elif mode == "int":
        ero = cv2.erode(img, se)
        return cv2.subtract(img, ero)
    else:  # "ext"
        dil = cv2.dilate(img, se)
        return cv2.subtract(dil, img)


# ==================================================
# TOP-HAT / BLACK-HAT
# ==================================================
def top_hat(img: np.ndarray, ksize: int = 3, shape: str = "rect") -> np.ndarray:
    """Resalta regiones más claras que su vecindad."""
    return cv2.morphologyEx(img, cv2.MORPH_TOPHAT, make_se(ksize, shape))


def black_hat(img: np.ndarray, ksize: int = 3, shape: str = "rect") -> np.ndarray:
    """Resalta regiones más oscuras que su vecindad."""
    return cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, make_se(ksize, shape))


# ==================================================
# FRONTERA (BOUNDARY)
# ==================================================
def boundary(img: np.ndarray, ksize: int = 3, shape: str = "rect") -> np.ndarray:
    """
    Frontera interna clásica: B(img) = img - erode(img).
    Para binario: devuelve contorno en 0/255.
    Para gris: resalta bordes finos.
    """
    se = make_se(ksize, shape)
    ero = cv2.erode(img, se)
    return cv2.subtract(img, ero)


# ==================================================
# HIT-OR-MISS (BINARIO)
# ==================================================
def hit_or_miss(binimg: np.ndarray,
                kernel_hit: np.ndarray,
                kernel_miss: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Transformada Hit-or-Miss para BINARIO (0/255).
    kernel_hit  : EE con 1s donde se exige 1 (255) y 0 donde 'no importa'.
    kernel_miss : EE con 1s donde se exige 0 y 0 donde 'no importa'.
                  Si es None, se calcula como complemento de kernel_hit (aprox).
    OpenCV espera binario en 0/1; convertimos y devolvemos 0/255.
    """
    img = _to_gray(binimg)
    if not _is_binary(img):
        raise ValueError("hit_or_miss requiere imagen binaria (0/255).")
    img01 = (img > 0).astype(np.uint8)

    kh = (kernel_hit > 0).astype(np.uint8)
    if kernel_miss is None:
        # Aproximación: donde kh == 0, asumimos 'miss' (exige 0)
        km = (kh == 0).astype(np.uint8)
    else:
        km = (kernel_miss > 0).astype(np.uint8)

    # Componemos hit-or-miss vía intersección: (img ⊖ kh) ∩ (~img ⊖ km)
    hit  = cv2.erode(img01, kh)
    miss = cv2.erode(1 - img01, km)
    out01 = cv2.bitwise_and(hit, miss)
    return (out01 * 255).astype(np.uint8)


# ==================================================
# ADELGAZAMIENTO (THINNING)
# ==================================================
def thinning(binimg: np.ndarray, max_iters: int = 0) -> np.ndarray:
    """
    Adelgazamiento (Zhang–Suen) sobre binario 0/255.
    max_iters=0 => itera hasta convergencia.
    Devuelve 0/255.
    """
    img = _to_gray(binimg)
    if not _is_binary(img):
        raise ValueError("thinning requiere imagen binaria (0/255).")
    # Convertir a 0/1
    th = (img > 0).astype(np.uint8)

    def neighbors(y, x):
        # p2..p9 en sentido horario empezando arriba (8-neighbors)
        return [th[y-1, x], th[y-1, x+1], th[y, x+1], th[y+1, x+1],
                th[y+1, x], th[y+1, x-1], th[y, x-1], th[y-1, x-1]]

    def transitions(nb):
        # número de transiciones 0->1 en la secuencia circular
        return sum((nb[i] == 0 and nb[(i+1) % 8] == 1) for i in range(8))

    changed = True
    iters = 0
    h, w = th.shape
    while changed and (max_iters == 0 or iters < max_iters):
        changed = False
        m1 = []
        for y in range(1, h-1):
            for x in range(1, w-1):
                p = th[y, x]
                if p != 1:
                    continue
                nb = neighbors(y, x)
                C = transitions(nb)
                N = sum(nb)
                if 2 <= N <= 6 and C == 1 and (nb[0]*nb[2]*nb[4] == 0) and (nb[2]*nb[4]*nb[6] == 0):
                    m1.append((y, x))
        if m1:
            changed = True
            for (y, x) in m1:
                th[y, x] = 0

        m2 = []
        for y in range(1, h-1):
            for x in range(1, w-1):
                p = th[y, x]
                if p != 1:
                    continue
                nb = neighbors(y, x)
                C = transitions(nb)
                N = sum(nb)
                if 2 <= N <= 6 and C == 1 and (nb[0]*nb[2]*nb[6] == 0) and (nb[0]*nb[4]*nb[6] == 0):
                    m2.append((y, x))
        if m2:
            changed = True
            for (y, x) in m2:
                th[y, x] = 0

        iters += 1

    return (th * 255).astype(np.uint8)


# ==================================================
# ESQUELETO MORFOLÓGICO
# ==================================================
def skeletonize(binimg: np.ndarray, ksize: int = 3, shape: str = "rect") -> np.ndarray:
    """
    Esqueleto morfológico por iteración: 
      S = ⋃ (Erode^k(img) - Open(Erode^k(img)))
    hasta que Erode^k(img) sea vacío.
    Entrada: binario 0/255. Salida: 0/255.
    """
    img = _to_gray(binimg)
    if not _is_binary(img):
        raise ValueError("skeletonize requiere imagen binaria (0/255).")

    se = make_se(ksize, shape)
    skel = np.zeros_like(img)
    eroded = img.copy()

    while True:
        opened = cv2.morphologyEx(eroded, cv2.MORPH_OPEN, se)
        temp = cv2.subtract(eroded, opened)
        skel = cv2.bitwise_or(skel, temp)
        eroded = cv2.erode(eroded, se)
        if cv2.countNonZero(eroded) == 0:
            break
    return skel


# ==================================================
# SUAVIZADO MORFOLÓGICO
# ==================================================
def smooth(img: np.ndarray, ksize: int = 3, shape: str = "rect",
           passes: int = 1, mode: Literal["open_close", "close_open"] = "open_close") -> np.ndarray:
    """
    Suavizado morfológico para gris/binario:
      - open_close: apertura seguida de cierre (elimina ruido claro y rellena huecos pequeños)
      - close_open: cierre seguido de apertura (inverso)
    """
    out = img.copy()
    for _ in range(max(1, int(passes))):
        if mode == "open_close":
            out = open_morph(out, ksize, shape)
            out = close_morph(out, ksize, shape)
        else:
            out = close_morph(out, ksize, shape)
            out = open_morph(out, ksize, shape)
    return out


# ==================================================
# ATAJOS "TRADICIONALES" (COMPATIBILIDAD)
# ==================================================
def apertura_tradicional(img: np.ndarray, ksize: int = 3, shape: str = "rect", iterations: int = 1) -> np.ndarray:
    """Apertura = erosión seguida de dilatación (implementación 'manual')."""
    se = make_se(ksize, shape)
    out = cv2.erode(img, se, iterations=iterations)
    out = cv2.dilate(out, se, iterations=iterations)
    return out


def cierre_tradicional(img: np.ndarray, ksize: int = 3, shape: str = "rect", iterations: int = 1) -> np.ndarray:
    """Cierre = dilatación seguida de erosión (implementación 'manual')."""
    se = make_se(ksize, shape)
    out = cv2.dilate(img, se, iterations=iterations)
    out = cv2.erode(out, se, iterations=iterations)
    return out


# ==================================================
# MORFOLOGÍA BINARIA / ESCALA DE GRISES (COMPATIBILIDAD)
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
    """Apertura morfológica: erosión seguida de dilatación (versión optimizada)."""
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((k, k), np.uint8)
    morph = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
    return cv2.cvtColor(morph, cv2.COLOR_GRAY2BGR)


def cierre(img_cv, k=5):
    """Cierre morfológico: dilatación seguida de erosión (versión optimizada)."""
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((k, k), np.uint8)
    morph = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    return cv2.cvtColor(morph, cv2.COLOR_GRAY2BGR)


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


# ==================================================
# UTILIDADES DE GUARDADO
# ==================================================
def save(path: str, img: np.ndarray) -> None:
    """Guarda imagen con cv2.imwrite; levanta excepción si falla."""
    if not cv2.imwrite(path, img):
        raise IOError(f"No se pudo guardar: {path}")
