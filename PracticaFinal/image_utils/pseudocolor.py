import cv2
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


# ====== Colormaps personalizados ======
_PASTEL_COLORS = [
    (1.0, 0.8, 0.9),  # rosa claro
    (0.8, 1.0, 0.8),  # verde menta
    (0.8, 0.9, 1.0),  # azul lavanda
    (1.0, 1.0, 0.8),  # amarillo suave
    (0.9, 0.8, 1.0)   # violeta claro
]
_PASTEL = LinearSegmentedColormap.from_list("PastelMap", _PASTEL_COLORS, N=256)

CUSTOM_CMAPS = {
    "Pastel (custom)": _PASTEL,
}
CUSTOM_LUTS = {}


def _cmap_to_lut(cmap: LinearSegmentedColormap) -> np.ndarray:
    """Convierte un colormap a una LUT (256, 3) uint8 para indexación rápida."""
    samples = np.linspace(0.0, 1.0, 256)
    rgba = cmap(samples)              # (256, 4)
    rgb = (rgba[:, :3] * 255).astype(np.uint8)  # (256, 3)
    return rgb


def make_random_colormap(name="RandomMap", n_anchors=5, seed=None) -> LinearSegmentedColormap:
    """Genera un colormap aleatorio con puntos de anclaje."""
    rng = np.random.default_rng(seed)
    anchors = np.linspace(0.0, 1.0, n_anchors)
    colors = rng.random((n_anchors, 3))
    colors[0]  = colors[0] * 0.2            # más oscuro
    colors[-1] = 0.8 + colors[-1] * 0.2     # más claro
    seg = list(zip(anchors, colors))
    return LinearSegmentedColormap.from_list(name, seg, N=256)


def install_random_cmap(label="Random (custom)", n_anchors=6, seed=None):
    """Instala un colormap aleatorio en el registro global."""
    rand_cmap = make_random_colormap("RandomMap", n_anchors=n_anchors, seed=seed)
    CUSTOM_CMAPS[label] = rand_cmap
    CUSTOM_LUTS[label] = _cmap_to_lut(rand_cmap)


# Inicialización de LUTs
CUSTOM_LUTS["Pastel (custom)"] = _cmap_to_lut(CUSTOM_CMAPS["Pastel (custom)"])
install_random_cmap(label="Random (custom)", n_anchors=6, seed=42)  # reproducible al arranque


# ====== Colormaps OpenCV disponibles ======
_COLORMAP_NAMES = {
    "JET": "COLORMAP_JET",
    "HOT": "COLORMAP_HOT",
    "OCEAN": "COLORMAP_OCEAN",
    "PARULA": "COLORMAP_PARULA",
    "RAINBOW": "COLORMAP_RAINBOW",
    "HSV": "COLORMAP_HSV",
    "AUTUMN": "COLORMAP_AUTUMN",
    "BONE": "COLORMAP_BONE",
    "COOL": "COLORMAP_COOL",
    "PINK": "COLORMAP_PINK",
    "SPRING": "COLORMAP_SPRING",
    "SUMMER": "COLORMAP_SUMMER",
    "WINTER": "COLORMAP_WINTER",
    "COPPER": "COLORMAP_COPPER",
    "INFERNO": "COLORMAP_INFERNO",
    "PLASMA": "COLORMAP_PLASMA",
    "MAGMA": "COLORMAP_MAGMA",
    "CIVIDIS": "COLORMAP_CIVIDIS",
    "VIRIDIS": "COLORMAP_VIRIDIS",
    "TWILIGHT": "COLORMAP_TWILIGHT",
    "TURBO": "COLORMAP_TURBO",
}


def _build_available_colormaps():
    """Construye dict de colormaps disponibles en la versión de OpenCV."""
    available = {}
    for name, const in _COLORMAP_NAMES.items():
        if hasattr(cv2, const):
            available[name] = getattr(cv2, const)
    return available


AVAILABLE_COLORMAPS = _build_available_colormaps()
GRAYSCALE_OPTION = "Escala de grises (sin pseudocolor)"


def get_menu_items():
    """Retorna lista de opciones de menú: Grises + OpenCV + Custom."""
    return [GRAYSCALE_OPTION] + list(AVAILABLE_COLORMAPS.keys()) + list(CUSTOM_CMAPS.keys())


def regenerate_random():
    """Genera un nuevo 'Random (custom)' con seed=None (distinto cada vez)."""
    install_random_cmap(label="Random (custom)", n_anchors=6, seed=None)


class Pseudocolor:

    def __init__(self, imagen_gris: np.ndarray):
        if imagen_gris is None:
            raise ValueError("La imagen en gris no puede ser None")

        if len(imagen_gris.shape) != 2:
            raise ValueError("La imagen debe estar en escala de grises")

        self.imagen_gris = imagen_gris

    # --------------------------------------------------
    # OpenCV pseudocolors
    # --------------------------------------------------
    def aplicar_opencv(self, nombre_mapa: str) -> np.ndarray:
        """Aplica un colormap de OpenCV por nombre."""
        nombre_mapa = nombre_mapa.upper()

        if nombre_mapa not in AVAILABLE_COLORMAPS:
            raise ValueError(f"Mapa OpenCV no soportado: {nombre_mapa}")

        return cv2.applyColorMap(
            self.imagen_gris,
            AVAILABLE_COLORMAPS[nombre_mapa]
        )

    # --------------------------------------------------
    # Custom & Pastel colormap
    # --------------------------------------------------
    def aplicar_custom(self, nombre_custom: str) -> np.ndarray:
        """Aplica un colormap personalizado (Pastel o Random)."""
        if nombre_custom not in CUSTOM_LUTS:
            raise ValueError(f"Colormap personalizado no encontrado: {nombre_custom}")

        lut = CUSTOM_LUTS[nombre_custom]  # (256, 3) uint8
        colored_rgb = lut[self.imagen_gris]
        return colored_rgb

    # --------------------------------------------------
    # Pseudocolor personalizado (dinámico)
    # --------------------------------------------------
    def aplicar_personalizado(self, colores: list, n=256) -> np.ndarray:
        """
        Aplica un colormap personalizado dinámico.
        colores: lista de tuplas RGB normalizadas (0.0 - 1.0)
        """
        if not colores or len(colores) < 2:
            raise ValueError("Se requieren al menos dos colores")

        cmap = LinearSegmentedColormap.from_list(
            "CustomMap", colores, N=n
        )

        norm = self.imagen_gris.astype(np.float32) / 255.0
        mapped = cmap(norm)[:, :, :3]  # RGB flotante

        return (mapped * 255).astype(np.uint8)
