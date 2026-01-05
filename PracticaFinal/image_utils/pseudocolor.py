import cv2
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


class Pseudocolor:

    MAPAS_OPENCV = {
        "JET": cv2.COLORMAP_JET,
        "HOT": cv2.COLORMAP_HOT,
        "OCEAN": cv2.COLORMAP_OCEAN,
        "BONE": cv2.COLORMAP_BONE,
        "RAINBOW": cv2.COLORMAP_RAINBOW
    }

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
        nombre_mapa = nombre_mapa.upper()

        if nombre_mapa not in self.MAPAS_OPENCV:
            raise ValueError(f"Mapa OpenCV no soportado: {nombre_mapa}")

        return cv2.applyColorMap(
            self.imagen_gris,
            self.MAPAS_OPENCV[nombre_mapa]
        )

    # --------------------------------------------------
    # Pseudocolor personalizado
    # --------------------------------------------------
    def aplicar_personalizado(self, colores: list, n=256) -> np.ndarray:
        """
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
