import cv2
import numpy as np
import os


class Normalizador:
    """
    Normalización de imágenes:
    - Resolución
    - Formato de archivo (PNG, JPG, BMP)
    """

    def __init__(self, imagen: np.ndarray):
        if imagen is None:
            raise ValueError("La imagen no puede ser None")

        self.img = imagen

    # ==================================================
    # NORMALIZACIÓN DE RESOLUCIÓN
    # ==================================================
    def normalizar_resolucion(self, ancho, alto, mantener_aspecto=True):
        h, w = self.img.shape[:2]

        if mantener_aspecto:
            escala = min(ancho / w, alto / h)
            nuevo_w = int(w * escala)
            nuevo_h = int(h * escala)
        else:
            nuevo_w = ancho
            nuevo_h = alto

        return cv2.resize(
            self.img,
            (nuevo_w, nuevo_h),
            interpolation=cv2.INTER_AREA
        )

    # ==================================================
    # NORMALIZACIÓN DE FORMATO (ARCHIVO)
    # ==================================================
    def guardar_como(self, ruta_salida: str, formato: str):
        """
        formato: 'png' | 'jpg' | 'bmp'
        """
        formato = formato.lower()

        if formato not in ("png", "jpg", "bmp"):
            raise ValueError("Formato no soportado")

        # Asegurar extensión correcta
        if not ruta_salida.lower().endswith(f".{formato}"):
            ruta_salida += f".{formato}"

        # JPG no soporta alfa
        img_guardar = self.img
        if formato == "jpg" and len(self.img.shape) == 3 and self.img.shape[2] == 4:
            img_guardar = self.img[:, :, :3]

        ok = cv2.imwrite(ruta_salida, img_guardar)

        if not ok:
            raise IOError("No se pudo guardar la imagen")

        return ruta_salida
