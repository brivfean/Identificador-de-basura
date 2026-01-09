import cv2
import numpy as np


class ModelosColor:

    # ==================================================
    # UTILIDAD INTERNA
    # ==================================================
    @staticmethod
    def _asegurar_bgr(img_cv):
        if img_cv.ndim == 2:
            return cv2.cvtColor(img_cv, cv2.COLOR_GRAY2BGR)
        return img_cv

    # ==================================================
    # ESCALA DE GRISES
    # ==================================================
    @staticmethod
    def a_grises(img_cv):
        if img_cv.ndim == 3:
            return cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        return img_cv

    # ==================================================
    # RGB (canales)
    # ==================================================
    @staticmethod
    def canal_rgb(img_cv, canal):
        img_cv = ModelosColor._asegurar_bgr(img_cv)

        b, g, r = cv2.split(img_cv)
        cero = np.zeros_like(b)

        if canal == "R":
            return cv2.merge([cero, cero, r])
        elif canal == "G":
            return cv2.merge([cero, g, cero])
        elif canal == "B":
            return cv2.merge([b, cero, cero])
        else:
            raise ValueError("Canal RGB inválido")

    # ==================================================
    # CMYK
    # ==================================================
    @staticmethod
    def canal_cmyk(img_cv, canal):
        img_cv = ModelosColor._asegurar_bgr(img_cv)

        rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]

        k = 1 - np.max(rgb, axis=2)
        c = (1 - r - k) / (1 - k + 1e-8)
        m = (1 - g - k) / (1 - k + 1e-8)
        y = (1 - b - k) / (1 - k + 1e-8)

        if canal == "C":
            img = np.stack([np.zeros_like(c), c, c], axis=2)
        elif canal == "M":
            img = np.stack([m, np.zeros_like(m), m], axis=2)
        elif canal == "Y":
            img = np.stack([y, y, np.zeros_like(y)], axis=2)
        elif canal == "K":
            gray = (k * 255).astype(np.uint8)
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        else:
            raise ValueError("Canal CMYK inválido")

        return (img * 255).astype(np.uint8)

    # ==================================================
    # HSL
    # ==================================================
    @staticmethod
    def canal_hsl(img_cv, canal):
        img_cv = ModelosColor._asegurar_bgr(img_cv)

        hls = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HLS)
        h, l, s = cv2.split(hls)

        if canal == "H":
            hls_vis = cv2.merge([h, np.full_like(h, 128), np.full_like(h, 255)])
        elif canal == "L":
            hls_vis = cv2.merge([h, l, s])
        elif canal == "S":
            hls_vis = cv2.merge([h, np.full_like(h, 128), s])
        else:
            raise ValueError("Canal HSL inválido")

        return cv2.cvtColor(hls_vis, cv2.COLOR_HLS2BGR)

    # ==================================================
    # BINARIZACIÓN
    # ==================================================
    @staticmethod
    def binarizar_umbral(img_cv, t):
        gray = ModelosColor.a_grises(img_cv)
        _, binaria = cv2.threshold(gray, t, 255, cv2.THRESH_BINARY)
        return binaria

    @staticmethod
    def binarizar_umbral_inverso(img_cv, t):
        gray = ModelosColor.a_grises(img_cv)
        _, binaria = cv2.threshold(gray, t, 255, cv2.THRESH_BINARY_INV)
        return binaria

    @staticmethod
    def binarizar_truncado(img_cv, t):
        gray = ModelosColor.a_grises(img_cv)
        _, binaria = cv2.threshold(gray, t, 255, cv2.THRESH_TRUNC)
        return binaria

    @staticmethod
    def binarizar_a_cero(img_cv, t):
        gray = ModelosColor.a_grises(img_cv)
        _, binaria = cv2.threshold(gray, t, 255, cv2.THRESH_TOZERO)
        return binaria

    @staticmethod
    def binarizar_otsu(img_cv):
        gray = ModelosColor.a_grises(img_cv)
        _, binaria = cv2.threshold(
            gray, 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        return binaria

    @staticmethod
    def binarizar_adaptativa_media(img_cv, block_size=11, c=2):
        gray = ModelosColor.a_grises(img_cv)
        return cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY,
            block_size,
            c
        )

    @staticmethod
    def binarizar_adaptativa_gauss(img_cv, block_size=11, c=2):
        gray = ModelosColor.a_grises(img_cv)
        return cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block_size,
            c
        )
