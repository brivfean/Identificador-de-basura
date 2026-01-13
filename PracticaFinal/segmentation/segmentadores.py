import cv2
import numpy as np


class Segmentador:
    """
    Segmentación e identificación de formas.
    La imagen principal YA debe venir preprocesada.
    Aquí SOLO se segmenta y se identifican objetos.
    """

    # ==================================================
    # CONSTRUCTOR
    # ==================================================
    def __init__(self, imagen: np.ndarray):
        if imagen is None:
            raise ValueError("La imagen no puede ser None")

        self.binaria = self._forzar_binaria(imagen)

    # ==================================================
    # FORZAR BINARIA (ÚNICO MÉTODO)
    # ==================================================
    def _forzar_binaria(self, img: np.ndarray) -> np.ndarray:
        if img is None:
            raise ValueError("Imagen inválida")

        # Color → gris (seguro)
        from ..image_utils.utils import to_gray
        img = to_gray(img)

        # Asegurar binaria real
        valores = np.unique(img)
        if not set(valores).issubset({0, 255}):
            _, img = cv2.threshold(
                img, 0, 255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )

        return img

    # ==================================================
    # CASO 1: SEGMENTACIÓN GEOMÉTRICA
    # ==================================================
    def formas_geometricas(self):
        contornos, _ = cv2.findContours(
            self.binaria,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        objetos = []

        for i, cnt in enumerate(contornos, start=1):
            area = cv2.contourArea(cnt)
            if area < 50:
                continue

            perimetro = cv2.arcLength(cnt, True)
            circ = self._circularidad(area, perimetro)
            forma = self._clasificar_forma(cnt, circ)

            objetos.append({
                "id": i,
                "forma": forma,
                "area": float(area),
                "perimetro": float(perimetro),
                "contorno": cnt
            })

        salida = self._dibujar(objetos)
        return salida, objetos

    def _circularidad(self, area, perimetro):
        if perimetro == 0:
            return 0.0
        return 4 * np.pi * area / (perimetro ** 2)

    def _clasificar_forma(self, cnt, circ):
        approx = cv2.approxPolyDP(
            cnt, 0.02 * cv2.arcLength(cnt, True), True
        )
        lados = len(approx)

        if circ > 0.8:
            return "Circular"
        if lados == 3:
            return "Triangular"
        if lados == 4:
            return "Cuadrilateral"
        if lados > 4:
            return "Poligonal"
        return "Irregular"

    def _dibujar(self, objetos):
        salida = cv2.cvtColor(self.binaria, cv2.COLOR_GRAY2BGR)

        for o in objetos:
            cv2.drawContours(
                salida, [o["contorno"]], -1, (0, 255, 0), 2
            )
            x, y, w, h = cv2.boundingRect(o["contorno"])
            cv2.putText(
                salida,
                o["forma"],
                (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1
            )

        return salida

    # ==================================================
    # CASO 2: HIT-OR-MISS CON PATRÓN DEL USUARIO
    # ==================================================
    def hit_or_miss(self, imagen_patron: np.ndarray):
        """
        imagen_patron:
        Imagen BINARIA seleccionada por el usuario
        que contiene la forma a buscar.
        """

        patron = self._forzar_binaria(imagen_patron)

        img01 = (self.binaria > 0).astype(np.uint8)
        pat01 = (patron > 0).astype(np.uint8)

        resultado = cv2.morphologyEx(
            img01,
            cv2.MORPH_HITMISS,
            pat01
        )

        resultado = (resultado > 0).astype(np.uint8) * 255

        salida = cv2.cvtColor(self.binaria, cv2.COLOR_GRAY2BGR)
        salida[resultado > 0] = (0, 0, 255)

        encontrados = int(np.count_nonzero(resultado))

        return salida, encontrados
