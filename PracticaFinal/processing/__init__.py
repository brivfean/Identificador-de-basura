# =========================
# Filtros
# =========================
from .filtros import *

# =========================
# Ruido
# =========================
from .ruido import *

# =========================
# Morfología
# =========================
from .morfologia import *

# =========================
# Aritméticas
# =========================
from .aritmeticas import *

# =========================
# Lógicas
# =========================
from .logicas import *

__all__ = [
    # filtros
    "filtro_media",
    "filtro_mediana",
    "filtro_gaussiano",
    "filtro_laplaciano",
    "filtro_sobel",
    "filtro_prewitt",
    "filtro_roberts",
    "filtro_canny",

    # ruido
    "ruido_sal_pimienta",
    "ruido_sal",
    "ruido_pimienta",
    "ruido_gaussiano",

    # morfología
    "erosion",
    "dilatacion",
    "apertura",
    "cierre",

    # aritméticas
    "suma",
    "resta",
    "multiplicacion",
    "division",

    # lógicas
    "op_and",
    "op_or",
    "op_xor",
    "op_not"
]
