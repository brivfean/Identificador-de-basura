# =========================
# Conversiones de imagen
# =========================
from .conversions import (
    convertir_pil_a_cv,
    convertir_cv_a_pil,
    separar_rgb_cv,
    convertir_a_grises_cv
)

# =========================
# Histogramas y estadísticas
# =========================
from .histogramas import (
    compute_histogramas_rgb_arrays_from_cv,
    compute_histograma_gris_array_from_cv,
    calcular_stats_rgb_from_cv
)

# =========================
# Componentes conexas
# =========================
from .componentes_conexas import (
    preparar_binaria,
    etiquetar_componentes,
    etiquetas_a_rgb,
    calcular_stats_cc
)

from .modelos_color import ModelosColor

from .pseudocolor import Pseudocolor

__all__ = [
    # conversiones
    "convertir_pil_a_cv",
    "convertir_cv_a_pil",
    "separar_rgb_cv",
    "convertir_a_grises_cv",

    # histogramas / stats
    "compute_histogramas_rgb_arrays_from_cv",
    "compute_histograma_gris_array_from_cv",
    "calcular_stats_rgb_from_cv",

    # componentes conexas
    "preparar_binaria",
    "etiquetar_componentes",
    "etiquetas_a_rgb",
    "calcular_stats_cc",

    # Modelos de color
    "ModelosColor",

    #pseudocolor
    "Pseudocolor"

]
