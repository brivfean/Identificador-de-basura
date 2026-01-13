# ===============================
# image_utils
# ===============================
from ..image_utils.conversions import convertir_a_grises_cv
from ..image_utils.modelos_color import ModelosColor
from ..image_utils.pseudocolor import Pseudocolor

# ===============================
# processing
# ===============================
from ..processing.filtros import *
from ..processing.morfologia import *
from ..processing.ruido import *
from ..processing.aritmeticas import *
from ..processing.logicas import *

# ===============================
# Normalizacion
# ===============================
from ..normalization.normalizador import Normalizador

# ==================================================
# REGISTRO DE FUNCIONES DE PREPROCESAMIENTO
# ==================================================
PREPROCESSING_FUNCTIONS = {

    # ----------------------------------
    # Conversión de color (YA EXISTENTES)
    # ----------------------------------
    "grises": convertir_a_grises_cv,

    # ----------------------------------
    # Filtros (YA EXISTENTES + NUEVOS)
    # ----------------------------------
    "media": filtro_media,
    "mediana": filtro_mediana,
    "gaussiano": filtro_gaussiano,

    # ----------------------------------
    # Ruido
    # ----------------------------------
    "ruido_sal_pimienta": ruido_sal_pimienta,
    "ruido_gaussiano": ruido_gaussiano,

    # ----------------------------------
    # Aritméticas
    # ----------------------------------
    "suma": suma,
    "resta": resta,
    "multiplicacion": multiplicacion,
    "division": division,

    # ----------------------------------
    # Lógicas
    # ----------------------------------
    "and": op_and,
    "or": op_or,
    "xor": op_xor,
    "not": op_not,

    # ----------------------------------
    # Modelos de color
    # ----------------------------------
    "rgb_r": lambda img: ModelosColor.canal_rgb(img, "R"),
    "rgb_g": lambda img: ModelosColor.canal_rgb(img, "G"),
    "rgb_b": lambda img: ModelosColor.canal_rgb(img, "B"),

    "cmyk_c": lambda img: ModelosColor.canal_cmyk(img, "C"),
    "cmyk_m": lambda img: ModelosColor.canal_cmyk(img, "M"),
    "cmyk_y": lambda img: ModelosColor.canal_cmyk(img, "Y"),
    "cmyk_k": lambda img: ModelosColor.canal_cmyk(img, "K"),

    "hsl_h": lambda img: ModelosColor.canal_hsl(img, "H"),
    "hsl_l": lambda img: ModelosColor.canal_hsl(img, "L"),
    "hsl_s": lambda img: ModelosColor.canal_hsl(img, "S"),

    # ----------------------------------
    # Binarización (YA EXISTENTES)
    # ----------------------------------
    "binarizar_umbral": ModelosColor.binarizar_umbral,
    "binarizar_otsu": ModelosColor.binarizar_otsu,

    # ----------------------------------
    # Pseudocolor
    # ----------------------------------
    "pseudocolor_jet": lambda img: Pseudocolor.aplicar_opencv(img, "JET"),
    "pseudocolor_hot": lambda img: Pseudocolor.aplicar_opencv(img, "HOT"),

    # ----------------------------------
    # Normalización
    # ----------------------------------
    "normalizar_resolucion": lambda img, **params: (
    Normalizador(img).normalizar_resolucion(**params)
    ),

    # ----------------------------------
    # Morfologia
    # ----------------------------------
    "erosion_color": erosion_color,
    "dilatacion_color": dilatacion_color,
    "gradiente_color": gradiente_morfologico_color,
    "tophat_color": tophat_color,
    "blackhat_color": blackhat_color,


}
