import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk

import cv2
import os

from tkinter import simpledialog

import numpy as np
from scipy import ndimage

from ..processing.filtros import (
    filtro_media,
    filtro_mediana,
    filtro_gaussiano,
    filtro_laplaciano,
    filtro_sobel,
    filtro_prewitt,
    filtro_roberts,
    filtro_canny
)

from ..processing.ruido import (
    ruido_sal_pimienta,
    ruido_sal,
    ruido_pimienta,
    ruido_gaussiano
)

from ..processing.morfologia import (
    erosion,
    dilatacion,
    apertura,
    cierre,
    erosion_color,
    dilatacion_color,
    gradiente_morfologico_color,
    tophat_color,
    blackhat_color
)

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from ..image_utils.histogramas import (
    compute_histogramas_rgb_arrays_from_cv,
    compute_histograma_gris_array_from_cv
)

from ..image_utils.histogramas import calcular_stats_rgb_from_cv

from ..image_utils import (
    preparar_binaria,
    etiquetar_componentes,
    etiquetas_a_rgb,
    calcular_stats_cc
)

from ..processing import (
    suma,
    resta,
    multiplicacion,
    division,
    op_and,
    op_or,
    op_xor,
    op_not
)

from ..image_utils.modelos_color import ModelosColor

from ..image_utils.pseudocolor import (
    Pseudocolor, get_menu_items, regenerate_random,
    GRAYSCALE_OPTION, AVAILABLE_COLORMAPS, CUSTOM_CMAPS
)

from ..image_utils.conversions import (
    convertir_pil_a_cv,
    convertir_cv_a_pil
)

from ..processing.frecuencia import (
    aplicar_filtro_fft, dct_compresion, fft2_imagen
)

from ..machine_learning import CNNClassifier
import threading
from tkinter import filedialog, messagebox

from ..segmentation import Segmentador

from ..preprocessing import PreprocessingPipeline, PREPROCESSING_FUNCTIONS

from ..normalization.normalizador import Normalizador

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Procesamiento de Imágenes")
        self.root.geometry("1200x800")

        # =========================
        # Estado de la aplicación
        # =========================
        self.imagen_original_cv = None
        self.imagen_actual_cv = None
        self.imagen_resultado_cv = None
        self.imagen_gris = None

        self.imagen_original_tk = None
        self.imagen_resultado_tk = None

        self.modificar_sobre_marcha = tk.BooleanVar(value=False)

        # =========================
        # Pila de transformaciones (para visualizar secuencia)
        # =========================
        self._transformacion_stack = []  # Lista de (nombre, imagen) guardadas paso a paso
        self._transformacion_nombres = []  # Nombres de transformaciones

        # =========================
        # Historial (Deshacer / Rehacer)
        # =========================
        # Cada snapshot guarda: imagen_actual_cv, imagen_resultado_cv, imagen_gris
        self._undo_stack = []
        self._redo_stack = []
        self._history_max = 30

        # Atajos de teclado
        self.root.bind_all("<Control-z>", lambda e: self._undo())
        self.root.bind_all("<Control-y>", lambda e: self._redo())
        # =========================
        # Construcción UI
        # =========================
        self._crear_menu()
        self._crear_layout_principal()

    # ==========================================================
    # UI - MENU SUPERIOR
    # ==========================================================
    def _crear_menu(self):
        menubar = tk.Menu(self.root)

        # -------- Archivo --------
        menu_archivo = tk.Menu(menubar, tearoff=0)
        menu_archivo.add_command(label="Abrir imagen", command=self._abrir_imagen)
        menu_archivo.add_command(label="Guardar imagen", command=self._guardar_imagen)
        menu_archivo.add_separator()
        menu_archivo.add_command(label="Salir", command=self.root.quit)
        
        menubar.add_cascade(label="Archivo", menu=menu_archivo)

        # -------- Edición --------
        self.menu_edicion = tk.Menu(menubar, tearoff=0)
        self.menu_edicion.add_command(
            label="Deshacer",
            command=self._undo,
            accelerator="Ctrl+Z"
        )
        self.menu_edicion.add_command(
            label="Rehacer",
            command=self._redo,
            accelerator="Ctrl+Y"
        )
        menubar.add_cascade(label="Edición", menu=self.menu_edicion)

        # -------- Filtros --------
        self.menu_filtros = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Filtros", menu=self.menu_filtros)

        # -------- Ruido --------
        self.menu_ruido = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Ruido", menu=self.menu_ruido)

        # -------- Morfología --------
        self.menu_morfologia = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Morfología", menu=self.menu_morfologia)

        # -------- Aritméticas --------
        self.menu_aritmeticas = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Aritméticas", menu=self.menu_aritmeticas)

        # -------- Lógicas --------
        self.menu_logicas = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Lógicas", menu=self.menu_logicas)

        # -------- Histograma --------
        self.menu_histogramas = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Histogramas", menu=self.menu_histogramas)

        # -------- Estadísticas --------
        self.menu_estadisticas = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Estadísticas", menu=self.menu_estadisticas)

        # -------- Análisis --------
        self.menu_analisis = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Análisis", menu=self.menu_analisis)

        # -------- Modelos de color --------
        self.menu_modelos_color = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Modelos de color", menu=self.menu_modelos_color)
        
        # -------- Pseudocolor --------
        self.menu_pseudocolor = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Pseudocolor", menu=self.menu_pseudocolor)

        # -------- Frecuencia (Filtros FFT) --------
        self.menu_frecuencia = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Frecuencia", menu=self.menu_frecuencia)
        
        # -------- Preprocesamiento --------
        self.menu_preprocesamiento = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Preprocesamiento", menu=self.menu_preprocesamiento)
        
        # -------- Normalizacion --------
        self.menu_normalizacion = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Normalizacion", menu=self.menu_normalizacion)

        # -------- Segmentacion --------
        self.menu_segmentacion = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Segmentacion", menu=self.menu_segmentacion)
        
        # -------- Machine Learning --------
        self.menu_ml = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Machine learning", menu=self.menu_ml)

        

        self.menu_preprocesamiento.add_command(
            label="Generar preprocesamiento",
            command=self._generar_preprocesamiento
        )

        self.menu_preprocesamiento.add_command(
            label="Usar preprocesamiento",
            command=self._usar_preprocesamiento
        )

        self.root.config(menu=menubar)

        self._cargar_menu_filtros()
        self._cargar_menu_ruido()
        self._cargar_menu_morfologia()
        self._cargar_menu_aritmeticas()
        self._cargar_menu_logicas()
        self._cargar_menu_histogramas()
        self._cargar_menu_estadisticas()
        self._cargar_menu_analisis()
        self._cargar_menu_pseudocolor()
        self._cargar_menu_modelos_color()
        self._cargar_menu_frecuencia()
        self._cargar_menu_segmentacion()
        self._cargar_menu_normalizacion()
        self._cargar_menu_ml()

    # ==========================================================
    # UI - LAYOUT PRINCIPAL
    # ==========================================================
    def _crear_layout_principal(self):
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # -------- Panel imágenes --------
        frame_imagenes = ttk.Frame(main_frame)
        frame_imagenes.pack(fill=tk.BOTH, expand=True)

        # Imagen original
        frame_original = ttk.LabelFrame(frame_imagenes, text="Imagen Original")
        frame_original.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

        self.label_original = ttk.Label(frame_original)
        self.label_original.pack(expand=True)

        # Imagen editada
        frame_resultado = ttk.LabelFrame(frame_imagenes, text="Imagen Editada")
        frame_resultado.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

        self.label_resultado = ttk.Label(frame_resultado)
        self.label_resultado.pack(expand=True)

        # -------- Controles (Deshacer / Rehacer + Checkbox) --------
        frame_controles = ttk.Frame(main_frame)
        frame_controles.pack(fill=tk.X, pady=10)

        self.btn_undo = ttk.Button(
            frame_controles,
            text="Deshacer",
            command=self._undo
        )
        self.btn_undo.pack(side=tk.LEFT, padx=(0, 6))

        self.btn_redo = ttk.Button(
            frame_controles,
            text="Rehacer",
            command=self._redo
        )
        self.btn_redo.pack(side=tk.LEFT, padx=(0, 12))

        self.btn_secuencia = ttk.Button(
            frame_controles,
            text="🎬 Visualizar Secuencia",
            command=self._visualizar_secuencia
        )
        self.btn_secuencia.pack(side=tk.LEFT, padx=(0, 12))

        ttk.Checkbutton(
            frame_controles,
            text="Modificar sobre la marcha",
            variable=self.modificar_sobre_marcha,
            command=self._reset_transformacion_stack
        ).pack(side=tk.LEFT)

        # Estado inicial
        self._update_undo_redo_ui()

        # -------- Panel inferior (stats / histogramas / info extra) --------
        frame_info = ttk.LabelFrame(
            main_frame,
            text="Estadísticas / Histogramas / Info"
        )
        frame_info.pack(fill=tk.BOTH, expand=False)

        # Contenedor para Text + Scrollbar
        frame_text = ttk.Frame(frame_info)
        frame_text.pack(fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(frame_text, orient=tk.VERTICAL)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.text_info = tk.Text(
            frame_text,
            height=12,
            wrap=tk.WORD,
            yscrollcommand=scrollbar.set
        )
        self.text_info.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar.config(command=self.text_info.yview)


    # ==========================================================
    # ARCHIVO
    # ==========================================================
    def _abrir_imagen(self):
        path = filedialog.askopenfilename(
            filetypes=[
                ("Imágenes", "*.png *.jpg *.jpeg *.bmp *.tif"),
                ("Todos", "*.*")
            ]
        )

        if not path:
            return

        try:
            img_pil = Image.open(path).convert("RGB")
            img_cv = convertir_pil_a_cv(img_pil)

            self.imagen_original_cv = img_cv
            self.imagen_actual_cv = img_cv.copy()
            self.imagen_resultado_cv = None
            self.imagen_gris = None

            # Reset historial
            self._undo_stack.clear()
            self._redo_stack.clear()
            self._update_undo_redo_ui()

            self._mostrar_imagen_original(img_pil)
            self._limpiar_resultado()

        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _guardar_imagen(self):
        if self.imagen_resultado_cv is None:
            messagebox.showwarning("Aviso", "No hay imagen para guardar")
            return

        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPG", "*.jpg")]
        )

        if not path:
            return

        img_pil = convertir_cv_a_pil(self.imagen_resultado_cv)
        img_pil.save(path)

    # ==========================================================
    # VISUALIZACIÓN
    # ==========================================================
    def _mostrar_imagen_original(self, img_pil):
        img = img_pil.copy()
        img.thumbnail((500, 500))
        self.imagen_original_tk = ImageTk.PhotoImage(img)
        self.label_original.config(image=self.imagen_original_tk)

    def _mostrar_imagen_resultado(self, img_cv):
        img_pil = convertir_cv_a_pil(img_cv)
        img_pil.thumbnail((500, 500))
        self.imagen_resultado_tk = ImageTk.PhotoImage(img_pil)
        self.label_resultado.config(image=self.imagen_resultado_tk)

    def _limpiar_resultado(self):
        self.label_resultado.config(image="")
        self.imagen_resultado_tk = None

    # ==========================================================
    # OPERACIONES (PATRÓN CENTRAL)
    # ==========================================================
    def aplicar_operacion(self, funcion, *args, nombre_transformacion="Transformación"):
        if self.imagen_original_cv is None:
            messagebox.showwarning("Aviso", "Primero carga una imagen")
            return

        # Guardar estado actual para deshacer
        self._push_undo()

        if self.modificar_sobre_marcha.get():
            fuente = self.imagen_actual_cv
        else:
            fuente = self.imagen_original_cv

        try:
            resultado = funcion(fuente, *args)
        except Exception as e:
            # Si falla, revertimos el push al historial
            if self._undo_stack:
                self._undo_stack.pop()
            messagebox.showerror("Error", str(e))
            return

        # Al aplicar una nueva operación, se invalida el rehacer
        self._redo_stack.clear()

        self.imagen_resultado_cv = resultado

        if self.modificar_sobre_marcha.get():
            self.imagen_actual_cv = resultado
            # Guardar en pila de transformaciones
            self._transformacion_stack.append(resultado.copy())
            self._transformacion_nombres.append(nombre_transformacion)

        # Cualquier cache dependiente de la imagen queda inválido
        self.imagen_gris = None

        self._mostrar_imagen_resultado(resultado)
        self._update_undo_redo_ui()

    # ==========================================================
    # HISTORIAL - DESHACER / REHACER
    # ==========================================================
    def _snapshot_state(self):
        return {
            "imagen_actual_cv": None if self.imagen_actual_cv is None else self.imagen_actual_cv.copy(),
            "imagen_resultado_cv": None if self.imagen_resultado_cv is None else self.imagen_resultado_cv.copy(),
            "imagen_gris": None if self.imagen_gris is None else self.imagen_gris.copy()
        }

    def _restore_state(self, snap):
        self.imagen_actual_cv = snap.get("imagen_actual_cv", None)
        self.imagen_resultado_cv = snap.get("imagen_resultado_cv", None)
        self.imagen_gris = snap.get("imagen_gris", None)

        if self.imagen_resultado_cv is None:
            self._limpiar_resultado()
        else:
            self._mostrar_imagen_resultado(self.imagen_resultado_cv)

        self._update_undo_redo_ui()

    def _push_undo(self):
        snap = self._snapshot_state()
        self._undo_stack.append(snap)
        if len(self._undo_stack) > self._history_max:
            self._undo_stack.pop(0)

    def _undo(self):
        if not self._undo_stack:
            return

        self._redo_stack.append(self._snapshot_state())
        snap = self._undo_stack.pop()
        self._restore_state(snap)

    def _redo(self):
        if not self._redo_stack:
            return

        self._undo_stack.append(self._snapshot_state())
        snap = self._redo_stack.pop()
        self._restore_state(snap)

    def _update_undo_redo_ui(self):
        if hasattr(self, "btn_undo"):
            self.btn_undo.config(state=("normal" if self._undo_stack else "disabled"))
        if hasattr(self, "btn_redo"):
            self.btn_redo.config(state=("normal" if self._redo_stack else "disabled"))

        if hasattr(self, "menu_edicion"):
            try:
                self.menu_edicion.entryconfig("Deshacer", state=("normal" if self._undo_stack else "disabled"))
                self.menu_edicion.entryconfig("Rehacer", state=("normal" if self._redo_stack else "disabled"))
            except tk.TclError:
                pass

    def _cargar_menu_filtros(self):
        # -------- Filtros básicos --------
        self.menu_filtros.add_command(
            label="Media",
            command=self._filtro_media
        )

        self.menu_filtros.add_command(
            label="Mediana",
            command=self._filtro_mediana
        )

        self.menu_filtros.add_command(
            label="Gaussiano",
            command=self._filtro_gaussiano
        )

        self.menu_filtros.add_command(
            label="Laplaciano",
            command=self._filtro_laplaciano
        )

        self.menu_filtros.add_separator()

        # -------- Bordes --------
        self.menu_filtros.add_command(
            label="Sobel",
            command=self._filtro_sobel
        )

        self.menu_filtros.add_command(
            label="Prewitt",
            command=self._filtro_prewitt
        )

        self.menu_filtros.add_command(
            label="Roberts",
            command=self._filtro_roberts
        )

        self.menu_filtros.add_command(
            label="Canny",
            command=self._filtro_canny
        )


    # =========================
    # FILTROS - CALLBACKS
    # =========================

    def _filtro_media(self):
        k = simpledialog.askinteger(
            "Filtro Media", "Tamaño del kernel (impar):",
            minvalue=1
        )
        if k:
            self.aplicar_operacion(filtro_media, k)

    def _filtro_mediana(self):
        k = simpledialog.askinteger(
            "Filtro Mediana",
            "Tamaño del kernel (impar):",
            minvalue=3
        )

        if k is None:
            return

        if k % 2 == 0:
            messagebox.showerror(
                "Error",
                "El tamaño del kernel debe ser impar."
            )
            return

        self.aplicar_operacion(filtro_mediana, k)

    def _filtro_gaussiano(self):
        k = simpledialog.askinteger(
            "Filtro Gaussiano", "Tamaño del kernel (impar):",
            minvalue=1
        )
        if k:
            self.aplicar_operacion(filtro_gaussiano, k)

    def _filtro_laplaciano(self):
        k = simpledialog.askinteger(
            "Filtro Laplaciano", "Tamaño del kernel (3 recomendado):",
            minvalue=1
        )
        if k:
            self.aplicar_operacion(filtro_laplaciano, k)

    def _filtro_sobel(self):
        # Sobel devuelve 3 imágenes, usamos la magnitud
        def sobel_wrapper(img):
            _, _, mag = filtro_sobel(img)
            return cv2.cvtColor(mag, cv2.COLOR_GRAY2BGR)

        self.aplicar_operacion(sobel_wrapper)

    def _filtro_prewitt(self):
        def prewitt_wrapper(img):
            _, _, mag = filtro_prewitt(img)
            return cv2.cvtColor(mag, cv2.COLOR_GRAY2BGR)

        self.aplicar_operacion(prewitt_wrapper)

    def _filtro_roberts(self):
        def roberts_wrapper(img):
            _, _, mag = filtro_roberts(img)
            return cv2.cvtColor(mag, cv2.COLOR_GRAY2BGR)

        self.aplicar_operacion(roberts_wrapper)

    def _filtro_canny(self):
        t1 = simpledialog.askinteger(
            "Canny", "Umbral inferior:",
            minvalue=0
        )
        t2 = simpledialog.askinteger(
            "Canny", "Umbral superior:",
            minvalue=0
        )

        if t1 is not None and t2 is not None:
            def canny_wrapper(img):
                edges = filtro_canny(img, t1, t2)
                return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

            self.aplicar_operacion(canny_wrapper)

    def _cargar_menu_ruido(self):
        self.menu_ruido.add_command(
            label="Sal y Pimienta",
            command=self._ruido_sal_pimienta
        )

        self.menu_ruido.add_command(
            label="Solo Sal",
            command=self._ruido_sal
        )

        self.menu_ruido.add_command(
            label="Solo Pimienta",
            command=self._ruido_pimienta
        )

        self.menu_ruido.add_separator()

        self.menu_ruido.add_command(
            label="Gaussiano",
            command=self._ruido_gaussiano
        )

    # =========================
    # RUIDO - CALLBACKS
    # =========================

    def _ruido_sal_pimienta(self):
        p = simpledialog.askfloat(
            "Ruido Sal y Pimienta",
            "Probabilidad (0.0 - 0.5):",
            minvalue=0.0,
            maxvalue=0.5
        )

        if p is not None:
            self.aplicar_operacion(ruido_sal_pimienta, p)

    def _ruido_gaussiano(self):
        sigma = simpledialog.askfloat(
            "Ruido Gaussiano",
            "Sigma (desviación estándar):",
            minvalue=0.0
        )

        if sigma is not None:
            self.aplicar_operacion(ruido_gaussiano, sigma)

    def _ruido_sal(self):
        p = simpledialog.askfloat(
            "Ruido Sal",
            "Probabilidad (0.0 - 0.5):",
            minvalue=0.0,
            maxvalue=0.5
        )

        if p is not None:
            self.aplicar_operacion(ruido_sal, p)


    def _ruido_pimienta(self):
        p = simpledialog.askfloat(
            "Ruido Pimienta",
            "Probabilidad (0.0 - 0.5):",
            minvalue=0.0,
            maxvalue=0.5
        )

        if p is not None:
            self.aplicar_operacion(ruido_pimienta, p)

    def _cargar_menu_morfologia(self):
        self.menu_morfologia.delete(0, tk.END)

        # ----------------------------------
        # Morfología binaria / gris
        # ----------------------------------
        self.menu_morfologia.add_command(
            label="Erosión (gris)",
            command=self._morfologia_erosion
        )

        self.menu_morfologia.add_command(
            label="Dilatación (gris)",
            command=self._morfologia_dilatacion
        )

        self.menu_morfologia.add_separator()

        self.menu_morfologia.add_command(
            label="Apertura (gris)",
            command=self._morfologia_apertura
        )

        self.menu_morfologia.add_command(
            label="Cierre (gris)",
            command=self._morfologia_cierre
        )

        self.menu_morfologia.add_separator()

        # ----------------------------------
        # Morfología en color
        # ----------------------------------
        self.menu_morfologia.add_command(
            label="Erosión (color)",
            command=self._morfologia_erosion_color
        )

        self.menu_morfologia.add_command(
            label="Dilatación (color)",
            command=self._morfologia_dilatacion_color
        )

        self.menu_morfologia.add_command(
            label="Gradiente morfológico (color)",
            command=self._morfologia_gradiente_color
        )

        self.menu_morfologia.add_command(
            label="Top-Hat (color)",
            command=self._morfologia_tophat_color
        )

        self.menu_morfologia.add_command(
            label="Black-Hat (color)",
            command=self._morfologia_blackhat_color
        )

    def _morfologia_erosion(self):
        k = simpledialog.askinteger(
            "Erosión (gris)",
            "Tamaño del kernel:",
            minvalue=1
        )
        if k:
            self.aplicar_operacion(erosion, k)


    def _morfologia_dilatacion(self):
        k = simpledialog.askinteger(
            "Dilatación (gris)",
            "Tamaño del kernel:",
            minvalue=1
        )
        if k:
            self.aplicar_operacion(dilatacion, k)


    def _morfologia_apertura(self):
        k = simpledialog.askinteger(
            "Apertura (gris)",
            "Tamaño del kernel:",
            minvalue=1
        )

        if k:
            self.aplicar_operacion(apertura, k)


    def _morfologia_cierre(self):
        k = simpledialog.askinteger(
            "Cierre (gris)",
            "Tamaño del kernel:",
            minvalue=1
        )

        if k:
            self.aplicar_operacion(cierre, k)

    def _morfologia_erosion_color(self):
        k = simpledialog.askinteger(
            "Erosión (color)",
            "Tamaño del kernel:",
            minvalue=1
        )
        if k:
            self.aplicar_operacion(erosion_color, k)


    def _morfologia_dilatacion_color(self):
        k = simpledialog.askinteger(
            "Dilatación (color)",
            "Tamaño del kernel:",
            minvalue=1
        )
        if k:
            self.aplicar_operacion(dilatacion_color, k)


    def _morfologia_gradiente_color(self):
        k = simpledialog.askinteger(
            "Gradiente morfológico (color)",
            "Tamaño del kernel:",
            minvalue=1
        )
        if k:
            self.aplicar_operacion(gradiente_morfologico_color, k)


    def _morfologia_tophat_color(self):
        k = simpledialog.askinteger(
            "Top-Hat (color)",
            "Tamaño del kernel:",
            minvalue=1
        )
        if k:
            self.aplicar_operacion(tophat_color, k)


    def _morfologia_blackhat_color(self):
        k = simpledialog.askinteger(
            "Black-Hat (color)",
            "Tamaño del kernel:",
            minvalue=1
        )
        if k:
            self.aplicar_operacion(blackhat_color, k)

    def _cargar_menu_histogramas(self):
        self.menu_histogramas.add_command(
            label="Histograma RGB",
            command=self._histograma_rgb
        )

        self.menu_histogramas.add_command(
            label="Histograma Escala de Grises",
            command=self._histograma_gris
        )

    def _seleccionar_imagen_para_analisis(self):
        opcion = messagebox.askquestion(
            "Seleccionar imagen",
            "¿Deseas usar la imagen EDITADA?\n\n"
            "Sí → Imagen editada\n"
            "No → Imagen original"
        )

        if opcion == "yes":
            if self.imagen_resultado_cv is not None:
                return self.imagen_resultado_cv
            else:
                messagebox.showwarning(
                    "Aviso",
                    "No hay imagen editada, se usará la original"
                )

        return self.imagen_original_cv

    def _histograma_rgb(self):
        if self.imagen_original_cv is None:
            messagebox.showwarning(
                "Aviso",
                "Primero debes cargar una imagen"
            )
            return

        img = self._seleccionar_imagen_para_analisis()
        if img is None:
            return

        histos = compute_histogramas_rgb_arrays_from_cv(img)

        ventana = tk.Toplevel(self.root)
        ventana.title("Histograma RGB")

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(histos["Rojo"], label="Rojo")
        ax.plot(histos["Verde"], label="Verde")
        ax.plot(histos["Azul"], label="Azul")

        ax.set_title("Histograma RGB")
        ax.set_xlabel("Intensidad")
        ax.set_ylabel("Frecuencia")
        ax.legend()
        ax.grid(True)

        canvas = FigureCanvasTkAgg(fig, master=ventana)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _histograma_gris(self):
        if self.imagen_original_cv is None:
            messagebox.showwarning(
                "Aviso",
                "Primero debes cargar una imagen"
            )
            return

        img = self._seleccionar_imagen_para_analisis()
        if img is None:
            return

        hist = compute_histograma_gris_array_from_cv(img)

        ventana = tk.Toplevel(self.root)
        ventana.title("Histograma en Escala de Grises")

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(hist, color="black")

        ax.set_title("Histograma Escala de Grises")
        ax.set_xlabel("Intensidad")
        ax.set_ylabel("Frecuencia")
        ax.grid(True)

        canvas = FigureCanvasTkAgg(fig, master=ventana)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _cargar_menu_estadisticas(self):
        self.menu_estadisticas.add_command(
            label="Calcular estadísticas",
            command=self._mostrar_estadisticas
        )

    def _mostrar_estadisticas(self):
        if self.imagen_original_cv is None:
            messagebox.showwarning(
                "Aviso",
                "Primero debes cargar una imagen"
            )
            return

        stats_original = calcular_stats_rgb_from_cv(
            self.imagen_original_cv
        )

        stats_editada = None
        if self.imagen_resultado_cv is not None:
            stats_editada = calcular_stats_rgb_from_cv(
                self.imagen_resultado_cv
            )

        self._ventana_estadisticas(
            stats_original,
            stats_editada
        )

    def _ventana_estadisticas(self, stats_original, stats_editada):
        ventana = tk.Toplevel(self.root)
        ventana.title("Estadísticas de la imagen")

        txt = tk.Text(ventana, width=70, height=25)
        txt.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        txt.insert(tk.END, "=== IMAGEN ORIGINAL ===\n")
        self._imprimir_stats(txt, stats_original)

        if stats_editada is not None:
            txt.insert(tk.END, "\n=== IMAGEN EDITADA ===\n")
            self._imprimir_stats(txt, stats_editada)
        else:
            txt.insert(
                tk.END,
                "\nNo hay imagen editada para comparar\n"
            )

        txt.config(state=tk.DISABLED)

    def _imprimir_stats(self, txt, stats):
        for canal, valores in stats.items():
            txt.insert(tk.END, f"\nCanal {canal}:\n")
            for nombre, valor in valores.items():
                txt.insert(
                    tk.END,
                    f"  {nombre}: {valor:.4f}\n"
                )

    def _cc_binarizar(self):
        if self.imagen_original_cv is None:
            messagebox.showwarning("Aviso", "Primero carga una imagen.")
            return

        umbral = simpledialog.askinteger(
            "Binarización",
            "Umbral (Cancelar = Otsu):",
            minvalue=0, maxvalue=255
        )

        bin_img = preparar_binaria(self.imagen_original_cv, umbral)
        bin_rgb = cv2.cvtColor(bin_img, cv2.COLOR_GRAY2BGR)

        self._actualizar_resultado(bin_rgb)
        self.text_info.delete("1.0", tk.END)
        self.text_info.insert(tk.END, "Imagen binarizada.\n")

    def _cc_etiquetar(self, vecindad=8):
        if self.imagen_original_cv is None:
            messagebox.showwarning("Aviso", "Primero carga una imagen.")
            return

        bin_img = preparar_binaria(self.imagen_original_cv)
        etiquetas, num_obj = etiquetar_componentes(bin_img, vecindad)
        rgb = etiquetas_a_rgb(etiquetas)

        self._actualizar_resultado(rgb)

        stats = calcular_stats_cc(etiquetas)

        self.text_info.delete("1.0", tk.END)
        self.text_info.insert(
            tk.END,
            f"Componentes conexas (vecindad {vecindad})\n"
            f"Objetos detectados: {num_obj}\n\n"
        )

        for idx, area, per in stats:
            self.text_info.insert(
                tk.END,
                f"Objeto {idx}: Área={area}px, Perímetro={per:.2f}px\n"
            )

    def _cargar_menu_analisis(self):

        self.menu_analisis.add_command(
            label="Etiquetar (Vecindad 4)",
            command=lambda: self._cc_etiquetar(vecindad=4)
        )

        self.menu_analisis.add_command(
            label="Etiquetar (Vecindad 8)",
            command=lambda: self._cc_etiquetar(vecindad=8)
        )

    def _operacion_doble(self, funcion):
        if self.imagen_original_cv is None:
            messagebox.showwarning("Aviso", "Primero carga una imagen base.")
            return

        ruta = filedialog.askopenfilename(
            title="Seleccione la segunda imagen"
        )

        if not ruta:
            return

        img2 = cv2.imread(ruta)

        if img2 is None:
            messagebox.showerror("Error", "No se pudo cargar la segunda imagen.")
            return

        # Redimensionar si es necesario
        if img2.shape != self.imagen_original_cv.shape:
            img2 = cv2.resize(
                img2,
                (self.imagen_original_cv.shape[1], self.imagen_original_cv.shape[0])
            )

        fuente = (
            self.imagen_resultado_cv
            if self.modificar_sobre_marcha.get() and self.imagen_resultado_cv is not None
            else self.imagen_original_cv
        )

        resultado = funcion(fuente, img2)

        self._actualizar_resultado(resultado)

    def _operacion_not(self):
        if self.imagen_original_cv is None:
            messagebox.showwarning("Aviso", "Primero carga una imagen.")
            return

        fuente = (
            self.imagen_resultado_cv
            if self.modificar_sobre_marcha.get() and self.imagen_resultado_cv is not None
            else self.imagen_original_cv
        )

        resultado = op_not(fuente)
        self._actualizar_resultado(resultado)

    def _cargar_menu_aritmeticas(self):
        self.menu_aritmeticas.add_command(
            label="Suma",
            command=lambda: self._operacion_doble(suma)
        )
        self.menu_aritmeticas.add_command(
            label="Resta",
            command=lambda: self._operacion_doble(resta)
        )
        self.menu_aritmeticas.add_command(
            label="Multiplicación",
            command=lambda: self._operacion_doble(multiplicacion)
        )
        self.menu_aritmeticas.add_command(
            label="División",
            command=lambda: self._operacion_doble(division)
        )


    def _cargar_menu_logicas(self):
        self.menu_logicas.add_command(
            label="AND",
            command=lambda: self._operacion_doble(op_and)
        )
        self.menu_logicas.add_command(
            label="OR",
            command=lambda: self._operacion_doble(op_or)
        )
        self.menu_logicas.add_command(
            label="XOR",
            command=lambda: self._operacion_doble(op_xor)
        )
        self.menu_logicas.add_command(
            label="NOT",
            command=self._operacion_not
        )

    def _actualizar_resultado(self, resultado_cv, nombre_transformacion="Transformación"):
        """
        Actualiza resultado y registra en pila de transformaciones si "Modificar sobre la marcha" está activo.
        """
        if self.imagen_original_cv is None:
            messagebox.showwarning("Aviso", "Primero carga una imagen")
            return

        self._push_undo()
        self._redo_stack.clear()

        self.imagen_resultado_cv = resultado_cv
        if self.modificar_sobre_marcha.get():
            self.imagen_actual_cv = resultado_cv
            # Guardar en pila de transformaciones
            self._transformacion_stack.append(resultado_cv.copy())
            self._transformacion_nombres.append(nombre_transformacion)

        self.imagen_gris = None

        self._mostrar_imagen_resultado(resultado_cv)
        self._update_undo_redo_ui()

    def _cargar_menu_pseudocolor(self):
        """
        Carga el menú de pseudocolores: Escala de grises + OpenCV + Custom + Random regenerable.
        """

        # Limpiar por si se recarga
        self.menu_pseudocolor.delete(0, tk.END)

        # Obtener lista completa de opciones
        menu_items = get_menu_items()

        for item in menu_items:
            if item == GRAYSCALE_OPTION:
                # Escala de grises
                self.menu_pseudocolor.add_command(
                    label=item,
                    command=lambda n=item: self._pseudocolor_aplicar(n)
                )
                self.menu_pseudocolor.add_separator()
            elif item in AVAILABLE_COLORMAPS:
                # Colormaps OpenCV
                self.menu_pseudocolor.add_command(
                    label=f"OpenCV - {item}",
                    command=lambda n=item: self._pseudocolor_aplicar(n)
                )
            elif item == "Random (custom)":
                # Random regenerable
                self.menu_pseudocolor.add_command(
                    label=f"🎲 {item}",
                    command=lambda n=item: self._pseudocolor_aplicar(n)
                )
                self.menu_pseudocolor.add_command(
                    label="  → Regenerar Random",
                    command=self._pseudocolor_regenerate_random
                )
            elif item in CUSTOM_CMAPS:
                # Otros custom (como Pastel)
                self.menu_pseudocolor.add_command(
                    label=f"Custom - {item}",
                    command=lambda n=item: self._pseudocolor_aplicar(n)
                )

        self.menu_pseudocolor.add_separator()
        self.menu_pseudocolor.add_command(
            label="Personalizado dinámico...",
            command=self._pseudocolor_personalizado
        )

    def _pseudocolor_aplicar(self, opcion: str):
        """Aplica pseudocolor según la opción seleccionada."""
        if self.imagen_original_cv is None:
            messagebox.showwarning("Aviso", "Primero carga una imagen.")
            return

        # Obtener gris si no existe
        if self.imagen_gris is None:
            self.imagen_gris = cv2.cvtColor(
                self.imagen_original_cv, cv2.COLOR_BGR2GRAY
            )

        pc = Pseudocolor(self.imagen_gris)

        if opcion == GRAYSCALE_OPTION:
            # Sin pseudocolor, solo gris
            self._actualizar_resultado(self.imagen_gris)
            self.text_info.delete("1.0", tk.END)
            self.text_info.insert(tk.END, "Escala de grises (sin pseudocolor).\n")
        elif opcion in AVAILABLE_COLORMAPS:
            # Colormap OpenCV
            resultado = pc.aplicar_opencv(opcion)
            resultado_rgb = cv2.cvtColor(resultado, cv2.COLOR_BGR2RGB)
            self._actualizar_resultado(resultado_rgb)
            self.text_info.delete("1.0", tk.END)
            self.text_info.insert(tk.END, f"Pseudocolor OpenCV: {opcion}\n")
        elif opcion in CUSTOM_CMAPS:
            # Colormap custom (Pastel, Random, etc.)
            resultado = pc.aplicar_custom(opcion)
            self._actualizar_resultado(resultado)
            self.text_info.delete("1.0", tk.END)
            self.text_info.insert(tk.END, f"Pseudocolor Custom: {opcion}\n")
        else:
            messagebox.showerror("Error", f"Opción no reconocida: {opcion}")

    def _pseudocolor_regenerate_random(self):
        """Regenera un nuevo colormap aleatorio."""
        regenerate_random()
        messagebox.showinfo("Éxito", "Colormap Random regenerado.\nVuelve a aplicarlo desde el menú.")

    def _pseudocolor_personalizado(self):
        """Aplica un pseudocolor personalizado dinámico."""
        if self.imagen_original_cv is None:
            messagebox.showwarning("Aviso", "Primero carga una imagen.")
            return

        if self.imagen_gris is None:
            self.imagen_gris = cv2.cvtColor(
                self.imagen_original_cv, cv2.COLOR_BGR2GRAY
            )

        # Ejemplo: pedir 3 colores base
        colores = [
            (1.0, 0.0, 0.0),  # rojo
            (1.0, 1.0, 0.0),  # amarillo
            (0.0, 0.0, 1.0)   # azul
        ]

        pc = Pseudocolor(self.imagen_gris)
        resultado = pc.aplicar_personalizado(colores)

        self._actualizar_resultado(resultado)

        self.text_info.delete("1.0", tk.END)
        self.text_info.insert(
            tk.END, "Pseudocolor personalizado aplicado.\n"
        )

    def _cargar_menu_modelos_color(self):
        self.menu_modelos_color.delete(0, tk.END)

        # -------- Grises --------
        self.menu_modelos_color.add_command(
            label="Escala de grises",
            command=self._modelo_grises
        )

        # -------- RGB --------
        menu_rgb = tk.Menu(self.menu_modelos_color, tearoff=0)
        self.menu_modelos_color.add_cascade(label="RGB", menu=menu_rgb)

        menu_rgb.add_command(label="Canal R", command=lambda: self._modelo_rgb("R"))
        menu_rgb.add_command(label="Canal G", command=lambda: self._modelo_rgb("G"))
        menu_rgb.add_command(label="Canal B", command=lambda: self._modelo_rgb("B"))

        # -------- CMYK --------
        menu_cmyk = tk.Menu(self.menu_modelos_color, tearoff=0)
        self.menu_modelos_color.add_cascade(label="CMYK", menu=menu_cmyk)

        for canal in ["C", "M", "Y", "K"]:
            menu_cmyk.add_command(
                label=f"Canal {canal}",
                command=lambda c=canal: self._modelo_cmyk(c)
            )

        # -------- HSL --------
        menu_hsl = tk.Menu(self.menu_modelos_color, tearoff=0)
        self.menu_modelos_color.add_cascade(label="HSL", menu=menu_hsl)

        for canal in ["H", "S", "L"]:
            menu_hsl.add_command(
                label=f"Canal {canal}",
                command=lambda c=canal: self._modelo_hsl(c)
            )

        # -------- Binarización --------
        menu_bin = tk.Menu(self.menu_modelos_color, tearoff=0)
        self.menu_modelos_color.add_cascade(label="Binarización", menu=menu_bin)

        menu_bin.add_command(
            label="Umbral manual",
            command=self._binarizacion_umbral
        )

        menu_bin.add_command(
            label="Umbral inverso",
            command=self._binarizacion_umbral_inv
        )

        menu_bin.add_command(
            label="Umbral truncado",
            command=self._binarizacion_truncado
        )

        menu_bin.add_command(
            label="Umbral a cero",
            command=self._binarizacion_a_cero
        )

        menu_bin.add_separator()

        menu_bin.add_command(
            label="Otsu",
            command=self._binarizacion_otsu
        )

        menu_bin.add_separator()

        menu_bin.add_command(
            label="Adaptativo (Media)",
            command=self._binarizacion_adaptativa_media
        )

        menu_bin.add_command(
            label="Adaptativo (Gauss)",
            command=self._binarizacion_adaptativa_gauss
        )

    def _modelo_grises(self):
        self.aplicar_operacion(ModelosColor.a_grises)

    def _modelo_rgb(self, canal):
        self.aplicar_operacion(
            lambda img: ModelosColor.canal_rgb(img, canal)
        )

    def _modelo_cmyk(self, canal):
        self.aplicar_operacion(
            lambda img: ModelosColor.canal_cmyk(img, canal)
        )

    def _modelo_hsl(self, canal):
        self.aplicar_operacion(
            lambda img: ModelosColor.canal_hsl(img, canal)
        )

    def _binarizacion_umbral(self):
        t = simpledialog.askinteger(
            "Binarización",
            "Valor de umbral (0–255):",
            minvalue=0,
            maxvalue=255
        )

        if t is not None:
            self.aplicar_operacion(
                lambda img: ModelosColor.binarizar_umbral(img, t)
            )

    def _binarizacion_otsu(self):
        self.aplicar_operacion(ModelosColor.binarizar_otsu)

    def _binarizacion_umbral_inv(self):
        t = simpledialog.askinteger(
            "Binarización inversa",
            "Valor de umbral (0–255):",
            minvalue=0,
            maxvalue=255
        )

        if t is not None:
            self.aplicar_operacion(
                lambda img: ModelosColor.binarizar_umbral_inverso(img, t)
            )


    def _binarizacion_truncado(self):
        t = simpledialog.askinteger(
            "Umbral truncado",
            "Valor de umbral (0–255):",
            minvalue=0,
            maxvalue=255
        )

        if t is not None:
            self.aplicar_operacion(
                lambda img: ModelosColor.binarizar_truncado(img, t)
            )


    def _binarizacion_a_cero(self):
        t = simpledialog.askinteger(
            "Umbral a cero",
            "Valor de umbral (0–255):",
            minvalue=0,
            maxvalue=255
        )

        if t is not None:
            self.aplicar_operacion(
                lambda img: ModelosColor.binarizar_a_cero(img, t)
            )


    def _binarizacion_adaptativa_media(self):
        block = simpledialog.askinteger(
            "Adaptativa (Media)",
            "Tamaño de bloque (impar):",
            minvalue=3
        )

        if block and block % 2 == 1:
            c = simpledialog.askinteger(
                "Adaptativa (Media)",
                "Constante C:",
                initialvalue=2
            )

            if c is not None:
                self.aplicar_operacion(
                    lambda img: ModelosColor.binarizar_adaptativa_media(img, block, c)
                )


    def _binarizacion_adaptativa_gauss(self):
        block = simpledialog.askinteger(
            "Adaptativa (Gauss)",
            "Tamaño de bloque (impar):",
            minvalue=3
        )

        if block and block % 2 == 1:
            c = simpledialog.askinteger(
                "Adaptativa (Gauss)",
                "Constante C:",
                initialvalue=2
            )

            if c is not None:
                self.aplicar_operacion(
                    lambda img: ModelosColor.binarizar_adaptativa_gauss(img, block, c)
                )

    def _cargar_menu_ml(self):
        menu_ml = tk.Menu(self.menu_ml, tearoff=0)

        menu_ml.add_command(
            label="Entrenar modelo",
            command=self._ml_entrenar
        )

        menu_ml.add_command(
            label="Probar modelo",
            command=self._ml_probar
        )

        self.menu_ml .add_cascade(label="ML", menu=menu_ml)

    def _ml_entrenar(self):
        dataset_dir = filedialog.askdirectory(
            title="Selecciona la carpeta del dataset"
        )
        if not dataset_dir:
            return

        def tarea():
            try:
                clf = CNNClassifier()
                # Pasar save_path="ml_model" para guardar solo ahí y generar reporte
                clf.train(dataset_dir, epochs=10, save_path="ml_model")

                messagebox.showinfo(
                    "ML",
                    "Entrenamiento finalizado.\n\n"
                    "Modelo guardado en: ml_model/\n"
                    "Reporte generado en: ml_model/training_report.txt"
                )
            except Exception as e:
                messagebox.showerror("Error", str(e))

        threading.Thread(target=tarea, daemon=True).start()

    def _ml_probar(self):
        # Seleccionar modelo .h5 disponible en ml_model
        clf_probe = CNNClassifier()
        model_dir = clf_probe.default_model_dir
        os.makedirs(model_dir, exist_ok=True)

        modelos_disponibles = [f for f in os.listdir(model_dir) if f.lower().endswith(".h5")]
        if not modelos_disponibles:
            messagebox.showerror("Modelos", "No se encontraron modelos .h5 en ml_model")
            return

        model_file = filedialog.askopenfilename(
            title="Selecciona un modelo (.h5)",
            initialdir=model_dir,
            filetypes=[("Modelos Keras", "*.h5")]
        )
        if not model_file:
            return

        image_paths = filedialog.askopenfilenames(
            title="Selecciona imágenes",
            filetypes=[("Imágenes", "*.jpg *.png *.jpeg")]
        )
        if not image_paths:
            return

        # Mostrar vista previa de las imágenes seleccionadas
        self._mostrar_preview_imagenes(image_paths)

        def tarea():
            try:
                import time
                
                clf = CNNClassifier()
                clf.load(os.path.dirname(model_file))
                
                # Nombre del modelo seleccionado
                model_name = os.path.basename(model_file)
                
                # Medir tiempo de clasificación
                inicio = time.time()
                results = clf.predict(image_paths)
                tiempo_total = time.time() - inicio
                tiempo_promedio = tiempo_total / len(image_paths)

                # Construir texto con formato mejorado
                texto = "═" * 60 + "\n"
                texto += "RESULTADOS DE CLASIFICACIÓN\n"
                texto += "═" * 60 + "\n\n"
                texto += f"Modelo usado: {model_name}\n"
                texto += f"Imágenes procesadas: {len(image_paths)}\n"
                texto += f"Tiempo promedio: {tiempo_promedio:.3f} segundos/imagen\n"
                texto += "═" * 60 + "\n\n"

                for idx, r in enumerate(results, 1):
                    path = r["image"]
                    clase = r["predicted_class"]
                    confianza = r["confidence"]
                    top = r["top_predictions"]

                    nombre = os.path.basename(path)

                    texto += f"[{idx}] IMAGEN: {nombre}\n"
                    texto += f"Ruta: {path}\n"
                    texto += "─" * 60 + "\n"
                    texto += f"✓ CLASIFICACIÓN: {clase} ({confianza:.2f}%)\n\n"
                    texto += "TOP 5 COINCIDENCIAS:\n"
                    
                    for i, p in enumerate(top[:5], 1):
                        barra = "█" * int(p['confidence'] / 5)  # Barra visual
                        texto += f"  {i}. {p['class']:<15} {p['confidence']:>6.2f}% {barra}\n"

                    texto += "\n" + "═" * 60 + "\n\n"

                messagebox.showinfo("Predicción ML", texto)

            except Exception as e:
                messagebox.showerror("Error", str(e))

        threading.Thread(target=tarea, daemon=True).start()

    def _mostrar_preview_imagenes(self, image_paths):
        """Despliega una ventana con miniaturas de las imágenes seleccionadas."""
        win = tk.Toplevel(self.root)
        win.title("Imágenes seleccionadas")

        contenedor = tk.Frame(win)
        contenedor.pack(fill="both", expand=True)

        canvas = tk.Canvas(contenedor, width=520, height=420)
        scrollbar = ttk.Scrollbar(contenedor, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)

        inner = tk.Frame(canvas)
        inner.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Guardar referencias para evitar que las imágenes se recolecten
        if not hasattr(self, "_preview_thumbs"):
            self._preview_thumbs = []
        self._preview_thumbs.clear()

        for path in image_paths:
            try:
                img = Image.open(path)
                img.thumbnail((160, 160))
                photo = ImageTk.PhotoImage(img)
                self._preview_thumbs.append(photo)

                marco = tk.Frame(inner, pady=6, padx=8)
                tk.Label(marco, image=photo, compound="top", text=os.path.basename(path)).pack()
                tk.Label(marco, text=path, wraplength=480, justify="left", fg="#444").pack(anchor="w")
                marco.pack(anchor="w")
            except Exception as exc:
                tk.Label(inner, text=f"No se pudo cargar {path}: {exc}", fg="red").pack(anchor="w")

    def _generar_preprocesamiento(self):
        nombre = simpledialog.askstring(
            "Nuevo preprocesamiento",
            "Nombre del preprocesamiento:"
        )
        if not nombre:
            return

        pipeline = PreprocessingPipeline(nombre)

        # Ejemplo fijo (luego lo haces dinámico con UI)
        pipeline.add_step("grises", PREPROCESSING_FUNCTIONS["grises"])
        pipeline.add_step("mediana", PREPROCESSING_FUNCTIONS["mediana"], k=5)
        pipeline.add_step("binarizar_otsu", PREPROCESSING_FUNCTIONS["binarizar_otsu"])

        pipeline.save("preprocesamientos")

        messagebox.showinfo(
            "Preprocesamiento",
            f"Preprocesamiento '{nombre}' guardado correctamente."
        )

    def _usar_preprocesamiento(self):
        # Seleccionar archivo de preprocesamiento
        prep_path = filedialog.askopenfilename(
            title="Selecciona un preprocesamiento",
            initialdir="preprocesamientos",
            filetypes=[("Preprocesamiento", "*.json")]
        )
        if not prep_path:
            return

        # Carpeta origen
        input_dir = filedialog.askdirectory(
            title="Carpeta con imágenes originales"
        )
        if not input_dir:
            return

        # Carpeta destino
        output_dir = filedialog.askdirectory(
            title="Carpeta destino para imágenes preprocesadas"
        )
        if not output_dir:
            return

        try:
            pipeline = PreprocessingPipeline.load(
                prep_path,
                PREPROCESSING_FUNCTIONS
            )

            pipeline.apply_to_folder(input_dir, output_dir)

            messagebox.showinfo(
                "Preprocesamiento",
                "Preprocesamiento aplicado correctamente."
            )

        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _cargar_menu_segmentacion(self):
        self.menu_segmentacion.delete(0, tk.END)

        self.menu_segmentacion.add_command(
            label="Detección de formas",
            command=self._segmentacion_formas
        )

        self.menu_segmentacion.add_separator()

        self.menu_segmentacion.add_command(
            label="Hit-or-Miss",
            command=self._segmentacion_hit_or_miss
        )


    def _segmentacion_formas(self):
        if self.imagen_original_cv is None:
            messagebox.showwarning("Aviso", "Primero carga una imagen.")
            return

        seg = Segmentador(self.imagen_original_cv)
        resultado, objetos = seg.formas_geometricas()

        self._actualizar_resultado(resultado)

        self.text_info.delete("1.0", tk.END)
        self.text_info.insert(
            tk.END, f"Objetos detectados: {len(objetos)}\n"
        )

        for o in objetos:
            self.text_info.insert(
                tk.END,
                f"- Objeto {o['id']}: {o['forma']} | Área={o['area']:.0f}\n"
            )

    def _segmentacion_hit_or_miss(self):
        if self.imagen_original_cv is None:
            messagebox.showwarning("Aviso", "No hay imagen para segmentar.")
            return

        ruta = filedialog.askopenfilename(
            title="Selecciona imagen patrón (binaria)",
            filetypes=[("Imágenes", "*.png *.jpg *.jpeg")]
        )
        if not ruta:
            return

        patron = cv2.imread(ruta)

        seg = Segmentador(self.imagen_original_cv)
        resultado, encontrados = seg.hit_or_miss(patron)

        self._actualizar_resultado(resultado)

        self.text_info.delete("1.0", tk.END)
        self.text_info.insert(
            tk.END,
            f"Hit-or-Miss aplicado\nPatrones encontrados: {encontrados}\n"
        )

    def _cargar_menu_normalizacion(self):
        self.menu_normalizacion.delete(0, tk.END)

        # -----------------------------
        # Resolución
        # -----------------------------
        self.menu_normalizacion.add_command(
            label="Normalizar resolución (256x256)",
            command=lambda: self._normalizar_resolucion(256, 256)
        )

        self.menu_normalizacion.add_command(
            label="Normalizar resolución (512x512)",
            command=lambda: self._normalizar_resolucion(512, 512)
        )

        self.menu_normalizacion.add_separator()

        # -----------------------------
        # Formato de archivo
        # -----------------------------
        self.menu_normalizacion.add_command(
            label="Guardar como PNG",
            command=lambda: self._guardar_formato("png")
        )

        self.menu_normalizacion.add_command(
            label="Guardar como JPG",
            command=lambda: self._guardar_formato("jpg")
        )

        self.menu_normalizacion.add_command(
            label="Guardar como BMP",
            command=lambda: self._guardar_formato("bmp")
        )

    def _normalizar_resolucion(self, w, h):
        if self.imagen_original_cv is None:
            messagebox.showwarning("Aviso", "Primero carga una imagen.")
            return

        norm = Normalizador(self.imagen_original_cv)
        resultado = norm.normalizar_resolucion(w, h)

        self._actualizar_resultado(resultado)

        self.text_info.delete("1.0", tk.END)
        self.text_info.insert(
            tk.END, f"Resolución normalizada a {w}x{h}.\n"
        )

    def _guardar_formato(self, formato):
        if self.imagen_resultado_cv is None:
            messagebox.showwarning(
                "Aviso", "No hay imagen para guardar."
            )
            return

        ruta = filedialog.asksaveasfilename(
            defaultextension=f".{formato}",
            filetypes=[(formato.upper(), f"*.{formato}")]
        )

        if not ruta:
            return

        norm = Normalizador(self.imagen_resultado_cv)
        ruta_final = norm.guardar_como(ruta, formato)

        self.text_info.delete("1.0", tk.END)
        self.text_info.insert(
            tk.END, f"Imagen guardada como {ruta_final}\n"
        )

    # ==========================================================
    # FRECUENCIA - FILTROS FFT (PASA BAJAS / PASA ALTAS)
    # ==========================================================
    def _cargar_menu_frecuencia(self):
        """Carga el menú de filtros en frecuencia (FFT)."""
        self.menu_frecuencia.delete(0, tk.END)

        # Pasa bajas
        self.menu_frecuencia.add_command(
            label="Pasa Bajas - Ideal",
            command=lambda: self._aplicar_fft_lowpass("ideal")
        )
        self.menu_frecuencia.add_command(
            label="Pasa Bajas - Gaussiano",
            command=lambda: self._aplicar_fft_lowpass("gaussiano")
        )
        self.menu_frecuencia.add_command(
            label="Pasa Bajas - Butterworth",
            command=lambda: self._aplicar_fft_lowpass("butterworth")
        )

        self.menu_frecuencia.add_separator()

        # Pasa altas
        self.menu_frecuencia.add_command(
            label="Pasa Altas - Ideal",
            command=lambda: self._aplicar_fft_highpass("ideal")
        )
        self.menu_frecuencia.add_command(
            label="Pasa Altas - Gaussiano",
            command=lambda: self._aplicar_fft_highpass("gaussiano")
        )
        self.menu_frecuencia.add_command(
            label="Pasa Altas - Butterworth",
            command=lambda: self._aplicar_fft_highpass("butterworth")
        )

        self.menu_frecuencia.add_separator()

        # DCT Compresión
        self.menu_frecuencia.add_command(
            label="Compresión DCT",
            command=self._aplicar_dct_compresion
        )

    def _aplicar_fft_lowpass(self, filtro_tipo: str):
        """Aplica filtro pasa bajas en frecuencia."""
        if self.imagen_original_cv is None:
            messagebox.showwarning("Aviso", "Primero carga una imagen.")
            return

        # Diálogo para parámetros
        cutoff = simpledialog.askfloat(
            "Filtro Pasa Bajas",
            "Radio de corte (0.05 - 0.5):",
            initialvalue=0.15,
            minvalue=0.01,
            maxvalue=0.5
        )
        if cutoff is None:
            return

        orden = 2
        if filtro_tipo == "butterworth":
            orden = simpledialog.askinteger(
                "Orden Butterworth",
                "Orden (1-10):",
                initialvalue=2,
                minvalue=1,
                maxvalue=10
            )
            if orden is None:
                return

        try:
            resultado, _ = aplicar_filtro_fft(
                self.imagen_original_cv,
                filtro=filtro_tipo,
                tipo="lowpass",
                cutoff=cutoff,
                orden=orden
            )
            self._actualizar_resultado(resultado)
            self.text_info.delete("1.0", tk.END)
            self.text_info.insert(
                tk.END,
                f"Filtro pasa bajas {filtro_tipo.upper()} aplicado.\n"
                f"Cutoff: {cutoff}\nOrden: {orden}\n"
            )
        except Exception as e:
            messagebox.showerror("Error", f"Error al aplicar filtro: {str(e)}")

    def _aplicar_fft_highpass(self, filtro_tipo: str):
        """Aplica filtro pasa altas en frecuencia."""
        if self.imagen_original_cv is None:
            messagebox.showwarning("Aviso", "Primero carga una imagen.")
            return

        # Diálogo para parámetros
        cutoff = simpledialog.askfloat(
            "Filtro Pasa Altas",
            "Radio de corte (0.05 - 0.5):",
            initialvalue=0.15,
            minvalue=0.01,
            maxvalue=0.5
        )
        if cutoff is None:
            return

        orden = 2
        if filtro_tipo == "butterworth":
            orden = simpledialog.askinteger(
                "Orden Butterworth",
                "Orden (1-10):",
                initialvalue=2,
                minvalue=1,
                maxvalue=10
            )
            if orden is None:
                return

        try:
            resultado, _ = aplicar_filtro_fft(
                self.imagen_original_cv,
                filtro=filtro_tipo,
                tipo="highpass",
                cutoff=cutoff,
                orden=orden
            )
            self._actualizar_resultado(resultado)
            self.text_info.delete("1.0", tk.END)
            self.text_info.insert(
                tk.END,
                f"Filtro pasa altas {filtro_tipo.upper()} aplicado.\n"
                f"Cutoff: {cutoff}\nOrden: {orden}\n"
            )
        except Exception as e:
            messagebox.showerror("Error", f"Error al aplicar filtro: {str(e)}")

    def _aplicar_dct_compresion(self):
        """Aplica compresión DCT."""
        if self.imagen_original_cv is None:
            messagebox.showwarning("Aviso", "Primero carga una imagen.")
            return

        q_factor = simpledialog.askfloat(
            "Compresión DCT",
            "Factor de cuantización (0.1 - 1.0):",
            initialvalue=0.5,
            minvalue=0.1,
            maxvalue=1.0
        )
        if q_factor is None:
            return

        try:
            resultado, psnr = dct_compresion(self.imagen_original_cv, q_factor=q_factor)
            self._actualizar_resultado(resultado)
            self.text_info.delete("1.0", tk.END)
            self.text_info.insert(
                tk.END,
                f"Compresión DCT aplicada.\n"
                f"Factor q: {q_factor}\n"
                f"PSNR: {psnr:.2f} dB\n"
            )
        except Exception as e:
            messagebox.showerror("Error", f"Error al comprimir: {str(e)}")

    # ==========================================================
    # VISUALIZACIÓN DE SECUENCIA DE TRANSFORMACIONES
    # ==========================================================
    def _reset_transformacion_stack(self):
        """Resetea la pila de transformaciones cuando se (des)marca 'Modificar sobre la marcha'."""
        if not self.modificar_sobre_marcha.get():
            self._transformacion_stack.clear()
            self._transformacion_nombres.clear()
        else:
            # Si se habilita, comenzar nueva secuencia desde imagen actual
            self._transformacion_stack = [self.imagen_actual_cv.copy()] if self.imagen_actual_cv is not None else []
            self._transformacion_nombres = ["Original (Inicio)"] if self.imagen_actual_cv is not None else []

    def _visualizar_secuencia(self):
        """Abre ventana con visualización paso a paso de todas las transformaciones."""
        if not self._transformacion_stack:
            messagebox.showinfo("Secuencia", "No hay transformaciones registradas.\n\nMarca 'Modificar sobre la marcha' y aplica operaciones.")
            return

        # Crear ventana nueva
        ventana_sec = tk.Toplevel(self.root)
        ventana_sec.title("Secuencia de Transformaciones")
        ventana_sec.geometry("1000x700")

        # Frame para controles
        frame_ctrl = ttk.Frame(ventana_sec)
        frame_ctrl.pack(fill=tk.X, padx=10, pady=10)

        lbl_info = ttk.Label(frame_ctrl, text=f"Total de pasos: {len(self._transformacion_stack)}")
        lbl_info.pack(side=tk.LEFT, padx=5)

        # Slider para navegar
        self.slider_secuencia = ttk.Scale(
            frame_ctrl,
            from_=0,
            to=len(self._transformacion_stack) - 1,
            orient=tk.HORIZONTAL,
            command=self._actualizar_preview_secuencia
        )
        self.slider_secuencia.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)
        self.slider_secuencia.set(0)

        # Label del paso actual
        self.lbl_paso = ttk.Label(frame_ctrl, text="Paso 0")
        self.lbl_paso.pack(side=tk.LEFT, padx=5)

        # Frame para imagen
        frame_img = ttk.LabelFrame(ventana_sec, text="Preview")
        frame_img.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.label_preview_sec = ttk.Label(frame_img)
        self.label_preview_sec.pack(expand=True)

        # Frame para historial de nombres
        frame_list = ttk.LabelFrame(ventana_sec, text="Historial de Transformaciones")
        frame_list.pack(fill=tk.X, padx=10, pady=10)

        # Listbox con scroll
        scrollbar = ttk.Scrollbar(frame_list)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        listbox = tk.Listbox(frame_list, yscrollcommand=scrollbar.set, height=6)
        listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=listbox.yview)

        for i, nombre in enumerate(self._transformacion_nombres):
            listbox.insert(tk.END, f"{i}: {nombre}")

        # Mostrar primer paso
        self._actualizar_preview_secuencia(0)

    def _actualizar_preview_secuencia(self, valor):
        """Actualiza preview de secuencia según posición del slider."""
        idx = int(float(valor))
        if idx < 0 or idx >= len(self._transformacion_stack):
            return

        img_cv = self._transformacion_stack[idx]
        nombre = self._transformacion_nombres[idx] if idx < len(self._transformacion_nombres) else "?"

        # Redimensionar para preview
        h, w = img_cv.shape[:2]
        max_dim = 800
        if max(h, w) > max_dim:
            scale = max_dim / float(max(h, w))
            img_cv = cv2.resize(img_cv, (int(w * scale), int(h * scale)))

        # Convertir a PIL y mostrar
        if len(img_cv.shape) == 2:
            # Gris
            img_pil = Image.fromarray(img_cv, mode='L')
        else:
            # Color BGR a RGB
            img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)

        img_tk = ImageTk.PhotoImage(img_pil)
        self.label_preview_sec.config(image=img_tk)
        self.label_preview_sec.image = img_tk

        self.lbl_paso.config(text=f"Paso {idx}: {nombre}")

