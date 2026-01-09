import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk

import cv2
import os

from tkinter import simpledialog

import numpy as np
from scipy import ndimage

from processing.filtros import (
    filtro_media,
    filtro_mediana,
    filtro_gaussiano,
    filtro_laplaciano,
    filtro_sobel,
    filtro_prewitt,
    filtro_roberts,
    filtro_canny
)

from processing.ruido import (
    ruido_sal_pimienta,
    ruido_sal,
    ruido_pimienta,
    ruido_gaussiano
)

from processing.morfologia import (
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

from image_utils.histogramas import (
    compute_histogramas_rgb_arrays_from_cv,
    compute_histograma_gris_array_from_cv
)

from image_utils.histogramas import calcular_stats_rgb_from_cv

from image_utils import (
    preparar_binaria,
    etiquetar_componentes,
    etiquetas_a_rgb,
    calcular_stats_cc
)

from processing import (
    suma,
    resta,
    multiplicacion,
    division,
    op_and,
    op_or,
    op_xor,
    op_not
)

from image_utils.modelos_color import ModelosColor

from image_utils.pseudocolor import Pseudocolor

from image_utils.conversions import (
    convertir_pil_a_cv,
    convertir_cv_a_pil
)

from machine_learning import CNNClassifier
import threading
from tkinter import filedialog, messagebox

from segmentation import Segmentador

from preprocessing import PreprocessingPipeline, PREPROCESSING_FUNCTIONS

from normalization.normalizador import Normalizador

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

        # -------- Checkbox --------
        frame_checkbox = ttk.Frame(main_frame)
        frame_checkbox.pack(fill=tk.X, pady=10)

        ttk.Checkbutton(
            frame_checkbox,
            text="Modificar sobre la marcha",
            variable=self.modificar_sobre_marcha
        ).pack(anchor="w")

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
    def aplicar_operacion(self, funcion, *args):
        if self.imagen_original_cv is None:
            messagebox.showwarning("Aviso", "Primero carga una imagen")
            return

        if self.modificar_sobre_marcha.get():
            fuente = self.imagen_actual_cv
        else:
            fuente = self.imagen_original_cv

        resultado = funcion(fuente, *args)

        self.imagen_resultado_cv = resultado

        if self.modificar_sobre_marcha.get():
            self.imagen_actual_cv = resultado

        self._mostrar_imagen_resultado(resultado)


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
            def wrapper(img):
                _, morph = apertura(img, k)
                return morph

            self.aplicar_operacion(wrapper)


    def _morfologia_cierre(self):
        k = simpledialog.askinteger(
            "Cierre (gris)",
            "Tamaño del kernel:",
            minvalue=1
        )

        if k:
            def wrapper(img):
                _, morph = cierre(img, k)
                return morph

            self.aplicar_operacion(wrapper)

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

    def _actualizar_resultado(self, resultado_cv):
        self.imagen_resultado_cv = resultado_cv
        self._mostrar_imagen_resultado(resultado_cv)

    def _cargar_menu_pseudocolor(self):
        """
        Carga el menú de pseudocolores (OpenCV y personalizado).
        """

        # Limpiar por si se recarga
        self.menu_pseudocolor.delete(0, tk.END)

        # -----------------------------
        # Pseudocolores OpenCV
        # -----------------------------
        pseudocolores_opencv = [
            "JET",
            "HOT",
            "OCEAN",
            "BONE",
            "RAINBOW"
        ]

        for nombre in pseudocolores_opencv:
            self.menu_pseudocolor.add_command(
                label=f"OpenCV - {nombre}",
                command=lambda n=nombre: self._pseudocolor_opencv(n)
            )

        self.menu_pseudocolor.add_separator()

        # -----------------------------
        # Pseudocolor personalizado
        # -----------------------------
        self.menu_pseudocolor.add_command(
            label="Personalizado...",
            command=self._pseudocolor_personalizado
        )

    def _pseudocolor_opencv(self, nombre):
        if self.imagen_original_cv is None:
            messagebox.showwarning("Aviso", "Primero carga una imagen.")
            return

        if self.imagen_gris is None:
            self.imagen_gris = cv2.cvtColor(
                self.imagen_original_cv, cv2.COLOR_BGR2GRAY
            )

        pc = Pseudocolor(self.imagen_gris)
        resultado = pc.aplicar_opencv(nombre)

        self._actualizar_resultado(resultado)

        self.text_info.delete("1.0", tk.END)
        self.text_info.insert(
            tk.END, f"Pseudocolor OpenCV {nombre} aplicado.\n"
        )

    def _pseudocolor_personalizado(self):
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
            label="Otsu",
            command=self._binarizacion_otsu
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
                clf.train(dataset_dir, epochs=10)
                clf.save("ml_model")

                messagebox.showinfo(
                    "ML",
                    "Entrenamiento finalizado y modelo guardado."
                )
            except Exception as e:
                messagebox.showerror("Error", str(e))

        threading.Thread(target=tarea, daemon=True).start()

    def _ml_probar(self):
        image_paths = filedialog.askopenfilenames(
            title="Selecciona imágenes",
            filetypes=[("Imágenes", "*.jpg *.png *.jpeg")]
        )
        if not image_paths:
            return

        def tarea():
            try:
                clf = CNNClassifier()
                clf.load_default()
                results = clf.predict(image_paths)

                texto = "Resultados:\n\n"
                for r in results:
                    path = r["image"]
                    clase = r["predicted_class"]
                    confianza = r["confidence"]
                    top = r["top_predictions"]

                    nombre = os.path.basename(path)

                    texto += f"{nombre} → {clase} ({confianza:.2f}%)\n"
                    texto += "   Siguientes similitudes:\n"

                    for p in top[1:]:
                        texto += f"      - {p['class']}: {p['confidence']:.2f}%\n"

                    texto += "\n"


                messagebox.showinfo("Predicción ML", texto)

            except Exception as e:
                messagebox.showerror("Error", str(e))

        threading.Thread(target=tarea, daemon=True).start()

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






