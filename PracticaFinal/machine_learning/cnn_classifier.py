import json
import os
import numpy as np
from PIL import Image

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class CNNClassifier:
    def __init__(self, input_size=(224, 224)):
        self.input_size = input_size
        self.model = None
        self.class_indices = None

        # Ruta por defecto del modelo entrenado
        self.default_model_dir = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "ml_model"
            )
        )

    # ==================================================
    # CONSTRUCCIÓN DEL MODELO
    # ==================================================
    def build_model(self):
        if self.class_indices is None:
            raise RuntimeError("Las clases no han sido definidas.")

        num_classes = len(self.class_indices)

        base_model = MobileNetV2(
            weights="imagenet",
            include_top=False,
            input_shape=(*self.input_size, 3)
        )

        # Congelar capas base
        for layer in base_model.layers:
            layer.trainable = False

        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(128, activation="relu")(x)
        outputs = Dense(num_classes, activation="softmax")(x)

        self.model = Model(inputs=base_model.input, outputs=outputs)

        self.model.compile(
            optimizer="adam",
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )

    # ==================================================
    # ENTRENAMIENTO
    # ==================================================
    def train(self, dataset_dir, batch_size=32, epochs=10, callback=None, save_path=None):
        """
        Entrena el modelo con las imágenes de dataset_dir.
        Se espera que haya subdirectorios (uno por clase).
        
        Args:
            dataset_dir: Ruta al directorio con subdirectorios por clase
            batch_size: Tamaño del batch
            epochs: Número máximo de épocas
            callback: Callback adicional (opcional)
            save_path: Ruta donde guardar el modelo (si es None, no guarda automáticamente)
        
        Returns:
            history: Objeto con el historial de entrenamiento
        """
        datagen = ImageDataGenerator(
            rescale=1.0 / 255.0,
            validation_split=0.2
        )

        train_gen = datagen.flow_from_directory(
            dataset_dir,
            target_size=self.input_size,
            batch_size=batch_size,
            class_mode="categorical",
            subset="training"
        )

        val_gen = datagen.flow_from_directory(
            dataset_dir,
            target_size=self.input_size,
            batch_size=batch_size,
            class_mode="categorical",
            subset="validation"
        )

        # Detectar clases desde el dataset
        self.class_indices = train_gen.class_indices

        # Construir modelo
        self.build_model()

        # Early stopping para evitar sobreentrenamiento
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        )

        callbacks_list = [early_stop]
        if callback:
            callbacks_list.append(callback)

        history = self.model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=epochs,
            callbacks=callbacks_list
        )

        # Guardar solo si se especifica una ruta
        if save_path:
            self.save(save_path)
            self.save_training_report(save_path, history, dataset_dir, train_gen, val_gen)

        return history

    # ==================================================
    # GUARDADO / CARGA
    # ==================================================
    
    def save_training_report(self, path, history, dataset_dir, train_gen, val_gen):
        """
        Guarda un reporte del entrenamiento con métricas importantes.
        
        Args:
            path: Ruta donde guardar el reporte (misma carpeta del modelo)
            history: Objeto history devuelto por model.fit()
            dataset_dir: Ruta del dataset usado
            train_gen: Generador de entrenamiento
            val_gen: Generador de validación
        """
        import datetime
        
        os.makedirs(path, exist_ok=True)
        report_path = os.path.join(path, "training_report.txt")
        
        # Obtener métricas finales
        final_epoch = len(history.history['loss'])
        final_loss = history.history['loss'][-1]
        final_acc = history.history['accuracy'][-1] * 100
        final_val_loss = history.history['val_loss'][-1]
        final_val_acc = history.history['val_accuracy'][-1] * 100
        
        # Mejor epoch (mayor val_accuracy)
        best_epoch = np.argmax(history.history['val_accuracy']) + 1
        best_val_acc = max(history.history['val_accuracy']) * 100
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("REPORTE DE ENTRENAMIENTO - CNN GARBAGE CLASSIFIER\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Fecha y hora: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Dataset: {os.path.basename(dataset_dir)}\n")
            f.write(f"Ruta completa: {dataset_dir}\n\n")
            
            f.write("-" * 60 + "\n")
            f.write("INFORMACIÓN DEL DATASET\n")
            f.write("-" * 60 + "\n")
            f.write(f"Número de clases: {len(self.class_indices)}\n")
            f.write(f"Clases: {', '.join(self.class_indices.values())}\n")
            f.write(f"Imágenes de entrenamiento: {train_gen.samples}\n")
            f.write(f"Imágenes de validación: {val_gen.samples}\n")
            f.write(f"Tamaño de entrada: {self.input_size}\n\n")
            
            f.write("-" * 60 + "\n")
            f.write("RESULTADOS DEL ENTRENAMIENTO\n")
            f.write("-" * 60 + "\n")
            f.write(f"Épocas completadas: {final_epoch}\n")
            f.write(f"Mejor época: {best_epoch}\n")
            f.write(f"Mejor precisión de validación: {best_val_acc:.2f}%\n\n")
            
            f.write(f"MÉTRICAS FINALES (Época {final_epoch}):\n")
            f.write(f"  - Loss (entrenamiento): {final_loss:.4f}\n")
            f.write(f"  - Accuracy (entrenamiento): {final_acc:.2f}%\n")
            f.write(f"  - Loss (validación): {final_val_loss:.4f}\n")
            f.write(f"  - Accuracy (validación): {final_val_acc:.2f}%\n\n")
            
            f.write("-" * 60 + "\n")
            f.write("HISTORIAL POR ÉPOCA\n")
            f.write("-" * 60 + "\n")
            f.write(f"{'Época':<8} {'Train Loss':<12} {'Train Acc':<12} {'Val Loss':<12} {'Val Acc':<12}\n")
            f.write("-" * 60 + "\n")
            
            for i in range(final_epoch):
                f.write(f"{i+1:<8} {history.history['loss'][i]:<12.4f} "
                       f"{history.history['accuracy'][i]*100:<12.2f} "
                       f"{history.history['val_loss'][i]:<12.4f} "
                       f"{history.history['val_accuracy'][i]*100:<12.2f}\n")
            
            f.write("\n" + "=" * 60 + "\n")
            f.write("Early Stopping activado con patience=5 en val_loss\n")
            f.write("=" * 60 + "\n")
    
    def save(self, path):
        if self.model is None:
            raise RuntimeError("No hay modelo para guardar.")

        os.makedirs(path, exist_ok=True)

        # Guardar modelo
        self.model.save(os.path.join(path, "model.h5"))

        # Guardar clases
        with open(os.path.join(path, "classes.json"), "w") as f:
            json.dump(self.class_indices, f, indent=4)

        # Guardar metadatos
        meta = {
            "input_size": self.input_size
        }
        with open(os.path.join(path, "meta.json"), "w") as f:
            json.dump(meta, f, indent=4)

    def load(self, path):
        model_path = os.path.join(path, "model.h5")

        # Intento de carga estándar (compatible con TF/Keras actuales)
        try:
            self.model = load_model(model_path, compile=False)
        except Exception:
            # Compatibilidad: algunos modelos guardados con versiones más nuevas
            # incluyen el parámetro 'groups' en DepthwiseConv2D, que no existe
            # en versiones anteriores de tf.keras. Creamos un wrapper que lo ignora.
            class DepthwiseConv2DCompat(tf.keras.layers.DepthwiseConv2D):
                def __init__(self, *args, **kwargs):
                    kwargs.pop("groups", None)
                    super().__init__(*args, **kwargs)

            self.model = load_model(
                model_path,
                compile=False,
                custom_objects={"DepthwiseConv2D": DepthwiseConv2DCompat}
            )

        with open(os.path.join(path, "classes.json"), "r") as f:
            self.class_indices = json.load(f)

        with open(os.path.join(path, "meta.json"), "r") as f:
            meta = json.load(f)
            self.input_size = tuple(meta["input_size"])

    # ==================================================
    # PREDICCIÓN
    # ==================================================
    def predict(self, image_paths, top_k=3):
        if self.model is None:
            raise RuntimeError("Modelo no cargado.")

        if self.class_indices is None:
            raise RuntimeError("Clases no cargadas.")

        index_to_class = {v: k for k, v in self.class_indices.items()}
        results = []

        for path in image_paths:
            img = Image.open(path).convert("RGB")
            img = img.resize(self.input_size)

            img_array = np.array(img, dtype=np.float32) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            predictions = self.model.predict(img_array, verbose=0)[0]

            # Obtener índices ordenados por probabilidad (desc)
            sorted_indices = np.argsort(predictions)[::-1][:top_k]

            top_predictions = []
            for idx in sorted_indices:
                class_name = index_to_class[idx]
                confidence = float(predictions[idx]) * 100.0
                top_predictions.append(
                    {
                        "class": class_name,
                        "confidence": round(confidence, 2)
                    }
                )

            results.append(
                {
                    "image": path,
                    "predicted_class": top_predictions[0]["class"],
                    "confidence": top_predictions[0]["confidence"],
                    "top_predictions": top_predictions
                }
            )

        return results

    def load_default(self):
        self.load(self.default_model_dir)
