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
                "models",
                "cnn_garbage"
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
    def train(self, dataset_dir, batch_size=32, epochs=10, callback=None):
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

        history = self.model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=epochs,
            callbacks=[callback] if callback else None
        )

        self.save(self.default_model_dir)

        return history

    # ==================================================
    # GUARDADO / CARGA
    # ==================================================
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
        self.model = load_model(os.path.join(path, "model.h5"))

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
