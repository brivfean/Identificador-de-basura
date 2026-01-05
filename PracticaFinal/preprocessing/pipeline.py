import os
import cv2
import json


class PreprocessingPipeline:
    def __init__(self, name):
        self.name = name
        self.steps = []

    # ==================================================
    # DEFINICIÓN DEL PIPELINE
    # ==================================================
    def add_step(self, step_name, func, **params):
        self.steps.append({
            "name": step_name,
            "func": func,
            "params": params
        })

    # ==================================================
    # APLICAR A UNA SOLA IMAGEN
    # ==================================================
    def apply_to_image(self, img_cv):
        result = img_cv
        for step in self.steps:
            result = step["func"](result, **step["params"])
        return result

    # ==================================================
    # APLICAR A UNA CARPETA
    # ==================================================
    def apply_to_folder(self, input_dir, output_dir):
        if not os.path.isdir(input_dir):
            raise ValueError("Directorio de entrada inválido")

        os.makedirs(output_dir, exist_ok=True)

        for filename in os.listdir(input_dir):
            if not filename.lower().endswith((".jpg", ".png", ".jpeg")):
                continue

            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)

            img = cv2.imread(input_path)
            if img is None:
                continue

            processed = self.apply_to_image(img)
            cv2.imwrite(output_path, processed)

    # ==================================================
    # GUARDADO
    # ==================================================
    def save(self, directory):
        os.makedirs(directory, exist_ok=True)

        data = {
            "name": self.name,
            "steps": [
                {
                    "name": step["name"],
                    "params": step["params"]
                } for step in self.steps
            ]
        }

        path = os.path.join(directory, f"{self.name}.json")
        with open(path, "w") as f:
            json.dump(data, f, indent=4)

    # ==================================================
    # CARGA
    # ==================================================
    @staticmethod
    def load(path, function_registry):
        with open(path, "r") as f:
            data = json.load(f)

        pipeline = PreprocessingPipeline(data["name"])

        for step in data["steps"]:
            name = step["name"]
            params = step["params"]

            if name not in function_registry:
                raise KeyError(f"Función no registrada: {name}")

            pipeline.add_step(
                name,
                function_registry[name],
                **params
            )

        return pipeline
