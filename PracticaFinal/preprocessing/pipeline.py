import json
from preprocessing import operations

OPERACIONES = {
    "grises": operations.grises,
    "gaussiano": operations.gaussiano,
    "binarizacion_umbral": operations.binarizacion_umbral
}

class PreprocessingPipeline:
    def __init__(self, steps=None):
        self.steps = steps or []

    def add_step(self, op_name, **params):
        self.steps.append({
            "op": op_name,
            "params": params
        })

    def apply(self, img):
        resultado = img
        for step in self.steps:
            op = OPERACIONES[step["op"]]
            resultado = op(resultado, **step["params"])
        return resultado

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.steps, f, indent=4)

    @staticmethod
    def load(path):
        with open(path, "r") as f:
            steps = json.load(f)
        return PreprocessingPipeline(steps)
