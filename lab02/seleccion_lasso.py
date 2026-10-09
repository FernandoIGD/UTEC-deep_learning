"""Selección de variables con LASSO y validación cruzada de 10 particiones (Sección 3.1 del paper).

Respuesta: caudal medio de las 48 horas futuras.
Predictores: promedio de las últimas 24 horas de cada una de las 12 variables.
Se usa una muestra aleatoria de 50 000 ventanas de entrenamiento.
"""
import json
from pathlib import Path

import h5py
import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler

from datos import RUTA_DATOS

NOMBRES = ["convective_fraction", "longwave_radiation", "potential_energy", "potential_evaporation",
           "pressure", "shortwave_radiation", "specific_humidity", "temperature",
           "total_precipitation", "wind_u", "wind_v", "specific_discharge"]

with h5py.File(RUTA_DATOS / "train.h5", "r") as f:
    idx_train = np.flatnonzero(f["split"][:] == 0)
    idx = np.sort(np.random.default_rng(42).choice(idx_train, 50000, replace=False))
    X = f["X"][idx]
    y = f["y"][idx]

predictores = StandardScaler().fit_transform(X[:, -24:, :].mean(axis=1))
respuesta = StandardScaler().fit_transform(y.mean(axis=1, keepdims=True)).ravel()

lasso = LassoCV(cv=10, random_state=42).fit(predictores, respuesta)
seleccion = [i for i, c in enumerate(lasso.coef_) if c != 0]

for nombre, c in zip(NOMBRES, lasso.coef_):
    print(f"{nombre:<24} {c:+.4f}")
print(f"alpha: {lasso.alpha_:.6f} | variables seleccionadas: {len(seleccion)} de 12")

carpeta = Path(__file__).parent / "resultados"
carpeta.mkdir(exist_ok=True)
with open(carpeta / "lasso.json", "w") as f:
    json.dump({"alpha": lasso.alpha_, "coeficientes": dict(zip(NOMBRES, lasso.coef_.tolist())),
               "seleccion": seleccion}, f, indent=2)
