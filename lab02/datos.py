from pathlib import Path

import h5py
import numpy as np

RUTA_DATOS = Path(__file__).parent / "Rainfall-Runoff"
CANAL_CAUDAL = 11


def cargar_train():
    """Carga train.h5 y separa entrenamiento (split=0) y validación (split=1).

    Retorna:
        Diccionario con X, y y basin_id para "train" y "val".
    """
    with h5py.File(RUTA_DATOS / "train.h5", "r") as f:
        X = f["X"][:]
        y = f["y"][:]
        split = f["split"][:]
        basin = f["basin_id"][:]

    datos = {}
    for nombre, codigo in [("train", 0), ("val", 1)]:
        m = split == codigo
        datos[nombre] = {"X": X[m], "y": y[m], "basin_id": basin[m]}
    return datos


def cargar_test():
    with h5py.File(RUTA_DATOS / "test.h5", "r") as f:
        return {"X": f["X"][:], "basin_id": f["basin_id"][:]}


def calcular_estadisticos(X_train):
    """Media y desviación estándar por variable, calculadas solo con entrenamiento."""
    suma = np.zeros(X_train.shape[2])
    suma_cuad = np.zeros(X_train.shape[2])
    for i in range(0, len(X_train), 20000):
        bloque = X_train[i:i + 20000].astype(np.float64)
        suma += bloque.sum(axis=(0, 1))
        suma_cuad += (bloque ** 2).sum(axis=(0, 1))
    n = X_train.shape[0] * X_train.shape[1]
    media = suma / n
    std = np.sqrt(suma_cuad / n - media ** 2)
    return media.astype(np.float32), std.astype(np.float32)


def normalizar(X, media, std):
    X -= media
    X /= std
    return X
