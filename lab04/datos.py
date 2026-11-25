"""Preparación del catálogo y de las consultas a partir de ESCI (versión reducida, español)."""

import re
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import snapshot_download

RAIZ = Path(__file__).parent
DATOS = RAIZ / "datos"
SEMILLA = 42
N_DESARROLLO = 100
N_EVALUACION = 200
# Productos sin anotaciones: evitan que el catálogo contenga solo candidatos de las consultas.
N_ADICIONALES = 1500
CAMPOS_TEXTO = ["product_title", "product_brand", "product_color", "product_bullet_point", "product_description"]


def descargar():
    carpeta = snapshot_download(
        "milistu/amazon-esci-data",
        repo_type="dataset",
        local_dir=DATOS / "esci",
        allow_patterns=["products/*", "queries/*"],
    )
    return Path(carpeta)


def leer_anotaciones(carpeta):
    anotaciones = pd.read_parquet(
        carpeta / "queries",
        columns=["query_id", "query", "product_id", "esci_label", "split"],
        filters=[("product_locale", "==", "es"), ("small_version", "==", 1)],
    )
    return anotaciones.drop_duplicates(["query_id", "product_id"])


def leer_productos(carpeta, ids):
    productos = pd.read_parquet(
        carpeta / "products",
        columns=["product_id"] + CAMPOS_TEXTO,
        filters=[("product_locale", "==", "es"), ("product_id", "in", list(ids))],
    )
    return productos.drop_duplicates("product_id")


def ids_espanol(carpeta):
    tabla = pd.read_parquet(carpeta / "products", columns=["product_id"], filters=[("product_locale", "==", "es")])
    return tabla["product_id"].drop_duplicates()


def limpiar(texto):
    texto = re.sub(r"<[^>]+>", " ", texto)
    return re.sub(r"\s+", " ", texto).strip()


def elegir_consultas(anotaciones, split, n, rng):
    grupo = anotaciones[anotaciones["split"] == split]
    completas = grupo.groupby("query_id")["disponible"].all()
    candidatas = np.sort(completas[completas].index.to_numpy())
    return rng.choice(candidatas, size=n, replace=False)


def preparar():
    carpeta = descargar()
    rng = np.random.default_rng(SEMILLA)
    anotaciones = leer_anotaciones(carpeta)
    disponibles = ids_espanol(carpeta)
    anotaciones["disponible"] = anotaciones["product_id"].isin(disponibles)

    desarrollo = elegir_consultas(anotaciones, "train", N_DESARROLLO, rng)
    evaluacion = elegir_consultas(anotaciones, "test", N_EVALUACION, rng)
    consultas = anotaciones[anotaciones["query_id"].isin(np.concatenate([desarrollo, evaluacion]))].drop(columns="disponible")

    anotados = set(consultas["product_id"])
    resto = np.sort(disponibles[~disponibles.isin(anotaciones["product_id"])].to_numpy())
    adicionales = rng.choice(resto, size=N_ADICIONALES, replace=False)

    catalogo = leer_productos(carpeta, anotados | set(adicionales)).fillna("")
    for campo in CAMPOS_TEXTO:
        catalogo[campo] = catalogo[campo].map(limpiar)
    catalogo["texto"] = catalogo[CAMPOS_TEXTO].agg(". ".join, axis=1).str.slice(0, 900)
    catalogo = catalogo.sort_values("product_id").reset_index(drop=True)

    faltantes = anotados - set(catalogo["product_id"])
    if faltantes:
        raise ValueError(f"Hay {len(faltantes)} productos anotados que no están en el catálogo")

    catalogo.to_parquet(DATOS / "catalogo.parquet", index=False)
    consultas.to_parquet(DATOS / "consultas.parquet", index=False)
    return catalogo, consultas


def cargar():
    return pd.read_parquet(DATOS / "catalogo.parquet"), pd.read_parquet(DATOS / "consultas.parquet")


if __name__ == "__main__":
    catalogo, consultas = preparar()
    resumen = consultas.groupby("split")["query_id"].nunique()
    print(f"Catálogo: {len(catalogo)} productos")
    print(f"Consultas: {resumen.to_dict()} | pares anotados: {len(consultas)}")
    print(consultas["esci_label"].value_counts().to_dict())
