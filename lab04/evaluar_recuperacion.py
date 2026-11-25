"""nDCG@10 del modelo base (BM25), del sistema (híbrido) y de las ablaciones."""

import json
import sys

import numpy as np
import pandas as pd

from datos import RAIZ, cargar
from recuperacion import Buscador

GANANCIAS = {"E": 1.0, "S": 0.1, "C": 0.01, "I": 0.0}
K = 10


def ndcg(ganancias_ordenadas, todas, k=K):
    descuento = 1 / np.log2(np.arange(2, k + 2))
    dcg = (np.asarray(ganancias_ordenadas[:k]) * descuento[: len(ganancias_ordenadas[:k])]).sum()
    ideal = np.sort(todas)[::-1][:k]
    idcg = (ideal * descuento[: len(ideal)]).sum()
    # Sin productos relevantes el nDCG no está definido; la consulta se excluye del promedio.
    return dcg / idcg if idcg > 0 else np.nan


def evaluar(buscador, consultas, metodo, sobre_catalogo=False):
    """nDCG@10 por consulta.

    Args:
        buscador: Buscador ya indexado.
        consultas: Pares consulta-producto anotados.
        metodo: "bm25", "denso" o "hibrido".
        sobre_catalogo: False ordena solo los candidatos anotados de cada consulta; True ordena
            todo el catálogo y los productos no anotados reciben ganancia 0.

    Returns:
        Serie con el nDCG@10 de cada consulta.
    """
    resultados = {}
    # Orden neutro de candidatos: con empates, argsort estable no debe heredar el orden del parquet.
    for query_id, grupo in consultas.sort_values("product_id").groupby("query_id"):
        ganancia = dict(zip(grupo["product_id"], grupo["esci_label"].map(GANANCIAS)))
        indices = None if sobre_catalogo else [buscador.posicion[pid] for pid in grupo["product_id"]]
        orden = buscador.ordenar(grupo["query"].iloc[0], metodo, indices)[:K]
        ordenadas = [ganancia.get(pid, 0.0) for pid in buscador.catalogo["product_id"].iloc[orden]]
        resultados[query_id] = ndcg(ordenadas, list(ganancia.values()))
    return pd.Series(resultados)


def tabla(buscadores, consultas):
    filas = []
    for nombre, (buscador, metodo) in buscadores.items():
        candidatos = evaluar(buscador, consultas, metodo)
        catalogo = evaluar(buscador, consultas, metodo, sobre_catalogo=True)
        filas.append({"sistema": nombre, "ndcg10_candidatos": candidatos.mean(), "ndcg10_catalogo": catalogo.mean()})
    return pd.DataFrame(filas).round(4)


if __name__ == "__main__":
    dispositivo = sys.argv[1] if len(sys.argv) > 1 else "cpu"
    catalogo, consultas = cargar()
    completo = Buscador(catalogo, dispositivo=dispositivo)
    solo_titulo = Buscador(catalogo, columna="product_title", dispositivo=dispositivo)

    desarrollo = consultas[consultas["split"] == "train"]
    ajuste = []
    for k_rrf in [10, 30, 60, 100]:
        completo.k_rrf = k_rrf
        ajuste.append({"k_rrf": k_rrf, "ndcg10_candidatos": evaluar(completo, desarrollo, "hibrido").mean()})
    ajuste = pd.DataFrame(ajuste).round(4)
    mejor = int(ajuste.loc[ajuste["ndcg10_candidatos"].idxmax(), "k_rrf"])
    completo.k_rrf = solo_titulo.k_rrf = mejor
    print("Ajuste en desarrollo:\n", ajuste.to_string(index=False))

    prueba = consultas[consultas["split"] == "test"]
    resultados = tabla(
        {
            "BM25 (modelo base)": (completo, "bm25"),
            "Híbrido BM25 + E5 (sistema)": (completo, "hibrido"),
            "Ablación: sin BM25 (solo E5)": (completo, "denso"),
            "Ablación: solo título": (solo_titulo, "hibrido"),
        },
        prueba,
    )
    sin_relevantes = prueba.groupby("query_id")["esci_label"].apply(lambda e: (e == "I").all()).sum()
    print(f"\nPrueba ({prueba['query_id'].nunique()} consultas, {sin_relevantes} sin relevantes excluidas), k_rrf = {mejor}:\n", resultados.to_string(index=False))

    salida = RAIZ / "resultados"
    salida.mkdir(exist_ok=True)
    (salida / "recuperacion.json").write_text(
        json.dumps({"k_rrf": mejor, "ajuste": ajuste.to_dict("records"), "prueba": resultados.to_dict("records")}, ensure_ascii=False, indent=2)
    )
