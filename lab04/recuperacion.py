"""Recuperación sobre el catálogo: BM25 (modelo base), embeddings E5 y combinación híbrida (RRF)."""

import bm25s
import numpy as np
import Stemmer
from sentence_transformers import SentenceTransformer

MODELO_EMBEDDINGS = "intfloat/multilingual-e5-small"
METODOS = ["bm25", "denso", "hibrido"]
# Elegido con las consultas de desarrollo en evaluar_recuperacion.py.
K_RRF = 100


class Buscador:
    """Índices léxico y denso sobre el mismo catálogo.

    Args:
        catalogo: DataFrame con `product_id` y la columna de texto indicada.
        columna: Columna de texto que se indexa.
        dispositivo: Dispositivo del modelo de embeddings.
        k_rrf: Constante de la fusión por rangos recíprocos.
    """

    def __init__(self, catalogo, columna="texto", dispositivo="cpu", k_rrf=K_RRF):
        self.catalogo = catalogo.reset_index(drop=True)
        self.k_rrf = k_rrf
        self.stemmer = Stemmer.Stemmer("spanish")
        textos = self.catalogo[columna].tolist()

        self.bm25 = bm25s.BM25()
        self.bm25.index(bm25s.tokenize(textos, stopwords="es", stemmer=self.stemmer, show_progress=False), show_progress=False)

        self.encoder = SentenceTransformer(MODELO_EMBEDDINGS, device=dispositivo)
        self.encoder.max_seq_length = 256
        self.vectores = self.encoder.encode(["passage: " + t for t in textos], batch_size=64, normalize_embeddings=True, show_progress_bar=False)
        self.posicion = {pid: i for i, pid in enumerate(self.catalogo["product_id"])}

    def puntajes_bm25(self, consulta):
        tokens = bm25s.tokenize(consulta, stopwords="es", stemmer=self.stemmer, return_ids=False, show_progress=False)[0]
        if not tokens:
            return np.zeros(len(self.catalogo))
        return self.bm25.get_scores(tokens)

    def puntajes_densos(self, consulta):
        vector = self.encoder.encode("query: " + consulta, normalize_embeddings=True)
        return self.vectores @ vector

    def ordenar(self, consulta, metodo="hibrido", indices=None):
        """Ordena el catálogo (o solo `indices`) de mayor a menor relevancia.

        Args:
            consulta: Texto de búsqueda.
            metodo: "bm25", "denso" o "hibrido".
            indices: Posiciones del catálogo a ordenar; None ordena todo el catálogo.

        Returns:
            Arreglo con las posiciones ordenadas.
        """
        indices = np.arange(len(self.catalogo)) if indices is None else np.asarray(indices)
        if metodo == "bm25":
            return indices[np.argsort(-self.puntajes_bm25(consulta)[indices], kind="stable")]
        if metodo == "denso":
            return indices[np.argsort(-self.puntajes_densos(consulta)[indices], kind="stable")]

        lexico = self.puntajes_bm25(consulta)[indices]
        denso = self.puntajes_densos(consulta)[indices]
        fusion = 1 / (self.k_rrf + rangos(denso))
        # Los productos sin ningún término en común no reciben aporte léxico.
        fusion += np.where(lexico > 0, 1 / (self.k_rrf + rangos(lexico)), 0)
        return indices[np.argsort(-fusion, kind="stable")]

    def buscar(self, consulta, k=5, metodo="hibrido"):
        return self.catalogo.iloc[self.ordenar(consulta, metodo)[:k]]


def rangos(puntajes):
    orden = np.argsort(-puntajes, kind="stable")
    resultado = np.empty(len(puntajes))
    resultado[orden] = np.arange(1, len(puntajes) + 1)
    return resultado
