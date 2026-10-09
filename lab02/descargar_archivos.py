"""Descarga los datos y las predicciones del laboratorio desde Google Drive."""

from pathlib import Path

import gdown


RAIZ = Path(__file__).resolve().parent
RECURSOS = (
    (
        "Rainfall-Runoff",
        "https://drive.google.com/drive/folders/1KDHpRTgFRU6oUB7mkpXtqQl9GjlwTFsQ",
        ("train.h5", "test.h5", "test_targets.csv"),
    ),
    (
        "resultados",
        "https://drive.google.com/drive/folders/1DhtRJeNOHRxfqvmlLl71GgGvQpq6t6oI",
        tuple(
            f"{modelo}/{archivo}"
            for modelo in ("lstm", "fdmlp", "mlp")
            for archivo in ("pred_val.npy", "pred_test.csv")
        ),
    ),
)


def descargar_archivos():
    """Descarga los archivos necesarios que todavía no estén disponibles.

    Raises:
        FileNotFoundError: Si falta algún archivo después de la descarga.
    """
    for carpeta, enlace, archivos in RECURSOS:
        destino = RAIZ / carpeta
        if all((destino / archivo).is_file() for archivo in archivos):
            continue

        print(f"Descargando {carpeta} desde Google Drive...")
        gdown.download_folder(
            url=enlace,
            output=str(destino),
            resume=True,
            use_cookies=False,
            retries=3,
        )
        faltantes = [archivo for archivo in archivos if not (destino / archivo).is_file()]
        if faltantes:
            raise FileNotFoundError(f"Faltan archivos en {destino}: {', '.join(faltantes)}")


if __name__ == "__main__":
    descargar_archivos()
