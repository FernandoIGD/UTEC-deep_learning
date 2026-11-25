"""Graba la voz de referencia y los turnos de los seis escenarios por voz con el micrófono del Mac."""

import subprocess
import sys
import time

from datos import RAIZ
from escenarios import EVALUACION
from voz import TEXTO_REFERENCIA

CARPETA = RAIZ / "audios"


def grabar(texto, archivo):
    print(f"\n{archivo.name}\n  Leer: «{texto}»")
    input("  Enter para comenzar...")
    proceso = subprocess.Popen(
        ["ffmpeg", "-loglevel", "error", "-y", "-f", "avfoundation", "-i", ":0", "-ac", "1", "-ar", "24000", str(archivo)],
        stdin=subprocess.PIPE,
    )
    # El micrófono (sobre todo el del iPhone por Continuidad) tarda en entregar audio; sin esta espera se pierde el inicio.
    time.sleep(1.5)
    input("  >>> HABLE AHORA. Enter para detener...")
    proceso.communicate(b"q")


def pendientes():
    lista = [(TEXTO_REFERENCIA, CARPETA / "referencia.wav")]
    for escenario in EVALUACION:
        if escenario.get("voz"):
            for numero, texto in enumerate(escenario["turnos"], start=1):
                lista.append((texto, CARPETA / f"{escenario['id']}_t{numero}.wav"))
    return lista


if __name__ == "__main__":
    CARPETA.mkdir(exist_ok=True)
    rehacer = set(sys.argv[1:])
    for texto, archivo in pendientes():
        if archivo.exists() and archivo.stem not in rehacer:
            continue
        grabar(texto, archivo)
    print(f"\nListo: {len(list(CARPETA.glob('*.wav')))} archivos en {CARPETA}")
