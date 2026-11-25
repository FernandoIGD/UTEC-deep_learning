"""Escenarios por voz: ASR -> agente -> TTS, con tiempos por etapa y métricas de voz."""

import json
import sys
import time

import pandas as pd
import soundfile as sf

from agente import Agente, cargar_llm
from datos import RAIZ, cargar
from escenarios import EVALUACION, ejecutar, guardar
from recuperacion import Buscador
from tienda import Tienda
from voz import cargar_asr, cargar_tts, clonar_voz, errores_asr, inteligibilidad, similitud_voz, sintetizar, transcribir

AUDIOS = RAIZ / "audios"
SALIDA = RAIZ / "resultados" / "audios"
PRODUCTOS = {
    "E01": [["fundas", "iphone", "auriculares", "bluetooth"], [], [], []],
    "E02": [["funda"], ["iphone"], ["funda"], []],
    "E03": [["cuadernos", "boligrafos"], ["cuadernos"], [], []],
    "E04": [["botella", "acero", "inoxidable"], [], []],
    "E05": [["tazas", "ceramica"], ["tazas"], [], []],
    "E06": [["toallas"], [], [], [], []],
}


def cargar_modelos(dispositivo):
    """Carga todos los modelos y mide el tiempo de carga de cada uno."""
    tiempos = {}
    inicio = time.perf_counter()
    catalogo, _ = cargar()
    buscador = Buscador(catalogo, dispositivo=dispositivo)
    tiempos["indice"] = time.perf_counter() - inicio

    inicio = time.perf_counter()
    tok, llm = cargar_llm(dispositivo)
    tiempos["llm"] = time.perf_counter() - inicio

    inicio = time.perf_counter()
    tts = cargar_tts(dispositivo)
    tiempos["tts"] = time.perf_counter() - inicio

    inicio = time.perf_counter()
    asr = cargar_asr()
    tiempos["asr"] = time.perf_counter() - inicio

    agente = Agente(tok, llm, buscador, Tienda(catalogo["product_id"]))
    return agente, asr, tts, tiempos


def ejecutar_voz(escenario, agente, asr, tts, voz):
    """Transcribe cada turno grabado, ejecuta el escenario y sintetiza cada respuesta.

    Returns:
        Registro del escenario con transcripciones, errores de ASR, métricas de TTS y tiempos por etapa.
    """
    archivos = [AUDIOS / f"{escenario['id']}_t{n}.wav" for n in range(1, len(escenario["turnos"]) + 1)]
    transcripciones = [transcribir(asr, str(archivo)) for archivo in archivos]
    r = ejecutar(escenario, agente, entradas=[texto for texto, _ in transcripciones])

    SALIDA.mkdir(parents=True, exist_ok=True)
    referencia = str(AUDIOS / "referencia.wav")
    for numero, (respuesta, (texto, segundos_asr)) in enumerate(zip(r["respuestas"], transcripciones), start=1):
        audio, sr, segundos_tts = sintetizar(tts, voz, respuesta)
        sf.write(SALIDA / f"{escenario['id']}_t{numero}_respuesta.wav", audio, sr)
        wer_tts, transcripcion_tts = inteligibilidad(asr, audio, sr, respuesta)
        tiempos = r["tiempos"][numero - 1]
        tiempos.update({"asr": segundos_asr, "tts": segundos_tts, "total": segundos_asr + tiempos["total"] + segundos_tts})
        r.setdefault("voz", []).append(
            {
                "turno": numero,
                "dicho": escenario["turnos"][numero - 1],
                "transcrito": texto,
                **errores_asr(escenario["turnos"][numero - 1], texto, PRODUCTOS[escenario["id"]][numero - 1]),
                "wer_tts": wer_tts,
                "transcripcion_tts": transcripcion_tts,
                "similitud": similitud_voz(tts, referencia, (audio, sr)),
            }
        )
    return r


def resumen_voz(registros):
    filas = []
    for r in registros:
        for v, t in zip(r["voz"], r["tiempos"]):
            filas.append({"id": r["id"], **{k: v[k] for k in v if k not in ("dicho", "transcrito", "transcripcion_tts")}, **{f"s_{k}": round(x, 2) for k, x in t.items()}})
    return pd.DataFrame(filas)


if __name__ == "__main__":
    dispositivo = sys.argv[1] if len(sys.argv) > 1 else "cuda"
    agente, asr, tts, carga = cargar_modelos(dispositivo)
    print("Carga inicial (s):", {k: round(v, 1) for k, v in carga.items()})
    voz = clonar_voz(tts, str(AUDIOS / "referencia.wav"))

    registros = []
    for escenario in [e for e in EVALUACION if e.get("voz")]:
        r = ejecutar_voz(escenario, agente, asr, tts, voz)
        guardar(r, "escenarios_voz")
        registros.append(r)
        print(r["id"], "completado" if r["completado"] else "fallido", r["violaciones"], flush=True)

    tabla = resumen_voz(registros)
    print(tabla.to_string(index=False))
    (RAIZ / "resultados" / "voz.json").write_text(
        json.dumps({"carga": carga, "turnos": tabla.to_dict("records"), "completados": [r["completado"] for r in registros]}, ensure_ascii=False, indent=2)
    )
