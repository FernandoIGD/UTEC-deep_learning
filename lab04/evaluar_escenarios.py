"""Ejecuta los escenarios operativos por texto y resume los resultados."""

import json
import sys

import pandas as pd

from agente import Agente, cargar_llm
from datos import RAIZ, cargar
from escenarios import DESARROLLO, EVALUACION, ejecutar, guardar
from recuperacion import Buscador
from tienda import Tienda


def resumir(registros):
    filas = [
        {
            "id": r["id"],
            "situacion": r["situacion"],
            "completado": r["completado"],
            "fallidos": "; ".join(v["criterio"] for v in r["verificaciones"] if not v["ok"]),
            "violaciones": "; ".join(r["violaciones"]),
            "turnos": len(r["entradas"]),
            "segundos_por_turno": round(sum(t["total"] for t in r["tiempos"]) / len(r["tiempos"]), 1),
        }
        for r in registros
    ]
    return pd.DataFrame(filas)


if __name__ == "__main__":
    conjunto = sys.argv[1]
    dispositivo = sys.argv[2] if len(sys.argv) > 2 else "cuda"
    escenarios = {"desarrollo": DESARROLLO, "evaluacion": EVALUACION}[conjunto]

    catalogo, _ = cargar()
    buscador = Buscador(catalogo, dispositivo=dispositivo)
    tok, llm = cargar_llm(dispositivo)
    agente = Agente(tok, llm, buscador, Tienda(catalogo["product_id"]))

    registros = []
    for escenario in escenarios:
        r = ejecutar(escenario, agente)
        guardar(r, f"escenarios_{conjunto}")
        registros.append(r)
        print(r["id"], "completado" if r["completado"] else "fallido", r["violaciones"], flush=True)

    tabla = resumir(registros)
    print(tabla.to_string(index=False))
    print(f"Completados: {tabla['completado'].mean():.0%} | violaciones: {sum(len(r['violaciones']) for r in registros)}")
    (RAIZ / "resultados" / f"escenarios_{conjunto}.json").write_text(json.dumps(tabla.to_dict("records"), ensure_ascii=False, indent=2))
