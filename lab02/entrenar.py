import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from datos import CANAL_CAUDAL, calcular_estadisticos, cargar_test, cargar_train, normalizar
from modelos import MODELOS


def metricas(y_obs, y_pred):
    """NSE, RMSE y R calculados sobre todas las muestras y horizontes (Ec. 16-18 del paper)."""
    y_obs, y_pred = y_obs.ravel(), y_pred.ravel()
    nse = 1 - np.sum((y_obs - y_pred) ** 2) / np.sum((y_obs - y_obs.mean()) ** 2)
    rmse = np.sqrt(np.mean((y_obs - y_pred) ** 2))
    r = np.corrcoef(y_obs, y_pred)[0, 1]
    return {"NSE": nse, "RMSE": rmse, "R": r}


def elegir_dispositivo():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sincronizar(dispositivo):
    if dispositivo.type == "cuda":
        torch.cuda.synchronize()
    elif dispositivo.type == "mps":
        torch.mps.synchronize()


@torch.no_grad()
def predecir(modelo, X, batch):
    modelo.eval()
    # .cpu() en cada lote sincroniza la GPU y evita que se encolen todos los lotes a la vez
    salidas = [modelo(X[i:i + batch]).cpu() for i in range(0, len(X), batch)]
    return torch.cat(salidas)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modelo", required=True, choices=list(MODELOS))
    parser.add_argument("--epocas", type=int, default=30)
    parser.add_argument("--paciencia", type=int, default=5)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--semilla", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.semilla)
    dispositivo = elegir_dispositivo()
    carpeta = Path(__file__).parent / "resultados" / args.modelo
    carpeta.mkdir(parents=True, exist_ok=True)
    print(f"Modelo: {args.modelo} | Dispositivo: {dispositivo}")

    datos = cargar_train()
    test = cargar_test()
    media, std = calcular_estadisticos(datos["train"]["X"])
    media_q, std_q = media[CANAL_CAUDAL], std[CANAL_CAUDAL]
    with open(carpeta.parent / "lasso.json") as f:
        seleccion = json.load(f)["seleccion"]
    print(f"Variables seleccionadas por LASSO: {seleccion}")

    X_train = torch.from_numpy(normalizar(datos["train"]["X"], media, std)[:, :, seleccion]).to(dispositivo)
    y_train = torch.from_numpy((datos["train"]["y"] - media_q) / std_q).to(dispositivo)
    X_val = torch.from_numpy(normalizar(datos["val"]["X"], media, std)[:, :, seleccion]).to(dispositivo)
    X_test = torch.from_numpy(normalizar(test["X"], media, std)[:, :, seleccion]).to(dispositivo)
    y_val = datos["val"]["y"]
    del datos, test

    modelo = MODELOS[args.modelo](len(seleccion)).to(dispositivo)
    n_parametros = sum(p.numel() for p in modelo.parameters())
    optimizador = torch.optim.Adam(modelo.parameters(), lr=args.lr)
    perdida_fn = torch.nn.MSELoss()
    print(f"Parámetros: {n_parametros:,}")

    historial = []
    mejor_nse, sin_mejora = -np.inf, 0
    for epoca in range(1, args.epocas + 1):
        inicio = time.time()
        modelo.train()
        orden = torch.randperm(len(X_train), device=dispositivo)
        suma_perdida = 0.0
        for i in range(0, len(orden), args.batch):
            idx = orden[i:i + args.batch]
            optimizador.zero_grad()
            perdida = perdida_fn(modelo(X_train[idx]), y_train[idx])
            perdida.backward()
            optimizador.step()
            suma_perdida += perdida.item() * len(idx)
        mse_train = suma_perdida / len(orden)
        segundos = time.time() - inicio

        pred_val = predecir(modelo, X_val, 256).cpu().numpy() * std_q + media_q
        m = metricas(y_val, pred_val)
        historial.append({"epoca": epoca, "mse_train": mse_train, **m, "segundos": segundos})
        print(f"Época {epoca:02d} | MSE train: {mse_train:.4f} | NSE val: {m['NSE']:.4f} "
              f"| RMSE val: {m['RMSE']:.4f} | {segundos:.0f} s", flush=True)

        if m["NSE"] > mejor_nse:
            mejor_nse, sin_mejora = m["NSE"], 0
            torch.save(modelo.state_dict(), carpeta / "modelo.pt")
        else:
            sin_mejora += 1
            if sin_mejora >= args.paciencia:
                print("Parada temprana")
                break

    pd.DataFrame(historial).to_csv(carpeta / "historial.csv", index=False)

    modelo.load_state_dict(torch.load(carpeta / "modelo.pt", map_location=dispositivo))
    pred_val = predecir(modelo, X_val, 256).cpu().numpy() * std_q + media_q
    sincronizar(dispositivo)
    inicio = time.time()
    pred_test = predecir(modelo, X_test, 256)
    sincronizar(dispositivo)
    segundos_inferencia = time.time() - inicio
    pred_test = pred_test.cpu().numpy() * std_q + media_q

    np.save(carpeta / "pred_val.npy", pred_val)
    columnas = [f"q_{h:02d}" for h in range(1, 49)]
    df_test = pd.DataFrame(pred_test, columns=columnas)
    df_test.insert(0, "Id", np.arange(len(df_test)))
    df_test.to_csv(carpeta / "pred_test.csv", index=False)

    info = {
        "modelo": args.modelo,
        "dispositivo": str(dispositivo),
        "parametros": n_parametros,
        "epocas_entrenadas": len(historial),
        "segundos_por_epoca": float(np.mean([h["segundos"] for h in historial])),
        "segundos_inferencia_test": segundos_inferencia,
        **{f"{k}_val": float(v) for k, v in metricas(y_val, pred_val).items()},
        **vars(args),
    }
    with open(carpeta / "info.json", "w") as f:
        json.dump(info, f, indent=2)
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
