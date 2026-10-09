# Deep Learning — UTEC

Implementaciones de los laboratorios del curso de Deep Learning.

## Laboratorio 2: pronóstico de caudal

Se compara FDMLP-LSTM con una LSTM base y una ablación sin FFT. El código y el análisis se encuentran en [`lab02/`](lab02/).

Los datos de entrada y las predicciones se descargan de Google Drive con `lab02/descargar_archivos.py` y se guardan localmente en `lab02/Rainfall-Runoff/` y `lab02/resultados/`. La descarga inicial requiere aproximadamente 5,2 GB.

Se incluyen las métricas y los historiales de ambas corridas.

```bash
uv sync --locked
uv run --locked python lab02/descargar_archivos.py
cd lab02
uv run --locked python seleccion_lasso.py
uv run --locked python entrenar.py --modelo lstm
uv run --locked python entrenar.py --modelo fdmlp
uv run --locked python entrenar.py --modelo mlp
uv run --locked jupyter lab resultados.ipynb
```

El cuaderno conserva las salidas de la ejecución original y descarga los archivos necesarios para el análisis.
