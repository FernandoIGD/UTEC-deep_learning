# Deep Learning — UTEC

Implementaciones de los laboratorios del curso de Deep Learning.

## Laboratorio 2: pronóstico de caudal

Se compara FDMLP-LSTM con una LSTM base y una ablación sin FFT. El código y el análisis se encuentran en [`lab02/`](lab02/).

Los datos de entrada y las predicciones se descargan de Google Drive con `lab02/descargar_archivos.py` y se guardan localmente en `lab02/Rainfall-Runoff/` y `lab02/resultados/`. La descarga inicial requiere aproximadamente 5,2 GB.

Se incluyen las métricas y los historiales de ambas corridas.

```bash
cd lab02
uv sync --locked
uv run --locked python descargar_archivos.py
uv run --locked python seleccion_lasso.py
uv run --locked python entrenar.py --modelo lstm
uv run --locked python entrenar.py --modelo fdmlp
uv run --locked python entrenar.py --modelo mlp
uv run --locked jupyter lab resultados.ipynb
```

El cuaderno conserva las salidas de la ejecución original y descarga los archivos necesarios para el análisis.

## Laboratorio 4: asistente de compras por voz

Asistente que recibe consultas habladas en español, recupera productos de Amazon ESCI con una búsqueda híbrida (BM25 + E5), ejecuta operaciones validadas sobre una tienda simulada y responde con voz clonada. El código y el análisis se encuentran en [`lab04/`](lab04/).

La ejecución principal es el cuaderno [`lab04/laboratorio4.ipynb`](lab04/laboratorio4.ipynb) en Google Colab con GPU; sus primeras celdas clonan el repositorio e instalan las versiones utilizadas. Las grabaciones de voz no se publican y se suben en la celda correspondiente.
