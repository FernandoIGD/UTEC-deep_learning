"""Reconocimiento de voz (Whisper), síntesis con clonación (Qwen3-TTS) y métricas de voz."""

import re
import time
import unicodedata
from collections import Counter

import jiwer
import librosa
import numpy as np
import torch
from faster_whisper import WhisperModel
from qwen_tts import Qwen3TTSModel

MODELO_ASR = "turbo"
MODELO_TTS = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
TEXTO_REFERENCIA = "Hola, esta es mi voz de referencia. Estoy probando un asistente que busca productos y responde de forma natural en español."

UNIDADES = {
    "cero": 0, "un": 1, "uno": 1, "una": 1, "dos": 2, "tres": 3, "cuatro": 4, "cinco": 5, "seis": 6, "siete": 7, "ocho": 8,
    "nueve": 9, "diez": 10, "once": 11, "doce": 12, "trece": 13, "catorce": 14, "quince": 15, "dieciseis": 16,
    "diecisiete": 17, "dieciocho": 18, "diecinueve": 19, "veinte": 20, "veintiuno": 21, "veintiun": 21, "veintiuna": 21,
    "veintidos": 22, "veintitres": 23, "veinticuatro": 24, "veinticinco": 25, "veintiseis": 26, "veintisiete": 27,
    "veintiocho": 28, "veintinueve": 29,
}
DECENAS = {"treinta": 30, "cuarenta": 40, "cincuenta": 50, "sesenta": 60, "setenta": 70, "ochenta": 80, "noventa": 90}
CENTENAS = {"cien": 100, "ciento": 100} | {
    f"{prefijo}cient{fin}": valor * 100
    for prefijo, valor in [("dos", 2), ("tres", 3), ("cuatro", 4), ("quinien", 5), ("seis", 6), ("sete", 7), ("ocho", 8), ("nove", 9)]
    for fin in ["os", "as"]
}
CENTENAS |= {"quinientos": 500, "quinientas": 500}


def cargar_asr():
    # Whisper en CPU con INT8 deja la memoria de la GPU para el LLM y el TTS.
    return WhisperModel(MODELO_ASR, device="cpu", compute_type="int8")


def transcribir(asr, audio):
    inicio = time.perf_counter()
    segmentos, _ = asr.transcribe(audio, language="es", beam_size=1)
    texto = " ".join(s.text.strip() for s in segmentos)
    return texto, time.perf_counter() - inicio


def cargar_tts(dispositivo):
    # FP32 y SDPA evitan depender de BF16 o FlashAttention, que la T4 no soporta.
    mapa = "cuda:0" if dispositivo == "cuda" else dispositivo
    return Qwen3TTSModel.from_pretrained(MODELO_TTS, device_map=mapa, dtype=torch.float32, attn_implementation="sdpa")


def clonar_voz(tts, audio_referencia, texto_referencia=TEXTO_REFERENCIA):
    return tts.create_voice_clone_prompt(ref_audio=audio_referencia, ref_text=texto_referencia)


def texto_para_voz(texto):
    texto = re.sub(r"[*#_`>]|^\s*[-•]\s*", " ", texto, flags=re.M)
    # "37,52 euros" se lee mal; la forma hablada habitual se sintetiza con claridad.
    texto = re.sub(r"(\d+),(\d{2}) ?(euros|€)", lambda m: f"{m[1]} euros con {int(m[2])} céntimos", texto)
    texto = re.sub(r"(\d+) ?€", r"\1 euros", texto)
    return re.sub(r"\s+", " ", texto).strip()


def sintetizar(tts, voz, texto):
    inicio = time.perf_counter()
    audios, sr = tts.generate_voice_clone(text=texto_para_voz(texto), language="Spanish", voice_clone_prompt=voz, max_new_tokens=1024)
    return audios[0], sr, time.perf_counter() - inicio


def similitud_voz(tts, audio_referencia, audio_generado):
    """Coseno entre los embeddings de hablante (x-vector) del codificador de Qwen3-TTS."""
    ref = tts.create_voice_clone_prompt(ref_audio=audio_referencia, x_vector_only_mode=True)[0].ref_spk_embedding
    gen = tts.create_voice_clone_prompt(ref_audio=audio_generado, x_vector_only_mode=True)[0].ref_spk_embedding
    return torch.nn.functional.cosine_similarity(ref.float().flatten(), gen.float().flatten(), dim=0).item()


def normalizar(texto):
    texto = unicodedata.normalize("NFD", texto.lower())
    texto = "".join(c for c in texto if unicodedata.category(c) != "Mn")
    texto = re.sub(r"(\d),(\d)", r"\1.\2", texto).replace("€", " euros")
    return re.sub(r"[^\w\s.]|(?<!\d)\.|\.(?!\d)", " ", texto)


def leer_numero(palabras, i):
    """Lee un número en palabras desde la posición i: [centena] [decena [y unidad] | unidad]."""
    total, j = 0, i
    if j < len(palabras) and palabras[j] in CENTENAS:
        total, j = CENTENAS[palabras[j]], j + 1
    if j < len(palabras) and palabras[j] in DECENAS:
        total, j = total + DECENAS[palabras[j]], j + 1
        if j + 1 < len(palabras) and palabras[j] == "y" and UNIDADES.get(palabras[j + 1], 10) < 10:
            total, j = total + UNIDADES[palabras[j + 1]], j + 2
    elif j < len(palabras) and palabras[j] in UNIDADES:
        total, j = total + UNIDADES[palabras[j]], j + 1
    return total, j


def palabras_a_numeros(texto):
    """Convierte números escritos en palabras (hasta 999) a cifras, por ejemplo "ochenta y dos" -> "82"."""
    palabras, salida, i = normalizar(texto).split(), [], 0
    while i < len(palabras):
        valor, j = leer_numero(palabras, i)
        salida.append(palabras[i] if j == i else str(valor))
        i = max(j, i + 1)
    return " ".join(salida)


def diferencias(referencia, reconocidas):
    faltan = Counter(referencia) - Counter(reconocidas)
    sobran = Counter(reconocidas) - Counter(referencia)
    # Una sustitución (catorce en lugar de cuatro) cuenta como un solo error.
    return max(sum(faltan.values()), sum(sobran.values()))


def cifras(texto):
    """Separa los números en importes (seguidos de "euros") y cantidades."""
    tokens = palabras_a_numeros(texto).split()
    importes, cantidades = [], []
    for i, token in enumerate(tokens):
        if re.fullmatch(r"\d+(\.\d+)?", token):
            destino = importes if i + 1 < len(tokens) and tokens[i + 1].startswith("euro") else cantidades
            destino.append(float(token))
    return importes, cantidades


def errores_asr(referencia, transcripcion, productos):
    """Errores del ASR en cantidades, importes y productos respecto de la transcripción correcta.

    Args:
        referencia: Texto realmente dicho.
        transcripcion: Texto reconocido por el ASR.
        productos: Palabras clave de producto presentes en la referencia.

    Returns:
        Diccionario con el WER y la cantidad de errores por tipo.
    """
    imp_ref, cant_ref = cifras(referencia)
    imp_asr, cant_asr = cifras(transcripcion)
    hipotesis = palabras_a_numeros(transcripcion)
    return {
        "wer": jiwer.wer(palabras_a_numeros(referencia), hipotesis),
        "errores_cantidades": diferencias(cant_ref, cant_asr),
        "errores_importes": diferencias(imp_ref, imp_asr),
        "errores_productos": sum(normalizar(p).strip() not in hipotesis.split() for p in productos),
    }


def inteligibilidad(asr, audio, sr, texto):
    """WER entre el texto sintetizado y su transcripción por Whisper (menor es mejor)."""
    audio = librosa.resample(np.asarray(audio, dtype=np.float32), orig_sr=sr, target_sr=16000)
    transcripcion, _ = transcribir(asr, audio)
    if not texto_para_voz(texto):
        return float("nan"), transcripcion
    return jiwer.wer(palabras_a_numeros(texto_para_voz(texto)), palabras_a_numeros(transcripcion)), transcripcion
