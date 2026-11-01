"""Herramientas para lanzar muchos entrenamientos a la vez.

Cada experimento se describe con un diccionario (una "configuración"), por ejemplo:

    {"mercado": "reversion", "algoritmo": "dqn", "alpha": 5e-4, "gamma": 0.5, "seed": 0}

mercado es uno de los cuatro generadores y algoritmo, uno de ALGORITMOS; el resto de
claves se pasan a la función de entrenamiento.
"""
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from .a3c import entrenar_a3c, entrenar_reinforce
from .agentes import crear_agente
from .dqn import entrenar_dqn
from .evaluation import SEMILLA_TEST, SEMILLA_VALIDACION, evaluar_lote, generar_series
from .prices import GENERADORES
from .redes import Normalizador, fijar_hilos

N_VALIDACION = 100
N_TEST = 100

ALGORITMOS = ("dqn", "doble_dqn", "duelo", "a3c", "reinforce")
ES_ACTOR = ("a3c", "reinforce")


def _matriz(mercado, n, semilla):
    return np.array(generar_series(GENERADORES[mercado], n, semilla))


def entrenar(config, precios_validacion=None):
    """Entrena según una configuración y devuelve (Resultado, Normalizador)."""
    params = dict(config)
    mercado, algoritmo = params.pop("mercado"), params.pop("algoritmo")
    generador = GENERADORES[mercado]
    normalizador = Normalizador.ajustar(generador)
    if precios_validacion is None:
        precios_validacion = _matriz(mercado, params.pop("n_validacion", N_VALIDACION), SEMILLA_VALIDACION)
    else:
        params.pop("n_validacion", None)
    params["precios_validacion"] = precios_validacion
    if algoritmo == "a3c":
        return entrenar_a3c(generador, normalizador, **params), normalizador
    if algoritmo == "reinforce":
        return entrenar_reinforce(generador, normalizador, **params), normalizador
    flags = {"doble_dqn": dict(doble=True), "duelo": dict(duelo=True), "dqn": {}}[algoritmo]
    return entrenar_dqn(generador, normalizador, **flags, **params), normalizador


def _resumen_validacion(res, normalizador, precios):
    """Resume en validación los pesos elegidos (los del mejor control)."""
    df = evaluar_lote(precios, crear_agente(res.tipo_red, res.ocultas, res.estado, normalizador).acciones_greedy(precios))
    return {"val_mediana": df["retorno_pct"].median(), "val_media": df["retorno_pct"].mean(),
            "val_log": df["log_capital"].mean(), "val_ganancia": df["ganancia"].mean(), "val_sharpe": df["sharpe"].mean(),
            "val_operaciones": df["operaciones"].mean(), "val_dentro": df["tiempo_dentro_pct"].mean()}


def ejecutar(config, en_test=False):
    """Entrena un agente, lo resume en validación y, solo si se pide, lo evalúa en test.

    El test solo debe usarse con configuraciones ya elegidas. Se evalúan los pesos del
    mejor control en validación, y como comprobación los del último control y, para
    los métodos de actor, la política estocástica en vez de la greedy.
    """
    fijar_hilos()
    validacion = _matriz(config["mercado"], config.get("n_validacion", N_VALIDACION), SEMILLA_VALIDACION)
    res, normalizador = entrenar(config, validacion)
    salida = {"config": config, "resultado": res, "normalizador": normalizador,
              "resumen": _resumen_validacion(res, normalizador, validacion)}
    if en_test:
        precios = _matriz(config["mercado"], N_TEST, SEMILLA_TEST)
        mejor = crear_agente(res.tipo_red, res.ocultas, res.estado, normalizador)
        ultimo = crear_agente(res.tipo_red, res.ocultas, res.estado_final, normalizador)
        salida["test"] = evaluar_lote(precios, mejor.acciones_greedy(precios))
        salida["test_ultimo"] = evaluar_lote(precios, ultimo.acciones_greedy(precios))
        if config["algoritmo"] in ES_ACTOR:
            salida["test_estocastica"] = evaluar_lote(precios, mejor.acciones_muestreadas(precios))
        salida["mapa"] = mejor.mapa_politica()
    return salida


def _ejecutar_en_test(config):
    return ejecutar(config, en_test=True)


def ejecutar_varios(configs, en_test=False, procesos=None):
    """Lanza varias configuraciones en paralelo y devuelve los resultados en el mismo orden.

    En un script, la llamada tiene que ir dentro de `if __name__ == "__main__":`, porque
    en Windows cada proceso hijo vuelve a importar el script.
    """
    funcion = _ejecutar_en_test if en_test else ejecutar
    with ProcessPoolExecutor(max_workers=procesos or max(1, os.cpu_count() - 2)) as pool:
        return list(pool.map(funcion, configs))
