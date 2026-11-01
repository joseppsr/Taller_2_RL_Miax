"""Prueba secundaria: DQN y A3C con precios reales de índices y fondos (ver rl_trader/reales.py).

No se ajustan hiperparámetros con datos reales, porque solo hay una historia y se
sobreajustaría: se usa la configuración elegida para el mercado de regímenes, el simulador
más parecido a un mercado real. Cada agente se entrena con 10 semillas.

Genera en results/ solo métricas, nunca precios: reales_resultados.csv.

Para lanzarlo, después de scripts/ajuste.py:  python scripts/datos_reales.py
Si el resultado ya existe no se repite. Con 16 núcleos tarda alrededor de media hora.
"""
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from rl_trader.a3c import entrenar_a3c  # noqa: E402
from rl_trader.agentes import crear_agente  # noqa: E402
from rl_trader.baselines import BASELINES  # noqa: E402
from rl_trader.dqn import entrenar_dqn  # noqa: E402
from rl_trader.evaluation import evaluar, evaluar_lote  # noqa: E402
from rl_trader.reales import ACTIVOS, GeneradorTramos, cargar, dividir  # noqa: E402
from rl_trader.redes import Normalizador, fijar_hilos  # noqa: E402

RESULTADOS = RAIZ / "results"
SEMILLAS = range(100, 110)
PROCESOS = {"dqn": 15, "a3c": 4}


def _entrenar(activo, familia, config):
    """Entrena un agente con tramos del periodo de entrenamiento y lo evalúa en cada periodo."""
    fijar_hilos()
    entrenamiento, validacion, test = dividir(cargar(activo))
    generador = GeneradorTramos(entrenamiento)
    normalizador = Normalizador.ajustar(generador)
    params = {k: v for k, v in config.items() if k not in ("mercado", "algoritmo")}
    entrenar_agente = entrenar_dqn if familia == "dqn" else entrenar_a3c
    res = entrenar_agente(generador, normalizador, precios_validacion=validacion[None], **params)
    agente = crear_agente(res.tipo_red, res.ocultas, res.estado, normalizador)
    filas = []
    for periodo, precios in (("entrenamiento", entrenamiento), ("validacion", validacion), ("test", test)):
        m = evaluar_lote(precios[None], agente.acciones_greedy(precios[None])).iloc[0]
        filas.append({"activo": activo, "agente": familia, "seed": config["seed"], "periodo": periodo, **m.to_dict()})
    return filas


def _referencias():
    filas = []
    for activo in ACTIVOS:
        tramos = dict(zip(("entrenamiento", "validacion", "test"), dividir(cargar(activo))))
        for periodo, precios in tramos.items():
            for nombre, crear in BASELINES.items():
                for s in (SEMILLAS if nombre == "aleatorio" else [None]):
                    politica = crear() if s is None else crear(s)
                    m = evaluar(politica, [precios]).iloc[0]
                    filas.append({"activo": activo, "agente": nombre, "seed": s, "periodo": periodo, **m.to_dict()})
    return filas


def main():
    ruta = RESULTADOS / "reales_resultados.csv"
    if ruta.exists():
        print("datos reales: ya calculados")
        return
    mejores = pd.read_csv(RESULTADOS / "mejores.csv").set_index(["familia", "mercado"])
    filas = _referencias()
    for familia in ("dqn", "a3c"):
        config = json.loads(mejores.loc[(familia, "regimenes"), "config"])
        with ProcessPoolExecutor(max_workers=PROCESOS[familia]) as pool:
            futuros = [pool.submit(_entrenar, a, familia, config | {"seed": s}) for a in ACTIVOS for s in SEMILLAS]
            for futuro in as_completed(futuros):
                filas += futuro.result()
        print(f"{familia}: terminado", flush=True)
    pd.DataFrame(filas).to_csv(ruta, index=False)


if __name__ == "__main__":
    main()
