"""Evaluación final en las series de test, con las configuraciones elegidas en scripts/ajuste.py.

Los agentes se entrenan de nuevo con 10 semillas que no se usaron en el ajuste (100 a 109)
y se prueban en las mismas 100 series de test que las estrategias de referencia.

Genera en results/:
- test_por_serie.csv.gz: una fila por agente, semilla y serie de test.
- entrenamientos.csv: una fila por entrenamiento (tiempo y episodio de convergencia).
- curvas_validacion.csv: la curva de aprendizaje de cada entrenamiento.
- ganancia_episodios.csv.gz: la ganancia G = Σ r de cada episodio de entrenamiento.
- baselines_validacion.csv: la ganancia de las referencias, para dibujarla en las curvas.
- mapas_politica.npz: la política continua de cada entrenamiento.
- modelos.pt: los pesos elegidos de cada entrenamiento.
- tiempos.csv: segundos por episodio con la máquina descargada.
- escalado_a3c.csv: tiempo y resultado de A3C según el número de trabajadores.

Para lanzarlo, desde la raíz del repositorio:  python scripts/resultados.py

Las partes ya calculadas no se repiten, y cada familia de algoritmos se guarda al terminar.
Con 16 núcleos tarda alrededor de 1 hora y media.
"""
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import torch

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from rl_trader.baselines import BASELINES  # noqa: E402
from rl_trader.evaluation import (SEMILLA_TEST, SEMILLA_VALIDACION, episodio_convergencia, evaluar,  # noqa: E402
                                  generar_series)
from rl_trader.experimentos import _ejecutar_en_test, ejecutar  # noqa: E402
from rl_trader.prices import GENERADORES  # noqa: E402

RESULTADOS = RAIZ / "results"
SEMILLAS = range(100, 110)
N_TEST = 100
VARIANTES = {"dqn": ["dqn", "doble_dqn", "duelo"], "reinforce": ["reinforce"], "a3c": ["a3c"]}
PROCESOS = {"dqn": 15, "reinforce": 15, "a3c": 4}
TRABAJADORES_ESCALADO = (1, 2, 4, 8)
SEMILLAS_ESCALADO = (200, 201)
MERCADOS_ESCALADO = ("ciclo", "reversion")
# entrenamientos.csv va el último: una familia solo cuenta como hecha cuando se ha escrito todo lo demás
ARCHIVOS = {"test": "test_por_serie.csv.gz", "curvas": "curvas_validacion.csv", "ganancia": "ganancia_episodios.csv.gz",
            "runs": "entrenamientos.csv"}


def configuraciones(familia):
    configs = []
    for f in pd.read_csv(RESULTADOS / "mejores.csv").query("familia == @familia").itertuples():
        base = json.loads(f.config)
        for algoritmo in VARIANTES[familia]:
            configs += [base | {"algoritmo": algoritmo, "seed": s} for s in SEMILLAS]
    return configs


def _guardar(salida, tablas):
    c = salida["config"]
    res = salida["resultado"]
    id_run = {"mercado": c["mercado"], "agente": c["algoritmo"], "seed": c["seed"]}
    for nombre, clave in (("test", "test"), ("test_ultimo", "test_ultimo"), ("test_estocastica", "test_estocastica")):
        if clave in salida:
            tablas["test"].append(salida[clave].assign(serie=range(N_TEST), version=nombre, **id_run))
    tablas["runs"].append({**id_run, "segundos": res.segundos, "pasos": res.pasos,
                           "n_episodios": c["n_episodios"],
                           "episodio_convergencia": episodio_convergencia(res.validacion),
                           **salida["resumen"], "config": json.dumps(c, sort_keys=True)})
    tablas["curvas"].append(res.validacion.assign(**id_run))
    tablas["ganancia"].append(res.historial.assign(**id_run))
    tablas["mapas"][f"{c['mercado']}|{c['algoritmo']}|{c['seed']}"] = salida["mapa"]
    tablas["modelos"][f"{c['mercado']}|{c['algoritmo']}|{c['seed']}"] = {"tipo_red": res.tipo_red, "ocultas": res.ocultas,
                                                                        "estado": res.estado}


def _familia_hecha(familia):
    ruta = RESULTADOS / ARCHIVOS["runs"]
    return ruta.exists() and set(VARIANTES[familia]) <= set(pd.read_csv(ruta, usecols=["agente"])["agente"])


def _escribir(tablas):
    """Añade a results/ lo que acaba de entrenarse, sin tocar lo que ya había de otras familias."""
    ruta_mapas, ruta_modelos = RESULTADOS / "mapas_politica.npz", RESULTADOS / "modelos.pt"
    mapas = dict(np.load(ruta_mapas)) if ruta_mapas.exists() else {}
    np.savez_compressed(ruta_mapas, **(mapas | tablas["mapas"]))
    modelos = torch.load(ruta_modelos, weights_only=False) if ruta_modelos.exists() else {}
    torch.save(modelos | tablas["modelos"], ruta_modelos)
    for clave, archivo in ARCHIVOS.items():
        ruta = RESULTADOS / archivo
        nuevas = pd.DataFrame(tablas[clave]) if clave == "runs" else pd.concat(tablas[clave])
        (pd.concat([pd.read_csv(ruta), nuevas]) if ruta.exists() else nuevas).to_csv(ruta, index=False)


def entrenar_finales():
    for familia in ("dqn", "reinforce", "a3c"):
        if _familia_hecha(familia):
            print(f"entrenamientos finales de {familia}: ya calculados", flush=True)
            continue
        tablas = {"test": [], "runs": [], "curvas": [], "ganancia": [], "mapas": {}, "modelos": {}}
        configs = configuraciones(familia)
        print(f"[{time.strftime('%H:%M:%S')}] {familia}: {len(configs)} entrenamientos", flush=True)
        with ProcessPoolExecutor(max_workers=PROCESOS[familia]) as pool:
            for futuro in as_completed([pool.submit(_ejecutar_en_test, c) for c in configs]):
                _guardar(futuro.result(), tablas)
        _escribir(tablas)


def evaluar_baselines():
    """Referencias en test y en validación (la ganancia en validación se dibuja junto a las curvas)."""
    ruta = RESULTADOS / "test_baselines.csv"
    if ruta.exists():
        return
    test, val = [], []
    for mercado, g in GENERADORES.items():
        series_test = generar_series(g, N_TEST, SEMILLA_TEST)
        series_val = generar_series(g, N_TEST, SEMILLA_VALIDACION)
        for nombre, crear in BASELINES.items():
            # El agente aleatorio se repite con 10 semillas, como los agentes entrenados
            semillas = SEMILLAS if nombre == "aleatorio" else [None]
            for s in semillas:
                politica = crear() if s is None else crear(s)
                test.append(evaluar(politica, series_test).assign(serie=range(N_TEST), mercado=mercado, agente=nombre,
                                                                  seed=s))
            politica = crear()
            val.append({"mercado": mercado, "agente": nombre,
                        "ganancia": evaluar(politica, series_val)["ganancia"].mean()})
    pd.concat(test).to_csv(ruta, index=False)
    pd.DataFrame(val).to_csv(RESULTADOS / "baselines_validacion.csv", index=False)


def medir_tiempos():
    """Mide los segundos por episodio con la máquina descargada: un entrenamiento detrás de otro.

    Los tiempos de los entrenamientos finales están inflados porque corren muchos a la vez.
    """
    ruta = RESULTADOS / "tiempos.csv"
    if ruta.exists():
        return
    filas = []
    for familia, algoritmos in VARIANTES.items():
        for algoritmo in algoritmos:
            f = pd.read_csv(RESULTADOS / "mejores.csv").query("familia == @familia and mercado == 'reversion'").iloc[0]
            config = json.loads(f.config) | {"algoritmo": algoritmo, "seed": 300, "n_episodios": 200,
                                             "evaluar_cada": 200}
            s = ejecutar(config)
            filas.append({"agente": algoritmo, "seg_por_episodio": s["resultado"].segundos / 200})
            print(filas[-1], flush=True)
    pd.DataFrame(filas).to_csv(ruta, index=False)


def medir_escalado():
    """A3C con distinto número de trabajadores, uno detrás de otro: mide lo que aporta la asincronía."""
    ruta = RESULTADOS / "escalado_a3c.csv"
    if ruta.exists():
        return
    filas = []
    mejores = pd.read_csv(RESULTADOS / "mejores.csv").query("familia == 'a3c'").set_index("mercado")
    for mercado in MERCADOS_ESCALADO:
        for n in TRABAJADORES_ESCALADO:
            for seed in SEMILLAS_ESCALADO:
                config = json.loads(mejores.loc[mercado, "config"]) | {"n_trabajadores": n, "seed": seed,
                                                                       "n_episodios": 800}
                s = ejecutar(config)
                filas.append({"mercado": mercado, "trabajadores": n, "seed": seed,
                              "segundos": s["resultado"].segundos, "val_mediana": s["resumen"]["val_mediana"],
                              "episodio_convergencia": episodio_convergencia(s["resultado"].validacion)})
                print(filas[-1], flush=True)
    pd.DataFrame(filas).to_csv(ruta, index=False)


def main():
    evaluar_baselines()
    entrenar_finales()
    medir_tiempos()
    medir_escalado()
    print(f"[{time.strftime('%H:%M:%S')}] resultados terminados", flush=True)


if __name__ == "__main__":
    main()
