"""Tablas de resultados a partir de lo que guardan scripts/ajuste.py y scripts/resultados.py."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .evaluation import episodio_convergencia, ic_bootstrap_pareado

RESULTADOS = Path(__file__).resolve().parent.parent / "results"
# Tolerancia absoluta del criterio de convergencia sobre G (la relativa sigue siendo del 10%)
TOLERANCIA_G = 0.02

MERCADOS = ["ciclo", "browniano", "reversion", "regimenes"]
NOMBRES_MERCADO = {"ciclo": "1. Tendencia + ciclo", "browniano": "2. Browniano",
                   "reversion": "3. Reversión a la media", "regimenes": "4. Regímenes"}
BASELINES = ["aleatorio", "buy_and_hold", "momentum", "contrarian"]
AGENTES_RL = ["dqn", "doble_dqn", "duelo", "a3c", "reinforce"]
NOMBRES_AGENTE = {"aleatorio": "Aleatorio", "buy_and_hold": "Buy & hold", "momentum": "Momentum",
                  "contrarian": "Contrarian", "dqn": "DQN", "doble_dqn": "Doble DQN", "duelo": "DQN en duelo",
                  "a3c": "A3C", "reinforce": "REINFORCE"}


def cargar_test():
    """Une las métricas por serie de test de los agentes (solo pesos elegidos) y de las referencias."""
    rl = pd.read_csv(RESULTADOS / "test_por_serie.csv.gz")
    return pd.concat([rl[rl["version"] == "test"], pd.read_csv(RESULTADOS / "test_baselines.csv")], ignore_index=True)


def tabla_resultados(test=None):
    """Una fila por mercado y agente: rendimiento en test y convergencia.

    El tiempo de convergencia es el episodio de convergencia (mediana de las semillas)
    por los segundos por episodio medidos con la máquina descargada. Se mide sobre la
    mediana del retorno y, aparte, sobre la ganancia G: en ciclo, una mediana de +80.000%
    varía más de un 10% entre controles por puro ruido, y solo la segunda es informativa.
    """
    test = cargar_test() if test is None else test
    entrenamientos = pd.read_csv(RESULTADOS / "entrenamientos.csv")
    curvas = pd.read_csv(RESULTADOS / "curvas_validacion.csv")
    tiempos = pd.read_csv(RESULTADOS / "tiempos.csv").set_index("agente")["seg_por_episodio"]
    filas = []
    for mercado in MERCADOS:
        bh = test[(test.mercado == mercado) & (test.agente == "buy_and_hold")].set_index("serie")["retorno_pct"]
        for agente in BASELINES + AGENTES_RL:
            d = test[(test.mercado == mercado) & (test.agente == agente)]
            fila = {"mercado": mercado, "agente": agente,
                    "mediana_pct": d["retorno_pct"].median(), "media_pct": d["retorno_pct"].mean(),
                    "ganancia_G": d["ganancia"].mean(), "sharpe": d["sharpe"].mean(),
                    "max_caida_pct": d["max_drawdown_pct"].mean(), "operaciones": d["operaciones"].mean(),
                    "dentro_pct": d["tiempo_dentro_pct"].mean(),
                    "supera_BH_pct": 100 * (d["retorno_pct"].to_numpy() > bh.reindex(d["serie"]).to_numpy()).mean()}
            e = entrenamientos[(entrenamientos.mercado == mercado) & (entrenamientos.agente == agente)]
            if len(e):
                por_semilla = d.groupby("seed")["retorno_pct"].median()
                conv = e["episodio_convergencia"].median()
                c = curvas[(curvas.mercado == mercado) & (curvas.agente == agente)]
                conv_g = np.median([episodio_convergencia(x.sort_values("episodio"), columna="ganancia",
                                                          tol_abs=TOLERANCIA_G) for _, x in c.groupby("seed")])
                fila |= {"mediana_min_semilla": por_semilla.min(), "mediana_max_semilla": por_semilla.max(),
                         "episodios": e["n_episodios"].iloc[0], "convergencia_ep": conv,
                         "convergencia_s": conv * tiempos[agente], "convergencia_G_ep": conv_g,
                         "convergencia_G_s": conv_g * tiempos[agente],
                         "entrenamiento_s": e["n_episodios"].iloc[0] * tiempos[agente]}
            filas.append(fila)
    return pd.DataFrame(filas)


def comparar(test, mercado, agente_a, agente_b):
    """Cociente geométrico medio del capital final de A frente a B, con IC bootstrap del 95% (series pareadas)."""
    def log_medio(agente):
        d = test[(test.mercado == mercado) & (test.agente == agente)]
        return d.groupby("serie")["log_capital"].mean().sort_index()
    media, baja, alta = ic_bootstrap_pareado(log_medio(agente_a), log_medio(agente_b))
    return np.exp(media), np.exp(baja), np.exp(alta)


def tabla_comparaciones(test=None, referencias=("buy_and_hold", "aleatorio", "momentum", "contrarian"),
                        agentes=tuple(AGENTES_RL)):
    """Cada agente frente a cada referencia, en cada mercado; significativo si el IC no incluye el 1."""
    test = cargar_test() if test is None else test
    filas = []
    for mercado in MERCADOS:
        for agente in agentes:
            for referencia in referencias:
                x, baja, alta = comparar(test, mercado, agente, referencia)
                filas.append({"mercado": mercado, "agente": agente, "referencia": referencia, "cociente": x,
                              "ic_bajo": baja, "ic_alto": alta, "significativo": not baja <= 1 <= alta})
    return pd.DataFrame(filas)


def tabla_perdidas():
    """Compara MSE y MAE: nota media en validación de cada pérdida (etapa C del ajuste)."""
    salida = []
    for familia, columna in (("dqn", "perdida"), ("a3c", "perdida_critico")):
        d = pd.read_csv(RESULTADOS / f"ajuste_{familia}_c.csv")
        agregado = d.groupby(["mercado", columna])[["val_mediana", "val_ganancia", "val_operaciones",
                                                    "episodio_convergencia"]].mean().reset_index()
        salida.append(agregado.rename(columns={columna: "perdida"}).assign(familia=familia))
    return pd.concat(salida, ignore_index=True)


def diagnostico_convergencia(curvas):
    """Mide por mercado y agente si la curva de validación se ha estabilizado, sobre la ganancia G.

    Compara, en cada semilla, el último cuarto de la curva con el cuarto anterior:
    - deriva: cuánto sigue cambiando el nivel. Si es grande, aún no ha llegado.
    - oscilacion: desviación típica dentro del último cuarto. Si es grande, no se asienta.
    - caida_desde_mejor: cuánto queda el nivel final por debajo del mejor control. Si es
      grande, el mejor control fue un pico puntual, no un nivel que se mantiene.
    - rango_semillas: distancia entre semillas al final. Si es grande, depende de la suerte.
    """
    filas = []
    for (mercado, agente), datos in curvas.groupby(["mercado", "agente"]):
        por_semilla = []
        for _, d in datos.groupby("seed"):
            g = d.sort_values("episodio")["ganancia"].to_numpy()
            cuarto = max(1, len(g) // 4)
            ultimo, anterior = g[-cuarto:], g[-2 * cuarto:-cuarto]
            por_semilla.append({"final": ultimo.mean(), "deriva": ultimo.mean() - anterior.mean(),
                                "oscilacion": ultimo.std(), "caida_desde_mejor": g.max() - ultimo.mean()})
        s = pd.DataFrame(por_semilla)
        filas.append({"mercado": mercado, "agente": agente, **s.mean().to_dict(),
                      "rango_semillas": s["final"].max() - s["final"].min()})
    return pd.DataFrame(filas)


def mejor_frente_a_ultimo():
    """Mediana en test con los pesos del mejor control y con los del último: si difieren mucho, la política oscila."""
    d = pd.read_csv(RESULTADOS / "test_por_serie.csv.gz")
    tabla = d[d["version"].isin(["test", "test_ultimo"])].groupby(["mercado", "agente", "version"])["retorno_pct"]
    return tabla.median().unstack("version")


def entropia_relativa(logits):
    """Entropía media de softmax(logits), en proporción a la máxima. Resta el máximo para que exp no desborde."""
    log_p = logits - logits.max(axis=-1, keepdims=True)
    log_p = log_p - np.log(np.exp(log_p).sum(axis=-1, keepdims=True))
    return float(-(np.exp(log_p) * log_p).sum(axis=-1).mean() / np.log(3))


def entropia_final():
    """Entropía media de la política final de los métodos de actor, en proporción a la máxima (log 3).

    Cerca de 1, la política sigue siendo casi aleatoria. Cerca de 0, ya es casi determinista.
    """
    mapas = np.load(RESULTADOS / "mapas_politica.npz")
    filas = []
    for clave in mapas.files:
        mercado, agente, seed = clave.split("|")
        if agente in ("a3c", "reinforce"):
            filas.append({"mercado": mercado, "agente": agente, "seed": int(seed),
                          "entropia_relativa": entropia_relativa(mapas[clave])})
    return pd.DataFrame(filas).groupby(["mercado", "agente"])["entropia_relativa"].mean().unstack()


def configuraciones_elegidas():
    """Una fila por familia y mercado con los hiperparámetros elegidos en el ajuste."""
    m = pd.read_csv(RESULTADOS / "mejores.csv")
    return pd.DataFrame([{"familia": f.familia, "mercado": f.mercado, **json.loads(f.config)} for f in m.itertuples()])
