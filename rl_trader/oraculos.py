"""Techo teórico de cada mercado: lo que lograría quien conociera la información oculta.

Sirven para comprobar si los resultados tienen sentido. Ningún agente ve esta información:
solo observa la tendencia y su posición. Un agente no debería superar a su oráculo, y el
hueco indica cuánto se pierde por observar solo la tendencia.

- Ciclo: conoce el retorno esperado del paso siguiente (la fase del ciclo).
- Browniano: no hay información que explotar; lo mejor en esperanza es estar siempre dentro.
- Reversión: conoce la media (100) y usa el nivel de precio, con una banda para ahorrar
  comisiones. Se queda con la mejor banda, así que es un techo optimista.
- Regímenes: conoce el régimen oculto vigente.
"""
import numpy as np
import pandas as pd

from .evaluation import evaluar_lote, generar_series
from .prices import GENERADORES

BANDAS_REVERSION = (0, 2, 5, 8, 12)


def _tablas(dentro):
    a = np.where(dentro, 2, 0)
    return np.stack([a, a], axis=1)


def _regimenes_con_estado(n_steps=500, start_price=100.0):
    """Copia de generar_precios_regimenes que devuelve además el régimen de cada paso."""
    estado, estados, retornos = 0, [], []
    for _ in range(n_steps):
        if np.random.rand() < 0.02:
            estado = 1 - estado
        retornos.append(np.random.normal(0.001, 0.01) if estado == 0 else np.random.normal(-0.0015, 0.025))
        estados.append(estado)
    return start_price * np.cumprod(1 + np.array(retornos)), np.array(estados)


def _ciclo(n, semilla):
    p = np.array(generar_series(GENERADORES["ciclo"], n, semilla))
    t = np.linspace(0, 50, p.shape[1])
    return evaluar_lote(p, _tablas(np.tile((0.0005 * t + 0.05 * np.sin(t))[6:] > 0, (n, 1))))


def _browniano(n, semilla):
    p = np.array(generar_series(GENERADORES["browniano"], n, semilla))
    return evaluar_lote(p, _tablas(np.ones((n, p.shape[1] - 6), dtype=bool)))


def _reversion(n, semilla):
    p = np.array(generar_series(GENERADORES["reversion"], n, semilla))
    mejor = None
    for banda in BANDAS_REVERSION:
        dentro = np.zeros((n, p.shape[1] - 6), dtype=bool)
        actual = np.zeros(n, dtype=bool)
        for k in range(dentro.shape[1]):
            nivel = p[:, 5 + k]
            actual = np.where(nivel < 100 - banda, True, np.where(nivel > 100 + banda, False, actual))
            dentro[:, k] = actual
        df = evaluar_lote(p, _tablas(dentro))
        if mejor is None or df["retorno_pct"].median() > mejor["retorno_pct"].median():
            mejor = df
    return mejor


def _regimenes(n, semilla):
    series, estados = [], []
    for i in range(n):
        np.random.seed(semilla + i)
        precios, estado = _regimenes_con_estado()
        series.append(precios)
        estados.append(estado)
    return evaluar_lote(np.array(series), _tablas(np.array(estados)[:, 6:] == 0))


ORACULOS = {"ciclo": _ciclo, "browniano": _browniano, "reversion": _reversion, "regimenes": _regimenes}


def tabla_oraculos(n, semilla):
    """Una fila por mercado con la mediana del retorno, la ganancia y las operaciones del oráculo."""
    filas = []
    for mercado, oraculo in ORACULOS.items():
        df = oraculo(n, semilla)
        filas.append({"mercado": mercado, "mediana_pct": df["retorno_pct"].median(), "ganancia_G": df["ganancia"].mean(),
                      "operaciones": df["operaciones"].mean()})
    return pd.DataFrame(filas)
