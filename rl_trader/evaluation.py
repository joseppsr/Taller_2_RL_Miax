"""Cómo se generan las series de precios y cómo se mide a un agente.

Las series de entrenamiento, validación y test salen de rangos de semillas que no
se solapan. Son los mismos rangos que en el taller anterior (repositorio
emszmtz/taller_RL), así que las 100 series de test son idénticas a las que usaron
los agentes tabulares y los resultados se pueden comparar directamente.
"""
import numpy as np
import pandas as pd

from .env import SimpleTradingEnv

SEMILLA_VALIDACION = 4_000_000_000
SEMILLA_TEST = 4_100_000_000
SEMILLA_NORMALIZADOR = 4_200_000_000

# Longitud de cada serie de precios, en pasos
N_STEPS = 500
# El entorno empieza en este paso para tener historia con la que calcular la tendencia
PASO_INICIAL = 5
CAPITAL_INICIAL = 1000.0
COMISION = 0.001


def semilla_entrenamiento(semilla_agente, episodio):
    """Devuelve la semilla de la serie que ve un agente en un episodio de entrenamiento."""
    if not (0 <= semilla_agente < 400 and 0 <= episodio < 10_000_000):
        raise ValueError("semilla o episodio fuera del rango reservado al entrenamiento")
    return semilla_agente * 10_000_000 + episodio


def generar_series(generador, n, semilla_base, n_steps=N_STEPS):
    """Genera n series de precios, siempre las mismas: la serie i usa la semilla semilla_base + i."""
    series = []
    for i in range(n):
        np.random.seed(semilla_base + i)
        series.append(generador(n_steps=n_steps))
    return series


def tendencias(precios):
    """Calcula la tendencia del entorno en cada paso, para una serie o para una matriz de series.

    Es la misma cuenta que hace SimpleTradingEnv._get_obs, de una vez: (precio −
    media de los últimos 6) / media. El resultado de la posición i corresponde al
    paso i + PASO_INICIAL de la serie.
    """
    ventanas = np.lib.stride_tricks.sliding_window_view(np.asarray(precios, dtype=float), 6, axis=-1)
    media = ventanas.mean(axis=-1)
    return np.clip((ventanas[..., -1] - media) / media, -1.0, 1.0).astype(np.float32)


def ejecutar_episodio(precios, politica):
    """Hace que una política recorra una serie en el entorno y devuelve (capital por paso, posición por paso)."""
    env = SimpleTradingEnv(precios)
    obs, _ = env.reset()
    balances, posiciones = [env.balance], []
    done = False
    while not done:
        obs, _, done, _, _ = env.step(politica(obs))
        balances.append(env.balance)
        posiciones.append(env.position)
    return np.array(balances), np.array(posiciones)


def simular_lote(precios, acciones):
    """Hace lo mismo que el entorno, pero para muchas series a la vez y mucho más rápido.

    precios tiene forma (series, T). acciones tiene forma (series, 2, T − 6) y da la
    acción que se tomaría en cada paso según la posición (0 o 1), así que no hace
    falta evaluar la red dentro del bucle. Devuelve las mismas matrices de capital
    y posiciones que ejecutar_episodio; tests/test_basico.py comprueba que coinciden
    exactamente con el entorno.
    """
    n, T = precios.shape
    pasos = T - 1 - PASO_INICIAL
    filas = np.arange(n)
    pos = np.zeros(n, dtype=int)
    efectivo = np.full(n, CAPITAL_INICIAL)
    titulos = np.zeros(n)
    balances = np.empty((n, pasos + 1))
    balances[:, 0] = CAPITAL_INICIAL
    posiciones = np.empty((n, pasos), dtype=int)
    for k in range(pasos):
        a = acciones[filas, pos, k]
        anterior = precios[:, PASO_INICIAL + k]
        compra = (a == 2) & (pos == 0)
        venta = (a == 0) & (pos == 1)
        titulos[compra] = efectivo[compra] * (1 - COMISION) / anterior[compra]
        efectivo[compra] = 0.0
        efectivo[venta] = titulos[venta] * anterior[venta] * (1 - COMISION)
        titulos[venta] = 0.0
        pos[compra] = 1
        pos[venta] = 0
        balances[:, k + 1] = efectivo + titulos * precios[:, PASO_INICIAL + k + 1]
        posiciones[:, k] = pos
    return balances, posiciones


def metricas(balances, posiciones):
    """Resume un episodio: retorno (%), ganancia G, Sharpe anualizado, máxima caída (%), operaciones y tiempo dentro (%).

    La ganancia es la suma de los retornos de cada paso, la G sin descontar que
    acumula el agente. Es prácticamente la suma de sus recompensas y, a diferencia del
    retorno en %, tiene una escala comparable entre mercados.
    """
    rets = balances[1:] / balances[:-1] - 1
    sd = rets.std()
    return {
        "retorno_pct": 100 * (balances[-1] / balances[0] - 1),
        "log_capital": float(np.log(balances[-1] / balances[0])),
        "ganancia": float(rets.sum()),
        "sharpe": np.sqrt(252) * rets.mean() / sd if sd > 0 else 0.0,
        "max_drawdown_pct": 100 * np.max(1 - balances / np.maximum.accumulate(balances)),
        "operaciones": int(np.abs(np.diff(posiciones, prepend=0)).sum()),
        "tiempo_dentro_pct": 100 * posiciones.mean(),
    }


def evaluar(politica, series):
    """Prueba una política en el entorno y devuelve una tabla con una fila de métricas por serie."""
    return pd.DataFrame([metricas(*ejecutar_episodio(p, politica)) for p in series])


def evaluar_lote(precios, acciones):
    """Como evaluar, pero con simular_lote: acciones son las tablas precalculadas de la red."""
    balances, posiciones = simular_lote(precios, acciones)
    return pd.DataFrame([metricas(b, p) for b, p in zip(balances, posiciones)])


def ic_bootstrap_pareado(x, y, n=10_000, nivel=0.95, seed=0):
    """Diferencia media pareada x − y (mismas series) con su intervalo de confianza bootstrap."""
    d = np.asarray(x, dtype=float) - np.asarray(y, dtype=float)
    rng = np.random.default_rng(seed)
    medias = d[rng.integers(0, len(d), size=(n, len(d)))].mean(axis=1)
    cola = (1 - nivel) / 2
    return d.mean(), np.quantile(medias, cola), np.quantile(medias, 1 - cola)


def episodio_convergencia(validacion, columna="retorno_mediana_pct", tol_rel=0.1, tol_abs=2.0, suavizado=3):
    """Calcula en qué episodio se estabilizó el resultado del agente.

    Las redes oscilan más que una tabla Q, así que primero se suaviza la curva de
    validación con una media móvil de 3 controles y se toma como valor final la
    media del último 20% de la curva. La convergencia es el primer control a partir
    del cual la curva ya no se aleja de ese valor más de un 10% o, si eso es muy
    poco, más de 2 puntos. Si no se estabiliza, devuelve el último control.
    """
    v = validacion[columna].rolling(suavizado, min_periods=1).mean().to_numpy()
    final = v[-max(1, len(v) // 5):].mean()
    lejos = np.flatnonzero(np.abs(v - final) > max(tol_rel * abs(final), tol_abs))
    k = min(lejos[-1] + 1, len(v) - 1) if len(lejos) else 0
    return int(validacion["episodio"].iloc[k])
