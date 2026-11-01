"""Estrategias fijas con las que comparamos a los agentes.

Cada estrategia es una función que recibe lo que ve el agente, (tendencia,
posición), y devuelve una acción: 0 = vender, 1 = mantener, 2 = comprar.
"""
import numpy as np


def aleatoria(seed=0):
    """El agente aleatorio del enunciado: en cada paso elige una acción al azar."""
    rng = np.random.default_rng(seed)
    return lambda obs: int(rng.integers(3))


def buy_and_hold(obs):
    """Compra en el primer paso y ya no vende nunca."""
    return 2


def momentum(obs):
    """Sigue la tendencia: está dentro cuando es positiva y fuera cuando es negativa."""
    return 2 if obs[0] > 0 else 0


def contrarian(obs):
    """Va contra la tendencia: está dentro cuando es negativa y fuera cuando es positiva."""
    return 2 if obs[0] < 0 else 0


# Para usar una estrategia: BASELINES[nombre]() devuelve la función lista para usar
BASELINES = {
    "aleatorio": aleatoria,
    "buy_and_hold": lambda: buy_and_hold,
    "momentum": lambda: momentum,
    "contrarian": lambda: contrarian,
}
