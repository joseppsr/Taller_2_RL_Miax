"""Lo que comparten DQN y A3C: el calendario de exploración y los controles en validación."""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .evaluation import evaluar_lote


def decaimiento_exponencial(episodio, n_episodios, inicial, final, frac_decaimiento):
    """Baja de forma exponencial de inicial a final en la primera parte del entrenamiento, y luego se queda fijo.

    Se usa para ε en DQN y para el coeficiente de entropía β en A3C.
    """
    n_dec = max(1, int(frac_decaimiento * n_episodios))
    if episodio >= n_dec:
        return final
    return inicial * (final / inicial) ** (episodio / n_dec)


@dataclass
class Resultado:
    """Lo que devuelve un entrenamiento.

    - tipo_red, ocultas y estado: para reconstruir la red. estado son los pesos del
      mejor control en validación; estado_final, los del último.
    - validacion: una fila por control (retorno, Sharpe, operaciones, tiempo acumulado).
    - historial: una fila por episodio de entrenamiento, con su ganancia G = Σ r y su
      retorno en %, explorando como se exploraba en ese momento.
    - segundos: tiempo real de entrenamiento, sin contar las evaluaciones.
    """
    algoritmo: str
    tipo_red: str
    ocultas: tuple
    estado: dict
    estado_final: dict
    validacion: pd.DataFrame
    historial: pd.DataFrame
    segundos: float
    pasos: int


def control_validacion(agente, precios):
    """Prueba la política greedy del agente en las series de validación y resume el resultado."""
    df = evaluar_lote(precios, agente.acciones_greedy(precios))
    return {
        "retorno_mediana_pct": df["retorno_pct"].median(),
        "retorno_pct": df["retorno_pct"].mean(),
        "log_capital": df["log_capital"].mean(),
        "ganancia": df["ganancia"].mean(),
        "sharpe": df["sharpe"].mean(),
        "operaciones": df["operaciones"].mean(),
        "dentro_pct": df["tiempo_dentro_pct"].mean(),
    }


class MejorControl:
    """Recuerda los pesos del mejor control en validación.

    Las políticas aprendidas con redes oscilan durante el entrenamiento, así que el
    último control no es necesariamente el mejor. La nota es la mediana del retorno;
    si hay empate (por ejemplo, varios controles que no operan), gana el que opera menos.
    """

    def __init__(self):
        self.nota, self.estado = (-np.inf, -np.inf), None

    def actualizar(self, control, estado):
        nota = (control["retorno_mediana_pct"], -control["operaciones"])
        if nota > self.nota:
            self.nota, self.estado = nota, estado
