"""Los agentes: una red en self.modelo y cómo decidir con ella.

La red se llama siempre directamente, self.modelo(x), con toda la serie de una vez: no hay
ningún predict que la parta por lotes. Cada paso de cada serie se evalúa con las dos
posiciones posibles, porque la que toque depende de las acciones anteriores.
"""
import numpy as np
import torch

from .evaluation import tendencias
from .redes import RedActorCritico, crear_red


class Agente:
    """Agente basado en valor: self.modelo(x) devuelve Q(s, a), y se elige la acción de mayor valor."""

    def __init__(self, modelo, normalizador):
        self.modelo = modelo
        self.normalizador = normalizador

    def preferencias(self, x):
        """Devuelve la salida de la red que ordena las acciones."""
        return self.modelo(x)

    @torch.inference_mode()
    def tablas(self, precios):
        """Evalúa la red en todos los pasos de todas las series: array (series, 2, pasos, 3)."""
        t = tendencias(precios)[:, :-1]
        salida = [self.preferencias(torch.from_numpy(self.normalizador.entrada(t, np.full_like(t, p)))).numpy()
                  for p in (0, 1)]
        return np.stack(salida, axis=1)

    def acciones_greedy(self, precios):
        """Tablas de acciones de la política greedy, listas para evaluation.simular_lote."""
        return self.tablas(precios).argmax(axis=-1)

    @torch.inference_mode()
    def __call__(self, obs):
        """Decide la acción greedy para una observación del entorno."""
        return int(self.preferencias(self.normalizador.obs(obs)).argmax())

    @torch.inference_mode()
    def mapa_politica(self, z=np.linspace(-4, 4, 81)):
        """Evalúa la política continua en una rejilla de tendencias (en desviaciones típicas).

        Devuelve las preferencias de cada acción, (2, len(z), 3). No hay intervalos: la
        frontera de decisión es una curva.
        """
        t = np.asarray(z, dtype=np.float32) * self.normalizador.escala_tendencia
        return np.stack([self.preferencias(torch.from_numpy(self.normalizador.entrada(t, np.full_like(t, p)))).numpy()
                         for p in (0, 1)])


class AgenteActorCritico(Agente):
    """Agente basado en política: self.modelo(x) devuelve (logits del actor, valor del crítico)."""

    def preferencias(self, x):
        return self.modelo(x)[0]

    def acciones_muestreadas(self, precios, seed=0):
        """Tablas de acciones muestreadas de la política estocástica del actor."""
        logits = self.tablas(precios)
        # Muestreo de una categórica con el truco de Gumbel: argmax(logits + ruido)
        ruido = np.random.default_rng(seed).gumbel(size=logits.shape)
        return (logits + ruido).argmax(axis=-1)


def crear_agente(tipo_red, ocultas, estado, normalizador):
    """Reconstruye un agente a partir de los pesos guardados."""
    modelo = crear_red(tipo_red, ocultas, estado)
    clase = AgenteActorCritico if isinstance(modelo, RedActorCritico) else Agente
    return clase(modelo, normalizador)
