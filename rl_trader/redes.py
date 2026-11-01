"""Las redes neuronales y la preparación de la entrada que reciben.

En el taller anterior la tendencia se discretizaba en intervalos para indexar una
tabla Q. Aquí no hay intervalos: la red recibe la tendencia continua directamente
y generaliza entre valores parecidos. Lo único que se hace es normalizar la escala
(Normalizador), porque varía mucho entre mercados.
"""
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from .evaluation import N_STEPS, SEMILLA_NORMALIZADOR, generar_series, tendencias

N_ACCIONES = 3
# Una tendencia de más de 5 desviaciones típicas no aporta nada distinto y desestabiliza la red
CLIP_TENDENCIA = 5.0


@dataclass
class Normalizador:
    """Lleva la observación y la recompensa a una escala parecida en todos los mercados.

    - La tendencia se divide por su desviación típica. En el mercado de ciclo se
      mueve en ±0,3 y en los demás en ±0,05; sin esto, la red apenas la vería.
    - La posición (0 o 1) pasa a ser −1 o +1.
    - La recompensa se divide por la desviación típica de los retornos del mercado.
      Multiplicar por una constante positiva no cambia la política óptima, y evita
      que los errores de Bellman sean diminutos frente a la inicialización de la red.
      Sustituye al recorte de refuerzos a ±1 de las slides 1, que aquí borraría la
      diferencia entre un retorno pequeño y uno grande.
    """
    escala_tendencia: float
    escala_recompensa: float

    @classmethod
    def ajustar(cls, generador, n_series=50, n_steps=N_STEPS):
        """Calcula las escalas con series reservadas para esto, distintas de las de validación y test."""
        series = np.array(generar_series(generador, n_series, SEMILLA_NORMALIZADOR, n_steps))
        return cls(float(tendencias(series).std()), float((series[:, 1:] / series[:, :-1] - 1).std()))

    def entrada(self, tendencia, posicion):
        """Devuelve la entrada de la red, para una observación o para arrays de ellas."""
        t = np.clip(np.asarray(tendencia, dtype=np.float32) / self.escala_tendencia, -CLIP_TENDENCIA, CLIP_TENDENCIA)
        p = 2.0 * np.asarray(posicion, dtype=np.float32) - 1.0
        return np.stack([t, p * np.ones_like(t)], axis=-1)

    def obs(self, obs):
        """Convierte una observación del entorno en un tensor con la entrada de la red."""
        return torch.from_numpy(self.entrada(obs[0], obs[1]))


def _mlp(entrada, ocultas, salida):
    capas, n = [], entrada
    for h in ocultas:
        capas += [nn.Linear(n, h), nn.ReLU()]
        n = h
    capas.append(nn.Linear(n, salida))
    return nn.Sequential(*capas)


class RedQ(nn.Module):
    """Red de DQN: una salida lineal Q(s, a) por acción (slides 1)."""

    def __init__(self, ocultas=(64, 64)):
        super().__init__()
        self.red = _mlp(2, ocultas, N_ACCIONES)

    def forward(self, x):
        return self.red(x)


class RedDuelo(nn.Module):
    """DQN en duelo: Q(s,a) = V(s) + A(s,a) − media_a'(A(s,a')) (slides 5)."""

    def __init__(self, ocultas=(64, 64)):
        super().__init__()
        self.base = _mlp(2, ocultas[:-1], ocultas[-1])
        self.valor = nn.Sequential(nn.ReLU(), nn.Linear(ocultas[-1], 1))
        self.ventaja = nn.Sequential(nn.ReLU(), nn.Linear(ocultas[-1], N_ACCIONES))

    def forward(self, x):
        h = self.base(x)
        a = self.ventaja(h)
        return self.valor(h) + a - a.mean(dim=-1, keepdim=True)


class RedActorCritico(nn.Module):
    """Dos modelos separados, como en las slides 7: el actor π(a|s) y el crítico v(s).

    Se separan porque el error del crítico, muy ruidoso con estos precios, tapa la
    señal de la política si comparten las capas ocultas.
    """

    def __init__(self, ocultas=(64, 64)):
        super().__init__()
        self.actor = _mlp(2, ocultas, N_ACCIONES)
        self.critico = _mlp(2, ocultas, 1)

    def forward(self, x):
        return self.actor(x), self.critico(x).squeeze(-1)


REDES = {"q": RedQ, "duelo": RedDuelo, "actor_critico": RedActorCritico}


def crear_red(tipo, ocultas=(64, 64), estado=None):
    """Crea una red, con los pesos indicados si se dan."""
    red = REDES[tipo](tuple(ocultas))
    if estado is not None:
        red.load_state_dict(estado)
    return red


def clonar_estado(red):
    """Copia los pesos de una red, para guardar un punto de control."""
    return {k: v.detach().clone() for k, v in red.state_dict().items()}


def fijar_hilos():
    """Cada entrenamiento usa un único hilo: estas redes son tan pequeñas que más hilos (o la GPU) van más lentos."""
    torch.set_num_threads(1)
