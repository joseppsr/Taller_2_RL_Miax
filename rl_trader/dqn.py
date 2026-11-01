"""DQN con memoria de experiencia y red objetivo (slides 1), y sus variantes doble y en duelo (slides 5).

Algoritmo:
1. Inicializar los pesos de Q y copiarlos en la red objetivo Q'.
2. En cada episodio, con una serie de precios nueva, elegir acciones ε-greedy y
   guardar cada experiencia (s, a, r, s', terminal) en un buffer limitado.
3. Cada pocos pasos, entrenar Q con un minibatch al azar del buffer, que rompe la
   correlación entre experiencias seguidas. El target es
       r + γ · max_a' Q'(s', a')                  en DQN
       r + γ · Q'(s', argmax_a' Q(s', a'))        en doble DQN (reduce la sobreestimación)
4. Cada C pasos, copiar Q en Q'.

Como comparación, con modo="episodio" no hay minibatches ni buffer: al terminar cada serie,
la red ve el episodio entero, de una vez y en sus pasos de gradiente. Se pierde la mezcla de
experiencias de series distintas, que es justo lo que el buffer aporta.
"""
import time

import numpy as np
import pandas as pd
import torch
from torch import nn

from .agentes import Agente
from .entrenamiento import MejorControl, Resultado, control_validacion, decaimiento_exponencial
from .env import SimpleTradingEnv
from .evaluation import N_STEPS, semilla_entrenamiento
from .redes import N_ACCIONES, clonar_estado, crear_red, fijar_hilos


class Buffer:
    """Memoria de experiencia circular con arrays de NumPy."""

    def __init__(self, capacidad):
        self.capacidad = capacidad
        self.s = np.empty((capacidad, 2), dtype=np.float32)
        self.s2 = np.empty((capacidad, 2), dtype=np.float32)
        self.a = np.empty(capacidad, dtype=np.int64)
        self.r = np.empty(capacidad, dtype=np.float32)
        self.terminal = np.empty(capacidad, dtype=np.float32)
        self.n = self.i = 0

    def guardar(self, s, a, r, s2, terminal):
        i = self.i
        self.s[i], self.a[i], self.r[i], self.s2[i], self.terminal[i] = s, a, r, s2, terminal
        self.i = (i + 1) % self.capacidad
        self.n = min(self.n + 1, self.capacidad)

    def muestra(self, tam, rng):
        idx = rng.integers(0, self.n, size=tam)
        return (torch.from_numpy(self.s[idx]), torch.from_numpy(self.a[idx]), torch.from_numpy(self.r[idx]),
                torch.from_numpy(self.s2[idx]), torch.from_numpy(self.terminal[idx]))

    def vaciar_episodio(self):
        """Devuelve todas las experiencias guardadas, en orden, y deja el buffer vacío."""
        n = self.n
        self.n = self.i = 0
        return (torch.from_numpy(self.s[:n].copy()), torch.from_numpy(self.a[:n].copy()),
                torch.from_numpy(self.r[:n].copy()), torch.from_numpy(self.s2[:n].copy()),
                torch.from_numpy(self.terminal[:n].copy()))


class AgenteDQN(Agente):
    """El agente que aprende: la red Q en self.modelo y la red objetivo Q' en self.modelo_objetivo."""

    def __init__(self, normalizador, ocultas, alpha, gamma, doble, duelo, perdida, capacidad, max_norma_grad, rng):
        self.tipo_red = "duelo" if duelo else "q"
        super().__init__(crear_red(self.tipo_red, ocultas), normalizador)
        self.modelo_objetivo = crear_red(self.tipo_red, ocultas, clonar_estado(self.modelo))
        self.optimizador = torch.optim.Adam(self.modelo.parameters(), lr=alpha)
        self.funcion_perdida = {"mse": nn.MSELoss(), "mae": nn.L1Loss()}[perdida]
        self.buffer = Buffer(capacidad)
        self.gamma, self.doble, self.max_norma_grad, self.rng = gamma, doble, max_norma_grad, rng

    def elegir_accion(self, s, eps):
        """Elige la acción con ε-greedy."""
        if self.rng.random() < eps:
            return int(self.rng.integers(N_ACCIONES))
        with torch.inference_mode():
            return int(self.modelo(torch.from_numpy(s)).argmax())

    def aprender(self, s, a, r, s2, terminal):
        """Un paso de gradiente hacia el target de Bellman con las experiencias dadas."""
        with torch.no_grad():
            if self.doble:
                accion = self.modelo(s2).argmax(dim=1, keepdim=True)
                siguiente = self.modelo_objetivo(s2).gather(1, accion).squeeze(1)
            else:
                siguiente = self.modelo_objetivo(s2).max(dim=1).values
            objetivo = r + self.gamma * (1.0 - terminal) * siguiente
        prediccion = self.modelo(s).gather(1, a.unsqueeze(1)).squeeze(1)
        loss = self.funcion_perdida(prediccion, objetivo)
        self.optimizador.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.modelo.parameters(), self.max_norma_grad)
        self.optimizador.step()

    def sincronizar(self):
        """Copia Q en Q'."""
        self.modelo_objetivo.load_state_dict(self.modelo.state_dict())


def entrenar_dqn(generador, normalizador, n_episodios=1000, n_steps=N_STEPS, alpha=5e-4, gamma=0.9,
                 eps_inicial=1.0, eps_final=0.01, frac_decaimiento=0.7, tam_lote=64, capacidad=100_000,
                 inicio_aprendizaje=1000, entrenar_cada=4, sincronizar_cada=1000, doble=False, duelo=False,
                 ocultas=(64, 64), seed=0, precios_validacion=None, evaluar_cada=50, max_norma_grad=10.0,
                 perdida="mse", modo="minibatch", pasadas_episodio=1):
    """Entrena un agente DQN y devuelve un Resultado.

    Cada episodio usa una serie de precios nueva del generador. Parámetros principales:
    - alpha: factor de aprendizaje de Adam.
    - gamma: factor de descuento.
    - eps_inicial, eps_final y frac_decaimiento: decaimiento de ε.
    - doble y duelo: activan las variantes de las slides 5.
    - perdida: "mse" como en las slides 1, o "mae". Con MSE, Q estima la media del target;
      con MAE, su mediana, y el error grande pesa menos.
    - modo: "minibatch" (el algoritmo de las slides) o "episodio" (la red ve la serie entera
      al terminarla, sin minibatches, durante pasadas_episodio pasos de gradiente).
    - precios_validacion y evaluar_cada: matriz de series para los controles periódicos.
    """
    fijar_hilos()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    agente = AgenteDQN(normalizador, ocultas, alpha, gamma, doble, duelo, perdida,
                       capacidad if modo == "minibatch" else n_steps, max_norma_grad, rng)
    mejor = MejorControl()
    filas, historial = [], []
    pasos, t_evaluando = 0, 0.0
    inicio = time.perf_counter()

    for ep in range(n_episodios):
        eps = decaimiento_exponencial(ep, n_episodios, eps_inicial, eps_final, frac_decaimiento)
        np.random.seed(semilla_entrenamiento(seed, ep))
        env = SimpleTradingEnv(generador(n_steps=n_steps))
        obs, _ = env.reset()
        s = normalizador.entrada(obs[0], obs[1])
        terminado = False
        ganancia = 0.0
        while not terminado:
            a = agente.elegir_accion(s, eps)
            obs, r, terminado, _, _ = env.step(a)
            ganancia += r
            s2 = normalizador.entrada(obs[0], obs[1])
            agente.buffer.guardar(s, a, r / normalizador.escala_recompensa, s2, float(terminado))
            s = s2
            pasos += 1

            if modo == "minibatch" and agente.buffer.n >= inicio_aprendizaje and pasos % entrenar_cada == 0:
                agente.aprender(*agente.buffer.muestra(tam_lote, rng))
            if pasos % sincronizar_cada == 0:
                agente.sincronizar()

        if modo == "episodio":
            episodio = agente.buffer.vaciar_episodio()
            for _ in range(pasadas_episodio):
                agente.aprender(*episodio)

        historial.append({"episodio": ep + 1, "ganancia": ganancia,
                          "retorno_pct": 100 * (env.balance / env.initial_balance - 1)})

        # Controles en validación: ese tiempo no cuenta como entrenamiento
        if precios_validacion is not None and ((ep + 1) % evaluar_cada == 0 or ep + 1 == n_episodios):
            t0 = time.perf_counter()
            control = control_validacion(agente, precios_validacion)
            mejor.actualizar(control, clonar_estado(agente.modelo))
            filas.append({"episodio": ep + 1, "pasos": pasos, "segundos": t0 - inicio - t_evaluando,
                          "epsilon": eps, **control})
            t_evaluando += time.perf_counter() - t0

    segundos = time.perf_counter() - inicio - t_evaluando
    final = clonar_estado(agente.modelo)
    algoritmo = "duelo" if duelo else ("doble_dqn" if doble else "dqn")
    return Resultado(algoritmo, agente.tipo_red, tuple(ocultas), mejor.estado or final, final,
                     pd.DataFrame(filas), pd.DataFrame(historial), segundos, pasos)
