"""A3C: Asynchronous Advantage Actor-Critic (slides 7), y REINFORCE (slides 6).

Varios procesos trabajadores recorren series de precios distintas a la vez, cada uno
con su copia de la red. Cada n pasos, el trabajador:
1. Copia los pesos globales en su red local.
2. Actúa n pasos muestreando de π(a|s) y calcula los retornos
   G_t = r_t + γ·r_{t+1} + ... + γ^(n−k)·v(s_{t+n}), con v = 0 si la serie ha terminado.
3. Calcula los gradientes de la ventaja δ_t = G_t − v(s_t):
   - actor: −log π(a_t|s_t) · δ_t − β · H(π(·|s_t)), donde la entropía H sustituye a ε
     como mecanismo de exploración;
   - crítico: δ_t² / 2 (MSE, como en las slides) o |δ_t| (MAE).
4. Aplica esos gradientes a los pesos globales sin esperar a que los demás terminen los suyos.
   Esa asincronía es la que descorrelaciona las experiencias, y por eso A3C no necesita buffer.

El paso de Adam, y solo él, va con un cerrojo corto. Sin él, un trabajador puede leer la media m
ya actualizada por otro con la varianza v aún a cero, y el paso m / (√v + 1e-8) lleva los pesos
de 0,7 a más de 1.000 en una sola actualización: la política se satura y el agente deja de operar.

Actor y crítico tienen cada uno su red y su tamaño de paso (α_θ y α_w), como en las slides 7.

REINFORCE (slides 6) es el caso límite con un solo trabajador y n = ∞: se actualiza al final
del episodio con el retorno completo G_t, sin bootstrap y sin crítico.

Diferencia deliberada con las slides: no se multiplica el gradiente del actor por I = γ^t.
Ese factor optimiza el valor del estado inicial, pero aquí el primer paso de la serie no tiene
nada de especial: con γ = 0,9 anularía el aprendizaje del actor a los pocos pasos de cada serie
(y con γ = 0, después del primero). Tampoco lo usa el artículo original de A3C.
"""
import multiprocessing as mp
import threading
import time

import numpy as np
import pandas as pd
import torch
from torch import nn

from .agentes import AgenteActorCritico
from .entrenamiento import MejorControl, Resultado, control_validacion, decaimiento_exponencial
from .env import SimpleTradingEnv
from .evaluation import N_STEPS, semilla_entrenamiento
from .redes import clonar_estado, crear_red, fijar_hilos

TIPO_RED = "actor_critico"
# Los n pasos de REINFORCE: más que cualquier episodio, así que se actualiza al final de cada uno
N_PASOS_EPISODIO = 10 ** 9


def _estado_adam_compartido(red):
    """Crea el estado de Adam en memoria compartida, para que los trabajadores compartan el optimizador."""
    return [{"step": torch.zeros(()).share_memory_(),
             "exp_avg": torch.zeros_like(p).share_memory_(),
             "exp_avg_sq": torch.zeros_like(p).share_memory_()} for p in red.parameters()]


class Trabajador:
    """Un trabajador de A3C: su red local en self.modelo y los pesos compartidos en self.modelo_global."""

    def __init__(self, indice, modelo_global, estado_adam, cerrojo, contador, ganancias, retornos_pct, generador,
                 normalizador, c):
        self.modelo_global, self.cerrojo, self.contador = modelo_global, cerrojo, contador
        self.ganancias, self.retornos_pct = ganancias, retornos_pct
        self.generador, self.normalizador, self.c = generador, normalizador, c
        torch.manual_seed(c["seed"] * 1000 + indice)
        self.rng = np.random.default_rng([c["seed"], indice])
        self.optimizador = torch.optim.Adam([{"params": modelo_global.actor.parameters(), "lr": c["alpha"]},
                                             {"params": modelo_global.critico.parameters(), "lr": c["alpha_critico"]}])
        for p, estado in zip(modelo_global.parameters(), estado_adam):
            self.optimizador.state[p] = estado
        self.modelo = crear_red(TIPO_RED, c["ocultas"])

    def _actualizar(self, estados, acciones, recompensas, obs, terminado, beta):
        """Calcula los retornos a n pasos y aplica los gradientes a los pesos globales."""
        c = self.c
        with torch.inference_mode():
            G = 0.0 if terminado else float(self.modelo.critico(self.normalizador.obs(obs)))
        retornos = np.empty(len(recompensas), dtype=np.float32)
        for k in range(len(recompensas) - 1, -1, -1):
            G = recompensas[k] + c["gamma"] * G
            retornos[k] = G

        logits, valores = self.modelo(torch.from_numpy(np.stack(estados)))
        log_probs = torch.log_softmax(logits, dim=-1)
        retornos_t = torch.from_numpy(retornos)
        ventaja = retornos_t - valores
        acciones_t = torch.as_tensor(acciones).unsqueeze(1)
        peso_actor = ventaja.detach() if c["usar_critico"] else retornos_t
        perdida_actor = -(log_probs.gather(1, acciones_t).squeeze(1) * peso_actor).mean()
        entropia = -(log_probs.exp() * log_probs).sum(dim=-1).mean()
        perdida_critico = 0.5 * ventaja.pow(2).mean() if c["perdida_critico"] == "mse" else ventaja.abs().mean()
        loss = perdida_actor - beta * entropia + perdida_critico

        self.modelo.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.modelo.parameters(), c["max_norma_grad"])
        # Los gradientes calculados en la red local se aplican a los pesos globales
        for p_global, p_local in zip(self.modelo_global.parameters(), self.modelo.parameters()):
            p_global.grad = p_local.grad
        with self.cerrojo:
            self.optimizador.step()

    def _episodio(self, ep):
        c = self.c
        beta = decaimiento_exponencial(ep, c["n_episodios"], c["beta_inicial"], c["beta_final"], c["frac_decaimiento"])
        np.random.seed(semilla_entrenamiento(c["seed"], ep))
        env = SimpleTradingEnv(self.generador(n_steps=c["n_steps"]))
        obs, _ = env.reset()
        terminado = False
        ganancia = 0.0

        while not terminado:
            self.modelo.load_state_dict(self.modelo_global.state_dict())
            estados, acciones, recompensas = [], [], []
            while not terminado and len(estados) < c["n_pasos"]:
                x = self.normalizador.entrada(obs[0], obs[1])
                with torch.inference_mode():
                    probs = torch.softmax(self.modelo.actor(torch.from_numpy(x)), dim=-1).numpy()
                a = int(np.searchsorted(np.cumsum(probs), self.rng.random() * probs.sum()))
                a = min(a, len(probs) - 1)
                obs, r, terminado, _, _ = env.step(a)
                ganancia += r
                estados.append(x)
                acciones.append(a)
                recompensas.append(r / self.normalizador.escala_recompensa)
            self._actualizar(estados, acciones, recompensas, obs, terminado, beta)

        self.ganancias[ep] = ganancia
        self.retornos_pct[ep] = 100 * (env.balance / env.initial_balance - 1)

    def ejecutar(self):
        """Repite episodios hasta agotar el total. El contador compartido reparte una serie distinta a cada uno."""
        while True:
            with self.contador.get_lock():
                ep = self.contador.value
                if ep >= self.c["n_episodios"]:
                    return
                self.contador.value += 1
            self._episodio(ep)


def _lanzar_trabajador(barrera, *args):
    """Lo que ejecuta cada proceso (o el hilo, si solo hay un trabajador)."""
    fijar_hilos()
    trabajador = Trabajador(*args)
    barrera.wait()
    trabajador.ejecutar()


def entrenar_a3c(generador, normalizador, n_episodios=2000, n_steps=N_STEPS, alpha=3e-4, alpha_critico=1e-3,
                 gamma=0.9, beta_inicial=0.05, beta_final=0.001, frac_decaimiento=0.7, n_pasos=20, n_trabajadores=4,
                 usar_critico=True, perdida_critico="mse", ocultas=(64, 64), seed=0, precios_validacion=None,
                 evaluar_cada=100, max_norma_grad=5.0, algoritmo="a3c"):
    """Entrena un agente A3C (o REINFORCE) y devuelve un Resultado.

    Parámetros principales:
    - alpha y alpha_critico: tamaños de paso α_θ del actor y α_w del crítico.
    - gamma: factor de descuento.
    - beta_inicial, beta_final y frac_decaimiento: decaimiento del coeficiente de
      entropía β, que hace en A3C el papel de la ε de DQN.
    - perdida_critico: "mse" o "mae".
    - n_pasos: cada cuántos pasos actualiza un trabajador.
    - n_trabajadores: procesos en paralelo. n_episodios es el total entre todos.
    - precios_validacion y evaluar_cada: controles periódicos.
    """
    fijar_hilos()
    torch.manual_seed(seed)
    c = dict(seed=seed, alpha=alpha, alpha_critico=alpha_critico, gamma=gamma, ocultas=tuple(ocultas),
             n_episodios=n_episodios, n_steps=n_steps, beta_inicial=beta_inicial, beta_final=beta_final,
             frac_decaimiento=frac_decaimiento, n_pasos=n_pasos, usar_critico=usar_critico,
             perdida_critico=perdida_critico, max_norma_grad=max_norma_grad)
    modelo = crear_red(TIPO_RED, ocultas)
    modelo.share_memory()
    estado_adam = _estado_adam_compartido(modelo)
    ctx = mp.get_context("spawn")
    contador = ctx.Value("i", 0)
    barrera = ctx.Barrier(n_trabajadores + 1)
    # Cada trabajador apunta aquí la ganancia de los episodios que le tocan
    ganancias = ctx.RawArray("d", n_episodios)
    retornos_pct = ctx.RawArray("d", n_episodios)
    args = (modelo, estado_adam, ctx.Lock(), contador, ganancias, retornos_pct, generador, normalizador, c)
    # Con un trabajador no hace falta otro proceso: un hilo evita el coste de arrancarlo
    trabajadores = [threading.Thread(target=_lanzar_trabajador, args=(barrera, 0, *args))] if n_trabajadores == 1 else \
        [ctx.Process(target=_lanzar_trabajador, args=(barrera, i, *args)) for i in range(n_trabajadores)]
    for t in trabajadores:
        t.start()
    barrera.wait()  # el reloj empieza cuando todos los trabajadores están listos
    inicio = time.perf_counter()

    # Mientras los trabajadores entrenan, se guardan copias de los pesos en cada control.
    # Se evalúan al terminar, para que evaluar no quite tiempo de CPU al entrenamiento.
    instantaneas = []
    siguiente = evaluar_cada
    while any(t.is_alive() for t in trabajadores):
        if contador.value >= siguiente and siguiente < n_episodios:
            instantaneas.append((siguiente, time.perf_counter() - inicio, clonar_estado(modelo)))
            siguiente += evaluar_cada
        time.sleep(0.01)
    segundos = time.perf_counter() - inicio
    for t in trabajadores:
        t.join()
        if getattr(t, "exitcode", 0):
            raise RuntimeError(f"un trabajador de A3C terminó con código {t.exitcode}")
    final = clonar_estado(modelo)
    instantaneas.append((n_episodios, segundos, final))

    mejor = MejorControl()
    filas = []
    if precios_validacion is not None:
        for ep, seg, estado in instantaneas:
            agente = AgenteActorCritico(crear_red(TIPO_RED, ocultas, estado), normalizador)
            control = control_validacion(agente, precios_validacion)
            mejor.actualizar(control, estado)
            filas.append({"episodio": ep, "pasos": ep * (n_steps - 6), "segundos": seg,
                          "beta": decaimiento_exponencial(ep - 1, n_episodios, beta_inicial, beta_final,
                                                          frac_decaimiento), **control})
    historial = pd.DataFrame({"episodio": np.arange(1, n_episodios + 1), "ganancia": np.array(ganancias),
                              "retorno_pct": np.array(retornos_pct)})
    return Resultado(algoritmo, TIPO_RED, tuple(ocultas), mejor.estado or final, final,
                     pd.DataFrame(filas), historial, segundos, n_episodios * (n_steps - 6))


def entrenar_reinforce(generador, normalizador, **params):
    """REINFORCE: un solo trabajador que actualiza con G_t al final de cada episodio, sin crítico."""
    return entrenar_a3c(generador, normalizador, n_pasos=N_PASOS_EPISODIO, n_trabajadores=1, usar_critico=False,
                        algoritmo="reinforce", **params)
