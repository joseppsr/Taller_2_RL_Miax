"""Compras y ventas de modelos guardados: solo evaluación, sin entrenar."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from rl_trader.agentes import crear_agente
from rl_trader.env import SimpleTradingEnv
from rl_trader.evaluation import SEMILLA_TEST, generar_series
from rl_trader.prices import GENERADORES
from rl_trader.redes import Normalizador, fijar_hilos

RAIZ = Path(__file__).resolve().parent.parent
MERCADOS = {"ciclo": "Ciclo", "browniano": "Browniano",
            "reversion": "Reversión a la media", "regimenes": "Alternancia de regímenes"}


def graficar_operaciones(semilla_modelo=100, serie=0):
    """Misma serie de test por mercado para DQN y A3C; devuelve figura y registros."""
    if not 0 <= serie < 100:
        raise ValueError("Elige una serie de test entre 0 y 99.")
    fijar_hilos()
    modelos = torch.load(RAIZ / "results/modelos.pt", map_location="cpu", weights_only=True)
    fig, ejes = plt.subplots(4, 2, figsize=(15, 12), sharex=True, sharey="row")
    registros = {}
    # Los generadores usan NumPy global: restaurar su estado al terminar.
    estado_azar = np.random.get_state()
    try:
        for fila, (mercado, titulo) in enumerate(MERCADOS.items()):
            generador = GENERADORES[mercado]
            precios = generar_series(generador, 1, SEMILLA_TEST + serie)[0]
            normalizador = Normalizador.ajustar(generador)
            for columna, algoritmo in enumerate(("dqn", "a3c")):
                agente = crear_agente(**modelos[f"{mercado}|{algoritmo}|{semilla_modelo}"],
                                      normalizador=normalizador)
                env = SimpleTradingEnv(precios)
                obs, _ = env.reset()
                compras, ventas, dentro = [], [], np.zeros(len(precios), dtype=bool)
                terminado = False
                while not terminado:
                    paso, posicion = env.current_step, env.position
                    obs, _, terminado, _, _ = env.step(agente(obs))
                    # El entorno ejecuta al precio ANTERIOR al avance del reloj.
                    if env.position != posicion:
                        (compras if env.position else ventas).append(paso)
                    dentro[paso] = bool(env.position)
                dentro[-1] = bool(env.position)
                retorno = 100 * (env.balance / env.initial_balance - 1)
                ax = ejes[fila, columna]
                ax.plot(precios, color="#526579", lw=1.3, label="Precio")
                ax.fill_between(np.arange(len(precios)), 0, 1, where=dentro, step="post",
                                transform=ax.get_xaxis_transform(), color="#2a78d6",
                                alpha=0.10, label="Con el activo")
                ax.scatter(compras, precios[compras], marker="^", color="#16834b", s=55,
                           edgecolors="white", linewidths=0.5, zorder=3, label="Compra")
                ax.scatter(ventas, precios[ventas], marker="v", color="#d13d3d", s=55,
                           edgecolors="white", linewidths=0.5, zorder=3, label="Venta")
                ax.set_title(f"{titulo} · {algoritmo.upper()} · Retorno: {retorno:+.1f}%\n"
                             f"{len(compras)} compras / {len(ventas)} ventas")
                ax.grid(alpha=0.2)
                if mercado == "ciclo":
                    ax.set_yscale("log")  # Permite ver las operaciones al inicio y al final.
                if columna == 0:
                    ax.set_ylabel("Precio (escala log)" if mercado == "ciclo" else "Precio")
                if fila == 3:
                    ax.set_xlabel("Paso de la serie")
                if not compras and not ventas:
                    ax.text(0.5, 0.85, "Sin operaciones", transform=ax.transAxes, ha="center")
                registros[(mercado, algoritmo)] = dict(precios=precios, compras=compras,
                    ventas=ventas, retorno=retorno, posicion_final=env.position)
    finally:
        np.random.set_state(estado_azar)
    fig.suptitle(f"Compras y ventas · Mejor control en validación · Semilla {semilla_modelo} · "
                 f"Serie de test {serie}", fontsize=14)
    fig.legend(*ejes[0, 0].get_legend_handles_labels(), loc="upper center",
               bbox_to_anchor=(0.5, 0.96), ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    destino = RAIZ / "figures"
    destino.mkdir(exist_ok=True)
    fig.savefig(destino / "compras_ventas_cuatro_mercados.png", dpi=160, bbox_inches="tight")
    return fig, registros
