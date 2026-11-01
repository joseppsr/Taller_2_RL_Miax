"""Figuras comunes del notebook y del informe, todas con el mismo estilo.

Cada agente conserva siempre su color, y las referencias van en gris: el color indica
qué algoritmo es, no su posición en el ranking.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from .analisis import AGENTES_RL, BASELINES, MERCADOS, NOMBRES_AGENTE, NOMBRES_MERCADO, RESULTADOS

TINTA, TINTA_2, GRIS, REJILLA, EJE, FONDO = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
COLORES = {"dqn": "#2a78d6", "doble_dqn": "#eb6834", "duelo": "#1baf7a", "a3c": "#eda100", "reinforce": "#e87ba4"}
GRISES = {"aleatorio": "#b5b4ad", "buy_and_hold": "#52514e", "momentum": "#898781", "contrarian": "#c3c2b7"}
SECUENCIAL = LinearSegmentedColormap.from_list(
    "azules", ["#f4f8fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
FAMILIAS = (("Basados en valor (DQN)", ["dqn", "doble_dqn", "duelo"]), ("Basados en política", ["a3c", "reinforce"]))
FIGURAS = RESULTADOS.parent / "figures"


def aplicar_estilo():
    """Aplica el estilo común a matplotlib."""
    plt.rcParams.update({
        "figure.facecolor": FONDO, "axes.facecolor": FONDO, "savefig.facecolor": FONDO, "savefig.dpi": 160,
        "axes.edgecolor": EJE, "axes.labelcolor": TINTA_2, "axes.titlecolor": TINTA, "text.color": TINTA,
        "xtick.color": GRIS, "ytick.color": GRIS, "axes.grid": True, "grid.color": REJILLA, "grid.linewidth": 0.8,
        "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2,
        "font.size": 9, "axes.titlesize": 10, "legend.frameon": False,
    })


def _guardar(fig, nombre):
    if nombre:
        FIGURAS.mkdir(exist_ok=True)
        fig.savefig(FIGURAS / f"{nombre}.png", bbox_inches="tight")


def _curva_agente(ax, datos, columna, agente, ventana):
    """Dibuja la media entre semillas de una curva, suavizada, con la banda entre los cuartiles de las semillas."""
    tabla = datos.pivot_table(index="episodio", columns="seed", values=columna)
    suave = tabla.rolling(ventana, min_periods=1).mean()
    x = suave.index.to_numpy()
    ax.fill_between(x, suave.quantile(0.25, axis=1), suave.quantile(0.75, axis=1), color=COLORES[agente], alpha=0.15,
                    linewidth=0)
    ax.plot(x, suave.mean(axis=1), color=COLORES[agente], label=NOMBRES_AGENTE[agente])


def _referencias(ax, referencias, mercado):
    """Líneas de las referencias en gris, con su nombre escrito junto a la línea."""
    if referencias is None:
        return
    fila = referencias[referencias.mercado == mercado].set_index("agente")["ganancia"]
    x = ax.get_xlim()[1]
    for agente, estilo in (("buy_and_hold", "--"), ("aleatorio", ":")):
        ax.axhline(fila[agente], color=GRISES[agente], linestyle=estilo, linewidth=1.2)
        ax.annotate(NOMBRES_AGENTE[agente], (x, fila[agente]), xytext=(-3, 3), textcoords="offset points", ha="right",
                    fontsize=7.5, color=TINTA_2)


def curvas_por_episodio(datos, columna, titulo, ylabel, referencias=None, frac_ventana=0.04, nombre=None):
    """Curvas por episodio: una fila por mercado y una columna por familia de algoritmos.

    Cada fila comparte el eje y; las familias tienen eje x propio porque usan distinto
    número de episodios.
    """
    fig, ejes = plt.subplots(len(MERCADOS), 2, figsize=(11, 10), sharey="row", squeeze=False)
    for i, mercado in enumerate(MERCADOS):
        for j, (familia, agentes) in enumerate(FAMILIAS):
            ax = ejes[i, j]
            for agente in agentes:
                d = datos[(datos.mercado == mercado) & (datos.agente == agente)]
                # La ventana depende de los puntos de cada curva: una por episodio o una por control
                ventana = max(1, int(frac_ventana * d["episodio"].nunique()))
                _curva_agente(ax, d, columna, agente, ventana)
            ax.margins(x=0)
            _referencias(ax, referencias, mercado)
            if i == 0:
                ax.set_title(familia, fontweight="bold")
                ax.legend(loc="lower right", ncols=len(agentes))
            if j == 0:
                ax.set_ylabel(f"{NOMBRES_MERCADO[mercado]}\n{ylabel}")
            if i == len(MERCADOS) - 1:
                ax.set_xlabel("Episodio de entrenamiento")
    fig.suptitle(titulo, fontsize=12, fontweight="bold")
    fig.tight_layout()
    _guardar(fig, nombre)
    return fig


def graficar_ganancia_episodios(nombre="ganancia_por_episodio"):
    """La ganancia G = Σ r de cada episodio de entrenamiento, mientras el agente aprende y aún explora."""
    datos = pd.read_csv(RESULTADOS / "ganancia_episodios.csv.gz")
    referencias = pd.read_csv(RESULTADOS / "baselines_validacion.csv")
    return curvas_por_episodio(datos, "ganancia", "Ganancia G = Σ r por episodio de entrenamiento "
                               "(media de 10 semillas, suavizada; banda: cuartiles)", "Ganancia G",
                               referencias, nombre=nombre)


def graficar_ganancia_validacion(nombre="ganancia_validacion"):
    """La ganancia de la política greedy, sin explorar, en las mismas series de validación."""
    datos = pd.read_csv(RESULTADOS / "curvas_validacion.csv")
    referencias = pd.read_csv(RESULTADOS / "baselines_validacion.csv")
    return curvas_por_episodio(datos, "ganancia", "Ganancia media de la política greedy en validación "
                               "(media de 10 semillas; banda: cuartiles)", "Ganancia G", referencias,
                               frac_ventana=0.0, nombre=nombre)


def graficar_capital(test, nombre="capital_final"):
    """Capital final en test: mediana (punto) y percentiles 25-75 (línea), en escala logarítmica."""
    agentes = BASELINES + AGENTES_RL
    fig, ejes = plt.subplots(1, len(MERCADOS), figsize=(12, 3.8), sharey=True)
    for ax, mercado in zip(ejes, MERCADOS):
        for y, agente in enumerate(agentes[::-1]):
            factor = 1 + test[(test.mercado == mercado) & (test.agente == agente)]["retorno_pct"] / 100
            q1, q2, q3 = factor.quantile([0.25, 0.5, 0.75])
            color = COLORES.get(agente, GRISES.get(agente))
            ax.plot([q1, q3], [y, y], color=color, linewidth=2)
            ax.plot(q2, y, "o", color=color, markersize=8, markeredgecolor=FONDO, markeredgewidth=1.5)
        ax.axvline(1, color=GRIS, linewidth=1, linestyle=":")
        ax.set_xscale("log")
        ax.set_title(NOMBRES_MERCADO[mercado], fontweight="bold")
        ax.set_xlabel("Capital final / inicial")
        ax.set_yticks(range(len(agentes)), [NOMBRES_AGENTE[a] for a in agentes[::-1]])
        ax.grid(axis="y", visible=False)
    fig.tight_layout()
    _guardar(fig, nombre)
    return fig


def _dentro(preferencias):
    """Traduce la acción preferida a "dentro" (1) o "fuera" (0): sin activo, comprar; con activo, no vender."""
    a = preferencias.argmax(axis=-1)
    return np.stack([a[0] == 2, a[1] != 0]).astype(float)


def graficar_politicas(nombre="politicas"):
    """Fracción de las 10 semillas que quieren estar dentro, según la tendencia continua y la posición.

    La frontera es continua: no hay intervalos. La tendencia está en desviaciones típicas
    del mercado.
    """
    mapas = np.load(RESULTADOS / "mapas_politica.npz")
    z = np.linspace(-4, 4, 81)
    fig, ejes = plt.subplots(len(MERCADOS), len(AGENTES_RL), figsize=(12, 5.6), sharex=True, sharey=True)
    for i, mercado in enumerate(MERCADOS):
        for j, agente in enumerate(AGENTES_RL):
            claves = [k for k in mapas.files if k.startswith(f"{mercado}|{agente}|")]
            media = np.mean([_dentro(mapas[k]) for k in claves], axis=0)
            imagen = ejes[i, j].imshow(media, aspect="auto", cmap=SECUENCIAL, vmin=0, vmax=1,
                                       extent=(z[0], z[-1], -0.5, 1.5), origin="lower", interpolation="nearest")
            ejes[i, j].axvline(0, color=TINTA_2, linewidth=0.8)
            ejes[i, j].grid(False)
            ejes[i, j].set_yticks([0, 1], ["Sin activo", "Con activo"])
            if i == 0:
                ejes[i, j].set_title(NOMBRES_AGENTE[agente], fontweight="bold")
            if j == 0:
                ejes[i, j].set_ylabel(NOMBRES_MERCADO[mercado].replace(" a la media", "").replace(" + ciclo", ""))
            if i == len(MERCADOS) - 1:
                ejes[i, j].set_xlabel("Tendencia (σ)")
    fig.tight_layout()
    barra = fig.colorbar(imagen, ax=ejes, fraction=0.015, pad=0.015)
    barra.set_label("Fracción de semillas dentro")
    _guardar(fig, nombre)
    return fig


def graficar_ajuste(familia, nombre=None):
    """Mapa de calor de la etapa A del ajuste: nota en validación según α y γ, para cada mercado."""
    d = pd.read_csv(RESULTADOS / f"ajuste_{familia}_a.csv")
    fig, ejes = plt.subplots(1, len(MERCADOS), figsize=(12, 2.6))
    for ax, mercado in zip(ejes, MERCADOS):
        tabla = d[d.mercado == mercado].pivot_table(index="alpha", columns="gamma", values="val_mediana")
        v = tabla.to_numpy()
        ax.imshow(v, cmap=SECUENCIAL, aspect="auto")
        ax.set_xticks(range(v.shape[1]), [f"{g:g}" for g in tabla.columns])
        ax.set_yticks(range(v.shape[0]), [f"{a:g}" for a in tabla.index])
        ax.grid(False)
        umbral = (np.nanmin(v) + np.nanmax(v)) / 2
        for (fila, col), valor in np.ndenumerate(v):
            texto = f"{valor / 1000:.0f}K" if abs(valor) >= 1e4 else f"{valor:.1f}"
            ax.text(col, fila, texto, ha="center", va="center", fontsize=8,
                    color="white" if valor > umbral else TINTA)
        ax.set_title(NOMBRES_MERCADO[mercado], fontweight="bold")
        ax.set_xlabel("γ")
    ejes[0].set_ylabel("α")
    fig.tight_layout()
    _guardar(fig, nombre)
    return fig


def graficar_perdidas(tabla, nombre="mse_vs_mae"):
    """MSE frente a MAE: ganancia en validación de cada pérdida, por mercado y familia (etapa C del ajuste)."""
    fig, ejes = plt.subplots(1, 4, figsize=(12, 2.8))
    colores = {"mse": "#2a78d6", "mae": "#eb6834"}
    for ax, mercado in zip(ejes, MERCADOS):
        d = tabla[tabla.mercado == mercado]
        for k, perdida in enumerate(("mse", "mae")):
            filas = d[d.perdida == perdida].set_index("familia").reindex(["dqn", "a3c"])
            ax.bar(np.arange(2) + (k - 0.5) * 0.36, filas["val_ganancia"], 0.34, color=colores[perdida],
                   label=perdida.upper())
        ax.axhline(0, color=GRIS, linewidth=0.8)
        ax.set_xticks([0, 1], ["DQN", "A3C (crítico)"])
        ax.grid(axis="x", visible=False)
        ax.set_title(NOMBRES_MERCADO[mercado], fontweight="bold")
    ejes[0].set_ylabel("Ganancia G en validación")
    ejes[-1].legend()
    fig.tight_layout()
    _guardar(fig, nombre)
    return fig
