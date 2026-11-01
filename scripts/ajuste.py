"""Búsqueda de hiperparámetros por etapas, mirando solo las series de validación.

Cada etapa parte de lo mejor de la anterior, por mercado y familia de algoritmos:

- A: factor de aprendizaje α × factor de descuento γ.
- B: exploración: calendario de ε en DQN; coeficiente de entropía β y n pasos en A3C.
- C: función de pérdida, MSE frente a MAE (en DQN, la pérdida TD; en A3C, la del crítico).
- D: número de episodios.

Cada configuración se entrena con tres semillas. Su nota es la mediana del retorno en
100 series de validación, promediada entre semillas; si hay empate, gana la que opera
menos. Las series de test no se usan.

Doble DQN y DQN en duelo usan la configuración de DQN: son la misma familia y así se mide
solo el efecto de cada variante.

Para lanzarlo, desde la raíz del repositorio:  python scripts/ajuste.py

Cada entrenamiento terminado se apunta en results/, así que si se interrumpe y se vuelve
a lanzar continúa donde se quedó. Con 16 núcleos tarda unas 2 horas.
"""
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from rl_trader.evaluation import episodio_convergencia  # noqa: E402
from rl_trader.experimentos import ejecutar  # noqa: E402
from rl_trader.prices import GENERADORES  # noqa: E402

RESULTADOS = RAIZ / "results"
SEMILLAS = (0, 1, 2)
MERCADOS = list(GENERADORES)
GAMMAS = [0.0, 0.5, 0.9, 0.99]
# Procesos a la vez: A3C ya usa 4 por entrenamiento
PROCESOS = {"dqn": 15, "a3c": 4, "reinforce": 15}

BASE = {
    "dqn": dict(algoritmo="dqn", n_episodios=400, alpha=5e-4, gamma=0.9, eps_inicial=1.0, eps_final=0.01,
                frac_decaimiento=0.7, perdida="mse", evaluar_cada=50),
    "a3c": dict(algoritmo="a3c", n_episodios=800, alpha=3e-4, alpha_critico=1e-3, gamma=0.9, beta_inicial=0.05,
                beta_final=0.001, frac_decaimiento=0.7, n_pasos=20, n_trabajadores=4, perdida_critico="mse",
                evaluar_cada=50),
    "reinforce": dict(algoritmo="reinforce", n_episodios=800, alpha=1e-3, gamma=0.9, beta_inicial=0.05,
                      beta_final=0.001, frac_decaimiento=0.7, evaluar_cada=50),
}
ETAPA_A = {
    "dqn": [dict(alpha=a, gamma=g) for a in (1e-4, 5e-4, 2e-3) for g in GAMMAS],
    # α_w se mueve con α_θ, para que el crítico aprenda siempre algo más deprisa que el actor
    "a3c": [dict(alpha=a, alpha_critico=3 * a, gamma=g) for a in (1e-4, 3e-4, 1e-3) for g in GAMMAS],
    "reinforce": [dict(alpha=a, gamma=g) for a in (3e-4, 1e-3) for g in (0.5, 0.9, 0.99)],
}
ETAPA_B = {
    "dqn": [dict(frac_decaimiento=f, eps_final=e) for f, e in ((0.3, 0.01), (0.7, 0.01), (0.7, 0.1))],
    "a3c": [dict(beta_inicial=b, n_pasos=n) for b in (0.01, 0.05, 0.2) for n in (5, 20)],
    "reinforce": [dict(beta_inicial=b) for b in (0.01, 0.05, 0.2)],
}
ETAPA_C = {
    "dqn": [dict(perdida=p) for p in ("mse", "mae")],
    "a3c": [dict(perdida_critico=p) for p in ("mse", "mae")],
    "reinforce": [],  # REINFORCE no tiene crítico ni target: la pérdida no es de regresión
}
ETAPA_D = {
    "dqn": [dict(n_episodios=n) for n in (400, 800, 1600)],
    "a3c": [dict(n_episodios=n) for n in (800, 1600, 3200)],
    "reinforce": [dict(n_episodios=n) for n in (800, 1600, 3200)],
}
# Una configuración mejor con más episodios solo se prefiere si gana por más de esto
MARGEN_REL, MARGEN_ABS = 0.05, 1.0


def _clave(config):
    return json.dumps(config, sort_keys=True)


def _fila(config):
    """Entrena una configuración y resume el resultado. Se ejecuta en un proceso aparte."""
    s = ejecutar(config)
    res = s["resultado"]
    return config, {
        **s["resumen"],
        "segundos": res.segundos,
        "episodio_convergencia": episodio_convergencia(res.validacion),
    }, res.validacion


def correr(familia, etapa, variantes):
    """Entrena todas las variantes de una etapa y devuelve una fila por entrenamiento."""
    ruta = RESULTADOS / f"ajuste_{familia}_{etapa}.csv"
    ruta_curvas = RESULTADOS / f"ajuste_{familia}_{etapa}_curvas.csv"
    hechas = pd.read_csv(ruta) if ruta.exists() else pd.DataFrame()
    claves = set(hechas["clave"]) if len(hechas) else set()
    configs = [v | {"seed": s} for v in variantes for s in SEMILLAS]
    pendientes = [c for c in configs if _clave(c) not in claves]
    print(f"[{time.strftime('%H:%M:%S')}] {familia}/{etapa}: {len(pendientes)} de {len(configs)} entrenamientos",
          flush=True)
    if pendientes:
        with ProcessPoolExecutor(max_workers=PROCESOS[familia]) as pool:
            for futuro in as_completed([pool.submit(_fila, c) for c in pendientes]):
                config, resumen, curva = futuro.result()
                clave = _clave(config)
                pd.DataFrame([{**config, **resumen, "clave": clave}]).to_csv(
                    ruta, mode="a", index=False, header=not ruta.exists())
                curva.assign(clave=clave).to_csv(ruta_curvas, mode="a", index=False, header=not ruta_curvas.exists())
    tabla = pd.read_csv(ruta)
    return tabla[tabla["clave"].isin({_clave(c) for c in configs})]


def elegir(filas, columnas):
    """Elige la mejor combinación de valores de columnas en cada mercado: mayor mediana y, si empata, menos operaciones."""
    medias = filas.groupby(["mercado", *columnas], as_index=False)[["val_mediana", "val_operaciones"]].mean()
    orden = medias.sort_values(["val_mediana", "val_operaciones"], ascending=[False, True], kind="mergesort")
    mejores = orden.groupby("mercado").head(1)
    return {f.mercado: {c: getattr(f, c) for c in columnas} for f in mejores.itertuples()}


def elegir_episodios(filas):
    """Elige el menor número de episodios cuya nota queda cerca de la mejor: entrenar más cuesta tiempo."""
    medias = filas.groupby(["mercado", "n_episodios"], as_index=False)["val_mediana"].mean()
    salida = {}
    for mercado, grupo in medias.groupby("mercado"):
        mejor = grupo["val_mediana"].max()
        margen = max(MARGEN_REL * abs(mejor), MARGEN_ABS)
        salida[mercado] = {"n_episodios": int(grupo[grupo["val_mediana"] >= mejor - margen]["n_episodios"].min())}
    return salida


def ajustar(familia):
    """Hace las cuatro etapas de una familia y devuelve la configuración elegida en cada mercado."""
    elegido = {m: dict(BASE[familia], mercado=m) for m in MERCADOS}

    def etapa(nombre, variantes, eleccion=None):
        if not variantes:
            return
        filas = correr(familia, nombre, [elegido[m] | v for m in MERCADOS for v in variantes])
        ganadores = eleccion(filas) if eleccion else elegir(filas, sorted({k for v in variantes for k in v}))
        for mercado, valores in ganadores.items():
            elegido[mercado].update(valores)

    etapa("a", ETAPA_A[familia])
    etapa("b", ETAPA_B[familia])
    etapa("c", ETAPA_C[familia])
    etapa("d", ETAPA_D[familia], elegir_episodios)
    return elegido


def main():
    RESULTADOS.mkdir(exist_ok=True)
    mejores = []
    for familia in ("dqn", "reinforce", "a3c"):
        for mercado, config in ajustar(familia).items():
            mejores.append({"familia": familia, "mercado": mercado, "config": json.dumps(config, sort_keys=True)})
        pd.DataFrame(mejores).to_csv(RESULTADOS / "mejores.csv", index=False)
    print(f"[{time.strftime('%H:%M:%S')}] ajuste terminado", flush=True)


if __name__ == "__main__":
    main()
