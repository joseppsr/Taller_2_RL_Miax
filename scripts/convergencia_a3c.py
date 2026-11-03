"""Diagnóstico de convergencia: A3C con 4 veces más episodios de los que permite el ajuste.

Con los hiperparámetros elegidos, entrena 12.800 episodios (3 semillas por mercado) y guarda
la curva de validación. Sirve para distinguir dos situaciones: que A3C necesite más
experiencia o que oscile sin estabilizarse aunque se le dé mucha más. No elige nada: la
configuración final no cambia.

Genera results/convergencia_a3c.csv. Para lanzarlo, después de ajuste.py:
    python scripts/convergencia_a3c.py
Si el resultado ya existe no se repite. Con 16 núcleos tarda alrededor de media hora.
"""
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from rl_trader.experimentos import ejecutar  # noqa: E402

RESULTADOS = RAIZ / "results"
SEMILLAS = (400 - 3, 400 - 2, 400 - 1)
N_EPISODIOS = 12_800


def main():
    ruta = RESULTADOS / "convergencia_a3c.csv"
    if ruta.exists():
        print("convergencia de A3C: ya calculada")
        return
    mejores = pd.read_csv(RESULTADOS / "mejores.csv").query("familia == 'a3c'")
    configs = [json.loads(f.config) | {"n_episodios": N_EPISODIOS, "evaluar_cada": 200, "seed": s}
               for f in mejores.itertuples() for s in SEMILLAS]
    curvas = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        for futuro in as_completed([pool.submit(ejecutar, c) for c in configs]):
            s = futuro.result()
            c = s["config"]
            curvas.append(s["resultado"].validacion.assign(mercado=c["mercado"], agente="a3c", seed=c["seed"]))
    pd.concat(curvas).to_csv(ruta, index=False)


if __name__ == "__main__":
    main()
