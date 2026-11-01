"""Comparación: DQN con minibatches del buffer frente a DQN viendo el episodio entero.

La base sigue siendo el minibatch, el algoritmo de las slides. Aquí se entrena lo mismo, con
las mismas semillas y los mismos hiperparámetros, pero con modo="episodio": al terminar cada
serie la red ve de una vez sus 494 transiciones, sin buffer. Se prueban dos intensidades:

- 1 pasada: un solo paso de gradiente por episodio.
- 16 pasadas: 16 pasos por episodio. Usan tantas experiencias como el minibatch
  (123 actualizaciones de 64 ≈ 16 pasadas de 494), así que la comparación es justa.

No elige nada: la configuración final no cambia sea cual sea el resultado.

Genera results/lotes_test.csv.gz. Para lanzarlo, después de resultados.py:
    python scripts/comparar_lotes.py
Si el resultado ya existe no se repite. Con 16 núcleos tarda unos 10 minutos.
"""
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from rl_trader.experimentos import _ejecutar_en_test  # noqa: E402

RESULTADOS = RAIZ / "results"
SEMILLAS = range(100, 110)
PASADAS = (1, 16)


def main():
    ruta = RESULTADOS / "lotes_test.csv.gz"
    if ruta.exists():
        print("comparación de lotes: ya calculada")
        return
    mejores = pd.read_csv(RESULTADOS / "mejores.csv").query("familia == 'dqn'")
    configs = [json.loads(f.config) | {"modo": "episodio", "pasadas_episodio": p, "seed": s}
               for f in mejores.itertuples() for p in PASADAS for s in SEMILLAS]
    tablas = []
    with ProcessPoolExecutor(max_workers=15) as pool:
        for futuro in as_completed([pool.submit(_ejecutar_en_test, c) for c in configs]):
            s = futuro.result()
            c = s["config"]
            tablas.append(s["test"].assign(mercado=c["mercado"], pasadas=c["pasadas_episodio"], seed=c["seed"],
                                           segundos=s["resultado"].segundos))
    pd.concat(tablas).to_csv(ruta, index=False)


if __name__ == "__main__":
    main()
