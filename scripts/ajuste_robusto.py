"""Reevaluar pocas configuraciones del ajuste original con semillas nuevas.

No modifica mejores.csv ni consulta test. Las celdas del notebook preparan la
lista, repiten los entrenamientos y muestran propuestas con su incertidumbre.
"""
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
from rl_trader.a3c import entrenar_a3c
from rl_trader.dqn import entrenar_dqn
from rl_trader.evaluation import semilla_entrenamiento
from rl_trader.experimentos import ejecutar


def _configuracion(config):
    """Completar valores por defecto para no contar dos veces la misma configuración."""
    funcion = {"dqn": entrenar_dqn, "a3c": entrenar_a3c}[config["algoritmo"]]
    omitidos = {"generador", "normalizador", "precios_validacion", "seed", "doble", "duelo"}
    normalizada = {k: v.default for k, v in inspect.signature(funcion).parameters.items()
                   if k not in omitidos and v.default is not inspect.Parameter.empty}
    normalizada.update({k: v for k, v in config.items() if k not in omitidos})
    normalizada.setdefault("n_validacion", 100)
    return json.dumps(normalizada, sort_keys=True)


def seleccionar_candidatas(mercados=("reversion",), familias=("dqn", "a3c"), cantidad=3,
                          resultados=None):
    """Incluir la actual y las mejores alternativas completas de las etapas A–D."""
    if cantidad < 2:
        raise ValueError("Selecciona al menos la configuración actual y una alternativa.")
    resultados = Path(resultados or RAIZ / "results")
    elegidas = pd.read_csv(resultados / "mejores.csv")
    candidatas = []
    for familia in familias:
        registros = []
        for etapa in "abcd":
            ruta = resultados / f"ajuste_{familia}_{etapa}.csv"
            if not ruta.exists():
                continue
            for fila in pd.read_csv(ruta).itertuples():
                config = json.loads(fila.clave)
                if config["mercado"] in mercados:
                    registros.append(dict(config=_configuracion(config), mercado=config["mercado"],
                                          seed=int(config["seed"]), nota=fila.val_mediana,
                                          operaciones=fila.val_operaciones))
        # Las etapas pueden repetir la misma configuración/semilla: usar una observación.
        datos = pd.DataFrame(registros).drop_duplicates(["config", "seed"], keep="last")
        for mercado in mercados:
            grupo = datos[datos.mercado == mercado]
            ranking = grupo.groupby("config").agg(media_previa=("nota", "mean"),
                desviacion_previa=("nota", "std"), operaciones=("operaciones", "mean"),
                n_previo=("seed", "nunique"))
            ranking = ranking[ranking.n_previo >= 3].sort_values(
                ["media_previa", "operaciones"], ascending=[False, True])
            actual = _configuracion(json.loads(elegidas.query(
                "familia == @familia and mercado == @mercado").iloc[0].config))
            if actual not in ranking.index:
                raise ValueError(f"La configuración actual de {familia}/{mercado} no tiene tres semillas registradas.")
            claves = [actual] + [c for c in ranking.index if c != actual][:cantidad-1]
            for clave in claves:
                config = json.loads(clave)
                candidatas.append(dict(familia=familia, mercado=mercado,
                    candidato=hashlib.sha256(clave.encode()).hexdigest()[:10], actual=clave == actual,
                    media_previa=ranking.loc[clave, "media_previa"],
                    error_previo=ranking.loc[clave, "desviacion_previa"] / np.sqrt(ranking.loc[clave, "n_previo"]),
                    episodios=config["n_episodios"], alpha=config["alpha"], gamma=config["gamma"],
                    perdida=config.get("perdida", config.get("perdida_critico")),
                    exploracion_inicial=config.get("eps_inicial", config.get("beta_inicial")),
                    n_pasos_a3c=config.get("n_pasos", 0),
                    semillas_previas=json.dumps(sorted(grupo.seed.unique().tolist())), config=clave))
    plan = pd.DataFrame(candidatas)
    print(f"Seleccionadas {len(plan)} configuraciones. Se conserva cada presupuesto original de episodios.")
    return plan


def _entrenar(ruta):
    trabajo = json.loads(Path(ruta).read_text(encoding="utf-8"))
    salida = ejecutar(trabajo["config"], en_test=False)
    destino = Path(ruta).parent
    salida["resultado"].validacion.to_csv(destino / "validacion.csv", index=False)
    resumen = salida["resumen"] | {"segundos": salida["resultado"].segundos}
    # El resumen se guarda al final y actúa como indicador de ensayo completo.
    (destino / "resumen.json").write_text(json.dumps(resumen), encoding="utf-8")


def _proceso(ruta):
    with ruta.with_name("ejecucion.log").open("w", encoding="utf-8") as log:
        proceso = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), str(ruta)],
            cwd=RAIZ, stdout=log, stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        try:
            codigo = proceso.wait()
        except KeyboardInterrupt:
            if sys.platform == "win32":
                subprocess.run(["taskkill", "/PID", str(proceso.pid), "/T", "/F"],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                proceso.terminate()
            proceso.wait()
            raise
    if codigo:
        raise RuntimeError(f"Falló el entrenamiento. Consulta {ruta.with_name('ejecucion.log')}")


def repetir(plan, semillas=(380, 381, 382, 383, 384), carpeta=None):
    """Usar exactamente las mismas semillas nuevas para todas las candidatas."""
    if len(set(semillas)) != len(semillas) or len(semillas) < 2:
        raise ValueError("Se necesitan al menos dos semillas distintas para estimar dispersión.")
    for fila in plan.itertuples():
        if set(semillas) & set(json.loads(fila.semillas_previas)):
            raise ValueError("Las semillas nuevas no pueden coincidir con las del ajuste inicial.")
        for seed in semillas:
            semilla_entrenamiento(seed, json.loads(fila.config)["n_episodios"] - 1)
    codigo = b"".join(p.read_bytes() for p in [Path(__file__), *sorted((RAIZ / "rl_trader").glob("*.py"))])
    firma = dict(plan=plan.to_dict("records"), codigo=hashlib.sha256(codigo).hexdigest())
    identificador = hashlib.sha256(json.dumps(firma, sort_keys=True).encode()).hexdigest()[:12]
    destino = Path(carpeta or RAIZ / "results/ajuste_robusto") / identificador
    destino.mkdir(parents=True, exist_ok=True)
    plan.to_csv(destino / "candidatas.csv", index=False)
    print(f"{len(plan) * len(semillas)} ensayos como máximo. Resultados: {destino}", flush=True)
    filas = []
    for fila in plan.itertuples():
        for seed in semillas:
            ensayo = destino / f"{fila.candidato}_{seed}"
            if not (ensayo / "resumen.json").exists():
                ensayo.mkdir(exist_ok=True)
                config = json.loads(fila.config) | {"seed": seed}
                ruta = ensayo / "configuracion.json"
                ruta.write_text(json.dumps({"config": config}, indent=2), encoding="utf-8")
                print(f"{fila.familia}/{fila.mercado}, {fila.candidato}, semilla {seed}...", flush=True)
                _proceso(ruta)
            resumen = json.loads((ensayo / "resumen.json").read_text(encoding="utf-8"))
            filas.append(dict(familia=fila.familia, mercado=fila.mercado, candidato=fila.candidato,
                              actual=fila.actual, seed=seed, **resumen))
    datos = pd.DataFrame(filas)
    datos.to_csv(destino / "repeticiones.csv", index=False)
    datos.attrs["carpeta"] = str(destino)
    return datos


def resumir(plan, datos):
    """Ranking con semillas nuevas y diferencias pareadas frente a la actual."""
    from IPython.display import display
    filas, propuestas = [], []
    rng = np.random.default_rng(2026)
    for (familia, mercado), grupo in datos.groupby(["familia", "mercado"]):
        notas = grupo.pivot(index="seed", columns="candidato", values="val_mediana")
        if notas.isna().any().any():
            raise ValueError("Faltan semillas comunes: no se puede comparar un resultado parcial.")
        actual = grupo.loc[grupo.actual, "candidato"].iloc[0]
        candidatos = []
        for candidato in notas.columns:
            x = notas[candidato].to_numpy()
            diferencias = x - notas[actual].to_numpy()
            remuestreo = rng.choice(diferencias, size=(5000, len(x)), replace=True).mean(axis=1)
            bajo, alto = np.quantile(remuestreo, [.025, .975])
            previa = plan[plan.candidato == candidato].iloc[0]
            fila = dict(familia=familia, mercado=mercado, candidato=candidato, actual=candidato == actual,
                n=len(x), media_previa=previa.media_previa, media_nueva=x.mean(),
                desviacion=x.std(ddof=1), error_media=x.std(ddof=1)/np.sqrt(len(x)),
                diferencia_pp=diferencias.mean(), intervalo_bajo=bajo, intervalo_alto=alto,
                semillas_favorables=int((diferencias > 0).sum()),
                segundos=grupo[grupo.candidato == candidato].segundos.mean())
            filas.append(fila)
            # Evitar cambiar por una ventaja pequeña o compatible con cero.
            margen = max(1.0, .05 * abs(notas[actual].mean()))
            if candidato != actual and bajo > 0 and diferencias.mean() > margen:
                candidatos.append(fila)
        elegido = max(candidatos, key=lambda f: f["media_nueva"])["candidato"] if candidatos else actual
        propuestas.append(dict(familia=familia, mercado=mercado, candidato=elegido,
                               cambia=elegido != actual, config=plan[plan.candidato == elegido].iloc[0].config))
        print(f"{familia}/{mercado}: " + (f"propuesta de cambio a {elegido}." if elegido != actual else
              "se mantiene la configuración actual; las alternativas no superan la regla de cambio."))
    resumen = pd.DataFrame(filas).sort_values(["familia", "mercado", "media_nueva"], ascending=[True, True, False])
    display(resumen.round(3))
    print("Intervalos bootstrap pareados orientativos: cinco semillas dan poca precisión y no hay corrección por comparaciones múltiples.")
    print("Se usan solo semillas nuevas en el ranking final. La incertidumbre describe el entrenamiento, no otros mercados.")
    print("No se ha consultado test ni modificado mejores.csv. Las propuestas requieren confirmación independiente.")
    carpeta = Path(datos.attrs["carpeta"])
    resumen.to_csv(carpeta / "comparacion.csv", index=False)
    pd.DataFrame(propuestas).to_csv(carpeta / "propuestas.csv", index=False)
    return resumen


if __name__ == "__main__":
    _entrenar(sys.argv[1])
