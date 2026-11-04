"""Comparación independiente de A3C original y A3C con aprendizaje decreciente.

El notebook llama a comparar(). Cada ensayo se ejecuta en un proceso separado
para que Windows pueda importar los trabajadores sin alterar el A3C del notebook.
"""
from functools import partial
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from rl_trader import a3c
from rl_trader.evaluation import SEMILLA_VALIDACION, generar_series, semilla_entrenamiento
from rl_trader.prices import GENERADORES
from rl_trader.redes import Normalizador, fijar_hilos


class TrabajadorProgramado(a3c.Trabajador):
    """Conserva Adam y reduce las tasas del actor y del crítico por episodio."""

    def _episodio(self, ep):
        progreso = min(1.0, ep / max(1, int(0.8 * self.c["n_episodios"])))
        factor = self.factor_final ** progreso
        for grupo, clave in zip(self.optimizador.param_groups, ("alpha", "alpha_critico")):
            grupo["lr"] = self.c[clave] * factor
        super()._episodio(ep)  # El calendario de entropía ya existe en A3C.


def lanzar_trabajador(barrera, *args, factor_final):
    fijar_hilos()
    trabajador = TrabajadorProgramado(*args)
    trabajador.factor_final = factor_final
    barrera.wait()
    trabajador.ejecutar()


def ejecutar_ensayo(ruta):
    """Solo se llama en el subproceso; no cambia el módulo del notebook."""
    trabajo = json.loads(Path(ruta).read_text(encoding="utf-8"))
    destino = Path(ruta).parent
    config = trabajo["config"]
    generador = GENERADORES["reversion"]
    normalizador = Normalizador.ajustar(generador)
    precios = np.array(generar_series(generador, trabajo["n_validacion"], SEMILLA_VALIDACION))
    lanzamiento_original = a3c._lanzar_trabajador
    # El destino de multiprocessing es una función importable, también con spawn.
    a3c._lanzar_trabajador = partial(lanzar_trabajador, factor_final=trabajo["factor_final"])
    try:
        resultado = a3c.entrenar_a3c(generador, normalizador, precios_validacion=precios, **config)
    finally:
        a3c._lanzar_trabajador = lanzamiento_original

    curva = resultado.validacion.assign(variante=trabajo["variante"], seed=config["seed"])
    progreso = np.minimum(1, (curva.episodio - 1) / max(1, int(0.8 * config["n_episodios"])))
    curva["alpha_actor"] = config.get("alpha", 3e-4) * trabajo["factor_final"] ** progreso
    curva["alpha_critico"] = config.get("alpha_critico", 1e-3) * trabajo["factor_final"] ** progreso
    curva.to_csv(destino / "validacion.csv", index=False)
    torch.save(dict(mejor=resultado.estado, ultimo=resultado.estado_final,
                    normalizador=vars(normalizador), ocultas=resultado.ocultas, trabajo=trabajo),
               destino / "pesos.pt")
    resultado.historial.to_csv(destino / "episodios.csv", index=False)
    # Se marca completo al final; si se interrumpe, solo se repite este ensayo.
    (destino / "completo.json").write_text(
        json.dumps({"segundos_entrenamiento": resultado.segundos}), encoding="utf-8")


def ejecutar_proceso(ruta):
    """Guarda errores en el log y permite interrumpir el árbol de procesos en Windows."""
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
        raise RuntimeError(f"El ensayo falló. Consulta {ruta.with_name('ejecucion.log')}")


def comparar(episodios=3200, semillas=(350, 351, 352), evaluar_cada=200,
             trabajadores=4, n_validacion=100, ejecutar=True, carpeta=None, solo_aprendizaje=False,
             factores=None, config_base=None):
    """Dos variantes, mismo presupuesto y validación. Reutiliza ensayos completos.

    No consulta test. El modo de 3.200 episodios es exploratorio; para comprobar
    estabilidad a largo plazo, repetir después con episodios=12800.
    Con solo_aprendizaje=True, ambas variantes usan el calendario original de entropía.
    factores permite comparar otros decaimientos conservando la entropía original.
    """
    if episodios < 4 or not 1 <= evaluar_cada <= episodios // 4:
        raise ValueError("Se necesitan al menos cuatro controles de validación.")
    if trabajadores < 1 or n_validacion < 1 or not semillas:
        raise ValueError("Trabajadores, series y número de semillas deben ser positivos.")
    for seed in semillas:
        semilla_entrenamiento(seed, episodios - 1)
    if config_base is None:
        elegidas = pd.read_csv(RAIZ / "results/mejores.csv")
        config = json.loads(elegidas.query("familia == 'a3c' and mercado == 'reversion'").iloc[0].config)
    else:
        config = dict(config_base)  # Congelar los demás parámetros entre etapas.
    config.pop("mercado", None)
    config.pop("n_validacion", None)
    config.update(n_episodios=episodios, evaluar_cada=evaluar_cada, n_trabajadores=trabajadores)
    beta = config.get("beta_final", 0.001)
    variantes = {"original": (1.0, beta), "decreciente": (0.1, min(1e-5, beta / 100))}
    if solo_aprendizaje:
        variantes = {"original": (1.0, beta), "solo_aprendizaje": (0.1, beta)}
    if factores is not None:
        if not factores or any(not 0 < f <= 1 for f in factores.values()):
            raise ValueError("Cada factor de aprendizaje debe estar entre 0 y 1.")
        variantes = {nombre: (factor, beta) for nombre, factor in factores.items()}
    # La huella distingue presupuestos y versiones de código; añadir semillas no invalida lo calculado.
    codigo = b"".join(p.read_bytes() for p in [Path(__file__), *sorted((RAIZ / "rl_trader").glob("*.py"))])
    firma = dict(config=config, variantes=variantes, n_validacion=n_validacion,
                 codigo=hashlib.sha256(codigo).hexdigest())
    identificador = hashlib.sha256(json.dumps(firma, sort_keys=True).encode()).hexdigest()[:12]
    salida = Path(carpeta or RAIZ / "results/estabilidad_a3c_reversion") / identificador
    print(f"Resultados: {salida}", flush=True)
    curvas = []
    for variante, (factor, beta_final) in variantes.items():
        for seed in semillas:
            destino = salida / f"{variante}_{seed}"
            if not (destino / "completo.json").exists():
                if not ejecutar:
                    print(f"Pendiente: {variante}, semilla {seed}")
                    continue
                destino.mkdir(parents=True, exist_ok=True)
                trabajo = dict(config=config | {"seed": seed, "beta_final": beta_final},
                               factor_final=factor, variante=variante, n_validacion=n_validacion)
                ruta = destino / "configuracion.json"
                ruta.write_text(json.dumps(trabajo, indent=2), encoding="utf-8")
                print(f"Entrenando {variante}, semilla {seed}...", flush=True)
                inicio = time.perf_counter()
                ejecutar_proceso(ruta)
                print(f"Completado en {(time.perf_counter() - inicio) / 60:.1f} min", flush=True)
            tiempo = json.loads((destino / "completo.json").read_text(encoding="utf-8"))
            curvas.append(pd.read_csv(destino / "validacion.csv").assign(
                ruta_pesos=str(destino / "pesos.pt"), factor_final=factor,
                segundos_entrenamiento=tiempo["segundos_entrenamiento"]))
    if len(curvas) != len(variantes) * len(semillas):
        print("Comparación incompleta: aún no se pueden comparar todas las semillas.")
    return pd.concat(curvas, ignore_index=True) if curvas else pd.DataFrame()


def mostrar_comparacion(curvas):
    """Curvas de validación y resumen del último cuarto del entrenamiento."""
    import matplotlib.pyplot as plt
    from IPython.display import display

    if curvas.empty:
        print("No hay resultados. Activa ejecutar=True para calcularlos.")
        return
    filas = []
    fig, ax = plt.subplots(figsize=(9, 4))
    for variante, datos in curvas.groupby("variante"):
        for seed, curva in datos.groupby("seed"):
            curva = curva.sort_values("episodio")
            g = curva.ganancia.to_numpy()
            cuarto = max(1, len(g) // 4)
            filas.append(dict(variante=variante, seed=seed, G_final=g[-cuarto:].mean(),
                              oscilacion=g[-cuarto:].std(),
                              deriva=g[-cuarto:].mean() - g[-2*cuarto:-cuarto].mean(),
                              retorno_final_pct=curva.retorno_mediana_pct.iloc[-1],
                              brecha_mejor_ultimo_pp=curva.retorno_mediana_pct.max()
                              - curva.retorno_mediana_pct.iloc[-1]))
            if "segundos_entrenamiento" in curva:
                filas[-1]["segundos_entrenamiento"] = curva.segundos_entrenamiento.iloc[0]
        grupo = datos.groupby("episodio").ganancia
        media, desviacion = grupo.mean(), grupo.std().fillna(0)
        linea, = ax.plot(media.index, media, label=variante)
        ax.fill_between(media.index, media - desviacion, media + desviacion,
                        color=linea.get_color(), alpha=0.15)
    detalle = pd.DataFrame(filas)
    resumen = detalle.groupby("variante").agg(
        semillas=("seed", "nunique"), G_final=("G_final", "mean"),
        oscilacion=("oscilacion", "mean"), deriva=("deriva", "mean"),
        rango_semillas=("G_final", lambda x: x.max() - x.min()),
        retorno_final_pct=("retorno_final_pct", "mean"),
        brecha_mejor_ultimo_pp=("brecha_mejor_ultimo_pp", "mean"))
    if "segundos_entrenamiento" in detalle:
        resumen["segundos_entrenamiento"] = detalle.groupby("variante").segundos_entrenamiento.mean()
    display(resumen.round(4))
    ax.set(xlabel="Episodio", ylabel="Ganancia G en validación",
           title="A3C en reversión: media y desviación entre semillas")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    plt.show()
    print("Menor oscilación solo mejora el resultado si se mantiene o aumenta el rendimiento.")
    print("Es una comparación exploratoria en validación, no un resultado final de test.")
    return detalle


def _opciones_mejores(detalle):
    """Regla exploratoria fijada antes de consultar test; no es un test estadístico."""
    medias = detalle.groupby("variante").mean(numeric_only=True)
    base = medias.loc["original"]
    return medias.loc[(medias.G_final >= base.G_final)
                      & (medias.retorno_final_pct >= base.retorno_final_pct)
                      & (medias.oscilacion < base.oscilacion)].sort_values("G_final", ascending=False)


def explorar_aprendizaje(**presupuesto):
    """Paso 1: comparar original, 10 % y 30 %, exclusivamente en validación."""
    factores = {"original": 1.0, "alpha_10": 0.1, "alpha_30": 0.3}
    curvas = comparar(factores=factores, **presupuesto)
    detalle = mostrar_comparacion(curvas)
    opciones = _opciones_mejores(detalle)
    candidato = opciones.index[0] if len(opciones) else "original"
    ruta = Path(curvas.ruta_pesos.iloc[0]).parent / "configuracion.json"
    config = json.loads(ruta.read_text(encoding="utf-8"))["config"]
    print(f"Candidato para confirmar: {candidato}.")
    print("Regla: G y rentabilidad final no inferiores al original, y menor oscilación.")
    if candidato == "original":
        print("Ninguna reducción cumple la regla: no hay mejora conjunta en esta búsqueda.")
    else:
        print("Cumple la regla en estas semillas; falta comprobar si se repite en semillas nuevas.")
    return dict(candidato=candidato, factores=factores, config=config, curvas=curvas,
                detalle=detalle, presupuesto=presupuesto)


def confirmar_aprendizaje(busqueda, semillas=(360, 361, 362, 363, 364)):
    """Paso 2: confirmar sin probar candidatos nuevos ni consultar test."""
    if len(set(semillas)) != len(semillas) or set(semillas) & set(busqueda["curvas"].seed):
        raise ValueError("La confirmación necesita semillas distintas entre sí y de la búsqueda.")
    candidato = busqueda["candidato"]
    factores = {"original": 1.0, candidato: busqueda["factores"][candidato]}
    presupuesto = busqueda["presupuesto"] | {"semillas": semillas}
    curvas = comparar(factores=factores, config_base=busqueda["config"], **presupuesto)
    detalle = mostrar_comparacion(curvas)
    decision = "original"
    if candidato != "original":
        pares = detalle.pivot(index="seed", columns="variante", values="G_final")
        victorias = int((pares[candidato] >= pares.original).sum())
        if candidato in _opciones_mejores(detalle).index and victorias > len(semillas) / 2:
            decision = candidato
        print(f"G no inferior al original en {victorias}/{len(semillas)} semillas nuevas.")
    print(f"Decisión antes del test: {decision}.")
    print("Mejora exploratoria confirmada." if decision != "original" else
          "No se confirma una mejora conjunta: se conserva A3C original.")
    # Guardar la decisión ANTES de generar las series reservadas.
    destino = Path(curvas.ruta_pesos.iloc[0]).parent.parent
    modelos = curvas[["variante", "seed", "ruta_pesos"]].drop_duplicates()
    modelos = modelos[modelos.variante.isin({"original", decision})]
    manifiesto = dict(decision=decision, candidato=candidato,
                      modelos=modelos.to_dict("records"), semilla_test=4_170_000_000, n_test=100)
    identificador = hashlib.sha256(json.dumps(manifiesto, sort_keys=True).encode()).hexdigest()[:12]
    ruta = destino / f"decision_{identificador}.json"
    contenido = json.dumps(manifiesto, indent=2)
    if not ruta.exists():
        ruta.write_text(contenido, encoding="utf-8")
    return ruta


def evaluar_test_reservado(decision):
    """Paso 3: evaluar una decisión congelada; nunca elegir parámetros con test."""
    from IPython.display import display
    from rl_trader.agentes import crear_agente
    from rl_trader.evaluation import evaluar_lote

    decision = Path(decision)
    plan = json.loads(decision.read_text(encoding="utf-8"))
    ruta = decision.with_name(decision.stem + "_test.csv")
    if ruta.exists():
        resultados = pd.read_csv(ruta)
        print("Se muestran los resultados del test ya realizado; no se genera otro test.")
    else:
        fijar_hilos()
        # Rango distinto de entrenamiento, validación, test anterior y normalización.
        precios = np.array(generar_series(GENERADORES["reversion"], plan["n_test"], plan["semilla_test"]))
        filas = []
        for modelo in plan["modelos"]:
            pesos = torch.load(modelo["ruta_pesos"], map_location="cpu", weights_only=True)
            normalizador = Normalizador(**pesos["normalizador"])
            # Último control: criterio principal de estabilidad. Mejor control: diagnóstico secundario.
            for control in ("ultimo", "mejor"):
                agente = crear_agente("actor_critico", pesos["ocultas"], pesos[control], normalizador)
                datos = evaluar_lote(precios, agente.acciones_greedy(precios))
                filas.append(datos.assign(variante=modelo["variante"], seed=modelo["seed"],
                                          control=control, serie=np.arange(len(datos))))
        resultados = pd.concat(filas, ignore_index=True)
        resultados.to_csv(ruta, index=False)
    por_semilla = resultados.groupby(["variante", "control", "seed"]).agg(
        ganancia_G=("ganancia", "mean"), retorno_mediana_pct=("retorno_pct", "median"))
    resumen = por_semilla.groupby(["variante", "control"]).mean()
    display(resumen.round(4))
    if plan["decision"] == "original":
        print("Se evalúa el original: ninguna alternativa superó el criterio previo al test.")
    else:
        ultimo = resumen.xs("ultimo", level="control")
        diferencia = ultimo.loc[plan["decision"]] - ultimo.loc["original"]
        print(f"Último control frente al original: ΔG={diferencia.ganancia_G:+.4f}; "
              f"Δrentabilidad={diferencia.retorno_mediana_pct:+.2f} puntos.")
        print("Mejora observada en ambas métricas de test." if (diferencia > 0).all() else
              "El test no muestra una mejora en ambas métricas.")
    print("Resultados descriptivos: no prueban significación estadística. No reajustar con este test.")
    return resultados


if __name__ == "__main__":
    ejecutar_ensayo(sys.argv[1])
