"""Experimento de estabilidad de DQN; no modifica rl_trader/dqn.py."""
from functools import partial
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from rl_trader import dqn
from rl_trader.evaluation import SEMILLA_VALIDACION, generar_series, semilla_entrenamiento
from rl_trader.prices import GENERADORES
from rl_trader.redes import Normalizador

RAIZ = Path(__file__).resolve().parent.parent


class AgenteProgramado(dqn.AgenteDQN):
    """Mismo DQN y mismo Adam; solo cambia la tasa en la segunda mitad."""

    def __init__(self, *args, episodios, pasos_episodio, factor_final, **kwargs):
        super().__init__(*args, **kwargs)
        self.episodios, self.pasos_episodio = episodios, pasos_episodio
        self.factor_final, self.pasos = factor_final, 0
        self.alpha_inicial = self.optimizador.param_groups[0]["lr"]

    def elegir_accion(self, s, eps):
        self.pasos += 1
        return super().elegir_accion(s, eps)

    def aprender(self, *experiencia):
        episodio = (self.pasos - 1) // self.pasos_episodio
        progreso = np.clip((episodio - self.episodios / 2) / max(1, self.episodios / 2 - 1), 0, 1)
        self.optimizador.param_groups[0]["lr"] = self.alpha_inicial * self.factor_final ** progreso
        super().aprender(*experiencia)


def comparar(episodios=400, semillas=(350, 351, 352), evaluar_cada=25,
             n_validacion=100, ejecutar=True, carpeta=None, solo_aprendizaje=False, config_base=None):
    """Cuatro variantes con la misma validación; guarda y reutiliza cada ensayo."""
    if episodios < 4 or not 1 <= evaluar_cada <= episodios // 4:
        raise ValueError("Se necesitan al menos cuatro controles de validación.")
    for seed in semillas:
        semilla_entrenamiento(seed, episodios - 1)
    if config_base is None:
        elegidas = pd.read_csv(RAIZ / "results/mejores.csv")
        config = json.loads(elegidas.query("familia == 'dqn' and mercado == 'reversion'").iloc[0].config)
    else:
        config = dict(config_base)
    for clave in ("mercado", "algoritmo", "n_validacion"):
        config.pop(clave, None)
    config.update(n_episodios=episodios, evaluar_cada=evaluar_cada)
    frecuencia = config.get("sincronizar_cada", inspect.signature(dqn.entrenar_dqn).parameters["sincronizar_cada"].default)
    variantes = {"original": (1.0, frecuencia), "solo_alpha": (0.3, frecuencia),
                 "alpha_objetivo_rapido": (0.3, max(1, frecuencia // 2)),
                 "alpha_objetivo_lento": (0.3, frecuencia * 2)}
    if solo_aprendizaje:
        variantes = {nombre: variantes[nombre] for nombre in ("original", "solo_alpha")}
    codigo = b"".join(p.read_bytes() for p in [Path(__file__), *sorted((RAIZ / "rl_trader").glob("*.py"))])
    firma = dict(config=config, variantes=variantes, n_validacion=n_validacion,
                 codigo=hashlib.sha256(codigo).hexdigest())
    identificador = hashlib.sha256(json.dumps(firma, sort_keys=True).encode()).hexdigest()[:12]
    salida = Path(carpeta or RAIZ / "results/estabilidad_dqn_reversion") / identificador
    print(f"Resultados separados: {salida}", flush=True)
    generador = GENERADORES["reversion"]
    normalizador = Normalizador.ajustar(generador)
    precios = np.array(generar_series(generador, n_validacion, SEMILLA_VALIDACION))
    curvas = []
    for variante, (factor, sincronizar_cada) in variantes.items():
        for seed in semillas:
            destino = salida / f"{variante}_{seed}"
            if not (destino / "completo.json").exists():
                if not ejecutar:
                    print(f"Pendiente: {variante}, semilla {seed}")
                    continue
                destino.mkdir(parents=True, exist_ok=True)
                params = config | {"seed": seed, "sincronizar_cada": sincronizar_cada}
                print(f"Entrenando {variante}, semilla {seed}...", flush=True)
                clase_original = dqn.AgenteDQN
                # Adaptador temporal; se restaura incluso al interrumpir o fallar.
                dqn.AgenteDQN = partial(AgenteProgramado, episodios=episodios,
                                       pasos_episodio=params.get("n_steps", 500) - 6, factor_final=factor)
                try:
                    res = dqn.entrenar_dqn(generador, normalizador, precios_validacion=precios, **params)
                finally:
                    dqn.AgenteDQN = clase_original
                trabajo = dict(config=params, factor_final=factor, variante=variante, n_validacion=n_validacion)
                (destino / "configuracion.json").write_text(json.dumps(trabajo, indent=2), encoding="utf-8")
                curva = res.validacion.assign(variante=variante, seed=seed,
                                               segundos_entrenamiento=res.segundos,
                                               sincronizar_cada=sincronizar_cada)
                progreso = np.clip((curva.episodio - 1 - episodios / 2) / max(1, episodios / 2 - 1), 0, 1)
                curva["alpha"] = params.get("alpha", 5e-4) * factor ** progreso
                curva.to_csv(destino / "validacion.csv", index=False)
                torch.save(dict(mejor=res.estado, ultimo=res.estado_final, tipo_red=res.tipo_red,
                                ocultas=res.ocultas, normalizador=vars(normalizador), trabajo=trabajo),
                           destino / "pesos.pt")
                res.historial.to_csv(destino / "episodios.csv", index=False)
                (destino / "completo.json").write_text(json.dumps({"segundos": res.segundos}), encoding="utf-8")
                print(f"Completado en {res.segundos:.0f} s de entrenamiento.", flush=True)
            curvas.append(pd.read_csv(destino / "validacion.csv"))
    if len(curvas) != len(variantes) * len(semillas):
        print("Comparación incompleta; faltan ensayos para extraer conclusiones.")
    resultado = pd.concat(curvas, ignore_index=True) if curvas else pd.DataFrame()
    resultado.attrs["completa"] = len(curvas) == len(variantes) * len(semillas)
    resultado.attrs["salida"] = str(salida)
    return resultado


def confirmar(semillas=(360, 361, 362, 363, 364), referencia=None, carpeta=None):
    """Repetir la comparación elegida con semillas nuevas, sin tocar test."""
    referencia = Path(referencia or RAIZ / "results/estabilidad_dqn_reversion/6c28a6b8b526")
    anteriores = sorted(referencia.glob("original_*/configuracion.json"))
    if not anteriores:
        raise FileNotFoundError(f"No se encuentra el experimento de referencia: {referencia}")
    trabajos = [json.loads(r.read_text(encoding="utf-8")) for r in anteriores]
    usadas = {t["config"]["seed"] for t in trabajos}
    if not semillas or len(set(semillas)) != len(semillas) or set(semillas) & usadas:
        raise ValueError("Las semillas deben ser distintas entre sí y de las del experimento anterior.")
    # Recuperar la configuración probada, aunque mejores.csv haya cambiado después.
    config = dict(trabajos[0]["config"])
    config.pop("seed", None)
    curvas = comparar(episodios=config["n_episodios"], evaluar_cada=config["evaluar_cada"],
                      semillas=semillas, n_validacion=trabajos[0]["n_validacion"],
                      solo_aprendizaje=True, config_base=config, carpeta=carpeta)
    detalle = mostrar_resultados(curvas)
    medias = detalle.groupby("variante").mean(numeric_only=True)
    base, nueva = medias.loc["original"], medias.loc["solo_alpha"]
    ganancias = detalle.pivot(index="seed", columns="variante", values="G_final")
    retornos = detalle.pivot(index="seed", columns="variante", values="retorno_final_pct")
    favorables_G = int((ganancias.solo_alpha >= ganancias.original).sum())
    favorables_retorno = int((retornos.solo_alpha >= retornos.original).sum())
    confirmada = bool(nueva.G_final >= base.G_final and nueva.retorno_final_pct >= base.retorno_final_pct
                     and nueva.oscilacion < base.oscilacion and nueva.brecha_pp < base.brecha_pp
                     and favorables_G > len(semillas) / 2 and favorables_retorno > len(semillas) / 2)
    print(f"G no inferior en {favorables_G}/{len(semillas)} semillas nuevas; "
          f"retorno final no inferior en {favorables_retorno}/{len(semillas)}.")
    print("La mejora supera el criterio exploratorio de confirmación." if confirmada else
          "La mejora no se confirma con el criterio fijado: se conserva DQN original.")
    print("No se ha consultado test. La evaluación independiente queda pendiente.")
    decision = dict(variante="solo_alpha" if confirmada else "original", confirmada=confirmada,
                    semillas=list(semillas), favorables_G=favorables_G,
                    favorables_retorno=favorables_retorno, config=config,
                    n_validacion=trabajos[0]["n_validacion"], referencia=str(referencia))
    salida = Path(curvas.attrs["salida"])
    identificador = hashlib.sha256(json.dumps(decision, sort_keys=True).encode()).hexdigest()[:12]
    ruta = salida / f"confirmacion_{identificador}.json"
    ruta.write_text(json.dumps(decision, indent=2), encoding="utf-8")
    print(f"Decisión guardada: {ruta}")
    return dict(curvas=curvas, detalle=detalle, decision=decision, archivo=ruta)


def mostrar_resultados(curvas):
    """Comparar estabilidad sin confundir menor oscilación con mejor rendimiento."""
    import matplotlib.pyplot as plt
    from IPython.display import display

    if curvas.empty:
        print("No hay ensayos completos; ejecuta primero la comparación.")
        return
    filas = []
    for (variante, seed), datos in curvas.groupby(["variante", "seed"]):
        datos = datos.sort_values("episodio")
        g = datos.ganancia.to_numpy()
        cuarto = max(1, len(g) // 4)
        filas.append(dict(variante=variante, seed=seed, G_final=g[-cuarto:].mean(),
                          oscilacion=g[-cuarto:].std(), retorno_final_pct=datos.retorno_mediana_pct.iloc[-1],
                          mejor_retorno_pct=datos.retorno_mediana_pct.max(),
                          brecha_pp=datos.retorno_mediana_pct.max()-datos.retorno_mediana_pct.iloc[-1],
                          segundos=datos.segundos_entrenamiento.iloc[-1]))
    detalle = pd.DataFrame(filas)
    resumen = detalle.groupby("variante").mean(numeric_only=True).drop(columns="seed")
    resumen["rango_G_semillas"] = detalle.groupby("variante").G_final.agg(lambda x: x.max()-x.min())
    display(resumen.round(4))
    display(detalle.round(4))
    fig, ejes = plt.subplots(1, 2, figsize=(12, 4))
    for variante, datos in curvas.groupby("variante"):
        for eje, metrica in zip(ejes, ("ganancia", "retorno_mediana_pct")):
            grupo = datos.groupby("episodio")[metrica]
            media, dispersion = grupo.mean(), grupo.std().fillna(0)
            linea, = eje.plot(media.index, media, label=variante)
            eje.fill_between(media.index, media-dispersion, media+dispersion, color=linea.get_color(), alpha=.12)
    for eje, titulo in zip(ejes, ("Ganancia G en validación", "Rentabilidad mediana en validación (%)")):
        eje.set(title=titulo, xlabel="Episodio")
        eje.grid(alpha=.3)
        eje.legend(fontsize=8)
    fig.tight_layout()
    plt.show()
    if curvas.attrs.get("completa", False) and "original" in resumen.index:
        base = resumen.loc["original"]
        for nombre, fila in resumen.drop(index="original").iterrows():
            mejora = (fila.G_final >= base.G_final and fila.retorno_final_pct >= base.retorno_final_pct
                      and fila.oscilacion < base.oscilacion and fila.brecha_pp < base.brecha_pp)
            print(f"{nombre}: Δretorno={fila.retorno_final_pct-base.retorno_final_pct:+.2f} puntos; "
                  f"Δbrecha={fila.brecha_pp-base.brecha_pp:+.2f} puntos. "
                  + ("Mejora conjunta observada en validación." if mejora else
                     "No cumple todos los criterios de mejora conjunta."))
    print("Bandas: desviación entre semillas, no intervalos de confianza. Confirmar con semillas nuevas antes de test.")
    return detalle
