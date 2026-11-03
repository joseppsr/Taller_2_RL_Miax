"""Comprobaciones rápidas de que las piezas hacen lo que dicen.  Lanzar con: pytest"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from rl_trader.a3c import entrenar_a3c, entrenar_reinforce
from rl_trader.agentes import Agente, crear_agente
from rl_trader.analisis import entropia_relativa
from rl_trader.baselines import BASELINES
from rl_trader.dqn import Buffer, entrenar_dqn
from rl_trader.entrenamiento import MejorControl, decaimiento_exponencial
from rl_trader.env import SimpleTradingEnv
from rl_trader.evaluation import (SEMILLA_TEST, SEMILLA_VALIDACION, ejecutar_episodio, episodio_convergencia, evaluar,
                                  generar_series, metricas, simular_lote, tendencias)
from rl_trader.prices import GENERADORES
from rl_trader.redes import Normalizador, RedDuelo, RedQ, crear_red

RAIZ = Path(__file__).resolve().parent.parent
MERCADOS = list(GENERADORES)


def _celdas_notebook():
    return [''.join(c["source"]) for c in json.load(open(RAIZ / "taller_trader2.ipynb", encoding="utf8"))["cells"]
            if c["cell_type"] == "code"]


@pytest.mark.parametrize("modulo,marcador", [("env.py", "class SimpleTradingEnv"),
                                             ("prices.py", "def generar_precios("),
                                             ("prices.py", "def generar_precios_browniano"),
                                             ("prices.py", "def generar_precios_reversion_media"),
                                             ("prices.py", "def generar_precios_regimenes")])
def test_codigo_identico_al_enunciado(modulo, marcador):
    """El entorno y los generadores no se han tocado respecto al notebook."""
    celda = next(c for c in _celdas_notebook() if c.lstrip().startswith(marcador))
    assert celda.strip() in (RAIZ / "rl_trader" / modulo).read_text(encoding="utf8")


def test_series_reproducibles_y_disjuntas():
    g = GENERADORES["browniano"]
    a, b = generar_series(g, 2, SEMILLA_TEST), generar_series(g, 2, SEMILLA_TEST)
    assert np.array_equal(a[1], b[1])
    assert not np.array_equal(a[0], generar_series(g, 1, SEMILLA_VALIDACION)[0])


@pytest.mark.parametrize("mercado", MERCADOS)
def test_tendencias_coinciden_con_entorno(mercado):
    precios = generar_series(GENERADORES[mercado], 1, 7)[0]
    env = SimpleTradingEnv(precios)
    obs, _ = env.reset()
    t = tendencias(precios)
    for k in range(50):
        assert obs[0] == t[k]
        obs, *_ = env.step(1)


@pytest.mark.parametrize("mercado", MERCADOS)
def test_simular_lote_coincide_con_el_entorno(mercado):
    """Con acciones al azar, el simulador rápido debe dar exactamente lo mismo que el entorno."""
    precios = np.array(generar_series(GENERADORES[mercado], 5, 123))
    rng = np.random.default_rng(0)
    A = rng.integers(0, 3, size=(5, 2, precios.shape[1] - 6))
    bal, pos = simular_lote(precios, A)
    for i in range(5):
        env = SimpleTradingEnv(precios[i])
        env.reset()
        balances, posiciones = [env.balance], []
        for k in range(precios.shape[1] - 6):
            env.step(int(A[i, env.position, k]))
            balances.append(env.balance)
            posiciones.append(env.position)
        np.testing.assert_allclose(bal[i], balances, rtol=1e-12)
        np.testing.assert_array_equal(pos[i], posiciones)


def test_tablas_coinciden_con_politica_en_el_entorno():
    """Las tablas precalculadas de la red deciden lo mismo que la red paso a paso en el entorno."""
    g = GENERADORES["regimenes"]
    norm = Normalizador.ajustar(g, n_series=5)
    torch.manual_seed(0)
    agente = Agente(RedQ(), norm)
    precios = np.array(generar_series(g, 3, 55))
    bal, _ = simular_lote(precios, agente.acciones_greedy(precios))
    for i in range(3):
        b, _ = ejecutar_episodio(precios[i], agente)
        np.testing.assert_allclose(bal[i], b, rtol=1e-12)


def test_estado_continuo_sin_intervalos():
    """Dos tendencias muy cercanas son entradas distintas para la red: no hay discretización."""
    norm = Normalizador(0.02, 0.02)
    a, b = norm.entrada(0.010, 1), norm.entrada(0.0101, 1)
    assert a[0] != b[0] and a[1] == b[1] == 1.0
    assert norm.entrada(0.5, 0)[0] == 5.0 and norm.entrada(0.5, 0)[1] == -1.0


def test_duelo_resta_la_media_de_la_ventaja():
    torch.manual_seed(0)
    red = RedDuelo()
    x = torch.rand(4, 2)
    h = red.base(x)
    a = red.ventaja(h)
    esperado = red.valor(h) + a - a.mean(dim=1, keepdim=True)
    assert torch.allclose(red(x), esperado)


def test_buffer_circular():
    b = Buffer(3)
    for i in range(5):
        b.guardar([i, i], i % 3, float(i), [i, i], 0.0)
    assert b.n == 3 and set(b.r.tolist()) == {2.0, 3.0, 4.0}


def test_decaimiento():
    assert decaimiento_exponencial(0, 100, 1.0, 0.01, 0.5) == 1.0
    assert decaimiento_exponencial(50, 100, 1.0, 0.01, 0.5) == 0.01
    assert decaimiento_exponencial(99, 100, 1.0, 0.01, 0.5) == 0.01


def test_mejor_control_desempata_por_operar_menos():
    m = MejorControl()
    m.actualizar({"retorno_mediana_pct": 0.0, "operaciones": 50}, "a")
    m.actualizar({"retorno_mediana_pct": 0.0, "operaciones": 0}, "b")
    m.actualizar({"retorno_mediana_pct": -1.0, "operaciones": 0}, "c")
    assert m.estado == "b"


def test_convergencia():
    v = pd.DataFrame({"episodio": [10, 20, 30, 40, 50, 60], "retorno_mediana_pct": [0, 0, 50, 100, 100, 100]})
    assert episodio_convergencia(v, suavizado=1) == 40


@pytest.mark.parametrize("flags", [{}, {"doble": True}, {"duelo": True}])
def test_dqn_se_ejecuta_y_aprende_el_ciclo(flags):
    g = GENERADORES["ciclo"]
    norm = Normalizador.ajustar(g, n_series=5)
    val = np.array(generar_series(g, 10, SEMILLA_VALIDACION))
    res = entrenar_dqn(g, norm, n_episodios=25, gamma=0.0, seed=3, precios_validacion=val, evaluar_cada=5,
                       inicio_aprendizaje=200, **flags)
    assert res.pasos == 25 * 494
    assert list(res.historial["episodio"]) == list(range(1, 26))
    assert len(res.validacion) == 5
    # El ciclo tiene mucha inercia: tras tan poco ya debe ganar al azar con claridad
    assert res.validacion["retorno_mediana_pct"].max() > 1000


def test_ganancia_es_la_suma_de_recompensas():
    """La G de los controles en validación coincide casi exactamente con Σ r del entorno."""
    precios = generar_series(GENERADORES["regimenes"], 1, 9)[0]
    env = SimpleTradingEnv(precios)
    env.reset()
    rng = np.random.default_rng(0)
    suma, balances, posiciones, done = 0.0, [env.balance], [], False
    while not done:
        _, r, done, _, _ = env.step(int(rng.integers(3)))
        suma += r
        balances.append(env.balance)
        posiciones.append(env.position)
    assert metricas(np.array(balances), np.array(posiciones))["ganancia"] == pytest.approx(suma, abs=2e-3)


@pytest.mark.parametrize("perdida", ["mse", "mae"])
def test_dqn_acepta_mse_y_mae(perdida):
    g = GENERADORES["browniano"]
    res = entrenar_dqn(g, Normalizador.ajustar(g, n_series=5), n_episodios=2, inicio_aprendizaje=100, perdida=perdida)
    assert res.pasos == 2 * 494


def test_a3c_con_procesos_se_ejecuta_y_cambia_los_pesos():
    g = GENERADORES["browniano"]
    norm = Normalizador.ajustar(g, n_series=5)
    val = np.array(generar_series(g, 5, SEMILLA_VALIDACION))
    res = entrenar_a3c(g, norm, n_episodios=8, n_trabajadores=2, seed=1, precios_validacion=val, evaluar_cada=4,
                       perdida_critico="mae")
    assert res.validacion["episodio"].iloc[-1] == 8
    # Cada episodio ha apuntado su ganancia en la memoria compartida
    assert (res.historial["ganancia"] != 0).all()
    assert res.segundos > 0
    inicial = crear_red("actor_critico")
    assert any(not torch.equal(v, inicial.state_dict()[k]) for k, v in res.estado_final.items())


def test_a3c_el_paso_de_adam_no_hace_saltar_los_pesos():
    # Con α = 3e-4 y 8 episodios, los pesos apenas se mueven. Sin cerrojo en el paso de Adam saltaban a más de 1.000
    # en más de la mitad de las ejecuciones, y el agente dejaba de operar.
    g = GENERADORES["browniano"]
    norm = Normalizador.ajustar(g, n_series=5)
    for _ in range(3):
        res = entrenar_a3c(g, norm, n_episodios=8, n_trabajadores=2, seed=1, perdida_critico="mae")
        assert max(v.abs().max().item() for v in res.estado_final.values()) < 5
        assert (res.historial["ganancia"] != 0).all()


def test_entropia_relativa_no_desborda_con_logits_enormes():
    assert entropia_relativa(np.array([[2000.0, -500.0, 0.0]])) == pytest.approx(0)
    assert entropia_relativa(np.zeros((1, 3))) == pytest.approx(1)


def test_reinforce_actualiza_al_final_del_episodio():
    g = GENERADORES["browniano"]
    norm = Normalizador.ajustar(g, n_series=5)
    val = np.array(generar_series(g, 5, SEMILLA_VALIDACION))
    res = entrenar_reinforce(g, norm, n_episodios=4, seed=1, precios_validacion=val, evaluar_cada=2)
    assert res.algoritmo == "reinforce" and res.validacion["episodio"].iloc[-1] == 4


def test_baselines_reproducen_el_taller_anterior():
    """Las series de test son las mismas que las del taller tabular: buy & hold debe dar lo mismo."""
    tabla = pd.read_csv(RAIZ / "results" / "referencia_tabular_fase3.csv")
    for mercado in MERCADOS:
        series = generar_series(GENERADORES[mercado], 100, SEMILLA_TEST)
        for nombre in ("buy_and_hold", "momentum", "contrarian"):
            mediana = evaluar(BASELINES[nombre](), series)["retorno_pct"].median()
            ref = tabla[(tabla.mercado == mercado) & (tabla.agente == nombre)]["mediana_%"].iloc[0]
            assert mediana == pytest.approx(ref, rel=1e-9), (mercado, nombre)


def test_tablas_preferencias_forma():
    norm = Normalizador(0.02, 0.02)
    p = np.array(generar_series(GENERADORES["browniano"], 2, 1))
    assert Agente(RedQ(), norm).tablas(p).shape == (2, 2, 494, 3)


def test_la_red_ve_la_serie_entera_de_una_vez():
    """Al evaluar, self.modelo recibe todos los pasos en una sola llamada por posición: no hay lotes."""
    norm = Normalizador(0.02, 0.02)
    p = np.array(generar_series(GENERADORES["browniano"], 2, 1))
    agente = Agente(RedQ(), norm)
    formas = []
    agente.modelo.register_forward_hook(lambda _m, entrada, _s: formas.append(tuple(entrada[0].shape)))
    agente.tablas(p)
    assert formas == [(2, 494, 2), (2, 494, 2)]


def test_dqn_por_episodio_entero_sin_minibatches():
    """Con modo="episodio" cada paso de gradiente ve las 494 transiciones de la serie."""
    g = GENERADORES["ciclo"]
    norm = Normalizador.ajustar(g, n_series=5)
    val = np.array(generar_series(g, 10, SEMILLA_VALIDACION))
    res = entrenar_dqn(g, norm, n_episodios=15, gamma=0.0, seed=3, precios_validacion=val, evaluar_cada=5,
                       modo="episodio", pasadas_episodio=16, alpha=2e-3)
    assert res.pasos == 15 * 494 and len(res.validacion) == 3


def test_diagnostico_distingue_oscilar_de_seguir_subiendo():
    """Una curva que aún sube tiene deriva; una que oscila sin tendencia tiene oscilación, pero no deriva."""
    from rl_trader.analisis import diagnostico_convergencia
    n = 40
    sube = pd.DataFrame({"mercado": "m", "agente": "sube", "seed": 0, "episodio": range(n),
                         "ganancia": np.linspace(0, 8, n)})
    oscila = pd.DataFrame({"mercado": "m", "agente": "oscila", "seed": 0, "episodio": range(n),
                           "ganancia": [4 + (-1) ** i for i in range(n)]})
    d = diagnostico_convergencia(pd.concat([sube, oscila])).set_index("agente")
    assert d.loc["sube", "deriva"] > 1.5 and abs(d.loc["oscila", "deriva"]) < 1e-9
    assert d.loc["oscila", "oscilacion"] == pytest.approx(1.0)


def test_los_oraculos_son_coherentes_con_la_teoria():
    """El oráculo gana a buy & hold donde hay información oculta, y en browniano coincide con él."""
    from rl_trader.oraculos import tabla_oraculos
    techos = tabla_oraculos(30, SEMILLA_VALIDACION).set_index("mercado")
    for mercado in GENERADORES:
        bh = evaluar(BASELINES["buy_and_hold"](), generar_series(GENERADORES[mercado], 30, SEMILLA_VALIDACION))
        if mercado == "browniano":
            assert techos.loc[mercado, "ganancia_G"] == pytest.approx(bh["ganancia"].mean())
        else:
            assert techos.loc[mercado, "ganancia_G"] > bh["ganancia"].mean()


def test_crear_agente_elige_la_clase():
    from rl_trader.agentes import AgenteActorCritico
    norm = Normalizador(0.02, 0.02)
    assert isinstance(crear_agente("actor_critico", (8,), None, norm), AgenteActorCritico)
    assert type(crear_agente("q", (8,), None, norm)) is Agente
