"""Construye la presentación de 10 minutos a partir de los CSV del proyecto."""
from pathlib import Path
import sys
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
from rl_trader.prices import GENERADORES

SALIDA = RAIZ / "report"
RECURSOS = SALIDA / "presentacion_recursos"
RECURSOS.mkdir(exist_ok=True)
AZUL, VERDE, NARANJA, GRIS = "18334B", "00877C", "D17B39", "66788A"
FONDO, CLARO = "F5F7FA", "E7F1F0"
plt.rcParams.update({"font.family": "Arial", "font.size": 13,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.spines.left": False, "axes.spines.bottom": False,
                     "axes.labelcolor": "#"+AZUL, "text.color": "#"+AZUL,
                     "xtick.color": "#"+GRIS, "ytick.color": "#"+GRIS,
                     "axes.titleweight": "bold", "savefig.facecolor": "white"})


def guardar(fig, nombre):
    ruta = RECURSOS / f"{nombre}.png"
    fig.savefig(ruta, dpi=190, bbox_inches="tight", pad_inches=.2)
    plt.close(fig)
    return ruta


def curvas_carpeta(carpeta):
    filas = []
    for f in sorted((RAIZ / carpeta).glob("*/validacion.csv")):
        d = pd.read_csv(f).sort_values("episodio")
        g = d.ganancia.to_numpy()
        cuarto = max(1, len(g)//4)
        filas.append(dict(variante=d.variante.iloc[0], seed=int(d.seed.iloc[0]),
                          G=g[-cuarto:].mean(), oscilacion=g[-cuarto:].std(),
                          retorno=d.retorno_mediana_pct.iloc[-1],
                          brecha=d.retorno_mediana_pct.max()-d.retorno_mediana_pct.iloc[-1]))
    return pd.DataFrame(filas)


def graficos():
    archivos = {}
    nombres = {"ciclo": "Ciclo", "browniano": "Browniano", "reversion": "Reversión", "regimenes": "Regímenes"}
    fig, ejes = plt.subplots(2, 2, figsize=(10.9, 4.15))
    for ax, (mercado, nombre) in zip(ejes.flat, nombres.items()):
        np.random.seed(12)
        p = GENERADORES[mercado](n_steps=500)
        ax.plot(p / p[0] * 100, color="#"+VERDE, lw=1.5)
        if mercado == "ciclo":
            ax.set_yscale("log")
            nombre += " · escala logarítmica"
        ax.set_title(nombre, fontsize=14, loc="left")
        ax.grid(axis="y", alpha=.15)
        ax.tick_params(labelsize=10, length=0)
        ax.set_xlabel("Paso", fontsize=10)
    fig.tight_layout(h_pad=1.1, w_pad=2)
    archivos["mercados"] = guardar(fig, "mercados")

    tabla = pd.read_csv(RAIZ / "results/tabla.csv")
    orden = ["aleatorio", "buy_and_hold", "contrarian", "dqn", "doble_dqn", "duelo", "a3c", "reinforce"]
    etiquetas = ["Aleatorio", "Buy & hold", "Contrarian", "DQN", "Doble DQN", "DQN en duelo", "A3C", "REINFORCE"]
    vals = tabla[tabla.mercado == "reversion"].set_index("agente").loc[orden].mediana_pct
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    colores = ["#A9B6C1"]*3 + ["#"+AZUL]*3 + ["#"+VERDE]*2
    ax.barh(etiquetas, vals, color=colores, height=.63)
    ax.invert_yaxis()
    ax.axvline(0, color="#A9B6C1", lw=.8)
    for i, valor in enumerate(vals):
        ax.text(valor + (1 if valor >= 0 else -1), i, f"{valor:.1f}%".replace(".", ","),
                va="center", ha="left" if valor >= 0 else "right", fontsize=13, fontweight="bold")
    ax.set_xlim(-24, 35)
    ax.set_xlabel("Rentabilidad mediana en test (%)")
    ax.grid(axis="x", alpha=.15)
    ax.set_axisbelow(True)
    fig.tight_layout()
    archivos["principal"] = guardar(fig, "test_reversion")

    test = pd.read_csv(RAIZ / "results/test_por_serie.csv.gz")
    t = test[(test.mercado == "reversion") & test.agente.isin(["dqn", "a3c"])]
    med = t.groupby(["agente", "version"]).retorno_pct.median()
    fig, ax = plt.subplots(figsize=(7.8, 4.1))
    x = np.arange(2)
    for offset, version, etiqueta, color in [(-.19, "test", "Mejor control (validación)", VERDE),
                                            (.19, "test_ultimo", "Último control", NARANJA)]:
        valores = [med[a, version] for a in ("dqn", "a3c")]
        bars = ax.bar(x+offset, valores, .34, label=etiqueta, color="#"+color)
        ax.bar_label(bars, labels=[f"{v:.1f}%".replace(".", ",") for v in valores], padding=5, fontsize=17)
    ax.set(xticks=x, xticklabels=["DQN", "A3C"], ylim=(0, 36), ylabel="Rentabilidad mediana en test (%)")
    ax.legend(loc="upper left", fontsize=11, frameon=False)
    ax.grid(axis="y", alpha=.15)
    ax.set_axisbelow(True)
    fig.tight_layout()
    archivos["deterioro"] = guardar(fig, "mejor_ultimo")

    d = curvas_carpeta("results/estabilidad_dqn_reversion/489842ba036f")
    fig, ax = plt.subplots(figsize=(7.4, 4.25))
    for variante, etiqueta, color in [("original", "DQN original", GRIS), ("solo_alpha", "Aprendizaje decreciente", VERDE)]:
        v = d[d.variante == variante].sort_values("seed")
        ax.plot(v.seed.astype(str), v.retorno, "o-", color="#"+color, lw=2.5, ms=7, label=etiqueta)
    ax.set(ylim=(-1, 30), xlabel="Semilla nueva de entrenamiento", ylabel="Rentabilidad final en validación (%)")
    ax.legend(loc="upper left", frameon=False, fontsize=12)
    ax.grid(axis="y", alpha=.18)
    fig.tight_layout()
    archivos["dqn"] = guardar(fig, "confirmacion_dqn")

    a = curvas_carpeta("results/estabilidad_a3c_reversion/b524af4c8dbb").groupby("variante").mean(numeric_only=True)
    fig, ejes = plt.subplots(1, 2, figsize=(10.8, 3.5))
    claves = ["original", "alpha_10", "alpha_30"]
    for ax, campo, titulo in zip(ejes, ["G", "oscilacion"], ["Ganancia G del tramo final ↑", "Oscilación temporal ↓"]):
        v = a.loc[claves, campo]
        bars = ax.bar(["Constante", "Final al 10%", "Final al 30%"], v, color=["#"+AZUL, "#"+NARANJA, "#"+VERDE], width=.6)
        ax.bar_label(bars, labels=[f"{z:.3f}".replace(".", ",") for z in v], padding=5, fontsize=15)
        ax.set_ylim(0, v.max()*1.3)
        ax.set_title(titulo, fontsize=15, loc="left")
        ax.grid(axis="y", alpha=.15)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", labelsize=12)
    fig.tight_layout(w_pad=2)
    archivos["a3c"] = guardar(fig, "calendarios_a3c")

    r = pd.read_csv(RAIZ / "results/ajuste_robusto/e48237edf3e5/comparacion.csv")
    fig, ax = plt.subplots(figsize=(7.3, 4.1))
    etiquetas, medias, bajos, altos = [], [], [], []
    for familia, candidato, etiqueta in [("a3c", "5bfee635b8", "A3C · 800 ep., MAE"),
        ("a3c", "57c270c084", "A3C · 800 ep., MSE"),
        ("dqn", "9a2375decf", "DQN · α=0,0001; γ=0"),
        ("dqn", "36248cdc07", "DQN · pérdida MAE")]:
        row = r[(r.familia == familia) & (r.candidato == candidato)].iloc[0]
        etiquetas.append(etiqueta); medias.append(row.diferencia_pp)
        bajos.append(row.intervalo_bajo); altos.append(row.intervalo_alto)
    medias, bajos, altos = map(np.array, (medias, bajos, altos))
    ax.errorbar(medias, np.arange(4), xerr=[medias-bajos, altos-medias], fmt="o", color="#"+AZUL,
                ecolor="#"+VERDE, capsize=6, lw=2, ms=7)
    ax.axvline(0, color="#"+NARANJA, ls="--", lw=1.5)
    ax.set(yticks=np.arange(4), yticklabels=etiquetas, xlabel="Ventaja frente a la configuración inicial (pp)")
    ax.invert_yaxis(); ax.grid(axis="x", alpha=.15)
    fig.tight_layout()
    archivos["ajuste"] = guardar(fig, "incertidumbre_ajuste")
    return archivos


def rgb(c):
    return RGBColor.from_string(c)


def crear_presentacion():
    grafs = graficos()
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
    prs.core_properties.title = "DQN y A3C para invertir en un activo"
    prs.core_properties.author = "Josep Peréz Segura; Emilio Sanchez Martínez; Natalia San José Ortega"
    guion = []

    def texto(s, x, y, w, h, contenido, size=22, color=AZUL, bold=False, align=PP_ALIGN.LEFT):
        caja = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
        tf = caja.text_frame
        tf.clear(); tf.word_wrap = True
        tf.margin_left = tf.margin_right = Inches(.02)
        tf.margin_top = tf.margin_bottom = Inches(.02)
        for i, linea in enumerate(contenido.split("\n")):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            p.text = linea; p.alignment = align
            p.font.name = "Arial"; p.font.size = Pt(size); p.font.bold = bold; p.font.color.rgb = rgb(color)
            p.space_after = Pt(7); p.line_spacing = 1.08
        return caja

    def rect(s, x, y, w, h, color, redondo=False):
        sh = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE if redondo else MSO_SHAPE.RECTANGLE,
                               Inches(x), Inches(y), Inches(w), Inches(h))
        sh.fill.solid(); sh.fill.fore_color.rgb = rgb(color); sh.line.fill.background()
        if redondo:
            sh.adjustments[0] = .08
        return sh

    def tarjeta(s, x, y, w, h, titulo, cuerpo, color=VERDE):
        rect(s, x, y, w, h, "FFFFFF", True)
        rect(s, x, y, .06, h, color)
        compacta = h < 1.4
        texto(s, x+.22, y+.12, w-.45, .5, titulo, 26 if compacta else 23, color, True)
        inicio = .65 if compacta else (.73 if h <= 1.8 else .85)
        texto(s, x+.22, y+inicio, w-.45, h-inicio-.05, cuerpo,
              14 if compacta else (18 if h <= 1.8 else 20))

    def imagen(s, ruta, x, y, w, h):
        from PIL import Image
        with Image.open(ruta) as img:
            iw, ih = img.size
        escala = min(w/iw, h/ih)
        ww, hh = iw*escala, ih*escala
        s.shapes.add_picture(str(ruta), Inches(x+(w-ww)/2), Inches(y+(h-hh)/2), width=Inches(ww), height=Inches(hh))

    def diapositiva(titulo, etapa, segundos, notas, fuente="Proyecto y report/informe.pdf"):
        s = prs.slides.add_slide(prs.slide_layouts[6])
        s.background.fill.solid(); s.background.fill.fore_color.rgb = rgb(FONDO)
        texto(s, .55, .28, 7, .28, "TALLER 2  /  APRENDIZAJE POR REFUERZO", 11, GRIS, True)
        texto(s, 8.5, .28, 4.25, .3, etapa, 11, VERDE, True, PP_ALIGN.RIGHT)
        texto(s, .55, .82, 12.2, .86, titulo, 30, AZUL, True)
        rect(s, .55, 7.02, 12.2, .012, "CCD6DE")
        texto(s, .55, 7.12, 10.5, .22, "DQN y A3C · Resultados del proyecto", 10, GRIS)
        texto(s, 11.5, 7.1, 1.25, .25, f"{len(prs.slides):02d} / 12", 10, GRIS, align=PP_ALIGN.RIGHT)
        s.notes_slide.notes_text_frame.text = f"TIEMPO ORIENTATIVO: {segundos} segundos\n\n{notas}\n\nFUENTES: {fuente}"
        guion.append(dict(titulo=titulo, segundos=segundos, notas=notas, fuente=fuente))
        return s

    s = diapositiva("", "", 20,
        "Somos Josep Peréz Segura, Emilio Sanchez Martínez y Natalia San José Ortega. Presentamos un agente que decide cuándo comprar, mantener o vender un único activo. Además de comparar DQN y A3C, hemos estudiado una pregunta práctica: cómo evitar que un agente pierda al final del entrenamiento lo que ya había aprendido.")
    rect(s, 0, 0, 13.333, 7.5, AZUL)
    rect(s, .65, 1.03, .15, 3.05, VERDE)
    texto(s, 1.02, 1.1, 11.1, 1.85, "Aprender a invertir\ncon DQN y A3C", 43, "FFFFFF", True)
    texto(s, 1.02, 3.32, 10.9, .85, "Resultados, estabilidad y selección de hiperparámetros", 25, "BDE5DF")
    texto(s, 1.02, 4.9, 11.1, 1.4, "Josep Peréz Segura\nEmilio Sanchez Martínez\nNatalia San José Ortega", 20, "FFFFFF")
    texto(s, 1.02, 6.48, 11, .4, "Taller 2 · 5 de noviembre de 2026", 15, "C4D0DA")

    s = diapositiva("¿Cuándo conviene estar dentro del mercado?", "EL PROBLEMA", 45,
        "El agente empieza con mil unidades de capital. Solo observa la tendencia calculada con los últimos seis precios y si tiene o no el activo. Decide vender, mantener o comprar. Al comprar invierte el efectivo disponible y al vender cierra la posición. Cada operación efectiva tiene una comisión del 0,1%. La recompensa recoge el retorno del siguiente paso si está invertido, menos la penalización de operar. Por eso no buscamos que opere mucho: puede aprender que mantenerse fuera es mejor. Conservamos el entorno del enunciado y dejamos que la red reciba la tendencia continua, sin convertirla en intervalos.", "rl_trader/env.py; report/informe.md, sección 1")
    tarjeta(s,.6,2,3.7,2.8,"1 · Observar","Tendencia de 6 precios\nPosición: dentro / fuera")
    texto(s,4.4,3, .4,.6,"→",28,VERDE,True)
    tarjeta(s,4.9,2,3.55,2.8,"2 · Decidir","Vender\nMantener\nComprar")
    texto(s,8.55,3,.4,.6,"→",28,VERDE,True)
    tarjeta(s,9.05,2,3.65,2.8,"3 · Aprender","Recompensa del paso\nComisión: 0,1%")
    rect(s,.65,5.35,12,.85,CLARO,True)
    texto(s,.9,5.52,11.5,.5,"Operar más no implica ganar más: también se puede aprender a no entrar.",22,AZUL,True)

    s = diapositiva("Cuatro mercados, cuatro comportamientos", "ENTORNO SIMULADO", 45,
        "Probamos los cuatro generadores del taller. En el ciclo esperamos que seguir la tendencia ayude. En el browniano hay poca señal aprovechable. En reversión, las caídas extremas pueden anticipar un rebote hacia la media. En regímenes alternan fases alcistas y bajistas, pero el agente no observa directamente el régimen. Estos gráficos son ejemplos generados con el código del proyecto, no precios reales. Todos parten de un índice cien; el ciclo usa escala logarítmica y los ejes no son comparables entre paneles. Esta diversidad permite comprobar si el agente aprende una respuesta coherente con cada mercado.", "rl_trader/prices.py; series ilustrativas, semilla 12, 500 pasos")
    imagen(s,grafs["mercados"],.7,1.8,11.9,4.5)
    texto(s,.8,6.45,11.8,.35,"Ejemplos simulados · Precio inicial = 100 · Ciclo en escala logarítmica",15,GRIS)

    s = diapositiva("DQN valora acciones; A3C aprende una política", "ALGORITMOS", 50,
        "DQN estima cuánto vale cada acción con una red neuronal. Guarda experiencias y entrena con lotes aleatorios; utiliza otra red como referencia para calcular los valores futuros. A3C separa el actor, que decide, del crítico, que estima el valor del estado. Cuatro trabajadores recorren series distintas y actualizan pesos compartidos. Encontramos una cuestión importante de implementación: el estado de Adam podía quedar inconsistente si varios procesos lo actualizaban a la vez. Un cerrojo solo en ese paso corrigió el problema y el resto siguió siendo asíncrono. También incluimos Doble DQN, DQN en duelo y REINFORCE como comparaciones adicionales.", "rl_trader/dqn.py; rl_trader/a3c.py; results/a3c_sin_cerrojo.csv")
    tarjeta(s,.65,1.95,5.85,3.25,"DQN · aprender el valor","Red Q: una salida por acción\nMemoria de experiencias\nRed objetivo cada 1.000 pasos",AZUL)
    tarjeta(s,6.82,1.95,5.85,3.25,"A3C · actor y crítico","Actor: qué acción elegir\nCrítico: cuánto esperar\n4 trabajadores asíncronos",VERDE)
    texto(s,.85,5.65,11.6,.55,"A3C: proteger el paso de Adam evitó confundir un fallo con falta de convergencia.",21,AZUL,True)
    texto(s,.85,6.4,11.6,.3,"Comparaciones extra: Doble DQN · DQN en duelo · REINFORCE",16,GRIS)

    s = diapositiva("Separar aprender, elegir y evaluar", "PROTOCOLO", 45,
        "Cada episodio de entrenamiento ve una serie nueva del mismo mercado. Ajustamos aprendizaje, descuento, exploración, pérdida y número de episodios usando validación. El ajuste inicial usa tres semillas y cien series. El test principal usa otras cien series y diez semillas nuevas de entrenamiento; las series de test no eligen hiperparámetros. Utilizamos dos métricas diferentes. La rentabilidad final mide la variación del capital. G es la suma sin descuento de las recompensas del episodio y no equivale al porcentaje de rentabilidad compuesta. Mirar ambas ayuda a distinguir rendimiento económico y estabilidad del aprendizaje. Las ampliaciones nuevas se identifican como validación o test reservado.", "scripts/ajuste.py; rl_trader/evaluation.py; results/entrenamientos.csv")
    tarjeta(s,.65,2,3.85,2.6,"Entrenar","Series nuevas\n500 precios por episodio")
    tarjeta(s,4.75,2,3.85,2.6,"Elegir en validación","100 series comunes\nAjuste inicial: 3 semillas")
    tarjeta(s,8.85,2,3.85,2.6,"Evaluar en test","100 series distintas\n10 semillas de entrenamiento")
    texto(s,.9,5.15,5.8,.45,"Rentabilidad = cambio del capital",22,AZUL,True)
    texto(s,.9,5.95,11.4,.65,"G = suma de recompensas del episodio\nG = 0,25 no implica una rentabilidad del 25%.",21,VERDE)

    s = diapositiva("En reversión, los métodos de actor rinden más", "TEST PRINCIPAL", 55,
        "Este gráfico muestra la rentabilidad mediana en el test principal de reversión. DQN obtiene aproximadamente un 22%, A3C un 27,5% y REINFORCE un 28,6%. Las referencias ayudan a interpretar la mejora: comprar y mantener queda ligeramente negativo y el agente aleatorio pierde más. Las variantes de DQN no cambian mucho el resultado. En los otros mercados la lectura es distinta: en ciclo los agentes siguen la tendencia, en browniano predominan no operar o comprar y mantener, y en regímenes se quedan fuera. Los porcentajes enormes del ciclo son una propiedad del generador sintético. No deben confundirse con una expectativa de rentabilidad en mercados reales.", "results/tabla.csv; mediana agregada en test, 100 series, 10 semillas por agente RL")
    imagen(s,grafs["principal"],.6,1.8,8.7,4.9)
    tarjeta(s,9.55,2.05,3.05,1.8,"27,5%","A3C en reversión")
    texto(s,9.7,4.2,2.85,2.1,"Ciclo: seguir tendencia\nBrowniano: poca señal\nRegímenes: no entrar",18,AZUL)

    s = diapositiva("El mejor control no siempre es el último", "TEST PRINCIPAL · REVERSIÓN", 45,
        "Aquí aparece el problema que motivó las mejoras. Con los mismos entrenamientos, evaluar el último control en vez del mejor elegido en validación reduce mucho la rentabilidad: DQN pasa del 22,1 al 2,6%, y A3C del 27,5 al 15,9%. Alcanzar un buen punto no significa conservarlo. Incluso con 12.800 episodios A3C seguía oscilando en reversión. Por eso probamos cambios en el aprendizaje y en la red objetivo. Las pruebas nuevas se realizan en validación, con comparación entre semillas, sin utilizar estos resultados de test para declarar que una alternativa ya generaliza.", "results/test_por_serie.csv.gz; versiones test y test_ultimo; report/informe.md")
    imagen(s,grafs["deterioro"],.75,1.8,8.6,4.8)
    tarjeta(s,9.6,2.2,3.0,3.4,"La pregunta","¿Cómo conservar\nlo aprendido\nal terminar?",NARANJA)

    s = diapositiva("DQN conserva mejor el rendimiento al final", "VALIDACIÓN · 5 SEMILLAS NUEVAS", 75,
        "Primero comparamos DQN original con tres variantes: reducir la tasa de aprendizaje en la segunda mitad y combinar esa reducción con una red objetivo más rápida o más lenta. La opción más sencilla, reducir solo el aprendizaje hasta el 30%, fue prometedora. La repetimos frente al original con cinco semillas nuevas, manteniendo cuatrocientos episodios y la copia objetivo cada mil pasos. En las cinco semillas mejoraron tanto G del tramo final como la rentabilidad del último control. La media de esta última pasó del 6,28 al 18,13%; la oscilación bajó un 47% y la diferencia entre mejor y último control un 68%. El tiempo fue similar, unos 42 segundos. Es una confirmación exploratoria en validación: el test independiente de esta mejora todavía está pendiente.", "results/estabilidad_dqn_reversion/489842ba036f; cinco semillas 360–364")
    texto(s,.8,1.72,11.7,.45,"400 episodios · α: 0,002 → 0,0006 desde la mitad · Objetivo: cada 1.000 pasos",18,GRIS)
    imagen(s,grafs["dqn"],.6,2.25,8.5,4.25)
    tarjeta(s,9.4,2.2,3.2,1.18,"5 de 5","Semillas favorables")
    tarjeta(s,9.4,3.65,3.2,1.18,"−47%","Oscilación temporal")
    tarjeta(s,9.4,5.1,3.2,1.18,"−68%","Brecha mejor–último")
    texto(s,.85,6.65,11.6,.26,"Confirmado en validación · Pendiente: test independiente",14,NARANJA,True)

    s = diapositiva("A3C: menos oscilación no basta para mejorar", "VALIDACIÓN · 3 SEMILLAS", 65,
        "En A3C estudiamos decaimiento del aprendizaje y de la entropía. Con 12.800 episodios algunas variantes parecían favorables, pero en el presupuesto de 3.200 episodios el resultado dependía del calendario y de la ejecución. Esta comparación muestra el compromiso: terminar con una tasa del 10% reduce mucho la oscilación, pero también la ganancia. Con el 30% la ganancia sigue algo por debajo del original. Ninguna alternativa cumplió a la vez conservar G y rentabilidad y reducir la oscilación. Conservamos A3C original, lo repetimos con cinco semillas nuevas y solo ese modelo llegó al test reservado. Allí obtuvo 26,67% con el mejor control y 16,17% con el último. Ese test no compara calendarios alternativos.", "results/estabilidad_a3c_reversion/b524af4c8dbb; test reservado: 1e9c1784f2f4/decision_66330b4a66cf_test.csv")
    imagen(s,grafs["a3c"],.7,1.9,11.9,3.8)
    rect(s,.8,5.9,11.7,.72,CLARO,True)
    texto(s,1.05,6.02,11.2,.47,"Decisión: mantener A3C original. Test reservado: 26,67% mejor / 16,17% último.",19,AZUL,True)

    s = diapositiva("Más semillas evitan elegir por una buena racha", "VALIDACIÓN · HIPERPARÁMETROS", 65,
        "Para hacer más fiable el ajuste no repetimos todas las combinaciones. Seleccionamos la configuración inicial y las dos alternativas más prometedoras para DQN y A3C en reversión. Son seis configuraciones por cinco semillas nuevas: treinta entrenamientos. El gráfico muestra diferencias pareadas frente a cada configuración inicial. La línea vertical es cero: si el intervalo la cruza, estos ensayos no separan claramente una ventaja. A3C mantiene aproximadamente un 30,58% y poca dispersión. En DQN una alternativa obtiene mayor media, pero su intervalo incluye cero. Por el criterio fijado conservamos ambas configuraciones. Son intervalos orientativos con pocas semillas, sin corrección por comparaciones múltiples; no una prueba de superioridad o de equivalencia.", "results/ajuste_robusto/e48237edf3e5/comparacion.csv; propuestas.csv")
    texto(s,.8,1.72,11.8,.4,"6 configuraciones × 5 semillas nuevas = 30 entrenamientos",22,VERDE,True)
    imagen(s,grafs["ajuste"],.65,2.35,8.8,4.15)
    tarjeta(s,9.65,2.55,2.95,1.6,"A3C","Elección inicial estable")
    tarjeta(s,9.65,4.45,2.95,1.6,"DQN","Ventaja aún incierta",NARANJA)
    texto(s,.85,6.65,11.7,.26,"Intervalos bootstrap orientativos: incluyen la incertidumbre entre semillas, no nuevos mercados.",13,GRIS)

    s = diapositiva("Las pruebas extra también revelan límites", "OTRAS PRUEBAS", 45,
        "Las pruebas complementarias ayudan a entender qué aporta cada decisión. MSE frente a MAE no mostró una ventaja clara con tres semillas. En DQN sin buffer, una pasada por episodio dio un 24,4% en reversión, pero repetir dieciséis veces el mismo episodio cayó al 10,8%, compatible con sobreajustar experiencias correlacionadas. En datos reales ninguno superó a comprar y mantener: en el S&P 500, A3C obtuvo cerca del 190%, frente al 193% de la referencia; DQN ganó menos, pero redujo la caída máxima. Solo hay una historia de test por activo. Por tanto, el buen resultado en simulación no demuestra una ventaja aplicable a cualquier mercado.", "results/lotes_test.csv.gz; results/reales_resultados.csv; report/informe.md, sección 3")
    tarjeta(s,.65,2,3.85,3.6,"Función de pérdida","MSE frente a MAE\nSin ventaja clara\ncon 3 semillas")
    tarjeta(s,4.75,2,3.85,3.6,"Repetir experiencias","1 pasada: 24,4%\n16 pasadas: 10,8%\nTest de reversión",AZUL)
    tarjeta(s,8.85,2,3.85,3.6,"Datos reales","S&P 500:\nA3C ≈190%\nBuy & hold ≈193%",NARANJA)
    texto(s,.9,6.1,11.5,.5,"Aprender en simulación no garantiza superar una estrategia simple en datos reales.",21,AZUL,True)

    s = diapositiva("Conclusiones", "LO QUE NOS LLEVAMOS", 45,
        "Nos quedamos con cuatro ideas. Primero, DQN y A3C aprenden estrategias coherentes con la información que observan, pero esa información limita el resultado. Segundo, reducir el aprendizaje al final mejora la estabilidad de DQN en validación y lo hemos confirmado con cinco semillas nuevas. Tercero, una curva más suave no basta: en A3C no aceptamos cambios que sacrificaban rendimiento. Cuarto, repetir pocas configuraciones con más semillas ayuda a reconocer cuándo la selección de hiperparámetros sigue siendo incierta. El paso pendiente más importante es evaluar la mejora confirmada de DQN en un test nuevo reservado. Conservar la separación entre ajuste y test es esencial para que la conclusión sea creíble.", "report/informe.pdf; resultados y limitaciones de los experimentos")
    tarjeta(s,.7,1.95,5.85,1.65,"01 · Aprender tiene límites","La información del estado marca el techo.")
    tarjeta(s,6.8,1.95,5.85,1.65,"02 · DQN conserva lo aprendido","Mejora confirmada en validación.")
    tarjeta(s,.7,3.9,5.85,1.65,"03 · Estabilidad ≠ rendimiento","A3C exige comprobar ambas métricas.",NARANJA)
    tarjeta(s,6.8,3.9,5.85,1.65,"04 · Decidir con más evidencia","Las semillas revelan la incertidumbre.")
    texto(s,.95,6.05,11.5,.55,"Siguiente paso: test independiente de DQN con aprendizaje decreciente.",22,VERDE,True)

    assert len(prs.slides) == 12
    assert sum(g["segundos"] for g in guion) == 600
    ruta = SALIDA / "presentacion_proyecto.pptx"
    prs.save(ruta)
    lineas = ["# Guion de presentación · 10 minutos", "", "Las notas también están incluidas en PowerPoint.", ""]
    acumulado = 0
    for i, g in enumerate(guion, 1):
        inicio = acumulado; acumulado += g["segundos"]
        lineas += [f"## {i}. {g['titulo'] or 'Portada'}", "",
                   f"Tiempo: {g['segundos']} s · tramo {inicio//60}:{inicio%60:02d}–{acumulado//60}:{acumulado%60:02d}", "",
                   g["notas"], "", "Fuente: " + g["fuente"], ""]
    (SALIDA / "guion_presentacion_10min.md").write_text("\n".join(lineas), encoding="utf-8")
    (RECURSOS / "estructura.json").write_text(json.dumps(guion, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Generada {ruta}: 12 diapositivas, 600 segundos de guion.")


if __name__ == "__main__":
    crear_presentacion()
