"""Prueba secundaria con precios reales de índices y fondos, descargados de Yahoo Finance.

De cada activo solo hay una historia, así que el protocolo cambia respecto a los mercados
simulados:

- Entrenamiento (hasta 2015): cada episodio usa un tramo de 500 días elegido al azar, para
  cumplir con que cada episodio vea una serie distinta. Los tramos solo reparten los
  episodios: el estado que ve la red sigue siendo la tendencia continua, sin intervalos.
- Validación (2016-2018): elige los pesos del mejor control.
- Test (desde 2019): se recorre de una sola vez, y solo se usa para el resultado final.

Los precios se guardan en data/, que no se sube al repositorio.
"""
from pathlib import Path

import numpy as np
import pandas as pd

DATOS = Path(__file__).resolve().parent.parent / "data"

# Dos índices y dos fondos. Los índices son de precio, sin dividendos; los fondos
# incluyen los dividendos en su valor liquidativo ajustado.
ACTIVOS = {
    "sp500": dict(ticker="^GSPC", nombre="S&P 500 (índice)"),
    "ibex": dict(ticker="^IBEX", nombre="IBEX 35 (índice)"),
    "mixto": dict(ticker="VWELX", nombre="Vanguard Wellington (fondo mixto)"),
    "bonos": dict(ticker="VBMFX", nombre="Vanguard Total Bond (fondo de bonos)"),
}
INICIO = "2005-01-01"
FIN_ENTRENAMIENTO = "2015-12-31"
INICIO_VALIDACION, FIN_VALIDACION = "2016-01-01", "2018-12-31"
INICIO_TEST = "2019-01-01"
N_STEPS_VENTANA = 500


def cargar(activo):
    """Devuelve el precio de cierre ajustado de un activo, indexado por fecha. Lo descarga la primera vez."""
    ruta = DATOS / f"{activo}.csv"
    if not ruta.exists():
        import yfinance as yf
        DATOS.mkdir(exist_ok=True)
        serie = yf.Ticker(ACTIVOS[activo]["ticker"]).history(start=INICIO, auto_adjust=True)["Close"].dropna()
        serie.index = serie.index.tz_localize(None)
        serie.rename("precio").to_csv(ruta)
    return pd.read_csv(ruta, index_col=0, parse_dates=True)["precio"]


def dividir(serie):
    """Separa la serie en tres tramos sin solapar: (entrenamiento, validación, test)."""
    return (serie[:FIN_ENTRENAMIENTO].to_numpy(), serie[INICIO_VALIDACION:FIN_VALIDACION].to_numpy(),
            serie[INICIO_TEST:].to_numpy())


class GeneradorTramos:
    """Hace de generador de precios con datos reales: cada llamada devuelve un tramo al azar.

    Usa el azar global de NumPy, igual que los generadores del enunciado, así que los
    entrenamientos son reproducibles con su semilla.
    """

    def __init__(self, precios):
        self.precios = np.asarray(precios, dtype=float)

    def __call__(self, n_steps=N_STEPS_VENTANA):
        inicio = np.random.randint(0, len(self.precios) - n_steps + 1)
        return self.precios[inicio:inicio + n_steps]
