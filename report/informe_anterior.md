---
# Para generar el PDF, desde la raíz del repositorio (máximo 2 páginas):
#   python -c "import pypandoc; pypandoc.convert_file('report/informe.md', 'pdf', outputfile='report/informe.pdf', extra_args=['--pdf-engine=tectonic', '--resource-path=.'])"
title-meta: "DQN y A3C para invertir en un activo"
author-meta: "[Autores]"
lang: es
fontsize: 9pt
documentclass: extarticle
geometry: "a4paper, top=1.1cm, bottom=1.1cm, left=1.4cm, right=1.4cm"
colorlinks: true
header-includes:
  - \usepackage{float}
  - \floatplacement{figure}{H}
  - \setlength{\parskip}{1.5pt}
  - \setlength{\parindent}{0pt}
  - \pagenumbering{gobble}
  - \usepackage{titlesec}
  - \titlespacing*{\section}{0pt}{5pt}{2pt}
  - \titleformat{\section}{\normalsize\bfseries}{}{0pt}{}
  - \usepackage{enumitem}
  - \setlist{nosep, leftmargin=12pt}
  - \setlength{\tabcolsep}{4pt}
---

\begin{center}
{\large\bfseries DQN y A3C para invertir en un activo: cuatro mercados simulados y datos reales}\par\vspace{2pt}
{\small [Autores] · Taller 2 de Aprendizaje por Refuerzo · noviembre de 2026}
\end{center}

**Resumen.** Entrenamos DQN y A3C, además de Doble DQN, DQN en duelo y REINFORCE, en el entorno del enunciado sin modificarlo. Los comparamos con el agente aleatorio, *buy & hold* (B&H) y dos reglas fijas en las **mismas 100 series de test del taller anterior**, lo que permite comparar también con sus agentes tabulares. Todos aprenden la estrategia esperada en cada mercado. En ciclo empatan entre sí y con el tabular; en reversión, los métodos de actor (A3C, REINFORCE) superan a DQN y al tabular. Además, encontramos y corregimos un fallo de concurrencia que hacía parecer que A3C no convergía.

# 1. Diseño

- **Estado continuo, sin intervalos.** La red recibe $(\text{tendencia}, \text{posición})$ y generaliza entre tendencias parecidas. La tendencia se divide por su desviación típica, porque vale $\pm0{,}3$ en ciclo y $\pm0{,}05$ en el resto. La recompensa también se normaliza, en vez de recortarla a $\pm1$ como proponen las transparencias: recortar borraría la diferencia entre retornos pequeños y grandes.
- **DQN:** red $64\times64$, buffer de $10^5$ transiciones, minibatch de 64 cada 4 pasos y red objetivo copiada cada 1.000 pasos. **A3C:** 4 procesos asíncronos con Adam compartido, actor y crítico separados ($\alpha_w=3\alpha_\theta$) y entropía $\beta$ como exploración. Omitimos el factor $I=\gamma^t$: optimiza solo el estado inicial, que aquí no tiene nada de especial. **REINFORCE** es el caso de un solo trabajador que actualiza al final del episodio, sin crítico.
- **Datos y métricas.** Cada episodio usa una serie nueva de 500 pasos. Validación y test (100 series cada uno) usan semillas separadas. Medimos la mediana del retorno en test, la ganancia $G=\sum r$ y el cociente geométrico del capital frente a cada referencia, con IC bootstrap pareado. Los pesos son los del mejor control en validación.
- **Ajuste** por etapas, solo en validación (3 semillas × 100 series): (A) $\alpha\times\gamma$; (B) exploración: en DQN, $\varepsilon$ decae de 1 a 0,01 o 0,1 en el 30% o el 70% de los episodios (gana el 70% en todos los mercados), y en A3C, $\beta_0$ y $n$ pasos; (C) pérdida MSE o MAE; (D) episodios. La evaluación final usa 10 semillas nuevas.

\begin{center}\footnotesize
\begin{tabular}{l|lllll|lllll|llll}
\hline
 & \multicolumn{5}{c|}{DQN} & \multicolumn{5}{c|}{A3C (crítico con MAE)} & \multicolumn{4}{c}{REINFORCE} \\
Mercado & $\alpha$ & $\gamma$ & $\varepsilon_f$ & pérdida & ep. & $\alpha_\theta$ & $\gamma$ & $\beta_0$ & $n$ & ep. & $\alpha$ & $\gamma$ & $\beta_0$ & ep. \\
\hline
Ciclo & 2e-3 & 0 & 0,01 & MAE & 400 & 3e-4 & 0 & 0,01 & 20 & 1.600 & 1e-3 & 0,5 & 0,01 & 800 \\
Browniano & 5e-4 & 0,99 & 0,01 & MSE & 400 & 3e-4 & 0,99 & 0,01 & 20 & 800 & 1e-3 & 0,99 & 0,01 & 800 \\
Reversión & 2e-3 & 0,5 & 0,01 & MSE & 400 & 3e-4 & 0,9 & 0,2 & 20 & 3.200 & 1e-3 & 0,9 & 0,05 & 1.600 \\
Regímenes & 5e-4 & 0,5 & 0,1 & MSE & 400 & 1e-4 & 0 & 0,01 & 5 & 800 & 3e-4 & 0,5 & 0,01 & 800 \\
\hline
\end{tabular}
\end{center}

**Por qué.** En ciclo, $\gamma=0$, como en el tabular: la inercia se ve en el paso siguiente. En reversión, $\gamma$ alto, porque hay que aguantar la posición hasta que el precio vuelve a la media. En browniano y regímenes, casi todo acaba sin operar, y la variación entre configuraciones es del tamaño del error de 3 semillas (DQN en reversión: 2,3 frente a 2,1 puntos), así que esa elección es casi azar. A3C necesita más episodios que DQN, pero cada uno le cuesta 3 veces menos (41 frente a 131 ms).

# 2. Resultados en test

\begin{center}\small
\begin{tabular}{lrrrrl}
\hline
Agente & Ciclo & Browniano & Reversión & Regímenes & Convergencia en $G$, ciclo / reversión \\
\hline
Aleatorio & +456 (3,11) & $-$14,7 & $-$15,2 & $-$24,0 & \\
Buy \& hold & +3.425 (6,38) & +0,7 & $-$1,1 & $-$20,4 & \\
Momentum & +48.189 (7,71) & $-$11,8 & $-$28,7 & $-$18,4 & \\
Contrarian & $-$92,9 & $-$14,3 & +6,9 & $-$22,6 & \\
DQN & +59.363 (8,13) & 0 & +22,1 (0,25) & 0 & 50 / 375 ep. $\approx$ 7 / 49 s \\
Doble DQN & +59.363 (8,13) & 0 & +21,0 (0,24) & 0 & 50 / 350 ep. $\approx$ 7 / 47 s \\
DQN en duelo & +60.723 (8,18) & 0 & +22,2 (0,25) & 0 & 50 / 375 ep. $\approx$ 8 / 61 s \\
\textbf{A3C} & \textbf{+63.619} (8,21) & 0 & \textbf{+27,5} (0,30) & 0 & 50 / 3.200 ep. $\approx$ 2 / 130 s \\
REINFORCE & +59.905 (8,24) & 0 & +28,6 (0,30) & 0 & 75 / 1.350 ep. $\approx$ 3 / 48 s \\
\hline
\textit{Taller ant.:} Q-learning & +64.989 & 0 & +22,0 & +1,7 & 2.600 / 3.300 ep.$^\dagger$ $\approx$ 24 / 30 s \\
\textit{Taller ant.:} Sarsa & +67.861 & 0 & +22,2 & +4,7 & 600 / 3.700 ep.$^\dagger$ $\approx$ 4 / 29 s \\
Oráculo (techo) & +1.193.433 (11,16) & +0,7 (0,09) & +77,3 (0,63) & +26,2 (0,26) & \\
\hline
\end{tabular}

\footnotesize Mediana del retorno (\%) en test, con $G$ media entre paréntesis; 100 series (× 10 semillas en los agentes entrenados).\\ $^\dagger$Criterio del taller anterior, sobre la mediana en \%. Con ese mismo criterio: DQN, 400 / 375 episodios; A3C, 1.425 / 3.200.
\end{center}

- **Significación.** Todos los agentes superan a B&H y al aleatorio con IC del 95% salvo en browniano, donde ninguno se distingue de B&H. En ciclo superan también a momentum, la mejor regla fija, por un factor de 1,25 a 1,35. Doble DQN coincide exactamente con DQN en ciclo: con $\gamma=0$ el target es solo $r$ y no hay `max` que sobreestimar, lo que sirve de comprobación.
- **¿Es la estrategia esperada?** Sí, según los mapas de la política sobre la tendencia continua:
  - **Ciclo:** *momentum* con histéresis. Compra por encima de $\approx+0{,}5\sigma$ y no vende hasta bajar de $\approx-0{,}7\sigma$, lo que ahorra comisiones.
  - **Reversión:** *contrarian* solo en los extremos. Compra por debajo de $\approx-1{,}5\sigma$ y mantiene hasta pasar de $+1$ a $+1{,}5\sigma$, que es donde el rebote compensa la comisión.
  - **Browniano:** no operar o B&H, equivalentes en un mercado sin señal.
  - **Regímenes:** no operar. La tendencia de 6 pasos no distingue los regímenes, y por eso el oráculo, que sí conoce el régimen, gana +26%.
- **¿Tiene sentido?** Los agentes capturan el 74% de la $G$ del oráculo en ciclo y el 40–48% en reversión: siempre por debajo del techo y por encima de la mejor regla fija (69% y 18%). La $G$ en validación y en test casi coincide (A3C en reversión: 0,30 y 0,30), así que la elección del mejor control apenas infla el resultado.
- **Frente al tabular.** En rendimiento, empate en ciclo, mejora en reversión con los métodos de actor y pérdida del pequeño margen en regímenes: con una sola variable continua, 8–40 intervalos ya aproximaban bien la frontera. En convergencia, con el mismo criterio, DQN generaliza y en reversión necesita unas 10 veces menos episodios (375 frente a 3.300–3.700), pero cada uno cuesta 13 veces más y tarda algo más (49 frente a 29–30 s); A3C necesita tantos episodios como el tabular.

![Política aprendida: fracción de las 10 semillas que quieren estar dentro del mercado, según la tendencia normalizada y la posición.](figures/politicas.png){width=52%}

# 3. Convergencia de A3C

**Un fallo que parecía falta de convergencia.** Con la misma configuración y semilla, A3C a veces aprendía y a veces dejaba de operar para siempre. Lo destapó un test intermitente. Los procesos compartían el estado de Adam sin cerrojo: uno podía leer la media $m$ recién escrita por otro con la varianza $v$ aún a cero, y el paso $\alpha\,m/(\sqrt v+10^{-8})$ llevaba los pesos de 0,7 a más de 1.000 en una sola actualización. Con un cerrojo solo en ese paso, los gradientes se siguen calculando de forma asíncrona y los pesos se quedan en 0,71 en 30 de 30 pruebas. Al rehacer A3C:

- en reversión, la peor semilla pasa de +5,3% a +25,2% y la mediana, de +23,6% a +27,5%;
- los ajustes que no operaban bajan del 10% al 4%;
- el resultado ya no depende del número de trabajadores (1, 2, 4 y 8 tardan 57, 33, 23 y 21 s), aunque el cerrojo cuesta algo: sin él, 8 trabajadores tardaban 13–16 s.

**Veredicto, con 12.800 episodios** (4 veces el máximo del ajuste, 3 semillas):

- **Ciclo:** converge pronto y se mantiene ($G$ entre 8,3 y 8,5, deriva 0,01, rango entre semillas 0,04).
- **Browniano:** salta entre B&H y no operar, que valen casi lo mismo.
- **Regímenes:** converge a no operar.
- **Reversión:** **no converge**. Cada semilla alterna entre $G\approx0{,}12$ y $\approx0{,}30$, y la oscilación no baja con más experiencia. DQN oscila incluso más: con los pesos del último control, en vez del mejor, DQN cae en test de +22,1% a +2,6% y A3C de +27,5% a +15,9%. Elegir el mejor control en validación es imprescindible en este mercado.

![Prueba larga de A3C: ganancia $G$ en validación.](figures/convergencia_a3c.png){width=86%}

# 4. Otras pruebas

- **MSE frente a MAE.** Ninguna diferencia supera el error de 3 semillas. En DQN, MSE es algo mejor en reversión (+28,3% frente a +26,6%) y con MAE deja de operar en browniano y regímenes. En el crítico de A3C es al revés (+26,6% con MAE frente a +23,8%). Lectura posible: la mediana del target ignora las colas que justifican entrar, mientras que un crítico que solo es línea base gana con la robustez de MAE.
- **Minibatch frente a episodio entero** (DQN, sin buffer). Una pasada por episodio iguala al minibatch: en reversión, +24,4% frente a +22,1%. Dieciséis pasadas, que suman tantas experiencias como el minibatch, hunden reversión a +10,8%: la red se ajusta a las experiencias correlacionadas de la última serie. El valor del buffer aquí es no olvidar las series anteriores.
- **Datos reales** (entrenamiento hasta 2015, validación 2016–18, test desde 2019; configuración de regímenes, sin ajustar). Ningún agente supera a B&H, igual que en el taller anterior (allí, con otro periodo: S&P 500 +228% frente a +232%; IBEX +9% frente a +117%). Con una sola historia de test por activo, en años muy alcistas, no se puede generalizar.
  - A3C acaba haciendo B&H: S&P 500 +190% frente a +193%; IBEX +101% frente a +111%; fondo mixto +106% frente a +120%. En el fondo de bonos no entra.
  - DQN cambia retorno por riesgo: en el S&P 500 gana +179%, con caída máxima del 25% frente al 34% de B&H.

# 5. Análisis crítico y conclusiones

- **DQN y A3C aprenden lo esperado.** Ninguno descubre nada que no capture ya una buena regla con histéresis. El techo lo pone el estado: sin la distancia a la media ni el régimen, el oráculo queda lejos (48% en reversión, 0% en regímenes).
- **A3C no es peor que DQN:** en reversión es mejor y más estable entre semillas. Pero es más sensible a sus hiperparámetros (en reversión, variación entre configuraciones de 7,7 puntos, frente a 2,9 de error) y, como DQN, oscila en reversión.
- **El criterio de convergencia sobre la mediana en % engaña en ciclo:** una mediana de +80.000% varía más de un 10% por puro ruido. Por eso medimos la convergencia sobre $G$.
- **Lección de método:** sin el test intermitente, habríamos atribuido a A3C la inestabilidad de nuestra implementación. **Limitaciones:** comisión fija, ajuste con 3 semillas y tiempos medidos en un portátil de 16 núcleos.
