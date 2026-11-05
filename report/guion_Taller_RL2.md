# Guion oral de Taller_RL2.pdf

Duración prevista: **13 minutos**, incluidas pequeñas pausas para señalar tablas y gráficos. Sigue las **10 páginas del PDF**, contando la portada como diapositiva 1. Los textos de «Guion» están escritos para decirlos en voz alta; las indicaciones y notas finales son para preparar la exposición.

| Diapositiva | Tiempo | Acumulado |
|---|---:|---:|
| 1. Portada | 0:35 | 0:35 |
| 2. El problema | 1:40 | 2:15 |
| 3. Lo que hicimos | 1:10 | 3:25 |
| 4. Resultados en test | 1:20 | 4:45 |
| 5. Lo que aprendieron | 1:15 | 6:00 |
| 6. ¿Converge? | 1:10 | 7:10 |
| 7. Pruebas complementarias | 1:30 | 8:40 |
| 8. Estabilizar el entrenamiento | 2:00 | 10:40 |
| 9. ¿Y con más semillas? | 1:30 | 12:10 |
| 10. Conclusiones | 0:50 | 13:00 |

## 1. DQN y A3C: ¿cuándo estar dentro de un activo?

**Guion**

«En este proyecto estudiamos cómo un agente de aprendizaje por refuerzo puede decidir cuándo comprar, mantener o vender un activo.

Comparamos principalmente DQN y A3C en cuatro mercados simulados. Además de medir cuánto ganan, analizamos qué estrategia aprenden y si conservan su rendimiento al seguir entrenando.

La pregunta central es si podemos distinguir una política que funciona de un resultado favorable debido al azar. Para responderla, repetimos los experimentos y probamos mejoras de estabilidad.»

**Al mostrarla:** presentar al equipo sin leer todos los datos de la portada.

## 2. El problema

**Guion**

«El agente empieza con un capital de mil unidades y observa dos datos: la posición que tiene y la tendencia del precio respecto a la media de los seis últimos precios. El estado es continuo: no lo dividimos en intervalos.

Puede comprar, vender o mantener. Cuando compra, invierte todo el efectivo; cuando vende, liquida la posición. No hay posiciones cortas. Cada operación efectiva paga una comisión del 0,1%, así que operar demasiado puede eliminar una ventaja pequeña.

La recompensa refleja la variación del precio cuando está invertido, menos el coste de operar. Normalizamos las entradas y las recompensas para facilitar el entrenamiento.

DQN aprende el valor esperado de cada acción. Guarda experiencias en una memoria y aprende de muestras mezcladas. Además, utiliza una red objetivo que se actualiza periódicamente para que la referencia de aprendizaje cambie más despacio. Probamos también Doble DQN y DQN en duelo.

A3C aprende directamente una política mediante un actor, acompañado de un crítico que evalúa las decisiones. Cuatro trabajadores exploran series distintas y actualizan un modelo compartido. La entropía favorece la exploración. REINFORCE sirve como comparación con un trabajador y sin crítico.

Los cuatro mercados permiten comprobar si estas decisiones se adaptan a tendencias, ruido, reversión a la media y cambios de régimen.»

**Al mostrarla:** señalar primero estado y acciones, después las dos familias. Reservar el detalle del cerrojo para la diapositiva 7.

## 3. Lo que hicimos

**Guion**

«Organizamos el trabajo en seis bloques, con más de mil entrenamientos en conjunto.

Primero implementamos los cinco agentes. Después ajustamos sus parámetros: cuánto aprenden en cada actualización, cuánto valoran las recompensas futuras, cómo exploran y durante cuánto tiempo entrenan. Esta búsqueda inicial utilizó tres semillas por configuración.

Una semilla controla parte del azar del experimento. Repetir con semillas distintas permite comprobar si un resultado depende de una inicialización o de unas experiencias especialmente favorables.

El experimento principal se evaluó con diez semillas nuevas por agente y mercado. Después añadimos pruebas de implementación, estabilidad y selección de parámetros.

La separación importante es entre entrenamiento, validación y test. Cada episodio usa una serie nueva de 500 precios. Las cien series de validación sirven para elegir parámetros y guardar el mejor modelo; las otras cien de test sirven para evaluarlo. Las mejoras posteriores se analizan por separado y no sustituyen automáticamente los resultados del test principal.»

**Al mostrarla:** recorrer los bloques sin leer cada cifra; destacar la separación entre validación y test.

## 4. Resultados en test

**Guion**

«Esta tabla resume la rentabilidad mediana en test. No hace falta leer todas las filas: hay tres resultados principales.

En el mercado cíclico, los agentes aprovechan un patrón muy marcado y consiguen rentabilidades enormes. Son porcentajes del simulador, no una expectativa realista de inversión. Las redes superan a momentum, pero no mejoran claramente a Q-learning y Sarsa, que son las referencias del taller anterior.

En reversión a la media aparece la diferencia más interesante. DQN y sus variantes se sitúan alrededor del 21 o 22%, mientras que A3C y REINFORCE alcanzan aproximadamente el 27 y el 29%. Aquí, aprender directamente una política ofrece mejores resultados en este experimento.

En el mercado browniano y en el de regímenes, los agentes entrenados obtienen cero porque no entran al mercado. Evitan pérdidas y comisiones, pero en regímenes tampoco recuperan la pequeña ventaja que conseguían los métodos tabulares.

Por tanto, aumentar la complejidad del modelo no garantiza una mejora en todos los escenarios. Ahora veremos qué decisiones explican estos resultados.»

**Al mostrarla:** señalar las columnas de ciclo, reversión y regímenes; no comparar cifras de esta tabla con las de validación como si fueran la misma evaluación.

## 5. Lo que aprendieron

**Guion**

«En el ciclo, el agente sigue la tendencia: compra cuando la señal es suficientemente positiva y vende cuando pasa a ser suficientemente negativa. Entre ambos límites mantiene su posición. Esta banda de inacción evita pagar comisiones ante fluctuaciones pequeñas.

En reversión hace lo contrario: compra tras caídas pronunciadas y espera al rebote para vender. No actúa contra cualquier movimiento, sino principalmente en los extremos. Los umbrales que vemos están expresados en unidades de tendencia normalizada, no en porcentajes de rentabilidad.

En browniano no encuentra una señal clara y, en regímenes, la tendencia reciente no permite identificar de forma fiable el estado oculto del mercado.

Abajo aparece G, que es la suma de las recompensas durante un episodio. No es el porcentaje de rentabilidad del capital: ese porcentaje incorpora la evolución compuesta de la cartera.

El oráculo de reversión conoce información adicional, como el nivel de referencia del precio, y logra un G bastante mayor. Esa diferencia sugiere que enriquecer el estado podría ayudar, aunque no demuestra que toda la distancia se explique únicamente por la información disponible.»

**Al mostrarla:** recorrer las bandas de decisión y cerrar señalando la comparación con el oráculo.

## 6. ¿Converge?

**Guion**

«Llegar a una buena política no significa conservarla hasta el final.

En ciclo, DQN y A3C alcanzan el criterio de convergencia medido sobre G en unos cincuenta episodios. En reversión necesitan más entrenamiento y las curvas oscilan.

El gráfico muestra la consecuencia práctica. Si evaluamos DQN con los pesos del mejor control elegido en validación, obtiene un 22,1% en test. Si usamos los pesos finales, baja al 2,6%. En A3C pasa del 27,5% al 15,9%.

Un control es una evaluación periódica durante el entrenamiento. Guardamos los pesos que funcionan mejor en validación; no escogemos el mejor modelo mirando el test.

Tampoco basta con entrenar más: incluso con 12.800 episodios, A3C sigue oscilando en reversión. Por eso planteamos dos acciones: conservar el mejor control y buscar cambios que reduzcan el deterioro al final.»

**Al mostrarla:** señalar la diferencia entre las dos barras de cada algoritmo. Aclarar que alcanzar el criterio no equivale a una estabilidad permanente.

## 7. Pruebas complementarias

**Guion**

«Una de las pruebas más útiles detectó un error de concurrencia en A3C. Varios procesos actualizaban el estado compartido del optimizador Adam y podían dejar sus estadísticas en un estado inconsistente. Un solo paso podía disparar los pesos y hacer que el agente dejara de operar.

La solución fue proteger únicamente la actualización de Adam con un cerrojo. Los trabajadores siguen calculando sus gradientes de forma asíncrona. Tras corregirlo, la peor semilla en reversión pasó del 5,3 al 25,2%. Es una corrección de implementación, distinta de ajustar hiperparámetros.

También comparamos funciones de pérdida. Con tres semillas no encontramos una ventaja clara de MSE o MAE.

En DQN, entrenar una vez sobre cada episodio completo funcionó bien, pero repetir dieciséis veces la última serie redujo el retorno al 10,8%. El resultado es compatible con sobreajustar experiencias recientes.

Con cuatro trabajadores, A3C redujo el tiempo de la prueba de unos 57 a 23 segundos; pasar a ocho aportó poco más.

Finalmente probamos datos reales, separando los periodos temporalmente. En el S&P 500 ninguno superó a comprar y mantener. DQN tuvo una caída máxima menor, pero una única historia de test por activo no permite generalizar.»

**Al mostrarla:** dedicar aproximadamente la mitad del tiempo al error de concurrencia y el resto a los cuatro bloques inferiores.

## 8. Estabilizar el entrenamiento

**Guion**

«Aquí buscamos que el agente conserve una buena política al terminar. Alpha es la tasa de aprendizaje: reducirla hace que las actualizaciones finales sean más pequeñas.

En DQN probamos cambios en esa tasa y en la frecuencia de copia de la red objetivo. La variante que confirmamos modifica únicamente el aprendizaje: mantiene la tasa inicial durante la primera mitad y después la reduce progresivamente hasta el 30%. La red objetivo sigue copiándose cada mil pasos.

Con cinco semillas nuevas, el retorno final medio pasa del 6,3 al 18,1%. La oscilación baja aproximadamente un 47% y la brecha entre mejor y último control pasa de 18,4 a 5,9 puntos porcentuales. El retorno final y G mejoran en las cinco semillas.

Lo interesante es que el mejor retorno alcanzado apenas cambia: ronda el 24% en ambas variantes. La mejora consiste en conservar más rendimiento al terminar. Estos resultados son de validación; falta un test independiente de esta modificación.

En A3C, la conclusión es más matizada. Con 12.800 episodios, reducir el aprendizaje ayudaba. Con 3.200, terminar al 10% suaviza mucho la curva, pero reduce el rendimiento. Terminar al 30% conserva mejor la rentabilidad, aunque G queda ligeramente por debajo de la referencia.

Como ninguna variante cumple todos los criterios fijados, mantenemos A3C original. Solo esa configuración pasa al test reservado: obtiene aproximadamente un 26,7% con el mejor control y un 16,2% con el último.

La lección es que una curva más suave no basta: hay que mantener también el rendimiento.»

**Al mostrarla:** explicar primero toda la mitad de DQN y luego A3C. En estas tablas, G final resume el último cuarto de los controles y la oscilación mide su variación temporal; el retorno final corresponde al último control.

## 9. ¿Y con más semillas?

**Guion**

«La segunda mejora afecta a cómo elegimos los hiperparámetros. Si comparamos muchas opciones con pocas semillas, podemos seleccionar una que haya tenido suerte.

Para comprobarlo, repetimos la configuración elegida inicialmente y las dos alternativas más prometedoras de cada algoritmo. Son seis configuraciones por cinco semillas nuevas: treinta entrenamientos. Esta comparación utiliza únicamente las nuevas semillas, aunque conserva las mismas series de validación.

En A3C, la configuración inicial mantiene un resultado muy parecido: pasa del 30,8 al 30,6%. Sus dos alternativas quedan por debajo.

En DQN, la configuración inicial baja del 28,3 al 16,1%. La alternativa con aprendizaje 0,0001 y descuento cero alcanza el 21,9%, una ventaja media de 5,85 puntos.

Pero las líneas del gráfico muestran la incertidumbre. Si un intervalo cruza el cero, los resultados no permiten afirmar con claridad que la alternativa mejore. En este caso, el intervalo de esa ventaja va aproximadamente de menos cinco a más diecisiete puntos.

Para cambiar exigíamos un intervalo completamente positivo y una ventaja superior al mayor de un punto porcentual y el 5% de la media de referencia. Ninguna alternativa supera esa regla.

Conservamos las configuraciones iniciales. En DQN es una decisión prudente ante la incertidumbre, no una demostración de que la configuración inicial sea superior.»

**Al mostrarla:** señalar el cero del gráfico y un intervalo que lo cruce. Esta prueba compara hiperparámetros del ajuste original; es distinta de confirmar el calendario decreciente de la diapositiva 8.

## 10. Conclusiones

**Guion**

«Nos quedamos con cinco conclusiones.

Los agentes aprenden estrategias coherentes con los mercados, y los métodos de actor destacan en reversión.

Un buen resultado intermedio no garantiza un buen resultado final: guardar el mejor control en validación es importante.

Las pruebas de implementación también importan, como mostró el fallo de concurrencia de A3C.

Reducir el aprendizaje estabiliza DQN en las cinco semillas de confirmación. En A3C, ganar estabilidad no asegura conservar el rendimiento.

Y repetir con semillas nuevas evita confiar demasiado en una configuración afortunada.

El siguiente paso sería evaluar la mejora de DQN en un test independiente. También queda estudiar estados más informativos y otros costes de operación. Los resultados actuales apoyan estas líneas de trabajo, pero no demuestran una ventaja general en mercados reales. Gracias.»

## Notas de preparación — no leer durante la exposición

- **Duración:** ensayar a un ritmo aproximado de 135–150 palabras por minuto, dejando las pausas indicadas. Si falta tiempo, resumir las variantes de DQN en la diapositiva 2 y las pruebas secundarias de la 7. Mantener las explicaciones de validación/test y de las diapositivas 8–9.
- **Reparto opcional entre tres personas:** diapositivas 1–3 (3:25), 4–7 (5:15) y 8–10 (4:20). Los tiempos incluyen las transiciones; no hace falta presentar de nuevo el proyecto en cada relevo.
- **G y rentabilidad:** G suma recompensas sin descuento en la métrica presentada. Los retornos a varios pasos que calcula A3C para aprender sí incorporan gamma. No convertir G=0,25 directamente en un 25% de rentabilidad.
- **Puntos porcentuales:** pasar del 6,3% al 18,1% son 11,8 puntos porcentuales de mejora; no un aumento del 11,8% relativo.
- **Diapositiva 5:** el texto «lo que falta está en el estado, no en el algoritmo» es más categórico que la evidencia. El guion lo expresa como una hipótesis apoyada por el oráculo, que utiliza información adicional y una selección optimista de su banda.
- **Diapositiva 8:** al 30%, A3C sí mejora ligeramente el retorno final y reduce la oscilación, pero pierde algo de G. Por eso conviene decir «no cumple todos los criterios» en lugar de «todas las variantes rinden menos».
- **Semillas nuevas:** permiten estudiar la variabilidad del entrenamiento; reutilizar la validación no equivale a evaluar nuevos mercados ni a un test independiente. En A3C, la ejecución asíncrona puede introducir además variación entre ejecuciones.
- **Discrepancia menor entre documentos:** la diapositiva 4 muestra REINFORCE en reversión como 28,6% y `informe_v2.pdf` como 28,5%. El guion usa «aproximadamente 29%» para no dar una precisión inconsistente. El oráculo aparece asimismo redondeado a 0,63 y 0,62; esta diferencia no cambia la interpretación.
- **Detalle técnico opcional, diapositiva 2:** no se aplica al gradiente del actor el factor adicional gamma elevado al instante del episodio. Así se evita atenuar casi por completo el aprendizaje de los pasos tardíos. Esto no elimina el descuento gamma de los retornos de A3C.

## Documentos y código consultados

- `report/Taller_RL2.pdf`: orden, títulos, tablas y mensajes de las diez diapositivas.
- `report/informe_v2.pdf`: protocolo, resultados principales y pruebas posteriores.
- `rl_trader/env.py`: observaciones, acciones, capital, recompensa y comisiones.
- `rl_trader/dqn.py` y `rl_trader/a3c.py`: algoritmos, memoria, red objetivo, trabajadores y protección de Adam.
- `rl_trader/entrenamiento.py`: selección del mejor control por rentabilidad mediana de validación y definición de G.
- `rl_trader/oraculos.py`: información adicional de los oráculos y carácter optimista de la referencia de reversión.
- `scripts/estabilidad_dqn.py`: calendario decreciente, confirmación con nuevas semillas y criterios de mejora.
- `scripts/ajuste_robusto.py`: selección de candidatas, diferencias pareadas, bootstrap y regla de cambio.
- `report/fuentes_resultados.md`: trazabilidad de los archivos de resultados utilizados en el informe.
