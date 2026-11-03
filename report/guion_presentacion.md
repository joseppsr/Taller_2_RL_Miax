# Guion de la presentación

Presentación: [DQN y A3C para invertir en un activo](https://claude.ai/artifact/Ab6qcEp1bQ8athoz8BVSL6). El mismo texto está en las notas del orador de cada diapositiva.

**Tiempo:** unos 9 minutos de exposición, para dejar unos 6 de discusión dentro de los 15 que da el profesor. Si vais justos, la diapositiva 10 (otras pruebas) se puede resumir en una frase.

| # | Diapositiva | Tiempo | Acumulado |
|--:|:--|--:|--:|
| 1 | Portada | 0:20 | 0:20 |
| 2 | ¿Cuándo estar dentro del mercado? | 0:50 | 1:10 |
| 3 | Dos familias de agentes | 0:50 | 2:00 |
| 4 | Ajuste por etapas | 0:50 | 2:50 |
| 5 | Resultados en test | 1:10 | 4:00 |
| 6 | La estrategia aprendida es la esperada | 0:50 | 4:50 |
| 7 | Era un fallo de concurrencia en nuestro A3C | 1:00 | 5:50 |
| 8 | A3C con 12.800 episodios | 1:00 | 6:50 |
| 9 | Frente al taller anterior | 0:50 | 7:40 |
| 10 | Otras pruebas | 0:50 | 8:30 |
| 11 | Conclusiones | 0:40 | 9:10 |

## Qué decir en cada diapositiva

**1. Portada.** Somos [nombres]. Hemos entrenado DQN y A3C, más Doble DQN, DQN en duelo y REINFORCE, para decidir cuándo estar dentro de un activo, en los cuatro mercados simulados y con datos reales. Contamos el diseño, el ajuste, los resultados, la convergencia de A3C y la comparación con el taller tabular.

**2. El problema.**
- El entorno del enunciado, sin tocar: vender, mantener o comprar, con una comisión del 0,1%.
- Lo nuevo es el estado: la tendencia continua y la posición entran directamente en la red, sin los intervalos del taller anterior.
- El test usa las mismas 100 series que el taller anterior. Lo comprobamos reproduciendo sus medianas de las reglas fijas.
- Antes de entrenar medimos qué retorno sigue a cada tendencia, y eso dice qué esperar: momentum en ciclo, nada en browniano, contrarian en los extremos en reversión y muy poco margen en regímenes.

**3. Los agentes.**
- **DQN**, como en las transparencias: buffer con minibatch al azar para romper la correlación entre experiencias, y red objetivo copiada cada 1.000 pasos.
- **A3C:** 4 procesos asíncronos sobre pesos compartidos, con actor y crítico separados. REINFORCE es un solo trabajador, sin crítico.
- **Tres decisiones nuestras:**
  - normalizar en vez de recortar la recompensa;
  - quitar el factor γᵗ;
  - quedarse con los pesos del mejor control en validación.

**4. Ajuste.**
- Por etapas y solo en validación: α y γ, luego la exploración, la pérdida y el número de episodios.
- **Ciclo:** γ = 0, porque la inercia se ve en el paso siguiente.
- **Reversión:** γ alto, para aguantar la posición hasta que el precio vuelve a la media.
- **A3C** usa más episodios, pero cada uno le cuesta 3 veces menos.
- **Browniano y regímenes:** elegir es casi azar, porque la diferencia entre configuraciones es del tamaño del error de 3 semillas.

**5. Resultados.**
- **Ciclo:** todos multiplican el capital unas 600 veces y superan a buy & hold y a momentum. Entre ellos casi empatan: la ganancia G va de 8,1 a 8,2.
- **Browniano:** 0%, que no se distingue de buy & hold.
- **Reversión:** ganan A3C y REINFORCE, con +27–29%, frente a +21–22% de DQN y del tabular.
- **Regímenes:** no operan, y así evitan el régimen malo.
- **Comprobación:** Doble DQN da exactamente lo mismo que DQN en ciclo, porque con γ = 0 no hay máximo que sobreestimar.

**6. Estrategia.** Sí es la esperada:
- **Ciclo:** momentum con histéresis. Compra por encima de +0,5σ y no vende hasta −0,7σ.
- **Reversión:** contrarian solo en los extremos.
- **Browniano:** no operar o buy & hold, que son equivalentes.
- **Regímenes:** no operar.

Un oráculo que conoce el régimen ganaría un 26%: lo que falta está en el estado, no en el algoritmo.

**7. El fallo.**
- **El síntoma:** A3C parecía no converger; con la misma semilla, unas veces aprendía y otras dejaba de operar.
- **Cómo lo encontramos:** un test que fallaba de vez en cuando.
- **La causa:** el estado de Adam compartido se actualizaba sin cerrojo. Un proceso leía la media de otro con la varianza aún a cero, y los pesos saltaban de 0,7 a más de 1.000.
- **El arreglo:** un cerrojo solo en ese paso.
- **El efecto:** la peor semilla en reversión pasa del 5% al 25%.

Sin ese test se lo habríamos atribuido al algoritmo.

**8. Convergencia con 12.800 episodios.**

| Mercado | ¿Converge? |
|:--|:--|
| Ciclo | Sí: converge y se mantiene |
| Browniano | Salta entre dos políticas equivalentes |
| Regímenes | Sí: converge a no operar |
| Reversión | No: cada semilla oscila entre G 0,12 y 0,30 |

DQN oscila todavía más. Con el último control en lugar del mejor, DQN cae de +22% a +3%, y A3C de +27,5% a +16%.

**9. Frente al tabular.**
- **En rendimiento**, quitar los intervalos no aporta: empate en ciclo, mejora en reversión con los métodos de actor y pérdida del margen en regímenes.
- **En convergencia**, con el mismo criterio que el taller anterior (la mediana):
  - DQN converge en reversión con unas 10 veces menos episodios, pero cada uno le cuesta 13 veces más (49 s frente a unos 30 s).
  - A3C necesita tantos episodios como la tabla.

**10. Otras pruebas.**
- **MSE frente a MAE:** sin diferencias claras.
- **Minibatch frente a episodio entero:** una pasada iguala al minibatch, pero 16 pasadas hunden reversión al 11%. El buffer sirve para no olvidar las series anteriores.
- **Datos reales:** nadie supera a buy & hold. A3C acaba comprando y manteniendo, y DQN cambia retorno por menos riesgo.

**11. Conclusiones.** Las cinco de la diapositiva. Cerrar con la lección de método y abrir las preguntas.

## Preguntas probables y respuestas

**¿Por qué normalizar la recompensa en vez de recortarla a ±1, como en las transparencias?**
Recortar borraría la diferencia entre un retorno pequeño y uno grande, que aquí es justo la información útil. Dividir por una constante positiva no cambia la política óptima.

**¿Por qué quitasteis el factor I = γᵗ del gradiente del actor?**
Ese factor optimiza solo el valor del estado inicial, y aquí el primer paso de la serie no tiene nada especial. Con γ = 0,9, a los pocos pasos de cada serie el peso es casi cero y el actor dejaría de aprender. El artículo original de A3C tampoco lo usa.

**¿El cerrojo no convierte A3C en síncrono?**
No. Solo se serializa el paso de Adam, que es muy corto; los episodios y los gradientes siguen siendo asíncronos, sobre pesos que otros procesos están cambiando. Con 1, 2, 4 y 8 trabajadores tarda 57, 33, 23 y 21 s, con el mismo resultado. Tiene un coste: sin cerrojo, 8 trabajadores tardaban 13–16 s, aunque con resultados que no servían.

**¿Por qué la mediana y no la media?**
En ciclo, unas pocas series disparan la media: DQN tiene una media de +243.139% y una mediana de +59.363%. Además de la mediana damos la ganancia G y el cociente geométrico del capital frente a cada referencia, con IC bootstrap pareado.

**¿Por qué Doble DQN da exactamente lo mismo que DQN en ciclo?**
Con γ = 0 el target es solo la recompensa, sin `max`, así que Doble DQN no cambia nada. Con las mismas semillas sale exactamente igual. Lo usamos como comprobación de la implementación.

**¿Por qué ningún agente opera en regímenes?**
La tendencia de 6 pasos no distingue el régimen bueno del malo. Un oráculo que lo conoce gana un 26%; los agentes se quedan fuera para no perder. Haría falta añadir al estado la volatilidad o una ventana más larga.

**¿Es justa la comparación con el taller anterior?**
Las series de test son las mismas, comprobado. No tenemos sus resultados por serie, así que no podemos dar significación frente al tabular. La convergencia la comparamos con su mismo criterio.

**¿Cuánto es ruido en la elección de hiperparámetros?**
En DQN, en reversión, la variación entre configuraciones (2,3 puntos) es casi igual al error de 3 semillas (2,1): esa elección es casi azar. En A3C sí se distinguen (7,7 frente a 2,9): es más sensible a sus hiperparámetros.

**¿Por qué A3C no converge en reversión?**
Es probable que se deba a que el paso de Adam no decrece y β no llega a cero, pero no lo hemos comprobado. DQN oscila incluso más. Por eso usamos el mejor control en validación.

**¿Qué mejoraríais?**
- Añadir al estado la distancia a la media y la volatilidad, que es donde está el hueco con el oráculo.
- Hacer decrecer el paso de aprendizaje.
- Usar más semillas en el ajuste.
