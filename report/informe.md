# DQN y A3C para invertir en un activo

**Josep Peréz Segura · Emilio Sanchez Martínez · Natalia San José Ortega**

Taller 2 de Aprendizaje por Refuerzo · 5 de noviembre de 2026

## 1. Diseño y protocolo experimental

Entrenamos DQN, Doble DQN, DQN en duelo, A3C y REINFORCE en cuatro mercados: ciclo, browniano, reversión a la media y alternancia de regímenes. El entorno conserva el estado continuo (tendencia de seis precios, posición), las acciones vender/mantener/comprar y la comisión del 0,1 %. Normalizamos tendencia y recompensa sin discretizar el estado. Cada episodio utiliza una serie nueva de 500 precios; validación y test contienen 100 series con semillas separadas.

DQN usa redes de 64×64, buffer de 100.000 transiciones, lotes de 64 cada cuatro pasos y red objetivo cada 1.000 pasos. A3C usa cuatro trabajadores, actor y crítico separados y Adam compartido; la entropía regula la exploración. Un cerrojo en el paso de Adam evita actualizaciones inconsistentes del estado compartido, manteniendo asíncrono el cálculo de gradientes.

El ajuste original compara, con tres semillas, aprendizaje y descuento, exploración, pérdidas MSE/MAE y episodios. Selecciona pesos mediante validación y evalúa con diez semillas nuevas de entrenamiento. En reversión quedan DQN con α=0,002, γ=0,5, MSE y 400 episodios; A3C con α_actor=0,0003, α_crítico=0,0009, γ=0,9, β_inicial=0,2, retornos a 20 pasos, MAE y 3.200 episodios. Las ampliaciones de la página 2 se analizan por separado: no sustituyen estos resultados de test.

## 2. Resultados del experimento principal: test

| Agente | Ciclo (%) | Browniano (%) | Reversión (%) | Regímenes (%) |
|---|---:|---:|---:|---:|
| Aleatorio | 456 | −14,7 | −15,2 | −23,9 |
| Buy & hold | 3.425 | +0,7 | −1,1 | −20,4 |
| Momentum | 48.189 | −11,8 | −28,7 | −18,4 |
| Contrarian | −92,9 | −14,3 | +6,9 | −22,6 |
| DQN | 59.363 | 0 | +22,1 | 0 |
| Doble DQN | 59.363 | 0 | +21,0 | 0 |
| DQN en duelo | 60.723 | 0 | +22,2 | 0 |
| A3C | 63.619 | 0 | +27,5 | 0 |
| REINFORCE | 59.905 | 0 | +28,6 | 0 |

Mediana del retorno sobre 100 series de test; diez semillas de entrenamiento para cada agente RL. Los retornos extremos del ciclo responden al generador sintético y no representan una expectativa financiera real.

**Estrategia aprendida.** En ciclo siguen la tendencia con una banda de inacción que ahorra comisiones; en reversión compran en caídas extremas y mantienen hasta el rebote. En browniano predominan no operar o buy & hold; en regímenes evitan entrar. El estado observado limita la información: el oráculo obtiene G≈0,63 en reversión, frente a G≈0,25–0,30 de los agentes. Frente al taller tabular, las redes no mejoran claramente ciclo; los métodos de actor mejoran reversión, mientras se pierde el pequeño margen de regímenes.

**Convergencia.** Medida sobre G, suma de recompensas, en ciclo DQN y A3C alcanzan el criterio en 50 episodios (≈6,6 y 2,0 s). En reversión, DQN lo alcanza en 375 episodios (≈49 s) y A3C en 3.200 (≈130 s), cerca del presupuesto: esto no implica estabilidad posterior. La prueba de A3C con 12.800 episodios y tres semillas mantiene oscilaciones en reversión. En el test principal, usar el último control en vez del mejor reduce el retorno de DQN de 22,1 % a 2,6 % y el de A3C de 27,5 % a 15,9 %.

## 3. Pruebas complementarias

MSE frente a MAE no ofrece una ventaja clara con tres semillas. En DQN sin buffer, una pasada por episodio logra 24,4 % en reversión, frente al 22,1 % del minibatch; repetir 16 veces el mismo episodio baja a 10,8 %. Mezclar experiencias ayuda a evitar sobreajustar la última serie. El arreglo de concurrencia de A3C elevó su peor semilla en reversión de 5,3 % a 25,2 %; al pasar de uno a cuatro trabajadores, el tiempo de la prueba bajó de unos 57 a 23 s, con poco beneficio adicional al usar ocho.

Con datos reales (entrenamiento hasta 2015, validación 2016–2018 y test desde 2019), ninguno supera a buy & hold. En el S&P 500, A3C obtiene aproximadamente 190 % frente a 193 %; DQN, 179 %, con caída máxima del 25 % frente al 34 %. Una única historia de test por activo no permite generalizar.

<!-- pagebreak -->

## 4. Estabilizar el entrenamiento: pruebas nuevas en reversión

En estas pruebas, **R_final** es la media entre semillas de la rentabilidad mediana del último control en validación; **G_final** es la ganancia media del último cuarto; **oscilación** es su desviación temporal; **brecha** es la diferencia entre mejor y último retorno, en puntos porcentuales (pp). No son las medianas agregadas del test principal.

**DQN: aprendizaje decreciente y red objetivo.** Con 400 episodios y tres semillas, reducir α hasta el 30 % durante la segunda mitad elevó G_final de 0,1325 a 0,2261 y R_final de 11,45 % a 17,74 %. Copiar la red objetivo cada 500 pasos logró R_final=18,04 %, pero no redujo la oscilación; cada 2.000 pasos logró G_final=0,2308, con mayor brecha que reducir solo α. Se confirmó esta última variante manteniendo la copia cada 1.000 pasos, sin cambiar exploración ni otros parámetros.

| Confirmación DQN: cinco semillas nuevas | Original | Solo α decreciente |
|---|---:|---:|
| G_final | 0,1331 | **0,2168** |
| Oscilación | 0,0749 | **0,0397** |
| R_final | 6,28 % | **18,13 %** |
| Brecha mejor–último | 18,39 pp | **5,88 pp** |

G_final y R_final mejoraron en **5 de 5 semillas**; la oscilación bajó un 47 % y la brecha un 68 %, con ≈42 s por entrenamiento en ambas variantes. Supera el criterio exploratorio fijado antes de confirmar. La mejor rentabilidad alcanzada apenas cambia (24,66 % frente a 24,01 %): la ventaja es conservar rendimiento al terminar. **Falta evaluar esta mejora de DQN en un test independiente.**

**A3C: decaimiento del aprendizaje y de la entropía.** A 12.800 episodios, tres semillas y cuatro variantes, reducir solo α hasta el 10 % aumentó G_final de 0,2156 a 0,2550 y redujo la oscilación de 0,0519 a 0,0147; reducir solo β o ambos dio G_final=0,2492 y 0,2525, respectivamente. Las repeticiones a 3.200 episodios fueron sensibles al calendario y a la ejecución; menor oscilación no siempre conservó el rendimiento. En la búsqueda posterior con entropía original:

| A3C: 3.200 episodios, tres semillas | G_final | Oscilación | R_final |
|---|---:|---:|---:|
| α constante | 0,2378 | 0,0529 | 16,09 % |
| α final al 10 % | 0,1955 | 0,0079 | 12,72 % |
| α final al 30 % | 0,2328 | 0,0374 | 16,58 % |

Ninguna reducción cumplió a la vez no perder G ni retorno y reducir oscilación. Se conservó A3C original y se repitió con cinco semillas nuevas. **Solo esa configuración** pasó al nuevo test reservado de 100 series: 26,67 % con el mejor control de validación y 16,17 % con el último (medias entre semillas de medianas por serie). Este test no demuestra superioridad de un calendario alternativo, pues ninguno llegó a él.

## 5. Selección de hiperparámetros con más semillas

Para DQN y A3C en reversión se seleccionaron la configuración actual y las dos alternativas mejor valoradas en las etapas A–D; se eliminaron duplicados y se conservaron sus presupuestos. Se realizaron **30 entrenamientos: seis configuraciones × cinco semillas nuevas**, con las mismas 100 series de validación. El ranking nuevo no mezcla las tres semillas utilizadas para preseleccionar. Se comparan el retorno del mejor control, la dispersión y diferencias pareadas con intervalos bootstrap orientativos.

En **A3C**, la configuración actual mantiene 30,58 % (antes 30,83 %) y desviación de 1,38 pp; las alternativas alcanzan 26,27 % y 23,42 %. En **DQN**, la actual baja a 16,08 % y una alternativa llega a 21,93 %, pero su ventaja de 5,85 pp tiene intervalo [−4,93; +16,71], compatible con cero. La otra logra 18,00 %, también con intervalo que incluye cero. La regla exigía intervalo positivo y ventaja mayor que 1 pp o el 5 % de la media actual, el mayor de ambos: **se mantienen ambas configuraciones**. Retener DQN es una decisión conservadora, no una prueba de superioridad.

## 6. Conclusiones y límites

El aprendizaje decreciente de DQN reduce el deterioro final y supera una confirmación de cinco semillas en validación. En A3C, más estabilidad puede acompañarse de menor rendimiento: no se acepta una mejora solo por suavizar la curva. Repetir pocas candidatas ayuda a detectar selecciones inciertas sin rehacer toda la búsqueda. Persisten limitaciones: pocas semillas, series de validación reutilizadas, selección del mejor control, intervalos sin corrección por comparaciones múltiples y simulación con comisión fija. Los resultados son descriptivos; no se afirma significación estadística de las mejoras nuevas. El test debe permanecer fuera del ajuste.
